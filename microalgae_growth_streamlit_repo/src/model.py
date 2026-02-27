from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional, Tuple
import math
import pandas as pd

TrophicMode = Literal["autotrophic", "mixotrophic", "heterotrophic"]


@dataclass
class ModelParams:
    # --- Growth kinetics ---
    mu_max_d1: float = 1.8              # 1/day (tunable)
    X_max_gL: float = 50.0              # logistic cap in your biomass unit

    # Light & self-shading
    K_I: float = 80.0                   # μmol m^-2 s^-1 (half-saturation of effective light)
    k_shade: float = 0.05               # shading coefficient (per biomass unit per cm)
    light_path_cm: float = 3.0          # optical path length (cm)

    # Nutrients (normalized)
    K_N: float = 1.0                    # mmol N/L half-sat
    K_P: float = 0.05                   # mmol P/L half-sat (rough)

    # Carbon (autotrophic) - two options:
    # A) direct CO2% limitation (old proxy): fCO2 = pCO2/(K_pCO2 + pCO2)
    # B) inorganic carbon pool (DIC) with transfer & consumption: fDIC = DIC/(K_DIC + DIC)
    K_pCO2_kPa: float = 5.0             # kPa half-sat for pCO2 driving force (if not using DIC pool)
    H_CO2_mM_per_kPa: float = 1.0       # DIC_eq (mM) ≈ H * pCO2(kPa)  (proxy; tune)
    K_DIC_mM: float = 5.0               # half-sat for DIC (mM) (proxy; tune)
    C_req_mM_per_gX: float = 40.0       # mmol C per (g/L) biomass increase (proxy; tune if biomass unit is not g/L)

    # Mass transfer (proxy)
    K_kLa_d1: float = 10.0              # 1/day half-sat for kLa proxy
    kla_from_rpm_a: float = 5.0         # kLa_d1 ≈ a * (rpm/100)^b   (proxy)
    kla_from_rpm_b: float = 0.7

    # Temperature / pH
    T_ref_C: float = 25.0
    Q10: float = 2.0
    pH_opt: float = 8.2
    pH_sigma: float = 1.5

    # Carbon availability drops at high pH (proxy for carbonate chemistry / CO2(aq))
    pH_carbon_crit: float = 9.0
    pH_carbon_alpha: float = 3.0        # larger => sharper drop above crit

    # Lag phase (smooth ramp)
    lag_d: float = 0.5                  # days (0 means no lag)

    # Organic carbon yield (for hetero/mixo)
    Y_XC: float = 0.5                   # gX per gC (proxy)

    # N/P requirement scaling (because your biomass may be proxy units)
    N_req_scale: float = 1.0
    P_req_scale: float = 1.0

    # Composition bounds (fractions of DW)
    protein_min: float = 0.20
    protein_max: float = 0.60
    lipid_min: float = 0.10
    lipid_max: float = 0.40
    K_protein_N: float = 1.0            # mmol/L controls protein allocation

    def to_dict(self) -> Dict:
        return asdict(self)


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _f_saturation(x: float, k: float) -> float:
    if x is None:
        return 1.0
    if k <= 0:
        return 1.0
    x = max(float(x), 0.0)
    return float(x) / (float(k) + float(x) + 1e-12)


def _f_temp(T: float, T_ref: float, Q10: float) -> float:
    val = float(Q10) ** ((float(T) - float(T_ref)) / 10.0)
    return _clip(val, 0.0, 3.0)


def _f_ph_opt(pH: float, pH_opt: float, sigma: float) -> float:
    sigma = float(sigma) if sigma else 1.0
    return float(math.exp(-((float(pH) - float(pH_opt)) / sigma) ** 2))


def _f_carbon_pH(pH: float, crit: float, alpha: float) -> float:
    # sigmoid drop above crit
    return 1.0 / (1.0 + math.exp(float(alpha) * (float(pH) - float(crit))))


def _f_lag(t_d: float, lag_d: float) -> float:
    if lag_d is None or lag_d <= 0:
        return 1.0
    tau = max(float(lag_d) / 3.0, 1e-6)
    return 1.0 - math.exp(-float(t_d) / tau)


def _compute_composition(N_mM: float, energy_status: float, p: ModelParams) -> Tuple[float, float, float]:
    # N status drives protein
    N_status = _f_saturation(max(N_mM, 0.0), p.K_protein_N)
    protein = p.protein_min + (p.protein_max - p.protein_min) * N_status

    # If N low but energy high => lipid increases
    lipid = p.lipid_min + (p.lipid_max - p.lipid_min) * (1.0 - N_status) * _clip(energy_status, 0.0, 1.0)

    protein = _clip(protein, 0.0, 0.95)
    lipid = _clip(lipid, 0.0, 0.95)
    if protein + lipid > 0.90:
        scale = 0.90 / (protein + lipid + 1e-12)
        protein *= scale
        lipid *= scale
    carb = max(0.0, 1.0 - protein - lipid)
    return protein, lipid, carb


def simulate(
    trophic_mode: TrophicMode,
    carbon_source: str,
    C0_gL: float,
    nitrogen_source: str,
    N0_mM: float,
    phosphorus_as_P_mM: Optional[float],
    X0_gL: float,
    pH0: float,
    pH_end: Optional[float],
    light_uE_m2_s: float,
    temperature_C: float,
    dissolved_oxygen_mgL: Optional[float],
    gas_co2_percent: Optional[float],
    gas_co2_end_percent: Optional[float],
    pressure_kPa: Optional[float],
    mixing_rpm: Optional[float],
    kla_d1: Optional[float],
    DIC0_mM: Optional[float],
    duration_d: float = 7.0,
    dt_d: float = 0.02,
    params: Optional[ModelParams] = None,
    enable_self_shading: bool = True,
    enable_pH_drift: bool = False,
    enable_N_dynamics: bool = True,
    enable_P_dynamics: bool = False,
    enable_mass_transfer: bool = False,
    enable_logistic_cap: bool = True,
    enable_inorganic_carbon_pool: bool = True,
) -> pd.DataFrame:
    """
    Dynamic growth + composition model (v7):
      - N/P are consumed over time when enabled (N always recommended).
      - Autotrophic carbon can be modeled as an inorganic carbon pool (DIC) that is replenished from gas (kLa) and consumed by growth.
      - Vacuum/pressure affects pCO2 = yCO2 * P, which affects DIC_eq and carbon limitation.
    """
    p = params or ModelParams()

    t = 0.0
    X = max(float(X0_gL), 1e-12)
    C = max(float(C0_gL), 0.0)
    N = max(float(N0_mM), 0.0)
    P = max(float(phosphorus_as_P_mM), 0.0) if phosphorus_as_P_mM is not None else None

    # Pressure & CO2
    P_kPa = float(pressure_kPa) if pressure_kPa is not None and not math.isnan(float(pressure_kPa)) else 101.3

    # Inorganic carbon pool (mM)
    DIC = None
    if trophic_mode == "autotrophic" and enable_inorganic_carbon_pool:
        if DIC0_mM is not None and not math.isnan(float(DIC0_mM)):
            DIC = max(float(DIC0_mM), 0.0)
        else:
            # initialize at equilibrium with start pCO2
            yCO2 = max(float(gas_co2_percent or 0.0), 0.0) / 100.0
            pCO2 = yCO2 * P_kPa
            DIC = max(p.H_CO2_mM_per_kPa * pCO2, 0.0)

    rows = []
    steps = int(max(1, round(duration_d / dt_d)))

    for _ in range(steps + 1):
        frac = min(1.0, t / max(duration_d, 1e-9))

        # time-varying pH / CO2 if enabled
        pH_t = float(pH0)
        if enable_pH_drift and (pH_end is not None) and (not math.isnan(float(pH_end))):
            pH_t = float(pH0) + (float(pH_end) - float(pH0)) * frac

        CO2_t = float(gas_co2_percent or 0.0)
        if (gas_co2_end_percent is not None) and (not math.isnan(float(gas_co2_end_percent))):
            CO2_t = float(gas_co2_percent or 0.0) + (float(gas_co2_end_percent) - float(gas_co2_percent or 0.0)) * frac

        yCO2 = max(CO2_t, 0.0) / 100.0
        pCO2_kPa = yCO2 * P_kPa

        # effective light (self-shading)
        I0 = max(float(light_uE_m2_s), 0.0)
        I_eff = I0
        if enable_self_shading:
            I_eff = I0 * math.exp(-p.k_shade * X * max(p.light_path_cm, 0.1))

        # limiting factors
        fI = _f_saturation(I_eff, p.K_I) if trophic_mode != "heterotrophic" else 1.0
        fN = _f_saturation(N, p.K_N)
        fP = 1.0
        if P is not None:
            fP = _f_saturation(P, p.K_P)

        fT = _f_temp(float(temperature_C), p.T_ref_C, p.Q10)
        fpH = _f_ph_opt(pH_t, p.pH_opt, p.pH_sigma)

        # Carbon limitation
        fC_pH = _f_carbon_pH(pH_t, p.pH_carbon_crit, p.pH_carbon_alpha)

        # mass transfer proxy (prefer kla_d1, else estimate from rpm)
        kla_use = None
        fMT = 1.0
        if enable_mass_transfer:
            if kla_d1 is not None and (not math.isnan(float(kla_d1))):
                kla_use = float(kla_d1)
            else:
                rpm = float(mixing_rpm) if mixing_rpm is not None else 0.0
                kla_use = p.kla_from_rpm_a * ((max(rpm, 0.0) / 100.0) ** p.kla_from_rpm_b) if rpm > 0 else 0.0
            fMT = _f_saturation(kla_use, p.K_kLa_d1)

        # Autotrophic carbon
        DIC_eq = None
        if trophic_mode == "autotrophic":
            if enable_inorganic_carbon_pool and DIC is not None:
                DIC_eq = max(p.H_CO2_mM_per_kPa * pCO2_kPa, 0.0)
                fDIC = _f_saturation(DIC, p.K_DIC_mM)
                fC = fDIC * fC_pH
            else:
                fCO2 = _f_saturation(pCO2_kPa, p.K_pCO2_kPa)
                fC = fCO2 * fC_pH
        else:
            # organics
            fC = _f_saturation(C, 2.0)

        # DO inhibition (mild)
        fO2 = 1.0
        DO = None
        if dissolved_oxygen_mgL is not None and (not math.isnan(float(dissolved_oxygen_mgL))):
            DO = max(float(dissolved_oxygen_mgL), 0.0)
            fO2 = 1.0 / (1.0 + (DO / 30.0) ** 2)

        fLag = _f_lag(t, p.lag_d)

        mu = p.mu_max_d1 * fLag * fI * fC * fN * fP * fT * fpH * fMT * fO2
        mu = _clip(mu, 0.0, p.mu_max_d1)

        # growth ODE
        if enable_logistic_cap and p.X_max_gL is not None and p.X_max_gL > 0:
            dXdt = mu * X * (1.0 - X / max(p.X_max_gL, 1e-9))
        else:
            dXdt = mu * X

        dX = dXdt * dt_d
        X_next = max(0.0, X + dX)

        # energy status for composition allocation
        energy_status = _clip(fI * fC, 0.0, 1.0)
        protein_f, lipid_f, carb_f = _compute_composition(N, energy_status, p)

        # Organic carbon usage (hetero/mixo) — consumed
        dC = 0.0
        if trophic_mode != "autotrophic":
            dC = -(1.0 / max(p.Y_XC, 1e-9)) * dX

        # Nitrogen usage — consumed
        dN = 0.0
        if enable_N_dynamics:
            mmolN_per_gX = (protein_f / 6.25) / 0.014  # mmol N per gX
            dN = -p.N_req_scale * mmolN_per_gX * dX

        # Phosphorus usage — optional
        dP = 0.0
        if (P is not None) and enable_P_dynamics:
            mmolP_per_gX = 0.01 / 0.031  # rough
            dP = -p.P_req_scale * mmolP_per_gX * dX

        # Inorganic carbon usage — consumed (autotrophic) if pool enabled
        dDIC = 0.0
        if trophic_mode == "autotrophic" and enable_inorganic_carbon_pool and DIC is not None:
            # transfer from gas to liquid if enabled
            transfer = 0.0
            if enable_mass_transfer and kla_use is not None and DIC_eq is not None:
                transfer = kla_use * (DIC_eq - DIC) * dt_d  # mM per step
            # growth-associated carbon demand (proxy)
            consume = p.C_req_mM_per_gX * dX  # mM
            dDIC = transfer - consume

        # update pools
        C_next = max(0.0, C + dC)
        N_next = max(0.0, N + dN)
        P_next = None
        if P is not None:
            P_next = max(0.0, P + dP)

        DIC_next = None
        if DIC is not None:
            DIC_next = max(0.0, DIC + dDIC)

        # proxy flux indices (for demo)
        carbon_uptake_rate = max(0.0, -dC / dt_d) if trophic_mode != "autotrophic" else 0.0
        flux_gly = (carbon_uptake_rate / 10.0) + (max(0.0, dXdt) / 10.0)
        flux_ppp = flux_gly * (0.2 + 0.8 * (lipid_f / max(lipid_f + 0.05, 1e-9)))
        flux_tca = max(0.0, dXdt) / 5.0

        rows.append({
            "time_d": t,
            "biomass_gL": X,
            "carbon_gL": C,
            "nitrogen_mM": N,
            "phosphorus_mM": P,
            "DIC_mM": DIC,
            "DIC_eq_mM": DIC_eq,
            "pH": pH_t,
            "CO2_percent": CO2_t,
            "pressure_kPa": P_kPa,
            "pCO2_kPa": pCO2_kPa,
            "I_eff": I_eff,
            "mu_d1": mu,
            "fI": fI,
            "fC": fC,
            "fN": fN,
            "fP": fP,
            "fT": fT,
            "fpH": fpH,
            "fLag": fLag,
            "fMT": fMT,
            "DO": DO,
            "kLa_d1": kla_use,
            "protein_pct_dw": 100.0 * protein_f,
            "lipid_pct_dw": 100.0 * lipid_f,
            "carb_pct_dw": 100.0 * carb_f,
            "protein_gL": X * protein_f,
            "lipid_gL": X * lipid_f,
            "flux_glycolysis": flux_gly,
            "flux_ppp": flux_ppp,
            "flux_tca": flux_tca,
        })

        t += dt_d
        X, C, N, P, DIC = X_next, C_next, N_next, P_next, DIC_next

    return pd.DataFrame(rows)
