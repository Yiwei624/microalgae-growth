from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Literal, Optional, Tuple
import math
import numpy as np
import pandas as pd

TrophicMode = Literal["autotrophic", "mixotrophic", "heterotrophic"]

@dataclass
class ModelParams:
    mu_max_d1: float = 1.2
    X_max_gL: float = 40.0
    K_I: float = 80.0
    K_C: float = 2.0
    K_N: float = 1.0
    K_CO2: float = 1.0
    K_O2: float = 2.0
    K_rpm: float = 300.0

    T_ref_C: float = 25.0
    Q10: float = 2.0
    pH_opt: float = 7.0
    pH_sigma: float = 1.5

    Y_XC: float = 0.5

    protein_min: float = 0.20
    protein_max: float = 0.60
    lipid_min: float = 0.10
    lipid_max: float = 0.40
    K_protein_N: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)

def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def _f_saturation(x: float, k: float) -> float:
    if k <= 0:
        return 1.0
    return float(x) / (float(k) + float(x) + 1e-12)

def _f_temp(T: float, T_ref: float, Q10: float) -> float:
    val = float(Q10) ** ((float(T) - float(T_ref)) / 10.0)
    return _clip(val, 0.0, 3.0)

def _f_ph(pH: float, pH_opt: float, sigma: float) -> float:
    sigma = float(sigma) if sigma else 1.0
    return float(math.exp(-((float(pH) - float(pH_opt)) / sigma) ** 2))

def _compute_composition(N_mM: float, C_gL: float, params: ModelParams) -> Tuple[float, float, float]:
    N_status = _f_saturation(max(N_mM, 0.0), params.K_protein_N)
    protein = params.protein_min + (params.protein_max - params.protein_min) * N_status

    C_status = _f_saturation(max(C_gL, 0.0), params.K_C)
    lipid = params.lipid_min + (params.lipid_max - params.lipid_min) * (1.0 - N_status) * C_status

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
    X0_gL: float,
    pH: float,
    light_uE_m2_s: float,
    temperature_C: float,
    dissolved_oxygen_mgL: Optional[float],
    gas_co2_percent: Optional[float],
    mixing_rpm: Optional[float],
    duration_d: float = 7.0,
    dt_d: float = 0.02,
    params: Optional[ModelParams] = None,
) -> pd.DataFrame:
    p = params or ModelParams()

    t = 0.0
    X = max(float(X0_gL), 1e-9)
    C = max(float(C0_gL), 0.0)
    N = max(float(N0_mM), 0.0)

    rows = []
    steps = int(max(1, round(duration_d / dt_d)))

    for _ in range(steps + 1):
        protein_f, lipid_f, carb_f = _compute_composition(N, C, p)

        fI = _f_saturation(max(float(light_uE_m2_s), 0.0), p.K_I) if trophic_mode != "heterotrophic" else 1.0
        if trophic_mode == "autotrophic":
            fC = _f_saturation(max(float(gas_co2_percent or 0.0), 0.0), p.K_CO2)
        else:
            fC = _f_saturation(max(C, 0.0), p.K_C)

        fN = _f_saturation(max(N, 0.0), p.K_N)
        fT = _f_temp(float(temperature_C), p.T_ref_C, p.Q10)
        fpH = _f_ph(float(pH), p.pH_opt, p.pH_sigma)
        fO2 = _f_saturation(max(float(dissolved_oxygen_mgL) if dissolved_oxygen_mgL is not None else 10.0, 0.0), p.K_O2)
        fmix = _f_saturation(max(float(mixing_rpm) if mixing_rpm is not None else 300.0, 0.0), p.K_rpm)

        mu = p.mu_max_d1 * fI * fC * fN * fT * fpH * fO2 * fmix
        mu = _clip(mu, 0.0, p.mu_max_d1)

        dXdt = mu * X * (1.0 - X / max(p.X_max_gL, 1e-6))
        dX = dXdt * dt_d
        X_next = max(0.0, X + dX)

        # carbon usage
        dC = 0.0
        if trophic_mode != "autotrophic":
            dC = -(1.0 / max(p.Y_XC, 1e-9)) * dX

        # nitrogen usage based on protein in new biomass
        mmolN_per_gX = (protein_f / 6.25) / 0.014
        dN = -mmolN_per_gX * dX

        C_next = max(0.0, C + dC)
        N_next = max(0.0, N + dN)

        carbon_uptake_rate = max(0.0, -dC / dt_d) if trophic_mode != "autotrophic" else 0.0
        lipid_prod_rate = max(0.0, dXdt) * lipid_f
        flux_gly = carbon_uptake_rate / 10.0
        flux_ppp = flux_gly * (0.2 + 0.8 * (lipid_prod_rate / (lipid_prod_rate + 0.05)))
        flux_tca = max(0.0, dXdt) / 5.0

        rows.append({
            "time_d": t,
            "biomass_gL": X,
            "carbon_gL": C,
            "nitrogen_mM": N,
            "mu_d1": mu,
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
        X, C, N = X_next, C_next, N_next

    return pd.DataFrame(rows)
