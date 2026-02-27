from __future__ import annotations
from io import BytesIO
from typing import Dict, List

import pandas as pd

TEMPLATE_COLUMNS: Dict[str, List[str]] = {
    "study": [
        "title", "authors", "year", "journal", "doi", "url", "cultivation_focus", "notes"
    ],
    "organism": [
        "genus", "species", "strain", "taxonomy_id", "source_collection", "engineered", "engineering_notes"
    ],
    "media": [
        "name", "mode", "carbon_source", "carbon_conc_gL",
        "nitrogen_source", "nitrogen_as_N_mM", "phosphorus_as_P_mM",
        "micronutrients", "salinity_gL", "c_to_n_molar", "notes"
    ],
    "reactor": [
        "reactor_type", "working_volume_L", "light_path_cm", "mixing_rpm",
        "aeration_vvm", "gas_co2_percent", "gas_flow_Lmin", "notes"
    ],
    "experiment": [
        "exp_code",
        "study_id", "organism_id", "media_id", "reactor_id",
        "study_doi",
        "organism_genus", "organism_species", "organism_strain",
        "media_name", "media_mode", "media_carbon_source", "media_carbon_conc_gL",
        "media_nitrogen_source", "media_nitrogen_as_N_mM", "media_phosphorus_as_P_mM",
        "reactor_reactor_type", "reactor_working_volume_L", "reactor_light_path_cm",
        "reactor_mixing_rpm", "reactor_aeration_vvm", "reactor_gas_co2_percent", "reactor_gas_flow_Lmin",
        "cultivation_mode", "trophic_mode",
        "temperature_C", "pH", "light_uE_m2_s",
        "photoperiod_h_on", "photoperiod_h_off",
        "dissolved_oxygen_mgL", "gas_o2_percent", "gas_co2_percent",
        "agitation", "mixing_rpm", "co2_control",
        "dilution_rate_d1", "initial_biomass_gL", "duration_d",
        "replicates_n", "stress_type", "notes"
    ],
    "outcome": [
        "experiment_id", "exp_code",
        "mu_d1", "biomass_gL", "biomass_prod_gL_d",
        "protein_pct_dw", "protein_prod_gL_d",
        "lipid_pct_dw", "carb_pct_dw",
        "measurement_basis", "stats_type", "sd_or_se", "n_points", "notes"
    ],
    "timeseries": [
        "experiment_id", "exp_code",
        "time_d",
        "biomass_gL", "nitrate_as_N_mM", "phosphate_as_P_mM", "protein_pct_dw",
        "notes"
    ],
}

def build_empty_template() -> BytesIO:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for sheet, cols in TEMPLATE_COLUMNS.items():
            pd.DataFrame(columns=cols).to_excel(writer, sheet_name=sheet, index=False)
    bio.seek(0)
    return bio
