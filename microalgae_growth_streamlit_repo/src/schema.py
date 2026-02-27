from __future__ import annotations
from sqlalchemy import (
    MetaData, Table, Column, Integer, Text, Float, Boolean, DateTime,
    ForeignKey, UniqueConstraint
)
from sqlalchemy.sql import func

metadata = MetaData()

study = Table(
    "study", metadata,
    Column("study_id", Integer, primary_key=True, autoincrement=True),
    Column("title", Text, nullable=False),
    Column("authors", Text),
    Column("year", Integer),
    Column("journal", Text),
    Column("doi", Text, unique=True),
    Column("url", Text),
    Column("cultivation_focus", Text),
    Column("notes", Text),
)

organism = Table(
    "organism", metadata,
    Column("organism_id", Integer, primary_key=True, autoincrement=True),
    Column("genus", Text, nullable=False),
    Column("species", Text, nullable=False),
    Column("strain", Text),
    Column("taxonomy_id", Text),
    Column("source_collection", Text),
    Column("engineered", Boolean, default=False),
    Column("engineering_notes", Text),
    UniqueConstraint("genus", "species", "strain", name="uq_organism_gss"),
)

media = Table(
    "media", metadata,
    Column("media_id", Integer, primary_key=True, autoincrement=True),
    Column("name", Text),
    Column("mode", Text),  # autotrophic/mixotrophic/heterotrophic
    Column("carbon_source", Text),
    Column("carbon_conc_gL", Float),
    Column("nitrogen_source", Text),
    Column("nitrogen_as_N_mM", Float),
    Column("phosphorus_as_P_mM", Float),
    Column("micronutrients", Text),
    Column("salinity_gL", Float),
    Column("c_to_n_molar", Float),
    Column("notes", Text),
)

reactor = Table(
    "reactor", metadata,
    Column("reactor_id", Integer, primary_key=True, autoincrement=True),
    Column("reactor_type", Text),
    Column("working_volume_L", Float),
    Column("light_path_cm", Float),
    Column("mixing_rpm", Float),
    Column("aeration_vvm", Float),
    Column("gas_co2_percent", Float),
    Column("gas_flow_Lmin", Float),
    Column("notes", Text),
)

experiment = Table(
    "experiment", metadata,
    Column("experiment_id", Integer, primary_key=True, autoincrement=True),
    Column("exp_code", Text, nullable=False, unique=True),

    Column("study_id", Integer, ForeignKey("study.study_id")),
    Column("organism_id", Integer, ForeignKey("organism.organism_id")),
    Column("media_id", Integer, ForeignKey("media.media_id")),
    Column("reactor_id", Integer, ForeignKey("reactor.reactor_id")),

    Column("cultivation_mode", Text),
    Column("trophic_mode", Text),
    Column("temperature_C", Float),
    Column("pH", Float),
    Column("light_uE_m2_s", Float),
    Column("photoperiod_h_on", Float),
    Column("photoperiod_h_off", Float),
    Column("dissolved_oxygen_mgL", Float),
    Column("gas_o2_percent", Float),
    Column("gas_co2_percent", Float),
    Column("agitation", Text),
    Column("mixing_rpm", Float),
    Column("co2_control", Text),
    Column("dilution_rate_d1", Float),
    Column("initial_biomass_gL", Float),
    Column("duration_d", Float),
    Column("replicates_n", Integer),
    Column("stress_type", Text),
    Column("notes", Text),
)

outcome = Table(
    "outcome", metadata,
    Column("outcome_id", Integer, primary_key=True, autoincrement=True),
    Column("experiment_id", Integer, ForeignKey("experiment.experiment_id"), nullable=False),

    Column("mu_d1", Float),
    Column("biomass_gL", Float),
    Column("biomass_prod_gL_d", Float),
    Column("protein_pct_dw", Float),
    Column("protein_prod_gL_d", Float),
    Column("lipid_pct_dw", Float),
    Column("carb_pct_dw", Float),

    Column("measurement_basis", Text),
    Column("stats_type", Text),
    Column("sd_or_se", Float),
    Column("n_points", Integer),
    Column("notes", Text),
)

timeseries = Table(
    "timeseries", metadata,
    Column("ts_id", Integer, primary_key=True, autoincrement=True),
    Column("experiment_id", Integer, ForeignKey("experiment.experiment_id"), nullable=False),
    Column("time_d", Float, nullable=False),
    Column("biomass_gL", Float),
    Column("nitrate_as_N_mM", Float),
    Column("phosphate_as_P_mM", Float),
    Column("protein_pct_dw", Float),
    Column("notes", Text),
)

provenance = Table(
    "provenance", metadata,
    Column("prov_id", Integer, primary_key=True, autoincrement=True),
    Column("table_name", Text, nullable=False),
    Column("record_id", Integer, nullable=False),
    Column("source_type", Text),
    Column("source_label", Text),
    Column("page_number", Text),
    Column("locator_detail", Text),
    Column("extraction_method", Text),
    Column("unit_conversion", Text),
    Column("confidence", Text),
    Column("extracted_by", Text),
    Column("extracted_at", DateTime(timezone=True), server_default=func.now()),
)
