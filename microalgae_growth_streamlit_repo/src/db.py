from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from sqlalchemy import create_engine, text, select, and_
from sqlalchemy.engine import Engine, Connection
from sqlalchemy.pool import StaticPool

from .schema import metadata, study, organism, media, reactor, experiment, outcome, timeseries, provenance

DEFAULT_SQLITE_PATH = Path("data/microalgae.db")


def get_engine() -> Engine:
    """
    Priority:
    1) env DATABASE_URL (set by Streamlit secrets or OS env)
    2) fallback SQLite file

    For SQLite: use StaticPool + timeout to reduce 'database is locked' issues.
    """
    db_url = os.environ.get("DATABASE_URL", "").strip()
    if db_url:
        return create_engine(db_url, pool_pre_ping=True)

    sqlite_url = f"sqlite:///{DEFAULT_SQLITE_PATH.as_posix()}"
    DEFAULT_SQLITE_PATH.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(
        sqlite_url,
        connect_args={"check_same_thread": False, "timeout": 30},
        poolclass=StaticPool,
        pool_pre_ping=True,
    )


def init_db(engine: Engine) -> None:
    metadata.create_all(engine)


def list_tables(engine: Engine) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )).fetchall()
        if rows:
            return [r[0] for r in rows]
        rows = conn.execute(text(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename"
        )).fetchall()
        return [r[0] for r in rows]


def table_count(engine: Engine, table_name: str) -> int:
    with engine.connect() as conn:
        return int(conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar() or 0)


def read_table(engine: Engine, table_name: str, limit: int = 2000) -> pd.DataFrame:
    return pd.read_sql(text(f"SELECT * FROM {table_name} LIMIT :lim"), engine, params={"lim": limit})


def read_training_join(engine: Engine, limit: int = 200000) -> pd.DataFrame:
    """Join experiment + outcome + organism + study (left joins)."""
    q = text("""
    SELECT
        e.*,
        o.mu_d1, o.biomass_gL, o.protein_pct_dw, o.lipid_pct_dw,
        s.doi AS study_doi,
        org.genus AS organism_genus,
        org.species AS organism_species,
        org.strain AS organism_strain
    FROM experiment e
    LEFT JOIN outcome o ON o.experiment_id = e.experiment_id
    LEFT JOIN study s ON s.study_id = e.study_id
    LEFT JOIN organism org ON org.organism_id = e.organism_id
    LIMIT :lim
    """)
    return pd.read_sql(q, engine, params={"lim": limit})


# --------- helpers ---------

def _normalize_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    x = str(x).strip()
    return x if x else None


def _clean_value(v):
    """Convert NaN/empty strings to None so DB matching works reliably."""
    if v is None:
        return None
    # pandas / numpy NaN
    try:
        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, str):
        v2 = v.strip()
        return v2 if v2 else None
    return v


def _keep_value(v) -> bool:
    v = _clean_value(v)
    return v is not None


# --------- upsert dimensions ---------

def upsert_study_df(engine: Engine, df: pd.DataFrame) -> Tuple[int, int]:
    inserted = 0
    updated = 0
    cols = [c for c in df.columns if c in study.c.keys() and c != "study_id"]
    df = df.copy()
    for c in ["doi", "title"]:
        if c in df.columns:
            df[c] = df[c].apply(_normalize_str)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            data = {c: (_clean_value(row[c]) if c in row else None) for c in cols}
            doi = data.get("doi")
            if doi:
                existing = conn.execute(select(study.c.study_id).where(study.c.doi == doi)).fetchone()
                if existing:
                    conn.execute(study.update().where(study.c.study_id == existing[0]).values(**data))
                    updated += 1
                else:
                    conn.execute(study.insert().values(**data))
                    inserted += 1
            else:
                conn.execute(study.insert().values(**data))
                inserted += 1
    return inserted, updated


def upsert_organism_df(engine: Engine, df: pd.DataFrame) -> Tuple[int, int]:
    inserted = 0
    updated = 0
    cols = [c for c in df.columns if c in organism.c.keys() and c != "organism_id"]
    df = df.copy()
    for c in ["genus", "species", "strain"]:
        if c in df.columns:
            df[c] = df[c].apply(_normalize_str)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            data = {c: (_clean_value(row[c]) if c in row else None) for c in cols}
            g = data.get("genus")
            s = data.get("species")
            stn = data.get("strain")
            if not g or not s:
                continue
            cond = and_(organism.c.genus == g, organism.c.species == s, organism.c.strain == stn)
            existing = conn.execute(select(organism.c.organism_id).where(cond)).fetchone()
            if existing:
                conn.execute(organism.update().where(organism.c.organism_id == existing[0]).values(**data))
                updated += 1
            else:
                conn.execute(organism.insert().values(**data))
                inserted += 1
    return inserted, updated


# --------- get-or-create media/reactor (use SAME connection to avoid SQLite locks) ---------

def _get_or_create_media(conn: Connection, row: Dict) -> Optional[int]:
    row = {k: _clean_value(v) for k, v in row.items()}
    # 1) by name
    if row.get("name"):
        ex = conn.execute(select(media.c.media_id).where(media.c.name == row["name"])).fetchone()
        if ex:
            return int(ex[0])

    # 2) soft key by available fields (ignore None)
    key_fields = ["mode", "carbon_source", "carbon_conc_gL", "nitrogen_source", "nitrogen_as_N_mM", "phosphorus_as_P_mM", "salinity_gL"]
    where = []
    for f in key_fields:
        if row.get(f) is not None:
            where.append(media.c[f] == row[f])
    if where:
        ex = conn.execute(select(media.c.media_id).where(and_(*where))).fetchone()
        if ex:
            return int(ex[0])

    # 3) create
    insert_cols = {c: row.get(c) for c in media.c.keys() if c != "media_id"}
    res = conn.execute(media.insert().values(**insert_cols))
    if res.inserted_primary_key:
        return int(res.inserted_primary_key[0])
    return None


def _get_or_create_reactor(conn: Connection, row: Dict) -> Optional[int]:
    row = {k: _clean_value(v) for k, v in row.items()}
    # 1) basic match by type + volume + mixing if present
    where = []
    if row.get("reactor_type") is not None:
        where.append(reactor.c.reactor_type == row["reactor_type"])
    if row.get("working_volume_L") is not None:
        where.append(reactor.c.working_volume_L == row["working_volume_L"])
    if row.get("mixing_rpm") is not None:
        where.append(reactor.c.mixing_rpm == row["mixing_rpm"])
    if where:
        ex = conn.execute(select(reactor.c.reactor_id).where(and_(*where))).fetchone()
        if ex:
            return int(ex[0])

    # 2) fallback by reactor_type only
    if row.get("reactor_type") is not None:
        ex = conn.execute(select(reactor.c.reactor_id).where(reactor.c.reactor_type == row["reactor_type"])).fetchone()
        if ex:
            return int(ex[0])

    # 3) create
    insert_cols = {c: row.get(c) for c in reactor.c.keys() if c != "reactor_id"}
    res = conn.execute(reactor.insert().values(**insert_cols))
    if res.inserted_primary_key:
        return int(res.inserted_primary_key[0])
    return None


def resolve_fk_ids(conn: Connection, exp_row: Dict) -> Dict:
    """Map natural keys in upload into IDs (same-connection, NaN-safe)."""
    out = dict(exp_row)

    # study
    doi = _normalize_str(out.pop("study_doi", None))
    if doi:
        ex = conn.execute(select(study.c.study_id).where(study.c.doi == doi)).fetchone()
        if ex:
            out["study_id"] = int(ex[0])

    # organism
    g = _normalize_str(out.pop("organism_genus", None))
    s = _normalize_str(out.pop("organism_species", None))
    stn = _normalize_str(out.pop("organism_strain", None))
    if g and s:
        ex = conn.execute(select(organism.c.organism_id).where(
            and_(organism.c.genus == g, organism.c.species == s, organism.c.strain == stn)
        )).fetchone()
        if ex:
            out["organism_id"] = int(ex[0])

    # media fields from experiment sheet (prefix media_)
    media_fields = {}
    for k in list(out.keys()):
        if k.startswith("media_"):
            key = k.replace("media_", "")
            val = _clean_value(out.pop(k))
            if val is not None:
                media_fields[key] = val
    if media_fields:
        out["media_id"] = _get_or_create_media(conn, media_fields)

    # reactor fields from experiment sheet (prefix reactor_)
    reactor_fields = {}
    for k in list(out.keys()):
        if k.startswith("reactor_"):
            key = k.replace("reactor_", "")
            val = _clean_value(out.pop(k))
            if val is not None:
                reactor_fields[key] = val
    if reactor_fields:
        out["reactor_id"] = _get_or_create_reactor(conn, reactor_fields)

    return out


# --------- facts insertion ---------

def insert_experiments(engine: Engine, df: pd.DataFrame, uploaded_by: str = "upload", source_label: str = "") -> Tuple[int, int]:
    """Insert experiments. If exp_code exists, update row (simple upsert)."""
    inserted = 0
    updated = 0
    df = df.copy()
    if "exp_code" in df.columns:
        df["exp_code"] = df["exp_code"].astype(str).str.strip()

    with engine.begin() as conn:
        for _, row in df.iterrows():
            r = {k: _clean_value(v) for k, v in row.to_dict().items()}
            r = resolve_fk_ids(conn, r)

            exp_code = _normalize_str(r.get("exp_code"))
            if not exp_code:
                continue

            data = {}
            for c in experiment.c.keys():
                if c == "experiment_id":
                    continue
                if c in r:
                    data[c] = _clean_value(r.get(c))

            existing = conn.execute(select(experiment.c.experiment_id).where(experiment.c.exp_code == exp_code)).fetchone()
            if existing:
                conn.execute(experiment.update().where(experiment.c.experiment_id == existing[0]).values(**data))
                updated += 1
            else:
                conn.execute(experiment.insert().values(**data))
                inserted += 1

    return inserted, updated


def insert_outcomes(engine: Engine, df: pd.DataFrame, source_label: str = "", uploaded_by: str = "upload") -> int:
    inserted = 0
    df = df.copy()
    if "exp_code" in df.columns:
        df["exp_code"] = df["exp_code"].astype(str).str.strip()

    with engine.begin() as conn:
        for _, row in df.iterrows():
            r = {k: _clean_value(v) for k, v in row.to_dict().items()}
            exp_id = r.get("experiment_id")

            if exp_id is None and r.get("exp_code"):
                ex = conn.execute(select(experiment.c.experiment_id).where(experiment.c.exp_code == r["exp_code"])).fetchone()
                if ex:
                    exp_id = int(ex[0])

            if exp_id is None:
                continue

            data = {"experiment_id": int(exp_id)}
            for c in outcome.c.keys():
                if c in ("outcome_id", "experiment_id"):
                    continue
                if c in r:
                    data[c] = _clean_value(r.get(c))

            res = conn.execute(outcome.insert().values(**data))
            oid = int(res.inserted_primary_key[0]) if res.inserted_primary_key else None
            inserted += 1

            if oid:
                conn.execute(provenance.insert().values(
                    table_name="outcome",
                    record_id=oid,
                    source_type="Upload",
                    source_label=source_label or "upload",
                    extraction_method="manual",
                    confidence="medium",
                    extracted_by=uploaded_by
                ))
    return inserted


def insert_timeseries(engine: Engine, df: pd.DataFrame, source_label: str = "", uploaded_by: str = "upload") -> int:
    inserted = 0
    df = df.copy()
    if "exp_code" in df.columns:
        df["exp_code"] = df["exp_code"].astype(str).str.strip()

    with engine.begin() as conn:
        for _, row in df.iterrows():
            r = {k: _clean_value(v) for k, v in row.to_dict().items()}
            exp_id = r.get("experiment_id")

            if exp_id is None and r.get("exp_code"):
                ex = conn.execute(select(experiment.c.experiment_id).where(experiment.c.exp_code == r["exp_code"])).fetchone()
                if ex:
                    exp_id = int(ex[0])

            if exp_id is None:
                continue

            if r.get("time_d") is None:
                continue

            data = {"experiment_id": int(exp_id), "time_d": float(r["time_d"])}
            for c in timeseries.c.keys():
                if c in ("ts_id", "experiment_id", "time_d"):
                    continue
                if c in r:
                    data[c] = _clean_value(r.get(c))

            conn.execute(timeseries.insert().values(**data))
            inserted += 1

    return inserted
