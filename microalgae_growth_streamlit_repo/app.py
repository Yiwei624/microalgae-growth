from __future__ import annotations

import os
import streamlit as st
import pandas as pd

from src.db import (
    get_engine, init_db, list_tables, table_count, read_table,
    upsert_study_df, upsert_organism_df,
    insert_experiments, insert_outcomes, insert_timeseries,
    read_training_join
)
from src.io_utils import read_excel_sheets, read_csv_files
from src.template import build_empty_template, TEMPLATE_COLUMNS
from src.validators import validate_df
from src.model import ModelParams, simulate

st.set_page_config(page_title="Microalgae Growth DB + Predictor", layout="wide")

# Secrets -> env
try:
    if "DATABASE_URL" in st.secrets and not os.environ.get("DATABASE_URL"):
        os.environ["DATABASE_URL"] = str(st.secrets["DATABASE_URL"])
except Exception:
    pass

engine = get_engine()

st.title("🧫 Microalgae Growth Database / 微藻生长数据库")
st.markdown(
    """
**GitHub:** `https://github.com/<YOUR-ACCOUNT>/microalgae-growth-db`  
> Replace the URL above with your repo link after you push to GitHub.
"""
)

# Sidebar (no indentation pitfalls)
st.sidebar.header("Database / 数据库")
if os.environ.get("DATABASE_URL", "").strip():
    st.sidebar.success("Backend: Postgres / 远程数据库")
else:
    st.sidebar.info("Backend: SQLite (local) / 本地数据库: data/microalgae.db")

if st.sidebar.button("Initialize / Create Tables\n初始化/建表", type="primary"):
    init_db(engine)
    st.toast("Database initialized / 已建表", icon="✅")

st.sidebar.divider()
st.sidebar.caption("Tip: For Streamlit Cloud persistence, use Postgres via `DATABASE_URL` in Secrets.")

tab_upload, tab_browse, tab_predict, tab_quality, tab_help = st.tabs([
    "Upload / Update 上传更新",
    "Browse 浏览",
    "Predict / 模型预测",
    "Quality 质控",
    "Help 帮助",
])

with tab_upload:
    st.subheader("Upload your data (Excel or CSV) / 上传数据（Excel 或 CSV）")
    st.markdown(
        """
**推荐：** 上传一个 Excel（多 sheet：study / organism / media / reactor / experiment / outcome / timeseries）  
或上传多个 CSV 文件，文件名必须为：`study.csv`, `organism.csv`, `media.csv`, `reactor.csv`, `experiment.csv`, `outcome.csv`, `timeseries.csv`
"""
    )

    template_bytes = build_empty_template()
    st.download_button(
        "⬇️ Download Excel Template / 下载模板",
        data=template_bytes,
        file_name="microalgae_db_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True
    )

    st.divider()

    mode = st.radio(
        "Import mode / 导入模式",
        options=[
            "Append (add new rows) / 追加",
            "Upsert dimensions (study/organism) + append facts / 维表更新+事实表追加",
        ],
        index=1
    )

    uploaded_excel = st.file_uploader(
        "Upload Excel (multi-sheet) / 上传 Excel（多 Sheet）",
        type=["xlsx"],
        accept_multiple_files=False
    )
    uploaded_csvs = st.file_uploader(
        "Or upload multiple CSV files / 或上传多个 CSV 文件",
        type=["csv"],
        accept_multiple_files=True
    )

    if st.button("🚀 Import / 导入", type="primary", use_container_width=True):
        init_db(engine)

        if uploaded_excel is None and not uploaded_csvs:
            st.warning("Please upload an Excel or CSV files first. / 请先上传 Excel 或 CSV。")
            st.stop()

        if uploaded_excel is not None:
            sheets = read_excel_sheets(uploaded_excel.getvalue())
            source_label = f"excel:{uploaded_excel.name}"
        else:
            files = [(f.name, f.getvalue()) for f in uploaded_csvs]
            sheets = read_csv_files(files)
            source_label = "csv:multiple"

        known = set(TEMPLATE_COLUMNS.keys())
        recognized = {k: v for k, v in sheets.items() if k in known}
        ignored = [k for k in sheets.keys() if k not in known]
        if ignored:
            st.info(f"Ignored sheets/files: {ignored}")

        all_errors, all_warnings = [], []
        for t, df in recognized.items():
            errs, warns = validate_df(t, df)
            all_errors += [f"[{t}] {e}" for e in errs]
            all_warnings += [f"[{t}] {w}" for w in warns]

        if all_errors:
            st.error("Validation failed / 校验失败")
            st.write("\n".join(all_errors))
            st.stop()

        if all_warnings:
            st.warning("Warnings / 警告（不阻止导入）")
            for w in all_warnings:
                st.write("- " + w)

        msgs = []

        if "study" in recognized and len(recognized["study"]) > 0:
            if "Upsert" in mode:
                ins, upd = upsert_study_df(engine, recognized["study"])
                msgs.append(f"study: inserted {ins}, updated {upd}")
            else:
                recognized["study"].to_sql("study", engine, if_exists="append", index=False)
                msgs.append(f"study: appended {len(recognized['study'])}")

        if "organism" in recognized and len(recognized["organism"]) > 0:
            if "Upsert" in mode:
                ins, upd = upsert_organism_df(engine, recognized["organism"])
                msgs.append(f"organism: inserted {ins}, updated {upd}")
            else:
                recognized["organism"].to_sql("organism", engine, if_exists="append", index=False)
                msgs.append(f"organism: appended {len(recognized['organism'])}")

        if "media" in recognized and len(recognized["media"]) > 0:
            recognized["media"].to_sql("media", engine, if_exists="append", index=False)
            msgs.append(f"media: appended {len(recognized['media'])}")

        if "reactor" in recognized and len(recognized["reactor"]) > 0:
            recognized["reactor"].to_sql("reactor", engine, if_exists="append", index=False)
            msgs.append(f"reactor: appended {len(recognized['reactor'])}")

        if "experiment" in recognized and len(recognized["experiment"]) > 0:
            ins, upd = insert_experiments(engine, recognized["experiment"], source_label=source_label)
            msgs.append(f"experiment: inserted {ins}, updated {upd}")

        if "outcome" in recognized and len(recognized["outcome"]) > 0:
            ins = insert_outcomes(engine, recognized["outcome"], source_label=source_label)
            msgs.append(f"outcome: inserted {ins}")

        if "timeseries" in recognized and len(recognized["timeseries"]) > 0:
            ins = insert_timeseries(engine, recognized["timeseries"], source_label=source_label)
            msgs.append(f"timeseries: inserted {ins}")

        st.success("Import completed / 导入完成 ✅")
        for m in msgs:
            st.write("- " + m)

with tab_browse:
    st.subheader("Browse tables / 浏览数据库表")
    init_db(engine)

    try:
        tables = list_tables(engine)
    except Exception:
        tables = []

    if not tables:
        st.info("No tables found. Click 'Initialize' first. / 先点初始化。")
    else:
        table = st.selectbox("Select table / 选择表", options=tables, index=0)
        n = table_count(engine, table)
        st.caption(f"Rows / 行数: {n}")
        df = read_table(engine, table, limit=2000)
        st.dataframe(df, use_container_width=True, height=520)
        st.download_button(
            "⬇️ Download preview CSV / 下载预览CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name=f"{table}_preview.csv",
            mime="text/csv",
            use_container_width=True
        )

with tab_predict:
    st.subheader("Predict growth & composition / 预测生长与组成（MVP）")
    st.caption("Flux charts are proxy indices (heuristic), not real GEM/FBA flux. / 通量图为代理指标（演示），非真实 FBA。")

    c1, c2 = st.columns([1, 1])
    with c1:
        trophic_mode = st.selectbox("Trophic mode / 营养模式", ["mixotrophic", "heterotrophic", "autotrophic"], index=0)
        carbon_source = st.selectbox("Carbon source / 碳源（类型）", ["glucose", "acetate", "glycerol", "none (CO2 only)"], index=0)
        C0 = st.number_input("Initial carbon (g/L) / 初始碳源浓度 (g/L)", min_value=0.0, value=10.0, step=0.5)
        nitrogen_source = st.selectbox("Nitrogen source / 氮源（类型）", ["nitrate", "ammonium", "urea"], index=0)
        N0 = st.number_input("Initial nitrogen as N (mmol/L) / 初始氮（以 N 计, mmol/L）", min_value=0.0, value=5.0, step=0.5)
        X0 = st.number_input("Initial biomass (g/L) / 初始生物量 (g/L)", min_value=0.0, value=0.1, step=0.05)
    with c2:
        pH = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0, step=0.1)
        I = st.number_input("Light intensity (μmol m⁻² s⁻¹) / 光照强度", min_value=0.0, value=150.0, step=10.0)
        T = st.number_input("Temperature (°C) / 温度", min_value=-10.0, max_value=60.0, value=25.0, step=1.0)
        DO = st.number_input("Dissolved O2 (mg/L) / 溶解氧", min_value=0.0, value=8.0, step=0.5)
        CO2 = st.number_input("Gas CO2 (%) / 气相 CO2(%)", min_value=0.0, max_value=100.0, value=2.0, step=0.2)
        rpm = st.number_input("Mixing (rpm) / 搅拌转速", min_value=0.0, value=300.0, step=10.0)

    st.divider()
    duration = st.number_input("Duration (days) / 模拟天数", min_value=0.5, value=7.0, step=0.5)
    use_db_guess = st.checkbox("Guess parameters from your DB / 从数据库估计参数（快速）", value=True)

    params = ModelParams()
    if use_db_guess:
        try:
            init_db(engine)
            train = read_training_join(engine)
            if len(train) > 5:
                mu_obs = pd.to_numeric(train.get("mu_d1"), errors="coerce").dropna()
                x_obs = pd.to_numeric(train.get("biomass_gL"), errors="coerce").dropna()
                p_obs = pd.to_numeric(train.get("protein_pct_dw"), errors="coerce").dropna()
                l_obs = pd.to_numeric(train.get("lipid_pct_dw"), errors="coerce").dropna()
                if len(mu_obs) > 0:
                    params.mu_max_d1 = float(mu_obs.quantile(0.9))
                if len(x_obs) > 0:
                    params.X_max_gL = float(max(5.0, x_obs.max() * 1.3))
                if len(p_obs) > 0:
                    params.protein_max = float(min(0.8, p_obs.max() / 100.0))
                if len(l_obs) > 0:
                    params.lipid_max = float(min(0.8, l_obs.max() / 100.0))
        except Exception:
            pass

    with st.expander("Model parameters / 模型参数（可选）", expanded=False):
        st.write("Keep defaults unless you want to tune. / 默认即可。")
        col1, col2, col3 = st.columns(3)
        with col1:
            params.mu_max_d1 = st.number_input("mu_max (1/day)", min_value=0.0, value=float(params.mu_max_d1), step=0.1)
            params.X_max_gL = st.number_input("X_max (g/L)", min_value=0.1, value=float(params.X_max_gL), step=1.0)
        with col2:
            params.K_I = st.number_input("K_I (μmol m⁻² s⁻¹)", min_value=0.0, value=float(params.K_I), step=10.0)
            params.K_C = st.number_input("K_C (g/L)", min_value=0.0, value=float(params.K_C), step=0.2)
            params.K_N = st.number_input("K_N (mmol/L)", min_value=0.0, value=float(params.K_N), step=0.2)
        with col3:
            params.pH_opt = st.number_input("pH_opt", min_value=0.0, max_value=14.0, value=float(params.pH_opt), step=0.1)
            params.pH_sigma = st.number_input("pH_sigma", min_value=0.1, value=float(params.pH_sigma), step=0.1)
            params.Q10 = st.number_input("Q10", min_value=0.5, value=float(params.Q10), step=0.1)

    if st.button("🧪 Simulate / 模拟预测", type="primary", use_container_width=True):
        C0_use = 0.0 if carbon_source.startswith("none") else float(C0)
        df_sim = simulate(
            trophic_mode=trophic_mode,
            carbon_source=carbon_source,
            C0_gL=C0_use,
            nitrogen_source=nitrogen_source,
            N0_mM=float(N0),
            X0_gL=float(X0),
            pH=float(pH),
            light_uE_m2_s=float(I),
            temperature_C=float(T),
            dissolved_oxygen_mgL=float(DO),
            gas_co2_percent=float(CO2),
            mixing_rpm=float(rpm),
            duration_d=float(duration),
            dt_d=0.02,
            params=params
        )

        st.success("Simulation done / 模拟完成 ✅")

        st.write("### Biomass & content / 生物量与组成")
        cA, cB = st.columns(2)
        with cA:
            st.line_chart(df_sim.set_index("time_d")[["biomass_gL"]])
        with cB:
            st.line_chart(df_sim.set_index("time_d")[["protein_pct_dw", "lipid_pct_dw", "carb_pct_dw"]])

        st.write("### Metabolic flux (proxy) / 代谢通量（代理指标）")
        st.line_chart(df_sim.set_index("time_d")[["flux_glycolysis", "flux_ppp", "flux_tca"]])

        st.download_button(
            "⬇️ Download simulation CSV / 下载模拟结果 CSV",
            data=df_sim.to_csv(index=False).encode("utf-8"),
            file_name="simulation_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.dataframe(df_sim.head(200), use_container_width=True, height=320)

with tab_quality:
    st.subheader("Quick quality checks / 快速质控")
    init_db(engine)
    try:
        tables = list_tables(engine)
    except Exception:
        tables = []
    if not tables:
        st.info("Initialize DB first. / 请先初始化数据库。")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("study rows", table_count(engine, "study") if "study" in tables else 0)
        with c2:
            st.metric("experiment rows", table_count(engine, "experiment") if "experiment" in tables else 0)
        with c3:
            st.metric("outcome rows", table_count(engine, "outcome") if "outcome" in tables else 0)

with tab_help:
    st.subheader("How to use / 使用说明")
    st.markdown(
        """
### 中文
- 这个 App = **数据库 + 预测模型（MVP）**
- Predict：输入培养条件 → 输出生物量/蛋白/油脂曲线 + 3 条代理通量曲线  
- 若你要真正的 FBA/dFBA 代谢通量图 + “点 ALA 自动出补料方案”，下一步需要接 COBRApy + 物种 GEM。

### English
- This app = **database + predictor (MVP)**
- Predict: input conditions → biomass/protein/lipid curves + 3 proxy flux curves  
- For real FBA/dFBA flux maps + “click ALA → feeding plan”, next step needs COBRApy + a species GEM.
"""
    )
