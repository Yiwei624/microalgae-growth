from __future__ import annotations

import os
import streamlit as st
import pandas as pd

from src.db import (
    get_engine, init_db, list_tables, table_count, read_table,
    upsert_study_df, upsert_organism_df,
    insert_experiments, insert_outcomes, insert_timeseries, insert_signal_timeseries,
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
    st.subheader("Predict growth & composition / 预测生长与组成（v7：碳/氮消耗 + 真空/压力）")
    st.caption(
        "Now includes substrate depletion: N is consumed (when enabled), organic C is consumed for mixo/hetero, "
        "and autotrophic carbon can be modeled as a DIC pool consumed by growth and replenished from gas via kLa. "
        "Vacuum/pressure affects pCO2 = yCO2·P. "
        "/ 已加入底物消耗：氮可随时间耗尽；混养/异养有机碳消耗；自养可用DIC无机碳池(消耗+传质补给)。真空/压力影响pCO2。"
    )

    # --- switches ---
    s1, s2, s3, s4, s5, s6, s7 = st.columns(7)
    with s1:
        enable_self_shading = st.checkbox("Self-shading / 自遮光", value=True)
    with s2:
        enable_pH_drift = st.checkbox("pH drift / pH漂移", value=False)
    with s3:
        enable_N_dynamics = st.checkbox("N depletion / 氮耗尽", value=True)
    with s4:
        enable_P_dynamics = st.checkbox("P depletion / 磷耗尽", value=False)
    with s5:
        enable_mass_transfer = st.checkbox("Mass transfer / 传质(kLa)", value=False)
    with s6:
        enable_logistic_cap = st.checkbox("Logistic cap / 承载上限", value=True)
    with s7:
        enable_DIC_pool = st.checkbox("DIC pool / 无机碳池", value=True)

    c1, c2 = st.columns([1, 1])

    with c1:
        trophic_mode = st.selectbox(
            "Trophic mode / 营养模式",
            ["autotrophic", "mixotrophic", "heterotrophic"],
            index=0
        )

        carbon_source = st.selectbox(
            "Carbon source / 碳源（类型）",
            ["none (CO2 only)", "glucose", "acetate", "glycerol"],
            index=0
        )

        C0 = st.number_input(
            "Initial organic carbon (g/L) / 初始有机碳(g/L)（自养填0）",
            min_value=0.0, value=0.0 if carbon_source.startswith("none") else 10.0, step=0.5
        )

        nitrogen_source = st.selectbox("Nitrogen source / 氮源（类型）", ["nitrate", "ammonium", "urea"], index=0)
        N0 = st.number_input("Initial N (mmol/L) / 初始氮（以N计, mmol/L）", min_value=0.0, value=5.0, step=0.5)

        P0_in = st.number_input("Initial P (mmol/L) / 初始磷（以P计, mmol/L）(可选)", min_value=0.0, value=0.05, step=0.01)

        X0 = st.number_input(
            "Initial biomass (g/L) / 初始生物量 (g/L)",
            min_value=0.0, value=0.0023, step=0.0001, format="%.6f"
        )

        # Optional DIC initial
        DIC0_in = st.number_input(
            "Initial DIC (mmol/L) / 初始无机碳DIC(mM, 可选)",
            min_value=0.0, value=0.0, step=1.0
        )
        DIC0_use = None if (DIC0_in <= 0) else float(DIC0_in)

    with c2:
        pH0 = st.number_input("Initial pH / 初始pH", min_value=0.0, max_value=14.0, value=8.2, step=0.1)

        use_pH_end = st.checkbox("Use final pH / 使用末端pH", value=False)
        pH_end = st.number_input("Final pH / 末端pH", min_value=0.0, max_value=14.0, value=9.6, step=0.1)
        pH_end_use = pH_end if (enable_pH_drift and use_pH_end) else None

        I0 = st.number_input("Light intensity I0 (μmol m⁻² s⁻¹) / 光强", min_value=0.0, value=300.0, step=10.0)
        T = st.number_input("Temperature (°C) / 温度", min_value=-10.0, max_value=60.0, value=25.0, step=1.0)

        DO = st.number_input("Dissolved O2 (mg/L) / 溶解氧(可选)", min_value=0.0, value=8.0, step=0.5)
        CO2 = st.number_input("Gas CO2 (%) / 气相CO2(%)", min_value=0.0, max_value=100.0, value=78.9, step=0.2)

        use_CO2_end = st.checkbox("Use CO2 end / 使用末端CO2", value=False)
        CO2_end = st.number_input("CO2 end (%) / 末端CO2(%)", min_value=0.0, max_value=100.0, value=float(CO2), step=0.2)
        CO2_end_use = CO2_end if use_CO2_end else None

        # Vacuum / pressure
        vacuum_cmHg = st.number_input("Vacuum (cmHg) / 真空度(cmHg)", min_value=0.0, value=0.0, step=1.0)
        vacuum_is_gauge = st.checkbox("Vacuum is gauge (below atm) / 真空表压(相对大气)", value=True)
        # 1 cmHg ≈ 1.33322 kPa
        vac_kPa = float(vacuum_cmHg) * 1.33322
        if vacuum_is_gauge:
            pressure_kPa = max(0.1, 101.3 - vac_kPa)
        else:
            pressure_kPa = max(0.1, vac_kPa)
        st.info(f"Pressure ≈ {pressure_kPa:.1f} kPa / 压力≈{pressure_kPa:.1f} kPa")

        rpm = st.number_input("Mixing (rpm) / 搅拌转速（仅用于估算kLa）", min_value=0.0, value=50.0, step=10.0)
        kla_d1_in = st.number_input("kLa (1/day) / kLa(1/天)（可选）", min_value=0.0, value=0.0, step=1.0)
        kla_d1_use = kla_d1_in if (enable_mass_transfer and kla_d1_in > 0) else None

    st.divider()

    duration = st.number_input("Duration (days) / 模拟天数", min_value=0.5, value=17.0, step=0.5)
    dt = st.number_input("Time step (days) / 时间步长(天)", min_value=0.001, value=0.02, step=0.005)

    # Derived indicators (proxy C/N)
    if N0 > 0:
        if trophic_mode == "autotrophic":
            cn_proxy = CO2 / N0
            st.info(f"Proxy CO2/N = {cn_proxy:.2f} (% per mmol/L). / 代理 CO2/N = {cn_proxy:.2f}")
        else:
            cn_proxy = C0 / N0
            st.info(f"Proxy C/N = {cn_proxy:.2f} (g/L per mmol/L). / 代理 C/N = {cn_proxy:.2f}")
    else:
        st.warning("N0 = 0 will stop growth in this model. / N0=0 会导致生长被完全限制。")

    use_db_guess = st.checkbox("Guess parameters from your DB / 从数据库估计参数（快速）", value=True)

    params = ModelParams()

    if use_db_guess:
        try:
            init_db(engine)
            train = read_training_join(engine)
            if len(train) > 5:
                mu_obs = pd.to_numeric(train.get("mu_d1"), errors="coerce").dropna()
                x_obs = pd.to_numeric(train.get("biomass_gL"), errors="coerce").dropna()
                if len(mu_obs) > 0:
                    params.mu_max_d1 = float(mu_obs.quantile(0.9))
                if len(x_obs) > 0:
                    params.X_max_gL = float(max(1.0, x_obs.max() * 1.2))
        except Exception:
            pass

    with st.expander("Model parameters / 模型参数（可选）", expanded=False):
        colA, colB, colC = st.columns(3)
        with colA:
            params.mu_max_d1 = st.number_input("mu_max (1/day)", min_value=0.0, value=float(params.mu_max_d1), step=0.1)
            params.X_max_gL = st.number_input("X_max (g/L)", min_value=0.1, value=float(params.X_max_gL), step=1.0)
            params.lag_d = st.number_input("lag (days) / 适应期(天)", min_value=0.0, value=float(params.lag_d), step=0.1)
        with colB:
            params.K_I = st.number_input("K_I (μmol m⁻² s⁻¹)", min_value=0.0, value=float(params.K_I), step=10.0)
            params.k_shade = st.number_input("k_shade (per g/L/cm) / 自遮光系数", min_value=0.0, value=float(params.k_shade), step=0.01)
            params.light_path_cm = st.number_input("light_path (cm) / 光程(cm)", min_value=0.1, value=float(params.light_path_cm), step=0.5)
            params.Q10 = st.number_input("Q10", min_value=0.5, value=float(params.Q10), step=0.1)
            params.T_ref_C = st.number_input("T_ref (°C)", min_value=-10.0, max_value=60.0, value=float(params.T_ref_C), step=1.0)
        with colC:
            params.K_N = st.number_input("K_N (mmol/L)", min_value=0.0, value=float(params.K_N), step=0.2)
            params.K_P = st.number_input("K_P (mmol/L)", min_value=0.0, value=float(params.K_P), step=0.01)
            params.pH_opt = st.number_input("pH_opt", min_value=0.0, max_value=14.0, value=float(params.pH_opt), step=0.1)
            params.pH_sigma = st.number_input("pH_sigma", min_value=0.1, value=float(params.pH_sigma), step=0.1)
            params.pH_carbon_crit = st.number_input("pH_carbon_crit / 高pH碳限制阈值", min_value=0.0, max_value=14.0, value=float(params.pH_carbon_crit), step=0.1)
            params.pH_carbon_alpha = st.number_input("pH_carbon_alpha / 陡峭度", min_value=0.1, value=float(params.pH_carbon_alpha), step=0.2)

        st.markdown("**Autotrophic carbon pool params / 自养碳池参数（可选）**")
        colD, colE, colF = st.columns(3)
        with colD:
            params.H_CO2_mM_per_kPa = st.number_input("H_CO2 (mM/kPa) / 溶解系数", min_value=0.0, value=float(params.H_CO2_mM_per_kPa), step=0.1)
            params.K_DIC_mM = st.number_input("K_DIC (mM) / DIC半饱和", min_value=0.0, value=float(params.K_DIC_mM), step=0.5)
        with colE:
            params.K_pCO2_kPa = st.number_input("K_pCO2 (kPa) / pCO2半饱和", min_value=0.0, value=float(params.K_pCO2_kPa), step=0.5)
            params.C_req_mM_per_gX = st.number_input("C_req (mM per ΔX) / 碳需求", min_value=0.0, value=float(params.C_req_mM_per_gX), step=5.0)
        with colF:
            params.K_kLa_d1 = st.number_input("K_kLa (1/day) / kLa半饱和", min_value=0.0, value=float(params.K_kLa_d1), step=1.0)
            params.kla_from_rpm_a = st.number_input("kLa a", min_value=0.0, value=float(params.kla_from_rpm_a), step=0.5)
            params.kla_from_rpm_b = st.number_input("kLa b", min_value=0.0, value=float(params.kla_from_rpm_b), step=0.1)

    if st.button("🧪 Simulate / 模拟预测", type="primary", use_container_width=True):
        C0_use = 0.0 if carbon_source.startswith("none") else float(C0)
        P0_use = None if P0_in <= 0 else float(P0_in)

        df_sim = simulate(
            trophic_mode=trophic_mode,
            carbon_source=carbon_source,
            C0_gL=float(C0_use),
            nitrogen_source=nitrogen_source,
            N0_mM=float(N0),
            phosphorus_as_P_mM=P0_use,
            X0_gL=float(X0),
            pH0=float(pH0),
            pH_end=pH_end_use,
            light_uE_m2_s=float(I0),
            temperature_C=float(T),
            dissolved_oxygen_mgL=float(DO) if DO > 0 else None,
            gas_co2_percent=float(CO2),
            gas_co2_end_percent=CO2_end_use,
            pressure_kPa=float(pressure_kPa),
            mixing_rpm=float(rpm),
            kla_d1=kla_d1_use,
            DIC0_mM=DIC0_use,
            duration_d=float(duration),
            dt_d=float(dt),
            params=params,
            enable_self_shading=enable_self_shading,
            enable_pH_drift=enable_pH_drift and use_pH_end,
            enable_N_dynamics=enable_N_dynamics,
            enable_P_dynamics=enable_P_dynamics,
            enable_mass_transfer=enable_mass_transfer,
            enable_logistic_cap=enable_logistic_cap,
            enable_inorganic_carbon_pool=(enable_DIC_pool and trophic_mode == "autotrophic"),
        )

        st.success("Simulation done / 模拟完成 ✅")

        st.write("### Biomass / 生物量")
        st.line_chart(df_sim.set_index("time_d")[["biomass_gL"]])

        st.write("### Substrates / 底物消耗（碳/氮/磷）")
        cols = ["nitrogen_mM"]
        if "DIC_mM" in df_sim.columns:
            cols.append("DIC_mM")
        if "carbon_gL" in df_sim.columns:
            cols.append("carbon_gL")
        if "phosphorus_mM" in df_sim.columns:
            cols.append("phosphorus_mM")
        st.line_chart(df_sim.set_index("time_d")[cols])

        st.write("### Limiting factors / 限制因子（用于看平台原因）")
        st.line_chart(df_sim.set_index("time_d")[["fI", "fC", "fN", "fP", "fpH", "fLag", "fMT"]])

        st.write("### Environment / 环境变量（压力/CO2/pCO2）")
        env_cols = ["pH", "CO2_percent", "pressure_kPa", "pCO2_kPa", "I_eff", "mu_d1"]
        keep = [c for c in env_cols if c in df_sim.columns]
        st.line_chart(df_sim.set_index("time_d")[keep])

        st.write("### Composition / 组成（%DW）")
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
