from __future__ import annotations

import os
import streamlit as st
import pandas as pd

from src.db import (
    get_engine, init_db, list_tables, table_count, read_table,
    upsert_study_df, upsert_organism_df,
    insert_experiments, insert_outcomes, insert_timeseries
)
from src.io_utils import read_excel_sheets, read_csv_files
from src.template import build_empty_template, TEMPLATE_COLUMNS
from src.validators import validate_df

st.set_page_config(page_title="Microalgae Growth DB", layout="wide")

# If Streamlit secrets has DATABASE_URL, copy it to env so src.db can pick it up
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


# ---------------- Sidebar (no 'with' block) ----------------
st.sidebar.header("Database / 数据库")
using_url = os.environ.get("DATABASE_URL", "").strip()
if using_url:
    st.sidebar.success("Backend: Postgres / 远程数据库")
    st.sidebar.code(using_url[:120] + ("..." if len(using_url) > 120 else ""))
else:
    st.sidebar.info("Backend: SQLite (local) / 本地数据库: data/microalgae.db")

if st.sidebar.button("Initialize / Create Tables\n初始化/建表", type="primary"):
    init_db(engine)
    st.toast("Database initialized / 已建表", icon="✅")

st.sidebar.divider()
st.sidebar.caption("Tip: For Streamlit Cloud persistence, use Postgres via `DATABASE_URL` in Secrets.")

tab_upload, tab_browse, tab_quality, tab_help = st.tabs([
    "Upload / Update 上传更新", "Browse 浏览", "Quality 质控", "Help 帮助"
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
            "Upsert dimensions (study/organism) + append facts / 维表更新+事实表追加"
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

        st.divider()

        checks = []
        if "experiment" in tables:
            exp = read_table(engine, "experiment", limit=100000)
            missing_code = exp["exp_code"].isna().sum() if "exp_code" in exp.columns else 0
            checks.append(("experiment", "missing exp_code", int(missing_code)))
        if "study" in tables:
            s = read_table(engine, "study", limit=100000)
            missing_doi = s["doi"].isna().sum() if "doi" in s.columns else 0
            checks.append(("study", "missing doi", int(missing_doi)))

        st.write("**Missingness / 缺失检查**")
        st.dataframe(pd.DataFrame(checks, columns=["table", "check", "count"]), use_container_width=True)

        st.write("**Tip / 提示：** If you want stricter QC (unit normalization, duplicate detection), tell me and I’ll add it.")

with tab_help:
    st.subheader("How to use / 使用说明")
    st.markdown(
        """
### 中文
1. 下载模板 Excel，按 sheet 名填写：study / organism / media / reactor / experiment / outcome / timeseries  
2. 最关键：**experiment 里必须有 exp_code**（你自己定义，唯一即可）  
3. outcome / timeseries 可以用 exp_code 关联（不用填 experiment_id）  
4. 上传后去 Browse 查看是否导入成功  
5. Streamlit Cloud 想长期保存数据：请用 Postgres，并在 Secrets 里设置 `DATABASE_URL`

### English
1. Download the Excel template and fill sheets: study / organism / media / reactor / experiment / outcome / timeseries  
2. Most important: **experiment must have exp_code** (your own unique id)  
3. outcome/timeseries can link via exp_code (no need for experiment_id)  
4. After upload, inspect in Browse  
5. For persistence on Streamlit Cloud, use Postgres via `DATABASE_URL` in Secrets
"""
    )
