# Microalgae Growth DB + Predictor (Streamlit)

中文在前 / English below.

## 你得到什么
- ✅ 数据库维护：上传 Excel/CSV → 入库（SQLite / Postgres）
- ✅ 预测模型（MVP）：输入 C/N/光照/温度/pH/气体/搅拌 → 输出生物量/蛋白/油脂曲线
- ✅ 代谢通量（MVP）：输出 TCA / Glycolysis / PPP 的 **proxy flux 指标**（用于演示；非真实 FBA）

> 说明：目前的 flux 是“代理指标（proxy）”，不是基因组尺度模型 (GEM) 的真实 FBA/dFBA 通量。
> 下一步可升级为 COBRApy + GEM（需要你选定物种模型）。

## 本地运行
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud 持久化（强烈建议）
Streamlit Cloud 的本地文件系统不保证长期保存，建议用 Postgres（Neon/Supabase 等）。
在 Streamlit Cloud **Secrets** 里添加：
```toml
DATABASE_URL="postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
```

---

## English

## What you get
- ✅ Database maintenance: upload Excel/CSV → store in SQLite/Postgres
- ✅ Predictor (MVP): input C/N/light/T/pH/gases/mixing → biomass/protein/lipid curves
- ✅ Metabolic flux (MVP): TCA/Glycolysis/PPP **proxy flux indices** (demo; not true FBA)

> Current flux is a proxy (heuristic). Upgrade path: COBRApy + genome-scale model (GEM).

## Local run
```bash
pip install -r requirements.txt
streamlit run app.py
```
