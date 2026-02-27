# Microalgae Growth DB (Streamlit)

中文在前 / English below.

## 功能
- 在 Streamlit 页面上传 **Excel（多 Sheet）或 CSV（多文件）** 来更新微藻生长数据库
- 数据会写入数据库（默认 SQLite；也支持 Postgres）
- 可浏览表、预览数据、下载导出 CSV
- 提供空模板 Excel：按 schema 的列名填就能导入

## 快速开始（本地）
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 数据库后端
### 1) 默认：SQLite（本地）
- 默认写入：`data/microalgae.db`

### 2) 可选：Postgres（推荐用于 Streamlit Cloud 持久化）
在 Streamlit Cloud 的 **Secrets** 里加：

```toml
DATABASE_URL="postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
```

本地也可以设置环境变量：
```bash
export DATABASE_URL="postgresql+psycopg2://..."
```

## 部署到 Streamlit Community Cloud
1. 把本仓库推到 GitHub
2. Streamlit Cloud 选择该仓库，入口文件选 `app.py`
3. （可选）在 Secrets 中填 `DATABASE_URL` 连接 Postgres

---

## English

## Features
- Upload **Excel (multi-sheet)** or **CSV (multiple files)** to update a microalgae growth database
- Writes into a database (default SQLite; Postgres supported)
- Browse tables, preview rows, export CSV
- Download an empty Excel template that matches the schema

## Local Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Database backend
### SQLite (default)
- Writes to `data/microalgae.db`

### Postgres (recommended for Streamlit Cloud persistence)
Add this to Streamlit Cloud **Secrets**:
```toml
DATABASE_URL="postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME"
```

## Deploy to Streamlit Community Cloud
1. Push this repo to GitHub
2. In Streamlit Cloud, pick the repo and set the main file to `app.py`
3. (Optional) set `DATABASE_URL` in Secrets for Postgres persistence
