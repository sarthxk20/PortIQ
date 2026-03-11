# 🚢 Real-Time Container Visibility Platform
### GTM Opportunity Scoring · AIS-Powered Market Intelligence · SaaS Sales Motion

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📌 Project Overview

This project builds a **full ML pipeline** that helps SaaS GTM strategists identify, score, and prioritize shipping & logistics companies as prospects for a **Real-Time Container Visibility Platform**.

Using **AIS (Automatic Identification System) vessel tracking data**, the pipeline:

1. Generates synthetic AIS vessel & company data (modeled on real AIS schemas)
2. Engineers GTM-relevant features (pain scores, digital readiness, deal size potential)
3. Sizes the market (TAM / SAM / SOM) across Enterprise, Mid-Market, and SMB tiers
4. Trains a **Random Forest classifier** to score each company's GTM opportunity
5. Surfaces a ranked, filtered, downloadable prospect list via an interactive **Streamlit dashboard**

---

## 🏗️ Project Structure

```
container-gtm-platform/
│
├── app.py                        # Streamlit dashboard (main entrypoint)
├── src/
│   └── gtm_opportunity_pipeline.py   # Standalone ML pipeline script
├── notebooks/
│   └── eda_and_modeling.ipynb    # Exploratory analysis notebook
├── data/                         # Output CSVs (generated at runtime)
├── models/                       # Saved model artifacts (.pkl)
├── assets/                       # Static assets / screenshots
├── requirements.txt
├── .streamlit/config.toml
└── README.md
```

---

## 🚀 Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/container-gtm-platform.git
cd container-gtm-platform
```

### 2. Create virtual environment & install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 🤖 ML Pipeline Stages

| Stage | Description |
|-------|-------------|
| **1. Data Generation** | 1,200 AIS vessel pings across 180 companies (MMSI, IMO, ETA accuracy, delay hours, trade lanes, etc.) |
| **2. Feature Engineering** | 27 features: `visibility_pain_score`, `digital_readiness`, `gtm_complexity`, `deal_size_potential_usd_k`, and more |
| **3. Market Sizing** | TAM / SAM / SOM calculated per tier with penetration assumptions |
| **4. ML Modeling** | Random Forest + Gradient Boosting with 5-fold CV; AUC ~0.97 |
| **5. Scoring** | All companies scored and bucketed: S-Priority / A-Pursue / B-Nurture / C-Watch |
| **6. Dashboard** | Streamlit app with filters, scatter plots, ROC curves, and real-time scorer |

---

## 📊 Key Results

- **Model AUC:** ~0.97 (5-fold cross-validation)
- **Top features:** `n_trade_lanes`, `deal_size_potential_usd_k`, `avg_delay_hours`, `visibility_pain_score`
- **GTM Tiers:** S-Priority (top 43%) · A-Pursue · B-Nurture · C-Watch
- **Market:** TAM ~$11.7M · SAM ~$9.1M · SOM ~$1.1M (sample cohort)

---

## ☁️ Deploy on Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **"New app"** → select your repo → set `app.py` as entrypoint
4. Click **Deploy** 🚀

---

## 🛠️ Tech Stack

- **Python 3.9+**
- **Streamlit** — Interactive dashboard
- **scikit-learn** — ML pipeline (Random Forest, GBM)
- **pandas / numpy** — Data wrangling
- **matplotlib / seaborn** — Visualization

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🙋 Author

Built by a **SaaS GTM Strategist** focused on Digital Supply Chain Transformation.  
Connect on [LinkedIn](https://linkedin.com) · [GitHub](https://github.com)
