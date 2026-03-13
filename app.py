"""
=============================================================================
  PortIQ — Port Intelligence Platform  v2.0
  GTM Opportunity Scoring — Streamlit Dashboard
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import json
import io
import base64
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve,
                              classification_report, mean_absolute_error, r2_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PALETTE = {
    "primary": "#0A2342", "accent1": "#1B6CA8",
    "accent2": "#27AE60", "accent3": "#E74C3C", "accent4": "#F39C12",
    "bg": "#F8F9FA", "grid": "#E0E0E0",
}
TIER_COLORS = {
    "S — Priority": "#E74C3C",
    "A — Pursue":   "#F39C12",
    "B — Nurture":  "#1B6CA8",
    "C — Watch":    "#95A5A6",
}
TIER_DESCRIPTIONS = {
    "S — Priority": "Immediate outreach — highest potential, high pain, digitally ready.",
    "A — Pursue":   "Active pipeline — strong fit, worth a structured sales motion.",
    "B — Nurture":  "Long-term prospect — educate and build relationship over time.",
    "C — Watch":    "Low priority — monitor for future signals before engaging.",
}

VESSEL_TYPES = ["Container Ship","Bulk Carrier","Tanker","Ro-Ro","General Cargo","Feeder Container","VLCC"]
TIERS        = ["Enterprise","Mid-Market","SMB"]
PORT_NAMES   = ["Shanghai","Singapore","Rotterdam","Los Angeles","Hamburg","Busan",
                "Antwerp","Dubai","New York","Shenzhen","Ningbo","Qingdao","Felixstowe","Valencia","Colombo"]
PORT_COORDS  = {
    "Shanghai":    (121.47, 31.23), "Singapore":   (103.82,  1.26),
    "Rotterdam":   (4.40,  51.89),  "Los Angeles":  (-118.27, 33.74),
    "Hamburg":     (9.99,  53.55),  "Busan":        (129.04, 35.10),
    "Antwerp":     (4.42,  51.23),  "Dubai":        (55.27,  25.20),
    "New York":    (-74.01, 40.71), "Shenzhen":     (114.06, 22.55),
    "Ningbo":      (121.55, 29.88), "Qingdao":      (120.38, 36.07),
    "Felixstowe":  (1.35,  51.96),  "Valencia":     (-0.31,  39.46),
    "Colombo":     (79.86,  6.93),
}
PORT_REGIONS = {
    "Shanghai":"Asia","Singapore":"Asia","Rotterdam":"Europe","Los Angeles":"Americas",
    "Hamburg":"Europe","Busan":"Asia","Antwerp":"Europe","Dubai":"Middle East",
    "New York":"Americas","Shenzhen":"Asia","Ningbo":"Asia","Qingdao":"Asia",
    "Felixstowe":"Europe","Valencia":"Europe","Colombo":"Asia",
}

FEATURES = [
    "fleet_size","annual_teu_volume","n_trade_lanes","n_home_ports",
    "years_in_operation","has_existing_tms","has_edi_integration","digital_maturity",
    "avg_delay_hours","max_delay_hours","avg_eta_accuracy","total_port_calls",
    "avg_data_gaps","n_vessels_active","n_distinct_origins","n_distinct_dests",
    "avg_distance_nm","avg_vessel_age","container_vessel_pct","transcontinental_pct",
    "visibility_pain_score","port_complexity_score","digital_readiness",
    "gtm_complexity","deal_size_potential_usd_k","tam_addressable_teu","tier_encoded",
]

FEATURE_EXPLANATIONS = {
    "fleet_size":                "Number of ships owned — larger fleets mean bigger deals.",
    "annual_teu_volume":         "Total containers shipped per year (1 TEU = one 20-ft container).",
    "n_trade_lanes":             "Number of global routes — more routes = more complexity = more need.",
    "n_home_ports":              "Number of ports the company regularly uses.",
    "years_in_operation":        "How long the company has been in business.",
    "has_existing_tms":          "Whether the company already uses Transport Management Software (1 = Yes).",
    "has_edi_integration":       "Whether the company uses electronic logistics data exchange (1 = Yes).",
    "digital_maturity":          "How digitally advanced the company is, rated 1 (basic) to 5 (advanced).",
    "avg_delay_hours":           "Average hours ships arrive late — high delay = strong need for our platform.",
    "max_delay_hours":           "Worst-case delay ever recorded — indicates operational risk.",
    "avg_eta_accuracy":          "How often arrival time predictions are correct (1.0 = always accurate).",
    "total_port_calls":          "Total port visits per year across all vessels.",
    "avg_data_gaps":             "How often AIS tracking signal goes missing (0 = perfect, 1 = very unreliable).",
    "n_vessels_active":          "Number of vessels tracked in the AIS dataset.",
    "n_distinct_origins":        "Number of unique departure ports used.",
    "n_distinct_dests":          "Number of unique destination ports used.",
    "avg_distance_nm":           "Average voyage distance in nautical miles.",
    "avg_vessel_age":            "Average fleet age — older ships often have worse tracking.",
    "container_vessel_pct":      "Share of the fleet that carries containers.",
    "transcontinental_pct":      "Proportion of voyages crossing continents — higher = more complex.",
    "visibility_pain_score":     "Composite score (0–1) of how urgently a company needs better visibility.",
    "port_complexity_score":     "How many different ports the company juggles.",
    "digital_readiness":         "Overall readiness to adopt our platform.",
    "gtm_complexity":            "How complex the sales motion will be.",
    "deal_size_potential_usd_k": "Estimated annual contract value in USD (thousands).",
    "tam_addressable_teu":       "Total container volume our platform could directly serve.",
    "tier_encoded":              "Company size category encoded as a number for the ML model.",
}

EXPORT_COLS = [
    "company_id","company_tier","fleet_size","annual_teu_volume","n_trade_lanes",
    "avg_delay_hours","avg_eta_accuracy","visibility_pain_score","digital_readiness",
    "gtm_score_proba","gtm_tier_label","deal_size_potential_usd_k",
]

QUARTERS = ["Q1 2024","Q2 2024","Q3 2024","Q4 2024"]

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & STYLING
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="PortIQ — Port Intelligence",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .section-header {
        font-size: 1.1rem; font-weight: 800;
        margin: 1.2rem 0 0.3rem 0;
        border-bottom: 2px solid #1B6CA8;
        padding-bottom: 0.3rem;
    }
    .insight-box {
        background: #EBF4FB;
        border-left: 4px solid #1B6CA8;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        font-size: 0.88rem;
        margin-bottom: 0.8rem;
        color: #1a1a1a;
    }
    .flow-step {
        background: #0A2342;
        color: white;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-size: 0.88rem;
        font-weight: 600;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def score_to_tier(score: float) -> str:
    if score >= 0.80: return "S — Priority"
    if score >= 0.65: return "A — Pursue"
    if score >= 0.40: return "B — Nurture"
    return "C — Watch"

def make_chart(figsize=(7, 3.5)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top","right"]].set_visible(False)
    return fig, ax

def info_box(text: str):
    st.markdown(f'<div class="insight-box">{text}</div>', unsafe_allow_html=True)

def tier_color_row(row):
    """Pandas Styler: color entire row by GTM tier."""
    color_map = {
        "S — Priority": "background-color: #fde8e8; color: #7b1a1a;",
        "A — Pursue":   "background-color: #fef3e2; color: #7b4a00;",
        "B — Nurture":  "background-color: #e8f0fb; color: #0a2342;",
        "C — Watch":    "background-color: #f4f4f4; color: #555;",
    }
    tier = row.get("GTM Tier", "")
    style = color_map.get(tier, "")
    return [style] * len(row)

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE (cached)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_pipeline(n_vessels: int = 1200, n_companies: int = 180, seed: int = 42):
    np.random.seed(seed)

    # ── Companies ─────────────────────────────────────────────────────────────
    companies = []
    for cid in range(n_companies):
        tier = np.random.choice(TIERS, p=[0.15, 0.35, 0.50])
        if tier == "Enterprise":
            fleet_size = np.random.randint(30, 120)
            annual_teu = np.random.randint(500_000, 3_000_000)
            rev        = np.random.uniform(200, 2000)
        elif tier == "Mid-Market":
            fleet_size = np.random.randint(8, 30)
            annual_teu = np.random.randint(50_000, 500_000)
            rev        = np.random.uniform(20, 200)
        else:
            fleet_size = np.random.randint(1, 8)
            annual_teu = np.random.randint(2_000, 50_000)
            rev        = np.random.uniform(1, 20)
        companies.append({
            "company_id":          f"COMP_{cid:04d}",
            "company_tier":        tier,
            "fleet_size":          int(fleet_size),
            "annual_teu_volume":   int(annual_teu),
            "revenue_usd_m":       round(rev, 2),
            "n_trade_lanes":       int(np.random.choice([1,2,3,4,5], p=[0.25,0.30,0.25,0.15,0.05])),
            "n_home_ports":        int(min(len(PORT_NAMES), np.random.randint(2, 10))),
            "years_in_operation":  int(np.random.randint(1, 45)),
            "has_existing_tms":    int(np.random.choice([0,1], p=[0.45,0.55])),
            "has_edi_integration": int(np.random.choice([0,1], p=[0.50,0.50])),
            "digital_maturity":    int(np.random.randint(1, 6)),
        })
    df_companies = pd.DataFrame(companies)

    # ── Vessels ───────────────────────────────────────────────────────────────
    vessels = []
    for vid in range(n_vessels):
        c      = df_companies.iloc[np.random.randint(0, n_companies)]
        vtype  = np.random.choice(VESSEL_TYPES, p=[0.35,0.20,0.18,0.07,0.10,0.08,0.02])
        port_a, port_b = np.random.choice(PORT_NAMES, 2, replace=False)
        dist   = float(np.random.uniform(800, 12_000))
        delay  = float(max(0.0, np.random.normal(12 if c["n_trade_lanes"] >= 3 else 4, 8)))
        vessels.append({
            "vessel_id":           f"VESSEL_{vid:05d}",
            "company_id":          c["company_id"],
            "vessel_type":         vtype,
            "origin_port":         port_a,
            "destination_port":    port_b,
            "origin_region":       PORT_REGIONS[port_a],
            "dest_region":         PORT_REGIONS[port_b],
            "distance_nm":         round(dist, 1),
            "speed_knots":         round(float(np.random.uniform(8, 22)), 1),
            "delay_hours":         round(delay, 1),
            "port_calls_per_year": int(np.random.randint(4, 52)),
            "eta_accuracy_rate":   round(float(np.clip(np.random.normal(0.68, 0.15), 0, 1)), 3),
            "bunker_cost_usd":     round(dist * float(np.random.uniform(25, 50)), 2),
            "cargo_teu":           int(np.random.randint(100, 24_000)) if "Container" in vtype else 0,
            "vessel_age_years":    int(np.random.randint(0, 25)),
            "data_gaps_pct":       round(float(np.random.beta(2, 8)), 3),
            "transceivers_count":  int(np.random.randint(1, 4)),
        })
    df_vessels = pd.DataFrame(vessels)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    agg = df_vessels.groupby("company_id").agg(
        avg_delay_hours      = ("delay_hours",         "mean"),
        max_delay_hours      = ("delay_hours",         "max"),
        avg_eta_accuracy     = ("eta_accuracy_rate",   "mean"),
        total_port_calls     = ("port_calls_per_year", "sum"),
        avg_data_gaps        = ("data_gaps_pct",       "mean"),
        n_vessels_active     = ("vessel_id",           "count"),
        n_distinct_origins   = ("origin_port",         "nunique"),
        n_distinct_dests     = ("destination_port",    "nunique"),
        avg_distance_nm      = ("distance_nm",         "mean"),
        avg_vessel_age       = ("vessel_age_years",    "mean"),
        total_teu            = ("cargo_teu",           "sum"),
        container_vessel_pct = ("vessel_type",         lambda x: x.str.contains("Container").mean()),
        transcontinental_pct = ("origin_region",       lambda x: (x != df_vessels.loc[x.index,"dest_region"]).mean()),
    ).reset_index()

    df = df_companies.merge(agg, on="company_id", how="left")
    num_cols = df.select_dtypes(include="number").columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # ── Feature engineering ───────────────────────────────────────────────────
    df["port_complexity_score"]   = (df["n_distinct_origins"] + df["n_distinct_dests"]) / 2
    df["visibility_pain_score"]   = np.clip(
        (df["avg_delay_hours"] / df["avg_delay_hours"].max()) * 0.35 +
        (1 - df["avg_eta_accuracy"])                          * 0.30 +
        (df["avg_data_gaps"]   / df["avg_data_gaps"].max())   * 0.20 +
        df["transcontinental_pct"]                            * 0.15,
        0, 1).round(4)
    df["tam_addressable_teu"]         = df["total_teu"] * df["container_vessel_pct"]
    df["gtm_complexity"]              = (df["n_trade_lanes"] * df["n_home_ports"] / (df["fleet_size"] + 1)).round(4)
    df["digital_readiness"]           = (df["digital_maturity"]*0.5 + df["has_existing_tms"]*1.5 + df["has_edi_integration"]*1.0).round(4)
    df["deal_size_potential_usd_k"]   = (df["fleet_size"]*1.2 + df["annual_teu_volume"]/10_000*0.8 + df["n_trade_lanes"]*5).round(2)

    # ── Classification labels ─────────────────────────────────────────────────
    composite = (
        df["visibility_pain_score"]                                     * 0.30 +
        (df["fleet_size"]        / df["fleet_size"].max())              * 0.20 +
        (df["annual_teu_volume"] / df["annual_teu_volume"].max())       * 0.15 +
        (df["n_trade_lanes"]     / df["n_trade_lanes"].max())           * 0.15 +
        (df["digital_readiness"] / df["digital_readiness"].max())       * 0.10 +
        df["container_vessel_pct"]                                      * 0.10
    )
    df["gtm_high_opportunity"] = (composite >= composite.quantile(0.45)).astype(int)
    df["composite_score"]      = composite.round(4)

    le = LabelEncoder()
    df["tier_encoded"] = le.fit_transform(df["company_tier"])

    X = df[FEATURES]
    y_cls = df["gtm_high_opportunity"]
    y_reg = df["deal_size_potential_usd_k"]

    X_tr, X_te, y_tr_cls, y_te_cls = train_test_split(X, y_cls, test_size=0.25, random_state=42, stratify=y_cls)
    _, _, y_tr_reg, y_te_reg       = train_test_split(X, y_reg, test_size=0.25, random_state=42)

    # ── Classification model ──────────────────────────────────────────────────
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=3,
                                          class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    clf.fit(X_tr, y_tr_cls)

    # ── Regression model (deal size predictor) ────────────────────────────────
    reg = Pipeline([
        ("scaler", StandardScaler()),
        ("model",  RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=3,
                                         random_state=42, n_jobs=-1)),
    ])
    reg.fit(X_tr, y_tr_reg)
    y_pred_reg  = reg.predict(X_te)
    reg_mae     = mean_absolute_error(y_te_reg, y_pred_reg)
    reg_r2      = r2_score(y_te_reg, y_pred_reg)
    reg_imp     = pd.Series(reg.named_steps["model"].feature_importances_, index=FEATURES).sort_values(ascending=False)

    df["gtm_score_proba"]   = clf.predict_proba(X)[:, 1]
    df["predicted_deal_usd_k"] = reg.predict(X).round(2)
    df["gtm_tier_label"]    = df["gtm_score_proba"].apply(score_to_tier)

    clf_feat_imp    = pd.Series(clf.named_steps["model"].feature_importances_, index=FEATURES).sort_values(ascending=False)
    y_pred_cls      = clf.predict(X_te)
    y_prob_cls      = clf.predict_proba(X_te)[:, 1]
    auc             = roc_auc_score(y_te_cls, y_prob_cls)
    fpr, tpr, _     = roc_curve(y_te_cls, y_prob_cls)

    return (df, df_vessels, clf, reg, clf_feat_imp, reg_imp,
            auc, fpr, tpr, y_te_cls, y_pred_cls, y_prob_cls,
            y_te_reg, y_pred_reg, reg_mae, reg_r2)


@st.cache_data(show_spinner=False)
def simulate_quarterly_trends(df_master, seed: int = 42):
    """Simulate how GTM scores and pain scores evolve over 4 quarters."""
    np.random.seed(seed)
    records = []
    for _, row in df_master.iterrows():
        base_score = float(row["gtm_score_proba"])
        base_pain  = float(row["visibility_pain_score"])
        for q_idx, quarter in enumerate(QUARTERS):
            # Gradual score improvement as market matures + some noise
            q_score = float(np.clip(base_score + q_idx * 0.015 + np.random.normal(0, 0.025), 0, 1))
            q_pain  = float(np.clip(base_pain  - q_idx * 0.010 + np.random.normal(0, 0.020), 0, 1))
            records.append({
                "company_id":   row["company_id"],
                "company_tier": row["company_tier"],
                "gtm_tier_label": row["gtm_tier_label"],
                "quarter":      quarter,
                "gtm_score":    round(q_score, 4),
                "pain_score":   round(q_pain,  4),
            })
    return pd.DataFrame(records)


# ─────────────────────────────────────────────────────────────────────────────
# PDF GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# TRADE LANE MAP (matplotlib-based world map)
# ─────────────────────────────────────────────────────────────────────────────

def draw_trade_lane_map(df_vessels, selected_ports=None, max_routes=80):
    """Draw a stylised world map with accurate continent outlines and trade route arcs."""

    # Detailed continent polygons (lon, lat) — significantly more accurate than prior version
    CONTINENT_POLYS = {
        "North America": [
            (-168,72),(-140,70),(-120,75),(-85,73),(-65,68),(-60,55),(-65,45),
            (-70,43),(-75,40),(-80,30),(-88,16),(-78,8),(-75,10),(-65,12),(-60,15),
            (-55,5),(-50,3),(-52,5),(-57,7),(-62,10),(-67,11),(-72,10),(-75,8),
            (-78,10),(-80,20),(-83,10),(-78,8),(-78,7),(-80,4),(-85,10),
            (-88,16),(-90,20),(-90,28),(-88,30),(-82,30),(-80,25),(-82,26),
            (-84,30),(-82,35),(-76,38),(-72,42),(-68,46),(-64,47),(-60,47),
            (-57,50),(-55,55),(-60,60),(-65,62),(-64,65),(-68,70),(-80,72),
            (-100,74),(-120,72),(-140,70),(-160,71),(-166,68),(-168,66),(-168,72),
        ],
        "South America": [
            (-80,12),(-75,11),(-70,13),(-62,12),(-60,7),(-52,4),(-50,2),(-48,0),
            (-45,-2),(-38,-4),(-35,-6),(-35,-10),(-38,-15),(-40,-22),(-44,-23),
            (-47,-24),(-48,-28),(-52,-33),(-53,-34),(-58,-38),(-62,-42),(-65,-46),
            (-67,-50),(-69,-54),(-68,-55),(-65,-55),(-60,-52),(-57,-48),(-55,-44),
            (-54,-40),(-52,-34),(-58,-36),(-62,-38),(-65,-42),(-68,-45),(-70,-50),
            (-72,-50),(-75,-46),(-73,-42),(-72,-38),(-70,-32),(-70,-28),(-70,-22),
            (-68,-18),(-70,-14),(-72,-12),(-76,-8),(-78,-4),(-80,0),(-80,5),(-80,12),
        ],
        "Europe": [
            (-10,36),(-6,37),(-2,36),(0,37),(3,39),(5,43),(8,44),(14,45),(18,43),
            (20,40),(22,38),(25,37),(28,38),(30,42),(32,45),(30,48),(25,50),(22,53),
            (20,55),(18,57),(14,57),(12,58),(10,58),(8,57),(5,58),(3,60),(5,62),
            (5,64),(8,65),(14,65),(18,68),(20,70),(18,72),(12,72),(10,70),(5,68),
            (0,65),(-3,62),(-5,60),(-8,58),(-8,55),(-10,52),(-8,50),(-5,48),
            (-2,48),(-1,46),(2,46),(5,44),(3,42),(0,41),(-2,39),(-5,38),(-9,39),
            (-10,38),(-10,36),
        ],
        "Africa": [
            (-18,15),(-16,12),(-15,10),(-13,9),(-10,8),(-8,5),(-3,5),(2,5),
            (3,6),(5,4),(8,4),(10,2),(12,1),(14,1),(16,2),(18,4),(20,4),(22,2),
            (25,1),(28,-1),(32,-2),(35,-5),(38,-8),(40,-10),(42,-12),(44,-12),
            (46,-14),(48,-16),(50,-16),(50,-20),(48,-24),(46,-24),(44,-22),(42,-20),
            (40,-18),(36,-22),(34,-26),(30,-30),(26,-34),(22,-34),(18,-30),(16,-28),
            (14,-22),(12,-18),(10,-16),(8,-14),(8,-10),(10,-6),(10,-2),(8,4),
            (5,5),(3,5),(0,5),(-3,5),(-5,5),(-8,4),(-10,8),(-13,10),(-14,12),
            (-16,13),(-18,15),
        ],
        "Asia": [
            (26,42),(30,43),(34,42),(36,38),(38,36),(40,38),(42,40),(44,43),
            (46,44),(48,44),(50,42),(52,40),(54,38),(56,38),(58,40),(60,42),
            (62,44),(66,44),(68,42),(70,40),(72,38),(74,37),(76,36),(78,35),
            (80,32),(82,30),(84,28),(86,28),(88,26),(90,25),(92,24),(94,22),
            (96,20),(98,18),(100,15),(102,12),(104,10),(106,10),(108,12),(110,14),
            (112,16),(114,18),(116,20),(118,22),(120,24),(122,25),(120,30),(122,32),
            (124,35),(126,38),(128,38),(130,35),(132,34),(134,35),(136,36),(138,38),
            (140,40),(142,44),(144,46),(142,48),(140,50),(136,52),(132,54),(128,56),
            (124,58),(120,60),(116,60),(112,58),(108,58),(104,60),(100,60),
            (96,58),(92,56),(88,54),(84,52),(80,52),(76,54),(72,56),(68,56),
            (62,54),(56,54),(50,52),(44,50),(40,48),(36,46),(30,46),(26,44),(26,42),
        ],
        "Australia": [
            (114,-22),(116,-20),(118,-18),(120,-18),(122,-18),(124,-18),(126,-16),
            (128,-14),(130,-12),(132,-12),(134,-12),(136,-14),(138,-14),(140,-16),
            (142,-18),(144,-20),(146,-20),(148,-20),(150,-22),(152,-24),(152,-26),
            (152,-28),(150,-30),(148,-34),(146,-38),(144,-38),(140,-36),(136,-35),
            (132,-34),(128,-34),(124,-33),(120,-34),(116,-34),(114,-32),(114,-28),
            (112,-24),(114,-22),
        ],
        "Greenland": [
            (-52,82),(-32,84),(-16,82),(-16,78),(-20,76),(-24,75),(-28,72),
            (-32,70),(-38,68),(-44,68),(-48,70),(-50,72),(-52,76),(-54,78),(-52,82),
        ],
    }

    fig, ax = plt.subplots(figsize=(16, 7.5), facecolor="#0D1B2A")
    ax.set_facecolor("#0D1B2A")
    ax.set_xlim(-175, 165)
    ax.set_ylim(-58, 78)
    ax.axis("off")

    # Subtle graticule
    for lat in range(-60, 90, 30):
        ax.plot([-175, 165], [lat, lat], color="#162535", lw=0.5, alpha=0.7)
    for lon in range(-180, 180, 30):
        ax.plot([lon, lon], [-58, 78], color="#162535", lw=0.5, alpha=0.7)

    # Draw continents
    for name, pts in CONTINENT_POLYS.items():
        poly = mpatches.Polygon(pts, closed=True, facecolor="#1B3A28",
                                edgecolor="#2E6B45", lw=0.8, alpha=0.92, zorder=1)
        ax.add_patch(poly)

    # Draw trade route arcs
    route_counts = df_vessels.groupby(["origin_port","destination_port"]).size().reset_index(name="count")
    route_counts  = route_counts.sort_values("count", ascending=False).head(max_routes)
    max_cnt = max(route_counts["count"].max(), 1)

    for _, r in route_counts.iterrows():
        p1, p2 = r["origin_port"], r["destination_port"]
        if p1 not in PORT_COORDS or p2 not in PORT_COORDS:
            continue
        x1, y1 = PORT_COORDS[p1]
        x2, y2 = PORT_COORDS[p2]
        frac  = float(r["count"]) / max_cnt
        alpha = 0.12 + 0.45 * frac
        lw    = 0.4  + 2.0  * frac
        # alternate arc direction for east-west routes to avoid overlap
        rad = 0.2 if (x2 > x1) else -0.2
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-", color="#4FC3F7",
                                   lw=lw, alpha=alpha,
                                   connectionstyle=f"arc3,rad={rad}"),
                    zorder=2)

    # Port dot + label — offset label to avoid overlap
    label_offsets = {
        "Shanghai":   (5, 4),   "Singapore":  (5, -8),  "Rotterdam":  (-5, 6),
        "Los Angeles":(-5, 6),  "Hamburg":    (5, 4),   "Busan":      (5, 4),
        "Antwerp":    (5, -8),  "Dubai":      (5, 4),   "New York":   (5, -8),
        "Shenzhen":   (-5, -8), "Ningbo":     (5, 4),   "Qingdao":    (5, -8),
        "Felixstowe": (-60, 4), "Valencia":   (-60, 4), "Colombo":    (5, -8),
    }
    for port, (lon, lat) in PORT_COORDS.items():
        if selected_ports and port not in selected_ports:
            continue
        ax.scatter(lon, lat, s=55, color="#F39C12", zorder=5,
                   edgecolors="white", lw=0.9)
        ox, oy = label_offsets.get(port, (5, 4))
        ax.annotate(port, (lon, lat), textcoords="offset points", xytext=(ox, oy),
                    fontsize=6.8, color="white", fontweight="bold",
                    alpha=0.95, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", facecolor="#0D1B2A",
                              edgecolor="none", alpha=0.55))

    ax.set_title("Global Trade Lane Network — Port Connections & Route Frequency",
                 color="white", fontsize=13, fontweight="bold", pad=10)
    ax.text(0.01, 0.02,
            "Arc brightness & thickness = route frequency  |  Orange dots = tracked ports",
            transform=ax.transAxes, color="#7AAABB", fontsize=8)
    plt.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## PortIQ")
    st.markdown("*Port Intelligence · GTM Scoring · Supply Chain*")
    st.divider()

    st.markdown("### Pipeline Settings")
    n_vessels   = st.slider("AIS Vessel Records",  500, 3000, 1200, 100)
    n_companies = st.slider("Company Profiles",     80,  400,  180,  20)
    seed        = st.number_input("Random Seed", value=42, step=1)
    st.divider()

    st.markdown("### Filters")
    tier_filter = st.multiselect("Company Tier", ["Enterprise","Mid-Market","SMB"],
                                  default=["Enterprise","Mid-Market","SMB"])
    score_min   = st.slider("Min GTM Score", 0.0, 1.0, 0.0, 0.05)
    gtm_tiers   = st.multiselect("GTM Tier", list(TIER_COLORS.keys()), default=list(TIER_COLORS.keys()))
    st.divider()
    st.caption("PortIQ · AIS Data Science · Streamlit")

# ─────────────────────────────────────────────────────────────────────────────
# RUN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Running ML Pipeline..."):
    try:
        (df, df_vessels, clf, reg, clf_feat_imp, reg_imp,
         auc, fpr, tpr, y_te_cls, y_pred_cls, y_prob_cls,
         y_te_reg, y_pred_reg, reg_mae, reg_r2) = run_pipeline(n_vessels, n_companies, int(seed))
        df_trends = simulate_quarterly_trends(df, int(seed))
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

df_filtered = df[
    df["company_tier"].isin(tier_filter) &
    (df["gtm_score_proba"] >= score_min) &
    df["gtm_tier_label"].isin(gtm_tiers)
].copy()

if df_filtered.empty:
    st.warning("No companies match the current filters. Adjust the sidebar.")
    st.stop()

# Market sizing values (used in multiple tabs)
tier_stats = df.groupby("company_tier").agg(
    n_companies  = ("company_id",               "count"),
    avg_deal_usd = ("deal_size_potential_usd_k", "mean"),
    high_opp_pct = ("gtm_high_opportunity",      "mean"),
).round(2)
tier_stats["TAM ($M)"] = (tier_stats["n_companies"] * tier_stats["avg_deal_usd"] / 1000).round(2)
tier_stats["SAM ($M)"] = (tier_stats["TAM ($M)"]    * tier_stats["high_opp_pct"]).round(2)
tier_stats["SOM ($M)"] = (tier_stats["SAM ($M)"]    * 0.12).round(2)
total_tam = tier_stats["TAM ($M)"].sum()
total_sam = tier_stats["SAM ($M)"].sum()
total_som = tier_stats["SOM ($M)"].sum()

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<h1 style='font-size:2rem; margin-bottom:0.2rem;'>
PortIQ — Port Intelligence Platform
</h1>
<p style='color:#888; font-size:1rem; margin-top:0;'>
GTM Opportunity Scoring · AIS-Powered Market Intelligence · SaaS Sales Motion
</p>
""", unsafe_allow_html=True)
st.divider()

# KPIs
tam_usd_m    = df_filtered["deal_size_potential_usd_k"].sum() / 1000
high_count   = int(df_filtered["gtm_high_opportunity"].sum())
priority     = int((df_filtered["gtm_tier_label"] == "S — Priority").sum())
hit_rate_pct = high_count / max(len(df_filtered), 1) * 100

c1,c2,c3,c4,c5 = st.columns(5)
for col, label, value, delta, help_txt in [
    (c1,"Companies Scored",    f"{len(df_filtered)}",   f"of {len(df)} total",          "Total companies analysed."),
    (c2,"Addressable Market",  f"${tam_usd_m:.1f}M",    "USD est.",                      "Revenue if all scored companies became customers."),
    (c3,"High Opportunity",    f"{high_count}",          f"{hit_rate_pct:.0f}% of filtered","Companies classified as strong GTM prospects."),
    (c4,"S-Priority Prospects",f"{priority}",            "Immediate outreach",            "Top-tier companies — call them first."),
    (c5,"Classification AUC",  f"{auc:.3f}",             f"Regression R²: {reg_r2:.3f}",  "AUC: classification accuracy. R²: deal size prediction accuracy."),
]:
    col.metric(label, value, delta, help=help_txt)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "How It Works",
    "Market Overview",
    "ML Models",
    "Prospect List",
    "Trade Lane Map",
    "Trend Analysis",
    "AIS Explorer",
    "Export & Scorer",
])
tab_how, tab_market, tab_ml, tab_prospects, tab_map, tab_trend, tab_ais, tab_export = tabs

# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 — HOW IT WORKS
# ══════════════════════════════════════════════════════════════════════════════
with tab_how:
    st.markdown('<p class="section-header">What Is PortIQ?</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.markdown("""
PortIQ is an **AI-powered GTM (Go-To-Market) intelligence platform** built for SaaS sales teams
selling container visibility solutions to the global shipping industry.

**The problem it solves:** Shipping companies lose millions of dollars every year because they
cannot reliably track where their containers are, when ships will arrive, or when delays are
happening. Our platform helps them solve this — but first, we need to find the right companies
to sell to.

**What PortIQ does:** It reads AIS (vessel tracking) data for hundreds of shipping companies,
scores each one on how urgently they need our product, and ranks them so the sales team
knows exactly who to call first, what the deal is worth, and why.

**Who uses it:** SaaS sales strategists, account executives, and GTM leaders in the
container logistics and maritime tech space.
        """)

    with col2:
        # Simple stats summary
        st.markdown("**Platform Stats**")
        stats_data = {
            "Companies Scored":     len(df),
            "AIS Vessel Records":   len(df_vessels),
            "ML Features Used":     len(FEATURES),
            "Classification AUC":   f"{auc:.3f}",
            "Regression R²":        f"{reg_r2:.3f}",
            "S-Priority Prospects": int((df["gtm_tier_label"] == "S — Priority").sum()),
            "Total TAM":            f"${total_tam:.1f}M",
        }
        stats_df = pd.DataFrame(list(stats_data.items()), columns=["Metric","Value"])
        st.dataframe(stats_df, use_container_width=True, hide_index=True, height=280)

    # Pipeline flowchart
    st.markdown('<p class="section-header">How the Pipeline Works</p>', unsafe_allow_html=True)
    info_box(
        "The diagram below shows the 6 stages PortIQ goes through to turn raw ship tracking data "
        "into a ranked list of sales-ready prospects."
    )

    fig, ax = plt.subplots(figsize=(14, 3), facecolor=PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"]); ax.axis("off")
    ax.set_xlim(0, 14); ax.set_ylim(0, 3)

    steps = [
        ("1. AIS Data\nGeneration",     "Simulate 1,200+\nvessel voyages\nacross 15 ports",     PALETTE["primary"]),
        ("2. Company\nAggregation",      "Roll vessel stats\nup to company\nlevel",              PALETTE["accent1"]),
        ("3. Feature\nEngineering",      "Create 27 signals:\npain score, readiness,\ncomplexity", "#6C3483"),
        ("4. ML Models",                 "Train classifier\n+ deal size\nregressor",             PALETTE["accent3"]),
        ("5. GTM Scoring",               "Score & bucket\nevery company\nS/A/B/C tier",         PALETTE["accent4"]),
        ("6. Dashboard\n& Export",       "Ranked prospects,\nPDF reports,\nlive scorer",        PALETTE["accent2"]),
    ]

    box_w, box_h = 1.8, 1.8
    gap = (14 - len(steps) * box_w) / (len(steps) + 1)

    for i, (title, subtitle, color) in enumerate(steps):
        x = gap + i * (box_w + gap)
        rect = mpatches.FancyBboxPatch((x, 0.6), box_w, box_h,
                                        boxstyle="round,pad=0.08", facecolor=color,
                                        edgecolor="white", lw=1.5)
        ax.add_patch(rect)
        ax.text(x + box_w/2, 0.6 + box_h*0.68, title, ha="center", va="center",
                color="white", fontsize=8.5, fontweight="bold")
        ax.text(x + box_w/2, 0.6 + box_h*0.28, subtitle, ha="center", va="center",
                color="white", fontsize=6.8, alpha=0.9, linespacing=1.3)
        if i < len(steps) - 1:
            ax.annotate("", xy=(x + box_w + gap, 0.6 + box_h/2),
                         xytext=(x + box_w, 0.6 + box_h/2),
                         arrowprops=dict(arrowstyle="->", color=PALETTE["primary"], lw=2))

    plt.tight_layout(pad=0)
    st.pyplot(fig, use_container_width=True); plt.close()

    # GTM Tier legend
    st.markdown('<p class="section-header">GTM Tier Definitions</p>', unsafe_allow_html=True)
    tier_cols = st.columns(4)
    for col, (tier, desc) in zip(tier_cols, TIER_DESCRIPTIONS.items()):
        color = TIER_COLORS[tier]
        col.markdown(f"""
        <div style='border-left:4px solid {color}; padding:0.6rem 0.8rem;
                    background:{color}12; border-radius:6px;'>
        <b style='color:{color}'>{tier}</b><br>
        <span style='font-size:0.82rem;'>{desc}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header">Key Metrics Explained</p>', unsafe_allow_html=True)
    with st.expander("Click to read plain-English explanations of every metric"):
        cols = st.columns(2)
        items = list(FEATURE_EXPLANATIONS.items())
        half  = len(items) // 2
        for col, chunk in zip(cols, [items[:half], items[half:]]):
            for feat, exp in chunk:
                col.markdown(f"**`{feat}`** — {exp}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — MARKET OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab_market:
    info_box(
        "<b>TAM</b> = every company we could sell to. "
        "<b>SAM</b> = companies that are a strong fit. "
        "<b>SOM</b> = what we can realistically win (~12% of SAM). All values in USD."
    )

    st.markdown('<p class="section-header">Market Sizing by Customer Tier</p>', unsafe_allow_html=True)
    col1, col2 = st.columns([1.5, 1])
    with col1:
        fig, ax = make_chart((7, 3.5))
        tiers = tier_stats.index.tolist()
        x, w  = np.arange(len(tiers)), 0.25
        ax.bar(x-w, tier_stats["TAM ($M)"], w, label="TAM", color=PALETTE["primary"], alpha=0.88)
        ax.bar(x,   tier_stats["SAM ($M)"], w, label="SAM", color=PALETTE["accent1"], alpha=0.88)
        ax.bar(x+w, tier_stats["SOM ($M)"], w, label="SOM", color=PALETTE["accent2"], alpha=0.88)
        ax.set_xticks(x); ax.set_xticklabels(tiers)
        ax.set_ylabel("USD Millions")
        ax.set_title("TAM / SAM / SOM by Customer Tier", fontweight="bold", color=PALETTE["primary"])
        ax.legend(); ax.grid(axis="y", color=PALETTE["grid"])
        st.pyplot(fig, use_container_width=True); plt.close()
    with col2:
        st.dataframe(tier_stats[["n_companies","avg_deal_usd","TAM ($M)","SAM ($M)","SOM ($M)"]].rename(
            columns={"n_companies":"Companies","avg_deal_usd":"Avg Deal ($k)"}),
            use_container_width=True)
        st.caption(f"Total — TAM: ${total_tam:.1f}M  |  SAM: ${total_sam:.1f}M  |  SOM: ${total_som:.1f}M")

    st.markdown('<p class="section-header">GTM Tier Distribution</p>', unsafe_allow_html=True)
    info_box(
        "<b>S-Priority</b>: Call now. "
        "<b>A-Pursue</b>: Add to pipeline. "
        "<b>B-Nurture</b>: Send content. "
        "<b>C-Watch</b>: Check back in 6 months."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        score_dist = df_filtered["gtm_tier_label"].value_counts()
        ordered    = [t for t in TIER_COLORS if t in score_dist.index]
        fig, ax    = make_chart((4.5, 4)); ax.axis("off")
        ax.pie([score_dist[t] for t in ordered], labels=ordered,
               colors=[TIER_COLORS[t] for t in ordered],
               autopct="%1.1f%%", startangle=140,
               textprops={"fontsize":8.5}, pctdistance=0.82)
        ax.set_title("GTM Tier Breakdown", fontweight="bold", color=PALETTE["primary"])
        st.pyplot(fig, use_container_width=True); plt.close()

    with col2:
        valid_tiers = [t for t in ["Enterprise","Mid-Market","SMB"] if t in df_filtered["company_tier"].values]
        pain_data   = [df_filtered[df_filtered["company_tier"]==t]["visibility_pain_score"].values for t in valid_tiers]
        fig, ax     = make_chart((4.5, 4))
        if pain_data:
            bp = ax.boxplot(pain_data, patch_artist=True)
            for patch, color in zip(bp["boxes"],[PALETTE["primary"],PALETTE["accent1"],PALETTE["accent2"]]):
                patch.set_facecolor(color); patch.set_alpha(0.7)
            ax.set_xticklabels(valid_tiers, fontsize=8)
        ax.set_title("Visibility Pain Score by Tier", fontweight="bold", color=PALETTE["primary"])
        ax.set_ylabel("Pain Score (0–1)"); ax.grid(axis="y", color=PALETTE["grid"])
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Higher pain = company loses money from poor visibility — they need us urgently.")

    with col3:
        td = df_filtered.groupby("gtm_tier_label")["deal_size_potential_usd_k"].mean()
        td = td.reindex([k for k in TIER_COLORS if k in td.index]).dropna()
        fig, ax = make_chart((4.5, 4))
        if not td.empty:
            ax.bar(range(len(td)), td.values,
                   color=[TIER_COLORS[t] for t in td.index], edgecolor="white", width=0.6)
            ax.set_xticks(range(len(td)))
            ax.set_xticklabels(td.index, rotation=30, ha="right", fontsize=7.5)
            for i, v in enumerate(td.values):
                ax.text(i, v+0.5, f"${v:.0f}k", ha="center", fontsize=8, fontweight="bold")
        ax.set_title("Avg Deal Size by GTM Tier", fontweight="bold", color=PALETTE["primary"])
        ax.set_ylabel("Deal Size ($k)"); ax.grid(axis="y", color=PALETTE["grid"])
        st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — ML MODELS
# ══════════════════════════════════════════════════════════════════════════════
with tab_ml:
    info_box(
        "PortIQ uses <b>two machine learning models</b>: "
        "(1) a <b>classifier</b> that predicts whether a company is a High or Low opportunity, and "
        "(2) a <b>regression model</b> that predicts exactly how large the deal could be."
    )

    model_tab1, model_tab2 = st.tabs(["Classification Model", "Deal Size Regression"])

    with model_tab1:
        st.markdown('<p class="section-header">Classification Performance</p>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = make_chart((5.5, 4))
            ax.plot(fpr, tpr, color=PALETTE["accent1"], lw=2.5, label=f"AUC = {auc:.3f}")
            ax.plot([0,1],[0,1], "k--", lw=1, alpha=0.4, label="Random baseline")
            ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["accent1"])
            ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
            ax.set_title("ROC Curve", fontweight="bold", color=PALETTE["primary"])
            ax.legend(fontsize=9); ax.grid(color=PALETTE["grid"])
            st.pyplot(fig, use_container_width=True); plt.close()
            st.caption(f"AUC of {auc:.3f} — the model correctly ranks a real High-Opp company "
                        f"above a Low-Opp one {auc*100:.1f}% of the time.")
        with col2:
            cm = confusion_matrix(y_te_cls, y_pred_cls)
            fig, ax = make_chart((5.5, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["Low Opp","High Opp"],
                        yticklabels=["Low Opp","High Opp"],
                        cbar=False, linewidths=1)
            ax.set_title("Confusion Matrix", fontweight="bold", color=PALETTE["primary"])
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            st.pyplot(fig, use_container_width=True); plt.close()
            tn, fp, fn, tp = cm.ravel()
            st.caption(f"Caught {tp} real high-value prospects. Missed {fn}. Raised {fp} false alarms.")

        st.markdown('<p class="section-header">Feature Importances</p>', unsafe_allow_html=True)
        info_box("Longer bar = the model relies on that signal more when scoring a company.")
        top_n = st.slider("Features to show", 5, len(FEATURES), 15, key="clf_topn")
        fi    = clf_feat_imp.head(top_n).sort_values()
        fig, ax = make_chart((9, top_n * 0.48 + 1))
        bar_colors = [PALETTE["accent3"] if i >= len(fi)-3 else PALETTE["accent1"] for i in range(len(fi))]
        ax.barh(fi.index, fi.values, color=bar_colors, edgecolor="white")
        ax.set_xlabel("Importance Score"); ax.grid(axis="x", color=PALETTE["grid"])
        ax.set_title("Top Feature Importances — GTM Classifier", fontweight="bold", color=PALETTE["primary"])
        for i, (feat, val) in enumerate(fi.items()):
            ax.text(val+0.001, i, f"{val:.3f}", va="center", fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

        with st.expander("Plain English: what do these features mean?"):
            for feat in clf_feat_imp.head(top_n).index:
                st.markdown(f"**`{feat}`** — {FEATURE_EXPLANATIONS.get(feat,'')}")

        st.markdown('<p class="section-header">Full Classification Report</p>', unsafe_allow_html=True)
        report_df = pd.DataFrame(classification_report(
            y_te_cls, y_pred_cls, target_names=["Low Opportunity","High Opportunity"],
            output_dict=True)).T.round(3)
        st.dataframe(report_df, use_container_width=True)

    with model_tab2:
        st.markdown('<p class="section-header">Deal Size Prediction Model</p>', unsafe_allow_html=True)
        info_box(
            "This model predicts the <b>exact deal size ($k)</b> for each company — not just whether "
            "they are a good prospect, but how much the contract could be worth. "
            f"R² = <b>{reg_r2:.3f}</b> (1.0 = perfect). MAE = <b>${reg_mae:.1f}k</b> average error."
        )

        col1, col2 = st.columns(2)
        with col1:
            fig, ax = make_chart((5.5, 4))
            ax.scatter(y_te_reg, y_pred_reg, alpha=0.5, color=PALETTE["accent1"],
                       edgecolors="white", lw=0.3, s=40)
            lims = [min(y_te_reg.min(), y_pred_reg.min()),
                    max(y_te_reg.max(), y_pred_reg.max())]
            ax.plot(lims, lims, "r--", lw=1.5, alpha=0.6, label="Perfect prediction")
            ax.set_xlabel("Actual Deal Size ($k)"); ax.set_ylabel("Predicted Deal Size ($k)")
            ax.set_title(f"Actual vs Predicted Deal Size  (R²={reg_r2:.3f})",
                          fontweight="bold", color=PALETTE["primary"])
            ax.legend(); ax.grid(color=PALETTE["grid"])
            st.pyplot(fig, use_container_width=True); plt.close()
            st.caption("Each dot is a company. Dots close to the red line = accurate predictions.")

        with col2:
            residuals = np.array(y_pred_reg) - np.array(y_te_reg)
            fig, ax   = make_chart((5.5, 4))
            ax.hist(residuals, bins=30, color=PALETTE["accent1"], edgecolor="white", alpha=0.85)
            ax.axvline(0, color=PALETTE["accent3"], lw=2, ls="--", label="Zero error")
            ax.set_xlabel("Prediction Error ($k)"); ax.set_ylabel("Count")
            ax.set_title("Residual Distribution", fontweight="bold", color=PALETTE["primary"])
            ax.legend(); ax.grid(color=PALETTE["grid"])
            st.pyplot(fig, use_container_width=True); plt.close()
            st.caption("A bell curve centred on zero means unbiased predictions — the model doesn't systematically over- or under-estimate.")

        st.markdown('<p class="section-header">Feature Importances — Deal Size Model</p>', unsafe_allow_html=True)
        top_n2 = st.slider("Features to show", 5, len(FEATURES), 12, key="reg_topn")
        ri     = reg_imp.head(top_n2).sort_values()
        fig, ax = make_chart((9, top_n2 * 0.48 + 1))
        ax.barh(ri.index, ri.values, color=PALETTE["accent2"], edgecolor="white")
        ax.set_xlabel("Importance Score"); ax.grid(axis="x", color=PALETTE["grid"])
        ax.set_title("Top Feature Importances — Deal Size Regressor", fontweight="bold", color=PALETTE["primary"])
        for i, (feat, val) in enumerate(ri.items()):
            ax.text(val+0.001, i, f"{val:.3f}", va="center", fontsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()

        # Predicted vs actual table
        pred_compare = pd.DataFrame({
            "Actual ($k)":    np.round(y_te_reg.values, 1),
            "Predicted ($k)": np.round(y_pred_reg, 1),
            "Error ($k)":     np.round(residuals, 1),
        }).head(20)
        st.markdown('<p class="section-header">Sample Predictions</p>', unsafe_allow_html=True)
        st.dataframe(pred_compare, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PROSPECT LIST  (color-coded + comparison)
# ══════════════════════════════════════════════════════════════════════════════
with tab_prospects:
    info_box(
        "<b>This is your sales hit list.</b> GTM Score ranges from 0 (low priority) to 1 (highest priority). "
        "Rows are <span style='background:#fde8e8;color:#7b1a1a;padding:1px 5px;border-radius:3px;'>red = S-Priority</span>, "
        "<span style='background:#fef3e2;color:#7b4a00;padding:1px 5px;border-radius:3px;'>orange = A-Pursue</span>, "
        "<span style='background:#e8f0fb;color:#0a2342;padding:1px 5px;border-radius:3px;'>blue = B-Nurture</span>."
    )

    list_tab, compare_tab = st.tabs(["Ranked List", "Company Comparison"])

    with list_tab:
        df_show = df_filtered[EXPORT_COLS].sort_values("gtm_score_proba", ascending=False).copy()
        for col in ["gtm_score_proba","visibility_pain_score","avg_eta_accuracy"]:
            df_show[col] = df_show[col].round(3)
        df_show = df_show.rename(columns={
            "company_id":"Company ID","company_tier":"Tier","fleet_size":"Fleet",
            "annual_teu_volume":"Annual TEU","n_trade_lanes":"Lanes",
            "avg_delay_hours":"Avg Delay (h)","avg_eta_accuracy":"ETA Acc.",
            "visibility_pain_score":"Pain Score","digital_readiness":"Digital",
            "gtm_score_proba":"GTM Score","gtm_tier_label":"GTM Tier",
            "deal_size_potential_usd_k":"Deal ($k)",
        })

        col1, col2 = st.columns([3,1])
        with col1:
            search = st.text_input("Search by Company ID", "", placeholder="e.g. COMP_0042")
        with col2:
            sort_col = st.selectbox("Sort by", ["GTM Score","Deal ($k)","Fleet","Avg Delay (h)"])

        if search:
            df_show = df_show[df_show["Company ID"].str.contains(search.upper(), na=False)]
        df_show = df_show.sort_values(sort_col, ascending=False).reset_index(drop=True)

        # Color-coded table
        styled = df_show.style.apply(tier_color_row, axis=1).format({
            "GTM Score": "{:.3f}", "Pain Score": "{:.3f}",
            "ETA Acc.":  "{:.3f}", "Deal ($k)":  "${:.0f}",
        })
        st.dataframe(styled, use_container_width=True, height=460)
        st.caption(f"Showing {len(df_show)} companies. Row color = GTM tier.")

        # Scatter
        st.markdown('<p class="section-header">Fleet Size vs GTM Score</p>', unsafe_allow_html=True)
        fig, ax = make_chart((10, 4.5))
        for tier, grp in df_filtered.groupby("gtm_tier_label"):
            ax.scatter(grp["fleet_size"], grp["gtm_score_proba"],
                       c=TIER_COLORS.get(str(tier),"#888"), s=grp["n_trade_lanes"]*18,
                       alpha=0.65, edgecolors="white", lw=0.5, label=str(tier))
        ax.axhline(0.65, ls="--", color=PALETTE["accent3"], lw=1.2, alpha=0.7, label="Pursue threshold")
        ax.axhline(0.80, ls="--", color=PALETTE["accent4"], lw=1.2, alpha=0.7, label="Priority threshold")
        ax.set_xlabel("Fleet Size (vessels)"); ax.set_ylabel("GTM Score")
        ax.set_title("Fleet Size vs GTM Score  (bubble = trade lane count)", fontweight="bold", color=PALETTE["primary"])
        ax.legend(fontsize=8); ax.grid(color=PALETTE["grid"])
        st.pyplot(fig, use_container_width=True); plt.close()

    with compare_tab:
        st.markdown('<p class="section-header">Side-by-Side Company Comparison</p>', unsafe_allow_html=True)
        info_box(
            "Select 2 or 3 companies to compare their key metrics side by side. "
            "Useful for deciding which prospect to prioritise when they have similar scores."
        )

        all_ids  = sorted(df["company_id"].tolist())
        selected = st.multiselect("Select companies to compare (2–3)", all_ids,
                                   default=df.nlargest(3,"gtm_score_proba")["company_id"].tolist(),
                                   max_selections=3)

        if len(selected) < 2:
            st.info("Please select at least 2 companies to compare.")
        else:
            cmp_df = df[df["company_id"].isin(selected)].set_index("company_id")
            cmp_cols = [
                "company_tier","fleet_size","annual_teu_volume","n_trade_lanes",
                "avg_delay_hours","avg_eta_accuracy","visibility_pain_score",
                "digital_readiness","gtm_score_proba","gtm_tier_label",
                "deal_size_potential_usd_k","predicted_deal_usd_k",
            ]
            cmp_labels = {
                "company_tier":"Tier","fleet_size":"Fleet Size",
                "annual_teu_volume":"Annual TEU","n_trade_lanes":"Trade Lanes",
                "avg_delay_hours":"Avg Delay (h)","avg_eta_accuracy":"ETA Accuracy",
                "visibility_pain_score":"Pain Score","digital_readiness":"Digital Readiness",
                "gtm_score_proba":"GTM Score","gtm_tier_label":"GTM Tier",
                "deal_size_potential_usd_k":"Deal Size ($k)","predicted_deal_usd_k":"Predicted Deal ($k)",
            }
            display_cmp = cmp_df[cmp_cols].T.rename(index=cmp_labels)

            # Highlight best value in each numeric row
            def highlight_best(row):
                styles = [""] * len(row)
                try:
                    vals = pd.to_numeric(row, errors="coerce")
                    if vals.notna().any():
                        best_idx = vals.idxmax()
                        idx_pos  = list(row.index).index(best_idx)
                        styles[idx_pos] = "background-color: #d4edda; font-weight: bold;"
                except Exception:
                    pass
                return styles

            st.dataframe(display_cmp.style.apply(highlight_best, axis=1),
                          use_container_width=True)
            st.caption("Green highlight = best value in that row.")

            # Radar chart
            radar_metrics = ["fleet_size","n_trade_lanes","avg_delay_hours","digital_readiness",
                              "visibility_pain_score","gtm_score_proba"]
            radar_labels  = ["Fleet Size","Trade Lanes","Delay","Digital Ready","Pain Score","GTM Score"]
            angles = np.linspace(0, 2*np.pi, len(radar_metrics), endpoint=False).tolist()
            angles += angles[:1]

            fig, ax = plt.subplots(figsize=(6, 5), subplot_kw=dict(polar=True), facecolor=PALETTE["bg"])
            ax.set_facecolor(PALETTE["bg"])
            radar_colors = [PALETTE["accent3"], PALETTE["accent1"], PALETTE["accent2"]]

            for i, cid in enumerate(selected):
                row_data = cmp_df.loc[cid, radar_metrics]
                # Normalise each metric to 0–1
                normed = []
                for col in radar_metrics:
                    col_min = df[col].min(); col_max = df[col].max()
                    val = (row_data[col] - col_min) / max(col_max - col_min, 1e-9)
                    normed.append(float(val))
                normed += normed[:1]
                color = radar_colors[i % len(radar_colors)]
                ax.plot(angles, normed, "o-", lw=2, color=color, label=cid)
                ax.fill(angles, normed, alpha=0.1, color=color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(radar_labels, fontsize=8.5)
            ax.set_yticklabels([]); ax.set_ylim(0, 1)
            ax.set_title("Company Profile Comparison (normalised)", fontweight="bold",
                          color=PALETTE["primary"], pad=20)
            ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — TRADE LANE MAP
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    info_box(
        "This map shows the global shipping network across all 15 tracked ports. "
        "<b>Brighter arcs</b> = more vessels on that route. "
        "<b>Orange dots</b> = major ports. "
        "Darker, busier routes indicate companies with higher operational complexity — our best prospects."
    )

    st.markdown('<p class="section-header">Global Trade Lane Network</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([3, 1])
    with col1:
        max_routes = st.slider("Max routes to display", 20, 150, 80, 10,
                                help="Limit the number of trade arcs drawn. Lower = cleaner map.")
    with col2:
        highlight_ports = st.multiselect("Highlight specific ports", PORT_NAMES, default=[],
                                          help="Leave blank to show all ports.")

    selected_ports = highlight_ports if highlight_ports else None
    with st.spinner("Drawing map..."):
        map_fig = draw_trade_lane_map(df_vessels, selected_ports, max_routes)
    st.pyplot(map_fig, use_container_width=True); plt.close()

    # Port statistics table
    st.markdown('<p class="section-header">Port Activity Summary</p>', unsafe_allow_html=True)
    port_stats = pd.concat([
        df_vessels.groupby("origin_port")["delay_hours"].mean().rename("Avg Delay (h) as Origin"),
        df_vessels.groupby("destination_port")["delay_hours"].mean().rename("Avg Delay (h) as Dest"),
        df_vessels.groupby("origin_port")["vessel_id"].count().rename("Departures"),
        df_vessels.groupby("destination_port")["vessel_id"].count().rename("Arrivals"),
    ], axis=1).fillna(0).round(2)
    port_stats["Region"] = [PORT_REGIONS.get(p,"Unknown") for p in port_stats.index]
    port_stats.index.name = "Port"
    st.dataframe(port_stats.sort_values("Departures", ascending=False),
                  use_container_width=True, height=350)
    st.caption("Ports with high delays and high traffic volumes are the richest source of prospects.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — TREND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_trend:
    info_box(
        "This tab simulates how GTM scores and visibility pain evolve over 4 quarters. "
        "It helps the sales team understand <b>pipeline maturity</b> — "
        "which companies are improving (rising score) vs stagnating (flat or declining)."
    )

    st.markdown('<p class="section-header">Quarterly GTM Score Trend</p>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col2:
        trend_tier  = st.multiselect("Filter by Tier", TIERS, default=TIERS, key="trend_tier")
        trend_gtm   = st.multiselect("Filter by GTM Tier", list(TIER_COLORS.keys()),
                                      default=["S — Priority","A — Pursue"], key="trend_gtm")
        n_companies_show = st.slider("Max companies to trace", 5, 40, 15,
                                      help="Show individual trend lines for top N companies.")

    df_t = df_trends[df_trends["company_tier"].isin(trend_tier) &
                     df_trends["gtm_tier_label"].isin(trend_gtm)]

    with col1:
        # Average trend by tier
        tier_trend = df_t.groupby(["quarter","company_tier"])["gtm_score"].mean().reset_index()
        fig, ax    = make_chart((7, 3.5))
        tier_color_map = {"Enterprise": PALETTE["primary"],"Mid-Market": PALETTE["accent1"],"SMB": PALETTE["accent2"]}
        for tier in trend_tier:
            sub = tier_trend[tier_trend["company_tier"]==tier]
            if not sub.empty:
                ax.plot(sub["quarter"], sub["gtm_score"], marker="o", lw=2,
                        color=tier_color_map.get(tier,"#888"), label=tier)
        ax.set_xlabel("Quarter"); ax.set_ylabel("Avg GTM Score")
        ax.set_title("Average GTM Score Trend by Tier", fontweight="bold", color=PALETTE["primary"])
        ax.legend(); ax.grid(color=PALETTE["grid"]); ax.set_ylim(0, 1)
        st.pyplot(fig, use_container_width=True); plt.close()

    # Individual company traces
    st.markdown('<p class="section-header">Individual Company Score Trajectories</p>', unsafe_allow_html=True)
    top_ids   = df[df["gtm_tier_label"].isin(trend_gtm)].nlargest(n_companies_show,"gtm_score_proba")["company_id"].tolist()
    df_traces = df_trends[df_trends["company_id"].isin(top_ids)]

    fig, ax = make_chart((12, 4.5))
    for cid, grp in df_traces.groupby("company_id"):
        tier_lbl = grp["gtm_tier_label"].iloc[0]
        color    = TIER_COLORS.get(tier_lbl, "#888")
        ax.plot(grp["quarter"], grp["gtm_score"], lw=1.2, alpha=0.55, color=color)
    ax.set_xlabel("Quarter"); ax.set_ylabel("GTM Score")
    ax.set_title(f"Score Trajectories — Top {n_companies_show} Prospects", fontweight="bold", color=PALETTE["primary"])
    ax.grid(color=PALETTE["grid"]); ax.set_ylim(0, 1)
    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in TIER_COLORS.items()]
    ax.legend(handles=legend_handles, fontsize=8)
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("Each line is a company. Rising lines = increasing urgency — companies to accelerate in the pipeline.")

    # Pain score trend
    st.markdown('<p class="section-header">Visibility Pain Score Over Time</p>', unsafe_allow_html=True)
    pain_trend = df_t.groupby(["quarter","company_tier"])["pain_score"].mean().reset_index()
    fig, ax    = make_chart((10, 3.5))
    for tier in trend_tier:
        sub = pain_trend[pain_trend["company_tier"]==tier]
        if not sub.empty:
            ax.plot(sub["quarter"], sub["pain_score"], marker="s", lw=2,
                    color=tier_color_map.get(tier,"#888"), label=tier, ls="--")
    ax.set_xlabel("Quarter"); ax.set_ylabel("Avg Pain Score")
    ax.set_title("Visibility Pain Score Trend by Tier", fontweight="bold", color=PALETTE["primary"])
    ax.legend(); ax.grid(color=PALETTE["grid"])
    st.pyplot(fig, use_container_width=True); plt.close()
    st.caption("Declining pain score over time could indicate competitors are addressing the problem — act fast.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — AIS EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_ais:
    info_box(
        "AIS (Automatic Identification System) is the GPS-like signal every commercial ship broadcasts. "
        "We use it to measure delay patterns, tracking reliability, and route complexity."
    )

    st.markdown('<p class="section-header">Raw AIS Vessel Records</p>', unsafe_allow_html=True)
    st.dataframe(df_vessels.head(200), use_container_width=True, height=300)

    with st.expander("Column Glossary"):
        st.markdown("""
| Column | Meaning |
|--------|---------|
| `vessel_id` | Unique ID for each ship |
| `vessel_type` | Container Ship, Tanker, Bulk Carrier, etc. |
| `origin_port / destination_port` | Departure and arrival ports |
| `distance_nm` | Voyage distance in nautical miles |
| `delay_hours` | Hours the ship arrived late |
| `eta_accuracy_rate` | How accurate arrival predictions were (1.0 = perfect) |
| `data_gaps_pct` | Proportion of tracking pings that went missing |
| `cargo_teu` | Containers on board (1 TEU = 20-foot container) |
| `bunker_cost_usd` | Fuel cost for the voyage in USD |
        """)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<p class="section-header">Fleet Composition</p>', unsafe_allow_html=True)
        vc = df_vessels["vessel_type"].value_counts()
        fig, ax = make_chart((5.5, 3.5))
        ax.barh(vc.index, vc.values, color=PALETTE["accent1"], edgecolor="white")
        ax.set_xlabel("Number of Vessels")
        ax.set_title("Vessels by Type", fontweight="bold", color=PALETTE["primary"])
        ax.grid(axis="x", color=PALETTE["grid"])
        st.pyplot(fig, use_container_width=True); plt.close()
        st.caption("Container Ships are our primary target — most TEUs, highest tracking complexity.")

    with col2:
        st.markdown('<p class="section-header">Delay Distribution</p>', unsafe_allow_html=True)
        fig, ax = make_chart((5.5, 3.5))
        for vt in df_vessels["vessel_type"].value_counts().head(4).index:
            ax.hist(df_vessels[df_vessels["vessel_type"]==vt]["delay_hours"],
                    bins=25, alpha=0.55, label=vt, edgecolor="none")
        ax.set_xlabel("Hours Late"); ax.set_ylabel("Voyages")
        ax.set_title("Delay by Vessel Type", fontweight="bold", color=PALETTE["primary"])
        ax.legend(fontsize=7.5); ax.grid(color=PALETTE["grid"])
        st.pyplot(fig, use_container_width=True); plt.close()

    st.markdown('<p class="section-header">Port Delay Heatmap</p>', unsafe_allow_html=True)
    info_box("Darker red = more delays = stronger prospect pain on that route.")
    pivot = df_vessels.groupby(["origin_port","dest_region"])["delay_hours"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 5), facecolor=PALETTE["bg"])
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, linewidths=0.5, annot=True, fmt=".1f",
                cbar_kws={"label":"Avg Delay (hrs)"})
    ax.set_title("Average Delay: Origin Port → Destination Region",
                  fontweight="bold", color=PALETTE["primary"])
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — EXPORT & SCORER
# ══════════════════════════════════════════════════════════════════════════════
with tab_export:

    # ── Downloads ─────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Download Data</p>', unsafe_allow_html=True)
    info_box("Download any dataset below for use in Excel, your CRM, or presentations.")

    col1, col2, col3 = st.columns(3)
    with col1:
        csv_data = df_filtered[EXPORT_COLS].sort_values("gtm_score_proba", ascending=False).to_csv(index=False)
        st.download_button("Download Prospects CSV", csv_data,
                            "portiq_prospects.csv", "text/csv", use_container_width=True)
        st.caption("Ranked prospect list — import into your CRM.")
    with col2:
        st.download_button("Download AIS Vessel Data", df_vessels.to_csv(index=False),
                            "portiq_ais.csv", "text/csv", use_container_width=True)
        st.caption("Raw AIS tracking records.")
    with col3:
        summary = {
            "model": "Random Forest", "classification_auc": round(auc,4),
            "regression_r2": round(reg_r2,4), "regression_mae_usd_k": round(reg_mae,2),
            "n_features": len(FEATURES), "n_companies_scored": len(df),
            "currency": "USD",
            "market_sizing_usd_m": {"TAM": round(total_tam,2), "SAM": round(total_sam,2), "SOM": round(total_som,2)},
            "gtm_tier_counts": {str(k): int(v) for k,v in df["gtm_tier_label"].value_counts().items()},
            "top5_features": clf_feat_imp.head(5).index.tolist(),
        }
        st.download_button("Download Summary JSON",
                            json.dumps(summary, indent=2), "portiq_summary.json",
                            "application/json", use_container_width=True)
        st.caption("Model metadata and market sizing.")
    st.divider()

    # ── Live Scorer ────────────────────────────────────────────────────────────
    st.markdown('<p class="section-header">Live Company Scorer</p>', unsafe_allow_html=True)
    info_box(
        "Enter any shipping company's details and instantly get their GTM score, "
        "predicted deal size, and a breakdown of exactly why they scored that way."
    )

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown("**Company Size**")
        inp_fleet = st.number_input("Fleet Size (ships)", 1, 200, 20)
        inp_teu   = st.number_input("Annual TEU Volume", 1000, 5_000_000, 100_000, step=5000)
        inp_lanes = st.slider("Number of Trade Routes", 1, 6, 3)
    with sc2:
        st.markdown("**Operational Pain**")
        inp_delay = st.number_input("Average Arrival Delay (hrs)", 0.0, 100.0, 12.0)
        inp_eta   = st.slider("ETA Accuracy (0=always wrong, 1=always right)", 0.0, 1.0, 0.70, 0.01)
        inp_dmat  = st.slider("Digital Maturity (1=basic, 5=advanced)", 1, 5, 3)
    with sc3:
        st.markdown("**Tech Profile**")
        inp_tms   = st.selectbox("Uses Transport Management Software?", ["Yes","No"])
        inp_edi   = st.selectbox("Has EDI Integration?", ["Yes","No"])
        inp_tier  = st.selectbox("Company Size Tier", ["SMB","Mid-Market","Enterprise"])

    if st.button("Score This Company", type="primary"):
        tms_val  = 1 if inp_tms == "Yes" else 0
        edi_val  = 1 if inp_edi == "Yes" else 0
        tier_map = {"SMB":2,"Mid-Market":1,"Enterprise":0}

        pain       = float(np.clip((inp_delay/80)*0.35 + (1-inp_eta)*0.30 + 0.05 + 0.15*0.6, 0, 1))
        deal_usd_k = round((inp_fleet*1.2 + inp_teu/10_000*0.8 + inp_lanes*5), 2)
        dig_ready  = round(inp_dmat*0.5 + tms_val*1.5 + edi_val*1.0, 4)

        feat_vals = {
            "fleet_size": inp_fleet, "annual_teu_volume": inp_teu, "n_trade_lanes": inp_lanes,
            "n_home_ports": 3, "years_in_operation": 10, "has_existing_tms": tms_val,
            "has_edi_integration": edi_val, "digital_maturity": inp_dmat,
            "avg_delay_hours": inp_delay, "max_delay_hours": inp_delay*1.8,
            "avg_eta_accuracy": inp_eta, "total_port_calls": 24, "avg_data_gaps": 0.05,
            "n_vessels_active": max(1, inp_fleet//3), "n_distinct_origins": 5,
            "n_distinct_dests": 5, "avg_distance_nm": 4000.0, "avg_vessel_age": 8.0,
            "container_vessel_pct": 0.7, "transcontinental_pct": 0.6,
            "visibility_pain_score": pain, "port_complexity_score": 5.0,
            "digital_readiness": dig_ready,
            "gtm_complexity": inp_lanes*3/max(inp_fleet,1),
            "deal_size_potential_usd_k": deal_usd_k,
            "tam_addressable_teu": inp_teu*0.7,
            "tier_encoded": tier_map[inp_tier],
        }

        try:
            row        = pd.DataFrame([feat_vals])[FEATURES]
            score      = float(clf.predict_proba(row)[0][1])
            pred_deal  = float(reg.predict(row)[0])
            tier_label = score_to_tier(score)
            color      = TIER_COLORS[tier_label]

            r1, r2, r3, r4 = st.columns(4)
            r1.metric("GTM Score",           f"{score:.3f}")
            r2.metric("GTM Tier",            tier_label)
            r3.metric("Est. Deal Size",      f"${deal_usd_k:.0f}k")
            r4.metric("Model-Predicted Deal",f"${pred_deal:.0f}k")

            st.markdown(f"""
            <div style='background:{color}18; border-left:5px solid {color};
                        padding:0.9rem 1.2rem; border-radius:8px; margin-top:0.5rem;'>
              <b style='color:{color}; font-size:1rem;'>{tier_label}</b><br>
              <span style='font-size:0.88rem;'>{TIER_DESCRIPTIONS[tier_label]}</span><br><br>
              <span style='font-size:0.85rem; color:#555;'>
                GTM Score: <b>{score:.3f}</b> &nbsp;|&nbsp;
                Deal: <b>${deal_usd_k:.0f}k</b> &nbsp;|&nbsp;
                Digital Readiness: <b>{dig_ready:.1f}</b> &nbsp;|&nbsp;
                Visibility Pain: <b>{pain:.3f}</b>
              </span>
            </div>
            """, unsafe_allow_html=True)

            # ── "Why this score?" feature contribution chart ──────────────────
            st.markdown('<p class="section-header">Why This Score? — Feature Contribution Breakdown</p>',
                         unsafe_allow_html=True)
            info_box(
                "This chart shows which of the company's inputs had the most influence on their final GTM score. "
                "Longer bars = that signal pulled the score up more. "
                "It makes the AI decision transparent and explainable."
            )

            # Compute contribution = feature_importance × normalised input value
            feat_importance = clf.named_steps["model"].feature_importances_
            row_vals        = row.values[0]
            # Normalise each feature value vs training data range
            X_train_arr = df[FEATURES].values
            col_mins    = X_train_arr.min(axis=0)
            col_maxs    = X_train_arr.max(axis=0)
            ranges      = np.where(col_maxs - col_mins > 0, col_maxs - col_mins, 1)
            norm_vals   = np.clip((row_vals - col_mins) / ranges, 0, 1)
            contributions = feat_importance * norm_vals

            contrib_series = pd.Series(contributions, index=FEATURES).sort_values(ascending=False).head(12)

            fig, ax = make_chart((9, 5))
            bar_colors = [PALETTE["accent3"] if v > contrib_series.median() else PALETTE["accent1"]
                          for v in contrib_series.values]
            bars = ax.barh(contrib_series.index[::-1], contrib_series.values[::-1],
                           color=bar_colors[::-1], edgecolor="white")
            ax.set_xlabel("Contribution to GTM Score")
            ax.set_title(f"Score Driver Analysis — {tier_label}  (GTM Score: {score:.3f})",
                          fontweight="bold", color=PALETTE["primary"])
            ax.grid(axis="x", color=PALETTE["grid"])
            for bar, val in zip(bars, contrib_series.values[::-1]):
                ax.text(val+0.0005, bar.get_y()+bar.get_height()/2,
                        f"{val:.4f}", va="center", fontsize=8)
            st.pyplot(fig, use_container_width=True); plt.close()

            # Plain English explanation
            top3_features = contrib_series.head(3).index.tolist()
            st.markdown("**Top 3 reasons for this score:**")
            for feat in top3_features:
                exp = FEATURE_EXPLANATIONS.get(feat,"")
                val = feat_vals.get(feat, "N/A")
                st.markdown(f"- **{feat}** = `{round(val,2) if isinstance(val,float) else val}` — {exp}")

        except Exception as e:
            st.error(f"Scoring failed: {e}")