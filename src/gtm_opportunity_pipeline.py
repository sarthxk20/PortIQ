"""
=============================================================================
  PortIQ — Port Intelligence Platform
  GTM Opportunity Scoring Pipeline — AIS Vessel Tracking Data
=============================================================================
  Author  : Sarthak Shandilya
  Purpose : Market Sizing + ML-based Prospect Scoring for SaaS Sales Motion
  Data    : Synthetic AIS (Automatic Identification System) vessel data
            modeled after real AIS schema (MMSI, IMO, position, speed, etc.)
=============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import json
import os

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, roc_auc_score,
                             confusion_matrix, roc_curve, precision_recall_curve)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import pickle

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. SYNTHETIC AIS DATA GENERATION
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STAGE 1 — SYNTHETIC AIS DATA GENERATION")
print("="*70)

N_VESSELS   = 1200
N_COMPANIES = 180

VESSEL_TYPES = ["Container Ship", "Bulk Carrier", "Tanker", "Ro-Ro",
                "General Cargo", "Feeder Container", "VLCC"]
TRADE_LANES  = ["Asia-Europe", "Trans-Pacific", "Trans-Atlantic",
                "Intra-Asia", "Middle East-Asia", "Latin America-US"]
REGISTRIES   = ["Panama", "Liberia", "Marshall Islands", "Singapore",
                "Hong Kong", "Bahamas", "Malta"]
TIERS        = ["Enterprise", "Mid-Market", "SMB"]

MAJOR_PORTS = {
    "Shanghai": (121.47, 31.23, "Asia"),
    "Singapore": (103.82, 1.26, "Asia"),
    "Rotterdam": (4.40, 51.89, "Europe"),
    "Los Angeles": (-118.27, 33.74, "Americas"),
    "Hamburg": (9.99, 53.55, "Europe"),
    "Busan": (129.04, 35.10, "Asia"),
    "Antwerp": (4.42, 51.23, "Europe"),
    "Dubai": (55.27, 25.20, "Middle East"),
    "New York": (-74.01, 40.71, "Americas"),
    "Shenzhen": (114.06, 22.55, "Asia"),
    "Ningbo": (121.55, 29.88, "Asia"),
    "Qingdao": (120.38, 36.07, "Asia"),
    "Felixstowe": (1.35, 51.96, "Europe"),
    "Valencia": (-0.31, 39.46, "Europe"),
    "Colombo": (79.86, 6.93, "Asia"),
}

PORT_NAMES = list(MAJOR_PORTS.keys())
PORT_REGIONS = {p: v[2] for p, v in MAJOR_PORTS.items()}

def generate_company_profile(cid):
    tier = np.random.choice(TIERS, p=[0.15, 0.35, 0.50])
    if tier == "Enterprise":
        fleet_size = np.random.randint(30, 120)
        annual_teu = np.random.randint(500_000, 3_000_000)
        revenue_usd_m = np.random.uniform(200, 2000)
    elif tier == "Mid-Market":
        fleet_size = np.random.randint(8, 30)
        annual_teu = np.random.randint(50_000, 500_000)
        revenue_usd_m = np.random.uniform(20, 200)
    else:
        fleet_size = np.random.randint(1, 8)
        annual_teu = np.random.randint(2_000, 50_000)
        revenue_usd_m = np.random.uniform(1, 20)

    n_lanes   = np.random.choice([1,2,3,4,5], p=[0.25,0.30,0.25,0.15,0.05])
    lanes     = np.random.choice(TRADE_LANES, n_lanes, replace=False).tolist()
    n_ports   = min(len(PORT_NAMES), np.random.randint(2, 10))
    home_ports = np.random.choice(PORT_NAMES, n_ports, replace=False).tolist()

    return {
        "company_id":         f"COMP_{cid:04d}",
        "company_tier":       tier,
        "fleet_size":         fleet_size,
        "annual_teu_volume":  annual_teu,
        "revenue_usd_m":      round(revenue_usd_m, 2),
        "trade_lanes":        lanes,
        "home_ports":         home_ports,
        "n_trade_lanes":      n_lanes,
        "n_home_ports":       n_ports,
        "registry":           np.random.choice(REGISTRIES),
        "years_in_operation": np.random.randint(1, 45),
        "has_existing_tms":   np.random.choice([0, 1], p=[0.45, 0.55]),
        "has_edi_integration":np.random.choice([0, 1], p=[0.50, 0.50]),
        "digital_maturity":   np.random.randint(1, 6),   # 1-5 Likert
    }

companies_raw = [generate_company_profile(i) for i in range(N_COMPANIES)]
df_companies  = pd.DataFrame(companies_raw)
print(f"  ✓ Generated {N_COMPANIES} company profiles")

# ── AIS Vessel Pings ──────────────────────────────────────────────────────────

def generate_vessel_record(vid):
    cid_idx   = np.random.randint(0, N_COMPANIES)
    company   = df_companies.iloc[cid_idx]
    vtype     = np.random.choice(VESSEL_TYPES, p=[0.35,0.20,0.18,0.07,0.10,0.08,0.02])
    port_a, port_b = np.random.choice(PORT_NAMES, 2, replace=False)
    dist_nm   = np.random.uniform(800, 12_000)
    speed_kt  = np.random.uniform(8, 22) if vtype != "VLCC" else np.random.uniform(10, 15)
    transit_h = dist_nm / speed_kt
    delay_h   = max(0, np.random.normal(
        loc=12 if company["n_trade_lanes"] >= 3 else 4,
        scale=8))
    port_calls_yr = np.random.randint(4, 52)
    eta_accuracy  = max(0, min(1, np.random.normal(0.68, 0.15)))
    bunker_cost_usd = dist_nm * np.random.uniform(25, 50)

    return {
        "vessel_id":            f"VESSEL_{vid:05d}",
        "mmsi":                 np.random.randint(200_000_000, 800_000_000),
        "imo":                  np.random.randint(9_000_000, 9_999_999),
        "company_id":           company["company_id"],
        "vessel_type":          vtype,
        "origin_port":          port_a,
        "destination_port":     port_b,
        "origin_region":        PORT_REGIONS[port_a],
        "dest_region":          PORT_REGIONS[port_b],
        "distance_nm":          round(dist_nm, 1),
        "speed_knots":          round(speed_kt, 1),
        "transit_hours":        round(transit_h, 1),
        "delay_hours":          round(delay_h, 1),
        "port_calls_per_year":  port_calls_yr,
        "eta_accuracy_rate":    round(eta_accuracy, 3),
        "bunker_cost_usd":      round(bunker_cost_usd, 2),
        "cargo_teu":            np.random.randint(100, 24_000) if "Container" in vtype else 0,
        "vessel_age_years":     np.random.randint(0, 25),
        "flag_state":           np.random.choice(REGISTRIES),
        "transceivers_count":   np.random.randint(1, 4),
        "data_gaps_pct":        round(np.random.beta(2, 8), 3),  # % of missing pings
    }

vessels_raw = [generate_vessel_record(i) for i in range(N_VESSELS)]
df_vessels  = pd.DataFrame(vessels_raw)
print(f"  ✓ Generated {N_VESSELS} AIS vessel records")
print(f"  ✓ Vessel types: {df_vessels['vessel_type'].value_counts().to_dict()}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STAGE 2 — FEATURE ENGINEERING")
print("="*70)

# ── Aggregate vessel stats → company level ────────────────────────────────────

vessel_agg = df_vessels.groupby("company_id").agg(
    avg_delay_hours       =("delay_hours",       "mean"),
    max_delay_hours       =("delay_hours",       "max"),
    avg_eta_accuracy      =("eta_accuracy_rate", "mean"),
    total_port_calls      =("port_calls_per_year","sum"),
    avg_data_gaps         =("data_gaps_pct",      "mean"),
    n_vessels_active      =("vessel_id",          "count"),
    n_distinct_origins    =("origin_port",        "nunique"),
    n_distinct_dests      =("destination_port",   "nunique"),
    avg_distance_nm       =("distance_nm",        "mean"),
    avg_speed             =("speed_knots",        "mean"),
    total_teu             =("cargo_teu",          "sum"),
    avg_vessel_age        =("vessel_age_years",   "mean"),
    container_vessel_pct  =("vessel_type",        lambda x: (x.str.contains("Container")).mean()),
    avg_bunker_cost       =("bunker_cost_usd",    "mean"),
    transcontinental_pct  =("origin_region",      lambda x: (x != df_vessels.loc[x.index, "dest_region"]).mean()),
).reset_index()

df_master = df_companies.merge(vessel_agg, on="company_id", how="left")
df_master.fillna(df_master.median(numeric_only=True), inplace=True)

# ── Derived / engineered features ─────────────────────────────────────────────

df_master["port_complexity_score"] = (
    df_master["n_distinct_origins"] + df_master["n_distinct_dests"]
) / 2

df_master["visibility_pain_score"] = (
    (df_master["avg_delay_hours"]   / df_master["avg_delay_hours"].max()) * 0.35 +
    (1 - df_master["avg_eta_accuracy"])                                   * 0.30 +
    (df_master["avg_data_gaps"]     / df_master["avg_data_gaps"].max())   * 0.20 +
    (df_master["transcontinental_pct"])                                   * 0.15
).round(4)

df_master["tam_addressable_teu"] = df_master["total_teu"] * df_master["container_vessel_pct"]

df_master["gtm_complexity"] = (
    df_master["n_trade_lanes"] * df_master["n_home_ports"] /
    (df_master["fleet_size"] + 1)
).round(4)

df_master["digital_readiness"] = (
    df_master["digital_maturity"] * 0.5 +
    df_master["has_existing_tms"] * 1.5 +
    df_master["has_edi_integration"] * 1.0
).round(4)

df_master["deal_size_potential_usd_k"] = (
    df_master["fleet_size"] * 1.2 +
    df_master["annual_teu_volume"] / 10_000 * 0.8 +
    df_master["n_trade_lanes"] * 5
).round(2)

print(f"  ✓ Engineered {df_master.shape[1]} features for {df_master.shape[0]} companies")

# ── Label generation — GTM Opportunity Score (binary: High vs Low) ────────────

composite = (
    df_master["visibility_pain_score"]                                        * 0.30 +
    (df_master["fleet_size"] / df_master["fleet_size"].max())                 * 0.20 +
    (df_master["annual_teu_volume"] / df_master["annual_teu_volume"].max())   * 0.15 +
    (df_master["n_trade_lanes"] / df_master["n_trade_lanes"].max())           * 0.15 +
    (df_master["digital_readiness"] / df_master["digital_readiness"].max())   * 0.10 +
    (df_master["container_vessel_pct"])                                        * 0.10
)

threshold = composite.quantile(0.45)
df_master["gtm_high_opportunity"] = (composite >= threshold).astype(int)
df_master["composite_score"]       = composite.round(4)

print(f"  ✓ Label distribution  →  High: {df_master['gtm_high_opportunity'].sum()}  "
      f"Low: {(1 - df_master['gtm_high_opportunity']).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. MARKET SIZING
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STAGE 3 — MARKET SIZING ANALYSIS")
print("="*70)

tier_stats = df_master.groupby("company_tier").agg(
    n_companies         =("company_id",            "count"),
    avg_fleet_size      =("fleet_size",            "mean"),
    total_teu           =("annual_teu_volume",      "sum"),
    avg_deal_k          =("deal_size_potential_usd_k", "mean"),
    high_opp_pct        =("gtm_high_opportunity",  "mean"),
    avg_pain_score      =("visibility_pain_score", "mean"),
).round(2)

tier_stats["tam_usd_m"] = (
    tier_stats["n_companies"] *
    tier_stats["avg_deal_k"] / 1000
).round(2)

tier_stats["sam_usd_m"] = (tier_stats["tam_usd_m"] * tier_stats["high_opp_pct"]).round(2)
tier_stats["som_usd_m"] = (tier_stats["sam_usd_m"] * 0.12).round(2)   # 12% attainable

print("\n  ── MARKET SIZING BY TIER ─────────────────────────────────────────")
print(tier_stats[["n_companies","avg_fleet_size","avg_deal_k",
                   "high_opp_pct","tam_usd_m","sam_usd_m","som_usd_m"]].to_string())

total_tam = tier_stats["tam_usd_m"].sum()
total_sam = tier_stats["sam_usd_m"].sum()
total_som = tier_stats["som_usd_m"].sum()
print(f"\n  TOTAL TAM  ${total_tam:.1f}M  |  SAM  ${total_sam:.1f}M  |  SOM  ${total_som:.1f}M")


# ─────────────────────────────────────────────────────────────────────────────
# 4. ML PIPELINE — GTM OPPORTUNITY SCORING MODEL
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STAGE 4 — ML PIPELINE: GTM OPPORTUNITY SCORING")
print("="*70)

FEATURE_COLS = [
    "fleet_size", "annual_teu_volume", "n_trade_lanes", "n_home_ports",
    "years_in_operation", "has_existing_tms", "has_edi_integration",
    "digital_maturity", "avg_delay_hours", "max_delay_hours",
    "avg_eta_accuracy", "total_port_calls", "avg_data_gaps",
    "n_vessels_active", "n_distinct_origins", "n_distinct_dests",
    "avg_distance_nm", "avg_vessel_age", "container_vessel_pct",
    "transcontinental_pct", "visibility_pain_score", "port_complexity_score",
    "digital_readiness", "gtm_complexity", "deal_size_potential_usd_k",
    "tam_addressable_teu",
]

# ── Encode tier ───────────────────────────────────────────────────────────────
le = LabelEncoder()
df_master["tier_encoded"] = le.fit_transform(df_master["company_tier"])
FEATURE_COLS.append("tier_encoded")

X = df_master[FEATURE_COLS].copy()
y = df_master["gtm_high_opportunity"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

print(f"  Train: {len(X_train)}  |  Test: {len(X_test)}")

# ── Model A — Random Forest ───────────────────────────────────────────────────
rf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)

# ── Model B — Gradient Boosting ───────────────────────────────────────────────
gb_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model",  GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42))
])
gb_pipeline.fit(X_train, y_train)

# ── Cross-validation ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_cv  = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc")
gb_cv  = cross_val_score(gb_pipeline, X, y, cv=cv, scoring="roc_auc")

print(f"\n  Random Forest  CV AUC: {rf_cv.mean():.4f} ± {rf_cv.std():.4f}")
print(f"  Grad Boosting  CV AUC: {gb_cv.mean():.4f} ± {gb_cv.std():.4f}")

# ── Pick best model ───────────────────────────────────────────────────────────
best_model = rf_pipeline if rf_cv.mean() >= gb_cv.mean() else gb_pipeline
best_name  = "Random Forest" if rf_cv.mean() >= gb_cv.mean() else "Gradient Boosting"
print(f"\n  ★ Best Model: {best_name}")

y_pred   = best_model.predict(X_test)
y_proba  = best_model.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, y_proba)
print(f"  Test AUC: {test_auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Low Opp", "High Opp"]))

# ── Feature Importance ────────────────────────────────────────────────────────
feat_imp = pd.Series(
    best_model.named_steps["model"].feature_importances_,
    index=FEATURE_COLS
).sort_values(ascending=False).head(15)

print("\n  Top 15 Feature Importances:")
for feat, imp in feat_imp.items():
    bar = "█" * int(imp * 200)
    print(f"    {feat:<35}  {imp:.4f}  {bar}")

# ── Score all companies ───────────────────────────────────────────────────────
df_master["gtm_score_proba"] = best_model.predict_proba(X)[:, 1]

df_master["gtm_tier_label"] = pd.cut(
    df_master["gtm_score_proba"],
    bins=[0, 0.40, 0.65, 0.80, 1.0],
    labels=["C — Watch", "B — Nurture", "A — Pursue", "S — Priority"]
)

score_dist = df_master["gtm_tier_label"].value_counts().sort_index()
print("\n  GTM Score Distribution:")
for tier, cnt in score_dist.items():
    print(f"    {tier:<20}  {cnt} companies")


# ─────────────────────────────────────────────────────────────────────────────
# 5. VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STAGE 5 — VISUALIZATIONS")
print("="*70)

PALETTE = {
    "primary":  "#0A2342",
    "accent1":  "#1B6CA8",
    "accent2":  "#27AE60",
    "accent3":  "#E74C3C",
    "accent4":  "#F39C12",
    "bg":       "#F8F9FA",
    "grid":     "#E0E0E0",
}
TIER_COLORS = {
    "S — Priority": "#E74C3C",
    "A — Pursue":   "#F39C12",
    "B — Nurture":  "#1B6CA8",
    "C — Watch":    "#95A5A6",
}
plt.rcParams.update({
    "figure.facecolor": PALETTE["bg"],
    "axes.facecolor":   PALETTE["bg"],
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "font.family":      "DejaVu Sans",
})

fig = plt.figure(figsize=(22, 28), facecolor=PALETTE["bg"])
fig.suptitle(
    "PortIQ — PortIQ — GTM Opportunity Scoring Dashboard",
    fontsize=20, fontweight="bold", color=PALETTE["primary"], y=0.98)

gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.45, wspace=0.35)

# ── (1) Market Sizing TAM/SAM/SOM ────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
tier_order  = ["Enterprise", "Mid-Market", "SMB"]
tier_colors = [PALETTE["primary"], PALETTE["accent1"], PALETTE["accent2"]]
x  = np.arange(len(tier_order))
w  = 0.25
tam_vals = [tier_stats.loc[t, "tam_usd_m"] for t in tier_order]
sam_vals = [tier_stats.loc[t, "sam_usd_m"] for t in tier_order]
som_vals = [tier_stats.loc[t, "som_usd_m"] for t in tier_order]
ax1.bar(x - w, tam_vals, w, label="TAM", color=PALETTE["primary"],   alpha=0.85)
ax1.bar(x,     sam_vals, w, label="SAM", color=PALETTE["accent1"],   alpha=0.85)
ax1.bar(x + w, som_vals, w, label="SOM", color=PALETTE["accent2"],   alpha=0.85)
ax1.set_xticks(x); ax1.set_xticklabels(tier_order, fontsize=9)
ax1.set_ylabel("USD Millions"); ax1.set_title("Market Sizing: TAM / SAM / SOM", fontweight="bold")
ax1.legend(fontsize=8); ax1.grid(axis="y", color=PALETTE["grid"])

# ── (2) GTM Score Distribution ────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
labels   = score_dist.index.tolist()
colors   = [TIER_COLORS[l] for l in labels]
wedges, texts, autotexts = ax2.pie(
    score_dist.values, labels=labels, colors=colors,
    autopct="%1.1f%%", startangle=140,
    textprops={"fontsize": 8.5}, pctdistance=0.82)
for at in autotexts: at.set_color("white"); at.set_fontweight("bold")
ax2.set_title("GTM Tier Distribution", fontweight="bold")

# ── (3) ROC Curve ─────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[0, 2])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_pipeline.predict_proba(X_test)[:,1])
fpr_gb, tpr_gb, _ = roc_curve(y_test, gb_pipeline.predict_proba(X_test)[:,1])
ax3.plot(fpr_rf, tpr_rf, color=PALETTE["accent1"], lw=2,
         label=f"Random Forest (AUC={roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:,1]):.3f})")
ax3.plot(fpr_gb, tpr_gb, color=PALETTE["accent4"], lw=2,
         label=f"Grad Boost    (AUC={roc_auc_score(y_test, gb_pipeline.predict_proba(X_test)[:,1]):.3f})")
ax3.plot([0,1],[0,1], "k--", lw=1, alpha=0.4)
ax3.set_xlabel("FPR"); ax3.set_ylabel("TPR")
ax3.set_title("ROC Curves — Model Comparison", fontweight="bold")
ax3.legend(fontsize=8); ax3.grid(color=PALETTE["grid"])

# ── (4) Feature Importance ────────────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, :2])
feat_plot = feat_imp.head(12).sort_values()
bar_colors = [PALETTE["accent1"] if i < 4 else PALETTE["primary"] for i in range(len(feat_plot))]
bars = ax4.barh(feat_plot.index, feat_plot.values, color=bar_colors[::-1], edgecolor="white")
ax4.set_title("Top Feature Importances (GTM Scoring Model)", fontweight="bold")
ax4.set_xlabel("Importance Score")
for bar, val in zip(bars, feat_plot.values):
    ax4.text(val + 0.001, bar.get_y() + bar.get_height()/2,
             f"{val:.3f}", va="center", fontsize=8)
ax4.grid(axis="x", color=PALETTE["grid"])

# ── (5) Confusion Matrix ──────────────────────────────────────────────────────
ax5 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax5,
            xticklabels=["Low Opp","High Opp"],
            yticklabels=["Low Opp","High Opp"],
            cbar=False, linewidths=1)
ax5.set_title("Confusion Matrix", fontweight="bold")
ax5.set_xlabel("Predicted"); ax5.set_ylabel("Actual")

# ── (6) Visibility Pain Score by Tier ────────────────────────────────────────
ax6 = fig.add_subplot(gs[2, 0])
pain_data = [df_master[df_master["company_tier"]==t]["visibility_pain_score"].values
             for t in tier_order]
bp = ax6.boxplot(pain_data, patch_artist=True, notch=True,
                 boxprops=dict(linewidth=1.5))
for patch, color in zip(bp["boxes"], [PALETTE["primary"], PALETTE["accent1"], PALETTE["accent2"]]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax6.set_xticklabels(tier_order, fontsize=9)
ax6.set_title("Visibility Pain Score by Tier", fontweight="bold")
ax6.set_ylabel("Pain Score (0-1)")
ax6.grid(axis="y", color=PALETTE["grid"])

# ── (7) Fleet Size vs GTM Score Bubble Chart ─────────────────────────────────
ax7 = fig.add_subplot(gs[2, 1])
scatter_colors = df_master["gtm_tier_label"].map(TIER_COLORS).fillna("#888888")
sc = ax7.scatter(
    df_master["fleet_size"],
    df_master["gtm_score_proba"],
    c=scatter_colors,
    s=df_master["n_trade_lanes"] * 18,
    alpha=0.65, edgecolors="white", linewidth=0.5)
ax7.set_xlabel("Fleet Size (Vessels)"); ax7.set_ylabel("GTM Score Probability")
ax7.set_title("Fleet Size vs GTM Score\n(bubble = trade lane breadth)", fontweight="bold")
from matplotlib.patches import Patch
legend_els = [Patch(fc=c, label=l) for l, c in TIER_COLORS.items()]
ax7.legend(handles=legend_els, fontsize=7.5, loc="lower right")
ax7.grid(color=PALETTE["grid"])
ax7.axhline(0.65, ls="--", color=PALETTE["accent3"], lw=1, alpha=0.6, label="Pursue threshold")

# ── (8) Deal Size Potential by GTM Tier ──────────────────────────────────────
ax8 = fig.add_subplot(gs[2, 2])
tier_deal = df_master.groupby("gtm_tier_label")["deal_size_potential_usd_k"].mean()
tier_deal_ordered = tier_deal.reindex(["C — Watch","B — Nurture","A — Pursue","S — Priority"])
ax8.bar(range(len(tier_deal_ordered)),
        tier_deal_ordered.values,
        color=[TIER_COLORS[t] for t in tier_deal_ordered.index],
        edgecolor="white", width=0.6)
ax8.set_xticks(range(len(tier_deal_ordered)))
ax8.set_xticklabels(tier_deal_ordered.index, rotation=30, ha="right", fontsize=8)
ax8.set_ylabel("Avg Deal Size (USD k)")
ax8.set_title("Avg Deal Size by GTM Tier", fontweight="bold")
ax8.grid(axis="y", color=PALETTE["grid"])
for i, v in enumerate(tier_deal_ordered.values):
    ax8.text(i, v + 0.5, f"${v:.0f}k", ha="center", fontsize=8.5, fontweight="bold")

# ── (9) AIS Data Quality vs ETA Accuracy ────────────────────────────────────
ax9 = fig.add_subplot(gs[3, :2])
high_opp = df_master[df_master["gtm_high_opportunity"] == 1]
low_opp  = df_master[df_master["gtm_high_opportunity"] == 0]
ax9.scatter(high_opp["avg_data_gaps"], high_opp["avg_eta_accuracy"],
            c=PALETTE["accent3"], alpha=0.6, s=60, label="High Opportunity", edgecolors="white")
ax9.scatter(low_opp["avg_data_gaps"],  low_opp["avg_eta_accuracy"],
            c=PALETTE["accent1"], alpha=0.4, s=40, label="Low Opportunity",  edgecolors="white")
ax9.set_xlabel("AIS Data Gap Rate (0=clean, 1=noisy)")
ax9.set_ylabel("Average ETA Accuracy")
ax9.set_title("AIS Data Quality vs ETA Accuracy — Core Pain Signal", fontweight="bold")
ax9.legend(fontsize=9); ax9.grid(color=PALETTE["grid"])

# ── (10) Top 15 Ranked Prospects ─────────────────────────────────────────────
ax10 = fig.add_subplot(gs[3, 2])
top15 = df_master.nlargest(15, "gtm_score_proba")[
    ["company_id","company_tier","gtm_tier_label","gtm_score_proba","deal_size_potential_usd_k"]
].reset_index(drop=True)
ax10.axis("off")
table_data = [[r["company_id"], r["company_tier"], f"{r['gtm_score_proba']:.2f}",
               f"${r['deal_size_potential_usd_k']:.0f}k"]
              for _, r in top15.iterrows()]
tbl = ax10.table(
    cellText=table_data,
    colLabels=["Company", "Tier", "Score", "Deal $"],
    loc="center", cellLoc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(7.5)
tbl.scale(1, 1.35)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_facecolor(PALETTE["primary"])
        cell.set_text_props(color="white", fontweight="bold")
    else:
        cell.set_facecolor("#EEF2F7" if row % 2 == 0 else "white")
    cell.set_edgecolor(PALETTE["grid"])
ax10.set_title("Top 15 Ranked Prospects", fontweight="bold", pad=12)

plt.savefig(f"{OUTPUT_DIR}/gtm_dashboard.png", dpi=150, bbox_inches="tight",
            facecolor=PALETTE["bg"])
print(f"  ✓ Dashboard saved → gtm_dashboard.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. EXPORT ARTIFACTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*70)
print("  STAGE 6 — EXPORT ARTIFACTS")
print("="*70)

# ── Ranked Prospect List ───────────────────────────────────────────────────────
export_cols = [
    "company_id","company_tier","fleet_size","annual_teu_volume",
    "n_trade_lanes","avg_delay_hours","avg_eta_accuracy",
    "visibility_pain_score","digital_readiness","gtm_score_proba",
    "gtm_tier_label","deal_size_potential_usd_k","composite_score"
]
df_export = df_master[export_cols].sort_values("gtm_score_proba", ascending=False)
df_export.to_csv(f"{OUTPUT_DIR}/gtm_ranked_prospects.csv", index=False)
print(f"  ✓ Ranked prospects CSV  → gtm_ranked_prospects.csv ({len(df_export)} rows)")

# ── Model artifact ────────────────────────────────────────────────────────────
with open(f"{OUTPUT_DIR}/gtm_scoring_model.pkl", "wb") as f:
    pickle.dump({"model": best_model, "features": FEATURE_COLS, "label_encoder": le}, f)
print(f"  ✓ Trained model pickle  → gtm_scoring_model.pkl")

# ── Summary JSON ──────────────────────────────────────────────────────────────
summary = {
    "pipeline_version":  "1.0.0",
    "best_model":        best_name,
    "test_auc":          round(test_auc, 4),
    "cv_auc_mean":       round(rf_cv.mean() if best_name=="Random Forest" else gb_cv.mean(), 4),
    "n_features":        len(FEATURE_COLS),
    "n_companies_scored":len(df_master),
    "market_sizing": {
        "TAM_usd_m": round(total_tam, 2),
        "SAM_usd_m": round(total_sam, 2),
        "SOM_usd_m": round(total_som, 2),
    },
    "gtm_tier_counts": {str(k): int(v) for k, v in score_dist.items()},
    "top5_features":   feat_imp.head(5).index.tolist(),
}
with open(f"{OUTPUT_DIR}/pipeline_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print(f"  ✓ Pipeline summary JSON → pipeline_summary.json")

print("\n" + "="*70)
print("  ✅  PIPELINE COMPLETE")
print("="*70)
print(f"""
  📊  Dashboard PNG         gtm_dashboard.png
  📋  Ranked Prospects CSV  gtm_ranked_prospects.csv
  🤖  Model Pickle          gtm_scoring_model.pkl
  📄  Summary JSON          pipeline_summary.json
""")
