"""
LMK ML Clustering & Pattern Analysis
=====================================
Run standalone:   streamlit run lmk_ml_clustering.py
Or import into the main dashboard as an extra tab.

What this does
--------------
1.  Encodes all 4 survey question responses + demographic fields as numbers
2.  Finds the optimal number of clusters (K-Means, Silhouette method)
3.  Reduces to 2D with PCA  →  interactive scatter coloured by cluster
4.  Reduces to 2D with UMAP →  second view (often reveals tighter structure)
5.  Profiles each cluster:  who is in it, how did they respond?
6.  Association analysis:   which demographic × response combinations
    co-occur more than chance? (lift > 1)
7.  Anomaly detection:      flags respondents that don't fit any cluster
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.ensemble import IsolationForest

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="LMK – ML Clustering", page_icon="🔬", layout="wide")

st.markdown("""
<style>
[data-testid="stSidebar"]          { background-color: #1E2A3A; }
[data-testid="stSidebar"] *        { color: #E8EDF2 !important; }
[data-testid="stSidebar"] label    { color: #A8C4DC !important; font-weight:600; }
.cluster-card {
    background: white; border-radius: 10px; padding: 16px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08); margin-bottom: 12px;
}
.insight-box {
    background: #EAF4FB; border-left: 4px solid #5B8DB8;
    padding: 12px 16px; border-radius: 6px; margin: 8px 0;
    font-size: 0.93rem;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette ─────────────────────────────────────────────────────────────
CLUSTER_COLORS = px.colors.qualitative.Bold
ACCENT = "#5B8DB8"
GREEN  = "#6BAE95"

# ── Response orderings (ordinal) ───────────────────────────────────────────────
USEFUL_MAP  = {"Definitely": 5, "Probably": 4, "Not sure": 3,
               "Not really": 2, "No": 1}
AGREE_MAP   = {"Strongly agree": 5, "Agree": 4,
               "Neither agree nor disagree": 3,
               "Disagree": 2, "Strongly disagree": 1}
RATING_MAP  = {"Excellent": 5, "Good": 4, "OK": 3, "Poor": 2, "Very poor": 1}

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # ── UPDATE THESE PATHS ────────────────────────────────────────────────────
    df_s = pd.read_csv("Sessions 23-24 and 24-25.csv")
    df_v = pd.read_csv("Impact surveys 23-24 and 24-25.csv")

    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()

    df_s = df_s.rename(columns={
        "Record ID": "record_id", "Module": "module",
        "Academic year": "academic_year", "Org Borough": "borough",
        "Org Type": "org_type", "Org % on school meals": "pct_school_meals",
        "Vulnerable group": "vulnerable_group",
        "Confirmed number of participants (Youth + Adult)": "actual_participants",
    })

    rename_map = {
        "Session Record ID": "session_id",
        "Record ID": "survey_id",
        "Workshop useful/helpful relationships (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8, AWP)":
            "workshop_useful",
        "Learnt something new about healthy and unhealthy behaviours in relationships (Y10SDD+AP+7/8, YIoP, YSII, AWP)":
            "learnt_healthy",
        "Changed understanding of healthy and unhealthy behaviours in relationships (ACPD)":
            "changed_understanding",
        "Leader rating (Y10SDD+AP+7/8, Y10SInc, YIoP+7/8, YSII+Inc+7/8, ACPD, AWP)":
            "leader_rating",
        "Know who and where to go if worried about relationship (Y10SDD+AP+7/8, Y10SInc+Pri, YDDPri, YIoP+7/8, YSII+Inc+7/8, AWP)":
            "know_where_to_go",
        "Age (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)": "age",
        "Gender (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8, ACPD)": "gender",
        "Ethnicity (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)": "ethnicity",
        "Disability (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)": "disability",
        "Sexuality (Y10SDD+AP, YIoP, YSII)": "sexuality",
        "Neurodivergent (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)": "neurodivergent",
    }
    df_v = df_v.rename(columns={k: v for k, v in rename_map.items()
                                 if k in df_v.columns})

    # Merge session fields
    meta = df_s[["record_id", "module", "academic_year", "borough",
                 "org_type", "pct_school_meals", "vulnerable_group"]].rename(
        columns={"record_id": "session_id"})
    df = df_v.merge(meta, on="session_id", how="left")

    # Combine learnt_healthy + changed_understanding → single "understanding" col
    if "learnt_healthy" in df.columns and "changed_understanding" in df.columns:
        df["understanding"] = df["learnt_healthy"].combine_first(df["changed_understanding"])
    elif "learnt_healthy" in df.columns:
        df["understanding"] = df["learnt_healthy"]
    elif "changed_understanding" in df.columns:
        df["understanding"] = df["changed_understanding"]
    else:
        df["understanding"] = np.nan

    return df, df_s


df_raw, df_sessions = load_data()

# ── Sidebar controls ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 ML Clustering")
    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    max_k = st.slider("Max clusters to test (K)", 2, 10, 7, key="max_k")
    n_clusters_override = st.slider(
        "Force cluster count (0 = auto-select)", 0, 10, 0, key="k_override")
    include_demographics = st.multiselect(
        "Demographics to include in clustering",
        ["gender", "ethnicity", "age", "disability", "sexuality",
         "neurodivergent", "module", "vulnerable_group"],
        default=["gender", "ethnicity", "age", "disability",
                 "sexuality", "module"],
        key="demog"
    )
    anomaly_contamination = st.slider(
        "Anomaly sensitivity (% flagged)", 1, 20, 5, key="anom") / 100

    st.markdown("---")
    st.markdown("**Filter data before clustering**")

    def ms(label, col, key):
        opts = sorted([str(o) for o in df_raw[col].dropna().unique()
                       if str(o).strip() not in ("", "nan", "Not Answered")]) \
               if col in df_raw.columns else []
        return st.multiselect(label, opts, key=key)

    sel_module = ms("Module",        "module",        "f_mod")
    sel_year   = ms("Academic Year", "academic_year", "f_yr")
    sel_borough= ms("Borough",       "borough",       "f_bor")


# ── Filter ─────────────────────────────────────────────────────────────────────
df = df_raw.copy()
if sel_module:  df = df[df["module"].isin(sel_module)]
if sel_year:    df = df[df["academic_year"].isin(sel_year)]
if sel_borough: df = df[df["borough"].isin(sel_borough)]


# ── Feature engineering ────────────────────────────────────────────────────────
@st.cache_data
def build_features(df_hash, demog_cols):
    """
    Encode all question responses (ordinal) + demographics (label-encoded).
    Returns:
      X_scaled   – scaled numeric matrix for clustering
      X_raw      – unscaled numeric matrix (for interpretation)
      mask       – boolean index of rows used
      feature_names
    """
    df = df_hash.copy()

    # Ordinal encode the 4 survey questions
    null_vals = {"Not Answered", "nan", "", "left blank", "(left blank)",
                 "N/A", "n/a", "Not sure", "I prefer not to answer"}

    def safe_map(series, mapping):
        return series.astype(str).str.strip().map(
            lambda x: np.nan if x in null_vals else mapping.get(x, np.nan)
        )

    df["q_useful"]      = safe_map(df.get("workshop_useful",   pd.Series()), USEFUL_MAP)
    df["q_understand"]  = safe_map(df.get("understanding",     pd.Series()), AGREE_MAP)
    df["q_leader"]      = safe_map(df.get("leader_rating",     pd.Series()), RATING_MAP)
    df["q_know_help"]   = safe_map(df.get("know_where_to_go",  pd.Series()), AGREE_MAP)

    q_cols = ["q_useful", "q_understand", "q_leader", "q_know_help"]

    # Label-encode demographic columns
    enc_cols = []
    encoders = {}
    for col in demog_cols:
        if col not in df.columns:
            continue
        s = df[col].astype(str).str.strip()
        s = s.replace({v: np.nan for v in null_vals})
        le = LabelEncoder()
        encoded = s.copy()
        valid_mask = s.notna()
        encoded[valid_mask] = le.fit_transform(s[valid_mask])
        encoded[~valid_mask] = np.nan
        df[f"enc_{col}"] = pd.to_numeric(encoded, errors="coerce")
        enc_cols.append(f"enc_{col}")
        encoders[col] = le

    all_cols = q_cols + enc_cols
    feature_names = q_cols + [f"enc_{c}" for c in demog_cols if f"enc_{c}" in df.columns]

    sub = df[all_cols].copy()

    # Only keep rows with at least 2 question answers
    q_filled = sub[q_cols].notna().sum(axis=1)
    mask = q_filled >= 2

    sub = sub[mask]
    # Fill remaining NaN with column median
    sub = sub.fillna(sub.median(numeric_only=True))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub)

    return X_scaled, sub.values, mask, feature_names, df[mask], encoders


# Build a hashable key from the filtered df
df_key = df.to_json()

with st.spinner("Engineering features…"):
    X_scaled, X_raw, row_mask, feat_names, df_ml, encoders = build_features(
        df, include_demographics)

n_rows = X_scaled.shape[0]
n_feats = X_scaled.shape[1]


# ── Silhouette sweep → optimal K ──────────────────────────────────────────────
@st.cache_data
def find_optimal_k(X_hash, max_k):
    X = np.frombuffer(X_hash, dtype=np.float64).reshape(-1, feat_names.__len__() if False else X_scaled.shape[1])
    scores = {}
    inertias = {}
    k_range = range(2, min(max_k + 1, n_rows))
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X)
        scores[k]   = silhouette_score(X, labels)
        inertias[k] = km.inertia_
    best_k = max(scores, key=scores.get)
    return scores, inertias, best_k


if n_rows < 4:
    st.warning(f"Only {n_rows} usable survey rows after filtering. "
               "Please broaden your filters.")
    st.stop()

with st.spinner("Testing cluster counts…"):
    sil_scores, inertias, best_k = find_optimal_k(
        X_scaled.tobytes(), max_k)

chosen_k = n_clusters_override if n_clusters_override >= 2 else best_k


# ── Final clustering ───────────────────────────────────────────────────────────
@st.cache_data
def run_clustering(X_bytes, k, shape):
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(shape)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)
    sil_vals = silhouette_samples(X, labels)
    return labels, sil_vals, km.cluster_centers_


with st.spinner("Running K-Means…"):
    cluster_labels, sil_vals, centers = run_clustering(
        X_scaled.tobytes(), chosen_k, X_scaled.shape)

df_ml = df_ml.copy()
df_ml["cluster"] = cluster_labels
df_ml["silhouette"] = sil_vals
df_ml["cluster_name"] = df_ml["cluster"].apply(lambda c: f"Cluster {c+1}")


# ── PCA (2D) ──────────────────────────────────────────────────────────────────
@st.cache_data
def run_pca(X_bytes, shape):
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(shape)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


pca_coords, pca_var = run_pca(X_scaled.tobytes(), X_scaled.shape)
df_ml["pca_x"] = pca_coords[:, 0]
df_ml["pca_y"] = pca_coords[:, 1]


# ── UMAP (2D) ─────────────────────────────────────────────────────────────────
if UMAP_AVAILABLE:
    @st.cache_data
    def run_umap(X_bytes, shape):
        X = np.frombuffer(X_bytes, dtype=np.float64).reshape(shape)
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
        return reducer.fit_transform(X)

    with st.spinner("Running UMAP…"):
        umap_coords = run_umap(X_scaled.tobytes(), X_scaled.shape)
    df_ml["umap_x"] = umap_coords[:, 0]
    df_ml["umap_y"] = umap_coords[:, 1]


# ── Anomaly detection ──────────────────────────────────────────────────────────
@st.cache_data
def run_anomaly(X_bytes, shape, contamination):
    X = np.frombuffer(X_bytes, dtype=np.float64).reshape(shape)
    iso = IsolationForest(contamination=contamination, random_state=42)
    return iso.fit_predict(X)  # -1 = anomaly, 1 = normal


anomaly_labels = run_anomaly(X_scaled.tobytes(), X_scaled.shape,
                              anomaly_contamination)
df_ml["anomaly"] = anomaly_labels == -1


# ── Cluster profiles ───────────────────────────────────────────────────────────
Q_LABELS = {
    "q_useful":     "Workshop Useful",
    "q_understand": "Changed Understanding",
    "q_leader":     "Leader Rating",
    "q_know_help":  "Know Where to Get Help",
}

def score_to_label(col, val):
    """Convert back numeric to text label for display."""
    rev_useful  = {v: k for k, v in USEFUL_MAP.items()}
    rev_agree   = {v: k for k, v in AGREE_MAP.items()}
    rev_rating  = {v: k for k, v in RATING_MAP.items()}
    maps = {"q_useful": rev_useful, "q_leader": rev_rating,
            "q_understand": rev_agree, "q_know_help": rev_agree}
    m = maps.get(col, {})
    rounded = round(val)
    return m.get(rounded, f"{val:.1f}")

def cluster_profile(df_ml, cluster_id):
    sub = df_ml[df_ml["cluster"] == cluster_id]
    n = len(sub)
    q_means = {col: sub[col].mean() for col in Q_LABELS if col in sub.columns}
    sil_mean = sub["silhouette"].mean()
    return n, q_means, sil_mean


# ─────────────────────────────────────────────────────────────────────────────
# PAGE LAYOUT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🔬 ML Pattern & Cluster Analysis")
st.caption(
    "Uses K-Means clustering on all 4 survey question responses + selected "
    "demographic features to discover hidden respondent groups.")

# ── Top summary metrics ────────────────────────────────────────────────────────
overall_sil = silhouette_score(X_scaled, cluster_labels)
pct_anomaly = df_ml["anomaly"].mean() * 100

h1, h2, h3, h4 = st.columns(4)
h1.metric("Rows used in clustering", f"{n_rows:,}")
h2.metric("Features", f"{n_feats}")
h3.metric("Clusters found", f"{chosen_k}",
          help="Optimal K by silhouette score (or forced by slider)")
h4.metric("Overall silhouette score", f"{overall_sil:.3f}",
          help="−1 to 1; higher = better-separated clusters")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Optimal K selection
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 1️⃣ Choosing the Right Number of Clusters")
st.markdown("""
Two diagnostics are used together:

- **Silhouette score** — measures how well each point fits its own cluster vs 
  the nearest other cluster. Higher = better. Peak = optimal K.
- **Elbow (inertia)** — total within-cluster variance. The "elbow" is where 
  adding more clusters stops helping much.
""")

k_vals = sorted(sil_scores.keys())
sil_vals_list = [sil_scores[k] for k in k_vals]
inert_vals    = [inertias[k]   for k in k_vals]

fig_k = make_subplots(rows=1, cols=2,
                      subplot_titles=("Silhouette Score (higher = better)",
                                      "Inertia / Elbow curve (lower = better)"))

fig_k.add_trace(go.Scatter(x=k_vals, y=sil_vals_list, mode="lines+markers",
                            marker=dict(size=10, color=ACCENT),
                            line=dict(width=2, color=ACCENT),
                            name="Silhouette"), row=1, col=1)
fig_k.add_trace(go.Scatter(x=[best_k], y=[sil_scores[best_k]],
                            mode="markers",
                            marker=dict(size=14, color="red", symbol="star"),
                            name=f"Best K={best_k}"), row=1, col=1)
fig_k.add_trace(go.Scatter(x=k_vals, y=inert_vals, mode="lines+markers",
                            marker=dict(size=10, color=GREEN),
                            line=dict(width=2, color=GREEN),
                            name="Inertia"), row=1, col=2)

fig_k.update_layout(height=340, plot_bgcolor="white", paper_bgcolor="white",
                    margin=dict(t=44, b=20), showlegend=False)
fig_k.update_xaxes(title_text="Number of clusters (K)", dtick=1)
fig_k.update_yaxes(title_text="Silhouette score", row=1, col=1)
fig_k.update_yaxes(title_text="Inertia",           row=1, col=2)
st.plotly_chart(fig_k, use_container_width=True)

st.markdown(
    f'<div class="insight-box">🔍 <b>Auto-selected K = {best_k}</b> '
    f'(silhouette = {sil_scores[best_k]:.3f}). '
    f'You can override this with the sidebar slider.</div>',
    unsafe_allow_html=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — PCA scatter
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 2️⃣ Cluster Map (PCA — 2D projection)")
st.markdown(
    f"Principal Component Analysis compresses {n_feats} features into 2 axes "
    f"(explaining **{pca_var[0]*100:.1f}% + {pca_var[1]*100:.1f}% = "
    f"{sum(pca_var)*100:.1f}%** of variance). Each dot is one survey respondent."
)

hover_cols = [c for c in ["gender", "ethnicity", "age", "module",
                           "q_useful", "q_understand", "q_leader", "q_know_help"]
              if c in df_ml.columns]

fig_pca = px.scatter(
    df_ml, x="pca_x", y="pca_y",
    color="cluster_name",
    color_discrete_sequence=CLUSTER_COLORS,
    hover_data=hover_cols,
    opacity=0.7,
    title=f"PCA – {chosen_k} clusters ({n_rows:,} respondents)",
    labels={"pca_x": f"PC1 ({pca_var[0]*100:.1f}% var.)",
            "pca_y": f"PC2 ({pca_var[1]*100:.1f}% var.)",
            "cluster_name": "Cluster"},
)
fig_pca.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
fig_pca.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                      height=480, margin=dict(t=44, b=20),
                      legend=dict(title="Cluster"))
st.plotly_chart(fig_pca, use_container_width=True)

if UMAP_AVAILABLE:
    st.markdown("## 3️⃣ Cluster Map (UMAP — non-linear projection)")
    st.markdown(
        "UMAP preserves local neighbourhood structure better than PCA, "
        "making tight natural clusters more visually apparent.")
    fig_umap = px.scatter(
        df_ml, x="umap_x", y="umap_y",
        color="cluster_name",
        color_discrete_sequence=CLUSTER_COLORS,
        hover_data=hover_cols,
        opacity=0.7,
        title=f"UMAP – {chosen_k} clusters",
        labels={"umap_x": "UMAP-1", "umap_y": "UMAP-2",
                "cluster_name": "Cluster"},
    )
    fig_umap.update_traces(marker=dict(size=7, line=dict(width=0.5, color="white")))
    fig_umap.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           height=480, margin=dict(t=44, b=20))
    st.plotly_chart(fig_umap, use_container_width=True)
else:
    st.info("Install `umap-learn` (`pip install umap-learn`) to enable the UMAP view.")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Cluster profiles
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 4️⃣ Cluster Profiles — Who is in each cluster?")
st.caption("Hover over any bar for exact values. The radar compares all clusters on the 4 questions.")

# Radar chart comparing all clusters on 4 questions
radar_cols = [c for c in Q_LABELS if c in df_ml.columns]
radar_labels = [Q_LABELS[c] for c in radar_cols]

fig_radar = go.Figure()
for cid in range(chosen_k):
    sub = df_ml[df_ml["cluster"] == cid]
    vals = [sub[c].mean() for c in radar_cols]
    vals_closed = vals + [vals[0]]
    labels_closed = radar_labels + [radar_labels[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=vals_closed, theta=labels_closed,
        fill="toself", name=f"Cluster {cid+1}",
        line_color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
        opacity=0.6,
    ))

fig_radar.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
    showlegend=True, height=420,
    title="Average Response Score per Cluster (1–5 scale)",
    paper_bgcolor="white", margin=dict(t=60, b=20),
)
st.plotly_chart(fig_radar, use_container_width=True)

# Per-cluster cards
st.markdown("### Cluster-by-cluster breakdown")
for cid in range(chosen_k):
    n_c, q_means, sil_c = cluster_profile(df_ml, cid)
    pct_c = n_c / n_rows * 100

    with st.expander(
        f"**Cluster {cid+1}**  —  {n_c:,} respondents ({pct_c:.1f}%)  "
        f"|  silhouette = {sil_c:.3f}",
        expanded=(cid == 0)
    ):
        col_q, col_d = st.columns([1, 1])

        with col_q:
            st.markdown("**📋 Average response to the 4 questions**")
            for q_col, q_label in Q_LABELS.items():
                if q_col in df_ml.columns:
                    mean_val = df_ml[df_ml["cluster"] == cid][q_col].mean()
                    label_text = score_to_label(q_col, mean_val)
                    bar_pct = int(mean_val / 5 * 100)
                    st.markdown(
                        f"**{q_label}**: {label_text} ({mean_val:.2f}/5)"
                    )
                    st.progress(bar_pct)

        with col_d:
            st.markdown("**👥 Demographic makeup**")
            sub = df_ml[df_ml["cluster"] == cid]
            for dcol in ["gender", "ethnicity", "age", "module",
                         "disability", "sexuality"]:
                if dcol in sub.columns:
                    top = sub[dcol].value_counts().head(3)
                    if not top.empty:
                        items = ", ".join(
                            [f"{v} ({c})" for v, c in top.items()])
                        st.markdown(f"**{dcol.title()}**: {items}")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Cluster composition charts
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 5️⃣ Demographic Composition of Each Cluster")

demog_display = [c for c in ["gender", "ethnicity", "age", "module",
                              "disability", "sexuality", "vulnerable_group"]
                 if c in df_ml.columns]

sel_demog = st.selectbox("Show cluster breakdown by:", demog_display,
                          key="demog_select")

ct = (df_ml.groupby(["cluster_name", sel_demog])
           .size()
           .reset_index(name="n"))
totals = ct.groupby("cluster_name")["n"].transform("sum")
ct["pct"] = (ct["n"] / totals * 100).round(1)

fig_comp = px.bar(
    ct, x="cluster_name", y="pct", color=sel_demog,
    barmode="stack",
    title=f"Cluster composition by {sel_demog.replace('_', ' ').title()}",
    labels={"cluster_name": "Cluster", "pct": "%", sel_demog: sel_demog.title()},
    color_discrete_sequence=px.colors.qualitative.Pastel,
    text=ct["pct"].apply(lambda v: f"{v:.0f}%" if v >= 5 else ""),
)
fig_comp.update_traces(textposition="inside", insidetextanchor="middle")
fig_comp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       height=420, margin=dict(t=44, b=20),
                       yaxis=dict(title="%", range=[0, 100]))
st.plotly_chart(fig_comp, use_container_width=True)

# Heatmap: mean question score per cluster
st.markdown("### Response score heatmap by cluster")
hm_data = []
for cid in range(chosen_k):
    sub = df_ml[df_ml["cluster"] == cid]
    row = {"Cluster": f"Cluster {cid+1}"}
    for q_col, q_label in Q_LABELS.items():
        if q_col in sub.columns:
            row[q_label] = round(sub[q_col].mean(), 2)
    hm_data.append(row)

hm_df = pd.DataFrame(hm_data).set_index("Cluster")
fig_hm = px.imshow(
    hm_df,
    text_auto=True,
    color_continuous_scale="RdYlGn",
    zmin=1, zmax=5,
    aspect="auto",
    title="Mean response score (1=lowest, 5=highest) per cluster × question",
)
fig_hm.update_layout(height=max(200, 60 * chosen_k + 80),
                     margin=dict(t=44, b=20),
                     paper_bgcolor="white")
st.plotly_chart(fig_hm, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Silhouette plot (quality check)
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 6️⃣ Cluster Quality — Silhouette Plot")
st.markdown("""
Each horizontal bar = one respondent. Width = silhouette score (how well they 
fit their cluster vs others). Bars extending past the dashed line are 
well-placed. Negative bars belong in the wrong cluster.
""")

sil_fig = go.Figure()
y_lower = 0
tick_positions = []
tick_labels    = []

for cid in range(chosen_k):
    mask_c = df_ml["cluster"] == cid
    vals = np.sort(df_ml.loc[mask_c, "silhouette"].values)
    size_c = len(vals)
    y_upper = y_lower + size_c

    sil_fig.add_trace(go.Bar(
        x=vals, y=list(range(y_lower, y_upper)),
        orientation="h",
        marker_color=CLUSTER_COLORS[cid % len(CLUSTER_COLORS)],
        name=f"Cluster {cid+1}",
        showlegend=True,
    ))
    tick_positions.append((y_lower + y_upper) // 2)
    tick_labels.append(f"C{cid+1}")
    y_lower = y_upper + 5

sil_fig.add_vline(x=overall_sil, line_dash="dash", line_color="red",
                  annotation_text=f"Mean={overall_sil:.3f}",
                  annotation_position="top right")
sil_fig.update_layout(
    title="Silhouette plot — width = cluster fit quality",
    xaxis_title="Silhouette coefficient",
    yaxis=dict(tickvals=tick_positions, ticktext=tick_labels, title="Cluster"),
    barmode="overlay", bargap=0,
    plot_bgcolor="white", paper_bgcolor="white",
    height=max(300, 3 * n_rows + 40),
    margin=dict(t=44, b=20),
    showlegend=True,
)
st.plotly_chart(fig_radar if False else sil_fig, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Association / lift analysis
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 7️⃣ Association Analysis — Which combinations co-occur?")
st.markdown("""
**Lift > 1** means two values appear *together more often than chance*.  
**Lift < 1** means they rarely co-occur.  
This highlights which demographic groups correlate with specific response patterns.
""")

assoc_demog = st.selectbox(
    "Demographic to cross with responses:",
    [c for c in ["gender", "ethnicity", "age", "disability",
                 "module", "sexuality", "vulnerable_group"]
     if c in df_ml.columns],
    key="assoc_d"
)
assoc_q = st.selectbox(
    "Survey question:",
    list(Q_LABELS.values()),
    key="assoc_q"
)
assoc_q_col = {v: k for k, v in Q_LABELS.items()}[assoc_q]

# Bin the question into Positive / Neutral / Negative
def bin_response(val):
    if pd.isna(val):
        return np.nan
    if val >= 4:
        return "Positive (4-5)"
    elif val == 3:
        return "Neutral (3)"
    else:
        return "Negative (1-2)"

if assoc_q_col in df_ml.columns and assoc_demog in df_ml.columns:
    df_assoc = df_ml[[assoc_demog, assoc_q_col]].copy()
    df_assoc["response_bin"] = df_assoc[assoc_q_col].apply(bin_response)
    df_assoc = df_assoc.dropna()

    N = len(df_assoc)
    rows = []
    for demog_val in df_assoc[assoc_demog].unique():
        for resp_val in df_assoc["response_bin"].unique():
            n_both  = ((df_assoc[assoc_demog] == demog_val) &
                       (df_assoc["response_bin"] == resp_val)).sum()
            n_demog = (df_assoc[assoc_demog] == demog_val).sum()
            n_resp  = (df_assoc["response_bin"] == resp_val).sum()
            if n_demog == 0 or n_resp == 0:
                continue
            support = n_both / N
            lift    = (n_both / N) / ((n_demog / N) * (n_resp / N))
            rows.append({
                "Demographic value":    demog_val,
                "Response":            resp_val,
                "Count":               int(n_both),
                "Support %":           round(support * 100, 1),
                "Lift":                round(lift, 2),
            })

    lift_df = pd.DataFrame(rows).sort_values("Lift", ascending=False)

    # Pivot for heatmap
    pivot = lift_df.pivot(index="Demographic value",
                          columns="Response", values="Lift").fillna(1)
    fig_lift = px.imshow(
        pivot,
        text_auto=True,
        color_continuous_scale="RdBu",
        color_continuous_midpoint=1,
        title=f"Lift: {assoc_demog.title()} × {assoc_q} response",
        aspect="auto",
    )
    fig_lift.update_layout(height=max(300, 30 * len(pivot) + 80),
                           paper_bgcolor="white",
                           margin=dict(t=44, b=20))
    st.plotly_chart(fig_lift, use_container_width=True)

    # Top associations table
    st.markdown("**Top 10 strongest associations (Lift > 1 = over-represented)**")
    top_lift = lift_df[lift_df["Count"] >= 5].head(10)
    st.dataframe(top_lift.style.background_gradient(subset=["Lift"], cmap="RdYlGn"),
                 use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — Anomaly detection
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 8️⃣ Anomaly Detection — Unusual Respondents")
st.markdown(f"""
**Isolation Forest** flags respondents whose answer pattern is statistically 
unusual (doesn't fit any cluster). Currently **{pct_anomaly:.1f}%** of 
respondents are flagged as anomalies (sensitivity set in sidebar).

These may represent: genuine edge-case experiences, data entry errors, 
or pupils who didn't engage with questions.
""")

fig_anom = px.scatter(
    df_ml, x="pca_x", y="pca_y",
    color=df_ml["anomaly"].map({True: "⚠️ Anomaly", False: "Normal"}),
    color_discrete_map={"⚠️ Anomaly": "red", "Normal": "#A8C4DC"},
    opacity=df_ml["anomaly"].map({True: 1.0, False: 0.4}),
    title=f"Anomalous respondents (PCA view) — {df_ml['anomaly'].sum()} flagged",
    labels={"pca_x": "PC1", "pca_y": "PC2", "color": "Status"},
    hover_data=[c for c in hover_cols if c in df_ml.columns],
)
fig_anom.update_traces(marker=dict(size=7))
fig_anom.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                       height=440, margin=dict(t=44, b=20))
st.plotly_chart(fig_anom, use_container_width=True)

# Anomaly breakdown by demo
anom_sub = df_ml[df_ml["anomaly"]]
if not anom_sub.empty:
    cols_a = st.columns(3)
    for i, dcol in enumerate(["gender", "module", "ethnicity"]):
        if dcol in anom_sub.columns:
            with cols_a[i % 3]:
                vc = anom_sub[dcol].value_counts().head(5).reset_index()
                vc.columns = [dcol.title(), "Count"]
                fig_a = px.bar(vc, x="Count", y=dcol.title(),
                               orientation="h",
                               title=f"Anomalies by {dcol.title()}",
                               color_discrete_sequence=["#D96B6B"])
                fig_a.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                    height=260, margin=dict(t=40, b=10, l=10, r=10))
                st.plotly_chart(fig_a, use_container_width=True)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — Summary insights
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("## 9️⃣ Key Findings Summary")

# Auto-generate cluster descriptions
findings = []
for cid in range(chosen_k):
    sub = df_ml[df_ml["cluster"] == cid]
    n_c = len(sub)
    pct_c = n_c / n_rows * 100
    q_avgs = {q: sub[q].mean() for q in Q_LABELS if q in sub.columns}

    best_q  = max(q_avgs, key=q_avgs.get,  default=None)
    worst_q = min(q_avgs, key=q_avgs.get, default=None)

    top_gender = sub["gender"].value_counts().idxmax() \
                 if "gender" in sub.columns and not sub["gender"].isna().all() else "N/A"
    top_module = sub["module"].value_counts().idxmax() \
                 if "module" in sub.columns and not sub["module"].isna().all() else "N/A"

    overall_pos = np.mean(list(q_avgs.values())) if q_avgs else 0
    profile_word = ("High-engagers" if overall_pos >= 4 else
                    "Moderate-engagers" if overall_pos >= 3 else "Low-engagers")

    msg = (
        f"**Cluster {cid+1}** ({n_c:,} respondents, {pct_c:.0f}%) — "
        f"*{profile_word}*. "
        f"Strongest on **{Q_LABELS.get(best_q, best_q)}** "
        f"({q_avgs.get(best_q, 0):.2f}/5), "
        f"weakest on **{Q_LABELS.get(worst_q, worst_q)}** "
        f"({q_avgs.get(worst_q, 0):.2f}/5). "
        f"Predominantly: {top_gender} | Module: {top_module}."
    )
    findings.append(msg)

for f in findings:
    st.markdown(f'<div class="insight-box">{f}</div>', unsafe_allow_html=True)

sil_interp = ("strong" if overall_sil >= 0.5 else
              "moderate" if overall_sil >= 0.3 else "weak")
st.markdown(
    f'<div class="insight-box">📐 <b>Cluster cohesion is {sil_interp}</b> '
    f'(silhouette = {overall_sil:.3f}). '
    + ("The clusters are well-separated and meaningful."
       if overall_sil >= 0.4 else
       "There is some overlap between clusters — the data has a continuum "
       "rather than sharp groupings, which is common in survey data.")
    + "</div>",
    unsafe_allow_html=True
)

st.markdown("---")
st.caption(
    "Methods: K-Means clustering · PCA · UMAP · Isolation Forest · "
    "Association Lift analysis. "
    "Ordinal encoding: question responses mapped to 1–5 scale before clustering."
)
