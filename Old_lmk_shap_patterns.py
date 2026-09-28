"""
LMK - SHAP Feature Importance + Pattern / Interaction Analysis
==============================================================
Run:  streamlit run lmk_shap_patterns.py

What this script does
---------------------
For EACH of the 4 survey outcome questions it:
  1. Trains a GradientBoostingClassifier (positive vs non-positive response)
  2. Computes SHAP values for every feature (demog + session-level)
  3. Plots SHAP summary bar, beeswarm, and dependence plots
  4. Detects pairwise interaction patterns between ALL feature combinations
     (Spearman correlation, Chi-squared, Cramer's V, and SHAP interaction)
  5. Highlights the strongest patterns in plain English
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings, itertools
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from scipy import stats
import shap

# -- Page setup ----------------------------------------------------------------
st.set_page_config(page_title="LMK - SHAP & Patterns", page_icon="-", layout="wide")

st.markdown("""
<style>
/* ── Sidebar: always dark navy regardless of theme ── */
[data-testid="stSidebar"]                    { background-color: #1A2535 !important; }
[data-testid="stSidebar"] *                  { color: #DCE8F5 !important; }
[data-testid="stSidebar"] label              { color: #90C0E0 !important; font-weight:600; }
[data-testid="stSidebar"] .stSelectbox > div { background:#243447 !important; }
[data-testid="stSidebar"] hr                 { border-color:#2E4460; }

/* ── Info / insight boxes: adapt to light AND dark theme ── */
@media (prefers-color-scheme: light) {
    .insight-box  { background:#EAF4FB; border-left:4px solid #3A7DC0;
                    color:#1A3A55; padding:10px 16px; border-radius:7px;
                    margin:6px 0; font-size:.92rem; }
    .warn-box     { background:#FFF8E6; border-left:4px solid #E09B00;
                    color:#5A3E00; padding:10px 16px; border-radius:7px;
                    margin:6px 0; font-size:.92rem; }
    .strong-box   { background:#E8F5EC; border-left:4px solid #2E9E52;
                    color:#1A4A28; padding:10px 16px; border-radius:7px;
                    margin:6px 0; font-size:.92rem; }
}
@media (prefers-color-scheme: dark) {
    .insight-box  { background:#1C3248; border-left:4px solid #5BA8E0;
                    color:#B8D8F5; padding:10px 16px; border-radius:7px;
                    margin:6px 0; font-size:.92rem; }
    .warn-box     { background:#3A2E10; border-left:4px solid #E0B040;
                    color:#F0D080; padding:10px 16px; border-radius:7px;
                    margin:6px 0; font-size:.92rem; }
    .strong-box   { background:#1A3828; border-left:4px solid #40C070;
                    color:#80E0A8; padding:10px 16px; border-radius:7px;
                    margin:6px 0; font-size:.92rem; }
}
/* Streamlit dark-mode class fallback (Streamlit sets data-theme) */
[data-theme="dark"] .insight-box  { background:#1C3248; border-left:4px solid #5BA8E0;
                                    color:#B8D8F5; }
[data-theme="dark"] .warn-box     { background:#3A2E10; border-left:4px solid #E0B040;
                                    color:#F0D080; }
[data-theme="dark"] .strong-box   { background:#1A3828; border-left:4px solid #40C070;
                                    color:#80E0A8; }
</style>
""", unsafe_allow_html=True)

# -- Colour palette (works on both light and dark backgrounds) -----------------
# Charts use explicit background so they always render correctly
ACCENT  = "#4A9FD4"   # blue
GREEN   = "#4EAF82"   # green
ORANGE  = "#E89050"   # orange
RED     = "#D95F5F"   # red
PURPLE  = "#9B72C0"   # purple

# Chart bg/text colours that adapt
CHART_BG   = "rgba(0,0,0,0)"   # transparent - inherits page bg
CHART_TEXT = "#555555"          # neutral grey - readable on both themes

# Plotly template: "plotly_white" for light, "plotly_dark" available via sidebar toggle
def chart_layout(height=340, extra=None):
    base = dict(
        plot_bgcolor="rgba(245,248,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#444444", size=12),
        margin=dict(t=44, b=20, l=10, r=10),
        height=height,
    )
    if extra:
        base.update(extra)
    return base

# -- Response orderings ---------------------------------------------------------
USEFUL_MAP = {"Definitely":5,"Probably":4,"Not sure":3,"Not really":2,"No":1}
AGREE_MAP  = {"Strongly agree":5,"Agree":4,"Neither agree nor disagree":3,
              "Disagree":2,"Strongly disagree":1}
RATING_MAP = {"Excellent":5,"Good":4,"OK":3,"Poor":2,"Very poor":1}

NULL_VALS = {"Not Answered","nan","","left blank","(left blank)","N/A","n/a",
             "I prefer not to answer","Not sure"}

QUESTIONS = {
    "workshop_useful":    ("Workshop Useful",         USEFUL_MAP,  {4,5}),
    "understanding":      ("Changed Understanding",   AGREE_MAP,   {4,5}),
    "leader_rating":      ("LMK Leader Rating",       RATING_MAP,  {4,5}),
    "know_where_to_go":   ("Know Where to Get Help",  AGREE_MAP,   {4,5}),
}

# -- Load data ------------------------------------------------------------------
@st.cache_data
def load_data():
    df_s = pd.read_csv("Sessions 23-24 and 24-25.csv")
    df_v = pd.read_csv("Impact surveys 23-24 and 24-25.csv")
    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()

    df_s = df_s.rename(columns={
        "Record ID":"record_id","Module":"module",
        "Academic year":"academic_year","Org Borough":"borough",
        "Org Type":"org_type","Org Sub Type":"org_sub_type",
        "Org % on school meals":"pct_school_meals",
        "Vulnerable group":"vulnerable_group",
        "Confirmed number of participants (Youth + Adult)":"actual_participants",
        "Confirmed number of expected participants (Youth + Adult)":"expected_participants",
        "Number of surveys":"num_surveys",
    })

    rn = {
        "Session Record ID":"session_id","Record ID":"survey_id",
        "Workshop useful/helpful relationships (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8, AWP)":"workshop_useful",
        "Learnt something new about healthy and unhealthy behaviours in relationships (Y10SDD+AP+7/8, YIoP, YSII, AWP)":"learnt_healthy",
        "Changed understanding of healthy and unhealthy behaviours in relationships (ACPD)":"changed_understanding",
        "Leader rating (Y10SDD+AP+7/8, Y10SInc, YIoP+7/8, YSII+Inc+7/8, ACPD, AWP)":"leader_rating",
        "Know who and where to go if worried about relationship (Y10SDD+AP+7/8, Y10SInc+Pri, YDDPri, YIoP+7/8, YSII+Inc+7/8, AWP)":"know_where_to_go",
        "Age (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":"age",
        "Gender (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8, ACPD)":"gender",
        "Ethnicity (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":"ethnicity",
        "Disability (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":"disability",
        "Sexuality (Y10SDD+AP, YIoP, YSII)":"sexuality",
        "Neurodivergent (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":"neurodivergent",
    }
    df_v = df_v.rename(columns={k:v for k,v in rn.items() if k in df_v.columns})

    # Combine understanding columns
    if "learnt_healthy" in df_v.columns and "changed_understanding" in df_v.columns:
        df_v["understanding"] = df_v["learnt_healthy"].combine_first(df_v["changed_understanding"])
    elif "learnt_healthy" in df_v.columns:
        df_v["understanding"] = df_v["learnt_healthy"]
    elif "changed_understanding" in df_v.columns:
        df_v["understanding"] = df_v["changed_understanding"]

    # Merge session fields onto survey
    meta = df_s[["record_id","module","academic_year","borough","org_type",
                 "org_sub_type","pct_school_meals","vulnerable_group"]].rename(
        columns={"record_id":"session_id"})
    df = df_v.merge(meta, on="session_id", how="left")

    return df, df_s

df_raw, df_sessions = load_data()

# -- Feature definitions --------------------------------------------------------
FEATURE_COLS = [
    # Demographic (pupil-level)
    "gender","ethnicity","age","disability","sexuality","neurodivergent",
    # Session-level
    "module","borough","org_type","org_sub_type","vulnerable_group",
    "academic_year",
    # Numeric session-level
    "pct_school_meals",
]
FEATURE_COLS = [c for c in FEATURE_COLS if c in df_raw.columns]

# -- Sidebar --------------------------------------------------------------------
with st.sidebar:
    st.markdown("## - SHAP & Patterns")
    st.markdown("---")

    def ms(label, col, key):
        opts = sorted([str(o) for o in df_raw[col].dropna().unique()
                       if str(o).strip() not in NULL_VALS]) \
               if col in df_raw.columns else []
        return st.multiselect(label, opts, key=key)

    st.markdown("**Filter data**")
    sel_module = ms("Module",        "module",        "sm")
    sel_year   = ms("Academic Year", "academic_year", "sy")
    sel_borough= ms("Borough",       "borough",       "sb")

    st.markdown("---")
    st.markdown("**Analysis settings**")
    min_support = st.slider("Min. responses per cell (pattern filter)",
                            5, 50, 10, key="minsup")
    cramers_thresh = st.slider("Cramer's V threshold (show strong patterns)",
                               0.1, 0.5, 0.15, step=0.01, key="cv")
    n_trees = st.slider("GBM trees (more = slower but more accurate SHAP)",
                        50, 300, 100, step=50, key="ntree")
    st.markdown("---")
    if st.button("- Reset", use_container_width=True):
        for k in ["sm","sy","sb"]:
            st.session_state[k] = []
        st.rerun()

# -- Filter ---------------------------------------------------------------------
df = df_raw.copy()
if sel_module:  df = df[df["module"].isin(sel_module)]
if sel_year:    df = df[df["academic_year"].isin(sel_year)]
if sel_borough: df = df[df["borough"].isin(sel_borough)]

# -- Feature matrix builder -----------------------------------------------------
@st.cache_data
def build_feature_matrix(df_json):
    df = pd.read_json(df_json)
    encoders = {}
    X = pd.DataFrame(index=df.index)

    for col in FEATURE_COLS:
        if col not in df.columns:
            continue
        if col == "pct_school_meals":
            X[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            s = df[col].astype(str).str.strip()
            s = s.where(~s.isin(NULL_VALS), np.nan)
            le = LabelEncoder()
            valid = s.notna()
            enc = s.copy().astype(object)
            if valid.sum() > 0:
                enc[valid] = le.fit_transform(s[valid])
                encoders[col] = le
            enc[~valid] = np.nan
            X[col] = pd.to_numeric(enc, errors="coerce")

    return X, encoders

X_all, encoders = build_feature_matrix(df.to_json())

# -- Ordinal-encode targets -----------------------------------------------------
def encode_target(series, mapping, positive_set):
    """Returns (numeric 1-5, binary 0/1) series."""
    def safe_map(x):
        x = str(x).strip()
        return np.nan if x in NULL_VALS else mapping.get(x, np.nan)
    num = series.apply(safe_map)
    binary = num.apply(lambda v: 1 if (not pd.isna(v) and int(v) in positive_set) else
                       (0 if not pd.isna(v) else np.nan))
    return num, binary

# -- Helper: Cramer's V --------------------------------------------------------
def cramers_v(x, y):
    """Cramer's V association between two categorical series."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return 0.0
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    v = np.sqrt(phi2 / min(k-1, r-1)) if min(k-1, r-1) > 0 else 0.0
    return round(float(v), 4), float(p)

# -- Helper: % positive breakdown for a crosstab -------------------------------
def pct_positive_crosstab(df, col_a, col_b, q_col, pos_set):
    """
    For each (col_a, col_b) combination, what % gave a positive response?
    Returns a pivot table.
    """
    sub = df[[col_a, col_b, q_col]].dropna()
    sub = sub[sub[q_col].notna()]
    sub["positive"] = sub[q_col].isin(pos_set).astype(int)
    grp = sub.groupby([col_a, col_b])["positive"].agg(["mean","count"]).reset_index()
    grp = grp[grp["count"] >= min_support]
    grp["pct"] = (grp["mean"]*100).round(1)
    return grp

# -- SHAP for one question ------------------------------------------------------
@st.cache_data
def compute_shap(df_json, q_col, n_est):
    df = pd.read_json(df_json)
    X_loc, _ = build_feature_matrix(df.to_json())

    # Build target
    mapping = (USEFUL_MAP if q_col=="workshop_useful" else
               RATING_MAP if q_col=="leader_rating" else AGREE_MAP)
    pos_set = {4,5}
    num_tgt, bin_tgt = encode_target(df[q_col] if q_col in df.columns
                                     else pd.Series(np.nan, index=df.index),
                                     mapping, pos_set)

    # Align rows
    valid_idx = bin_tgt.dropna().index
    X_m = X_all.loc[valid_idx].copy()
    y_m = bin_tgt.loc[valid_idx].astype(int)

    # Fill NaN with median
    X_m = X_m.fillna(X_m.median(numeric_only=True))

    if len(y_m) < 20 or y_m.nunique() < 2:
        return None, None, None, 0, 0

    model = GradientBoostingClassifier(n_estimators=n_est, max_depth=3,
                                       random_state=42)
    model.fit(X_m, y_m)

    auc = cross_val_score(model, X_m, y_m, cv=3,
                          scoring="roc_auc").mean()

    explainer = shap.TreeExplainer(model)
    shap_vals  = explainer.shap_values(X_m)

    return shap_vals, X_m, model, auc, len(y_m)

# -- Main layout ----------------------------------------------------------------
st.markdown("# - SHAP Feature Importance + Pattern Analysis")
st.caption(
    "Each survey outcome is modelled with a GradientBoostingClassifier. "
    "SHAP values reveal **which features drive positive vs negative responses**. "
    "Interaction analysis then finds patterns across ALL feature combinations.")

st.markdown(f"**Dataset:** {len(df):,} survey rows after filters applied.")
st.markdown("---")

# ------------------------------------------------------------------------------
# PART 1 -- SHAP PER QUESTION
# ------------------------------------------------------------------------------
st.markdown("## - Part 1 -- SHAP Analysis for Each Outcome Question")
st.markdown("""
**How to read SHAP:**
- **Bar chart** = mean absolute impact of each feature on predictions
- **Beeswarm** = each dot is one person; colour = feature value (red=high, blue=low); 
  x-position = how much it pushed the prediction toward positive (right) or negative (left)
- **AUC** = model accuracy (0.5 = random, 1.0 = perfect)
""")

q_tabs = st.tabs([v[0] for v in QUESTIONS.values()])

all_shap_results = {}

for tab, (q_col, (q_label, q_map, q_pos)) in zip(q_tabs, QUESTIONS.items()):
    with tab:
        if q_col not in df.columns:
            st.warning(f"Column `{q_col}` not found in the filtered data.")
            continue

        with st.spinner(f"Computing SHAP for: {q_label}..."):
            shap_vals, X_m, model, auc, n_used = compute_shap(
                df.to_json(), q_col, n_trees)

        if shap_vals is None:
            st.warning("Not enough data to model this question with current filters.")
            continue

        all_shap_results[q_col] = (shap_vals, X_m, model, auc)

        # KPI strip
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows modelled", f"{n_used:,}")
        c2.metric("Model AUC (3-fold CV)", f"{auc:.3f}",
                  help="0.5=random, 1.0=perfect. >0.6 is informative.")
        c3.metric("Features used", f"{X_m.shape[1]}")

        auc_note = ("- Model is informative -- SHAP values are reliable."
                    if auc >= 0.6 else
                    "-- AUC near chance -- treat SHAP patterns as indicative only.")
        st.markdown(f'<div class="{"strong-box" if auc>=0.6 else "warn-box"}">'
                    f'{auc_note}</div>', unsafe_allow_html=True)

        st.markdown("---")

        # -- SHAP Bar (mean |SHAP|) -----------------------------------------
        mean_shap = np.abs(shap_vals).mean(axis=0)
        feat_imp  = pd.DataFrame({
            "Feature": X_m.columns,
            "Mean |SHAP|": mean_shap
        }).sort_values("Mean |SHAP|", ascending=True)

        fig_bar = px.bar(
            feat_imp, x="Mean |SHAP|", y="Feature", orientation="h",
            title=f"SHAP Feature Importance -- {q_label}",
            color="Mean |SHAP|", color_continuous_scale="Blues",
            text=feat_imp["Mean |SHAP|"].apply(lambda v: f"{v:.3f}"),
        )
        fig_bar.update_traces(textposition="outside", marker_line_width=0)
        fig_bar.update_layout(
            plot_bgcolor="rgba(245,248,252,1)", paper_bgcolor="rgba(0,0,0,0)",
            height=max(300, 28*len(feat_imp)+60),
            margin=dict(t=44,b=20,l=10,r=60),
            coloraxis_showscale=False,
            yaxis=dict(title=""),
            xaxis=dict(title="Mean |SHAP value| - more impact"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # -- SHAP Beeswarm --------------------------------------------------
        st.markdown("#### SHAP Beeswarm -- direction and magnitude of each feature")
        st.caption("Red = high feature value - Blue = low - Right = pushed toward Positive response")

        # Build beeswarm data manually (no matplotlib needed)
        beeswarm_rows = []
        for i, feat in enumerate(X_m.columns):
            sv   = shap_vals[:, i]
            fv   = X_m[feat].values
            fv_n = (fv - np.nanmin(fv)) / (np.nanmax(fv) - np.nanmin(fv) + 1e-9)
            # Add jitter on y-axis for readability
            jitter = np.random.uniform(-0.35, 0.35, size=len(sv))
            for s, f, fn, j in zip(sv, fv, fv_n, jitter):
                beeswarm_rows.append({
                    "feature": feat,
                    "SHAP value": round(float(s), 4),
                    "feature_norm": round(float(fn), 3),
                    "y_jitter": i + float(j),
                })

        bee_df = pd.DataFrame(beeswarm_rows)

        fig_bee = go.Figure()
        fig_bee.add_trace(go.Scatter(
            x=bee_df["SHAP value"],
            y=bee_df["y_jitter"],
            mode="markers",
            marker=dict(
                color=bee_df["feature_norm"],
                colorscale="RdBu_r",
                size=4,
                opacity=0.6,
                colorbar=dict(title="Feature value<br>(normalised)",
                              thickness=12, len=0.6),
                showscale=True,
            ),
            text=bee_df["feature"],
            hovertemplate="%{text}<br>SHAP: %{x:.3f}<extra></extra>",
        ))

        feat_order = list(X_m.columns)
        fig_bee.update_layout(
            title=f"SHAP Beeswarm -- {q_label}",
            xaxis=dict(title="SHAP value  (- Negative | Positive -)",
                       zeroline=True, zerolinecolor="black", zerolinewidth=1),
            yaxis=dict(tickvals=list(range(len(feat_order))),
                       ticktext=feat_order, title=""),
            plot_bgcolor="rgba(245,248,252,1)", paper_bgcolor="rgba(0,0,0,0)",
            height=max(350, 30*len(feat_order)+60),
            margin=dict(t=44, b=20, l=10, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig_bee, use_container_width=True)

        # -- Top-3 feature insights -----------------------------------------
        top3 = feat_imp.sort_values("Mean |SHAP|", ascending=False).head(3)
        st.markdown("#### - Top drivers for this question")
        for _, row in top3.iterrows():
            feat = row["Feature"]
            imp  = row["Mean |SHAP|"]
            # Direction: positive or negative on average?
            idx  = list(X_m.columns).index(feat)
            mean_sv = shap_vals[:, idx].mean()
            direction = "increases" if mean_sv > 0 else "decreases"
            # Decode top value
            if feat in encoders:
                le = encoders[feat]
                fv_vals = X_m[feat].dropna()
                if len(fv_vals) > 0:
                    top_code = fv_vals.value_counts().idxmax()
                    try:
                        top_label = le.inverse_transform([int(top_code)])[0]
                    except Exception:
                        top_label = str(top_code)
                else:
                    top_label = "N/A"
                txt = (f"<b>{feat.replace('_',' ').title()}</b> is the #{list(top3['Feature']).index(feat)+1} "
                       f"driver (impact={imp:.3f}). Having higher values generally "
                       f"<b>{direction}</b> the chance of a positive response. "
                       f"Most common value in data: <i>{top_label}</i>.")
            else:
                txt = (f"<b>{feat.replace('_',' ').title()}</b> is the #{list(top3['Feature']).index(feat)+1} "
                       f"driver (impact={imp:.3f}). Higher values generally "
                       f"<b>{direction}</b> the chance of a positive response.")
            st.markdown(f'<div class="insight-box">{txt}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # -- SHAP Dependence plots for top 2 features -----------------------
        st.markdown("#### SHAP Dependence -- how each top feature affects the outcome")
        top2 = feat_imp.sort_values("Mean |SHAP|", ascending=False).head(2)["Feature"].tolist()
        dep_cols = st.columns(len(top2))

        for dcol, feat in zip(dep_cols, top2):
            with dcol:
                idx = list(X_m.columns).index(feat)
                sv  = shap_vals[:, idx]
                fv  = X_m[feat].values

                dep_df = pd.DataFrame({feat: fv, "SHAP": sv}).dropna()

                # Manual smoothing line (no statsmodels needed)
                dep_sorted = dep_df.sort_values(feat)
                window = max(1, len(dep_sorted) // 10)
                dep_sorted["smooth"] = (dep_sorted["SHAP"]
                                        .rolling(window, center=True, min_periods=1)
                                        .mean())

                fig_dep = go.Figure()
                fig_dep.add_trace(go.Scatter(
                    x=dep_df[feat], y=dep_df["SHAP"],
                    mode="markers",
                    marker=dict(color=ACCENT, size=5, opacity=0.45),
                    name="SHAP value",
                ))
                fig_dep.add_trace(go.Scatter(
                    x=dep_sorted[feat], y=dep_sorted["smooth"],
                    mode="lines",
                    line=dict(color=ORANGE, width=2.5),
                    name="Trend",
                ))
                fig_dep.update_layout(
                    title=f"Dependence: {feat.replace('_',' ').title()}",
                    xaxis_title=feat.replace("_"," ").title(),
                    yaxis_title="SHAP value",
                    plot_bgcolor="rgba(245,248,252,1)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#444444"),
                    height=300,
                    margin=dict(t=44,b=20,l=10,r=10),
                    showlegend=False,
                )
                st.plotly_chart(fig_dep, use_container_width=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# PART 2 -- ALL-PAIR PATTERN / INTERACTION ANALYSIS
# ------------------------------------------------------------------------------
st.markdown("## - Part 2 -- Pairwise Pattern Detection Across ALL Feature Combinations")
st.markdown("""
We test **every pair of categorical features** for statistical association using 
**Cramer's V** (ranges 0-1: 0=no association, 1=perfect). We also check specific 
combinations you asked about: Free School Meals - Borough - Gender, and extend 
to all other triple combinations.

**Thresholds used:** Weak < 0.1 - Moderate 0.1-0.2 - Strong > 0.2
""")

# -- Categorical features for pair testing -------------------------------------
CAT_FEATS = [c for c in [
    "gender","ethnicity","age","disability","sexuality","neurodivergent",
    "module","borough","org_type","org_sub_type","vulnerable_group",
    "academic_year",
] if c in df.columns]

# Numeric feature for bucketing
NUM_FEAT = "pct_school_meals"

# Bucket FSM
if NUM_FEAT in df.columns:
    fsm_num = pd.to_numeric(df[NUM_FEAT], errors="coerce")
    df["fsm_bucket"] = pd.cut(fsm_num,
        bins=[0,20,40,60,80,101],
        labels=["0-20%","21-40%","41-60%","61-80%","81-100%"],
        right=False)
    CAT_FEATS_FULL = CAT_FEATS + ["fsm_bucket"]
else:
    CAT_FEATS_FULL = CAT_FEATS

# -- Compute all pairwise Cramer's V -------------------------------------------
@st.cache_data
def compute_all_pairs(df_json, feats):
    df = pd.read_json(df_json)
    results = []
    for f1, f2 in itertools.combinations(feats, 2):
        if f1 not in df.columns or f2 not in df.columns:
            continue
        sub = df[[f1, f2]].dropna()
        sub = sub[~sub[f1].astype(str).isin(NULL_VALS)]
        sub = sub[~sub[f2].astype(str).isin(NULL_VALS)]
        if len(sub) < min_support:
            continue
        try:
            cv, pval = cramers_v(sub[f1].astype(str), sub[f2].astype(str))
        except Exception:
            continue
        results.append({
            "Feature A": f1.replace("_"," ").title(),
            "Feature B": f2.replace("_"," ").title(),
            "feat_a": f1, "feat_b": f2,
            "Cramer's V": cv,
            "p-value": round(pval, 5),
            "n": len(sub),
            "Strength": ("Strong" if cv >= 0.2 else
                         "Moderate" if cv >= 0.1 else "Weak"),
        })
    return pd.DataFrame(results).sort_values("Cramer's V", ascending=False)

with st.spinner("Computing all pairwise associations..."):
    pair_df = compute_all_pairs(df.to_json(), CAT_FEATS_FULL)

# -- Heatmap of all Cramer's V values ------------------------------------------
st.markdown("### -- Association Heatmap -- All Feature Pairs")
st.caption("Darker = stronger association between the two features.")

if not pair_df.empty:
    feat_labels = sorted(set(pair_df["Feature A"]) | set(pair_df["Feature B"]))
    mat = pd.DataFrame(0.0, index=feat_labels, columns=feat_labels)
    for _, row in pair_df.iterrows():
        mat.loc[row["Feature A"], row["Feature B"]] = row["Cramer's V"]
        mat.loc[row["Feature B"], row["Feature A"]] = row["Cramer's V"]
    np.fill_diagonal(mat.values, 1.0)

    fig_heat = px.imshow(
        mat, text_auto=".2f",
        color_continuous_scale="Blues",
        zmin=0, zmax=1,
        title="Cramer's V -- Pairwise Feature Associations (0=none, 1=perfect)",
        aspect="auto",
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(400, 40*len(feat_labels)+80),
        margin=dict(t=60,b=20,l=10,r=10),
        coloraxis_colorbar=dict(title="Cramer's V"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # -- Ranked table -------------------------------------------------------
    st.markdown("### - All Pairs Ranked by Strength")
    disp_df = pair_df[["Feature A","Feature B","Cramer's V","p-value","n","Strength"]].copy()
    disp_df["Cramer's V"] = disp_df["Cramer's V"].round(3)
    disp_df["p-value"]    = disp_df["p-value"].round(5)

    def color_strength(val):
        if val == "Strong":   return "background-color:#D4EDDA; color:#155724"
        if val == "Moderate": return "background-color:#FFF3CD; color:#856404"
        return "background-color:#F8D7DA; color:#721c24"

    st.dataframe(
        disp_df.style.applymap(color_strength, subset=["Strength"])
                     .background_gradient(subset=["Cramer's V"], cmap="Blues"),
        use_container_width=True, height=380,
    )

    # Strong patterns summary
    strong = pair_df[pair_df["Cramer's V"] >= cramers_thresh]
    if not strong.empty:
        st.markdown("### - Strong Patterns Found")
        for _, row in strong.iterrows():
            cv_val = row["Cramer's V"]
            pval   = row["p-value"]
            sig    = "statistically significant (p<0.05)" if pval < 0.05 else "not statistically significant"
            st.markdown(
                f'<div class="strong-box">- <b>{row["Feature A"]}</b> and '
                f'<b>{row["Feature B"]}</b> have a <b>{row["Strength"].lower()} '
                f'association</b> (Cramer\'s V = {cv_val:.3f}, {sig}, n={row["n"]:,}). '
                f'These two features move together -- knowing one helps predict the other.'
                f'</div>', unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div class="warn-box">No pairs above the Cramer\'s V threshold '
            f'of {cramers_thresh}. Try lowering the threshold in the sidebar.</div>',
            unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# PART 3 -- SPECIFIC COMBINATIONS: FSM - BOROUGH - GENDER (+ all triples)
# ------------------------------------------------------------------------------
st.markdown("## - Part 3 -- Specific Patterns: FSM - Borough - Gender (and more)")
st.markdown("""
For each outcome question, we show what % gave a **positive response** 
broken down by pairs and triples of features you care about.
""")

Q_SEL = st.selectbox(
    "Choose outcome question to analyse:",
    list(QUESTIONS.keys()),
    format_func=lambda k: QUESTIONS[k][0],
    key="q_sel"
)
q_label_sel, q_map_sel, q_pos_sel = QUESTIONS[Q_SEL]

# Encode target for the selected question
if Q_SEL in df.columns:
    def safe_map_target(x, mapping):
        x = str(x).strip()
        return np.nan if x in NULL_VALS else mapping.get(x, np.nan)
    df["_target_num"] = df[Q_SEL].apply(lambda x: safe_map_target(x, q_map_sel))
    df["_target_pos"] = df["_target_num"].apply(
        lambda v: 1 if (not pd.isna(v) and int(v) in q_pos_sel) else
                  (0 if not pd.isna(v) else np.nan))
else:
    df["_target_num"] = np.nan
    df["_target_pos"] = np.nan

# -- A) The asked combination: FSM - Borough - Gender -------------------------
st.markdown(f"### A) Free School Meals - Borough - Gender - *{q_label_sel}*")

fsm_ok     = "fsm_bucket" in df.columns
borough_ok = "borough"    in df.columns
gender_ok  = "gender"     in df.columns

if fsm_ok and borough_ok and gender_ok:
    sub3 = df[["fsm_bucket","borough","gender","_target_pos"]].dropna()
    sub3 = sub3[~sub3["gender"].astype(str).isin(NULL_VALS)]
    sub3 = sub3[~sub3["borough"].astype(str).isin(NULL_VALS)]

    agg3 = (sub3.groupby(["fsm_bucket","borough","gender"])["_target_pos"]
               .agg(["mean","count"]).reset_index())
    agg3 = agg3[agg3["count"] >= min_support]
    agg3["% Positive"] = (agg3["mean"]*100).round(1)
    agg3.columns = ["FSM Band","Borough","Gender","mean","n","% Positive"]

    if not agg3.empty:
        # Faceted heatmap: FSM band on rows, Borough on columns, Gender as facet
        genders = agg3["Gender"].unique()
        gender_tabs = st.tabs([f"Gender: {g}" for g in sorted(genders)])
        for gtab, gval in zip(gender_tabs, sorted(genders)):
            with gtab:
                sub_g = agg3[agg3["Gender"]==gval]
                if sub_g.empty:
                    st.info("Not enough data for this gender with current filters.")
                    continue
                pivot = sub_g.pivot(index="FSM Band", columns="Borough",
                                    values="% Positive")
                fig_3 = px.imshow(
                    pivot, text_auto=True,
                    color_continuous_scale="RdYlGn",
                    zmin=0, zmax=100,
                    title=f"% Positive ({q_label_sel}) -- Gender: {gval}",
                    aspect="auto",
                    labels=dict(color="% Positive"),
                )
                fig_3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                                    height=max(250,50*len(pivot)+100),
                                    margin=dict(t=50,b=20))
                st.plotly_chart(fig_3, use_container_width=True)

                # Best and worst cells
                best  = sub_g.loc[sub_g["% Positive"].idxmax()]
                worst = sub_g.loc[sub_g["% Positive"].idxmin()]
                st.markdown(
                    f'<div class="strong-box">- <b>Highest</b>: '
                    f'{best["FSM Band"]} FSM / {best["Borough"]} - '
                    f'{best["% Positive"]}% positive (n={int(best["n"])})</div>',
                    unsafe_allow_html=True)
                st.markdown(
                    f'<div class="warn-box">-- <b>Lowest</b>: '
                    f'{worst["FSM Band"]} FSM / {worst["Borough"]} - '
                    f'{worst["% Positive"]}% positive (n={int(worst["n"])})</div>',
                    unsafe_allow_html=True)
    else:
        st.info("Not enough data cells meeting the minimum support threshold. "
                "Try lowering 'Min. responses per cell' in the sidebar.")
else:
    missing = [c for c, ok in [("fsm_bucket",fsm_ok),("borough",borough_ok),
                                ("gender",gender_ok)] if not ok]
    st.warning(f"Missing columns: {missing}")

st.markdown("---")

# -- B) All meaningful pairwise combinations - selected question ---------------
st.markdown(f"### B) All Feature Pairs vs *{q_label_sel}* (% Positive Heatmaps)")
st.caption(f"Each heatmap shows % positive response for '{q_label_sel}' "
           f"across every pair of features. Only cells with -{min_support} responses shown.")

# Identify top pairs by Cramer's V (limit to top 8 most interesting pairs)
TOP_N_PAIRS = 8
if not pair_df.empty:
    top_pairs = pair_df.head(TOP_N_PAIRS)[["feat_a","feat_b"]].values.tolist()
else:
    # Fallback: manual key pairs
    top_pairs = [
        ["gender","borough"],
        ["gender","module"],
        ["ethnicity","borough"],
        ["fsm_bucket","borough"],
        ["fsm_bucket","gender"],
        ["age","gender"],
        ["disability","gender"],
        ["vulnerable_group","borough"],
    ]
    top_pairs = [(a,b) for a,b in top_pairs
                 if a in df.columns and b in df.columns]

for feat_a, feat_b in top_pairs:
    fa_label = feat_a.replace("_"," ").title()
    fb_label = feat_b.replace("_"," ").title()

    sub_p = df[[feat_a, feat_b, "_target_pos"]].dropna()
    sub_p = sub_p[~sub_p[feat_a].astype(str).isin(NULL_VALS)]
    sub_p = sub_p[~sub_p[feat_b].astype(str).isin(NULL_VALS)]

    agg_p = (sub_p.groupby([feat_a, feat_b])["_target_pos"]
                  .agg(["mean","count"]).reset_index())
    agg_p = agg_p[agg_p["count"] >= min_support]
    agg_p["pct"] = (agg_p["mean"]*100).round(1)

    if agg_p.empty:
        continue

    pivot_p = agg_p.pivot(index=feat_a, columns=feat_b, values="pct")

    # Find CV for this pair
    cv_row = pair_df[(pair_df["feat_a"]==feat_a) & (pair_df["feat_b"]==feat_b)]
    if cv_row.empty:
        cv_row = pair_df[(pair_df["feat_a"]==feat_b) & (pair_df["feat_b"]==feat_a)]
    cv_col = "Cramer's V"
    cv_str = (f"  |  Cramer's V = {cv_row[cv_col].values[0]:.3f}"
              if not cv_row.empty else "")

    fig_pair = px.imshow(
        pivot_p, text_auto=True,
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=100,
        title=f"{fa_label} - {fb_label} - % Positive: {q_label_sel}{cv_str}",
        aspect="auto",
        labels=dict(color="% Positive"),
    )
    fig_pair.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=max(250, 40*len(pivot_p)+100),
        margin=dict(t=50,b=20,l=10,r=10),
    )
    st.plotly_chart(fig_pair, use_container_width=True)

    # Pattern insight for this pair
    if not agg_p.empty:
        best_r  = agg_p.loc[agg_p["pct"].idxmax()]
        worst_r = agg_p.loc[agg_p["pct"].idxmin()]
        range_v = best_r["pct"] - worst_r["pct"]
        if range_v >= 15:
            msg_cls = "strong-box"
            msg_icon = "-"
        elif range_v >= 5:
            msg_cls = "insight-box"
            msg_icon = "-"
        else:
            msg_cls = "warn-box"
            msg_icon = "--"

        st.markdown(
            f'<div class="{msg_cls}">{msg_icon} '
            f'<b>{fa_label}</b> - <b>{fb_label}</b>: '
            f'Range of {range_v:.1f}pp across groups. '
            f'Best: <b>{best_r[feat_a]}</b> / <b>{best_r[feat_b]}</b> '
            f'({best_r["pct"]}%, n={int(best_r["count"])}). '
            f'Lowest: <b>{worst_r[feat_a]}</b> / <b>{worst_r[feat_b]}</b> '
            f'({worst_r["pct"]}%, n={int(worst_r["count"])}).'
            f'</div>', unsafe_allow_html=True)

st.markdown("---")

# ------------------------------------------------------------------------------
# PART 4 -- SHAP Interaction: top feature pairs from model
# ------------------------------------------------------------------------------
st.markdown("## - Part 4 -- SHAP Interaction Effects")
st.markdown("""
SHAP interaction values measure how much **two features jointly** affect predictions 
beyond their individual effects. Positive = they reinforce each other; 
negative = they counteract.
""")

if Q_SEL in all_shap_results:
    shap_vals_sel, X_m_sel, model_sel, auc_sel = all_shap_results[Q_SEL]

    with st.spinner("Computing SHAP interaction values (this may take ~30s)..."):
        try:
            explainer_int = shap.TreeExplainer(model_sel)
            # Use a sample for speed
            sample_size = min(200, len(X_m_sel))
            X_sample = X_m_sel.sample(sample_size, random_state=42)
            shap_int  = explainer_int.shap_interaction_values(X_sample)

            # Mean absolute interaction per pair
            mean_int = np.abs(shap_int).mean(axis=0)
            np.fill_diagonal(mean_int, 0)  # zero out self-interaction

            int_df = pd.DataFrame(mean_int,
                                  index=X_m_sel.columns,
                                  columns=X_m_sel.columns)

            fig_int = px.imshow(
                int_df.round(4),
                text_auto=".3f",
                color_continuous_scale="RdBu_r",
                title=f"SHAP Interaction Values -- {q_label_sel} "
                      f"(sample n={sample_size})",
                aspect="auto",
                labels=dict(color="Mean |Interaction|"),
            )
            fig_int.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                height=max(350, 40*len(int_df)+80),
                margin=dict(t=60,b=20,l=10,r=10),
            )
            st.plotly_chart(fig_int, use_container_width=True)

            # Top interactions
            int_pairs = []
            for i in range(len(X_m_sel.columns)):
                for j in range(i+1, len(X_m_sel.columns)):
                    int_pairs.append({
                        "Feature A": X_m_sel.columns[i].replace("_"," ").title(),
                        "Feature B": X_m_sel.columns[j].replace("_"," ").title(),
                        "Interaction": round(mean_int[i, j], 4),
                    })
            int_top = pd.DataFrame(int_pairs).sort_values("Interaction", ascending=False).head(5)

            st.markdown("**Top 5 feature interactions:**")
            for _, r in int_top.iterrows():
                st.markdown(
                    f'<div class="insight-box">- <b>{r["Feature A"]}</b> - '
                    f'<b>{r["Feature B"]}</b> -- interaction strength: '
                    f'<b>{r["Interaction"]:.4f}</b>. '
                    f'These two features jointly affect {q_label_sel} '
                    f'beyond their individual contributions.</div>',
                    unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"SHAP interactions could not be computed: {e}")
else:
    st.info("Run the SHAP analysis above first (select the question tab).")

st.markdown("---")

# ------------------------------------------------------------------------------
# PART 5 -- SUMMARY ACROSS ALL QUESTIONS
# ------------------------------------------------------------------------------
st.markdown("## - Part 5 -- Cross-Question Feature Importance Comparison")
st.markdown("Which features matter most, and does it vary across the 4 questions?")

if all_shap_results:
    summary_rows = []
    for q_col, (shap_v, X_m, mdl, auc) in all_shap_results.items():
        mean_abs = np.abs(shap_v).mean(axis=0)
        for feat, imp in zip(X_m.columns, mean_abs):
            summary_rows.append({
                "Question": QUESTIONS[q_col][0],
                "Feature":  feat.replace("_"," ").title(),
                "Mean |SHAP|": round(imp, 4),
            })

    summary_df = pd.DataFrame(summary_rows)

    fig_sum = px.bar(
        summary_df,
        x="Mean |SHAP|", y="Feature",
        color="Question",
        barmode="group",
        orientation="h",
        title="SHAP Feature Importance -- All 4 Questions Compared",
        color_discrete_sequence=px.colors.qualitative.Bold,
        height=max(400, 35*summary_df["Feature"].nunique()+80),
    )
    fig_sum.update_layout(
        plot_bgcolor="rgba(245,248,252,1)", paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=44,b=20,l=10,r=10),
        yaxis=dict(title=""),
        xaxis=dict(title="Mean |SHAP value|"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1),
    )
    st.plotly_chart(fig_sum, use_container_width=True)

    # Which single feature dominates?
    top_overall = (summary_df.groupby("Feature")["Mean |SHAP|"]
                             .mean()
                             .sort_values(ascending=False)
                             .head(5))
    st.markdown("**Overall most impactful features (averaged across all 4 questions):**")
    for feat, imp in top_overall.items():
        st.markdown(
            f'<div class="strong-box">- <b>{feat}</b> -- '
            f'avg SHAP impact = {imp:.4f} across all questions</div>',
            unsafe_allow_html=True)

st.markdown("---")
st.caption(
    "Methods: GradientBoostingClassifier - TreeSHAP - Cramer's V (---based) - "
    "SHAP Interaction Values. "
    "Positive responses defined as score -4/5 (Definitely/Probably/Agree/Excellent)."
)