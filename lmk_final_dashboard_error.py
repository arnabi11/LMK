"""
LMK Impact Dashboard - Streamlit version
Run:  streamlit run lmk_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

import warnings, itertools
warnings.filterwarnings("ignore")

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text, _tree
from sklearn.inspection import permutation_importance
from plotly.subplots import make_subplots
from scipy import stats
import shap

# -- Page config -----------------------------------------------
st.set_page_config(
    page_title="LMK Impact Dashboard",
    page_icon="📊",
    layout="wide",
)

st.markdown("""
<style>
/* -- Sidebar: always dark navy ------------------------------- */
[data-testid="stSidebar"] { background-color: #1E2A3A !important; }
[data-testid="stSidebar"] * { color: #E8EDF2 !important; }
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSelectbox label { color: #A8C4DC !important; font-weight:600; }
[data-testid="stSidebar"] hr { border-color:#3a506b; }

/* -- Main content: explicit font colour for both themes ------ */
.block-container { padding-top:1.5rem; }
.main .block-container { color:#1a1a1a; }
[data-theme="dark"] .main .block-container { color:#e8edf2; }

/* -- Insight / warn / strong boxes: light mode defaults ------ */
.insight-box {
    background:#EAF4FB; border-left:4px solid #3A7DC0;
    color:#1A3A55 !important;
    padding:10px 16px; border-radius:7px; margin:6px 0;
    font-size:clamp(0.78rem,2vw,0.92rem); line-height:1.55;
}
.warn-box {
    background:#FFF8E6; border-left:4px solid #E09B00;
    color:#5A3E00 !important;
    padding:10px 16px; border-radius:7px; margin:6px 0;
    font-size:clamp(0.78rem,2vw,0.92rem); line-height:1.55;
}
.strong-box {
    background:#E8F5EC; border-left:4px solid #2E9E52;
    color:#1A4A28 !important;
    padding:10px 16px; border-radius:7px; margin:6px 0;
    font-size:clamp(0.78rem,2vw,0.92rem); line-height:1.55;
}

/* -- Dark-mode overrides (Streamlit sets data-theme on <html>) */
[data-theme="dark"] .insight-box {
    background:#1C3248 !important; border-left:4px solid #5BA8E0;
    color:#B8D8F5 !important;
}
[data-theme="dark"] .warn-box {
    background:#3A2E10 !important; border-left:4px solid #E0B040;
    color:#F0D080 !important;
}
[data-theme="dark"] .strong-box {
    background:#1A3828 !important; border-left:4px solid #40C070;
    color:#80E0A8 !important;
}

/* -- Interpretation box --------------------------------------- */
.interp-box {
    background:rgba(74,159,212,0.10); border-left:3px solid #4A9FD4;
    color:#1a3a55 !important;
    border-radius:6px; padding:9px 14px; margin:4px 0 14px 0;
    font-size:clamp(0.78rem,2vw,0.91rem); line-height:1.55;
}
[data-theme="dark"] .interp-box {
    background:rgba(74,159,212,0.20) !important;
    color:#c8e4f8 !important;
}

/* -- Responsive layout ---------------------------------------- */
@media (max-width:768px) {
    .block-container { padding-left:0.5rem !important; padding-right:0.5rem !important; }
    .stPlotlyChart { overflow-x:auto !important; }
}
@media (max-width:1024px) {
    .block-container { padding-left:1rem !important; padding-right:1rem !important; }
}
</style>
""", unsafe_allow_html=True)

# -- Colour palette --------------------------------------------
ACCENT  = "#5B8DB8"
GREEN   = "#6BAE95"
ORANGE  = "#E8A87C"
PURPLE  = "#9B72B0"
RED     = "#D96B6B"

# Response orderings and positive sets for each question type
USEFUL_ORDER    = ["Definitely", "Probably", "Not sure", "Not really", "No"]
AGREE_ORDER     = ["Strongly agree", "Agree", "Neither agree nor disagree",
                   "Disagree", "Strongly disagree"]
RATING_ORDER    = ["Excellent", "Good", "OK", "Poor", "Very poor"]

POSITIVE_USEFUL = {"Definitely", "Probably"}
POSITIVE_AGREE  = {"Strongly agree", "Agree"}
POSITIVE_RATING = {"Excellent", "Good"}

USEFUL_COLORS   = [GREEN, "#A8D5C2", "#F5D78E", ORANGE, RED]
AGREE_COLORS    = [GREEN, "#A8D5C2", "#F5D78E", ORANGE, RED]
RATING_COLORS   = [GREEN, "#A8D5C2", "#F5D78E", ORANGE, RED]


# -- Load & cache data -----------------------------------------
@st.cache_data
def load_data():
    # -- UPDATE THESE PATHS to point at your CSVs --------------
    df_s = pd.read_csv("Sessions 23-24 and 24-25.csv")
    df_v = pd.read_csv("Impact surveys 23-24 and 24-25.csv")

    df_s.columns = df_s.columns.str.strip()
    df_v.columns = df_v.columns.str.strip()

    # Sessions renames
    df_s = df_s.rename(columns={
        "Record ID":                                                       "record_id",
        "Module":                                                          "module",
        "Session start date and time":                                     "session_date",
        "Academic year":                                                   "academic_year",
        "Organisation Name":                                               "org_name",
        "Org Borough":                                                     "borough",
        "Confirmed number of expected participants (Youth + Adult)":       "expected_participants",
        "Confirmed number of participants (Youth + Adult)":                "actual_participants",
        "Number of surveys":                                               "num_surveys",
        "Org Type":                                                        "org_type",
        "Org Sub Type":                                                    "org_sub_type",
        "Org % on school meals":                                           "pct_school_meals",
        "Vulnerable group":                                                "vulnerable_group",
    })

    # Survey renames - exact column names from the real CSV
    rename_map = {
        "Session Record ID":   "session_id",
        "Session Name":        "session_name",
        "Record ID":           "record_id",
        # 4 question columns
        "Workshop useful/helpful relationships (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8, AWP)":
            "workshop_useful",
        "Learnt something new about healthy and unhealthy behaviours in relationships (Y10SDD+AP+7/8, YIoP, YSII, AWP)":
            "learnt_healthy_behaviours",
        "Leader rating (Y10SDD+AP+7/8, Y10SInc, YIoP+7/8, YSII+Inc+7/8, ACPD, AWP)":
            "leader_rating",
        "Know who and where to go if worried about relationship (Y10SDD+AP+7/8, Y10SInc+Pri, YDDPri, YIoP+7/8, YSII+Inc+7/8, AWP)":
            "know_where_to_go",
        # Changed understanding (ACPD module uses Agree scale)
        "Changed understanding of healthy and unhealthy behaviours in relationships (ACPD)":
            "changed_understanding",
        # Demographics
        "Age (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":                      "age",
        "Gender (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8, ACPD)":             "gender",
        "Ethnicity (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":                "ethnicity",
        "Disability (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":               "disability",
        "Sexuality (Y10SDD+AP, YIoP, YSII)":                             "sexuality",
        "Neurodivergent (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":           "neurodivergent",
    }
    # Only rename columns that exist
    df_v = df_v.rename(columns={k: v for k, v in rename_map.items() if k in df_v.columns})

    # Normalise blanks / non-answers to NaN
    null_vals = {"Not Answered", "nan", "", "left blank", "(left blank)", "N/A", "n/a"}

    def clean_col(s):
        return s.astype(str).str.strip().apply(
            lambda x: np.nan if x in null_vals else x
        )

    for c in ["workshop_useful", "learnt_healthy_behaviours",
              "leader_rating", "know_where_to_go", "changed_understanding"]:
        if c in df_v.columns:
            df_v[c] = clean_col(df_v[c])

    return df_s, df_v


df_sessions_raw, df_survey_raw = load_data()


# -- Merge session metadata onto survey rows -------------------
def enrich(dfs, dfv):
    meta = dfs[["record_id", "vulnerable_group", "pct_school_meals", "module"]].copy()
    meta = meta.rename(columns={"record_id": "session_id",
                                "module": "session_module"})
    return dfv.merge(meta, on="session_id", how="left", suffixes=("", "_sess"))


# -------------------------------------------------------------
# CHART HELPERS
# -------------------------------------------------------------
# -- Theme-safe chart defaults ----------------------------------
# plot_bgcolor: very light grey - visible on white AND dark canvas
# paper_bgcolor: transparent so it inherits the page background
# font color #333 is readable on light; charts also get template
BASE_LAYOUT = dict(
    plot_bgcolor="rgba(248,249,252,1)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=48, b=44, l=10, r=10),
    height=340,
    font=dict(size=11, color="#333333"),
    autosize=True,
)

def _trunc(labels, n=20):
    """Truncate long tick labels so they don't overlap."""
    return [str(l)[:n] + ("-" if len(str(l)) > n else "") for l in labels]


def pct_positive(series, positive_set):
    valid = series.dropna()
    if len(valid) == 0:
        return None, 0
    pct = round(valid.isin(positive_set).sum() / len(valid) * 100, 1)
    return pct, len(valid)


def donut_chart(series, order, colors, title):
    counts = series.dropna().value_counts()
    labels = [r for r in order if r in counts.index]
    values = [counts[r] for r in labels]
    if not labels:
        return empty_fig("No responses recorded")
    short_labels = _trunc(labels, 18)
    fig = go.Figure(go.Pie(
        labels=short_labels, values=values,
        hole=0.50,
        marker_colors=colors[:len(labels)],
        textinfo="percent",          # label shown in legend, not on slice
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
        sort=False,
    ))
    fig.update_layout(
        **BASE_LAYOUT, title=title,
        showlegend=True,
        legend=dict(orientation="v", font=dict(size=9),
                    yanchor="middle", y=0.5,
                    xanchor="left", x=1.01),
        margin=dict(t=48, b=20, l=10, r=130),  # extra right for legend
    )
    return fig


def stacked_bar(dfv, col, group_col, order, colors, title):
    """100% stacked bar: group on x-axis, responses as colour bands."""
    if col not in dfv.columns or group_col not in dfv.columns:
        return empty_fig(f"Column not found")

    sub = dfv[[col, group_col]].dropna()
    sub = sub[sub[col].isin(order)]
    if sub.empty:
        return empty_fig("No data")

    ct = sub.groupby([group_col, col]).size().reset_index(name="n")
    totals = ct.groupby(group_col)["n"].transform("sum")
    ct["pct"] = (ct["n"] / totals * 100).round(1)

    color_map = dict(zip(order, colors))
    fig = go.Figure()
    for resp in order:
        d = ct[ct[col] == resp]
        if d.empty:
            continue
        fig.add_trace(go.Bar(
            name=resp,
            x=d[group_col],
            y=d["pct"],
            marker_color=color_map.get(resp, "#ccc"),
            text=d["pct"].apply(lambda v: f"{v:.0f}%" if v >= 5 else ""),
            textposition="inside",
            insidetextanchor="middle",
        ))

    # Truncate long x-axis category names
    unique_groups = ct[group_col].unique().tolist()
    tick_text = _trunc(unique_groups, 18)
    fig.update_layout(
        **BASE_LAYOUT,
        title=title,
        barmode="stack",
        yaxis=dict(title="%", range=[0, 100], tickfont=dict(size=10)),
        xaxis=dict(title="", tickangle=-35, tickfont=dict(size=9),
                   tickmode="array",
                   tickvals=unique_groups,
                   ticktext=tick_text),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=9)),
        margin=dict(t=48, b=80, l=10, r=10),
    )
    return fig


def pct_positive_bar(dfv, col, group_col, positive_set, color, title):
    """Horizontal bar: % positive per group."""
    if col not in dfv.columns or group_col not in dfv.columns:
        return empty_fig("Column not found")

    sub = dfv[[col, group_col]].dropna()
    if sub.empty:
        return empty_fig("No data")

    agg = (sub.groupby(group_col)
              .apply(lambda g: round(g[col].isin(positive_set).sum() / len(g) * 100, 1))
              .reset_index(name="% Positive"))
    agg["n"] = sub.groupby(group_col).size().values
    agg = agg.sort_values("% Positive", ascending=True)

    agg["label"] = agg.apply(lambda r: f"{r['% Positive']}%", axis=1)
    agg["ylabel"] = _trunc(agg[group_col].tolist(), 22)
    fig = px.bar(agg, x="% Positive", y="ylabel", orientation="h",
                 title=title, color_discrete_sequence=[color],
                 range_x=[0, 115],
                 text="label",
                 hover_data={group_col: True, "n": True,
                             "ylabel": False, "label": False})
    fig.update_traces(
        textposition="outside",
        textfont=dict(size=9),
        marker_line_width=0,
        cliponaxis=False,
    )
    fig.update_layout(
        **BASE_LAYOUT,
        xaxis_title="% Positive Response",
        yaxis_title="",
        yaxis=dict(tickfont=dict(size=9)),
        xaxis=dict(tickfont=dict(size=10)),
        margin=dict(t=48, b=44, l=10, r=60),
    )
    return fig


def fsm_bar(dfs, dfv, col, positive_set, color, title):
    """% positive bucketed by % free school meals (session-level)."""
    if col not in dfv.columns or "pct_school_meals" not in dfs.columns:
        return empty_fig("Free school meals data not available")

    fsm_map = pd.to_numeric(
        dfs.set_index("record_id")["pct_school_meals"], errors="coerce"
    ).to_dict()
    dfv2 = dfv.copy()
    dfv2["_fsm"] = dfv2["session_id"].map(fsm_map)
    dfv2 = dfv2.dropna(subset=["_fsm", col])
    if dfv2.empty:
        return empty_fig("No FSM data linked to these responses")

    bins   = [0, 20, 40, 60, 80, 101]
    labels = ["0-20%", "21-40%", "41-60%", "61-80%", "81-100%"]
    dfv2["fsm_bucket"] = pd.cut(dfv2["_fsm"], bins=bins, labels=labels, right=False)
    return pct_positive_bar(dfv2, col, "fsm_bucket", positive_set, color, title)


def empty_fig(msg="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#888888"))
    fig.update_layout(**BASE_LAYOUT)
    return fig


# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("## 📊 LMK Dashboard")
    st.markdown("---")
    st.markdown("### 🔽 Filters")
    st.caption("All filters apply across every tab.")

    def ms(label, col, df, key):
        opts = sorted([str(o) for o in df[col].dropna().unique()
                       if str(o).strip() not in ("", "nan", "Not Answered")]) \
               if col in df.columns else []
        return st.multiselect(label, opts, key=key)

    st.markdown("**Session-level**")
    sel_borough  = ms("Borough",          "borough",          df_sessions_raw, "borough")
    sel_year     = ms("Academic Year",    "academic_year",    df_sessions_raw, "year")
    sel_module   = ms("Module",           "module",           df_sessions_raw, "module")
    sel_orgtype  = ms("Org Type",         "org_type",         df_sessions_raw, "orgtype")
    sel_vgroup   = ms("Vulnerable Group", "vulnerable_group", df_sessions_raw, "vgroup")

    sel_fsm = None
    if "pct_school_meals" in df_sessions_raw.columns:
        fsm_num = pd.to_numeric(df_sessions_raw["pct_school_meals"], errors="coerce").dropna()
        if not fsm_num.empty:
            sel_fsm = st.slider(
                "% Eligible for Free School Meals",
                int(fsm_num.min()), int(fsm_num.max()),
                (int(fsm_num.min()), int(fsm_num.max())), key="fsm"
            )

    st.markdown("**Pupil-level**")
    sel_gender    = ms("Gender",                              "gender",        df_survey_raw, "gender")
    sel_ethnicity = ms("Ethnicity",                           "ethnicity",     df_survey_raw, "ethnicity")
    sel_age       = ms("Age",                                 "age",           df_survey_raw, "age")
    sel_disability= ms("Disability",                          "disability",    df_survey_raw, "disability")
    sel_sexuality = ms("Sexuality",                           "sexuality",     df_survey_raw, "sexuality")
    sel_nd        = ms("Neurodivergent / Learning Difficulty","neurodivergent",df_survey_raw, "nd")

    st.markdown("---")
    if st.button("🔄 Reset All Filters", use_container_width=True):
        for k in ["borough","year","module","orgtype","vgroup",
                  "gender","ethnicity","age","disability","sexuality","nd"]:
            st.session_state[k] = []
        st.rerun()


# -------------------------------------------------------------
# APPLY FILTERS  (cached so repeated identical selections
#                 return instantly without recomputing)
# -------------------------------------------------------------
@st.cache_data(show_spinner=False)
def apply_filters(
    boroughs, years, modules, orgtypes, vgroups, fsm_range,
    genders, ethnicities, ages, disabilities, sexualities, nds,
):
    """All heavy pandas work done once per unique filter combination."""
    dfs = df_sessions_raw.copy()
    if boroughs:  dfs = dfs[dfs["borough"].isin(boroughs)]
    if years:     dfs = dfs[dfs["academic_year"].isin(years)]
    if modules:   dfs = dfs[dfs["module"].isin(modules)]
    if orgtypes and "org_type" in dfs.columns:
        dfs = dfs[dfs["org_type"].isin(orgtypes)]
    if vgroups and "vulnerable_group" in dfs.columns:
        dfs = dfs[dfs["vulnerable_group"].isin(vgroups)]
    if fsm_range is not None and "pct_school_meals" in dfs.columns:
        _fsm = pd.to_numeric(dfs["pct_school_meals"], errors="coerce")
        dfs = dfs[_fsm.between(fsm_range[0], fsm_range[1]) | _fsm.isna()]

    dfv = df_survey_raw[df_survey_raw["session_id"].isin(dfs["record_id"])].copy()
    dfv = enrich(dfs, dfv)

    if genders      and "gender"        in dfv.columns: dfv = dfv[dfv["gender"].isin(genders)]
    if ethnicities  and "ethnicity"     in dfv.columns: dfv = dfv[dfv["ethnicity"].isin(ethnicities)]
    if ages         and "age"           in dfv.columns: dfv = dfv[dfv["age"].isin(ages)]
    if disabilities and "disability"    in dfv.columns: dfv = dfv[dfv["disability"].isin(disabilities)]
    if sexualities  and "sexuality"     in dfv.columns: dfv = dfv[dfv["sexuality"].isin(sexualities)]
    if nds          and "neurodivergent" in dfv.columns: dfv = dfv[dfv["neurodivergent"].isin(nds)]

    return dfs, dfv

# Hashable tuples for cache key (lists are not hashable)
dfs, dfv = apply_filters(
    tuple(sel_borough),
    tuple(sel_year),
    tuple(sel_module),
    tuple(sel_orgtype),
    tuple(sel_vgroup),
    tuple(sel_fsm) if sel_fsm is not None else None,
    tuple(sel_gender),
    tuple(sel_ethnicity),
    tuple(sel_age),
    tuple(sel_disability),
    tuple(sel_sexuality),
    tuple(sel_nd),
)


# -------------------------------------------------------------
# HEADER
# -------------------------------------------------------------
st.markdown("# 📊 LMK Impact Dashboard")

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("Sessions", f"{len(dfs):,}")
with k2:
    ap = int(dfs["actual_participants"].sum()) if "actual_participants" in dfs.columns else 0
    st.metric("Confirmed Participants", f"{ap:,}")
with k3:
    st.metric("Survey Responses", f"{len(dfv):,}")
with k4:
    pct_u, n_u = pct_positive(dfv.get("workshop_useful", pd.Series()), POSITIVE_USEFUL)
    st.metric("Found Workshop Useful",
              f"{pct_u}%" if pct_u is not None else "N/A",
              help="% answering Definitely or Probably")

st.markdown("---")


# -------------------------------------------------------------
# REUSABLE TAB BODY
# -------------------------------------------------------------
def render_tab(col, q_title, order, positive_set, colors, pos_label, note=None):
    st.subheader(q_title)
    if note:
        st.info(note)

    series = dfv[col] if col in dfv.columns else pd.Series([], dtype=str)
    pct, n  = pct_positive(series, positive_set)
    top_val = series.dropna().value_counts()
    top_r   = top_val.idxmax() if not top_val.empty else "N/A"
    top_pct = round(top_val.iloc[0] / n * 100, 1) if n > 0 else 0

    # KPI strip
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(pos_label, f"{pct}%" if pct is not None else "N/A",
                  help=f"n = {n:,}")
    with m2:
        st.metric("Most Common Response", f"{top_r} ({top_pct}%)")
    with m3:
        st.metric("Responses for this question", f"{n:,}")

    st.markdown("---")

    # Row 1 - Overall donut | by Gender (stacked)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            donut_chart(series, order, colors, "Overall Response Breakdown"),
            use_container_width=True)
    with c2:
        st.plotly_chart(
            stacked_bar(dfv, col, "gender", order, colors, "Response by Gender"),
            use_container_width=True)

    # Row 2 - % positive by Ethnicity | by Age (stacked)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            pct_positive_bar(dfv, col, "ethnicity", positive_set, ACCENT,
                             "% Positive by Ethnicity"),
            use_container_width=True)
    with c4:
        st.plotly_chart(
            stacked_bar(dfv, col, "age", order, colors, "Response by Age"),
            use_container_width=True)

    # Row 3 - Disability | Neurodivergent / Learning Difficulty
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(
            stacked_bar(dfv, col, "disability", order, colors,
                        "Response by Disability"),
            use_container_width=True)
    with c6:
        st.plotly_chart(
            stacked_bar(dfv, col, "neurodivergent", order, colors,
                        "Response by Neurodivergent / Learning Difficulty"),
            use_container_width=True)

    # Row 4 - Sexuality | Vulnerable group
    c7, c8 = st.columns(2)
    with c7:
        st.plotly_chart(
            stacked_bar(dfv, col, "sexuality", order, colors,
                        "Response by Sexuality"),
            use_container_width=True)
    with c8:
        vg = "vulnerable_group" if "vulnerable_group" in dfv.columns else "vulnerable_group_sess"
        st.plotly_chart(
            stacked_bar(dfv, col, vg, order, colors,
                        "Response by Vulnerable / Not Vulnerable Group"),
            use_container_width=True)

    # Row 5 - % Free School Meals (full width)
    st.plotly_chart(
        fsm_bar(dfs, dfv, col, positive_set, GREEN,
                "% Positive by Free School Meals Eligibility (session-level)"),
        use_container_width=True)


# ---------------------------------------------------------------
# PATTERN ANALYSIS TAB  (SHAP + ML pattern discovery)
# ---------------------------------------------------------------
def render_pattern_analysis_tab():
    """Self-contained SHAP / pattern-discovery analysis (Parts 1-6).
    Loads and filters its own copy of the data via the widgets this
    function adds to the sidebar, independent of the main dashboard's
    filters above.
    """
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

    def chart_layout(height=340, extra=None):
        """Theme-safe Plotly layout: transparent bg, neutral font, responsive."""
        base = dict(
            plot_bgcolor="rgba(248,249,252,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333333", size=11),
            margin=dict(t=48, b=48, l=10, r=10),
            height=height,
            autosize=True,
        )
        if extra:
            base.update(extra)
        return base

    def _trunc_p(labels, n=20):
        return [str(l)[:n] + ("-" if len(str(l)) > n else "") for l in labels]

    # -- Dynamic interpretation helper (uses CSS class for theme support) ----------
    def interpret(lines):
        """Render exactly 2 auto-generated interpretation lines below a chart."""
        assert len(lines) == 2, "Always pass exactly 2 interpretation lines"
        st.markdown(
            f'<div class="interp-box"><b>Interpretation:</b><br>{lines[0]}<br>{lines[1]}</div>',
            unsafe_allow_html=True,
        )

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

    # -- Filter (cached per unique filter combo for <1s repeats) -------------------
    @st.cache_data(show_spinner=False)
    def _apply_pat_filter(df_json, modules, years, boroughs):
        import io
        df = pd.read_json(io.StringIO(df_json))
        if modules:  df = df[df["module"].isin(list(modules))]
        if years:    df = df[df["academic_year"].isin(list(years))]
        if boroughs: df = df[df["borough"].isin(list(boroughs))]
        return df

    _df_raw_json = df_raw.to_json()
    df = _apply_pat_filter(
        _df_raw_json,
        tuple(sel_module),
        tuple(sel_year),
        tuple(sel_borough),
    )

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
                plot_bgcolor="rgba(248,249,252,1)", paper_bgcolor="rgba(0,0,0,0)",
                height=max(300, 28*len(feat_imp)+60),
                margin=dict(t=44,b=20,l=10,r=60),
                coloraxis_showscale=False,
                yaxis=dict(title=""),
                xaxis=dict(title="Mean |SHAP value| - more impact"),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            _top1 = feat_imp.sort_values("Mean |SHAP|", ascending=False).iloc[0]
            _top2b = feat_imp.sort_values("Mean |SHAP|", ascending=False).iloc[1] if len(feat_imp)>1 else _top1
            _bot1 = feat_imp.sort_values("Mean |SHAP|", ascending=True).iloc[0]
            interpret([
                f"<b>{_top1['Feature'].replace('_',' ').title()}</b> is the single strongest "
                f"driver of <b>{q_label}</b> (mean |SHAP| = {_top1['Mean |SHAP|']:.3f}), "
                f"followed by <b>{_top2b['Feature'].replace('_',' ').title()}</b> "
                f"({_top2b['Mean |SHAP|']:.3f}) - these two features together account for the "
                f"most variation in how respondents answered this question.",
                f"<b>{_bot1['Feature'].replace('_',' ').title()}</b> has the lowest impact "
                f"({_bot1['Mean |SHAP|']:.3f}), suggesting it contributes little to predicting "
                f"whether someone gives a positive response to this question.",
            ])

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
                plot_bgcolor="rgba(248,249,252,1)", paper_bgcolor="rgba(0,0,0,0)",
                height=max(350, 30*len(feat_order)+60),
                margin=dict(t=44, b=20, l=10, r=20),
                showlegend=False,
            )
            st.plotly_chart(fig_bee, use_container_width=True)
            _top_bee = feat_imp.sort_values("Mean |SHAP|", ascending=False).iloc[0]
            _bee_idx = list(X_m.columns).index(_top_bee["Feature"])
            _bee_dir = "pushes predictions toward POSITIVE" if shap_vals[:, _bee_idx].mean() > 0 else "pushes predictions toward NEGATIVE"
            _spread = shap_vals[:, _bee_idx].max() - shap_vals[:, _bee_idx].min()
            interpret([
                f"Dots spread far to the <b>right</b> (positive SHAP) indicate feature values "
                f"that increase the likelihood of a positive response; dots to the <b>left</b> "
                f"reduce it. <b>{_top_bee['Feature'].replace('_',' ').title()}</b> shows the "
                f"widest spread ({_spread:.3f}), meaning its values create the largest swing "
                f"in predicted outcome.",
                f"The colour gradient (red=high value, blue=low value) on "
                f"<b>{_top_bee['Feature'].replace('_',' ').title()}</b> reveals that "
                f"higher values of this feature generally <b>{_bee_dir}</b> - "
                f"look for features where red dots cluster on the right and blue on the left "
                f"(or vice versa) to identify the strongest directional effects.",
            ])

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
                        plot_bgcolor="rgba(248,249,252,1)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#444444"),
                        height=300,
                        margin=dict(t=44,b=20,l=10,r=10),
                        showlegend=False,
                    )
                    st.plotly_chart(fig_dep, use_container_width=True)
                    _dep_corr = float(np.corrcoef(dep_df[feat].values, dep_df["SHAP"].values)[0,1])
                    _dep_dir  = "positive" if _dep_corr > 0.05 else ("negative" if _dep_corr < -0.05 else "flat")
                    _dep_word = ("as values increase, positive response becomes more likely"
                                 if _dep_corr > 0.05 else
                                 ("as values increase, positive response becomes less likely"
                                  if _dep_corr < -0.05 else
                                  "no clear linear trend - the effect is non-linear or negligible"))
                    interpret([
                        f"The trend line shows a <b>{_dep_dir} relationship</b> "
                        f"(correlation = {_dep_corr:.2f}): {_dep_word} for "
                        f"<b>{feat.replace('_',' ').title()}</b>.",
                        f"Scattered dots above the zero SHAP line represent respondents where this "
                        f"feature <b>increased</b> their predicted positive response; dots below "
                        f"represent those where it <b>decreased</b> it - wide vertical scatter at "
                        f"a given x-value signals that other features also play a role.",
                    ])

    st.markdown("---")

    # ------------------------------------------------------------------------------
    # PART 2 -- ALL-PAIR PATTERN / INTERACTION ANALYSIS
    # ------------------------------------------------------------------------------
    st.markdown("## - Part 2 -- Pairwise Pattern Detection Across ALL Feature Combinations")
    st.markdown("""
    We test **every pair of categorical features** for statistical association using 
    **Cramer's V** (ranges 0-1: 0=no association, 1=perfect). We also check specific 
    combinations you asked about: Age - Borough - Gender, and extend 
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
            font=dict(color="#333333", size=10),
            autosize=True,
            height=max(420, 42*len(feat_labels)+80),
            margin=dict(t=60, b=70, l=10, r=10),
            coloraxis_colorbar=dict(title="Cramer's V", tickfont=dict(size=9)),
            xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_heat, use_container_width=True)
        _cv_max_row = pair_df.iloc[0]
        _cv_min_row = pair_df.iloc[-1]
        _n_strong   = (pair_df["Cramer's V"] >= 0.2).sum()
        _n_weak     = (pair_df["Cramer's V"] <  0.1).sum()
        _cv_col_name = "Cramer's V"
        _cv_val_str  = f"{_cv_max_row[_cv_col_name]:.3f}"
        _cv_feat_a   = _cv_max_row['Feature A']
        _cv_feat_b   = _cv_max_row['Feature B']
        _cv_summary  = ('most features move independently'
                        if _n_weak > _n_strong
                        else 'several feature combinations carry shared information')
        interpret([
            (f"The darkest cell in the heatmap is "
             f"<b>{_cv_feat_a}</b> x <b>{_cv_feat_b}</b> "
             f"(Cramer's V = {_cv_val_str}), indicating the strongest "
             f"co-movement between any two features - knowing one of these values helps "
             f"predict the other across the dataset."),
            (f"There are <b>{_n_strong} strongly associated pairs</b> (V >= 0.2) and "
             f"<b>{_n_weak} weakly associated pairs</b> (V < 0.1) out of "
             f"{len(pair_df)} total pairs tested - {_cv_summary}."),
        ])

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
    # PART 3 -- SPECIFIC COMBINATIONS: AGE - BOROUGH - GENDER (+ all triples)
    # ------------------------------------------------------------------------------
    st.markdown("## - Part 3 -- Specific Patterns: Age - Borough - Gender (and more)")
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

    # -- A) The asked combination: Age - Borough - Gender -------------------------
    st.markdown(f"### A) Age - Borough - Gender - *{q_label_sel}*")

    age_ok     = "age"     in df.columns
    borough_ok = "borough" in df.columns
    gender_ok  = "gender"  in df.columns

    if age_ok and borough_ok and gender_ok:
        sub3 = df[["age","borough","gender","_target_pos"]].dropna()
        sub3 = sub3[~sub3["age"].astype(str).isin(NULL_VALS)]
        sub3 = sub3[~sub3["gender"].astype(str).isin(NULL_VALS)]
        sub3 = sub3[~sub3["borough"].astype(str).isin(NULL_VALS)]

        agg3 = (sub3.groupby(["age","borough","gender"])["_target_pos"]
                   .agg(["mean","count"]).reset_index())
        agg3 = agg3[agg3["count"] >= min_support]
        agg3["% Positive"] = (agg3["mean"]*100).round(1)
        agg3.columns = ["Age","Borough","Gender","mean","n","% Positive"]

        if not agg3.empty:
            # Faceted heatmap: Age on rows, Borough on columns, Gender as tab
            genders = agg3["Gender"].unique()
            gender_tabs = st.tabs([f"Gender: {g}" for g in sorted(genders)])
            for gtab, gval in zip(gender_tabs, sorted(genders)):
                with gtab:
                    sub_g = agg3[agg3["Gender"]==gval]
                    if sub_g.empty:
                        st.info("Not enough data for this gender with current filters.")
                        continue
                    pivot = sub_g.pivot(index="Age", columns="Borough",
                                        values="% Positive")
                    fig_3 = px.imshow(
                        pivot, text_auto=True,
                        color_continuous_scale="RdYlGn",
                        zmin=0, zmax=100,
                        title=f"% Positive ({q_label_sel}) -- Gender: {gval}",
                        aspect="auto",
                        labels=dict(color="% Positive"),
                    )
                    fig_3.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)",
                        font=dict(color="#333333", size=10),
                        autosize=True,
                        height=max(280, 54*len(pivot)+100),
                        margin=dict(t=50, b=80, l=10, r=10),
                        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
                        yaxis=dict(tickfont=dict(size=9)),
                    )
                    st.plotly_chart(fig_3, use_container_width=True)

                    # Best and worst cells
                    best  = sub_g.loc[sub_g["% Positive"].idxmax()]
                    worst = sub_g.loc[sub_g["% Positive"].idxmin()]
                    _range3  = best["% Positive"] - worst["% Positive"]
                    _n_cells = len(sub_g)
                    interpret([
                        f"For <b>{gval}</b> respondents, Age <b>{best['Age']}</b> in "
                        f"<b>{best['Borough']}</b> produces the highest positive response rate "
                        f"({best['% Positive']}%, n={int(best['n'])}), while Age "
                        f"<b>{worst['Age']}</b> in <b>{worst['Borough']}</b> "
                        f"produces the lowest ({worst['% Positive']}%, n={int(worst['n'])}).",
                        f"The {_range3:.1f} percentage-point gap across {_n_cells} Age-Borough "
                        f"combinations suggests that "
                        f"{'age and geography together significantly shape outcomes for this group' if _range3 >= 15 else 'the combined effect of age and borough is modest for this group - other factors may be stronger drivers'}.",
                    ])
                    st.markdown(
                        f'<div class="strong-box">- <b>Highest</b>: '
                        f'Age {best["Age"]} / {best["Borough"]} - '
                        f'{best["% Positive"]}% positive (n={int(best["n"])})</div>',
                        unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="warn-box">-- <b>Lowest</b>: '
                        f'Age {worst["Age"]} / {worst["Borough"]} - '
                        f'{worst["% Positive"]}% positive (n={int(worst["n"])})</div>',
                        unsafe_allow_html=True)
        else:
            st.info("Not enough data cells meeting the minimum support threshold. "
                    "Try lowering 'Min. responses per cell' in the sidebar.")
    else:
        missing = [c for c, ok in [("age",age_ok),("borough",borough_ok),
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
            font=dict(color="#333333", size=10),
            autosize=True,
            height=max(280, 46*len(pivot_p)+100),
            margin=dict(t=50, b=80, l=10, r=10),
            xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
            yaxis=dict(tickfont=dict(size=9)),
        )
        st.plotly_chart(fig_pair, use_container_width=True)
        _pair_best  = agg_p.loc[agg_p["pct"].idxmax()]
        _pair_worst = agg_p.loc[agg_p["pct"].idxmin()]
        _pair_range = _pair_best["pct"] - _pair_worst["pct"]
        _pair_cv_val= cv_row["Cramer's V"].values[0] if not cv_row.empty else 0
        interpret([
            f"The highest-performing combination is "
            f"<b>{_pair_best[feat_a]}</b> + <b>{_pair_best[feat_b]}</b> "
            f"({_pair_best['pct']}% positive, n={int(_pair_best['count'])}) and the lowest is "
            f"<b>{_pair_worst[feat_a]}</b> + <b>{_pair_worst[feat_b]}</b> "
            f"({_pair_worst['pct']}%, n={int(_pair_worst['count'])}) - "
            f"a gap of {_pair_range:.1f} percentage points.",
            f"The Cramer's V of {_pair_cv_val:.3f} between these two features indicates a "
            f"'{'strong' if _pair_cv_val>=0.2 else ('moderate' if _pair_cv_val>=0.1 else 'weak')}' "
            f"statistical association - "
            f"{'the two features genuinely co-vary and this pattern is unlikely to be random' if _pair_cv_val>=0.15 else 'the two features are largely independent, so the outcome difference is driven by each feature individually rather than their combination'}.",
        ])

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
                _int_flat = mean_int.copy()
                np.fill_diagonal(_int_flat, 0)
                _int_max_idx = np.unravel_index(_int_flat.argmax(), _int_flat.shape)
                _int_feat_a  = X_m_sel.columns[_int_max_idx[0]].replace("_"," ").title()
                _int_feat_b  = X_m_sel.columns[_int_max_idx[1]].replace("_"," ").title()
                _int_max_val = _int_flat[_int_max_idx]
                _int_avg     = _int_flat[_int_flat > 0].mean()
                interpret([
                    f"The strongest interaction is between <b>{_int_feat_a}</b> and "
                    f"<b>{_int_feat_b}</b> (interaction = {_int_max_val:.4f}), meaning these "
                    f"two features jointly influence <b>{q_label_sel}</b> beyond what either "
                    f"contributes alone - the combined effect is greater than the sum of parts.",
                    f"The average non-zero interaction strength across all pairs is "
                    f"{_int_avg:.4f}. "
                    f"{'High average interaction suggests the outcome is driven by feature combinations, not single features in isolation.' if _int_avg > 0.005 else 'Low average interaction suggests most features act independently - the outcome is largely driven by individual features rather than combinations.'}",
                ])

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
            plot_bgcolor="rgba(248,249,252,1)", paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333333", size=11),
            autosize=True,
            margin=dict(t=56, b=40, l=10, r=10),
            yaxis=dict(title="", tickfont=dict(size=10), categoryorder="total ascending"),
            xaxis=dict(title="Mean |SHAP value|", tickfont=dict(size=10)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02,
                        xanchor="right", x=1, font=dict(size=10)),
        )
        st.plotly_chart(fig_sum, use_container_width=True)
        _top_overall = (summary_df.groupby("Feature")["Mean |SHAP|"]
                                  .mean().sort_values(ascending=False))
        _top1_all = _top_overall.index[0]
        _top1_val = _top_overall.iloc[0]
        _consistent = summary_df.groupby("Feature")["Mean |SHAP|"].std().idxmin()
        interpret([
            f"<b>{_top1_all}</b> is the most impactful feature on average across all "
            f"4 questions (avg SHAP = {_top1_val:.4f}), making it the most universal "
            f"driver of survey outcomes - changes in this feature shift predictions "
            f"regardless of which question is being predicted.",
            f"<b>{_consistent.replace('_',' ').title()}</b> has the most consistent "
            f"importance across all 4 questions (lowest variation), meaning its influence "
            f"is stable and not question-specific - "
            f"features with high variation in importance are question-specific drivers.",
        ])

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
    # ==============================================================================
    # PART 6 -- ML-DRIVEN KEY PATTERN PLOTS
    # What columns co-occur to determine each outcome score
    # ==============================================================================
    st.markdown("---")
    st.markdown("## 7 - ML-Driven Key Pattern Discovery")
    st.markdown("""
    Three complementary ML techniques reveal **which columns co-occur to determine outcomes**:

    1. **Decision Tree rules** -- explicit if/then paths showing exact column combinations
    2. **Random Forest co-importance** -- which feature pairs jointly drive predictions
    3. **Sankey flow diagram** -- visualises the path from demographics to outcome
    4. **Bubble co-occurrence chart** -- size = how often a combination appears, colour = outcome rate
    """)

    from sklearn.tree import DecisionTreeClassifier, export_text
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.inspection import permutation_importance

    # ---- helper: decode a label-encoded column back to readable text --------------
    def decode_col(df, col, encoders):
        if col not in df.columns:
            return df[col] if col in df.columns else pd.Series(["?"] * len(df))
        if col in encoders:
            s = df[col].copy()
            valid = s.notna()
            out = s.astype(object)
            try:
                out[valid] = encoders[col].inverse_transform(
                    s[valid].astype(int).values)
            except Exception:
                pass
            return out.fillna("Unknown")
        return df[col].fillna("Unknown")

    # ---- Question selector: only show questions with enough data -----------------

    def build_target6(df, q_col, q_map, q_pos):
        def sm(x):
            x = str(x).strip()
            return np.nan if x in NULL_VALS else q_map.get(x, np.nan)
        num = (df[q_col].apply(sm)
               if q_col in df.columns
               else pd.Series(np.nan, index=df.index))
        binary = num.apply(
            lambda v: 1 if (not pd.isna(v) and int(v) in q_pos)
                      else (0 if not pd.isna(v) else np.nan))
        return binary

    # Work out which questions have enough data BEFORE showing the selector
    _available_qs = {}
    for _qk, (_ql, _qm, _qp) in QUESTIONS.items():
        _tgt = build_target6(df, _qk, _qm, _qp)
        _n   = _tgt.dropna()
        if len(_n) >= 30 and _n.nunique() >= 2:
            _available_qs[_qk] = _ql

    if not _available_qs:
        st.warning(
            "No survey questions have enough data (>=30 responses) "
            "for the selected module/filter combination. "
            "Please select a different module or clear the filter."
        )
        return

    st.markdown("### Select outcome question to analyse patterns for:")
    st.caption(
        f"Only questions with data in the current filter are shown "
        f"({len(_available_qs)} of {len(QUESTIONS)} available)."
    )
    q_sel6 = st.selectbox(
        "Question",
        list(_available_qs.keys()),
        format_func=lambda k: _available_qs[k],
        key="q_sel6",
    )
    q_label6, q_map6, q_pos6 = QUESTIONS[q_sel6]

    df["_y6"]  = build_target6(df, q_sel6, q_map6, q_pos6)
    valid_idx6 = df["_y6"].dropna().index
    _med       = X_all.median(numeric_only=True)
    X6  = X_all.loc[valid_idx6].fillna(_med)
    y6  = df.loc[valid_idx6, "_y6"].astype(int)

    # Final safety check (shouldn't trigger given filter above, but belt-and-braces)
    if len(y6) < 30 or y6.nunique() < 2:
        st.warning(
            f"Not enough valid responses for **{q_label6}** "
            f"with the current module filter (found {len(y6)} rows). "
            "Try selecting a different question or clearing the Module filter."
        )
        return

    # ==============================================================================
    # PLOT 1 -- Decision Tree: explicit co-occurrence rules
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 1 - Decision Tree: Column Co-occurrence Rules")
    st.markdown("""
    A shallow Decision Tree finds the **exact combinations of column values** that
    together produce positive or negative outcomes. Each leaf = a group defined by
    multiple columns co-occurring.
    """)

    max_depth_dt = st.slider("Tree depth (deeper = more specific rules)", 2, 5, 3, key="dt_depth")

    @st.cache_data
    def run_decision_tree(X_bytes, y_bytes, shape_X, max_depth):
        X = np.frombuffer(X_bytes, dtype=np.float64).reshape(shape_X)
        y = np.frombuffer(y_bytes, dtype=np.float64).astype(int)
        dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=10,
                                     random_state=42, class_weight="balanced")
        dt.fit(X, y)
        acc = dt.score(X, y)
        return dt, acc

    try:
        dt_model, dt_acc = run_decision_tree(
            X6.values.tobytes(), y6.values.astype(float).tobytes(),
            X6.shape, max_depth_dt)
        _dt_ok = True
    except Exception as _dt_err:
        st.warning(f"Decision Tree could not be trained: {_dt_err}")
        _dt_ok = False

    if not _dt_ok:
        st.info("Skipping Decision Tree rules for this question/filter combination.")
        dt_model = None
        dt_acc   = 0.0

    # Extract all leaf paths as rules
    from sklearn.tree import _tree

    def extract_rules(tree, feature_names, encoders_map, class_names=["Negative","Positive"]):
        tree_ = tree.tree_
        fn    = feature_names

        rules = []

        def recurse(node, depth, conditions):
            if tree_.feature[node] == _tree.TREE_UNDEFINED:
                # Leaf node
                val      = tree_.value[node][0]
                total    = int(val.sum())
                pos_pct  = round(val[1] / total * 100, 1) if total > 0 else 0
                pred     = "POSITIVE" if val[1] > val[0] else "NEGATIVE"
                rules.append({
                    "conditions": list(conditions),
                    "n": total,
                    "pos_pct": pos_pct,
                    "prediction": pred,
                    "depth": depth,
                })
                return

            feat      = fn[tree_.feature[node]]
            threshold = tree_.threshold[node]

            # Left branch: feature <= threshold
            recurse(tree_.children_left[node], depth+1,
                    conditions + [f"{feat} <= {threshold:.2f}"])
            # Right branch: feature > threshold
            recurse(tree_.children_right[node], depth+1,
                    conditions + [f"{feat} > {threshold:.2f}"])

        recurse(0, 0, [])
        return rules

    if dt_model is None:
        raw_rules = []
    else:
        raw_rules = extract_rules(dt_model, list(X6.columns), encoders)

    # Convert threshold rules back to readable labels where possible
    def readable_rule(conditions, encoders):
        readable = []
        for cond in conditions:
            # parse "feature <= X.XX" or "feature > X.XX"
            for op in [" <= ", " > "]:
                if op in cond:
                    feat, thresh_str = cond.split(op)
                    thresh = float(thresh_str)
                    feat   = feat.strip()
                    if feat in encoders:
                        le     = encoders[feat]
                        # map threshold to nearest class label
                        idx    = min(int(round(thresh)), len(le.classes_)-1)
                        idx    = max(0, idx)
                        label  = le.classes_[idx]
                        direction = "is" if op.strip() == "<=" else "is not"
                        readable.append(f"{feat.replace('_',' ').title()} {direction} '{label}'")
                    else:
                        readable.append(
                            f"{feat.replace('_',' ').title()} {'<=' if op.strip()=='<=' else '>'} {thresh:.1f}")
                    break
        return " AND ".join(readable) if readable else "All"

    rules_df = pd.DataFrame([{
        "Rule (column co-occurrence)": readable_rule(r["conditions"], encoders),
        "n (respondents)": r["n"],
        "% Positive": r["pos_pct"],
        "Prediction": r["prediction"],
        "Columns involved": len(r["conditions"]),
    } for r in raw_rules])

    if rules_df.empty:
        st.info(
            f"No decision tree rules could be extracted for **{q_label6}** "
            "with the current module filter. "
            "This question may not have enough responses in the selected module."
        )
    else:
        rules_df = rules_df.sort_values("% Positive", ascending=False)

        # Colour code by prediction
        def style_pred(val):
            if val == "POSITIVE": return "background-color:#D4EDDA;color:#155724"
            return "background-color:#F8D7DA;color:#721c24"

        st.markdown(f"**Model accuracy on training data: {dt_acc*100:.1f}%** | "
                f"**{len(rules_df)} leaf rules extracted**")

        st.dataframe(
        rules_df.style.applymap(style_pred, subset=["Prediction"])
                      .background_gradient(subset=["% Positive"], cmap="RdYlGn",
                                           vmin=0, vmax=100),
        use_container_width=True, height=420,
        )

        # Top positive and negative rules as insight boxes
        top_pos = rules_df[rules_df["Prediction"]=="POSITIVE"].head(3)
        top_neg = rules_df[rules_df["Prediction"]=="NEGATIVE"].tail(3)

        st.markdown("#### Top positive patterns (columns that co-occur for HIGH scores)")
        for _, r in top_pos.iterrows():
            st.markdown(
            f'<div class="strong-box">- <b>{r["% Positive"]}% positive</b> '
            f'(n={r["n (respondents)"]:,}) when: {r["Rule (column co-occurrence)"]}</div>',
            unsafe_allow_html=True)

        st.markdown("#### Top negative patterns (columns that co-occur for LOW scores)")
        for _, r in top_neg.iterrows():
            st.markdown(
            f'<div class="warn-box">-- <b>{r["% Positive"]}% positive</b> '
            f'(n={r["n (respondents)"]:,}) when: {r["Rule (column co-occurrence)"]}</div>',
            unsafe_allow_html=True)

    # ==============================================================================
    # PLOT 2 -- Random Forest: feature co-importance matrix
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 2 - Random Forest: Feature Co-importance Matrix")
    st.markdown("""
    A Random Forest is trained and we measure how much **each pair of features
    together** reduces prediction error (joint Gini importance). Darker cells =
    that combination of columns jointly matters more for the outcome.
    """)

    @st.cache_data
    def run_rf_coimportance(X_bytes, y_bytes, shape_X, n_est):
        X = np.frombuffer(X_bytes, dtype=np.float64).reshape(shape_X)
        y = np.frombuffer(y_bytes, dtype=np.float64).astype(int)
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=5,
                                     random_state=42, n_jobs=-1,
                                     class_weight="balanced")
        rf.fit(X, y)
        # Individual importances
        imp = rf.feature_importances_
        # Co-importance: product of pair importances (proxy for joint contribution)
        n = len(imp)
        co_mat = np.outer(imp, imp)
        np.fill_diagonal(co_mat, imp)  # diagonal = individual importance
        return co_mat, imp, rf.oob_score_ if rf.oob_score else 0

    # RF needs oob_score
    @st.cache_data
    def run_rf_full(X_bytes, y_bytes, shape_X, n_est):
        X = np.frombuffer(X_bytes, dtype=np.float64).reshape(shape_X)
        y = np.frombuffer(y_bytes, dtype=np.float64).astype(int)
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=5,
                                     random_state=42, n_jobs=-1,
                                     class_weight="balanced",
                                     oob_score=True)
        rf.fit(X, y)
        imp = rf.feature_importances_
        co_mat = np.outer(imp, imp)
        np.fill_diagonal(co_mat, imp)
        return co_mat, imp, rf

    try:
        co_mat, rf_imp, rf_model = run_rf_full(
            X6.values.tobytes(), y6.values.astype(float).tobytes(),
            X6.shape, n_trees)
        _rf_ok = True
    except Exception as _rf_err:
        st.warning(f"Random Forest could not be trained: {_rf_err}. "
                   "Try a different question or clear filters.")
        _rf_ok = False

    if not _rf_ok:
        return

    feat_labels_rf = [f.replace("_"," ").title() for f in X6.columns]
    co_df = pd.DataFrame(co_mat, index=feat_labels_rf, columns=feat_labels_rf)

    fig_co = px.imshow(
        co_df.round(5),
        color_continuous_scale="Blues",
        title=f"Feature Co-importance Matrix -- {q_label6}",
        labels=dict(color="Co-importance"),
        aspect="auto",
        text_auto=".4f",
    )
    fig_co.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333333", size=10),
        autosize=True,
        height=max(420, 46*len(feat_labels_rf)+80),
        margin=dict(t=60, b=70, l=10, r=10),
        xaxis=dict(tickangle=-40, tickfont=dict(size=9)),
        yaxis=dict(tickfont=dict(size=9)),
    )
    st.plotly_chart(fig_co, use_container_width=True)

    # Top co-important pairs
    co_pairs = []
    for i in range(len(feat_labels_rf)):
        for j in range(i+1, len(feat_labels_rf)):
            co_pairs.append({
                "Feature A": feat_labels_rf[i],
                "Feature B": feat_labels_rf[j],
                "Co-importance": round(float(co_mat[i,j]), 6),
            })
    co_pairs_df = pd.DataFrame(co_pairs).sort_values("Co-importance", ascending=False).head(8)

    _co_top      = co_pairs_df.iloc[0] if not co_pairs_df.empty else None
    _co_diag_max = feat_labels_rf[np.argmax(rf_imp)]
    _co_diag_val = rf_imp.max()
    interpret([
        f"The darkest off-diagonal cell represents the strongest jointly important "
        f"feature pair: "
        + (f"<b>{_co_top['Feature A']}</b> x <b>{_co_top['Feature B']}</b> "
           f"(co-importance = {_co_top['Co-importance']:.5f}), meaning these two features "
           f"together explain the most variance in <b>{q_label6}</b>."
           if _co_top is not None else "no pairs found."),
        f"The diagonal shows individual feature importance: "
        f"<b>{_co_diag_max}</b> is the single most important feature "
        f"(importance = {_co_diag_val:.4f}). Features with high diagonal AND high "
        f"off-diagonal values are both individually important and strongly co-predictive "
        f"with other features.",
    ])

    st.markdown("**Top 8 co-important feature pairs (jointly drive the outcome most):**")
    fig_copairs = px.bar(
        co_pairs_df,
        x="Co-importance",
        y=co_pairs_df["Feature A"] + "  x  " + co_pairs_df["Feature B"],
        orientation="h",
        color="Co-importance",
        color_continuous_scale="Blues",
        title="Top Feature Pairs by Joint Importance",
        labels={"y": "Feature Pair"},
        text=co_pairs_df["Co-importance"].apply(lambda v: f"{v:.5f}"),
    )
    fig_copairs.update_traces(textposition="outside", marker_line_width=0)
    fig_copairs.update_layout(
        plot_bgcolor="rgba(248,249,252,1)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333333", size=10),
        autosize=True,
        height=max(360, 44*len(co_pairs_df)+80),
        margin=dict(t=48, b=40, l=10, r=80),
        coloraxis_showscale=False,
        yaxis=dict(title="", tickfont=dict(size=9)),
        xaxis=dict(title="Co-importance score"),
        showlegend=False,
    )
    st.plotly_chart(fig_copairs, use_container_width=True)

    for _, r in co_pairs_df.head(3).iterrows():
        st.markdown(
            f'<div class="insight-box">- <b>{r["Feature A"]}</b> and <b>{r["Feature B"]}</b> jointly account for the most variation in <b>{q_label6}</b> (co-importance = {r["Co-importance"]:.5f}).</div>',
            unsafe_allow_html=True)

    # ==============================================================================
    # PLOT 3 -- Sankey: flow from top features through values to outcome
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 3 - Sankey Flow: How Column Combinations Flow to Outcome")
    st.markdown("""
    The Sankey diagram traces respondents from their **top-2 most important
    feature values** through to Positive / Negative outcome. Width of each band =
    number of respondents. This makes the co-occurrence pattern immediately visible.
    """)

    # Get top 2 features by RF importance
    top2_idx  = np.argsort(rf_imp)[::-1][:2]
    top2_feat = [list(X6.columns)[i] for i in top2_idx]
    top2_label= [f.replace("_"," ").title() for f in top2_feat]

    # Decode top2 columns back to text for display
    df6 = df.loc[valid_idx6].copy()
    for feat in top2_feat:
        df6[feat+"_lbl"] = decode_col(df6, feat, encoders)

    df6["outcome_lbl"] = df6["_y6"].map({1:"Positive", 0:"Negative"})

    # Build Sankey nodes and links
    feat_a = top2_feat[0] + "_lbl"
    feat_b = top2_feat[1] + "_lbl"

    # Nodes: unique values of feat_a, feat_b, outcome
    vals_a   = df6[feat_a].dropna().unique().tolist()
    vals_b   = df6[feat_b].dropna().unique().tolist()
    outcomes = ["Positive", "Negative"]

    all_nodes = (
        [f"{top2_label[0]}: {v}" for v in vals_a] +
        [f"{top2_label[1]}: {v}" for v in vals_b] +
        outcomes
    )
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    links_src, links_tgt, links_val, links_lbl = [], [], [], []

    # feat_a -> feat_b
    ab = df6.groupby([feat_a, feat_b]).size().reset_index(name="n")
    for _, row in ab.iterrows():
        src = f"{top2_label[0]}: {row[feat_a]}"
        tgt = f"{top2_label[1]}: {row[feat_b]}"
        if src in node_idx and tgt in node_idx and row["n"] >= 3:
            links_src.append(node_idx[src])
            links_tgt.append(node_idx[tgt])
            links_val.append(int(row["n"]))
            links_lbl.append(f"n={row['n']}")

    # feat_b -> outcome
    bo = df6.groupby([feat_b, "outcome_lbl"]).size().reset_index(name="n")
    for _, row in bo.iterrows():
        src = f"{top2_label[1]}: {row[feat_b]}"
        tgt = row["outcome_lbl"]
        if src in node_idx and tgt in node_idx and row["n"] >= 3:
            links_src.append(node_idx[src])
            links_tgt.append(node_idx[tgt])
            links_val.append(int(row["n"]))
            links_lbl.append(f"n={row['n']}")

    # Node colours
    def node_colour(n):
        if n == "Positive":  return "rgba(78,175,130,0.85)"
        if n == "Negative":  return "rgba(217,95,95,0.85)"
        if top2_label[0] in n: return "rgba(74,159,212,0.7)"
        return "rgba(155,114,192,0.7)"

    node_colours = [node_colour(n) for n in all_nodes]

    fig_sankey = go.Figure(go.Sankey(
        node=dict(
            pad=18, thickness=20,
            line=dict(color="rgba(100,100,100,0.3)", width=0.5),
            label=all_nodes,
            color=node_colours,
            hovertemplate="%{label}<extra></extra>",
        ),
        link=dict(
            source=links_src,
            target=links_tgt,
            value=links_val,
            label=links_lbl,
            color="rgba(150,180,210,0.35)",
        ),
    ))
    fig_sankey.update_layout(
        title=f"Sankey: {top2_label[0]} + {top2_label[1]} -> {q_label6}",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#333333", size=10),
        autosize=True,
        height=500,
        margin=dict(t=60, b=40, l=10, r=10),
    )
    st.plotly_chart(fig_sankey, use_container_width=True)
    st.caption(f"Driven by top-2 RF features: **{top2_label[0]}** and **{top2_label[1]}**")
    _sank_pos = df6["outcome_lbl"].value_counts().get("Positive", 0)
    _sank_neg = df6["outcome_lbl"].value_counts().get("Negative", 0)
    _sank_pct = round(_sank_pos / (_sank_pos + _sank_neg) * 100, 1) if (_sank_pos + _sank_neg) > 0 else 0
    _widest_link = max(links_val) if links_val else 0
    interpret([
        f"Each band's width represents the number of respondents flowing through that "
        f"path - the widest band (n={_widest_link}) shows the most common "
        f"<b>{top2_label[0]} + {top2_label[1]}</b> combination in the data. "
        f"Trace the green (Positive) bands back to see which feature value combinations "
        f"produce the most positive outcomes.",
        f"Overall, <b>{_sank_pct}%</b> of respondents in this filtered view gave a "
        f"positive response ({_sank_pos} of {_sank_pos+_sank_neg}). "
        f"{'Bands that stay green all the way through indicate feature value combinations that consistently drive high positive rates.' if _sank_pct >= 50 else 'Many bands flow toward Negative - look for thin green bands to identify which specific combinations still achieve positive outcomes despite the overall lower rate.'}",
    ])

    # ==============================================================================
    # PLOT 4 -- Bubble chart: column combination co-occurrence vs outcome rate
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 4 - Bubble Chart: Feature Combination Co-occurrence vs Outcome Rate")
    st.markdown("""
    Each bubble = a specific combination of two column values.
    - **X-axis** = how common that combination is (% of all respondents)
    - **Y-axis** = % who gave a positive response in that combination
    - **Bubble size** = number of respondents
    - **Colour** = outcome rate (red=low, green=high)

    This directly shows which co-occurring column values drive the score.
    """)

    # User picks which two columns to bubble-plot
    avail_cols = [c for c in FEATURE_COLS if c in df6.columns]
    col_labels  = [c.replace("_"," ").title() for c in avail_cols]
    col_map     = dict(zip(col_labels, avail_cols))

    bc1, bc2 = st.columns(2)
    with bc1:
        feat_x_label = st.selectbox("Feature A (x-grouping)",
                                     col_labels,
                                     index=col_labels.index(top2_label[0])
                                           if top2_label[0] in col_labels else 0,
                                     key="bc_feat_a")
    with bc2:
        feat_y_label = st.selectbox("Feature B (y-grouping)",
                                     col_labels,
                                     index=col_labels.index(top2_label[1])
                                           if top2_label[1] in col_labels else
                                           min(1, len(col_labels)-1),
                                     key="bc_feat_b")

    feat_x_raw = col_map[feat_x_label]
    feat_y_raw = col_map[feat_y_label]

    # Decode both to text
    df6["_bx"] = decode_col(df6, feat_x_raw, encoders)
    df6["_by"] = decode_col(df6, feat_y_raw, encoders)

    bubble_grp = (df6.groupby(["_bx","_by"])
                     .agg(n=("_y6","count"),
                          pos_pct=("_y6","mean"))
                     .reset_index())
    bubble_grp = bubble_grp[bubble_grp["n"] >= min_support]
    bubble_grp["pos_pct"] = (bubble_grp["pos_pct"]*100).round(1)
    bubble_grp["freq_pct"] = (bubble_grp["n"] / bubble_grp["n"].sum() * 100).round(1)
    bubble_grp["combo"]    = (bubble_grp["_bx"].astype(str)
                              + " + " + bubble_grp["_by"].astype(str))

    if bubble_grp.empty:
        st.info("Not enough data for bubble chart with current filters. "
                "Try lowering the min support threshold.")
    else:
        fig_bubble = px.scatter(
            bubble_grp,
            x="freq_pct",
            y="pos_pct",
            size="n",
            color="pos_pct",
            color_continuous_scale="RdYlGn",
            range_color=[0, 100],
            hover_name="combo",
            hover_data={"n": True,
                        "pos_pct": ":.1f",
                        "freq_pct": ":.1f",
                        "_bx": False,
                        "_by": False},
            title=f"Co-occurrence Bubble: {feat_x_label} + {feat_y_label} --> {q_label6}",
            labels={"freq_pct": "% of all respondents (how common)",
                    "pos_pct":  "% Positive response",
                    "n":        "Respondents"},
            size_max=60,
        )
        fig_bubble.update_traces(
            marker=dict(line=dict(width=1, color="rgba(100,100,100,0.4)")),
        )
        fig_bubble.add_hline(y=50, line_dash="dot",
                              line_color="grey",
                              annotation_text="50% threshold",
                              annotation_position="bottom right")
        fig_bubble.update_layout(
            plot_bgcolor="rgba(248,249,252,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333333", size=11),
            autosize=True,
            height=520,
            margin=dict(t=60, b=50, l=10, r=10),
            coloraxis_colorbar=dict(title="% Positive", tickfont=dict(size=9)),
            xaxis=dict(title="% of Respondents (how common)", tickfont=dict(size=10)),
            yaxis=dict(title="% Positive Response", range=[-5, 105]),
        )
        st.plotly_chart(fig_bubble, use_container_width=True)
        _bub_best  = bubble_grp.loc[bubble_grp["pos_pct"].idxmax()]
        _bub_worst = bubble_grp.loc[bubble_grp["pos_pct"].idxmin()]
        _bub_most_common = bubble_grp.loc[bubble_grp["freq_pct"].idxmax()]
        interpret([
            f"The <b>top-right quadrant</b> (common AND high positive rate) contains the most "
            f"actionable combinations. The best performing is "
            f"<b>{_bub_best['combo']}</b> ({_bub_best['pos_pct']}% positive, "
            f"{_bub_best['freq_pct']}% of respondents), while the lowest is "
            f"<b>{_bub_worst['combo']}</b> ({_bub_worst['pos_pct']}% positive).",
            f"The most common combination is <b>{_bub_most_common['combo']}</b> "
            f"({_bub_most_common['freq_pct']}% of all respondents, n={int(_bub_most_common['n'])}) "
            f"with a {_bub_most_common['pos_pct']}% positive rate - "
            f"{'this dominant group already performs well above the 50% baseline' if _bub_most_common['pos_pct'] >= 50 else 'this dominant group falls below the 50% baseline, suggesting broad intervention may be needed rather than targeting rare subgroups'}.",
        ])

        # Auto insight: top-right quadrant = common AND positive
        high_freq   = bubble_grp["freq_pct"].quantile(0.5)
        common_pos  = bubble_grp[(bubble_grp["freq_pct"] >= high_freq) &
                                  (bubble_grp["pos_pct"]  >= 60)]
        common_neg  = bubble_grp[(bubble_grp["freq_pct"] >= high_freq) &
                                  (bubble_grp["pos_pct"]  <  50)]

        if not common_pos.empty:
            st.markdown("**Common combinations with HIGH positive rate (top-right quadrant):**")
            for _, r in common_pos.sort_values("pos_pct", ascending=False).head(4).iterrows():
                st.markdown(
                    f'<div class="strong-box">- <b>{r["combo"]}</b>: {r["pos_pct"]}% positive, {r["n"]} respondents ({r["freq_pct"]}% of total)</div>',
                    unsafe_allow_html=True)

        if not common_neg.empty:
            st.markdown("**Common combinations with LOW positive rate (bottom-right quadrant):**")
            for _, r in common_neg.sort_values("pos_pct").head(4).iterrows():
                st.markdown(
                    f'<div class="warn-box">- <b>{r["combo"]}</b>: {r["pos_pct"]}% positive, {r["n"]} respondents ({r["freq_pct"]}% of total)</div>',
                    unsafe_allow_html=True)

    # ==============================================================================
    # PLOT 5 -- Multi-column co-occurrence: 3 readable charts
    # ==============================================================================
    st.markdown("---")
    st.markdown("### 5 - Multi-Column Co-occurrence: Which Value Combinations Drive the Outcome")
    st.markdown("""
    Instead of spaghetti lines, three complementary views show the same story clearly:

    - **5a** -- Combination heatmap: % positive for every top-feature value combination
    - **5c** -- Pattern summary table: every meaningful co-occurrence ranked by outcome
    """)

    # Top 4 RF features
    top4_idx   = np.argsort(rf_imp)[::-1][:4]
    top4_feats = [list(X6.columns)[i] for i in top4_idx]
    top4_labels= [f.replace("_"," ").title() for f in top4_feats]

    # Decode top4 to text labels
    df6_top = df6[top4_feats + ["_y6"]].copy()
    for feat in top4_feats:
        df6_top[feat+"_lbl"] = decode_col(df6_top, feat, encoders)

    # ---- 5a: Combination heatmap (top2 features) ----------------------------------
    st.markdown("#### 5a - Combination Heatmap: % Positive for Each Value Pair")
    st.caption(
        "Rows = values of the most important feature. "
        "Columns = values of the second most important feature. "
        "Cell colour = % of respondents in that combination who gave a positive response."
    )

    feat_a5 = top4_feats[0] + "_lbl"
    feat_b5 = top4_feats[1] + "_lbl"
    la5 = top4_labels[0]
    lb5 = top4_labels[1]

    hm_grp = (df6_top.dropna(subset=[feat_a5, feat_b5, "_y6"])
                      .groupby([feat_a5, feat_b5])["_y6"]
                      .agg(["mean","count"])
                      .reset_index())
    hm_grp = hm_grp[hm_grp["count"] >= min_support]
    hm_grp["pct"] = (hm_grp["mean"] * 100).round(1)
    hm_grp["label"] = hm_grp["pct"].astype(str) + "%" + chr(10) + "n=" + hm_grp["count"].astype(str)

    if not hm_grp.empty:
        pivot5 = hm_grp.pivot(index=feat_a5, columns=feat_b5, values="pct")
        pivot_n= hm_grp.pivot(index=feat_a5, columns=feat_b5, values="count")

        # Annotate cells with both % and n
        annot = pivot5.copy().astype(object)
        for r in pivot5.index:
            for c in pivot5.columns:
                v = pivot5.loc[r, c]
                n = pivot_n.loc[r, c] if r in pivot_n.index and c in pivot_n.columns else ""
                if pd.notna(v):
                    annot.loc[r, c] = f"{v:.0f}%  n={int(n)}"
                else:
                    annot.loc[r, c] = ""

        fig_hm5 = go.Figure(go.Heatmap(
            z=pivot5.values,
            x=[str(c)[:20] for c in pivot5.columns],
            y=[str(r)[:25] for r in pivot5.index],
            text=annot.values,
            texttemplate="%{text}",
            textfont=dict(size=11, color="black"),
            colorscale="RdYlGn",
            zmin=0, zmax=100,
            colorbar=dict(title="% Positive", thickness=14),
            hoverongaps=False,
            hovertemplate=(
                f"{la5}: %{{y}}<br>"
                f"{lb5}: %{{x}}<br>"
                "% Positive: %{z:.1f}%<extra></extra>"
            ),
        ))
        fig_hm5.update_layout(
            title=f"% Positive: {la5} (rows) x {lb5} (columns)",
            xaxis=dict(title=lb5, tickangle=-40, tickfont=dict(size=9)),
            yaxis=dict(title=la5, tickfont=dict(size=9)),
            plot_bgcolor="rgba(248,249,252,1)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#333333", size=10),
            autosize=True,
            height=max(320, 58 * len(pivot5) + 120),
            margin=dict(t=60, b=80, l=10, r=10),
        )
        st.plotly_chart(fig_hm5, use_container_width=True)

        # Key insight for heatmap
        best5  = hm_grp.loc[hm_grp["pct"].idxmax()]
        worst5 = hm_grp.loc[hm_grp["pct"].idxmin()]
        _hm5_range = best5["pct"] - worst5["pct"]
        _hm5_cells = len(hm_grp)
        interpret([
            f"The green cell (<b>{best5[feat_a5]}</b> + <b>{best5[feat_b5]}</b>) achieves "
            f"the highest positive rate at <b>{best5['pct']:.0f}%</b> (n={int(best5['count'])}), "
            f"while the red cell (<b>{worst5[feat_a5]}</b> + <b>{worst5[feat_b5]}</b>) "
            f"achieves the lowest at <b>{worst5['pct']:.0f}%</b> (n={int(worst5['count'])}).",
            f"The {_hm5_range:.1f} percentage-point spread across {_hm5_cells} combinations "
            f"{'is substantial - the two features together create meaningfully different outcomes across groups' if _hm5_range >= 20 else 'is moderate - the features have some influence on outcomes but the effect is not dramatic across all combinations'}.",
        ])
        st.markdown(
            f'<div class="strong-box">'
            f'Highest: <b>{best5[feat_a5]}</b> + <b>{best5[feat_b5]}</b> '
            f'-- {best5["pct"]:.0f}% positive (n={int(best5["count"])})'
            f'</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="warn-box">'
            f'Lowest: <b>{worst5[feat_a5]}</b> + <b>{worst5[feat_b5]}</b> '
            f'-- {worst5["pct"]:.0f}% positive (n={int(worst5["count"])})'
            f'</div>', unsafe_allow_html=True)
    else:
        st.info("Not enough data for combination heatmap with current filters.")


    # ---- 5c: Pattern summary table -----------------------------------------------
    st.markdown("---")
    st.markdown("#### 5b - Pattern Summary Table: All Meaningful Co-occurrences Ranked")
    st.caption(
        "Every combination of top-feature values with enough respondents, "
        "ranked from highest to lowest positive response rate. "
        "Use this as a quick reference for which profiles perform best/worst."
    )

    # Build all pairwise combos of top4 decoded features
    summary_rows = []
    feat_pairs5 = list(itertools.combinations(
        [(f+"_lbl", l) for f, l in zip(top4_feats, top4_labels)], 2))

    for (fa, la), (fb, lb) in feat_pairs5:
        grp = (df6_top.dropna(subset=[fa, fb, "_y6"])
                       .groupby([fa, fb])["_y6"]
                       .agg(["mean","count"])
                       .reset_index())
        grp = grp[grp["count"] >= min_support]
        for _, row in grp.iterrows():
            summary_rows.append({
                "Feature A": la,
                "Value A":   str(row[fa])[:25],
                "Feature B": lb,
                "Value B":   str(row[fb])[:25],
                "n":         int(row["count"]),
                "% Positive": round(row["mean"] * 100, 1),
                "Pattern": (
                    "High" if row["mean"] >= 0.65 else
                    "Low"  if row["mean"] <= 0.40 else
                    "Mid"
                ),
            })

    if summary_rows:
        summary5 = (pd.DataFrame(summary_rows)
                      .sort_values("% Positive", ascending=False)
                      .reset_index(drop=True))

        # Colour pattern column
        def style_pattern(val):
            if val == "High": return "background-color:#D4EDDA; color:#155724; font-weight:600"
            if val == "Low":  return "background-color:#F8D7DA; color:#721c24; font-weight:600"
            return "background-color:#FFF3CD; color:#856404"

        st.dataframe(
            summary5.style
                .applymap(style_pattern, subset=["Pattern"])
                .background_gradient(subset=["% Positive"], cmap="RdYlGn", vmin=0, vmax=100)
                .format({"% Positive": "{:.1f}%"}),
            use_container_width=True,
            height=460,
        )
        _n_high = (summary5["Pattern"] == "High").sum()
        _n_low  = (summary5["Pattern"] == "Low").sum()
        _n_mid  = (summary5["Pattern"] == "Mid").sum()
        _tbl_best  = summary5.iloc[0]
        _tbl_worst = summary5.iloc[-1]
        interpret([
            f"Across all top-feature co-occurrence combinations, "
            f"<b>{_n_high} are High-performing</b> (-65% positive), "
            f"<b>{_n_mid} are Mid-range</b>, and "
            f"<b>{_n_low} are Low-performing</b> (-40% positive). "
            f"The best combination is <b>{_tbl_best['Value A']}</b> + "
            f"<b>{_tbl_best['Value B']}</b> at {_tbl_best['% Positive']:.0f}%.",
            f"The lowest combination is <b>{_tbl_worst['Value A']}</b> + "
            f"<b>{_tbl_worst['Value B']}</b> at {_tbl_worst['% Positive']:.0f}% - "
            f"{'consider targeted support for respondents in this profile' if _tbl_worst['% Positive'] <= 40 else 'even the lowest-performing combination is above 40%, suggesting relatively consistent outcomes across groups'}.",
        ])

        # Auto-written key findings
        high_patterns = summary5[summary5["Pattern"] == "High"].head(3)
        low_patterns  = summary5[summary5["Pattern"] == "Low"].tail(3)

        if not high_patterns.empty:
            st.markdown("**Key HIGH-performing co-occurrences found by ML:**")
            for _, r in high_patterns.iterrows():
                st.markdown(
                    f'<div class="strong-box">'
                    f'When <b>{r["Feature A"]}</b> = <i>{r["Value A"]}</i> '
                    f'AND <b>{r["Feature B"]}</b> = <i>{r["Value B"]}</i>: '
                    f'<b>{r["% Positive"]:.0f}% positive response</b> '
                    f'(n={r["n"]})</div>',
                    unsafe_allow_html=True)

        if not low_patterns.empty:
            st.markdown("**Key LOW-performing co-occurrences found by ML:**")
            for _, r in low_patterns.sort_values("% Positive").iterrows():
                st.markdown(
                    f'<div class="warn-box">'
                    f'When <b>{r["Feature A"]}</b> = <i>{r["Value A"]}</i> '
                    f'AND <b>{r["Feature B"]}</b> = <i>{r["Value B"]}</i>: '
                    f'only <b>{r["% Positive"]:.0f}% positive response</b> '
                    f'(n={r["n"]})</div>',
                    unsafe_allow_html=True)
    else:
        st.info("Not enough co-occurrence data with current filters.")

    st.markdown("---")
    st.caption(
        "Part 6 methods: DecisionTreeClassifier (sklearn) - "
        "RandomForestClassifier joint importance - "
        "Sankey via Plotly - Bubble co-occurrence - Parallel Coordinates. "
        "All patterns are data-driven, not hand-picked."
    )

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🌟 Workshop Usefulness",
    "🧠 Changed Understanding",
    "⭐ LMK Leader Rating",
    "🆘 Know Where to Get Help",
    "🔎 Pattern Analysis",
])

with tab1:
    render_tab(
        col="workshop_useful",
        q_title="Do you think today's workshop will be useful in your relationships "
                "either right now or in future situations?",
        order=USEFUL_ORDER,
        positive_set=POSITIVE_USEFUL,
        colors=USEFUL_COLORS,
        pos_label="Definitely / Probably",
    )

with tab2:
    # Prefer changed_understanding (Agree scale, ACPD); fall back to
    # learnt_healthy_behaviours which uses the same scale for other modules
    has_cu = ("changed_understanding" in dfv.columns and
              dfv["changed_understanding"].dropna().shape[0] > 0)
    use_col = "changed_understanding" if has_cu else "learnt_healthy_behaviours"
    render_tab(
        col=use_col,
        q_title="Today's workshop has changed my understanding of what behaviours "
                "are healthy and unhealthy in relationships.",
        order=AGREE_ORDER,
        positive_set=POSITIVE_AGREE,
        colors=AGREE_COLORS,
        pos_label="Strongly Agree / Agree",
        note="Responses use an agree/disagree scale. "
             "Positive = 'Strongly agree' or 'Agree'.",
    )

with tab3:
    render_tab(
        col="leader_rating",
        q_title="How would you rate your LMK leader today?",
        order=RATING_ORDER,
        positive_set=POSITIVE_RATING,
        colors=RATING_COLORS,
        pos_label="Excellent / Good",
        note="Positive = 'Excellent' or 'Good'.",
    )

with tab4:
    render_tab(
        col="know_where_to_go",
        q_title="(For 10 Signs and Delving Deeper modules only) I know where to go for "
                "help or advice if either myself or a friend is in an unhealthy or "
                "abusive situation in a relationship.",
        order=AGREE_ORDER,
        positive_set=POSITIVE_AGREE,
        colors=AGREE_COLORS,
        pos_label="Strongly Agree / Agree",
        note="ℹ️ This question applies to **10 Signs** and **Delving Deeper** modules. "
             "Use the Module filter in the sidebar to narrow results.",
    )

with tab5:
    render_pattern_analysis_tab()
