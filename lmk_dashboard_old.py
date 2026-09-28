"""
LMK Impact Dashboard – Streamlit version
Run:  streamlit run lmk_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="LMK Impact Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1E2A3A; }
    [data-testid="stSidebar"] * { color: #E8EDF2 !important; }
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stMultiSelect label { color: #A8C4DC !important; font-weight: 600; }
    [data-testid="stSidebar"] h2 { color: #5B8DB8 !important; font-size: 1.1rem; letter-spacing: 0.05em; }

    /* Main area */
    .main { background-color: #F7F9FC; }

    /* KPI cards */
    .kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 20px; }
    .kpi-card {
        flex: 1; min-width: 140px; background: white;
        border-radius: 10px; padding: 16px 20px; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .kpi-value { font-size: 1.9rem; font-weight: 700; margin: 0; }
    .kpi-label { font-size: 0.78rem; color: #666; margin-top: 4px; }

    /* Tab headings */
    .stTabs [data-baseweb="tab"] { font-size: 0.88rem; font-weight: 600; }

    /* Section header */
    .section-header {
        font-size: 0.85rem; font-weight: 700; color: #5B8DB8;
        text-transform: uppercase; letter-spacing: 0.08em;
        margin: 18px 0 6px; border-bottom: 1px solid #d0dce8; padding-bottom: 4px;
    }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────
@st.cache_data
def load_data():
    # ----------------------------------------------------------
    # Replace these paths with your actual CSV file locations
    # ----------------------------------------------------------
    df_sessions = pd.read_csv("Sessions 23-24 and 24-25.csv")
    df_survey   = pd.read_csv("Impact surveys 23-24 and 24-25.csv")

    df_sessions.columns = df_sessions.columns.str.strip()
    df_survey.columns   = df_survey.columns.str.strip()

    df_sessions = df_sessions.rename(columns={
        "Record ID":                                                      "record_id",
        "Module":                                                         "module",
        "Session start date and time":                                    "session_date",
        "Academic year":                                                  "academic_year",
        "Organisation Name":                                              "org_name",
        "Org Borough":                                                    "borough",
        "Confirmed number of expected participants (Youth + Adult)":      "expected_participants",
        "Confirmed number of participants (Youth + Adult)":               "actual_participants",
        "Number of surveys":                                              "num_surveys",
        "Org Type":                                                       "org_type",
        "Org Sub Type":                                                   "org_sub_type",
        "Org % on school meals":                                          "pct_school_meals",
        "Vulnerable group":                                               "vulnerable_group",
    })

    df_survey = df_survey.rename(columns={
        "Session Record ID":                                              "session_id",
        "Session Name":                                                   "session_name",
        "Record ID":                                                      "record_id",
        "Age (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":                     "age",
        "Gender (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8, ACPD)":            "gender",
        "Ethnicity (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":               "ethnicity",
        "Disability (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8)":              "disability",
        "Sexuality":                                                      "sexuality",
        "Learning difficulty":                                            "learning_difficulty",
        "Leader rating (Y10SDD+AP+7/8, Y10SInc, YIoP+7/8, YSII+Inc+7/8, ACPD, AWP)":
            "leader_rating",
        "Learnt something new about healthy and unhealthy behaviours in relationships (Y10SDD+AP+7/8, YIoP, YSII, AWP)":
            "learnt_healthy_behaviours",
        "Know who and where to go if worried about relationship (Y10SDD+AP+7/8, Y10SInc+Pri, YDDPri, YIoP+7/8, YSII+Inc+7/8, AWP)":
            "know_where_to_go",
        "Workshop useful/helpful relationships (Y10SDD+AP+7/8, YIoP+7/8, YSII+7/8, AWP)":
            "workshop_useful",
    })

    # Numeric coercion
    for col in ["leader_rating", "learnt_healthy_behaviours",
                "know_where_to_go", "workshop_useful"]:
        if col in df_survey.columns:
            df_survey[col] = pd.to_numeric(df_survey[col], errors="coerce")

    return df_sessions, df_survey


df_sessions_raw, df_survey_raw = load_data()


# ── Helper: merge session metadata onto survey rows ───────────
def enrich_survey(dfs: pd.DataFrame, dfv: pd.DataFrame) -> pd.DataFrame:
    """Join session-level fields onto the filtered survey dataframe."""
    session_meta = dfs[["record_id", "vulnerable_group", "pct_school_meals"]].copy()
    session_meta = session_meta.rename(columns={"record_id": "session_id"})
    merged = dfv.merge(session_meta, on="session_id", how="left",
                       suffixes=("", "_session"))
    return merged


# ── Response-rate helper ──────────────────────────────────────
def pct_agree(series: pd.Series, threshold: int = 4) -> float:
    """% of non-null values >= threshold (agree/strongly agree on 1-5 scale)."""
    valid = series.dropna()
    if len(valid) == 0:
        return np.nan
    return round((valid >= threshold).sum() / len(valid) * 100, 1)


# ── Chart helpers ─────────────────────────────────────────────
ACCENT   = "#5B8DB8"
GREEN    = "#8BC4A8"
ORANGE   = "#E8A87C"
PURPLE   = "#C47EC4"
PALETTE  = px.colors.qualitative.Pastel

CHART_LAYOUT = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    margin=dict(t=40, b=30, l=10, r=10),
    height=340,
    font=dict(size=12),
)


def bar_by_group(dfv: pd.DataFrame, col: str, group: str, title: str,
                 color: str = ACCENT) -> go.Figure:
    """Avg score for `col` broken down by `group`."""
    if col not in dfv.columns or group not in dfv.columns:
        return empty_fig(f"Column '{col}' or '{group}' not found in data")
    agg = (dfv.dropna(subset=[col, group])
               .groupby(group)[col]
               .mean()
               .reset_index()
               .rename(columns={col: "Avg Score", group: group.replace("_", " ").title()}))
    fig = px.bar(agg, x=agg.columns[0], y="Avg Score",
                 title=title, color_discrete_sequence=[color],
                 range_y=[0, 5])
    fig.update_layout(**CHART_LAYOUT)
    fig.update_traces(marker_line_width=0)
    return fig


def dist_chart(dfv: pd.DataFrame, col: str, title: str,
               color: str = ACCENT) -> go.Figure:
    """Distribution of Likert scores (1-5)."""
    if col not in dfv.columns:
        return empty_fig(f"Column '{col}' not found in data")
    counts = (dfv[col].dropna()
                      .astype(int)
                      .value_counts()
                      .sort_index()
                      .reset_index())
    counts.columns = ["Score", "Count"]
    total = counts["Count"].sum()
    counts["Pct"] = (counts["Count"] / total * 100).round(1)
    fig = px.bar(counts, x="Score", y="Count",
                 text=counts["Pct"].astype(str) + "%",
                 title=title, color_discrete_sequence=[color])
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(**CHART_LAYOUT, xaxis=dict(tickmode="linear", dtick=1))
    return fig


def pie_chart(dfv: pd.DataFrame, col: str, title: str) -> go.Figure:
    if col not in dfv.columns:
        return empty_fig(f"Column '{col}' not found in data")
    vals = dfv[col].dropna().value_counts().reset_index()
    vals.columns = [col, "Count"]
    fig = px.pie(vals, names=col, values="Count", title=title,
                 color_discrete_sequence=PALETTE)
    fig.update_layout(**CHART_LAYOUT)
    return fig


def empty_fig(msg: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#aaa"))
    fig.update_layout(**CHART_LAYOUT, title="")
    return fig


def kpi_metric(col: str, dfv: pd.DataFrame, label: str, color: str,
               threshold: int = 4):
    """Render a KPI metric box in a Streamlit column."""
    pct = pct_agree(dfv[col], threshold) if col in dfv.columns else None
    n   = dfv[col].dropna().shape[0] if col in dfv.columns else 0
    val = f"{pct}%" if pct is not None else "N/A"
    st.markdown(f"""
    <div class="kpi-card">
        <p class="kpi-value" style="color:{color}">{val}</p>
        <p class="kpi-label">{label}<br><span style="font-size:0.7rem;color:#999">n={n:,}</span></p>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 LMK Dashboard")
    st.markdown("---")
    st.markdown("### 🔽 Filters")

    def ms(label, options, key):
        return st.multiselect(label, sorted([o for o in options if pd.notna(o)]), key=key)

    # ── Session-level filters ─────────────────────────────────
    st.markdown('<p class="section-header">Session filters</p>', unsafe_allow_html=True)
    sel_borough  = ms("Borough",       df_sessions_raw["borough"].unique(),       "borough")
    sel_year     = ms("Academic Year", df_sessions_raw["academic_year"].unique(), "year")
    sel_module   = ms("Module",        df_sessions_raw["module"].unique(),         "module")
    sel_orgtype  = ms("Org Type",      df_sessions_raw["org_type"].unique()
                      if "org_type" in df_sessions_raw.columns else [],           "orgtype")

    # Vulnerable group (session-derived)
    vg_opts = df_sessions_raw["vulnerable_group"].dropna().unique() \
              if "vulnerable_group" in df_sessions_raw.columns else []
    sel_vulnerable = ms("Vulnerable / Not Vulnerable Group", vg_opts, "vulnerable")

    # % Free school meals slider (session-derived)
    if "pct_school_meals" in df_sessions_raw.columns:
        fsm_vals = pd.to_numeric(df_sessions_raw["pct_school_meals"], errors="coerce").dropna()
        if not fsm_vals.empty:
            fsm_min, fsm_max = int(fsm_vals.min()), int(fsm_vals.max())
            sel_fsm = st.slider("% Eligible for Free School Meals",
                                fsm_min, fsm_max, (fsm_min, fsm_max), key="fsm")
        else:
            sel_fsm = None
    else:
        sel_fsm = None

    # ── Pupil-level (survey) filters ─────────────────────────
    st.markdown('<p class="section-header">Pupil filters</p>', unsafe_allow_html=True)

    def survey_ms(label, col, key):
        opts = df_survey_raw[col].dropna().unique() if col in df_survey_raw.columns else []
        return ms(label, opts, key)

    sel_gender    = survey_ms("Gender",             "gender",             "gender")
    sel_ethnicity = survey_ms("Ethnicity",           "ethnicity",          "ethnicity")
    sel_age       = survey_ms("Age",                 "age",                "age")
    sel_disability = survey_ms("Disability",         "disability",         "disability")
    sel_sexuality  = survey_ms("Sexuality",          "sexuality",          "sexuality")
    sel_ld         = survey_ms("Learning Difficulty","learning_difficulty","ld")

    st.markdown("---")
    if st.button("🔄 Reset All Filters", use_container_width=True):
        for k in ["borough","year","module","orgtype","vulnerable",
                  "gender","ethnicity","age","disability","sexuality","ld"]:
            st.session_state[k] = []
        st.rerun()


# ── Apply session-level filters ───────────────────────────────
dfs = df_sessions_raw.copy()
if sel_borough:   dfs = dfs[dfs["borough"].isin(sel_borough)]
if sel_year:      dfs = dfs[dfs["academic_year"].isin(sel_year)]
if sel_module:    dfs = dfs[dfs["module"].isin(sel_module)]
if sel_orgtype and "org_type" in dfs.columns:
    dfs = dfs[dfs["org_type"].isin(sel_orgtype)]
if sel_vulnerable and "vulnerable_group" in dfs.columns:
    dfs = dfs[dfs["vulnerable_group"].isin(sel_vulnerable)]
if sel_fsm is not None and "pct_school_meals" in dfs.columns:
    dfs["pct_school_meals_num"] = pd.to_numeric(dfs["pct_school_meals"], errors="coerce")
    dfs = dfs[dfs["pct_school_meals_num"].between(sel_fsm[0], sel_fsm[1]) |
              dfs["pct_school_meals_num"].isna()]

# Filter survey to matching session IDs
dfv = df_survey_raw[df_survey_raw["session_id"].isin(dfs["record_id"])].copy()

# Enrich survey with session-derived fields
dfv = enrich_survey(dfs, dfv)

# ── Apply pupil-level filters ─────────────────────────────────
if sel_gender    and "gender"             in dfv.columns: dfv = dfv[dfv["gender"].isin(sel_gender)]
if sel_ethnicity and "ethnicity"          in dfv.columns: dfv = dfv[dfv["ethnicity"].isin(sel_ethnicity)]
if sel_age       and "age"               in dfv.columns: dfv = dfv[dfv["age"].isin(sel_age)]
if sel_disability and "disability"       in dfv.columns: dfv = dfv[dfv["disability"].isin(sel_disability)]
if sel_sexuality  and "sexuality"        in dfv.columns: dfv = dfv[dfv["sexuality"].isin(sel_sexuality)]
if sel_ld         and "learning_difficulty" in dfv.columns: dfv = dfv[dfv["learning_difficulty"].isin(sel_ld)]


# ─────────────────────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────────────────────
st.markdown("# 📊 LMK Impact Dashboard")

# ── Top KPI strip ─────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.metric("Total Sessions",      f"{len(dfs):,}")
with k2:
    ep = int(dfs["expected_participants"].sum()) if "expected_participants" in dfs.columns else 0
    st.metric("Expected Participants", f"{ep:,}")
with k3:
    ap = int(dfs["actual_participants"].sum()) if "actual_participants" in dfs.columns else 0
    st.metric("Confirmed Participants", f"{ap:,}")
with k4:
    st.metric("Survey Responses", f"{len(dfv):,}")
with k5:
    avg_lr = dfv["leader_rating"].mean() if "leader_rating" in dfv.columns else None
    st.metric("Avg Leader Rating",
              f"{avg_lr:.2f}/5" if avg_lr and not np.isnan(avg_lr) else "N/A")

st.markdown("---")

# ── 4 Tabs ────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌟 Workshop Usefulness",
    "🧠 Changed Understanding",
    "⭐ LMK Leader Rating",
    "🆘 Know Where to Get Help",
])


# ═══════════════════════════════════════════════════════════════
# TAB 1 — Workshop Usefulness
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Do you think today's workshop will be useful in your relationships "
                 "either right now or in future situations?")
    col = "workshop_useful"

    # KPI row
    c1, c2, c3 = st.columns(3)
    with c1:
        pct = pct_agree(dfv[col]) if col in dfv.columns else None
        st.metric("Agreed / Strongly Agreed (4-5)", f"{pct}%" if pct else "N/A",
                  help="Score ≥ 4 out of 5")
    with c2:
        avg = dfv[col].mean() if col in dfv.columns else None
        st.metric("Average Score", f"{avg:.2f}/5" if avg and not np.isnan(avg) else "N/A")
    with c3:
        n = dfv[col].dropna().shape[0] if col in dfv.columns else 0
        st.metric("Responses", f"{n:,}")

    st.markdown("---")

    # Row 1 charts
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(dist_chart(dfv, col,
            "Score Distribution – Workshop Usefulness", ACCENT),
            use_container_width=True)
    with r1c2:
        st.plotly_chart(bar_by_group(dfv, col, "gender",
            "Avg Score by Gender", GREEN),
            use_container_width=True)

    # Row 2 charts
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(bar_by_group(dfv, col, "ethnicity",
            "Avg Score by Ethnicity", ORANGE),
            use_container_width=True)
    with r2c2:
        st.plotly_chart(bar_by_group(dfv, col, "age",
            "Avg Score by Age Group", PURPLE),
            use_container_width=True)

    # Row 3 charts
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(bar_by_group(dfv, col, "disability",
            "Avg Score by Disability", "#E07B6A"),
            use_container_width=True)
    with r3c2:
        st.plotly_chart(bar_by_group(dfv, col, "learning_difficulty",
            "Avg Score by Learning Difficulty", "#6AB8B8"),
            use_container_width=True)

    # Row 4
    r4c1, r4c2 = st.columns(2)
    with r4c1:
        st.plotly_chart(bar_by_group(dfv, col, "sexuality",
            "Avg Score by Sexuality", "#A08EC2"),
            use_container_width=True)
    with r4c2:
        st.plotly_chart(bar_by_group(dfv, col, "vulnerable_group",
            "Avg Score by Vulnerable / Not Vulnerable Group", "#D4A05A"),
            use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — Changed Understanding
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Today's workshop has changed my understanding of what behaviours "
                 "are healthy and unhealthy in relationships.")
    col = "learnt_healthy_behaviours"

    c1, c2, c3 = st.columns(3)
    with c1:
        pct = pct_agree(dfv[col]) if col in dfv.columns else None
        st.metric("Agreed / Strongly Agreed (4-5)", f"{pct}%" if pct else "N/A")
    with c2:
        avg = dfv[col].mean() if col in dfv.columns else None
        st.metric("Average Score", f"{avg:.2f}/5" if avg and not np.isnan(avg) else "N/A")
    with c3:
        n = dfv[col].dropna().shape[0] if col in dfv.columns else 0
        st.metric("Responses", f"{n:,}")

    st.markdown("---")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(dist_chart(dfv, col,
            "Score Distribution – Changed Understanding", ACCENT),
            use_container_width=True)
    with r1c2:
        st.plotly_chart(bar_by_group(dfv, col, "gender",
            "Avg Score by Gender", GREEN),
            use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(bar_by_group(dfv, col, "ethnicity",
            "Avg Score by Ethnicity", ORANGE),
            use_container_width=True)
    with r2c2:
        st.plotly_chart(bar_by_group(dfv, col, "age",
            "Avg Score by Age Group", PURPLE),
            use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(bar_by_group(dfv, col, "disability",
            "Avg Score by Disability", "#E07B6A"),
            use_container_width=True)
    with r3c2:
        st.plotly_chart(bar_by_group(dfv, col, "learning_difficulty",
            "Avg Score by Learning Difficulty", "#6AB8B8"),
            use_container_width=True)

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        st.plotly_chart(bar_by_group(dfv, col, "sexuality",
            "Avg Score by Sexuality", "#A08EC2"),
            use_container_width=True)
    with r4c2:
        st.plotly_chart(bar_by_group(dfv, col, "vulnerable_group",
            "Avg Score by Vulnerable / Not Vulnerable Group", "#D4A05A"),
            use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — LMK Leader Rating
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("How would you rate your LMK leader today?")
    col = "leader_rating"

    c1, c2, c3 = st.columns(3)
    with c1:
        pct = pct_agree(dfv[col]) if col in dfv.columns else None
        st.metric("Rated 4 or 5 out of 5", f"{pct}%" if pct else "N/A")
    with c2:
        avg = dfv[col].mean() if col in dfv.columns else None
        st.metric("Average Rating", f"{avg:.2f}/5" if avg and not np.isnan(avg) else "N/A")
    with c3:
        n = dfv[col].dropna().shape[0] if col in dfv.columns else 0
        st.metric("Responses", f"{n:,}")

    st.markdown("---")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(dist_chart(dfv, col,
            "Leader Rating Distribution", PURPLE),
            use_container_width=True)
    with r1c2:
        st.plotly_chart(bar_by_group(dfv, col, "gender",
            "Avg Rating by Gender", GREEN),
            use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(bar_by_group(dfv, col, "ethnicity",
            "Avg Rating by Ethnicity", ORANGE),
            use_container_width=True)
    with r2c2:
        st.plotly_chart(bar_by_group(dfv, col, "age",
            "Avg Rating by Age Group", ACCENT),
            use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(bar_by_group(dfv, col, "disability",
            "Avg Rating by Disability", "#E07B6A"),
            use_container_width=True)
    with r3c2:
        st.plotly_chart(bar_by_group(dfv, col, "learning_difficulty",
            "Avg Rating by Learning Difficulty", "#6AB8B8"),
            use_container_width=True)

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        st.plotly_chart(bar_by_group(dfv, col, "sexuality",
            "Avg Rating by Sexuality", "#A08EC2"),
            use_container_width=True)
    with r4c2:
        st.plotly_chart(bar_by_group(dfv, col, "vulnerable_group",
            "Avg Rating by Vulnerable / Not Vulnerable Group", "#D4A05A"),
            use_container_width=True)


# ═══════════════════════════════════════════════════════════════
# TAB 4 — Know Where to Get Help
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("(For 10 Signs and Delving Deeper modules only) I know where to go for "
                 "help or advice if either myself or a friend is in an unhealthy or abusive "
                 "situation in a relationship.")
    col = "know_where_to_go"

    # Highlight relevant modules
    relevant_modules = ["10 Signs", "Delving Deeper"]
    dfv_filtered = dfv.copy()
    # If session name / module info is available, we can further narrow
    # (keep all rows if session module info isn't on survey rows)

    c1, c2, c3 = st.columns(3)
    with c1:
        pct = pct_agree(dfv_filtered[col]) if col in dfv_filtered.columns else None
        st.metric("Agreed / Strongly Agreed (4-5)", f"{pct}%" if pct else "N/A")
    with c2:
        avg = dfv_filtered[col].mean() if col in dfv_filtered.columns else None
        st.metric("Average Score", f"{avg:.2f}/5" if avg and not np.isnan(avg) else "N/A")
    with c3:
        n = dfv_filtered[col].dropna().shape[0] if col in dfv_filtered.columns else 0
        st.metric("Responses", f"{n:,}")

    # Module note
    module_note = ", ".join(sel_module) if sel_module else "All (filter by Module sidebar for 10 Signs / Delving Deeper)"
    st.info(f"ℹ️ Module filter active: **{module_note}**")

    st.markdown("---")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(dist_chart(dfv_filtered, col,
            "Score Distribution – Know Where to Get Help", GREEN),
            use_container_width=True)
    with r1c2:
        st.plotly_chart(bar_by_group(dfv_filtered, col, "gender",
            "Avg Score by Gender", ACCENT),
            use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(bar_by_group(dfv_filtered, col, "ethnicity",
            "Avg Score by Ethnicity", ORANGE),
            use_container_width=True)
    with r2c2:
        st.plotly_chart(bar_by_group(dfv_filtered, col, "age",
            "Avg Score by Age Group", PURPLE),
            use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(bar_by_group(dfv_filtered, col, "disability",
            "Avg Score by Disability", "#E07B6A"),
            use_container_width=True)
    with r3c2:
        st.plotly_chart(bar_by_group(dfv_filtered, col, "learning_difficulty",
            "Avg Score by Learning Difficulty", "#6AB8B8"),
            use_container_width=True)

    r4c1, r4c2 = st.columns(2)
    with r4c1:
        st.plotly_chart(bar_by_group(dfv_filtered, col, "sexuality",
            "Avg Score by Sexuality", "#A08EC2"),
            use_container_width=True)
    with r4c2:
        st.plotly_chart(bar_by_group(dfv_filtered, col, "vulnerable_group",
            "Avg Score by Vulnerable / Not Vulnerable Group", "#D4A05A"),
            use_container_width=True)
