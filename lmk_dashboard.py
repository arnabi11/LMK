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

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #1E2A3A; }
[data-testid="stSidebar"] * { color: #E8EDF2 !important; }
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSelectbox label { color: #A8C4DC !important; font-weight: 600; }
[data-testid="stSidebar"] hr { border-color: #3a506b; }
.block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ────────────────────────────────────────────
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


# ── Load & cache data ─────────────────────────────────────────
@st.cache_data
def load_data():
    # ── UPDATE THESE PATHS to point at your CSVs ──────────────
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

    # Survey renames — exact column names from the real CSV
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


# ── Merge session metadata onto survey rows ───────────────────
def enrich(dfs, dfv):
    meta = dfs[["record_id", "vulnerable_group", "pct_school_meals", "module"]].copy()
    meta = meta.rename(columns={"record_id": "session_id",
                                "module": "session_module"})
    return dfv.merge(meta, on="session_id", how="left", suffixes=("", "_sess"))


# ─────────────────────────────────────────────────────────────
# CHART HELPERS
# ─────────────────────────────────────────────────────────────
BASE_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    margin=dict(t=44, b=20, l=10, r=10),
    height=340, font=dict(size=12),
)


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
    fig = go.Figure(go.Pie(
        labels=labels, values=values,
        hole=0.50,
        marker_colors=colors[:len(labels)],
        textinfo="percent+label",
        sort=False,
    ))
    fig.update_layout(**BASE_LAYOUT, title=title, showlegend=False)
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

    fig.update_layout(
        **BASE_LAYOUT,
        title=title,
        barmode="stack",
        yaxis=dict(title="%", range=[0, 100]),
        xaxis=dict(title="", tickangle=-20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=10)),
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

    fig = px.bar(agg, x="% Positive", y=group_col, orientation="h",
                 title=title, color_discrete_sequence=[color],
                 range_x=[0, 105],
                 text=agg.apply(lambda r: f"{r['% Positive']}% (n={r['n']})", axis=1))
    fig.update_traces(textposition="outside", marker_line_width=0)
    fig.update_layout(**BASE_LAYOUT, xaxis_title="% Positive Response", yaxis_title="")
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
    labels = ["0–20%", "21–40%", "41–60%", "61–80%", "81–100%"]
    dfv2["fsm_bucket"] = pd.cut(dfv2["_fsm"], bins=bins, labels=labels, right=False)
    return pct_positive_bar(dfv2, col, "fsm_bucket", positive_set, color, title)


def empty_fig(msg="No data available"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(size=13, color="#aaa"))
    fig.update_layout(**BASE_LAYOUT)
    return fig


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────
dfs = df_sessions_raw.copy()
if sel_borough: dfs = dfs[dfs["borough"].isin(sel_borough)]
if sel_year:    dfs = dfs[dfs["academic_year"].isin(sel_year)]
if sel_module:  dfs = dfs[dfs["module"].isin(sel_module)]
if sel_orgtype and "org_type" in dfs.columns:
    dfs = dfs[dfs["org_type"].isin(sel_orgtype)]
if sel_vgroup and "vulnerable_group" in dfs.columns:
    dfs = dfs[dfs["vulnerable_group"].isin(sel_vgroup)]
if sel_fsm is not None and "pct_school_meals" in dfs.columns:
    dfs["_fsm_n"] = pd.to_numeric(dfs["pct_school_meals"], errors="coerce")
    dfs = dfs[dfs["_fsm_n"].between(sel_fsm[0], sel_fsm[1]) | dfs["_fsm_n"].isna()]

dfv = df_survey_raw[df_survey_raw["session_id"].isin(dfs["record_id"])].copy()
dfv = enrich(dfs, dfv)

if sel_gender     and "gender"        in dfv.columns: dfv = dfv[dfv["gender"].isin(sel_gender)]
if sel_ethnicity  and "ethnicity"     in dfv.columns: dfv = dfv[dfv["ethnicity"].isin(sel_ethnicity)]
if sel_age        and "age"           in dfv.columns: dfv = dfv[dfv["age"].isin(sel_age)]
if sel_disability and "disability"    in dfv.columns: dfv = dfv[dfv["disability"].isin(sel_disability)]
if sel_sexuality  and "sexuality"     in dfv.columns: dfv = dfv[dfv["sexuality"].isin(sel_sexuality)]
if sel_nd         and "neurodivergent" in dfv.columns: dfv = dfv[dfv["neurodivergent"].isin(sel_nd)]


# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# REUSABLE TAB BODY
# ─────────────────────────────────────────────────────────────
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

    # Row 1 — Overall donut | by Gender (stacked)
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            donut_chart(series, order, colors, "Overall Response Breakdown"),
            use_container_width=True)
    with c2:
        st.plotly_chart(
            stacked_bar(dfv, col, "gender", order, colors, "Response by Gender"),
            use_container_width=True)

    # Row 2 — % positive by Ethnicity | by Age (stacked)
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

    # Row 3 — Disability | Neurodivergent / Learning Difficulty
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

    # Row 4 — Sexuality | Vulnerable group
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

    # Row 5 — % Free School Meals (full width)
    st.plotly_chart(
        fsm_bar(dfs, dfv, col, positive_set, GREEN,
                "% Positive by Free School Meals Eligibility (session-level)"),
        use_container_width=True)


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🌟 Workshop Usefulness",
    "🧠 Changed Understanding",
    "⭐ LMK Leader Rating",
    "🆘 Know Where to Get Help",
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
