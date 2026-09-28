# 📊 LMK Impact Dashboard — Streamlit Application

A comprehensive multi-tab impact analysis dashboard for LMK (Let Me Know) workshop sessions, combining survey response visualisation, machine learning pattern discovery, SHAP feature importance analysis, and free-text word analysis across academic years 2023/24 and 2024/25.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.35+-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## 📋 Features

### 🌟 Workshop Usefulness Tab
- Donut chart: overall response breakdown (Definitely / Probably / Not sure / Not really / No)
- Stacked 100% bar charts broken down by Gender, Ethnicity, Age, Disability, Neurodivergent/Learning Difficulty, Sexuality, and Vulnerable Group
- % Positive horizontal bar charts by Ethnicity and Free School Meals eligibility band
- KPI strip: % positive rate, most common response, total response count

### 🧠 Changed Understanding Tab
- Same chart suite as Workshop Usefulness, using the Agree/Disagree response scale
- Combines `Learnt something new about healthy and unhealthy behaviours` and `Changed understanding of healthy and unhealthy behaviours` columns across modules

### ⭐ LMK Leader Rating Tab
- Donut, stacked bars and % positive charts for the leader rating question (Excellent / Good / OK / Poor / Very poor)
- Breakdown by all demographic and session-level filters

### 🆘 Know Where to Get Help Tab
- Charts for the *10 Signs* and *Delving Deeper* module question on knowing where to seek help
- Uses Agree/Disagree scale; filtered to relevant modules via the sidebar Module filter

### 🔎 Pattern Analysis Tab (SHAP + ML)
Seven integrated analysis sections:

**Part 1 — SHAP Analysis per Outcome Question**
- GradientBoostingClassifier trained on each of the 4 survey questions
- SHAP Feature Importance bar chart (mean |SHAP| per feature)
- SHAP Beeswarm plot showing direction and magnitude for each respondent
- SHAP Dependence plots for the top 2 most important features
- Auto-generated 2-line interpretation below every chart
- 3-fold CV AUC score per model

**Part 2 — Pairwise Association Heatmap**
- Cramér's V computed for every pair of demographic + session features
- Full association heatmap with ranked table (Weak / Moderate / Strong)
- Plain-English highlights for pairs above configurable threshold

**Part 3 — Age × Borough × Gender Patterns**
- Faceted heatmaps showing % positive response for each Age/Borough combination, tabbed by Gender
- Top-8 most associated feature pairs as % positive heatmaps
- Per-heatmap best/worst cell callout boxes

**Part 4 — SHAP Interaction Effects**
- SHAP interaction values heatmap (joint feature effects beyond individual contributions)
- Top 5 interacting feature pairs ranked

**Part 5 — Cross-Question Feature Importance**
- Grouped horizontal bar comparing SHAP importance across all 4 questions simultaneously
- Interactive feature checkboxes (select/deselect any feature, Select All / Clear All)
- Auto-ranked top features averaged across all questions

**Part 6 — ML-Driven Key Pattern Discovery**
- Decision Tree leaf rules: plain-English if/then conditions → % positive outcome table
- Random Forest co-importance matrix heatmap + top 8 co-important feature pairs
- Sankey flow diagram: top-2 RF features → Positive / Negative outcome
- Bubble co-occurrence chart: frequency vs % positive per feature combination
- 5a: Combination heatmap (top-2 feature value pairs)
- 5b: Pattern summary table (all meaningful co-occurrences ranked)

### 📝 Word Analysis Tab
Four sub-tabs (one per outcome question), each showing:
- **Word Cloud** — visual frequency map of free-text responses (requires `wordcloud`)
- **Sentiment Bar** — rule-based Positive / Neutral / Negative response counts
- **Top N Words** — most frequent words across all responses (slider: 5–30)
- **Positive vs Non-Positive word bars** — side-by-side comparison of vocabulary
- **Top Bigrams** — most common word pairs across all responses
- **Differential Vocabulary** — words used more by positive vs non-positive responders
- **Raw Response Table** — expandable view of actual text per source column

Free-text columns used: *One thing LMK should change*, *One thing learned*, *One thing good/liked about workshop*

---

## 🗃️ Data

The dashboard expects two CSV files in the same directory as the script:

| File | Description |
|---|---|
| `Sessions 23-24 and 24-25.csv` | One row per workshop session — module, date, borough, org type, % free school meals, vulnerable group |
| `Impact surveys 23-24 and 24-25.csv` | One row per survey response — 4 outcome questions + demographic fields + free-text columns |

The two files are joined on `Session Record ID` → `Record ID` (sessions).

### Key columns used

**Sessions CSV**

| Column | Used as |
|---|---|
| Record ID | Join key |
| Module | Filter + feature |
| Academic year | Filter + feature |
| Org Borough | Filter + feature |
| Org Type / Org Sub Type | Filter + feature |
| Org % on school meals | FSM slider filter + pattern feature |
| Vulnerable group | Filter + feature |

**Impact Surveys CSV**

| Column | Used as |
|---|---|
| Workshop useful/helpful relationships | Outcome Q1 |
| Learnt something new / Changed understanding | Outcome Q2 |
| Leader rating | Outcome Q3 |
| Know who and where to go if worried | Outcome Q4 |
| Age / Gender / Ethnicity | Demographic filters + features |
| Disability / Sexuality / Neurodivergent | Demographic filters + features |
| One thing learned / good / LMK should change | Word analysis |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone or download the project folder
cd LMK

# (Recommended) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # Mac / Linux
venv\Scripts\activate          # Windows

# Install dependencies
pip install -r requirements.txt

# Run the main dashboard
streamlit run lmk_final_dashboard_final.py
```

Visit **http://localhost:8501**

Place both CSV files in the same directory as `lmk_final_dashboard_final.py` before launching.

---

## 📁 Project Structure

```
LMK/
├── lmk_final_dashboard_final.py     # Main dashboard (6 tabs, 3,156 lines)
├── lmk_shap_patterns_final.py       # Standalone SHAP + pattern analysis app (2,091 lines)
├── lmk_ml_clustering.py             # Standalone ML clustering app (859 lines)
│
├── Sessions 23-24 and 24-25.csv     # Session-level data
├── sample_data.numbers              # Sample data (Numbers format)
│
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

> **Note:** Files ending in `_old.py`, `_error.py`, `_bug.py`, `_err.py` are earlier development versions kept for reference. Only `lmk_final_dashboard_final.py` is the production file.

---

## 📦 Requirements

```
streamlit
pandas
numpy
plotly
scikit-learn
shap
scipy
matplotlib
wordcloud
umap-learn
```

Install all at once:

```bash
pip install streamlit pandas numpy plotly scikit-learn shap scipy matplotlib wordcloud umap-learn
```

> `wordcloud` and `umap-learn` are optional. The dashboard degrades gracefully if they are not installed — word cloud sections show an install prompt, and UMAP views are skipped.

---

## 🎛️ Sidebar Filters

All filters apply simultaneously across every tab and chart.

### Session-level filters
| Filter | Description |
|---|---|
| Borough | London borough of the organisation |
| Academic Year | 23/24 or 24/25 |
| Module | 10 Signs, Delving Deeper, Amended Module, etc. |
| Org Type | Education Setting, Community Organisation, etc. |
| Vulnerable Group | Session flagged as vulnerable group (Yes / No) |
| % Free School Meals | Slider — session-level FSM eligibility band (0–100%) |

### Pupil-level filters
| Filter | Description |
|---|---|
| Gender | Male, Female, Non-binary, Other, I prefer not to answer |
| Ethnicity | All ONS ethnic group categories |
| Age | Pupil age group |
| Disability | Disability status |
| Sexuality | Sexuality category |
| Neurodivergent / Learning Difficulty | Neurodivergent status |

---

## 🤖 Machine Learning Methods

| Technique | Where used | Purpose |
|---|---|---|
| GradientBoostingClassifier | Pattern Analysis — Part 1 | Predict positive vs non-positive response per question |
| TreeSHAP (via `shap`) | Pattern Analysis — Parts 1, 4 | Feature importance + interaction values |
| RandomForestClassifier | Pattern Analysis — Part 6 | Co-importance matrix, Sankey, bubble chart |
| DecisionTreeClassifier | Pattern Analysis — Part 6 | Explicit if/then co-occurrence rules |
| K-Means | ML Clustering app | Respondent cluster discovery |
| PCA | ML Clustering app | 2D projection of clusters |
| UMAP | ML Clustering app | Non-linear 2D projection |
| IsolationForest | ML Clustering app | Anomaly / outlier detection |
| Cramér's V (χ²) | Pattern Analysis — Part 2 | Pairwise feature association strength |
| Silhouette analysis | ML Clustering app | Optimal cluster count selection |

**Positive response thresholds:**

| Question | Positive = |
|---|---|
| Workshop Useful | Definitely or Probably |
| Changed Understanding | Strongly agree or Agree |
| LMK Leader Rating | Excellent or Good |
| Know Where to Get Help | Strongly agree or Agree |

---

## ⚡ Performance

Filter application is cached with `@st.cache_data` — repeated identical filter combinations return in under 1 second without recomputing the session/survey merge. SHAP values are computed on first render and cached per question and filter combination.

---

## 🌗 Theme Support

The dashboard supports Streamlit's built-in **light and dark themes**. All chart backgrounds, axis labels, and insight boxes are styled to be readable in both modes:
- Charts use `paper_bgcolor="#FFFFFF"` (word analysis) or `plot_bgcolor="rgba(248,249,252,1)"` with transparent paper for other tabs
- Insight, warning, and interpretation boxes use `[data-theme="dark"]` CSS overrides with chained selectors to beat Streamlit's injected stylesheet specificity
- Word analysis charts wrap in a white card `<div>` so axis labels are always readable regardless of page theme

---

## 📊 Running the Standalone Apps

The project also includes two standalone Streamlit apps for deeper analysis:

```bash
# SHAP + pattern analysis (standalone version)
streamlit run lmk_shap_patterns_final.py

# ML clustering analysis
streamlit run lmk_ml_clustering.py
```

Both expect the same two CSV files in the same directory.

---

## 🔄 Version History

### v3.0 (Current)
- Added **Word Analysis tab** with 4 sub-tabs (word cloud, sentiment, bigrams, differential vocabulary)
- Added `@st.cache_data` filter caching — filter changes now load in under 3 seconds
- Fixed duplicate Streamlit element ID errors in word analysis loop
- Fixed all chart axis labels for dark theme (solid white `paper_bgcolor` on word analysis charts)
- Added `wa_chart()` wrapper: white card with shadow for word analysis charts in dark mode
- Added free-text column fuzzy fallback matching (handles minor CSV header variants)

### v2.0
- Added **Pattern Analysis tab** with SHAP, Cramér's V, RF co-importance, Decision Tree rules, Sankey, and bubble charts
- Added **ML Pattern Discovery** section (Part 6) with 5 chart types
- Feature checkbox selector for cross-question SHAP comparison
- All interpretation boxes use `color:inherit` inline styles for theme compatibility
- Legend moved below x-axis on stacked bar charts to fix title overlap
- Donut charts use `textposition="inside"` with per-slice threshold to prevent floating labels

### v1.0
- Initial 4-tab survey response dashboard (Workshop Usefulness, Changed Understanding, Leader Rating, Know Where to Get Help)
- Sidebar filters for all demographic and session-level variables
- Stacked bar, % positive bar, donut, and FSM bucket charts per question

---

**Version:** 3.0  
**Last Updated:** September 2026  
**Status:** Active ✅
