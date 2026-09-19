"""
Mental Health in Tech Survey - Streamlit dashboard
==================================================
Project type : Exploratory Data Analysis (EDA)
Contribution : Individual
Dataset      : Open Sourcing Mental Illness (OSMI) 2014 survey  ->  survey.csv

Run it
------
    pip install -U streamlit pandas numpy plotly
    streamlit run app.py          (keep survey.csv in the same folder as app.py)

Needs Streamlit 1.36 or newer (uses st.navigation for real multi-page navigation).

Optional: for a perfect match with the colours used here, create
.streamlit/config.toml next to app.py with:

    [theme]
    base = "light"
    primaryColor = "#0E8C8C"
    backgroundColor = "#F5F7FB"
    secondaryBackgroundColor = "#EAEFF7"
    textColor = "#14213D"
"""

import io
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# 1. PAGE CONFIG (must be the first Streamlit call)
# =========================================================
st.set_page_config(
    page_title="Mental Health in Tech | Survey Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

if not hasattr(st, "navigation"):
    st.error(
        "This dashboard needs Streamlit 1.36 or newer. "
        "Run `pip install -U streamlit` and start the app again."
    )
    st.stop()

# =========================================================
# 2. DESIGN TOKENS
# =========================================================
INK = "#14213D"
INK_SOFT = "#2A3550"
MUTED = "#5B6780"
TEAL = "#0E8C8C"
TEAL_DARK = "#0A5F63"
TEAL_LIGHT = "#8CCFCF"
AMBER = "#F2A93B"
SAND = "#F7D08A"
SLATE = "#93A1BC"
SLATE_LIGHT = "#C9D2E3"
BERRY = "#B83A5B"
ROSE = "#D98099"
BG = "#F5F7FB"
LINE = "#E2E8F1"

FONT = "'Figtree', 'Segoe UI', system-ui, -apple-system, sans-serif"
HEAD_FONT = "'Bricolage Grotesque', 'Figtree', 'Segoe UI', system-ui, sans-serif"

MIN_N = 30  # groups smaller than this are flagged as unreliable

BLUE_RAMP = ["#CFE3F0", "#A5C8E0", "#7BABD0", "#528DBF", "#2F6FA8", "#14497D"]
QUALITATIVE = ["#0E8C8C", "#3F6C9E", "#F2A93B", "#7A5CA8", "#B83A5B", "#5FA8D3", "#8DA05A", "#93A1BC", "#C9D2E3"]

# Same answer -> same colour on every chart. Amber = "unsure", the gap this project is about.
CAT_COLORS = {
    "Yes": TEAL, "No": SLATE, "Maybe": SAND, "Don't know": AMBER, "Not sure": AMBER,
    "Some of them": SAND, "Not applicable": SLATE_LIGHT, "Other": SLATE_LIGHT,
}

# =========================================================
# 3. SURVEY METADATA (order of answers, labels, dictionary)
# =========================================================
YN = ["Yes", "No"]
DK = "Don't know"
YND = ["Yes", "No", DK]
YNM = ["No", "Maybe", "Yes"]
YSN = ["Yes", "Some of them", "No"]
YMN = ["Yes", "Maybe", "No"]
SIZE_ORDER = ["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
AGE_ORDER = ["18-25", "26-35", "36-45", "46-55", "56+"]
LEAVE_ORDER = ["Very easy", "Somewhat easy", "Don't know", "Somewhat difficult", "Very difficult"]

ORDER = {
    "treatment": YN, "family_history": YN, "remote_work": YN, "tech_company": YN,
    "self_employed": YN, "obs_consequence": YN,
    "Gender_clean": ["Male", "Female", "Other"], "Age_group": AGE_ORDER,
    "no_employees": SIZE_ORDER,
    "work_interfere": ["Never", "Rarely", "Sometimes", "Often", "Not applicable"],
    "benefits": YND, "wellness_program": YND, "seek_help": YND, "anonymity": YND,
    "mental_vs_physical": YND, "care_options": ["Yes", "No", "Not sure"],
    "leave": LEAVE_ORDER,
    "mental_health_consequence": YNM, "phys_health_consequence": YNM,
    "coworkers": YSN, "supervisor": YSN,
    "mental_health_interview": YMN, "phys_health_interview": YMN,
}

COLOR_MAPS = {
    "treatment": {"Yes": TEAL, "No": "#AEBBD3"},
    "Gender_clean": {"Male": "#3F6C9E", "Female": "#7A5CA8", "Other": SLATE},
    "leave": {"Very easy": TEAL_DARK, "Somewhat easy": TEAL_LIGHT, "Don't know": AMBER,
              "Somewhat difficult": ROSE, "Very difficult": BERRY},
    "work_interfere": {"Never": "#CFE8E8", "Rarely": TEAL_LIGHT, "Sometimes": TEAL,
                       "Often": TEAL_DARK, "Not applicable": SLATE_LIGHT},
    "no_employees": dict(zip(SIZE_ORDER, BLUE_RAMP)),
    "Age_group": dict(zip(AGE_ORDER, BLUE_RAMP[1:])),
}

LABEL = {
    "Age": "Age", "Age_group": "Age group", "Gender_clean": "Gender", "Country": "Country",
    "Country_grouped": "Country (top 8)", "state": "US state", "self_employed": "Self-employed",
    "family_history": "Family history of mental illness", "treatment": "Sought treatment",
    "work_interfere": "Work interference", "no_employees": "Company size",
    "remote_work": "Works remotely", "tech_company": "Tech company",
    "benefits": "Mental health benefits", "care_options": "Knows care options",
    "wellness_program": "Wellness program discussed", "seek_help": "Help resources provided",
    "anonymity": "Anonymity protected", "leave": "Ease of medical leave",
    "mental_health_consequence": "Fears consequences (mental health)",
    "phys_health_consequence": "Fears consequences (physical health)",
    "coworkers": "Would tell coworkers", "supervisor": "Would tell supervisor",
    "mental_health_interview": "Would raise in interview (mental)",
    "phys_health_interview": "Would raise in interview (physical)",
    "mental_vs_physical": "Employer treats both equally",
    "obs_consequence": "Saw negative consequences at work",
}

# Columns used as "possible drivers" of treatment.
# work_interfere is left out on purpose: it is only asked of people who already report a condition,
# so it almost contains the answer (see the Treatment drivers page).
DRIVER_COLS = [
    "Age_group", "Gender_clean", "Country_grouped", "self_employed", "family_history", "no_employees",
    "remote_work", "tech_company", "benefits", "care_options", "wellness_program", "seek_help",
    "anonymity", "leave", "mental_health_consequence", "phys_health_consequence", "coworkers",
    "supervisor", "mental_health_interview", "phys_health_interview", "mental_vs_physical",
    "obs_consequence",
]

SUPPORT_COLS = ["benefits", "care_options", "wellness_program", "seek_help", "anonymity", "mental_vs_physical"]

DATA_DICT = [
    ("Timestamp", "Other", "Date and time the survey was filled", "Datetime"),
    ("Age", "Demographics", "Age of the respondent", "Number"),
    ("Gender", "Demographics", "Gender identity (free text, needs cleaning)", "Text"),
    ("Country", "Demographics", "Country of residence", "Text"),
    ("state", "Demographics", "US state or territory (US residents only)", "Text"),
    ("self_employed", "Employment", "Are you self-employed?", "Yes / No"),
    ("family_history", "Mental health", "Do you have a family history of mental illness?", "Yes / No"),
    ("treatment", "Mental health", "Have you sought treatment for a mental health condition? (target variable)", "Yes / No"),
    ("work_interfere", "Mental health", "If you have a mental health condition, does it interfere with your work?", "Never / Rarely / Sometimes / Often"),
    ("no_employees", "Employment", "How many employees does your company or organization have?", "Company size bands"),
    ("remote_work", "Employment", "Do you work remotely at least 50% of the time?", "Yes / No"),
    ("tech_company", "Employment", "Is your employer primarily a tech company?", "Yes / No"),
    ("benefits", "Support", "Does your employer provide mental health benefits?", "Yes / No / Don't know"),
    ("care_options", "Support", "Do you know the mental health care options your employer provides?", "Yes / No / Not sure"),
    ("wellness_program", "Support", "Has your employer discussed mental health as part of a wellness program?", "Yes / No / Don't know"),
    ("seek_help", "Support", "Does your employer provide resources to learn about mental health and seeking help?", "Yes / No / Don't know"),
    ("anonymity", "Support", "Is your anonymity protected if you use mental health or substance abuse treatment resources?", "Yes / No / Don't know"),
    ("leave", "Support", "How easy is it to take medical leave for a mental health condition?", "Very easy to Very difficult"),
    ("mental_health_consequence", "Attitudes", "Would discussing a mental health issue with your employer have negative consequences?", "Yes / No / Maybe"),
    ("phys_health_consequence", "Attitudes", "Would discussing a physical health issue with your employer have negative consequences?", "Yes / No / Maybe"),
    ("coworkers", "Attitudes", "Would you discuss a mental health issue with your coworkers?", "Yes / No / Some of them"),
    ("supervisor", "Attitudes", "Would you discuss a mental health issue with your direct supervisor(s)?", "Yes / No / Some of them"),
    ("mental_health_interview", "Attitudes", "Would you bring up a mental health issue with a potential employer in an interview?", "Yes / No / Maybe"),
    ("phys_health_interview", "Attitudes", "Would you bring up a physical health issue with a potential employer in an interview?", "Yes / No / Maybe"),
    ("mental_vs_physical", "Attitudes", "Do you feel your employer takes mental health as seriously as physical health?", "Yes / No / Don't know"),
    ("obs_consequence", "Attitudes", "Have you heard of or observed negative consequences for coworkers with mental health conditions?", "Yes / No"),
    ("comments", "Other", "Any additional notes or comments", "Free text"),
]

# =========================================================
# 4. GLOBAL STYLE
# =========================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,600;12..96,700&family=Figtree:wght@400;500;600;700&display=swap');

.stApp, [data-testid="stAppViewContainer"] { background: %BG%; color: %INK%; }
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.4rem; padding-bottom: 4rem; max-width: 1180px; }

.stApp, [data-testid="stSidebar"], button, input, textarea,
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td, [data-testid="stMarkdownContainer"] th,
button[data-baseweb="tab"], label { font-family: %FONT%; }
.kpi, .callout, .waffle-card, .page-head p, .section-sub, .panel-sub, .chip, .rec-tag, .hero-lead { font-family: %FONT%; }
[data-testid="stMarkdownContainer"] h1, [data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3, [data-testid="stMarkdownContainer"] h4 {
    font-family: %HEAD%; color: %INK%; letter-spacing: -0.01em; }
[data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] li { color: %INK_SOFT%; line-height: 1.6; }

[data-testid="stSidebar"] { background: #EAEFF7; border-right: 1px solid %LINE%; }
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
footer, #MainMenu { visibility: hidden; }

button[data-baseweb="tab"] { font-weight: 600; }
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p,
button[data-baseweb="tab"] p, [data-testid="stExpander"] summary p, [data-testid="stRadio"] label p { color: %INK_SOFT%; }
button[data-baseweb="tab"][aria-selected="true"] p { color: %TEAL_DARK%; }
[data-baseweb="tab-highlight"] { background-color: %TEAL% !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: %LINE%; border-radius: 12px; }

.page-head { margin: 0 0 1.2rem 0; }
.page-head h1 { font-family: %HEAD%; font-size: 2.15rem; line-height: 1.15; margin: 0 0 .35rem 0; color: %INK%; }
.page-head p { color: %MUTED%; font-size: 1.05rem; max-width: 64ch; margin: 0; line-height: 1.55; }

.hero-title { font-family: %HEAD%; font-size: 2.9rem; line-height: 1.08; margin: 0 0 .8rem 0; color: %INK%; font-weight: 700; }
.hero-lead { color: %MUTED%; font-size: 1.12rem; line-height: 1.55; max-width: 52ch; margin: 0 0 1.1rem 0; }
.chip { display: inline-block; padding: 4px 12px; border: 1px solid %LINE%; border-radius: 999px;
        font-size: .85rem; background: #fff; color: %INK_SOFT%; margin: 0 6px 6px 0; }

.waffle-card { background: #fff; border: 1px solid %LINE%; border-radius: 12px; padding: 22px 24px; }
.waffle { display: grid; grid-template-columns: repeat(10, 1fr); gap: 8px; max-width: 330px; margin: 0 auto 14px auto; }
.waffle i { display: block; aspect-ratio: 1 / 1; border-radius: 50%; background: #D5DBE8; }
.waffle i.on { background: %TEAL%; }
.waffle-cap { font-size: 1rem; margin: 0 0 4px 0; color: %INK%; }
.waffle-note { font-size: .82rem; color: %MUTED%; margin: 0; }

.section-title { font-family: %HEAD%; font-size: 1.4rem; font-weight: 700; margin: 1.8rem 0 .15rem 0; color: %INK%; }
.section-sub { color: %MUTED%; margin: 0 0 .9rem 0; font-size: .97rem; max-width: 80ch; }
.panel-title { font-weight: 700; font-size: 1rem; margin: 0; color: %INK%; }
.panel-sub { color: %MUTED%; font-size: .85rem; margin: 0 0 .3rem 0; }

.kpi { background: #fff; border: 1px solid %LINE%; border-radius: 12px; padding: 16px 18px; height: 100%; }
.kpi-value { font-family: %HEAD%; font-size: 2rem; font-weight: 700; line-height: 1.1; color: %INK%; }
.kpi-label { font-size: .92rem; font-weight: 600; margin-top: 6px; color: %INK_SOFT%; }
.kpi-note { font-size: .82rem; color: %MUTED%; margin-top: 4px; line-height: 1.4; }
.kpi-teal { border-top: 4px solid %TEAL%; }
.kpi-amber { border-top: 4px solid %AMBER%; }
.kpi-berry { border-top: 4px solid %BERRY%; }
.kpi-slate { border-top: 4px solid %SLATE%; }

.callout { background: #fff; border: 1px solid %LINE%; border-left: 4px solid %SLATE%; border-radius: 6px;
           padding: 12px 16px; margin: 8px 0 16px 0; }
.callout-finding { border-left-color: %TEAL%; }
.callout-gap { border-left-color: %AMBER%; }
.callout-risk { border-left-color: %BERRY%; }
.callout-title { font-weight: 700; font-size: .95rem; margin-bottom: 2px; color: %INK%; }
.callout-body { font-size: .93rem; line-height: 1.55; color: %INK_SOFT%; }

.rec-tag { display: inline-block; font-size: .78rem; font-weight: 600; padding: 2px 10px; border-radius: 999px;
           background: #E3F3F3; color: %TEAL_DARK%; margin-right: 6px; }
.rec-tag-amber { background: #FDF0D8; color: #8A5A0B; }
.rec-tag-slate { background: #E6EBF3; color: %INK_SOFT%; }
.rec-title { font-family: %HEAD%; font-size: 1.2rem; font-weight: 700; margin: 4px 0 6px 0; color: %INK%; }
</style>
"""
for _k, _v in {
    "%BG%": BG, "%INK%": INK, "%INK_SOFT%": INK_SOFT, "%MUTED%": MUTED, "%TEAL%": TEAL,
    "%TEAL_DARK%": TEAL_DARK, "%AMBER%": AMBER, "%SLATE%": SLATE, "%BERRY%": BERRY,
    "%LINE%": LINE, "%FONT%": FONT, "%HEAD%": HEAD_FONT,
}.items():
    CSS = CSS.replace(_k, _v)
st.markdown(CSS, unsafe_allow_html=True)

# =========================================================
# 5. DATA LOADING AND CLEANING (same rules as the EDA notebook)
# =========================================================
try:
    _HERE = Path(__file__).resolve().parent
except NameError:
    _HERE = Path.cwd()
DATA_CANDIDATES = [_HERE / "survey.csv", Path.cwd() / "survey.csv"]


@st.cache_data(show_spinner="Loading survey data...")
def read_csv_path(path):
    return pd.read_csv(path)


@st.cache_data(show_spinner="Reading uploaded file...")
def read_csv_bytes(raw_bytes):
    return pd.read_csv(io.BytesIO(raw_bytes))


def get_raw():
    """Load survey.csv from disk; if it is missing, ask for an upload instead of crashing."""
    for p in DATA_CANDIDATES:
        if p.exists():
            return read_csv_path(str(p))
    st.markdown(
        '<div class="page-head"><h1>Upload survey.csv to begin</h1>'
        "<p>survey.csv was not found next to app.py. Upload it here, or copy it into the app folder "
        "and refresh the page.</p></div>",
        unsafe_allow_html=True,
    )
    up = st.file_uploader("Survey file", type="csv")
    if up is None:
        st.stop()
    return read_csv_bytes(up.getvalue())


def clean_gender(gender):
    """Map the free-text Gender answers into Male / Female / Other (identical to the EDA notebook)."""
    if pd.isna(gender):
        return "Other"
    g = str(gender).strip().lower()
    male_keywords = ["male", "m", "male-ish", "maile", "cis male", "mal", "male (cis)", "make", "man",
                     "msle", "mail", "malr", "cis man", "male ", "guy"]
    female_keywords = ["female", "f", "woman", "female ", "cis female", "femake", "female (cis)",
                       "femail", "trans-female", "trans woman", "female (trans)"]
    if g in male_keywords or ("male" in g and "female" not in g and "trans" not in g):
        return "Male"
    if g in female_keywords or "female" in g or g == "f":
        return "Female"
    return "Other"


@st.cache_data(show_spinner="Cleaning data...")
def prepare(raw):
    """Return (clean dataframe, gender mapping table, removed age values)."""
    d = raw.copy()
    d["Age"] = pd.to_numeric(d["Age"], errors="coerce")

    keep = d["Age"].between(18, 75)
    removed_ages = d.loc[~keep, "Age"]
    d = d.loc[keep].copy()

    entry = d["Gender"].astype(str).str.strip().str.lower()
    mapped = d["Gender"].apply(clean_gender)
    gender_map = (
        pd.DataFrame({"Entry": entry, "Mapped to": mapped})
        .groupby(["Mapped to", "Entry"]).size().reset_index(name="Respondents")
        .sort_values(["Mapped to", "Respondents"], ascending=[True, False])
        .reset_index(drop=True)
    )

    d["Gender_clean"] = mapped
    d["self_employed"] = d["self_employed"].fillna("No")
    d["work_interfere"] = d["work_interfere"].fillna("Not applicable")
    d["Age_group"] = pd.cut(d["Age"], bins=[17, 25, 35, 45, 55, 75], labels=AGE_ORDER).astype(str)
    top_countries = d["Country"].value_counts().nlargest(8).index
    d["Country_grouped"] = d["Country"].where(d["Country"].isin(top_countries), "Other")
    d = d.drop(columns=["comments", "Timestamp", "Gender"], errors="ignore").reset_index(drop=True)
    return d, gender_map, removed_ages


RAW = get_raw()
DFC, GENDER_MAP, REMOVED_AGES = prepare(RAW)

# =========================================================
# 6. SMALL STATS HELPERS
# =========================================================
def pct(x, digits=1):
    return "n/a" if x is None or pd.isna(x) else f"{x:.{digits}f}%"


def share(data, col, val):
    """Percentage of rows in `data` where col == val."""
    return float((data[col] == val).mean() * 100) if len(data) else float("nan")


def order_for(col, data):
    """Sensible display order for a column (survey order if known, otherwise most frequent first)."""
    present = data[col].dropna().astype(str).unique().tolist()
    if col in ("Country", "Country_grouped"):
        counts = data[col].astype(str).value_counts()
        return [c for c in counts.index if c != "Other"] + (["Other"] if "Other" in present else [])
    base = ORDER.get(col)
    if base is None:
        return data[col].astype(str).value_counts().index.tolist()
    return [o for o in base if o in present] + sorted(p for p in present if p not in base)


def wilson(k, n, z=1.96):
    """95% Wilson confidence interval for a proportion, returned in percent."""
    k = np.asarray(k, dtype=float)
    n = np.asarray(n, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denom = 1 + z * z / n
        centre = (p + z * z / (2 * n)) / denom
        half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half) * 100, (centre + half) * 100


def rate_table(data, col, target="treatment", pos="Yes", order=None):
    """Share of respondents with target == pos for every answer in `col`, plus n and a 95% interval."""
    tmp = pd.DataFrame({"Answer": data[col].astype(str), "_t": (data[target] == pos).astype(int)})
    g = tmp.groupby("Answer")["_t"].agg(["size", "sum"]).rename(columns={"size": "n", "sum": "treated"})
    g = g.reindex(order or order_for(col, data)).dropna().reset_index()
    g.columns = ["Answer", "n", "treated"]
    g["n"] = g["n"].astype(int)
    g["treated"] = g["treated"].astype(int)
    g["rate"] = g["treated"] / g["n"] * 100
    g["lo"], g["hi"] = wilson(g["treated"], g["n"])
    return g


def cramers_v(x, y):
    """Strength of association between two categorical variables (0 = none, 1 = perfect)."""
    ct = pd.crosstab(x, y)
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return float("nan")
    obs = ct.values.astype(float)
    n = obs.sum()
    exp = np.outer(obs.sum(axis=1), obs.sum(axis=0)) / n
    chi2 = ((obs - exp) ** 2 / exp).sum()
    return float(np.sqrt(chi2 / (n * (min(ct.shape) - 1))))


def cmap_for(col, cats):
    """Colour for every answer in `cats`, consistent across the whole app."""
    m = COLOR_MAPS.get(col, {})
    out, spare = {}, itertools.cycle(QUALITATIVE)
    for c in cats:
        out[c] = m.get(c) or CAT_COLORS.get(c) or next(spare)
    return out


def _text_on(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#FFFFFF"


# =========================================================
# 7. CHART HELPERS
# =========================================================
_KEYS = itertools.count()


def style(fig, height=340):
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=10, b=6),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=INK),
        hoverlabel=dict(font_family=FONT, bgcolor="#FFFFFF", font_color=INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
        bargap=0.28,
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE, tickfont_size=12, title_font_size=12, automargin=True)
    fig.update_yaxes(gridcolor=LINE, zeroline=False, tickfont_size=12, title_font_size=12, automargin=True)
    return fig


def plot(fig):
    st.plotly_chart(
        fig, key=f"chart_{next(_KEYS)}",
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]},
    )


def show_table(data, height=None):
    kw = {"hide_index": True}
    if height:
        kw["height"] = height
    try:
        st.dataframe(data, width="stretch", **kw)
    except Exception:  # older Streamlit versions
        st.dataframe(data, use_container_width=True, **kw)


def count_bar(data, col, height=320):
    """Number of respondents per answer, with the share printed on each bar."""
    order = order_for(col, data)
    vc = data[col].astype(str).value_counts().reindex(order).fillna(0).astype(int)
    total = max(int(vc.sum()), 1)
    cm = cmap_for(col, order)
    fig = go.Figure(go.Bar(
        x=list(vc.index), y=vc.values, marker_color=[cm[a] for a in vc.index],
        text=[f"{v / total * 100:.1f}%" for v in vc.values], textposition="outside", cliponaxis=False,
        customdata=vc.values / total * 100,
        hovertemplate="<b>%{x}</b><br>%{y:,} respondents (%{customdata:.1f}%)<extra></extra>",
    ))
    fig.update_yaxes(title="Respondents", range=[0, max(int(vc.max()), 1) * 1.2])
    return style(fig, height)


def rate_bar(data, col, height=None, target="treatment", pos="Yes", ref_label="All respondents"):
    """% who sought treatment for each answer. Whiskers = 95% confidence range; lighter bars = n < 30."""
    t = rate_table(data, col, target, pos)
    overall = share(data, target, pos)
    horizontal = len(t) > 6 or (len(t) > 0 and t["Answer"].str.len().max() > 16)
    labels = [f"{r:.1f}% (n={n:,})" for r, n in zip(t["rate"], t["n"])]
    colors = [TEAL if n >= MIN_N else TEAL_LIGHT for n in t["n"]]
    err = dict(type="data", symmetric=False, array=(t["hi"] - t["rate"]).values,
               arrayminus=(t["rate"] - t["lo"]).values, color=INK_SOFT, thickness=1.2, width=4)
    hover = "<b>%{customdata[0]}</b><br>%{customdata[1]:.1f}% sought treatment<br>n = %{customdata[2]:,}<extra></extra>"
    cd = np.c_[t["Answer"], t["rate"], t["n"]]
    top = float(t["hi"].max()) if len(t) else 100
    if horizontal:
        fig = go.Figure(go.Bar(y=t["Answer"], x=t["rate"], orientation="h", marker_color=colors,
                               error_x=err, customdata=cd, hovertemplate=hover))
        fig.add_trace(go.Scatter(x=t["hi"], y=t["Answer"], mode="text", text=labels, textposition="middle right",
                                 hoverinfo="skip", showlegend=False, textfont=dict(size=12, color=INK)))
        fig.add_vline(x=overall, line_dash="dot", line_color=INK, opacity=0.6)
        fig.update_xaxes(range=[0, min(140, top * 1.55 + 10)], tickvals=[0, 20, 40, 60, 80, 100],
                         ticksuffix="%", title="Sought treatment")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        h = height or max(260, 64 + 46 * len(t))
    else:
        fig = go.Figure(go.Bar(x=t["Answer"], y=t["rate"], marker_color=colors,
                               error_y=err, customdata=cd, hovertemplate=hover))
        fig.add_trace(go.Scatter(x=t["Answer"], y=t["hi"], mode="text", text=labels, textposition="top center",
                                 hoverinfo="skip", showlegend=False, textfont=dict(size=12, color=INK)))
        fig.add_hline(y=overall, line_dash="dot", line_color=INK, opacity=0.6,
                      annotation_text=f"{ref_label} {overall:.1f}%", annotation_position="bottom right",
                      annotation_font_size=11, annotation_font_color=MUTED)
        fig.update_yaxes(range=[0, min(125, top * 1.22 + 6)], tickvals=[0, 20, 40, 60, 80, 100],
                         ticksuffix="%", title="Sought treatment")
        h = height or 340
    fig.update_layout(showlegend=False)
    return style(fig, h)


def crosstab_pct(data, row, col, rows=None, cols=None):
    ct = pd.crosstab(data[row].astype(str), data[col].astype(str), normalize="index") * 100
    rows = [r for r in (rows or order_for(row, data)) if r in ct.index]
    cols = [c for c in (cols or order_for(col, data)) if c in ct.columns]
    return ct.reindex(index=rows, columns=cols).fillna(0)


def stacked_pct(table, color_col=None, height=None):
    """100% stacked horizontal bars from a table (rows = groups, columns = answers, values = %)."""
    cm = cmap_for(color_col, list(table.columns))
    fig = go.Figure()
    for c in table.columns:
        vals = table[c].values
        fig.add_trace(go.Bar(
            y=list(table.index), x=vals, name=c, orientation="h", marker_color=cm[c],
            text=[f"{v:.0f}%" if v >= 6 else "" for v in vals], textposition="inside", insidetextanchor="middle",
            textfont=dict(color=_text_on(cm[c]), size=12),
            hovertemplate="<b>%{y}</b><br>" + c + ": %{x:.1f}%<extra></extra>",
        ))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(range=[0, 100], ticksuffix="%")
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    return style(fig, height or max(240, 70 + 46 * len(table)))


def treatment_grouped(data, col, height=320):
    """Counts of respondents per answer, split by treatment (like the EDA's grouped bar charts)."""
    order = order_for(col, data)
    ct = pd.crosstab(data[col].astype(str), data["treatment"]).reindex(order).fillna(0)
    fig = go.Figure()
    for tval in [v for v in YN if v in ct.columns]:
        fig.add_trace(go.Bar(
            x=list(ct.index), y=ct[tval].values, name="Sought treatment" if tval == "Yes" else "Did not seek treatment",
            marker_color=COLOR_MAPS["treatment"][tval],
            hovertemplate="<b>%{x}</b><br>%{y:,} respondents<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Respondents")
    return style(fig, height)


def heatmap(z, text=None, x=None, y=None, height=380, diverging=False, zmin=None, zmax=None, colorbar_title=""):
    scale = ([[0, BERRY], [0.5, "#F5F7FB"], [1, TEAL_DARK]] if diverging else [[0, "#EAF4F4"], [1, TEAL_DARK]])
    fig = go.Figure(go.Heatmap(
        z=z, x=x, y=y, colorscale=scale, zmin=zmin, zmax=zmax, xgap=3, ygap=3,
        text=text, texttemplate="%{text}" if text is not None else None,
        colorbar=dict(title=colorbar_title, thickness=12, outlinewidth=0),
        hovertemplate="%{y} / %{x}<br>%{z:.1f}<extra></extra>",
    ))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(side="top")
    return style(fig, height)


# =========================================================
# 8. UI HELPERS
# =========================================================
def page_head(title, lead=""):
    st.markdown(f'<div class="page-head"><h1>{title}</h1><p>{lead}</p></div>', unsafe_allow_html=True)


def section(title, sub=""):
    st.markdown(f'<div class="section-title">{title}</div><div class="section-sub">{sub}</div>', unsafe_allow_html=True)


def panel_title(title, sub=""):
    st.markdown(f'<p class="panel-title">{title}</p><p class="panel-sub">{sub}</p>', unsafe_allow_html=True)


def kpi(label, value, note="", tone="teal"):
    return (f'<div class="kpi kpi-{tone}"><div class="kpi-value">{value}</div>'
            f'<div class="kpi-label">{label}</div><div class="kpi-note">{note}</div></div>')


def kpi_row(items):
    cols = st.columns(len(items), gap="medium")
    for c, item in zip(cols, items):
        c.markdown(kpi(*item), unsafe_allow_html=True)


def callout(kind, title, body):
    """kind: finding (teal) | gap (amber) | risk (berry) | note (slate)."""
    st.markdown(
        f'<div class="callout callout-{kind}"><div class="callout-title">{title}</div>'
        f'<div class="callout-body">{body}</div></div>',
        unsafe_allow_html=True,
    )


def read_analysis(why, insight, impact):
    """The three questions the EDA asks under every chart, kept as a collapsible block."""
    with st.expander("Read the analysis"):
        st.markdown(f"**Why this chart.** {why}")
        st.markdown(f"**What it shows.** {insight}")
        st.markdown(f"**Why it matters.** {impact}")


def need_data():
    """Stop a page politely when the sidebar filters leave nothing (or too little) to analyse."""
    if len(DF) == 0:
        st.warning("No respondents match the current filters. Use **Reset filters** in the sidebar.")
        st.stop()
    if len(DF) < MIN_N:
        callout("risk", "Very small sample",
                f"Only {len(DF)} respondents match these filters. Percentages on this page can swing a lot "
                "with a handful of answers, so treat them as anecdotes.")


QUESTION = {row[0]: row[2] for row in DATA_DICT}


def rate_of(data, col, val):
    """% who sought treatment among respondents whose `col` answer equals `val`."""
    sub = data[data[col] == val]
    return share(sub, "treatment", "Yes") if len(sub) else float("nan")


def gap_sentence(data, col):
    """One-line, data-driven comparison of the highest and lowest treatment rate for a question."""
    t = rate_table(data, col)
    t = t[t["n"] >= MIN_N]
    if len(t) < 2:
        return "There are not enough respondents in each answer group (30+) to compare them reliably."
    hi = t.loc[t["rate"].idxmax()]
    lo = t.loc[t["rate"].idxmin()]
    return (f"{pct(hi['rate'])} of people answering '{hi['Answer']}' sought treatment, versus "
            f"{pct(lo['rate'])} of people answering '{lo['Answer']}' (a gap of {hi['rate'] - lo['rate']:.1f} points).")


def dont_know_share(data, col):
    return float(data[col].isin(["Don't know", "Not sure"]).mean() * 100) if len(data) else float("nan")


# =========================================================
# 9. PAGES - START
# =========================================================
def page_overview():
    d = DFC
    n = len(d)
    t = share(d, "treatment", "Yes")
    on = int(round(t))

    left, right = st.columns([1.45, 1], gap="large")
    with left:
        st.markdown(
            f'<h1 class="hero-title">What {n:,} people in tech said about mental health at work</h1>'
            '<p class="hero-lead">A walk through the 2014 OSMI survey: who has sought treatment, '
            "what employers offer, and how many people are left guessing.</p>"
            '<span class="chip">Exploratory data analysis</span>'
            '<span class="chip">Individual project</span>'
            '<span class="chip">OSMI survey, 2014</span>'
            f'<span class="chip">{len(RAW):,} rows in, {n:,} analysed</span>',
            unsafe_allow_html=True,
        )
    with right:
        dots = "".join(f'<i class="{"on" if i < on else ""}"></i>' for i in range(100))
        st.markdown(
            f'<div class="waffle-card"><div class="waffle">{dots}</div>'
            f'<p class="waffle-cap"><b>{on} of every 100</b> respondents have sought treatment '
            "for a mental health condition.</p>"
            '<p class="waffle-note">This is a self-selected online survey, so read it as a sample, '
            "not as a rate for all tech workers.</p></div>",
            unsafe_allow_html=True,
        )

    section("Three questions, three answers",
            "The project brief asks where treatment differs, what predicts it, and where employer support falls short.")
    c1, c2, c3 = st.columns(3, gap="medium")

    top4 = d["Country"].value_counts().head(4).index
    ct = rate_table(d[d["Country"].isin(top4)], "Country")
    fam_yes, fam_no = rate_of(d, "family_history", "Yes"), rate_of(d, "family_history", "No")
    anon_dk, ben_dk = dont_know_share(d, "anonymity"), dont_know_share(d, "benefits")

    with c1, st.container(border=True):
        st.markdown("**Does treatment differ by location?**")
        st.markdown(
            f"Among the four biggest countries, {ct['rate'].min():.0f}% to {ct['rate'].max():.0f}% "
            "sought treatment. The wide gaps only show up in countries with a handful of respondents."
        )
        st.page_link(PG["geo"], label="Open Geography")
    with c2, st.container(border=True):
        st.markdown("**What predicts treatment?**")
        st.markdown(
            f"Family history is the clearest signal: {fam_yes:.0f}% treated with it, "
            f"{fam_no:.0f}% without. Age and remote work barely matter."
        )
        st.page_link(PG["drivers"], label="Open Treatment drivers")
    with c3, st.container(border=True):
        st.markdown("**Where does employer support fall short?**")
        st.markdown(
            f"{anon_dk:.0f}% do not know if anonymity is protected and {ben_dk:.0f}% do not know "
            "if their employer offers benefits at all."
        )
        st.page_link(PG["support"], label="Open Workplace support")

    section("Headline numbers")
    fear = float(d["mental_health_consequence"].isin(["Yes", "Maybe"]).mean() * 100)
    kpi_row([
        ("Sought treatment", pct(t), "of survey respondents", "teal"),
        ("Have a family history", pct(share(d, "family_history", "Yes")), "of mental illness", "slate"),
        ("Fear career harm", pct(fear), "answered Yes or Maybe when asked about discussing mental health at work", "amber"),
        ("Saw others penalised", pct(share(d, "obs_consequence", "Yes")), "have seen or heard of negative consequences for a coworker", "berry"),
    ])

    section("The problem and the goal")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown(
            "**Problem.** Mental health issues are common in tech, but stigma, unclear support and fear of "
            "career damage keep people from asking for help. Employers lack data-backed answers on "
            "where the biggest gaps are, so wellness spending often misses."
        )
    with b:
        st.markdown(
            "**Goal.** Show where support falls short, what goes together with seeking treatment, and which "
            "low-cost changes are most likely to help. Every number on these pages is computed live from "
            "survey.csv."
        )

    section("How to use this dashboard")
    st.markdown(
        "- **Start** pages show the raw data and every cleaning decision.\n"
        "- **Analysis** pages answer the three questions. The sidebar filters apply to them.\n"
        "- Under each chart, open **Read the analysis** for why the chart was chosen, what it shows and why it matters.\n"
        "- Whiskers on treatment-rate bars are 95% confidence ranges. Lighter bars mean fewer than 30 respondents."
    )


def page_dataset():
    page_head("Data understanding", "A first look at the raw file: size, types, missing values and what each column means.")
    dup = int(RAW.duplicated().sum())
    miss_cells = RAW.isna().sum().sum() / RAW.size * 100
    kpi_row([
        ("Rows", f"{len(RAW):,}", "survey responses", "teal"),
        ("Columns", f"{RAW.shape[1]}", f"{RAW.select_dtypes('number').shape[1]} numeric, the rest text", "teal"),
        ("Duplicate rows", f"{dup}", "exact copies", "slate"),
        ("Missing cells", pct(miss_cells), "mostly the optional comments field", "amber"),
    ])
    st.write("")
    tab1, tab2, tab3, tab4 = st.tabs(["Preview", "Structure and missing values", "Data dictionary", "Value explorer"])

    with tab1:
        rows = st.select_slider("Rows to show", options=[5, 10, 25, 50, 100], value=10)
        show_table(RAW.head(rows))
        callout("gap", "Two columns need cleaning before any analysis",
                f"Age runs from {RAW['Age'].min():,.0f} to {RAW['Age'].max():,.0f}, and Gender holds "
                f"{RAW['Gender'].nunique()} different spellings. Both are fixed on the Data cleaning page.")

    with tab2:
        info = pd.DataFrame({
            "Column": RAW.columns,
            "Type": RAW.dtypes.astype(str).values,
            "Non-null": RAW.notna().sum().values,
            "Missing": RAW.isna().sum().values,
            "Missing %": (RAW.isna().mean() * 100).round(1).values,
            "Unique values": RAW.nunique().values,
        })
        show_table(info, height=420)
        miss = RAW.isna().sum()
        miss = miss[miss > 0].sort_values()
        a, b = st.columns(2, gap="medium")
        with a, st.container(border=True):
            panel_title("Missing values by column", "Share of rows left blank")
            fig = go.Figure(go.Bar(
                y=list(miss.index), x=(miss.values / len(RAW) * 100), orientation="h", marker_color=AMBER,
                text=[f"{v:,} rows" for v in miss.values], textposition="outside", cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>%{x:.1f}% missing<extra></extra>",
            ))
            fig.update_xaxes(range=[0, 115], ticksuffix="%")
            plot(style(fig, 300))
        with b, st.container(border=True):
            panel_title("Where the gaps sit", "Each amber mark is a blank answer; rows run top to bottom")
            fig = go.Figure(go.Heatmap(
                z=RAW.isna().values.astype(int), x=list(RAW.columns),
                colorscale=[[0, "#E4EAF4"], [1, AMBER]], showscale=False, hoverinfo="skip",
            ))
            fig.update_yaxes(autorange="reversed", showticklabels=False, gridcolor="rgba(0,0,0,0)")
            fig.update_xaxes(tickangle=-60, tickfont_size=10)
            plot(style(fig, 300))
        callout("finding", "What the file tells us",
                "Nearly every column is categorical; only Age is numeric. There are no duplicate rows. "
                "comments is mostly empty (optional free text), state is blank for people outside the US, "
                "and work_interfere is blank for people who did not report a condition. "
                "\"Don't know\" and \"Not sure\" answers are not missing data. They are real answers, and they turn out to be the main story.")

    with tab3:
        dd = pd.DataFrame(DATA_DICT, columns=["Column", "Group", "Question", "Answer type"])
        f1, f2 = st.columns([2, 1])
        q = f1.text_input("Search columns or questions", placeholder="for example: anonymity")
        groups = f2.multiselect("Group", sorted(dd["Group"].unique()), placeholder="All groups")
        if q:
            hay = dd["Column"] + " " + dd["Question"]
            dd = dd[hay.str.contains(q, case=False, regex=False)]
        if groups:
            dd = dd[dd["Group"].isin(groups)]
        show_table(dd, height=520)
        st.caption(f"{len(dd)} of {len(DATA_DICT)} columns shown. Question wording follows the survey.")

    with tab4:
        cols = [c for c in RAW.columns if c not in ("Timestamp", "comments")]
        col = st.selectbox("Column", cols, index=cols.index("Gender") if "Gender" in cols else 0)
        s = RAW[col]
        if pd.api.types.is_numeric_dtype(s):
            desc = s.describe().round(2).rename_axis("Statistic").reset_index(name="Value")
            outside = int((~s.between(18, 75)).sum())
            a, b = st.columns([1, 2], gap="large")
            with a:
                show_table(desc)
            with b:
                callout("gap", f"{outside} ages fall outside 18 to 75",
                        "Negative ages, single digits and values in the hundreds or billions are typing errors. "
                        "They stretch the mean and hide the real shape of the data.")
        else:
            vc = s.fillna("(blank)").astype(str).value_counts()
            st.caption(f"{s.nunique()} distinct answers" + (" (top 15 shown in the chart)" if len(vc) > 15 else ""))
            a, b = st.columns([1, 1.4], gap="large")
            with a:
                tbl = vc.rename_axis("Answer").reset_index(name="Respondents")
                show_table(tbl, height=420)
            with b, st.container(border=True):
                top = vc.head(15)
                fig = go.Figure(go.Bar(y=list(top.index), x=top.values, orientation="h", marker_color=TEAL,
                                       hovertemplate="<b>%{y}</b><br>%{x:,} respondents<extra></extra>"))
                fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
                plot(style(fig, max(260, 30 * len(top) + 40)))


def page_cleaning():
    page_head("Data cleaning", "Every change made to the raw file, with the reason and the effect. These are the same rules as the EDA notebook.")
    tabs = st.tabs(["Age", "Gender", "Missing values", "New columns", "Final dataset"])

    with tabs[0]:
        kpi_row([
            ("Lowest age in raw file", f"{RAW['Age'].min():,.0f}", "", "berry"),
            ("Highest age in raw file", f"{RAW['Age'].max():,.0f}", "", "berry"),
            ("Rows removed", f"{len(REMOVED_AGES)}", "ages outside 18 to 75", "amber"),
            ("Rows kept", f"{len(DFC):,}", f"{len(DFC) / len(RAW) * 100:.1f}% of the raw file", "teal"),
        ])
        vals = ", ".join(f"{v:,.0f}" for v in sorted(REMOVED_AGES.dropna()))
        callout("gap", "Values that were removed", vals or "None")
        st.markdown("**Rule.** Keep ages from 18 to 75, a realistic working-age range. Everything outside is a typing error, not a real respondent.")
        with st.container(border=True):
            panel_title("Age after cleaning", "Histogram with a box plot on top")
            fig = px.histogram(DFC, x="Age", nbins=20, marginal="box", color_discrete_sequence=[TEAL])
            fig.update_traces(marker_line_color="#fff", marker_line_width=1, selector=dict(type="histogram"))
            fig.update_yaxes(title_text="")
            fig.update_xaxes(title_text="Age")
            plot(style(fig, 340))
        st.code("df = df[(df['Age'] >= 18) & (df['Age'] <= 75)]", language="python")

    with tabs[1]:
        gc = DFC["Gender_clean"].value_counts()
        kpi_row([
            ("Different spellings", f"{RAW['Gender'].nunique()}", "in the raw Gender column", "amber"),
            ("Male", f"{gc.get('Male', 0):,}", pct(gc.get("Male", 0) / len(DFC) * 100), "slate"),
            ("Female", f"{gc.get('Female', 0):,}", pct(gc.get("Female", 0) / len(DFC) * 100), "slate"),
            ("Other", f"{gc.get('Other', 0):,}", pct(gc.get("Other", 0) / len(DFC) * 100), "berry"),
        ])
        st.write("")
        which = st.radio("Show entries mapped to", ["All", "Male", "Female", "Other"], horizontal=True)
        gm = GENDER_MAP if which == "All" else GENDER_MAP[GENDER_MAP["Mapped to"] == which]
        show_table(gm, height=380)
        callout("risk", "Judgement calls in this mapping",
                "Three hedged answers (\"male leaning androgynous\", \"ostensibly male, unsure what that really means\", "
                "\"something kinda male?\") end up in Male, and trans respondents who wrote \"trans woman\", "
                "\"trans-female\" or \"female (trans)\" end up in Female. "
                f"Only {gc.get('Other', 0)} respondents land in Other, so any percentage for that group is unreliable.")
        st.code(
            "def clean_gender(gender):\n"
            "    g = str(gender).strip().lower()\n"
            "    if g in male_keywords or ('male' in g and 'female' not in g and 'trans' not in g):\n"
            "        return 'Male'\n"
            "    elif g in female_keywords or 'female' in g or g == 'f':\n"
            "        return 'Female'\n"
            "    return 'Other'",
            language="python",
        )

    with tabs[2]:
        plan = [
            ("self_employed", "Filled with 'No'", "The overwhelming majority are not self-employed, and only a few rows are affected."),
            ("work_interfere", "Filled with 'Not applicable'", "The question is only shown to people who report a condition, so a blank most likely means no condition."),
            ("state", "Left blank", "Only US residents are asked, so a blank means 'not in the US'."),
            ("comments", "Column dropped", "Optional free text, mostly empty."),
        ]
        rows = []
        for col, how, why in plan:
            after = "dropped" if col not in DFC.columns else f"{int(DFC[col].isna().sum()):,}"
            rows.append({"Column": col, "Blank in raw file": f"{int(RAW[col].isna().sum()):,}",
                         "What was done": how, "Blank afterwards": after, "Why": why})
        show_table(pd.DataFrame(rows))
        na_rate = rate_of(DFC, "work_interfere", "Not applicable")
        never_rate = rate_of(DFC, "work_interfere", "Never")
        often_rate = rate_of(DFC, "work_interfere", "Often")
        callout("finding", "Why 'Not applicable' and not 'Never'",
                f"Only {pct(na_rate)} of the 'Not applicable' group sought treatment, which fits the idea that they have no condition. "
                "Filling those blanks with 'Never' would wrongly say they have a condition that never affects work. "
                f"It also matters later: {pct(often_rate)} of people who say it 'Often' interferes sought treatment versus {pct(never_rate)} for 'Never', "
                "so this column is left out of the driver ranking.")

    with tabs[3]:
        a, b = st.columns(2, gap="large")
        with a:
            st.markdown("**Age_group** (from Age)")
            ag = DFC["Age_group"].value_counts().reindex(AGE_ORDER).fillna(0).astype(int)
            show_table(ag.rename_axis("Age group").reset_index(name="Respondents"))
        with b:
            st.markdown("**Country_grouped** (top 8 countries, the rest as Other)")
            cg = DFC["Country_grouped"].value_counts()
            show_table(cg.rename_axis("Country").reset_index(name="Respondents"))
        st.markdown("Age bands keep charts readable, and grouping small countries avoids dozens of bars with only a few respondents each.")
        st.code(
            "df['Age_group'] = pd.cut(df['Age'], bins=[17, 25, 35, 45, 55, 75],\n"
            "                         labels=['18-25', '26-35', '36-45', '46-55', '56+'])\n"
            "top8 = df['Country'].value_counts().nlargest(8).index\n"
            "df['Country_grouped'] = df['Country'].where(df['Country'].isin(top8), 'Other')",
            language="python",
        )

    with tabs[4]:
        kpi_row([
            ("Raw file", f"{RAW.shape[0]:,} x {RAW.shape[1]}", "rows by columns", "slate"),
            ("Analysis-ready", f"{DFC.shape[0]:,} x {DFC.shape[1]}", "rows by columns", "teal"),
            ("Added", "3 columns", "Gender_clean, Age_group, Country_grouped", "teal"),
            ("Dropped", "3 columns", "Timestamp, comments, raw Gender", "amber"),
        ])
        st.write("")
        show_table(DFC.head(10))
        st.download_button("Download cleaned data (CSV)", DFC.to_csv(index=False).encode("utf-8"),
                           file_name="survey_cleaned.csv", mime="text/csv")


# =========================================================
# 10. PAGES - ANALYSIS
# =========================================================
def page_snapshot():
    need_data()
    d = DF
    page_head("Key metrics", "The whole survey on one screen. Sidebar filters update every number and chart here.")
    dk = float(np.mean([dont_know_share(d, c) for c in SUPPORT_COLS]))
    kpi_row([
        ("Respondents", f"{len(d):,}", f"of {len(DFC):,} analysed", "slate"),
        ("Sought treatment", pct(share(d, "treatment", "Yes")), "of respondents", "teal"),
        ("Family history", pct(share(d, "family_history", "Yes")), "of mental illness", "slate"),
        ("Median age", f"{d['Age'].median():.0f}", "years", "slate"),
        ("Unsure about support", pct(dk), "average 'Don't know' across six support questions", "amber"),
    ])
    st.write("")
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Have you sought treatment?", "Number of respondents and share")
        plot(count_bar(d, "treatment"))
    with b, st.container(border=True):
        panel_title("Gender", "Number of respondents and share")
        plot(count_bar(d, "Gender_clean"))
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Age and treatment", "Number of respondents by age")
        fig = px.histogram(d, x="Age", color="treatment", nbins=20, barmode="overlay", opacity=0.8,
                           color_discrete_map=COLOR_MAPS["treatment"], category_orders={"treatment": YN})
        fig.for_each_trace(lambda tr: tr.update(name="Sought treatment" if tr.name == "Yes" else "Did not seek treatment"))
        fig.update_yaxes(title_text="Respondents")
        plot(style(fig, 320))
    with b, st.container(border=True):
        panel_title("Family history and treatment", "% who sought treatment; whiskers show the 95% range")
        plot(rate_bar(d, "family_history"))

    section("Auto-generated readout", "Recomputed for whatever the filters currently select.")
    fam_yes, fam_no = rate_of(d, "family_history", "Yes"), rate_of(d, "family_history", "No")
    callout("finding", "Treatment is common in this sample",
            f"{pct(share(d, 'treatment', 'Yes'))} of the {len(d):,} respondents sought treatment. "
            f"With a family history it is {pct(fam_yes)}, without it {pct(fam_no)}.")
    callout("gap", "Many people are guessing about support",
            f"On average {pct(dk)} answered 'Don't know' or 'Not sure' across benefits, care options, wellness program, "
            f"help resources, anonymity and equal treatment. For anonymity alone it is {pct(dont_know_share(d, 'anonymity'))}.")
    callout("note", "Who is in the sample",
            f"Median age is {d['Age'].median():.0f}, {pct(share(d, 'Gender_clean', 'Male'))} are men and "
            f"{pct(share(d, 'Country', 'United States'))} live in the United States.")


def page_demographics():
    need_data()
    d = DF
    page_head("Who responded", "Age, gender, company size, country and work setup. This is the population every other chart speaks for.")

    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Age", "Histogram with a box plot on top")
        fig = px.histogram(d, x="Age", nbins=20, marginal="box", color_discrete_sequence=[TEAL])
        fig.update_traces(marker_line_color="#fff", marker_line_width=1, selector=dict(type="histogram"))
        fig.update_yaxes(title_text="")
        plot(style(fig, 340))
    with b, st.container(border=True):
        panel_title("Gender", "Number of respondents and share")
        plot(count_bar(d, "Gender_clean", 340))
    read_analysis(
        "A histogram with a box plot shows the shape, centre and any leftover outliers of Age in one view; a count plot is the simplest way to show the gender mix.",
        f"Median age is {d['Age'].median():.0f} and {pct(share(d, 'Age_group', '26-35'))} are aged 26 to 35. "
        f"{pct(share(d, 'Gender_clean', 'Male'))} are men, {pct(share(d, 'Gender_clean', 'Female'))} women and "
        f"{pct(share(d, 'Gender_clean', 'Other'))} other.",
        "Programs should fit an early-to-mid-career workforce (career pressure, work-life balance). The gender skew means women and "
        "non-binary people are thinly sampled, so listening sessions should complement survey data.",
    )

    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Age group", "Number of respondents and share")
        plot(count_bar(d, "Age_group"))
    with b, st.container(border=True):
        panel_title("Company size", "Number of employees in the respondent's company")
        plot(count_bar(d, "no_employees"))
    read_analysis(
        "Ordered bar charts show how respondents spread across career stage and organisation size, which decides whether the findings can be compared across firms.",
        f"Small firms (up to 100 people) make up {pct(float(d['no_employees'].isin(['1-5', '6-25', '26-100']).mean() * 100))} of respondents; "
        f"firms above 1,000 make up {pct(share(d, 'no_employees', 'More than 1000'))}.",
        "Startups and large enterprises have different budgets. Recommendations are split by company size on the Workplace support page.",
    )

    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Country", "Top 8 countries, the rest grouped as Other")
        plot(count_bar(d, "Country_grouped", 340))
    with b, st.container(border=True):
        panel_title("Work setup", "Share of respondents answering Yes")
        prof = [("Employer is a tech company", share(d, "tech_company", "Yes")),
                ("Works remotely 50%+ of the time", share(d, "remote_work", "Yes")),
                ("Self-employed", share(d, "self_employed", "Yes"))]
        fig = go.Figure(go.Bar(
            y=[p[0] for p in prof], x=[p[1] for p in prof], orientation="h", marker_color=TEAL,
            text=[f"{p[1]:.1f}%" for p in prof], textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>%{x:.1f}%<extra></extra>",
        ))
        fig.update_xaxes(range=[0, 105], ticksuffix="%")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        plot(style(fig, 340))
    read_analysis(
        "A bar chart of top countries shows where the data comes from; a horizontal bar of Yes-shares compares three yes/no traits on one scale.",
        f"{pct(share(d, 'Country', 'United States'))} of respondents live in the United States and "
        f"{pct(share(d, 'Country', 'United Kingdom'))} in the United Kingdom. {pct(share(d, 'tech_company', 'Yes'))} work for a tech company.",
        "Results lean towards the US context. Employers elsewhere should check whether local healthcare rules change what they need to offer.",
    )


def page_drivers():
    need_data()
    d = DF
    page_head("Treatment drivers", "Which answers go together with having sought treatment, and how strongly. Association, not proof of cause.")
    overall = share(d, "treatment", "Yes")

    section("1. Which questions matter most?",
            "Cramer's V measures how strongly each question is linked to treatment: 0 means no link, 1 a perfect link. "
            "As a rough guide, under 0.1 is weak, 0.1 to 0.3 moderate and above 0.3 strong.")
    cand = [c for c in DRIVER_COLS if d[c].nunique() > 1]
    if not cand:
        st.info("The respondents selected all give the same answer to every question, so there is nothing to compare. Widen the sidebar filters.")
        return
    vs = pd.DataFrame({"col": cand, "V": [cramers_v(d[c], d["treatment"]) for c in cand]}).dropna()
    vs = vs.sort_values("V", ascending=False).reset_index(drop=True)
    if len(vs) > 5:
        topn = st.slider("How many questions to show", 5, len(vs), min(12, len(vs)))
    else:
        topn = len(vs)
    show = vs.head(topn)
    with st.container(border=True):
        colors = [TEAL if i < 3 else SLATE for i in range(len(show))]
        fig = go.Figure(go.Bar(
            y=[LABEL[c] for c in show["col"]], x=show["V"], orientation="h", marker_color=colors,
            text=[f"{v:.2f}" for v in show["V"]], textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{y}</b><br>Cramer's V = %{x:.3f}<extra></extra>",
        ))
        fig.update_xaxes(range=[0, max(float(show["V"].max()) * 1.2, 0.1)], title="Strength of link with treatment")
        fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
        plot(style(fig, max(300, 34 * len(show) + 60)))
    if len(vs):
        top_names = ", ".join(LABEL[c].lower() for c in vs["col"].head(3))
        callout("finding", "Strongest links",
                f"The three strongest links with treatment are: {top_names}. "
                "Questions with tiny answer groups (country, or gender 'Other') can look stronger than they are, so check the sample sizes in section 2.")
    na_rate = rate_of(d, "work_interfere", "Not applicable")
    often = rate_of(d, "work_interfere", "Often")
    never = rate_of(d, "work_interfere", "Never")
    callout("risk", "Work interference is left out on purpose",
            f"It is only asked of people who report a condition. {pct(often)} of those answering 'Often' sought treatment against "
            f"{pct(never)} for 'Never' and {pct(na_rate)} for 'Not applicable'. That is close to restating the outcome, so ranking it would push every real driver down.")

    section("2. Look at any question", "Pick a question to see the treatment rate for each answer, with sample sizes.")
    pick = st.selectbox("Question", cand, format_func=lambda c: LABEL[c],
                        index=cand.index("family_history") if "family_history" in cand else 0)
    a, b = st.columns([1.5, 1], gap="medium")
    with a, st.container(border=True):
        panel_title(LABEL[pick], QUESTION.get(pick, ""))
        plot(rate_bar(d, pick))
    with b:
        tt = rate_table(d, pick)
        tt = tt.assign(**{"95% range": [f"{lo:.0f}% to {hi:.0f}%" for lo, hi in zip(tt["lo"], tt["hi"])]})
        tt["rate"] = tt["rate"].round(1)
        show_table(tt[["Answer", "n", "treated", "rate", "95% range"]].rename(
            columns={"n": "Respondents", "treated": "Sought treatment", "rate": "Rate %"}))
        st.caption("The dotted line marks the overall rate of " + pct(overall) + ". Lighter bars have fewer than 30 respondents.")
    st.markdown(gap_sentence(d, pick))

    section("3. Featured comparisons", "The comparisons the EDA notebook charts, with counts on the left and rates on the right.")
    featured = [
        ("Family history", "family_history",
         "A grouped bar chart compares a yes/no outcome across the groups of one question.",
         "Employers can run universal awareness campaigns that make it normal to ask for help. Nobody should be asked to disclose family history."),
        ("Gender", "Gender_clean",
         "A grouped bar chart lets you compare treatment rates across gender groups directly.",
         "Reducing stigma among men is a clear opportunity. The 'Other' group is too small to draw conclusions from."),
        ("Age group", "Age_group",
         "Grouped bars across age bands show whether help-seeking changes over a career.",
         "Support should be available across the whole career span rather than aimed at one age band."),
        ("Remote work", "remote_work",
         "A grouped bar chart compares remote and office-based staff.",
         "Remote staff need the same visibility of support as everyone else. The data gives no reason to treat them differently."),
        ("Care options", "care_options",
         "This tests whether knowing the care options goes together with using them.",
         "Making existing care options visible costs little. Note that people who have been treated are also more likely to know the options."),
    ]
    tabs = st.tabs([f[0] for f in featured])
    for tab, (name, col, why, impact) in zip(tabs, featured):
        with tab:
            a, b = st.columns(2, gap="medium")
            with a, st.container(border=True):
                panel_title(f"{LABEL[col]}: counts", "Respondents who did and did not seek treatment")
                plot(treatment_grouped(d, col))
            with b, st.container(border=True):
                panel_title(f"{LABEL[col]}: rate", "% who sought treatment")
                plot(rate_bar(d, col))
            extra = ""
            if col == "Age_group":
                med = d.groupby("treatment")["Age"].median()
                extra = f" Median age is {med.get('Yes', float('nan')):.0f} for those who sought treatment and {med.get('No', float('nan')):.0f} for those who did not."
            if col == "Gender_clean":
                n_other = int((d["Gender_clean"] == "Other").sum())
                if 0 < n_other < MIN_N:
                    extra = f" Only {n_other} respondents are in 'Other', so ignore that bar."
            read_analysis(why, gap_sentence(d, col) + extra, impact)


def page_support():
    need_data()
    d = DF
    page_head("Workplace support", "What employers offer, and how much of it employees actually know about.")

    section("1. The awareness gap", "Answers to the six support questions. Amber is 'Don't know' (or 'Not sure'), the people who are left guessing.")
    rows = {}
    for c in SUPPORT_COLS:
        s = d[c].replace({"Not sure": "Don't know"})
        vc = s.value_counts(normalize=True) * 100
        rows[LABEL[c]] = [float(vc.get(k, 0)) for k in YND]
    tbl = pd.DataFrame(rows, index=YND).T.sort_values("Don't know", ascending=False)
    with st.container(border=True):
        plot(stacked_pct(tbl, None, 330))
    worst = tbl.index[0]
    callout("gap", "Biggest blind spot",
            f"'{worst}' has the highest share of 'Don't know': {pct(tbl.iloc[0][DK])}. "
            f"On average {pct(float(tbl[DK].mean()))} of answers across these six questions are 'Don't know' or 'Not sure'. "
            "These are not missing data. They mean the employer may be paying for support that employees cannot find.")

    section("2. One question at a time", "Left: how people answered. Right: the treatment rate for each answer.")
    tab_cfg = [
        ("Benefits", ["benefits"],
         "A count plot is the clearest way to show how many employees know, do not have, or are unsure about benefits.",
         "Better communication (onboarding, intranet, manager talking points) can raise use of existing benefits at almost no cost."),
        ("Care options", ["care_options"],
         "This links awareness of care options to treatment-seeking.",
         "Awareness is one of the cheapest levers, but part of the link is reverse: people who have sought care learn what is available."),
        ("Anonymity", ["anonymity"],
         "This shows whether employees trust that using resources will stay private.",
         "Stating confidentiality guarantees clearly and repeatedly removes a barrier at no extra cost."),
        ("Leave", ["leave"],
         "An ordered chart shows how easy or hard employees think it is to take mental health leave.",
         "A simple, published leave process reduces the pressure to keep working while unwell."),
        ("Wellness and resources", ["wellness_program", "seek_help"],
         "Count plots show whether employers talk about mental health and provide learning resources.",
         "Regular, visible programs normalise the topic and show employees where to go."),
    ]
    tabs = st.tabs([t[0] for t in tab_cfg])
    for tab, (name, cols, why, impact) in zip(tabs, tab_cfg):
        with tab:
            texts = []
            for col in cols:
                a, b = st.columns(2, gap="medium")
                with a, st.container(border=True):
                    panel_title(LABEL[col], QUESTION.get(col, ""))
                    plot(count_bar(d, col))
                with b, st.container(border=True):
                    panel_title("Treatment rate by answer", "% who sought treatment")
                    plot(rate_bar(d, col))
                if col == "leave":
                    easy = float(d[col].isin(["Very easy", "Somewhat easy"]).mean() * 100)
                    hard = float(d[col].isin(["Somewhat difficult", "Very difficult"]).mean() * 100)
                    mix = f"{pct(easy)} find it easy, {pct(hard)} find it difficult and {pct(dont_know_share(d, col))} do not know."
                else:
                    mix = (f"{pct(share(d, col, 'Yes'))} answered Yes, {pct(share(d, col, 'No'))} No and "
                           f"{pct(dont_know_share(d, col))} Don't know or Not sure.")
                texts.append(f"{LABEL[col]}: {mix} {gap_sentence(d, col)}")
            read_analysis(why, " ".join(texts), impact)

    section("3. Does company size change the picture?", "Small firms and large firms have different problems.")
    sizes = order_for("no_employees", d)
    qs = SUPPORT_COLS + ["leave"]
    z, txt = [], []
    for c in qs:
        row, trow = [], []
        for s in sizes:
            sub = d[d["no_employees"] == s]
            v = dont_know_share(sub, c) if len(sub) else float("nan")
            row.append(v)
            trow.append("" if pd.isna(v) else f"{v:.0f}%")
        z.append(row)
        txt.append(trow)
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Share answering 'Don't know'", "Darker means more people are unsure")
        plot(heatmap(z, txt, x=sizes, y=[LABEL[c] for c in qs], height=380, zmin=0, zmax=80))
    with b, st.container(border=True):
        panel_title("Employer provides mental health benefits?", "Share of respondents by company size")
        plot(stacked_pct(crosstab_pct(d, "no_employees", "benefits"), "benefits", 380))
    yes_by_size = {s: share(d[d["no_employees"] == s], "benefits", "Yes") for s in sizes if (d["no_employees"] == s).sum() >= MIN_N}
    if len(yes_by_size) >= 2:
        first, last = list(yes_by_size)[0], list(yes_by_size)[-1]
        callout("finding", "Small firms mostly lack benefits, and even large firms leave many people guessing",
                f"Only {pct(yes_by_size[first])} of respondents at firms of size {first} say benefits exist, against {pct(yes_by_size[last])} at firms of size {last}. "
                f"Yet at the larger firms {pct(dont_know_share(d[d['no_employees'] == last], 'leave'))} still do not know how easy it is to take leave.")
    callout("risk", "Read the treatment rates with care",
            "People who have sought treatment have had a reason to find out what benefits exist, so 'Yes' answers naturally go with higher treatment rates. "
            "That link does not prove that telling people about benefits will make more of them seek help.")


def compare_bar(data, pairs, answers, height=320):
    """Grouped bars comparing the answer shares of two or more questions."""
    fig = go.Figure()
    colors = [TEAL, SLATE, AMBER]
    top = 1.0
    for (col, name), colr in zip(pairs, colors):
        vals = [share(data, col, a) for a in answers]
        top = max([top] + [v for v in vals if not pd.isna(v)])
        fig.add_trace(go.Bar(
            x=answers, y=vals, name=name, marker_color=colr,
            text=[f"{v:.1f}%" for v in vals], textposition="outside", cliponaxis=False,
            hovertemplate="<b>%{x}</b><br>" + name + ": %{y:.1f}%<extra></extra>",
        ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, top * 1.2], ticksuffix="%", title="Share of respondents")
    return style(fig, height)


def page_attitudes():
    need_data()
    d = DF
    page_head("Stigma and openness", "How comfortable people are talking about mental health at work, and how that compares with physical health.")

    section("1. Mental health versus physical health", "The same two questions asked about a mental health issue and a physical one.")
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Would discussing it with your employer have negative consequences?", "Share of respondents")
        plot(compare_bar(d, [("mental_health_consequence", "Mental health"), ("phys_health_consequence", "Physical health")], YNM))
    with b, st.container(border=True):
        panel_title("Would you bring it up in a job interview?", "Share of respondents")
        plot(compare_bar(d, [("mental_health_interview", "Mental health"), ("phys_health_interview", "Physical health")], YMN))
    m_yes, p_yes = share(d, "mental_health_consequence", "Yes"), share(d, "phys_health_consequence", "Yes")
    mi_yes, pi_yes = share(d, "mental_health_interview", "Yes"), share(d, "phys_health_interview", "Yes")
    fear = float(d["mental_health_consequence"].isin(["Yes", "Maybe"]).mean() * 100)
    callout("gap", "The stigma gap is large",
            f"{pct(m_yes)} expect negative consequences for a mental health issue versus {pct(p_yes)} for a physical one. "
            f"Only {pct(mi_yes)} would raise a mental health issue in an interview, against {pct(pi_yes)} for a physical one. "
            f"Counting 'Maybe', {pct(fear)} are not confident it would be safe.")
    read_analysis(
        "Side-by-side bars for the same question about mental and physical health isolate the stigma from the general fear of disclosing any health issue.",
        f"The difference between the two sets of bars is the extra risk people attach to mental health. Fear of consequences is {m_yes / p_yes:.1f} times higher for mental health." if p_yes and not pd.isna(p_yes) and p_yes > 0 else "There are too few answers to compare.",
        "High fear delays help-seeking, so issues get more severe before anyone acts. Leadership messaging, non-retaliation policies and manager training address this directly.",
    )

    section("2. Who would people talk to?", "Willingness to discuss a mental health issue at work.")
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Supervisor versus coworkers", "Share of respondents")
        plot(compare_bar(d, [("supervisor", "Direct supervisor"), ("coworkers", "Coworkers")], YSN))
    with b, st.container(border=True):
        panel_title("Willingness to talk to a supervisor", "Number of respondents and share")
        plot(count_bar(d, "supervisor"))
    read_analysis(
        "Comparing supervisor and coworker willingness on one chart shows which relationship people trust more.",
        f"{pct(share(d, 'supervisor', 'Yes'))} would talk to their supervisor and {pct(share(d, 'coworkers', 'Yes'))} to coworkers. "
        f"{pct(share(d, 'supervisor', 'No'))} would not talk to their supervisor at all.",
        "The supervisor is the natural first point of contact, so training managers to respond supportively and confidentially moves more people into the 'Yes' group.",
    )

    section("3. Is the workplace safe?", "Whether employers walk the talk, and what people have seen.")
    a, b = st.columns(2, gap="medium")
    with a, st.container(border=True):
        panel_title("Employer takes mental health as seriously as physical health?", "Number of respondents and share")
        plot(count_bar(d, "mental_vs_physical"))
    with b, st.container(border=True):
        panel_title("Seen or heard of negative consequences for a coworker?", "Number of respondents and share")
        plot(count_bar(d, "obs_consequence"))
    obs_yes, obs_no = rate_of(d, "obs_consequence", "Yes"), rate_of(d, "obs_consequence", "No")
    read_analysis(
        "These count plots show whether employees believe in the parity claim and whether they have seen stigma in practice.",
        f"Only {pct(share(d, 'mental_vs_physical', 'Yes'))} say their employer treats mental and physical health equally; {pct(share(d, 'mental_vs_physical', DK))} do not know. "
        f"{pct(share(d, 'obs_consequence', 'Yes'))} have seen or heard of negative consequences for a coworker, and {pct(obs_yes)} of them sought treatment versus {pct(obs_no)} of those who did not.",
        "Even a minority of visible stories of harm spread fear. Investigating real cases and sharing positive examples of support counters that effect.",
    )

    section("4. How often does it get in the way of work?", "Asked only of people who report a condition; everyone else is 'Not applicable'.")
    with st.container(border=True):
        plot(count_bar(d, "work_interfere", 320))
    answered = d[d["work_interfere"] != "Not applicable"]
    if len(answered):
        so = float(answered["work_interfere"].isin(["Sometimes", "Often"]).mean() * 100)
        callout("finding", "Interference is common among those with a condition",
                f"Of the {len(answered):,} respondents who answered, {pct(so)} say it interferes with work 'Sometimes' or 'Often'. "
                "That is a direct productivity cost, and it is the business case for flexible leave and early support.")


def page_geography():
    need_data()
    d = DF
    page_head("Geography", "Does treatment-seeking, or workplace support, differ by where people live? Bars show how sure we can be.")

    section("1. Treatment rate by country", "Countries are listed by number of respondents. Whiskers show the 95% range: wide whiskers mean a small sample.")
    min_n = st.slider("Only show countries with at least this many respondents", 5, 60, 20)
    cc = d["Country"].value_counts()
    keep = cc[cc >= min_n].index
    if len(keep) < 2:
        st.info("Fewer than two countries meet this minimum. Lower the slider or widen the sidebar filters.")
    else:
        sub = d[d["Country"].isin(keep)]
        with st.container(border=True):
            plot(rate_bar(sub, "Country", ref_label="All"))
        tt = rate_table(sub, "Country")
        big = tt[tt["n"] >= 60] if (tt["n"] >= 60).sum() >= 2 else tt
        spread = float(big["rate"].max() - big["rate"].min())
        bigc = cc[cc >= 50].index
        v_big = float("nan")
        if len(bigc) >= 2:
            sb = d[d["Country"].isin(bigc)]
            v_big = cramers_v(sb["Country"], sb["treatment"])
        v_txt = f" Among countries with 50+ respondents, Cramer's V for country is {v_big:.2f}, a weak link." if not pd.isna(v_big) else ""
        callout("finding", "Compare the whiskers before reading a ranking",
                f"Among the biggest countries the treatment rate sits between {big['rate'].min():.0f}% and {big['rate'].max():.0f}%. "
                "Bigger gaps tend to appear where samples are small, and those are the least reliable bars." + v_txt)
        with st.expander("See the numbers"):
            tshow = tt.copy()
            tshow["rate"] = tshow["rate"].round(1)
            tshow["95% range"] = [f"{lo:.0f}% to {hi:.0f}%" for lo, hi in zip(tshow["lo"], tshow["hi"])]
            show_table(tshow[["Answer", "n", "treated", "rate", "95% range"]].rename(
                columns={"Answer": "Country", "n": "Respondents", "treated": "Sought treatment", "rate": "Rate %"}))
        read_analysis(
            "A bar chart per country answers the brief's question on location directly; confidence whiskers stop small samples from being over-read.",
            (f"Across the larger countries the treatment rate differs by {spread:.0f} percentage points, "
             + ("a small spread." if spread <= 10 else "a noticeable spread, so check the whiskers.")),
            "Treatment-seeking does not need a country-by-country message. What differs is the benefits people actually get (see section 3).",
        )

    section("2. United States by state", "Only US residents were asked for a state.")
    us = d[(d["Country"] == "United States")].dropna(subset=["state"])
    if len(us) < 20:
        st.info("Too few US respondents remain after filtering to draw a state map.")
    else:
        c1, c2 = st.columns([1.2, 1])
        metric = c1.radio("Colour states by", ["% who sought treatment", "Number of respondents"], horizontal=True)
        min_s = c2.slider("Minimum respondents per state", 1, 30, 10)
        g = rate_table(us, "state")
        if metric.startswith("%"):
            g = g[g["n"] >= min_s]
            zcol, ctitle = "rate", "% treated"
        else:
            zcol, ctitle = "n", "Respondents"
        if len(g) == 0:
            st.info("No state has that many respondents. Lower the minimum.")
        else:
            fig = go.Figure(go.Choropleth(
                locations=g["Answer"], z=g[zcol], locationmode="USA-states",
                colorscale=[[0, "#EAF4F4"], [1, TEAL_DARK]], marker_line_color="#FFFFFF", marker_line_width=1,
                colorbar=dict(title=ctitle, thickness=12, outlinewidth=0),
                customdata=np.c_[g["n"], g["rate"]],
                hovertemplate="<b>%{location}</b><br>%{customdata[0]:,} respondents<br>%{customdata[1]:.1f}% sought treatment<extra></extra>",
            ))
            fig.update_geos(scope="usa", showlakes=False, bgcolor="rgba(0,0,0,0)")
            with st.container(border=True):
                plot(style(fig, 420))
            top_states = us["state"].value_counts().head(5)
            st.caption("Most respondents: " + ", ".join(f"{s} ({n})" for s, n in top_states.items()) + ". Most states have very few respondents, so treat state-level rates as indicative only.")

    section("3. Does employer support differ by country?", "Treatment rates are similar, but the benefits people get are not.")
    top = d["Country"].value_counts()
    top = top[top >= max(min_n, MIN_N)].index[:5]
    if len(top) >= 2:
        sub = d[d["Country"].isin(top)]
        a, b = st.columns(2, gap="medium")
        with a, st.container(border=True):
            panel_title("Employer provides mental health benefits?", "Share of respondents by country")
            plot(stacked_pct(crosstab_pct(sub, "Country", "benefits", rows=list(top)), "benefits", 360))
        with b, st.container(border=True):
            panel_title("Knows the care options?", "Share of respondents by country")
            plot(stacked_pct(crosstab_pct(sub, "Country", "care_options", rows=list(top)), "care_options", 360))
        by = {c: share(sub[sub["Country"] == c], "benefits", "Yes") for c in top}
        hi_c, lo_c = max(by, key=by.get), min(by, key=by.get)
        callout("gap", "Benefits differ far more than treatment does",
                f"{pct(by[hi_c])} of respondents in {hi_c} say their employer offers mental health benefits versus {pct(by[lo_c])} in {lo_c}. "
                "Public healthcare systems and employment rules differ, so what an employer needs to add differs too.")
    else:
        st.info("Not enough countries meet the minimum sample size for this comparison.")


def _flag_frame(d, cols):
    """Numeric frame: Age plus a 0/1 flag for every chosen Yes/No question (1 = Yes)."""
    out = pd.DataFrame({"Age": d["Age"].astype(float)})
    for c in cols:
        out[f"{LABEL[c]}: Yes"] = (d[c] == "Yes").astype(int)
    out["Gender: Female"] = (d["Gender_clean"] == "Female").astype(int)
    return out


def page_relationships():
    need_data()
    d = DF
    page_head("Relationships", "Look at several variables at once: a correlation map, a segment heatmap and a cross-tab builder.")
    tab1, tab2, tab3 = st.tabs(["Correlation map", "Segment heatmap", "Cross-tab builder"])

    with tab1:
        flag_opts = ["treatment", "family_history", "remote_work", "tech_company", "self_employed", "obs_consequence",
                     "benefits", "care_options", "wellness_program", "seek_help", "anonymity", "mental_vs_physical"]
        default = ["treatment", "family_history", "remote_work", "tech_company", "self_employed", "obs_consequence"]
        chosen = st.multiselect("Yes/No questions to include (Age and Female are always added)", flag_opts,
                                default=default, format_func=lambda c: LABEL[c])
        if "treatment" not in chosen:
            chosen = ["treatment"] + chosen
        frame = _flag_frame(d, chosen)
        corr = frame.corr()
        txt = [["" if pd.isna(v) else f"{v:.2f}" for v in row] for row in corr.values]
        with st.container(border=True):
            plot(heatmap(corr.values, txt, x=list(corr.columns), y=list(corr.index), height=520,
                         diverging=True, zmin=-1, zmax=1, colorbar_title="r"))
        tcol = "Sought treatment: Yes"
        s = corr[tcol].drop(labels=[tcol]).dropna()
        if len(s):
            best = s.abs().idxmax()
            read_analysis(
                "A correlation heatmap is the standard first look at linear links between numeric and 0/1 variables. Between two 0/1 flags it is the phi coefficient.",
                f"The strongest link with treatment is '{best}' (r = {s[best]:.2f}). Age shows r = {s.get('Age', float('nan')):.2f}, a weak link.",
                "Family history and having seen others penalised move together with treatment far more than age or remote work do, so programs should tackle stigma and awareness before targeting demographics.",
            )
        st.caption("Correlation shows how variables move together. It cannot say which one causes the other.")

    with tab2:
        opts = [c for c in DRIVER_COLS if d[c].nunique() > 1]
        c1, c2, c3 = st.columns(3)
        rv = c1.selectbox("Rows", opts, index=opts.index("family_history") if "family_history" in opts else 0, format_func=lambda c: LABEL[c])
        cv = c2.selectbox("Columns", opts, index=opts.index("Gender_clean") if "Gender_clean" in opts else 1, format_func=lambda c: LABEL[c])
        min_cell = c3.slider("Hide cells with fewer than", 1, 40, 10)
        if rv == cv:
            st.info("Pick two different questions.")
        else:
            rows_, cols_ = order_for(rv, d), order_for(cv, d)
            z, txt = [], []
            for r in rows_:
                zr, tr = [], []
                for c in cols_:
                    sub = d[(d[rv].astype(str) == r) & (d[cv].astype(str) == c)]
                    if len(sub) >= min_cell:
                        v = share(sub, "treatment", "Yes")
                        zr.append(v)
                        tr.append(f"{v:.0f}%<br>n={len(sub)}")
                    else:
                        zr.append(float("nan"))
                        tr.append("")
                z.append(zr)
                txt.append(tr)
            with st.container(border=True):
                panel_title(f"% who sought treatment: {LABEL[rv]} by {LABEL[cv]}", "Each cell shows the rate and the number of respondents")
                plot(heatmap(z, txt, x=cols_, y=rows_, height=max(320, 90 * len(rows_)), zmin=0, zmax=100, colorbar_title="% treated"))
            st.caption("Empty cells have too few respondents to show a stable percentage.")

    with tab3:
        allopts = ["treatment", "work_interfere"] + [c for c in DRIVER_COLS if d[c].nunique() > 1]
        c1, c2, c3 = st.columns(3)
        a = c1.selectbox("Rows ", allopts, index=allopts.index("family_history") if "family_history" in allopts else 0, format_func=lambda c: LABEL[c])
        b = c2.selectbox("Columns ", allopts, index=0, format_func=lambda c: LABEL[c])
        mode = c3.radio("Show", ["Counts", "% of row", "% of column"], horizontal=True)
        if a == b:
            st.info("Pick two different questions.")
        else:
            norm = {"Counts": False, "% of row": "index", "% of column": "columns"}[mode]
            ct = pd.crosstab(d[a].astype(str), d[b].astype(str), normalize=norm)
            ct = ct.reindex(index=[r for r in order_for(a, d) if r in ct.index], columns=[c for c in order_for(b, d) if c in ct.columns])
            vals = ct.values * (100 if norm else 1)
            txt = [[f"{v:.1f}%" if norm else f"{int(v):,}" for v in row] for row in vals]
            with st.container(border=True):
                plot(heatmap(vals, txt, x=list(ct.columns), y=list(ct.index), height=max(300, 80 * len(ct))))
            out = pd.DataFrame(vals, index=ct.index, columns=ct.columns).round(1)
            show_table(out.reset_index().rename(columns={"index": LABEL[a]}))
            st.download_button("Download this table (CSV)", out.to_csv().encode("utf-8"),
                               file_name=f"crosstab_{a}_by_{b}.csv", mime="text/csv")


# =========================================================
# 11. PAGES - CONCLUSIONS
# =========================================================
def page_recommendations():
    need_data()
    d = DF
    page_head("Recommendations", "What an employer could do, with the evidence behind each step. Numbers follow the sidebar filters.")

    dk = lambda c: dont_know_share(d, c)
    fear = float(d["mental_health_consequence"].isin(["Yes", "Maybe"]).mean() * 100)
    fear_phys = float(d["phys_health_consequence"].isin(["Yes", "Maybe"]).mean() * 100)
    hard_leave = float(d["leave"].isin(["Somewhat difficult", "Very difficult"]).mean() * 100)
    care_yes, care_no = rate_of(d, "care_options", "Yes"), rate_of(d, "care_options", "No")
    fam_yes, fam_no = rate_of(d, "family_history", "Yes"), rate_of(d, "family_history", "No")
    fem, mal = rate_of(d, "Gender_clean", "Female"), rate_of(d, "Gender_clean", "Male")
    sizes = [s for s in order_for("no_employees", d) if (d["no_employees"] == s).sum() >= MIN_N]
    b_small = share(d[d["no_employees"] == sizes[0]], "benefits", "Yes") if sizes else float("nan")
    b_large = share(d[d["no_employees"] == sizes[-1]], "benefits", "Yes") if sizes else float("nan")
    ctop = d["Country"].value_counts()
    ctop = ctop[ctop >= MIN_N].index[:4]
    by_c = {c: share(d[d["Country"] == c], "benefits", "Yes") for c in ctop}
    c_txt = ", ".join(f"{c} {pct(v, 0)}" for c, v in by_c.items()) if by_c else "not enough respondents per country"

    recs = [
        ("High impact", "Low cost", "1. Close the awareness gap",
         f"{pct(dk('anonymity'), 0)} do not know if anonymity is protected, {pct(dk('leave'), 0)} do not know how easy leave is, "
         f"and {pct(dk('benefits'), 0)} do not know if benefits exist. People who know their care options sought treatment "
         f"{pct(care_yes, 0)} of the time versus {pct(care_no, 0)} for those who do not.",
         ["Put a one-page summary of benefits, confidentiality and leave in onboarding and on the intranet.",
          "Give managers a short talking-points sheet and repeat the message each year.",
          "Track the 'Don't know' rate in an annual pulse survey and aim to push it down."]),
        ("High impact", "Medium cost", "2. Reduce the fear of negative consequences",
         f"{pct(fear, 0)} say discussing mental health with an employer might have negative consequences (Yes or Maybe), against "
         f"{pct(fear_phys, 0)} for a physical health issue. {pct(share(d, 'obs_consequence', 'Yes'), 0)} have seen or heard of a coworker being penalised.",
         ["Senior leaders speak openly about mental health, and share real recovery stories with permission.",
          "Publish a clear non-retaliation policy and act on any breach.",
          "Train managers to respond to a disclosure calmly and confidentially."]),
        ("Medium impact", "Low cost", "3. Make mental health leave clear and easy",
         f"{pct(dk('leave'), 0)} do not know how easy it is to take leave and {pct(hard_leave, 0)} find it somewhat or very difficult.",
         ["Publish the steps for taking mental health leave, in plain language.",
          "Treat it like any other medical leave, with the same paperwork and the same manager response."]),
        ("Medium impact", "Medium cost", "4. Treat mental health like physical health in practice",
         f"Only {pct(share(d, 'mental_vs_physical', 'Yes'), 0)} say their employer takes mental health as seriously as physical health, and "
         f"{pct(dk('mental_vs_physical'), 0)} do not know.",
         ["Align benefits, leave and manager behaviour so parity is visible, not just stated.",
          "Include mental health in the same health and safety conversations as physical health."]),
        ("High impact", "Varies", "5. Fit the fix to the size of the company",
         f"Benefits exist for {pct(b_small, 0)} of respondents at the smallest firms in this view and {pct(b_large, 0)} at the largest.",
         ["Small firms: low-cost stipends or an outside counselling service, flexible leave, and open conversation.",
          "Large firms: benefits already exist, so invest in communication and manager training."]),
        ("Medium impact", "Low cost", "6. Use the predictors to shape outreach, never to screen people",
         f"Family history goes with treatment ({pct(fam_yes, 0)} versus {pct(fam_no, 0)}), and women report treatment more often than men "
         f"({pct(fem, 0)} versus {pct(mal, 0)}).",
         ["Run open awareness campaigns that make asking for help normal, with messages aimed at men.",
          "Never ask employees for family or health history, and never use these patterns to profile individuals."]),
        ("Medium impact", "Low cost", "7. Strengthen manager capability",
         f"{pct(share(d, 'supervisor', 'Yes'), 0)} would talk to their supervisor, {pct(share(d, 'supervisor', 'No'), 0)} would not, "
         f"and only {pct(share(d, 'coworkers', 'Yes'), 0)} would talk to coworkers.",
         ["Give every manager basic mental health literacy training.",
          "Teach a simple script: listen, keep it private, point to support, follow up."]),
        ("Weaker evidence", "Low cost", "8. Check local benefits, not local treatment rates",
         f"Treatment rates look similar across large countries, but benefits do not (share saying benefits exist: {c_txt}).",
         ["For each country, check what public healthcare covers and what the employer still needs to add.",
          "Keep the message consistent worldwide and adjust only the practical details."]),
    ]
    for impact, effort, title, evidence, actions in recs:
        with st.container(border=True):
            left, right = st.columns([1.1, 1], gap="large")
            with left:
                tag2 = "rec-tag-amber" if impact == "Weaker evidence" else ""
                st.markdown(
                    f'<span class="rec-tag {tag2}">{impact}</span><span class="rec-tag rec-tag-slate">{effort}</span>'
                    f'<div class="rec-title">{title}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(evidence)
            with right:
                st.markdown("**What to do**")
                st.markdown("\n".join(f"- {a}" for a in actions))

    section("What this data cannot tell you")
    callout("risk", "Association, not cause",
            "The survey is a one-time snapshot. Seeing that treated people know more about their benefits does not show that telling people "
            "about benefits will make more of them seek treatment. Treat these steps as well-motivated bets and measure the result.")
    callout("note", "Who answered",
            "A mental health nonprofit ran this as an open online survey in 2014, so people interested in the topic were more likely to respond. "
            "It is best read as a sample of engaged respondents, not a headcount of all tech workers.")
    section("Expected business impact")
    st.markdown(
        "- Higher use of benefits the company already pays for\n"
        "- Earlier help-seeking, before problems become severe\n"
        "- Less presenteeism and burnout\n"
        "- Better trust in managers, and better retention"
    )


def page_about():
    page_head("About and limits", "Where the data comes from, how it was handled, and what it cannot support.")
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("### Dataset")
        st.markdown(
            "- **Source:** Open Sourcing Mental Illness (OSMI), 2014 Mental Health in Tech Survey\n"
            "- **Purpose:** measure attitudes towards mental health and how common mental health disorders are in the tech workplace\n"
            f"- **Size:** {len(RAW):,} responses and {RAW.shape[1]} columns; {len(DFC):,} responses after cleaning\n"
            "- **Coverage:** respondents from many countries, mostly the United States and the United Kingdom"
        )
        st.markdown("### Method")
        st.markdown(
            "- Cleaned Age and Gender, filled two columns, added Age_group and Country_grouped\n"
            "- Univariate, bivariate and multivariate views, 20+ charts in the EDA notebook\n"
            "- Cramer's V to rank drivers, Wilson 95% intervals to show uncertainty\n"
            "- Groups under 30 respondents are flagged with lighter bars"
        )
    with b:
        st.markdown("### Limits to keep in mind")
        us_share = share(DFC, "Country", "United States")
        male_share = share(DFC, "Gender_clean", "Male")
        n_other = int((DFC["Gender_clean"] == "Other").sum())
        st.markdown(
            "- **Self-selected sample.** An open online survey attracts people who care about the topic, so the treatment rate is not an industry rate.\n"
            "- **2014 snapshot.** Benefits and attitudes have changed since.\n"
            f"- **US-heavy.** {pct(us_share, 0)} of respondents live in the United States.\n"
            f"- **Gender skew.** {pct(male_share, 0)} are men, and only {n_other} fall in 'Other'.\n"
            "- **Self-reported.** 'Sought treatment' is not a diagnosis, and 'Don't know' does not mean the benefit is missing.\n"
            "- **Association only.** Nothing here proves that one factor causes another."
        )
    section("Run it yourself")
    st.code("pip install -U streamlit pandas numpy plotly\nstreamlit run app.py", language="bash")
    st.markdown(
        "Keep `survey.csv` in the same folder as `app.py`. If it is missing, the app asks you to upload it.\n\n"
        "**Project type:** Exploratory Data Analysis. **Contribution:** Individual. "
        "**Built with:** Streamlit, pandas, NumPy, Plotly. **Data:** Open Sourcing Mental Illness (OSMI)."
    )


# =========================================================
# 12. NAVIGATION, SIDEBAR FILTERS AND RUN
# =========================================================
PG = {
    "overview": st.Page(page_overview, title="Overview", default=True),
    "dataset": st.Page(page_dataset, title="Data understanding", url_path="data"),
    "cleaning": st.Page(page_cleaning, title="Data cleaning", url_path="cleaning"),
    "snapshot": st.Page(page_snapshot, title="Key metrics", url_path="metrics"),
    "who": st.Page(page_demographics, title="Who responded", url_path="respondents"),
    "drivers": st.Page(page_drivers, title="Treatment drivers", url_path="drivers"),
    "support": st.Page(page_support, title="Workplace support", url_path="support"),
    "attitudes": st.Page(page_attitudes, title="Stigma and openness", url_path="attitudes"),
    "geo": st.Page(page_geography, title="Geography", url_path="geography"),
    "relations": st.Page(page_relationships, title="Relationships", url_path="relationships"),
    "recs": st.Page(page_recommendations, title="Recommendations", url_path="recommendations"),
    "about": st.Page(page_about, title="About and limits", url_path="about"),
}
NAV = st.navigation(
    {
        "Start": [PG["overview"], PG["dataset"], PG["cleaning"]],
        "Analysis": [PG["snapshot"], PG["who"], PG["drivers"], PG["support"], PG["attitudes"], PG["geo"], PG["relations"]],
        "Conclusions": [PG["recs"], PG["about"]],
    },
    position="sidebar",
    expanded=True,
)
FILTER_TITLES = {PG[k].title for k in ["snapshot", "who", "drivers", "support", "attitudes", "geo", "relations", "recs"]}
FILTER_ACTIVE = NAV.title in FILTER_TITLES

_AGE_MIN, _AGE_MAX = int(DFC["Age"].min()), int(DFC["Age"].max())
_DEFAULTS = {"f_gender": [], "f_country": [], "f_size": [], "f_age": (_AGE_MIN, _AGE_MAX)}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)


def _reset_filters():
    for k, v in _DEFAULTS.items():
        st.session_state[k] = v


with st.sidebar:
    st.markdown("### Filter respondents")
    st.caption("Applies to the Analysis pages. Leave a box empty to include everyone." if FILTER_ACTIVE
               else "Filters apply to the Analysis pages, not to the raw data pages.")
    st.multiselect("Gender", ORDER["Gender_clean"], key="f_gender", placeholder="All", disabled=not FILTER_ACTIVE)
    st.multiselect("Country", order_for("Country_grouped", DFC), key="f_country", placeholder="All", disabled=not FILTER_ACTIVE)
    st.multiselect("Company size", SIZE_ORDER, key="f_size", placeholder="All", disabled=not FILTER_ACTIVE)
    st.slider("Age", _AGE_MIN, _AGE_MAX, key="f_age", disabled=not FILTER_ACTIVE)
    st.button("Reset filters", on_click=_reset_filters, disabled=not FILTER_ACTIVE)

_mask = pd.Series(True, index=DFC.index)
if FILTER_ACTIVE:
    if st.session_state.f_gender:
        _mask &= DFC["Gender_clean"].isin(st.session_state.f_gender)
    if st.session_state.f_country:
        _mask &= DFC["Country_grouped"].isin(st.session_state.f_country)
    if st.session_state.f_size:
        _mask &= DFC["no_employees"].isin(st.session_state.f_size)
    _lo, _hi = st.session_state.f_age
    _mask &= DFC["Age"].between(_lo, _hi)
DF = DFC[_mask]

with st.sidebar:
    if FILTER_ACTIVE:
        st.progress(len(DF) / len(DFC))
        st.caption(f"{len(DF):,} of {len(DFC):,} respondents selected")
    st.markdown("---")
    st.caption("Mental Health in Tech Survey, OSMI 2014. Individual EDA project.")

NAV.run()