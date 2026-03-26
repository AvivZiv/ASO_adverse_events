# Python 3.9 compatible
from typing import Optional, Dict, List, Tuple, Set
import math
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# ====================== App setup ======================
st.set_page_config(page_title="ASO Analytics", layout="wide", page_icon="🧬")

# ---------- Theme System ----------
PALETTES = {
    "Dark": {
        "bg":         "#0f172a",
        "sidebar_bg": "#1e293b",
        "card_bg":    "#1e293b",
        "input_bg":   "#0f172a",
        "border":     "rgba(255,255,255,0.10)",
        "text":       "#e2e8f0",
        "muted":      "#94a3b8",
        "accent":     "#facc15",
        "chip_bg":    "#334155",
        "chip_text":  "#e2e8f0",
        "plot": ["#006A4E", "#FFC72C", "#94a3b8", "#14b8a6", "#3b82f6", "#ef4444"],
    },
    "Bright": {
        "bg":         "#f1f5f9",
        "sidebar_bg": "#e2e8f0",
        "card_bg":    "#ffffff",
        "input_bg":   "#ffffff",
        "border":     "rgba(0,0,0,0.10)",
        "text":       "#0f172a",
        "muted":      "#475569",
        "accent":     "#0f172a",
        "chip_bg":    "#e2e8f0",
        "chip_text":  "#334155",
        "plot": ["#334155", "#0ea5e9", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"],
    },
}

def inject_theme(p: Dict[str, str]):
    st.markdown(
        f"""
        <style>
        /* ---- global tokens ---- */
        :root {{
            --aso-bg:         {p['bg']};
            --aso-sidebar:    {p['sidebar_bg']};
            --aso-card:       {p['card_bg']};
            --aso-input:      {p['input_bg']};
            --aso-border:     {p['border']};
            --aso-text:       {p['text']};
            --aso-muted:      {p['muted']};
            --aso-accent:     {p['accent']};
            --aso-chip-bg:    {p['chip_bg']};
            --aso-chip-text:  {p['chip_text']};
        }}

        /* ---- main background ---- */
        .stApp,
        .stApp > div,
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewContainer"] > section,
        .main .block-container {{
            background-color: var(--aso-bg) !important;
        }}
        .block-container {{ padding-top: 2.5rem; max-width: 1500px; }}

        /* ---- sidebar ---- */
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] > div,
        [data-testid="stSidebar"] > div > div {{
            background-color: var(--aso-sidebar) !important;
        }}

        /* ---- all text ---- */
        body, h1, h2, h3, h4, h5, h6, p, label, span, li, td, th,
        .stMarkdown, .stText, .stCaption,
        [data-testid="stSidebar"] * {{
            color: var(--aso-text) !important;
        }}

        /* ---- every white/light surface ---- */
        div[data-baseweb="base-input"],
        div[data-baseweb="input"],
        div[data-baseweb="select"] > div,
        div[data-baseweb="popover"],
        div[data-baseweb="menu"],
        ul[data-baseweb="menu"],
        li[role="option"],
        [data-testid="stNumberInput"] div[data-baseweb="base-input"],
        input, textarea, select {{
            background-color: var(--aso-input) !important;
            color: var(--aso-text) !important;
        }}

        /* ---- input borders ---- */
        div[data-baseweb="base-input"],
        div[data-baseweb="select"] > div:first-child {{
            border: 1px solid var(--aso-border) !important;
            border-radius: 6px !important;
        }}

        /* ---- multiselect tags ---- */
        [data-baseweb="tag"] {{
            background-color: var(--aso-chip-bg) !important;
            color: var(--aso-chip-text) !important;
        }}
        [data-baseweb="tag"] span {{
            color: var(--aso-chip-text) !important;
        }}

        /* ---- number input +/- buttons ---- */
        [data-testid="stNumberInput"] button,
        [data-testid="baseButton-secondary"] {{
            background-color: var(--aso-card) !important;
            color: var(--aso-text) !important;
            border: 1px solid var(--aso-border) !important;
        }}

        /* ---- all st.button ---- */
        .stButton > button {{
            background-color: var(--aso-card) !important;
            color: var(--aso-text) !important;
            border: 1px solid var(--aso-border) !important;
        }}
        .stButton > button:hover {{
            border-color: var(--aso-accent) !important;
            color: var(--aso-accent) !important;
        }}

        /* ---- toggle / checkbox ---- */
        [data-testid="stCheckbox"] label span,
        [data-testid="stToggle"] label span {{
            color: var(--aso-text) !important;
        }}

        /* ---- expanders ---- */
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] details summary,
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] details summary {{
            background-color: var(--aso-card) !important;
            border: 1px solid var(--aso-border) !important;
            border-radius: 8px;
            color: var(--aso-text) !important;
        }}

        /* ---- tables / dataframes ---- */
        [data-testid="stDataFrame"],
        [data-testid="stDataFrame"] > div,
        [data-testid="stDataFrame"] iframe,
        .stDataFrame, .stDataEditor,
        .stDataFrame > div {{
            background-color: var(--aso-card) !important;
            border: 1px solid var(--aso-border) !important;
            border-radius: 8px;
        }}

        /* ---- custom HTML tables ---- */
        .aso-table-wrap {{
            overflow: auto;
            border-radius: 8px;
            border: 1px solid var(--aso-border);
            background: var(--aso-card);
            width: 100%;
            font-size: 0.85rem;
        }}
        .aso-table-wrap table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--aso-card);
            color: var(--aso-text);
        }}
        .aso-table-wrap thead tr {{
            background: var(--aso-sidebar);
        }}
        .aso-table-wrap th {{
            padding: 8px 12px;
            text-align: left;
            font-weight: 600;
            border-bottom: 2px solid var(--aso-border);
            white-space: nowrap;
            color: var(--aso-text);
        }}
        .aso-table-wrap td {{
            padding: 5px 12px;
            border-bottom: 1px solid var(--aso-border);
            color: var(--aso-text);
        }}
        .aso-table-wrap tbody tr:hover td {{
            background: var(--aso-sidebar);
        }}

        /* ---- metric cards ---- */
        [data-testid="stMetric"] {{
            background: var(--aso-card) !important;
            border: 1px solid var(--aso-border) !important;
            border-radius: 8px;
            padding: 10px 12px;
        }}
        [data-testid="stMetricValue"],
        [data-testid="stMetricLabel"] {{
            color: var(--aso-text) !important;
        }}

        /* ---- download button ---- */
        [data-testid="stDownloadButton"] > button {{
            background-color: var(--aso-card) !important;
            color: var(--aso-text) !important;
            border: 1px solid var(--aso-border) !important;
        }}

        /* ---- tabs ---- */
        [data-testid="stTabs"] [role="tablist"] {{
            background-color: var(--aso-card) !important;
            border-bottom: 1px solid var(--aso-border) !important;
        }}
        [data-testid="stTabs"] button[role="tab"] {{
            color: var(--aso-muted) !important;
            background-color: transparent !important;
        }}
        [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
            color: var(--aso-text) !important;
            border-bottom: 2px solid var(--aso-accent) !important;
        }}
        [data-testid="stTabsContent"] {{
            background-color: var(--aso-bg) !important;
        }}

        /* ---- info / warning / error banners ---- */
        [data-testid="stAlert"] {{
            background-color: var(--aso-card) !important;
            border: 1px solid var(--aso-border) !important;
            color: var(--aso-text) !important;
        }}

        /* ---- custom components ---- */
        .aso-card {{
            background: var(--aso-card);
            border: 1px solid var(--aso-border);
            border-radius: 12px;
            padding: 14px 16px;
        }}
        .aso-chip {{
            display: inline-block; padding: .25rem .6rem; border-radius: 9999px;
            font-size: .78rem; font-weight: 600; letter-spacing: .2px;
            background: var(--aso-chip-bg); color: var(--aso-chip-text) !important;
        }}
        .aso-note {{
            border-radius: 8px; padding: 10px 12px;
            background: var(--aso-card); border: 1px solid var(--aso-border);
            color: var(--aso-text); line-height: 1.35; white-space: pre-wrap;
        }}
        .aso-section-title {{ margin: 0 0 6px 0; color: var(--aso-text); letter-spacing: .2px; }}
        .aso-muted {{ color: var(--aso-muted) !important; }}
        .aso-spacer-xxl {{ height: 48px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    settings_expander = st.expander("⚙️ Settings", expanded=False)

with settings_expander:
    st.markdown("#### 🎨 Theme")
    theme_name = st.selectbox("Theme", list(PALETTES.keys()), index=0, key="sel_palette")
_palette = PALETTES[theme_name]


inject_theme(_palette)
COLOR_SEQ    = _palette["plot"]
PLOT_BG      = _palette["card_bg"]
PLOT_PAPER   = _palette["bg"]
PLOT_TEXT    = _palette["text"]
PLOT_GRID    = _palette["border"]

_TABLE_ROW_LIMIT = 300  # cap HTML table rows to keep page responsive

def show_table(df: pd.DataFrame, height: int = 440):
    """Render a DataFrame as a themed HTML table that respects the active CSS palette."""
    if df is None or df.empty:
        st.info("No data to display.")
        return
    total = len(df)
    display = df.head(_TABLE_ROW_LIMIT).copy()
    for col in display.select_dtypes(include="float").columns:
        display[col] = display[col].map(lambda v: f"{v:,.2f}" if pd.notna(v) else "—")
    display = display.fillna("—")
    html = display.to_html(index=False, escape=True, border=0, classes="")
    if total > _TABLE_ROW_LIMIT:
        st.caption(f"Showing first {_TABLE_ROW_LIMIT} of {total} rows — download CSV for full data.")
    st.markdown(
        f'<div class="aso-table-wrap" style="max-height:{height}px;overflow:auto;">{html}</div>',
        unsafe_allow_html=True,
    )

# ====================== Header ======================
st.title("🧬 SafeSense")
st.markdown("##### ASO Human Trials Global Safety Atlas")


# ====================== DB selection ======================
DB_FILENAME = "final_so_far_13_11.db"
DBP = Path(DB_FILENAME)

if not DBP.exists():
    st.error(f"Database file '{DB_FILENAME}' not found in repository. Please ensure it is committed.")
    st.stop()

# ====================== DB helpers ======================
def run_sql(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    with sqlite3.connect(DBP) as con:
        return pd.read_sql(sql, con, params=params or {})

def _get_columns_cached(db_path: str, table: str) -> List[Tuple[str, str]]:
    try:
        with sqlite3.connect(db_path) as con:
            cur = con.execute(f'PRAGMA table_info("{table}")')
            return [(r[1], (r[2] or "").upper()) for r in cur.fetchall()]
    except Exception:
        return []

def get_columns(table: str) -> List[str]:
    return [n for n, _ in _get_columns_cached(str(DBP), table)]

def get_col_type(table: str, col: str) -> str:
    for n, t in _get_columns_cached(str(DBP), table):
        if n == col: return t
    return ""

def col_exists(table: str, col: str) -> bool:
    return col in get_columns(table)

def table_exists(name: str) -> bool:
    try:
        df = run_sql("SELECT 1 FROM sqlite_master WHERE type='table' AND name=:n;", {"n": name})
        return not df.empty
    except Exception:
        return False

# ====================== References Loader (ReferencesV3.xlsx) ======================
refs_xlsx_path = Path("ReferencesV3.xlsx")

def _load_references_xlsx():
    """Load ReferencesV3.xlsx into the DB as references_v2 table, matching drug -> treatment_id."""
    df = pd.read_excel(refs_xlsx_path)
    treatments = pd.read_sql('SELECT treatment_id, generic_name FROM treatments', sqlite3.connect(DBP))
    name_to_id = {row["generic_name"].strip().lower(): row["treatment_id"] for _, row in treatments.iterrows()}

    manual_map = {
        "\u200bIONIS-DGAT2(Rx) - ION224 (CS-2)": "ionis-dgat2-rx-ion224-cs-2",
        "AZD8233":                                "azd8233-ion449",
        "Baliforsen (ISIS 598769":                "baliforsen-isis-598769",
        "Casimersen":                             "casimersen-amondys45",
        "Custirsen":                              "custirsen-ogx-011",
        "Danvatirsen":                            "danvatirsen-azd9150-ionis-stat3-2-5r-isis-481464",
        "Eplontersen":                            "eplontersen-wainua",
        "Eteplirsen":                             "eteplirsen-exon-dys51",
        "Golodirsen":                             "golodirsen-vyondys53",
        "Olezarsen":                              "olezarsen-tryngolza",
        "Pelacarsen":                             "pelacarsen-ionis-akcea-apo-a-lr-ionis-681257-isis-681257",
        "Salanersen/BIIB115/ION306":              "salanersen-biib115-ion306",
        "Sepofarsen(QR\u2011110)":                "sepofarsen-qr110",
        "SHJ002 (Anti\u2013microRNA-328 Ophthalmic Solution)": "shj002-anti-microrna-328-ophthalmic-solution",
        "Tadnersen":                              "tadnersen-biib078",
        "Tofersen":                               "tofersen-qalsody",
        "Ultevursen (QR\u2011421a)":              "ultevursen-qr421a",
        "Viltolarsen":                            "viltolarsen-viltepso",
        "Sefaxersen":                             "sefaxersen-ionis-fb-lrx-ro7434656",
        "Volanesorsen":                           "volanesorsen-waylivra",
    }

    def resolve(drug_val):
        if pd.isna(drug_val):
            return None
        s = str(drug_val).strip()
        if s in manual_map:
            return manual_map[s]
        norm = s.lower().replace("\xa0", " ").replace("\u200b", "").strip()
        return name_to_id.get(norm)

    df["treatment_id"] = df["drug"].apply(resolve)
    df = df.rename(columns={
        "reference": "ref_value",
        "category":  "ref_type",
        "Type":      "ref_source_type",
    })
    df = df[df["treatment_id"].notna()]
    keep = ["treatment_id", "ref_type", "ref_source_type", "ref_value"]
    df = df[[c for c in keep if c in df.columns]]
    with sqlite3.connect(DBP) as con:
        df.to_sql("references_v2", con, if_exists="replace", index=False)

if refs_xlsx_path.exists():
    try:
        # Reload if table missing or schema doesn't have ref_source_type (V3 column)
        if not table_exists("references_v2") or not col_exists("references_v2", "ref_source_type"):
            _load_references_xlsx()
    except Exception:
        pass

# ====================== Treatments CSV Loader ======================
treatments_csv_path = Path("treatments_cursor.csv")

def _load_treatments_csv():
    df = pd.read_csv(treatments_csv_path)
    # Fix mojibake on conjugate column (Â\xa0 -> \xa0)
    df.columns = [c.replace("Â\xa0", "\xa0").replace("Â ", " ") for c in df.columns]
    with sqlite3.connect(DBP) as con:
        df.to_sql("treatments", con, if_exists="replace", index=False)

if treatments_csv_path.exists():
    try:
        # Reload if treatments table row count differs from CSV
        csv_row_count = sum(1 for _ in open(treatments_csv_path)) - 1
        db_row_count = 0
        try:
            db_row_count = run_sql('SELECT COUNT(*) AS n FROM treatments')["n"].iloc[0]
        except Exception:
            pass
        if int(db_row_count) != csv_row_count:
            _load_treatments_csv()
    except Exception:
        pass

# ====================== CSV Loader ======================
csv_path = Path("adverse_events_gold_cursor.csv")
loaded_custom = False

if csv_path.exists():
    try:
        # Check if we already loaded it this session or if table exists
        if table_exists("adverse_events_custom_upload"):
            # Check if columns are correct
            existing_cols = get_columns("adverse_events_custom_upload")
            required_db_cols = ["pts_observed_severe_n", "pts_observed_severe_percent"]
            if all(c in existing_cols for c in required_db_cols):
                loaded_custom = True
            else:
                loaded_custom = False # Force reload
        
        if not loaded_custom:
            # Auto-load on startup if not present or missing columns
            csv_df = pd.read_csv(csv_path)
            
            # Check required columns
            required_csv_cols = ["treatment_id", "ae_term", "ae_group", "treated", "observed", "percent"]
            missing = [c for c in required_csv_cols if c not in csv_df.columns]
            
            if not missing:
                # Rename to match DB schema
                rename_map = {
                    "treated": "total_treated",
                    "observed": "pts_observed_n",
                    "percent": "pts_observed_percent",
                    "observed_severe": "pts_observed_severe_n",
                    "percent_severe": "pts_observed_severe_percent"
                }
                # Only rename columns that exist
                csv_df = csv_df.rename(columns={k: v for k, v in rename_map.items() if k in csv_df.columns})
                
                # Add missing schema columns with defaults
                if "source_type" not in csv_df.columns:
                    csv_df["source_type"] = "Custom CSV"
                if "severity" not in csv_df.columns:
                    csv_df["severity"] = None
                
                # Write to DB as a new table
                table_name = "adverse_events_custom_upload"
                with sqlite3.connect(DBP) as con:
                    csv_df.to_sql(table_name, con, if_exists="replace", index=False)
                loaded_custom = True
    except Exception:
        pass


# ====================== AE table autodetect ======================
candidate_ae = [
    "adverse_events_custom_upload", # Prioritize custom upload if exists
    "adverse_events_normalized_v8_validated",
    "adverse_events_normalized_v8v",
    "adverse_events_normalized_v8",
    "adverse_events_normalized_v7v",
    "adverse_events_normalized_v7",
    "adverse_events_13_11"
]
available_ae = [t for t in candidate_ae if table_exists(t)]
if not available_ae:
    st.error("No AE table found. Expected one of: " + ", ".join(candidate_ae))
    st.stop()
AE_TABLE = available_ae[0]

ALIASES = {AE_TABLE: "ae", "treatments": "t", "approvals": "ap", "refs": "rf", "trials": "tr"}
JOINS: Dict[Tuple[str, str], str] = {
    (AE_TABLE, "treatments"): "ae.treatment_id = t.treatment_id",
    ("approvals", "treatments"): "ap.treatment_id = t.treatment_id",
    ("refs", "treatments"): "rf.treatment_id = t.treatment_id",
    ("trials", "treatments"): "tr.treatment_id = t.treatment_id",
}

# ===== Numeric columns =====
def _ae_num_cast(col: str) -> Optional[str]:
    if not col_exists(AE_TABLE, col): return None
    ctype = (get_col_type(AE_TABLE, col) or "").upper()
    is_numeric = any(k in ctype for k in ("INT", "REAL", "NUM", "DEC"))
    if is_numeric:
        return f'CAST(ae."{col}" AS FLOAT)'
    if col == "pts_observed_percent":
        return f'CAST(REPLACE(ae."{col}", "%", "") AS FLOAT)'
    return f'CAST(ae."{col}" AS FLOAT)'

AE_NUM: Dict[str, str] = {}
for col in ("total_treated", "pts_observed_n", "pts_observed_percent", "pts_observed_severe_n", "pts_observed_severe_percent"):
    expr = _ae_num_cast(col)
    if expr: AE_NUM[col] = expr

# Force strict mapping for custom upload to ensure columns are picked up even if schema detection lags
if AE_TABLE == "adverse_events_custom_upload":
    AE_NUM["pts_observed_severe_n"] = 'CAST(ae."pts_observed_severe_n" AS FLOAT)'
    AE_NUM["pts_observed_severe_percent"] = 'CAST(ae."pts_observed_severe_percent" AS FLOAT)'
    AE_NUM["total_treated"] = 'CAST(ae."total_treated" AS FLOAT)'
    AE_NUM["pts_observed_n"] = 'CAST(ae."pts_observed_n" AS FLOAT)'
    AE_NUM["pts_observed_percent"] = 'CAST(ae."pts_observed_percent" AS FLOAT)'

NUMERIC_CAST_EXPR: Dict[str, str] = {}
if col_exists(AE_TABLE, "total_treated"):
    NUMERIC_CAST_EXPR["Treated Population"] = _ae_num_cast("total_treated")
if col_exists(AE_TABLE, "pts_observed_n"):
    NUMERIC_CAST_EXPR["AE Reports (Incidence)"] = _ae_num_cast("pts_observed_n")
if col_exists(AE_TABLE, "pts_observed_percent"):
    NUMERIC_CAST_EXPR["AE Reports (Rate)"] = _ae_num_cast("pts_observed_percent")
if col_exists("treatments", "chem_length_nt"):
    NUMERIC_CAST_EXPR["Nucleotide Length"] = 'CAST(t."chem_length_nt" AS FLOAT)'

# ====================== Dimensions ======================
DIMENSIONS: Dict[str, Dict[str, str]] = {}

DIMENSIONS["Publication Source Type"] = {
    "expr": (
        "CASE ae.source_type "
        "WHEN 'P' THEN 'Peer Review' "
        "WHEN 'N' THEN 'Non-Peer Review' "
        "WHEN 'G' THEN 'Gray Literature' "
        "WHEN 'F' THEN 'FAERS Database' "
        "WHEN 'L' THEN 'Labeling' "
        "ELSE ae.source_type END"
    ),
    "table": AE_TABLE,
}
if col_exists(AE_TABLE, "ae_term"):
    DIMENSIONS["Adverse Event"] = {"expr": 'ae."ae_term"', "table": AE_TABLE}
if col_exists(AE_TABLE, "ae_group"):
    DIMENSIONS["Adverse Event Category"] = {"expr": 'ae."ae_group"', "table": AE_TABLE}
if col_exists(AE_TABLE, "severity"):
    # Only add Severity dimension if the column has actual non-null values
    try:
        _sev_count = run_sql(f'SELECT COUNT(*) AS n FROM "{AE_TABLE}" WHERE "severity" IS NOT NULL AND TRIM(CAST("severity" AS TEXT)) <> ""')["n"].iloc[0]
    except Exception:
        _sev_count = 0
    if int(_sev_count) > 0:
        DIMENSIONS["Severity"] = {
            "expr": (
                'CASE CAST(ae."severity" AS INTEGER) '
                "WHEN 0 THEN 'Mild' "
                "WHEN 1 THEN 'Severe' "
                'ELSE TRIM(COALESCE(ae."severity", "")) END'
            ),
            "table": AE_TABLE,
        }
if col_exists(AE_TABLE, "total_treated"):
    DIMENSIONS["Treated Population"] = {"expr": 'ae."total_treated"', "table": AE_TABLE}
if col_exists(AE_TABLE, "pts_observed_n"):
    DIMENSIONS["AE Reports (Incidence)"] = {"expr": 'ae."pts_observed_n"', "table": AE_TABLE}
if col_exists(AE_TABLE, "pts_observed_percent"):
    DIMENSIONS["AE Reports (Rate)"] = {"expr": 'ae."pts_observed_percent"', "table": AE_TABLE}

# Treatments
# Handle schema variations dynamically
struct_col = 't."Structure"'
if col_exists("treatments", "structure "):
    struct_col = 't."structure "'
elif col_exists("treatments", "Structure"):
    struct_col = 't."Structure"'

conj_col = 't."conjugate"'
if col_exists("treatments", "conjugate\xa0"):
    conj_col = 't."conjugate\xa0"'
elif col_exists("treatments", "conjugate"):
    conj_col = 't."conjugate"'

DIMENSIONS.update({
    "Drug Name": {"expr": 't."generic_name"', "table": "treatments"},
    "Target Gene": {"expr": 't."Target gene"', "table": "treatments"},
    "Mechanism of Action": {"expr": 't."mechanism_summary"', "table": "treatments"},
    "Route of Administration": {"expr": 't."route"', "table": "treatments"},
    "Backbone": {"expr": 't."backbone"', "table": "treatments"},
    "Sugar Modification": {"expr": 't."sugar"', "table": "treatments"},
    "Chemical Structure": {"expr": struct_col, "table": "treatments"},
    "Gapmer Configuration": {"expr": 't."gapmer_notes"', "table": "treatments"},
    "Conjugate Status": {"expr": conj_col, "table": "treatments"},
    "Nucleotide Length": {"expr": 't."chem_length_nt"', "table": "treatments"},
    "Treatment Indication": {"expr": 't."treatment_group"', "table": "treatments"},
})

# Approvals / Trials
DIMENSIONS.update({
    "Clinical Trial Phase": {"expr": 'tr."phase"', "table": "trials"},
})

def numeric_expr_for(label: str) -> Optional[str]:
    return NUMERIC_CAST_EXPR.get(label)

# ====================== Metrics ======================
METRICS: Dict[str, Dict[str, str]] = {"Row Count": {"agg": "COUNT", "expr": "*"}}
if "pts_observed_n" in AE_NUM:
    METRICS["Total AE Incidence"] = {"agg": "SUM", "expr": AE_NUM["pts_observed_n"]}
if "pts_observed_percent" in AE_NUM:
    METRICS["Accumulated AE Rate (%)"] = {"agg": "AVG", "expr": AE_NUM["pts_observed_percent"]}
    METRICS["Mean AE Rate (%)"] = {"agg": "AVG", "expr": AE_NUM["pts_observed_percent"]}
if "total_treated" in AE_NUM:
    METRICS["Avg. Treated Population"] = {"agg": "AVG", "expr": AE_NUM["total_treated"], "dedup_by": "ae.treatment_id"}
    METRICS["Treated Population"] = {"agg": "SUM", "expr": AE_NUM["total_treated"], "dedup_by": "ae.treatment_id"}

# ====================== SQL builders ======================
def resolve_tables(fields: List[str], metrics: List[str], filters: Dict[str, dict]) -> List[str]:
    needed: Set[str] = set()

    def uses(alias: str, sql: str) -> bool:
        return f"{alias}." in sql

    for f in fields:
        if f in DIMENSIONS:
            info = DIMENSIONS[f]
            needed.add(info["table"])
            expr = info["expr"]
            if uses("ae", expr): needed.add(AE_TABLE)
            if uses("t",  expr): needed.add("treatments")
            if uses("tr", expr): needed.add("trials")
            if uses("ap", expr): needed.add("approvals")
            if uses("rf", expr): needed.add("refs")

    for m in metrics:
        if m in METRICS:
            expr = METRICS[m]["expr"]
            if uses("ae", expr): needed.add(AE_TABLE)
            if uses("tr", expr): needed.add("trials")
            if uses("ap", expr): needed.add("approvals")
            if uses("rf", expr): needed.add("refs")

    for f, spec in filters.items():
        if f in DIMENSIONS:
            info = DIMENSIONS[f]
            needed.add(info["table"])
            expr = info["expr"]
            if uses("ae", expr): needed.add(AE_TABLE)
            if uses("t",  expr): needed.add("treatments")
            if uses("tr", expr): needed.add("trials")
            if uses("ap", expr): needed.add("approvals")
            if uses("rf", expr): needed.add("refs")

    if any(t in needed for t in [AE_TABLE, "approvals", "refs", "trials"]):
        needed.add("treatments")

    order = [AE_TABLE, "trials", "approvals", "refs", "treatments"]
    return [t for t in order if t in needed]

def build_from_join(used_tables: List[str]) -> str:
    if not used_tables: return ""
    base = used_tables[-1] if used_tables[-1] == "treatments" else used_tables[0]
    sql = f'FROM "{base}" {ALIASES[base]}'
    used = {base}
    pending = [t for t in used_tables if t != base]

    while pending:
        progressed = False
        for t in list(pending):
            jc = None
            for u in list(used):
                if (t, u) in JOINS: jc = (t, u, JOINS[(t, u)]); break
                if (u, t) in JOINS: jc = (u, t, JOINS[(u, t)]); break
            if jc:
                sql += f'\nINNER JOIN "{t}" {ALIASES[t]} ON {jc[2]}'
                used.add(t); pending.remove(t); progressed = True
        if not progressed:
            raise ValueError(f"Cannot connect tables: {used_tables}")
    return sql

def _series_to_frame(s: pd.Series, name: str, group_cols: List[str]) -> pd.DataFrame:
    if group_cols:
        return s.reset_index(name=name)
    return pd.DataFrame([{name: s.iloc[0] if len(s) else 0}])

def aggregate_metrics_from_rows(raw_df: pd.DataFrame, group_cols: List[str], selected_metrics: List[str]) -> pd.DataFrame:
    if raw_df.empty:
        out_cols = list(group_cols) + list(selected_metrics)
        return pd.DataFrame(columns=out_cols)

    if group_cols:
        result = raw_df[group_cols].drop_duplicates().reset_index(drop=True)
    else:
        result = pd.DataFrame([{}])

    def merge_metric(metric_df: pd.DataFrame) -> None:
        nonlocal result
        if group_cols:
            result = result.merge(metric_df, on=group_cols, how="left")
        else:
            for col in metric_df.columns:
                result[col] = metric_df.iloc[0][col]

    if "Row Count" in selected_metrics:
        count_s = raw_df.groupby(group_cols, dropna=False).size() if group_cols else pd.Series([len(raw_df)])
        merge_metric(_series_to_frame(count_s, "Row Count", group_cols))

    if "Total AE Incidence" in selected_metrics:
        inc_s = (
            raw_df.groupby(group_cols, dropna=False)["__metric_incidence"].sum()
            if group_cols else pd.Series([raw_df["__metric_incidence"].sum()])
        )
        merge_metric(_series_to_frame(inc_s, "Total AE Incidence", group_cols))

    if "Accumulated AE Rate (%)" in selected_metrics:
        # Use the full displayed grouping context for the denominator so
        # category-by-group views (e.g. sugar x AE category) match the
        # expected stacked-rate plots instead of pooling all categories together.
        denom_group_cols = list(group_cols)
        denom_dedup_keys = denom_group_cols + ["__treatment_id"] if denom_group_cols else ["__treatment_id"]
        denom_df = (
            raw_df[denom_dedup_keys + ["__total_treated"]]
            .groupby(denom_dedup_keys, dropna=False)["__total_treated"]
            .max()
            .reset_index()
        )
        if denom_group_cols:
            denom_by_group = denom_df.groupby(denom_group_cols, dropna=False)["__total_treated"].sum().reset_index(name="__denom_treated")
        else:
            denom_by_group = pd.DataFrame([{"__denom_treated": denom_df["__total_treated"].sum()}])

        num_s = (
            raw_df.groupby(group_cols, dropna=False)["__metric_incidence"].sum()
            if group_cols else pd.Series([raw_df["__metric_incidence"].sum()])
        )
        num_df = _series_to_frame(num_s, "__numerator", group_cols)
        if denom_group_cols:
            rate_df = num_df.merge(denom_by_group, on=denom_group_cols, how="left")
        else:
            rate_df = num_df.copy()
            rate_df["__denom_treated"] = float(denom_by_group["__denom_treated"].iloc[0]) if not denom_by_group.empty else 0.0
        rate_df["Accumulated AE Rate (%)"] = rate_df.apply(
            lambda r: (float(r["__numerator"]) / float(r["__denom_treated"]) * 100.0) if float(r["__denom_treated"] or 0) > 0 else 0.0,
            axis=1,
        )
        keep = list(group_cols) + ["Accumulated AE Rate (%)"]
        merge_metric(rate_df[keep])

    if "Mean AE Rate (%)" in selected_metrics:
        rate_s = (
            raw_df.groupby(group_cols, dropna=False)["__metric_row_rate"].mean()
            if group_cols else pd.Series([raw_df["__metric_row_rate"].mean()])
        )
        merge_metric(_series_to_frame(rate_s, "Mean AE Rate (%)", group_cols))

    if "Avg. Treated Population" in selected_metrics or "Treated Population" in selected_metrics:
        treated_dedup_keys = group_cols + ["__treatment_id"] if group_cols else ["__treatment_id"]
        treated_df = (
            raw_df[treated_dedup_keys + ["__total_treated"]]
            .groupby(treated_dedup_keys, dropna=False)["__total_treated"]
            .max()
            .reset_index()
        )
        if "Avg. Treated Population" in selected_metrics:
            avg_s = (
                treated_df.groupby(group_cols, dropna=False)["__total_treated"].mean()
                if group_cols else pd.Series([treated_df["__total_treated"].mean()])
            )
            merge_metric(_series_to_frame(avg_s, "Avg. Treated Population", group_cols))
        if "Treated Population" in selected_metrics:
            sum_s = (
                treated_df.groupby(group_cols, dropna=False)["__total_treated"].sum()
                if group_cols else pd.Series([treated_df["__total_treated"].sum()])
            )
            merge_metric(_series_to_frame(sum_s, "Treated Population", group_cols))

    ordered_cols = list(group_cols) + [m for m in selected_metrics if m in result.columns]
    return result[ordered_cols]

@st.cache_data(show_spinner=False)
def _distinct_cached(db_path: str, table: str, expr: str, alias: str = "ae") -> List[str]:
    q = f"""
        SELECT DISTINCT TRIM(COALESCE({expr}, '')) AS v
        FROM "{table}" {alias}
        WHERE {expr} IS NOT NULL AND TRIM(CAST({expr} AS TEXT)) <> ''
        ORDER BY 1
        LIMIT 1000
    """
    try:
        with sqlite3.connect(db_path) as con:
            df = pd.read_sql(q, con)
        return [str(x) for x in df["v"].tolist()]
    except Exception:
        return []

def distinct_for_display(col_label: str) -> List[str]:
    if col_label not in DIMENSIONS: return []
    info = DIMENSIONS[col_label]
    expr, table = info["expr"], info["table"]
    # For complex expressions (CASE, functions) use the full expr with the table alias.
    # For simple column refs strip the alias prefix and query the raw table.
    alias_map = {AE_TABLE: "ae", "treatments": "t", "trials": "tr",
                 "approvals": "ap", "refs": "rf"}
    alias = alias_map.get(table, "ae")
    return _distinct_cached(str(DBP), table, expr, alias)

# ====================== Build analysis UI ======================
st.markdown("### 📊 Build analysis")
dim_choices = list(DIMENSIONS.keys())
group_dim_choices = [
    d for d in dim_choices
    if d not in {"Publication Source Type", "Treated Population", "AE Reports (Incidence)", "AE Reports (Rate)"}
]
metric_choices = list(METRICS.keys())

c1, c2 = st.columns([2, 1])
with c1:
    group_by = st.multiselect("Group By (Max 3)", group_dim_choices, max_selections=3, key="ms_group_by")
with c2:
    stratify_by = st.selectbox("Stratify By", ["(none)"] + group_dim_choices, index=0, key="sel_stratify")

metric_sel = st.multiselect("Select Metrics", metric_choices, default=["Row Count"], max_selections=4, key="ms_metrics")

_severe_capable = ("pts_observed_severe_n" in AE_NUM and "pts_observed_severe_percent" in AE_NUM)
only_severe = st.toggle("Only Severe AEs", value=False, key="tgl_only_severe", disabled=not _severe_capable)

# When Only Severe AEs is on, redirect AE Incidence/Rate metrics to their severe equivalents
if only_severe:
    if "Total AE Incidence" in METRICS and "pts_observed_severe_n" in AE_NUM:
        METRICS["Total AE Incidence"] = {"agg": "SUM", "expr": AE_NUM["pts_observed_severe_n"]}
    if "Accumulated AE Rate (%)" in METRICS and "pts_observed_severe_percent" in AE_NUM:
        METRICS["Accumulated AE Rate (%)"] = {"agg": "AVG", "expr": AE_NUM["pts_observed_severe_percent"]}
    if "Mean AE Rate (%)" in METRICS and "pts_observed_severe_percent" in AE_NUM:
        METRICS["Mean AE Rate (%)"] = {"agg": "AVG", "expr": AE_NUM["pts_observed_severe_percent"]}

# -------- Advanced Filters --------
with st.expander("🎛️ Filters (optional)", expanded=False):
    filter_specs: Dict[str, dict] = {}
    filter_cols = st.multiselect("Filter Columns", dim_choices, default=[], key="ms_filter_cols")
    for col in filter_cols[:8]:
        info = DIMENSIONS[col]
        expr = info["expr"]

        st.markdown(f"**{col}**")
        f1, f2, f3 = st.columns([1.1, 2.2, 1.2])
        mode = f1.selectbox(
            "Mode", ["Include", "Exclude", "Greater than", "Less than", "Between"],
            key=f"mode_{col}"
        )
        exclude_null = f3.checkbox("Exclude Empty/Null", value=True, key=f"nonnull_{col}")

        spec = {"mode": mode, "exclude_null": exclude_null}

        if mode in ("Include", "Exclude"):
            vals = distinct_for_display(col)
            chosen = f2.multiselect("Values", vals, key=f"vals_{col}")
            if chosen:
                spec["values"] = chosen
        else:
            # numeric comparison modes
            def numeric_expr_for(label: str) -> Optional[str]:
                return NUMERIC_CAST_EXPR.get(label)
            num_expr = numeric_expr_for(col)
            if not num_expr:
                f2.warning("Numeric comparison not available for this column.")
            else:
                if mode == "Between":
                    low = f2.number_input("Minimum", value=0.0, key=f"min_{col}")
                    high = f2.number_input("Maximum", value=0.0, key=f"max_{col}")
                    spec["min"] = low; spec["max"] = high
                elif mode == "Greater than":
                    val = f2.number_input("Value", value=0.0, key=f"gt_{col}")
                    spec["value"] = val
                elif mode == "Less than":
                    val = f2.number_input("Value", value=0.0, key=f"lt_{col}")
                    spec["value"] = val

        filter_specs[col] = spec

c3, c4 = st.columns([1, 1])
with c3:
    limit_rows = st.number_input("Row limit", min_value=10, max_value=20000, value=1000, step=10, key="num_row_limit")
with c4:
    show_sql = st.toggle("Show generated SQL", value=False, key="tgl_show_sql")

# Compose grouping list
gb_all: List[str] = list(group_by)
if stratify_by != "(none)" and stratify_by not in gb_all:
    if len(gb_all) < 3: gb_all.append(stratify_by)
    else: st.warning("Stratify ignored: already using 3 grouping columns.")

_use_trials_phase_dedup = (
    "Clinical Trial Phase" in gb_all
    or "Clinical Trial Phase" in filter_specs
)

# ====================== Build SQL ======================
used_tables = resolve_tables(gb_all, metric_sel, filter_specs)
try:
    from_join_sql = build_from_join(used_tables)
except Exception as e:
    st.error(str(e)); st.stop()

if _use_trials_phase_dedup and "trials" in used_tables:
    _trials_phase_subq = (
        '(SELECT treatment_id, MAX(phase) AS phase '
        'FROM trials '
        'WHERE phase IS NOT NULL AND TRIM(COALESCE(phase, \'\')) <> \'\' '
        'GROUP BY treatment_id)'
    )
    from_join_sql = from_join_sql.replace('"trials"', _trials_phase_subq, 1)

_use_custom_rate_logic = any(m in metric_sel for m in ["Accumulated AE Rate (%)", "Mean AE Rate (%)"])

select_parts: List[str] = []
group_positions: List[int] = []

for i, d in enumerate(gb_all, start=1):
    expr = DIMENSIONS[d]["expr"]
    select_parts.append(f'{expr} AS "{d}"')
    group_positions.append(i)

for m in metric_sel:
    agg  = METRICS[m]["agg"]
    expr = METRICS[m]["expr"]
    if expr == "*":
        select_parts.append(f'{agg}(*) AS "{m}"')
    else:
        select_parts.append(f"{agg}({expr}) AS \"{m}\"")

select_sql = "SELECT " + ", ".join(select_parts) if select_parts else "SELECT COUNT(*) AS \"Row Count\""

# WHERE
where_parts: List[str] = []
params: Dict[str, str] = {}
pidx = 0

def norm_sql(s: str) -> str:
    return f"TRIM(COALESCE({s}, ''))"

for col, spec in filter_specs.items():
    expr = DIMENSIONS[col]["expr"]
    mode = spec.get("mode")
    exclude_null = spec.get("exclude_null", False)

    if exclude_null:
        where_parts.append(f"{expr} IS NOT NULL AND {norm_sql(expr)} <> ''")

    if mode in ("Include", "Exclude"):
        values = spec.get("values") or []
        if values:
            ph = []
            for val in values:
                key = f"p{pidx}"; pidx += 1
                params[key] = (val or "").strip()
                ph.append(f":{key}")
            op = "IN" if mode == "Include" else "NOT IN"
            where_parts.append(f"{norm_sql(expr)} COLLATE NOCASE {op} ({', '.join(ph)})")

    elif mode in ("Greater than", "Less than", "Between"):
        # reuse previously defined numeric_expr_for
        def numeric_expr_for(label: str) -> Optional[str]:
            return NUMERIC_CAST_EXPR.get(label)
        num_expr = numeric_expr_for(col)
        if num_expr:
            if mode == "Greater than":
                key = f"p{pidx}"; pidx += 1
                params[key] = float(spec.get("value", 0))
                where_parts.append(f"{num_expr} > :{key}")
            elif mode == "Less than":
                key = f"p{pidx}"; pidx += 1
                params[key] = float(spec.get("value", 0))
                where_parts.append(f"{num_expr} < :{key}")
            elif mode == "Between":
                key1 = f"p{pidx}"; pidx += 1
                key2 = f"p{pidx}"; pidx += 1
                params[key1] = float(spec.get("min", 0))
                params[key2] = float(spec.get("max", 0))
                where_parts.append(f"({num_expr} >= :{key1} AND {num_expr} <= :{key2})")

where_sql = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
group_sql = "GROUP BY " + ", ".join(str(i) for i in group_positions) if group_positions else ""
order_sql = f"ORDER BY \"{metric_sel[0]}\" DESC" if metric_sel else ("ORDER BY 1" if group_positions else "")

# If any selected metric requires per-treatment deduplication (e.g. Avg/Sum Treated Population),
# wrap the AE table in a subquery that collapses to one row per treatment_id.
# We include ALL ae columns referenced anywhere in the query so WHERE clauses still work.
_dedup_metrics = [m for m in metric_sel if METRICS[m].get("dedup_by")]
if _dedup_metrics and group_positions and AE_TABLE in used_tables:
    import re as _re
    # Collect every ae."col" reference from select + where + group
    _full_sql_so_far = select_sql + " " + where_sql + " " + group_sql
    _all_ae_cols = set(_re.findall(r'ae\."(\w+)"', _full_sql_so_far))
    _all_ae_cols.discard("treatment_id")
    if _all_ae_cols:
        _inner_cols = ", ".join([f'MAX("{c}") AS "{c}"' for c in sorted(_all_ae_cols)])
        _dedup_subq = (
            f"(SELECT treatment_id, {_inner_cols} "
            f"FROM {AE_TABLE} GROUP BY treatment_id)"
        )
        from_join_sql = from_join_sql.replace(f'"{AE_TABLE}"', _dedup_subq, 1)

if _use_custom_rate_logic:
    raw_select_parts = []
    for d in gb_all:
        expr = DIMENSIONS[d]["expr"]
        raw_select_parts.append(f'{expr} AS "{d}"')
    raw_select_parts.extend([
        'ae."treatment_id" AS "__treatment_id"',
        f'{AE_NUM.get("total_treated", "0")} AS "__total_treated"',
    ])
    metric_incidence_expr = AE_NUM.get("pts_observed_severe_n" if only_severe else "pts_observed_n", "0")
    metric_rate_expr = AE_NUM.get("pts_observed_severe_percent" if only_severe else "pts_observed_percent", "NULL")
    raw_select_parts.append(f'{metric_incidence_expr} AS "__metric_incidence"')
    raw_select_parts.append(f'{metric_rate_expr} AS "__metric_row_rate"')
    raw_sql = f"""
SELECT {", ".join(raw_select_parts)}
{from_join_sql}
{where_sql}
""".strip()
    final_sql = raw_sql
else:
    final_sql = f"""
{select_sql}
{from_join_sql}
{where_sql}
{group_sql}
{order_sql}
LIMIT {int(limit_rows)};
""".strip()

if show_sql:
    with st.expander("🧾 Generated SQL", expanded=True):
        st.code(final_sql, language="sql")
        if _use_custom_rate_logic:
            st.caption("`Accumulated AE Rate (%)` uses grouped incidence over deduplicated treated population, while `Mean AE Rate (%)` averages the raw AE-row percentage values.")
        with st.expander("Filter debug", expanded=False):
            st.write("Filter specs:", filter_specs)
            st.write("SQL params:", params)

# ====================== Run query ======================
try:
    if _use_custom_rate_logic:
        raw_df = run_sql(final_sql, params)
        df = aggregate_metrics_from_rows(raw_df, gb_all, metric_sel)
        if metric_sel and metric_sel[0] in df.columns:
            df = df.sort_values(by=metric_sel[0], ascending=False, na_position="last")
        elif gb_all:
            df = df.sort_values(by=gb_all)
        df = df.head(int(limit_rows)).reset_index(drop=True)
    else:
        df = run_sql(final_sql, params)
except Exception as e:
    st.exception(e); st.stop()

# ====================== Plot helper ======================
def render_plotly(fig):
    fig.update_xaxes(automargin=True, gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, color=PLOT_TEXT)
    fig.update_yaxes(automargin=True, gridcolor=PLOT_GRID, zerolinecolor=PLOT_GRID, color=PLOT_TEXT)
    fig.update_layout(
        height=560,
        margin=dict(l=30, r=30, t=60, b=170),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
            font=dict(color=PLOT_TEXT),
        ),
        paper_bgcolor=PLOT_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(color=PLOT_TEXT),
    )
    if not fig.data or len(fig.data) == 0:
        st.info("Nothing to plot with the current selections (no non-null values after filtering).")
        return
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

# ====================== Results ======================
st.markdown("### 📈 Results")

if df.empty:
    st.info("No rows returned. Try relaxing filters or changing dimensions/metrics.")
else:
    metric_cols_present = [c for c in df.columns if c in METRICS.keys()]
    if metric_cols_present:
        try:
            summary = df[metric_cols_present].sum(numeric_only=True)
            csm = st.columns(min(3, len(metric_cols_present)))
            for i, col in enumerate(metric_cols_present[:3]):
                csm[i].metric(col, f"{summary[col]:,.2f}")
        except Exception:
            pass

    show_table(df, height=440)
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name="aso_analytics.csv", mime="text/csv")

# ====================== Chart ======================
st.markdown("### 🎨 Chart")

if df.empty:
    st.info("No data to chart.")
else:
    chart_df = edited_df if "edited_df" in locals() else df

    dims = [c for c in chart_df.columns if c in gb_all]
    output_metric_cols = [c for c in chart_df.columns if c in METRICS.keys()]
    dim_labels = {d: d for d in gb_all}

    cc1, cc2, cc3, cc4 = st.columns([1.2, 1, 1, 1])
    chart_type = cc1.selectbox("Chart type", ["Bar", "Line", "Pie"], index=0)
    if not output_metric_cols:
        st.info("Add at least one metric to draw a chart.")
        st.stop()

    metric_for_chart = cc2.selectbox("Metric", output_metric_cols, index=0)
    sort_x = cc3.toggle("Sort Descending", value=True)
    apply_log1p = cc4.toggle("Apply log1p", value=False)

    work = chart_df.copy()
    work[metric_for_chart] = pd.to_numeric(work[metric_for_chart], errors="coerce")
    work = work.dropna(subset=[metric_for_chart])
    for d in dims:
        work[d] = work[d].astype(str)

    chart_metric_col = metric_for_chart
    chart_metric_label = metric_for_chart
    if apply_log1p:
        chart_metric_col = f"__log1p__{metric_for_chart}"
        work = work[work[metric_for_chart] > -1].copy()
        work[chart_metric_col] = work[metric_for_chart].map(lambda v: math.log1p(v) if pd.notna(v) else None)
        chart_metric_label = f"log1p({metric_for_chart})"

    if work.empty:
        st.info("No valid numeric data for chart.")
        st.stop()

    stack_bars = False
    if chart_type == "Bar" and len(dims) >= 2:
        stack_bars = st.toggle("Stack bars", value=False, key="tgl_stack_bars")

    # ----- PIE -----
    if chart_type == "Pie":
        if len(dims) == 0:
            st.info("Pick at least one grouping column to draw a pie chart.")
        else:
            if apply_log1p:
                st.info("Pie charts use the raw metric. `log1p` is applied only to bar and line charts.")
            names_col = dims[0]
            names_label = dim_labels.get(names_col, names_col)
            donut = st.checkbox("Donut Chart", value=True)

            fig = px.pie(
                work,
                names=names_col,
                values=metric_for_chart,
                hole=0.45 if donut else 0.0,
                color_discrete_sequence=COLOR_SEQ,
                title=f"{metric_for_chart} share by {names_label}"
            )
            render_plotly(fig)

    # ----- BAR / LINE -----
    else:
        # 0 dimensions
        if len(dims) == 0:
            st.info("Select at least one 'Group by' dimension to draw a chart.")

        # 1 dimension
        elif len(dims) == 1:
            x = dims[0]
            x_label = dim_labels.get(x, x)
            if sort_x:
                work = work.sort_values(by=chart_metric_col, ascending=False)

            if chart_type == "Bar":
                fig = px.bar(work, x=x, y=chart_metric_col,
                             color_discrete_sequence=COLOR_SEQ,
                             title=f"{chart_metric_label} by {x_label}")
            else:
                fig = px.line(work, x=x, y=chart_metric_col,
                              color_discrete_sequence=COLOR_SEQ,
                              title=f"{chart_metric_label} by {x_label}")

            fig.update_yaxes(title_text=chart_metric_label)
            render_plotly(fig)

        # 2 dimensions
        elif len(dims) == 2:
            x, color = dims[0], dims[1]
            x_label = dim_labels.get(x, x)
            color_label = dim_labels.get(color, color)
            if sort_x:
                if stack_bars:
                    x_order = (
                        work.groupby(x, dropna=False)[chart_metric_col]
                        .sum()
                        .sort_values(ascending=False)
                        .index
                        .tolist()
                    )
                else:
                    x_order = (
                        work.groupby(x, dropna=False)[chart_metric_col]
                        .max()
                        .sort_values(ascending=False)
                        .index
                        .tolist()
                    )
                work[x] = pd.Categorical(work[x], categories=x_order, ordered=True)
                work = work.sort_values(by=[x, chart_metric_col], ascending=[True, False])

            if chart_type == "Bar":
                fig = px.bar(work, x=x, y=chart_metric_col, color=color, barmode=("stack" if stack_bars else "group"),
                             color_discrete_sequence=COLOR_SEQ,
                             title=f"{chart_metric_label} by {x_label} and {color_label}")
            else:
                fig = px.line(work, x=x, y=chart_metric_col, color=color,
                              color_discrete_sequence=COLOR_SEQ,
                              title=f"{chart_metric_label} by {x_label} and {color_label}")

            fig.update_yaxes(title_text=chart_metric_label)
            render_plotly(fig)

        # 3 dimensions
        elif len(dims) >= 3:
            x, color, facet = dims[0], dims[1], dims[2]
            x_label = dim_labels.get(x, x)
            color_label = dim_labels.get(color, color)
            facet_label = dim_labels.get(facet, facet)
            if sort_x:
                if chart_type == "Bar" and stack_bars:
                    x_order = (
                        work.groupby(x, dropna=False)[chart_metric_col]
                        .sum()
                        .sort_values(ascending=False)
                        .index
                        .tolist()
                    )
                else:
                    x_order = (
                        work.groupby(x, dropna=False)[chart_metric_col]
                        .max()
                        .sort_values(ascending=False)
                        .index
                        .tolist()
                    )
                work[x] = pd.Categorical(work[x], categories=x_order, ordered=True)
                work = work.sort_values(by=[x, chart_metric_col], ascending=[True, False])

            if chart_type == "Bar":
                fig = px.bar(
                    work, x=x, y=chart_metric_col, color=color, facet_col=facet,
                    facet_col_wrap=3, barmode=("stack" if stack_bars else "group"),
                    color_discrete_sequence=COLOR_SEQ,
                    title=f"{chart_metric_label} by {x_label}, {color_label} (facet: {facet_label})"
                )
            else:
                fig = px.line(
                    work, x=x, y=chart_metric_col, color=color, facet_col=facet,
                    facet_col_wrap=3, color_discrete_sequence=COLOR_SEQ,
                    title=f"{chart_metric_label} by {x_label}, {color_label} (facet: {facet_label})"
                )

            fig.update_yaxes(title_text=chart_metric_label)

            fig.update_layout(
                margin=dict(l=30, r=30, t=60, b=220), height=680,
                paper_bgcolor=PLOT_PAPER, plot_bgcolor=PLOT_BG,
                font=dict(color=PLOT_TEXT),
            )
            render_plotly(fig)



# ====================== Treatment info tab ======================
info_tab = st.tabs(["🧪 Treatment info"])[0]
with info_tab:
    st.markdown('<h4 class="aso-section-title">Select a treatment to view chemistry and evidence</h4>', unsafe_allow_html=True)

    try:
        names_df = run_sql('SELECT DISTINCT TRIM("generic_name") AS name FROM treatments WHERE "generic_name" IS NOT NULL AND TRIM("generic_name")<>"" ORDER BY 1;')
        name_options = names_df["name"].astype(str).tolist()
    except Exception as e:
        name_options = []
        st.warning(f"Could not load treatment names: {e}")

    sel_name = st.selectbox("Generic Name", name_options, key="sel_treatment")

    if sel_name:
        try:
            q_info = (
                'SELECT '
                '  "Target gene"           AS "Target Gene", '
                '  "mechanism_summary"     AS "Mechanism of Action", '
                '  "route"                 AS "Route of Administration", '
                f'  {conj_col.replace("t.", "")}             AS "Conjugate Status", '
                f'  {struct_col.replace("t.", "")}            AS "Chemical Structure", '
                '  "backbone"              AS "Backbone", '
                '  "sugar"                 AS "Sugar Modification", '
                '  "Nof1"                  AS "Single Patient Study (N=1)?", '
                '  "treatment_group"       AS "Treatment Indication", '
                '  "gapmer_notes"          AS "Gapmer Configuration", '
                '  "chem_length_nt"        AS "Nucleotide Length", '
                '  "indication_primary"    AS "Primary Indication" '
                'FROM treatments '
                'WHERE TRIM(LOWER("generic_name")) = TRIM(LOWER(:n)) '
                'ORDER BY rowid DESC LIMIT 1;'
            )
            info_df = run_sql(q_info, {"n": sel_name})
        except Exception as e:
            info_df = pd.DataFrame()
            st.warning(f"Could not load info: {e}")

        if info_df.empty:
            st.info("No info found for this treatment.")
        else:
            row = info_df.iloc[0]

            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Target Gene", str(row.get("Target Gene", "")))
            a2.metric("Mechanism of Action", str(row.get("Mechanism of Action", "")))
            a3.metric("Route of Administration", str(row.get("Route of Administration", "")))
            
            conj_raw = str(row.get("Conjugate Status", "")).strip()
            conj_val = "None" if conj_raw == "N" else conj_raw
            a4.metric("Conjugate Status", conj_val)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Chemical Structure", str(row.get("Chemical Structure", "")))
            b2.metric("Backbone", str(row.get("Backbone", "")))
            
            sugar_raw = str(row.get("Sugar Modification", "")).strip()
            sugar_val = "None" if sugar_raw == "N" else sugar_raw
            b3.metric("Sugar Modification", sugar_val)
            
            val_raw = str(row.get("Single Patient Study (N=1)?", "")).strip()
            # 1/1.0 -> Yes, 0/0.0 -> No
            if val_raw in ("1", "1.0"):
                val = "Yes"
            elif val_raw in ("0", "0.0"):
                val = "No"
            else:
                val = val_raw
            b4.metric("Single Patient Study (N=1)?", val)


            c1, c2, c3 = st.columns(3)
            c1.metric("Treatment Indication", str(row.get("Treatment Indication", "")))

            gapmer_notes = str(row.get("Gapmer Configuration", "") or "").strip()
            if gapmer_notes:
                c2.metric("Gapmer Configuration", gapmer_notes)

            nt_len = row.get("Nucleotide Length", None)
            nt_len = "" if nt_len is None else str(nt_len).strip()
            if nt_len:
                c3.metric("Nucleotide Length", nt_len)

            prim = str(row.get("Primary Indication", "") or "").strip()
            st.markdown("#### Primary Indication")
            if prim:
                st.markdown(f'<div class="aso-note">{prim}</div>', unsafe_allow_html=True)
            else:
                st.info("No primary indication recorded.")

        try:
            refs_source = "references_v2" if table_exists("references_v2") else "refs"
            q_refs = f"""SELECT DISTINCT ref_type AS "Category", ref_value AS "Reference"
                   FROM {refs_source}
                   WHERE treatment_id = (
                       SELECT treatment_id FROM treatments
                       WHERE TRIM(LOWER("generic_name")) = TRIM(LOWER(:n))
                       LIMIT 1
                   )
                   AND ref_type IS NOT NULL AND TRIM(ref_type) <> ''
                   AND ref_value IS NOT NULL AND TRIM(ref_value) <> ''
                   ORDER BY 1, 2;"""
            refs_df = run_sql(q_refs, {"n": sel_name})
        except Exception as e:
            refs_df = pd.DataFrame()
            st.warning(f"Could not load references: {e}")

        st.markdown("#### References")
        if refs_df.empty:
            st.info("No references found for this treatment.")
        else:
            show_table(refs_df, height=400)

        st.markdown("#### Adverse Effects")
        try:
            pts_pct_expr = AE_NUM.get("pts_observed_percent", "NULL")
            total_treated_expr = AE_NUM.get("total_treated", 'ae."total_treated"')
            pts_obs_n_expr = AE_NUM.get("pts_observed_n", 'ae."pts_observed_n"')
            
            pts_obs_sev_n_expr = AE_NUM.get("pts_observed_severe_n", "0")
            pts_obs_sev_pct_expr = AE_NUM.get("pts_observed_severe_percent", "NULL")
            
            q_ae_src = (
                f"""
                SELECT
                    ae.ae_term  AS "Adverse Event",
                    ae.ae_group AS "Adverse Event Category",
                    {total_treated_expr}           AS "Total Treated",
                    {pts_obs_n_expr}               AS "AE Incidence",
                    {pts_pct_expr}                 AS "AE Rate",
                    {pts_obs_sev_n_expr}           AS "Severe AE Incidence",
                    {pts_obs_sev_pct_expr}         AS "Severe AE Rate"
                FROM {AE_TABLE} ae
                WHERE ((ae.treatment_id COLLATE NOCASE IN (
                    SELECT treatment_id FROM treatments
                    WHERE TRIM(LOWER("generic_name")) = TRIM(LOWER(:n))
                ))
                OR (TRIM(ae.treatment_id) COLLATE NOCASE = TRIM(:n) COLLATE NOCASE))
                AND {pts_obs_n_expr} > 0
                ORDER BY 3 DESC;
                """
            )
            ae_by_source_df = run_sql(q_ae_src, {"n": sel_name})

        except Exception as e:
            ae_by_source_df = pd.DataFrame()
            st.warning(f"Could not load adverse effects: {e}")

        if ae_by_source_df.empty:
            st.info("No adverse effects found for this treatment.")
        else:
            show_table(ae_by_source_df, height=400)

        st.markdown("#### Adverse Event Category Distribution")
        try:
            q_ae_group_counts = (
                f"""
                SELECT ae_group AS "Adverse Event Category", COUNT(*) AS rows
                FROM {AE_TABLE}
                WHERE (treatment_id COLLATE NOCASE IN (
                    SELECT treatment_id FROM treatments
                    WHERE TRIM(LOWER("generic_name")) = TRIM(LOWER(:n))
                ))
                OR (TRIM(treatment_id) COLLATE NOCASE = TRIM(:n) COLLATE NOCASE)
                GROUP BY 1
                ORDER BY 2 DESC;
                """
            )
            ae_group_counts_df = run_sql(q_ae_group_counts, {"n": sel_name})
        except Exception as e:
            ae_group_counts_df = pd.DataFrame()
            st.warning(f"Could not load AE group counts: {e}")

        if ae_group_counts_df.empty:
            st.info("No AE groups found for this treatment.")
        else:
            pie = px.pie(
                ae_group_counts_df,
                names="Adverse Event Category",
                values="rows",
                hole=0.45,
                color_discrete_sequence=COLOR_SEQ,
                title="AE Category Share"
            )
            render_plotly(pie)

# ====================== Abbreviations Legend ======================
st.markdown("---")
with st.expander("📖 Abbreviations & Terminology"):
    legend = {
        "Backbone": {
            "PS":    "Phosphorothioate",
            "PMO":   "Phosphorodiamidate Morpholino Oligomers",
            "PO/PS": "Phosphodiester / Phosphorothioate",
        },
        "Gapmer Configuration": {
            "G":  "Gapmer",
            "FM": "Fully Modified",
            "NM": "Non-Modified",
        },
        "Sugar Modification": {
            "N":     "Unmodified DNA (sugar position only)",
            "2'MOE": "2'-O-Methoxyethyl",
            "2'OMe": "2'-O-Methylation",
            "LNA":   "Locked Nucleic Acid",
            "cEt":   "Constrained Ethyl",
            "MIX":   "Mixed Sugar Modification",
        },
    }
    for section, terms in legend.items():
        st.markdown(f"**{section}**")
        rows = "".join(
            f"<tr><td style='padding:3px 16px 3px 0;font-weight:600;white-space:nowrap'>{abbr}</td>"
            f"<td style='padding:3px 0'>{full}</td></tr>"
            for abbr, full in terms.items()
        )
        st.markdown(
            f"<table style='border-collapse:collapse;margin-bottom:10px'>{rows}</table>",
            unsafe_allow_html=True,
        )
