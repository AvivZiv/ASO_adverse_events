# Python 3.9 compatible
from typing import Optional, Dict, List, Tuple, Set
import sqlite3
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# ====================== App setup ======================
st.set_page_config(page_title="ASO Analytics", layout="wide", page_icon="🧬")

# ---------- Theme System (colorful + elegant) ----------
PALETTES = {
    "Aurora": {
        "bg_grad": "linear-gradient(135deg,#0ea5e9 0%,#6366f1 50%,#a855f7 100%)",
        "card_bg": "rgba(255,255,255,.7)",
        "glass_border": "rgba(255,255,255,.35)",
        "text_primary": "#0b1221",
        "muted": "#4b5563",
        "accent": "#7c3aed",
        "chip_bg": "rgba(124,58,237,.12)",
        "chip_text": "#2e1065",
        "plot": ["#0ea5e9", "#7c3aed", "#10b981", "#f59e0b", "#ef4444", "#14b8a6"],
    },
    "Sunset": {
        "bg_grad": "linear-gradient(135deg,#fb7185 0%,#f59e0b 50%,#22c55e 100%)",
        "card_bg": "rgba(255,255,255,.72)",
        "glass_border": "rgba(255,255,255,.38)",
        "text_primary": "#101418",
        "muted": "#475569",
        "accent": "#f97316",
        "chip_bg": "rgba(249,115,22,.12)",
        "chip_text": "#7c2d12",
        "plot": ["#fb7185", "#f59e0b", "#22c55e", "#06b6d4", "#8b5cf6", "#ef4444"],
    },
    "Oceanic": {
        "bg_grad": "linear-gradient(135deg,#14b8a6 0%,#0ea5e9 50%,#22d3ee 100%)",
        "card_bg": "rgba(255,255,255,.76)",
        "glass_border": "rgba(255,255,255,.4)",
        "text_primary": "#0b1221",
        "muted": "#334155",
        "accent": "#0891b2",
        "chip_bg": "rgba(8,145,178,.12)",
        "chip_text": "#083344",
        "plot": ["#14b8a6", "#0ea5e9", "#22d3ee", "#f59e0b", "#8b5cf6", "#ef4444"],
    },
    "Orchid": {
        "bg_grad": "linear-gradient(135deg,#9333ea 0%,#f472b6 50%,#f43f5e 100%)",
        "card_bg": "rgba(255,255,255,.74)",
        "glass_border": "rgba(255,255,255,.38)",
        "text_primary": "#100a1c",
        "muted": "#3f3d56",
        "accent": "#c026d3",
        "chip_bg": "rgba(192,38,211,.12)",
        "chip_text": "#4a044e",
        "plot": ["#9333ea", "#f472b6", "#f43f5e", "#10b981", "#0ea5e9", "#f59e0b"],
    },
}

def inject_theme(palette: Dict[str, str]):
    st.markdown(
        f"""
        <style>
        :root {{
            --aso-bg-grad: {palette['bg_grad']};
            --aso-card-bg: {palette['card_bg']};
            --aso-glass-border: {palette['glass_border']};
            --aso-text: {palette['text_primary']};
            --aso-muted: {palette['muted']};
            --aso-accent: {palette['accent']};
            --aso-chip-bg: {palette['chip_bg']};
            --aso-chip-text: {palette['chip_text']};
            --aso-shadow-1: 0 8px 30px rgba(2,6,23,.20);
            --aso-shadow-2: 0 6px 22px rgba(2,6,23,.12);
        }}
        .block-container {{ padding-top: 0; max-width: 1500px; }}

        .aso-banner {{
            background: var(--aso-bg-grad); color: white; border-radius: 22px;
            padding: 22px 28px; box-shadow: var(--aso-shadow-1);
            margin: 18px 0 12px 0; position: relative; overflow: hidden;
        }}
        .aso-banner h1 {{ font-size: 2.05rem; line-height: 1.1; margin: 0 0 6px 0; }}
        .aso-sub {{ opacity: .95; font-weight: 500; letter-spacing:.2px; }}

        .aso-card {{
            background: var(--aso-card-bg); backdrop-filter: blur(10px);
            border: 1px solid var(--aso-glass-border); border-radius: 18px; box-shadow: var(--aso-shadow-2); padding: 14px 16px;
        }}
        .aso-chip {{
            display:inline-block; padding:.25rem .6rem; border-radius:9999px; font-size:.78rem; font-weight:600;
            letter-spacing:.2px; background: var(--aso-chip-bg); color: var(--aso-chip-text);
        }}
        .aso-note {{
            border-radius: 14px; padding: 10px 12px; background: white;
            border: 1px solid var(--aso-glass-border); box-shadow: var(--aso-shadow-2); color: var(--aso-text);
            line-height: 1.35; white-space: pre-wrap;
        }}
        [data-testid="stMetric"] {{
            border-radius: 12px; background: var(--aso-card-bg); border: 1px solid var(--aso-glass-border);
            padding: 10px 12px; box-shadow: var(--aso-shadow-2); color: var(--aso-text);
        }}
        [data-testid="stDataFrame"] {{ border: 1px solid var(--aso-glass-border); border-radius: 14px; box-shadow: var(--aso-shadow-2); }}
        [data-testid="stExpander"] > details {{ border: 1px solid var(--aso-glass-border); border-radius: 14px; background: var(--aso-card-bg); }}
        .aso-section-title {{ margin: 0 0 6px 0; color: var(--aso-text); letter-spacing:.2px; }}
        .aso-muted {{ color: var(--aso-muted); }}

        .aso-spacer-xxl {{ height: 48px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.markdown("### 🎨 Theme")
    theme_name = st.selectbox("Palette", list(PALETTES.keys()), index=0, key="sel_palette")
inject_theme(PALETTES[theme_name])
COLOR_SEQ = PALETTES[theme_name]["plot"]

# ====================== Header ======================
st.markdown(
    """
    <div class="aso-banner">
        <h1>🧬 SafeSense: ASO Human Trials Global Safety Atlas</h1>
        <div class="aso-sub">A colorful, elegant explorer for treatments, adverse effects, and trials.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ====================== DB selection ======================
DEFAULT_DB_CANDIDATES = [
    "final_so_far_13_11.db",   # the DB inside your repo
]

existing = [str(p) for p in map(Path, DEFAULT_DB_CANDIDATES) if p.exists()]

with st.sidebar:
    st.markdown('<div class="aso-card">', unsafe_allow_html=True)
    st.markdown("#### 📦 Database")
    db_path = st.text_input(
        "SQLite DB path",
        value=(existing[0] if existing else "/Users/avivziv/Downloads/streamlit_db.db"),
        key="db_path_input",
    )
    st.caption("Tip: Update the path if your DB lives elsewhere.")
    st.markdown('<span class="aso-chip">Schema-aware</span> &nbsp; <span class="aso-chip">No ORM</span> &nbsp; <span class="aso-chip">Local-only</span>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

if not db_path:
    st.info("Enter path to your `.db` in the sidebar.")
    st.stop()

DBP = Path(db_path)
if not DBP.exists():
    st.error(f"DB not found: {DBP.resolve()}")
    st.stop()

st.caption(f"📁 Using DB: `{DBP.resolve()}`")

# ====================== DB helpers ======================
def run_sql(sql: str, params: Optional[dict] = None) -> pd.DataFrame:
    with sqlite3.connect(DBP) as con:
        return pd.read_sql(sql, con, params=params or {})

@st.cache_data(show_spinner=False)
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

# ====================== AE table autodetect ======================
candidate_ae = [
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

with st.sidebar:
    st.markdown('<div class="aso-card">', unsafe_allow_html=True)
    AE_TABLE = st.selectbox("AE table to use", available_ae, index=0, key="sel_ae_table")
    st.caption(f"Using AE table: `{AE_TABLE}`")
    st.markdown('</div>', unsafe_allow_html=True)

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
for col in ("total_treated", "pts_observed_n", "pts_observed_percent"):
    expr = _ae_num_cast(col)
    if expr: AE_NUM[col] = expr

NUMERIC_CAST_EXPR: Dict[str, str] = {}
if col_exists(AE_TABLE, "total_treated"):
    NUMERIC_CAST_EXPR["Total treated (row)"] = _ae_num_cast("total_treated")
if col_exists(AE_TABLE, "pts_observed_n"):
    NUMERIC_CAST_EXPR["Patients with AE (row)"] = _ae_num_cast("pts_observed_n")
if col_exists(AE_TABLE, "pts_observed_percent"):
    NUMERIC_CAST_EXPR["Patients with AE % (row)"] = _ae_num_cast("pts_observed_percent")
if col_exists("treatments", "chem_length_nt"):
    NUMERIC_CAST_EXPR["n of nucleotides"] = 'CAST(t."chem_length_nt" AS FLOAT)'

# ====================== Dimensions ======================
DIMENSIONS: Dict[str, Dict[str, str]] = {}

DIMENSIONS["Source type"] = {
    "expr": (
        "CASE ae.source_type "
        "WHEN 'P' THEN 'Peer review' "
        "WHEN 'N' THEN 'Nonpeer review' "
        "WHEN 'G' THEN 'Gray literature' "
        "WHEN 'F' THEN 'FAERS database' "
        "WHEN 'L' THEN 'Labeling' "
        "ELSE ae.source_type END"
    ),
    "table": AE_TABLE,
}
if col_exists(AE_TABLE, "ae_term"):
    DIMENSIONS["Adverse effect"] = {"expr": 'ae."ae_term"', "table": AE_TABLE}
if col_exists(AE_TABLE, "ae_group"):
    DIMENSIONS["Adverse effect group"] = {"expr": 'ae."ae_group"', "table": AE_TABLE}
if col_exists(AE_TABLE, "severity"):
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
    DIMENSIONS["Total treated (row)"] = {"expr": 'ae."total_treated"', "table": AE_TABLE}
if col_exists(AE_TABLE, "pts_observed_n"):
    DIMENSIONS["Patients with AE (row)"] = {"expr": 'ae."pts_observed_n"', "table": AE_TABLE}
if col_exists(AE_TABLE, "pts_observed_percent"):
    DIMENSIONS["Patients with AE % (row)"] = {"expr": 'ae."pts_observed_percent"', "table": AE_TABLE}

# Treatments
DIMENSIONS.update({
    "Name": {"expr": 't."generic_name"', "table": "treatments"},
    "Target gene": {"expr": 't."Target gene"', "table": "treatments"},
    "Mechanism of action": {"expr": 't."mechanism_summary"', "table": "treatments"},
    "Route of administration": {"expr": 't."route"', "table": "treatments"},
    "Backbone": {"expr": 't."backbone"', "table": "treatments"},
    "Sugar modification": {"expr": 't."sugar"', "table": "treatments"},
    "Structure": {"expr": 't."structure "', "table": "treatments"},  # trailing space in column
    "Gapmer notes": {"expr": 't."gapmer_notes"', "table": "treatments"},
    "Conjugate": {"expr": 't."conjugate"', "table": "treatments"},
    "n of nucleotides": {"expr": 't."chem_length_nt"', "table": "treatments"},
    "Treatment classification": {"expr": 't."treatment_group"', "table": "treatments"},
})

# Approvals / Trials
DIMENSIONS.update({
    "phase": {"expr": 'tr."phase"', "table": "trials"},
    "Approval date": {"expr": 'ap."decision_date"', "table": "approvals"},
})

def numeric_expr_for(label: str) -> Optional[str]:
    return NUMERIC_CAST_EXPR.get(label)

# ====================== Metrics ======================
METRICS: Dict[str, Dict[str, str]] = {"Count rows": {"agg": "COUNT", "expr": "*"}}
if "pts_observed_n" in AE_NUM:
    METRICS["Sum patients with AE"] = {"agg": "SUM", "expr": AE_NUM["pts_observed_n"]}
if "pts_observed_percent" in AE_NUM:
    METRICS["Avg patients with AE %"] = {"agg": "AVG", "expr": AE_NUM["pts_observed_percent"]}
if "total_treated" in AE_NUM:
    METRICS["Avg total treated"] = {"agg": "AVG", "expr": AE_NUM["total_treated"]}
if col_exists("trials", "n_treated"):
    METRICS["Total treated (trials)"] = {"agg": "SUM", "expr": 'tr."n_treated"'}
if col_exists("trials", "N_in_trial"):
    METRICS["Max N in trial"] = {"agg": "MAX", "expr": 'tr."N_in_trial"'}

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

@st.cache_data(show_spinner=False)
def _distinct_cached(db_path: str, table: str, expr: str) -> List[str]:
    q = f"""
        SELECT DISTINCT TRIM(COALESCE({expr}, '')) AS v
        FROM "{table}"
        WHERE {expr} IS NOT NULL AND TRIM({expr}) <> ''
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
    simple = expr
    for pref in ["ae.", "t.", "tr.", "ap.", "rf."]:
        if simple.startswith(pref): simple = simple[len(pref):]
    return _distinct_cached(str(DBP), table, simple)

# ====================== Build analysis UI ======================
st.markdown('<div class="aso-card">', unsafe_allow_html=True)
st.markdown("### 📊 Build analysis")
dim_choices = list(DIMENSIONS.keys())
metric_choices = list(METRICS.keys())

c1, c2 = st.columns([2, 1])
with c1:
    group_by = st.multiselect("Group by (up to 3)", dim_choices, max_selections=3, key="ms_group_by")
with c2:
    stratify_by = st.selectbox("Stratify by (optional)", ["(none)"] + dim_choices, index=0, key="sel_stratify")

metric_sel = st.multiselect("Metrics (one or more)", metric_choices, default=["Count rows"], max_selections=4, key="ms_metrics")

# -------- Advanced Filters --------
with st.expander("🎛️ Filters (optional)", expanded=False):
    filter_specs: Dict[str, dict] = {}
    filter_cols = st.multiselect("Choose filter columns", dim_choices, default=[], key="ms_filter_cols")
    for col in filter_cols[:8]:
        info = DIMENSIONS[col]
        expr = info["expr"]

        st.markdown(f"**{col}**")
        f1, f2, f3 = st.columns([1.1, 2.2, 1.2])
        mode = f1.selectbox(
            "Mode", ["Include", "Exclude", "Greater than", "Less than", "Between"],
            key=f"mode_{col}"
        )
        exclude_null = f3.checkbox("Exclude NULL/blank", value=True, key=f"nonnull_{col}")

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
                    low = f2.number_input("Min (inclusive)", value=0.0, key=f"min_{col}")
                    high = f2.number_input("Max (inclusive)", value=0.0, key=f"max_{col}")
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
st.markdown('</div>', unsafe_allow_html=True)

# Compose grouping list
gb_all: List[str] = list(group_by)
if stratify_by != "(none)" and stratify_by not in gb_all:
    if len(gb_all) < 3: gb_all.append(stratify_by)
    else: st.warning("Stratify ignored: already using 3 grouping columns.")

# ====================== Build SQL ======================
used_tables = resolve_tables(gb_all, metric_sel, filter_specs)
try:
    from_join_sql = build_from_join(used_tables)
except Exception as e:
    st.error(str(e)); st.stop()

select_parts: List[str] = []
group_positions: List[int] = []

for i, d in enumerate(gb_all, start=1):
    expr = DIMENSIONS[d]["expr"]
    select_parts.append(f"{expr} AS g{i}")
    group_positions.append(i)

for m in metric_sel:
    agg = METRICS[m]["agg"]; expr = METRICS[m]["expr"]
    if expr == "*": select_parts.append(f'{agg}(*) AS "{m}"')
    else:           select_parts.append(f"{agg}({expr}) AS \"{m}\"")

select_sql = "SELECT " + ", ".join(select_parts) if select_parts else "SELECT COUNT(*) AS \"Count rows\""

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
        with st.expander("Filter debug", expanded=False):
            st.write("Filter specs:", filter_specs)
            st.write("SQL params:", params)

# ====================== Run query ======================
try:
    df = run_sql(final_sql, params)
except Exception as e:
    st.exception(e); st.stop()

# ====================== Plot helper ======================
def render_plotly(fig):
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    fig.update_layout(
        height=560,
        margin=dict(l=30, r=30, t=60, b=170),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    if not fig.data or len(fig.data) == 0:
        st.info("Nothing to plot with the current selections (no non-null values after filtering).")
        return
    st.plotly_chart(fig, use_container_width=True, config={"responsive": True})

# ====================== Results ======================
st.markdown('<div class="aso-card">', unsafe_allow_html=True)
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

    # Use data_editor so interactive filters/edits are reflected in the returned dataframe
    edited_df = st.data_editor(df, use_container_width=True, height=440, key="main_table")
    csv = edited_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download CSV", csv, file_name="aso_analytics.csv", mime="text/csv")
st.markdown('</div>', unsafe_allow_html=True)

# ====================== Chart (SINGLE SECTION with unique keys) ======================
st.markdown('<div class="aso-card">', unsafe_allow_html=True)
st.markdown("### 🎨 Chart")

if df.empty:
    st.info("No data to chart.")
else:
    # Use the edited / filtered dataframe from the table above if available
    chart_df = edited_df if "edited_df" in locals() else df

    dims = [c for c in chart_df.columns if c.startswith("g")]
    output_metric_cols = [c for c in chart_df.columns if c in METRICS.keys()]

    # Map grouping aliases (g1, g2, ...) back to their human-readable labels
    dim_labels = {f"g{i}": gb_all[i - 1] for i in range(1, len(gb_all) + 1)}

    cc1, cc2, cc3, cc4 = st.columns([1.2, 1, 1, 1])
    chart_type = cc1.selectbox("Chart type", ["Bar", "Line", "Pie"], index=0, key="chart_type_main")

    if not output_metric_cols:
        st.info("Add at least one metric to draw a chart.")
    else:
        metric_for_chart = cc2.selectbox("Metric", output_metric_cols, index=0, key="chart_metric_main")
        sort_x = cc3.toggle("Sort by metric (desc)", value=True, key="sort_desc_main")
        log_y = cc4.toggle("Log scale (y)", value=False, key="logy_main")

        work = chart_df.copy()
        work[metric_for_chart] = pd.to_numeric(work[metric_for_chart], errors="coerce")
        work = work.dropna(subset=[metric_for_chart])
        for d in dims: work[d] = work[d].astype(str)

        if work.empty:
            st.info("No rows with numeric values for the selected metric after filtering.")
        else:
            if sort_x and chart_type != "Pie" and dims:
                work = work.sort_values(by=metric_for_chart, ascending=False)

            if chart_type == "Pie":
                if len(dims) == 0:
                    st.info("Pick at least one grouping column to draw a pie chart.")
                else:
                    donut = st.checkbox("Donut (ring) style", value=True, key="donut_main")
                    names_col = dims[0]
                    names_label = dim_labels.get(names_col, names_col)
                    fig = px.pie(
                        work, names=names_col, values=metric_for_chart,
                        hole=0.45 if donut else 0.0, color_discrete_sequence=COLOR_SEQ,
                        title=f"{metric_for_chart} share by {names_label}",
                    )

                    render_plotly(fig)

            elif len(dims) == 1:
                x = dims[0]
                x_label = dim_labels.get(x, x)
                if chart_type == "Bar":
                    fig = px.bar(work, x=x, y=metric_for_chart,
                                 color_discrete_sequence=COLOR_SEQ,
                                 title=f"{metric_for_chart} by {x_label}")
                else:
                    fig = px.line(work, x=x, y=metric_for_chart,
                                  color_discrete_sequence=COLOR_SEQ,
                                  title=f"{metric_for_chart} by {x_label}")

                if log_y: fig.update_yaxes(type="log")
                render_plotly(fig)

            elif len(dims) == 2:
                x, color = dims[0], dims[1]
                x_label = dim_labels.get(x, x)
                color_label = dim_labels.get(color, color)
                if chart_type == "Bar":
                    fig = px.bar(work, x=x, y=metric_for_chart, color=color, barmode="group",
                                 color_discrete_sequence=COLOR_SEQ,
                                 title=f"{metric_for_chart} by {x_label} and {color_label}")
                else:
                    fig = px.line(work, x=x, y=metric_for_chart, color=color,
                                  color_discrete_sequence=COLOR_SEQ,
                                  title=f"{metric_for_chart} by {x_label} and {color_label}")

                if log_y: fig.update_yaxes(type="log")
                render_plotly(fig)

            else:
                x, color, facet = dims[0], dims[1], dims[2]
                x_label = dim_labels.get(x, x)
                color_label = dim_labels.get(color, color)
                facet_label = dim_labels.get(facet, facet)
                if chart_type == "Bar":
                    fig = px.bar(work, x=x, y=metric_for_chart, color=color, facet_col=facet,
                                 facet_col_wrap=3, barmode="group",
                                 color_discrete_sequence=COLOR_SEQ,
                                 title=f"{metric_for_chart} by {x_label}, {color_label} (facet: {facet_label})")
                else:
                    fig = px.line(work, x=x, y=metric_for_chart, color=color, facet_col=facet, facet_col_wrap=3,
                                  color_discrete_sequence=COLOR_SEQ,
                                  title=f"{metric_for_chart} by {x_label}, {color_label} (facet: {facet_label})")

                if log_y: fig.update_yaxes(type="log")
                fig.update_layout(margin=dict(l=30, r=30, t=60, b=220), height=680)
                render_plotly(fig)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown('<div class="aso-spacer-xxl"></div>', unsafe_allow_html=True)

# ====================== Reference: row counts ======================
with st.expander("🗂️ Tables & row counts (reference)"):
    try:
        names = run_sql("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY 1;")
        names = names["name"].tolist()
        rc = []
        for tname in names:
            try:
                n = run_sql(f'SELECT COUNT(*) AS n FROM "{tname}"')["n"].iloc[0]
            except Exception:
                n = "ERR"
            rc.append((tname, n))
        ref_df = pd.DataFrame(rc, columns=["table", "rows"])
        st.dataframe(ref_df, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not list tables: {e}")

# ====================== Treatment info tab ======================
info_tab = st.tabs(["🧪 Treatment info"])[0]
with info_tab:
    st.markdown('<div class="aso-card">', unsafe_allow_html=True)
    st.markdown('<h4 class="aso-section-title">Select a treatment to view chemistry and evidence</h4>', unsafe_allow_html=True)

    try:
        names_df = run_sql('SELECT DISTINCT TRIM("generic_name") AS name FROM treatments WHERE "generic_name" IS NOT NULL AND TRIM("generic_name")<>"" ORDER BY 1;')
        name_options = names_df["name"].astype(str).tolist()
    except Exception as e:
        name_options = []
        st.warning(f"Could not load treatment names: {e}")

    sel_name = st.selectbox("Treatment generic name", name_options, key="sel_treatment")

    if sel_name:
        try:
            q_info = (
                'SELECT '
                '  "Target gene"           AS "Target gene", '
                '  "mechanism_summary"     AS "Mechanism of action", '
                '  "route"                 AS "Route of administration", '
                '  "conjugate"             AS "Conjugate", '
                '  "structure "            AS "Structure", '
                '  "backbone"              AS "Backbone", '
                '  "sugar"                 AS "Sugar modification", '
                '  "Nof1"                  AS "Is it n=1?", '
                '  "treatment_group"       AS "Treatment classification", '
                '  "gapmer_notes"          AS "Gapmer notes", '
                '  "chem_length_nt"        AS "Nucleotide length (nt)", '
                '  "indication_primary"    AS "Primary indication" '
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
            a1.metric("Target gene", str(row.get("Target gene", "")))
            a2.metric("Mechanism of action", str(row.get("Mechanism of action", "")))
            a3.metric("Route of administration", str(row.get("Route of administration", "")))
            a4.metric("Conjugate", str(row.get("Conjugate", "")))

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Structure", str(row.get("Structure", "")))
            b2.metric("Backbone", str(row.get("Backbone", "")))
            b3.metric("Sugar modification", str(row.get("Sugar modification", "")))
            val_raw = str(row.get("Is it n=1?", "")).strip()
            val = "Yes" if val_raw in ("1", "1.0") else ("No" if val_raw in ("0", "0.0") else val_raw)
            b4.metric("Is it n=1?", val)


            c1, c2, c3 = st.columns(3)
            c1.metric("Treatment classification", str(row.get("Treatment classification", "")))

            gapmer_notes = str(row.get("Gapmer notes", "") or "").strip()
            if gapmer_notes:
                c2.metric("Gapmer notes", gapmer_notes)

            nt_len = row.get("Nucleotide length (nt)", None)
            nt_len = "" if nt_len is None else str(nt_len).strip()
            if nt_len:
                c3.metric("Nucleotide length (nt)", nt_len)

            prim = str(row.get("Primary indication", "") or "").strip()
            st.markdown("#### Primary indication")
            if prim:
                st.markdown(f'<div class="aso-note">{prim}</div>', unsafe_allow_html=True)
            else:
                st.info("No primary indication recorded.")

        try:
            q_refs = (
                """SELECT DISTINCT ref_type AS "Type", ref_value AS "Reference"
                   FROM refs
                   WHERE treatment_id IN (
                       SELECT treatment_id FROM treatments
                       WHERE TRIM(LOWER("generic_name")) = TRIM(LOWER(:n))
                   )
                   AND ref_type IS NOT NULL AND TRIM(ref_type) <> ''
                   AND ref_value IS NOT NULL AND TRIM(ref_value) <> ''
                   ORDER BY 1, 2;"""
            )
            refs_df = run_sql(q_refs, {"n": sel_name})
        except Exception as e:
            refs_df = pd.DataFrame()
            st.warning(f"Could not load references: {e}")

        st.markdown("#### References")
        if refs_df.empty:
            st.info("No references found for this treatment.")
        else:
            st.dataframe(refs_df, use_container_width=True)

        st.markdown("#### Adverse effects by source type")
        try:
            pts_pct_expr = AE_NUM.get("pts_observed_percent", "NULL")
            total_treated_expr = AE_NUM.get("total_treated", 'ae."total_treated"')
            pts_obs_n_expr = AE_NUM.get("pts_observed_n", 'ae."pts_observed_n"')
            q_ae_src = (
                f"""
                SELECT
                    CASE ae.source_type
                        WHEN 'P' THEN 'Peer review'
                        WHEN 'N' THEN 'Nonpeer review'
                        WHEN 'G' THEN 'Gray literature'
                        WHEN 'F' THEN 'FAERS database'
                        WHEN 'L' THEN 'Labeling'
                        ELSE ae.source_type
                    END AS "Source type",
                    ae.ae_term AS "Adverse effect",
                    SUM({total_treated_expr})      AS "Total treated",
                    SUM({pts_obs_n_expr})          AS "Total with AE",
                    AVG({pts_pct_expr})            AS "Percent with AE"
                FROM {AE_TABLE} ae
                WHERE ae.treatment_id IN (
                    SELECT treatment_id FROM treatments
                    WHERE TRIM(LOWER("generic_name")) = TRIM(LOWER(:n))
                )
                GROUP BY 1, 2
                ORDER BY 1, 3 DESC;
                """
            )
            ae_by_source_df = run_sql(q_ae_src, {"n": sel_name})

        except Exception as e:
            ae_by_source_df = pd.DataFrame()
            st.warning(f"Could not load adverse effects by source type: {e}")

        if ae_by_source_df.empty:
            st.info("No adverse effects found for this treatment.")
        else:
            st.dataframe(ae_by_source_df, use_container_width=True)

        st.markdown("#### Adverse effect groups — distribution (rows)")
        try:
            q_ae_group_counts = (
                f"""
                SELECT ae_group AS "Adverse effect group", COUNT(*) AS rows
                FROM {AE_TABLE}
                WHERE treatment_id IN (
                    SELECT treatment_id FROM treatments
                    WHERE TRIM(LOWER("generic_name")) = TRIM(LOWER(:n))
                )
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
                names="Adverse effect group",
                values="rows",
                hole=0.45,
                color_discrete_sequence=COLOR_SEQ,
                title="AE groups share (by row count)"
            )
            render_plotly(pie)

    st.markdown('</div>', unsafe_allow_html=True)

# ====================== Add New Treatment tab ======================
add_tab = st.tabs(["➕ Add New Treatment"])[0]

with add_tab:
    st.markdown('<div class="aso-card">', unsafe_allow_html=True)
    st.markdown("## ➕ Add a New Treatment and Associated Data")

    st.markdown("### 🧪 Treatment Details")

    # ---- Load distinct lists for dropdowns ----
    try:
        existing_t_ids = run_sql('SELECT treatment_id FROM treatments')['treatment_id'].tolist()
        existing_names = run_sql('SELECT generic_name FROM treatments')['generic_name'].astype(str).tolist()

        target_genes = run_sql('SELECT DISTINCT "Target gene" AS g FROM treatments ORDER BY 1')['g'].astype(str).tolist()
        routes = run_sql('SELECT DISTINCT route AS r FROM treatments ORDER BY 1')['r'].astype(str).tolist()
        backbones = run_sql('SELECT DISTINCT backbone AS b FROM treatments ORDER BY 1')['b'].astype(str).tolist()
        sugars = run_sql('SELECT DISTINCT sugar AS s FROM treatments ORDER BY 1')['s'].astype(str).tolist()
        structures = run_sql('SELECT DISTINCT "Structure " AS s FROM treatments ORDER BY 1')['s'].astype(str).tolist()
        conjugates = run_sql('SELECT DISTINCT "conjugate " AS s FROM treatments ORDER BY 1')['s'].astype(str).tolist()
        t_groups = run_sql('SELECT DISTINCT treatment_group AS g FROM treatments ORDER BY 1')['g'].astype(str).tolist()
    except Exception as e:
        st.error(f"Error loading dropdowns: {e}")
        st.stop()

    # ---- Treatment fields ----
    t_id = st.text_input("🔑 Treatment ID (unique)", key="new_t_id")
    t_name = st.text_input("🧬 Generic name", key="new_t_name")

    c1, c2 = st.columns(2)
    with c1:
        t_gene = st.selectbox("Target gene", target_genes + ["(other)"], key="new_t_gene")
        t_mech = st.text_area("Mechanism of action", key="new_t_mech")
        t_route = st.selectbox("Route of administration", routes + ["(other)"], key="new_t_route")
        t_conj = st.selectbox("Conjugate", conjugates + ["(other)"], key="new_t_conj")

    with c2:
        t_backbone = st.selectbox("Backbone", backbones + ["(other)"], key="new_t_backbone")
        t_sugar = st.selectbox("Sugar modification", sugars + ["(other)"], key="new_t_sugar")
        t_nt = st.number_input("Nucleotide length (nt)", min_value=0, max_value=200, value=0, key="new_t_nt")
        t_group = st.selectbox("Treatment classification", t_groups + ["(other)"], key="new_t_group")

    t_indication = st.text_area("Primary indication", key="new_t_indication")
    t_gapmer = st.text_area("Gapmer notes", key="new_t_gapmer")

    st.markdown("---")
    st.markdown("### 📚 References (optional)")
    st.caption("Add any number of rows. Leave table empty if none.")

    ref_template = pd.DataFrame([{
        "ref_type": "",
        "ref_value": "",
        "year": "",
        "title_or_note": ""
    }])

    ref_editor = st.data_editor(
        ref_template,
        num_rows="dynamic",
        key="ref_editor",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 📜 Approvals (optional)")

    app_template = pd.DataFrame([{
        "region": "",
        "pathway": "",
        "decision_date": "",
        "decision_date_precision": "",
        "label_ref": "",
        "status_note": "",
        "decision_date_full": ""
    }])

    app_editor = st.data_editor(
        app_template,
        num_rows="dynamic",
        key="app_editor",
        use_container_width=True,
    )

    st.markdown("---")
    st.markdown("### 📊 FAERS Summary (optional)")

    faers_template = pd.DataFrame([{
        "data_window": "",
        "total_reports": "",
        "serious_reports": "",
        "top_terms_json": "",
        "notes": ""
    }])

    faers_editor = st.data_editor(
        faers_template,
        num_rows="dynamic",
        key="faers_editor",
        use_container_width=True,
    )

    # ---- Load allowed AE values ----
    st.markdown("---")
    st.markdown("### ⚠️ Adverse Events (bulk entry)")

    try:
        ae_terms = run_sql('SELECT DISTINCT ae_term FROM adverse_events_13_11 ORDER BY 1')['ae_term'].astype(str).tolist()
        ae_groups = run_sql('SELECT DISTINCT ae_group FROM adverse_events_13_11 ORDER BY 1')['ae_group'].astype(str).tolist()
        severities = run_sql('SELECT DISTINCT severity FROM adverse_events_13_11 ORDER BY 1')['severity'].astype(str).tolist()
        source_types = run_sql('SELECT DISTINCT source_type FROM adverse_events_13_11 ORDER BY 1')['source_type'].astype(str).tolist()
    except Exception as e:
        st.error(f"Error loading AE dropdowns: {e}")
        st.stop()

    ae_template = pd.DataFrame([{
        "ae_term": "",
        "ae_group": "",
        "severity": "",
        "source_type": "",
        "source_id": "",
        "total_treated": "",
        "pts_observed_n": "",
        "pts_observed_percent": "",
        "ae_comments": "",
        "n_events": "",
        "time_window": "",
        "notes": ""
    }])
    def suggest_ae_terms(prefix: str, all_terms: list) -> list:
        prefix = (prefix or "").strip().lower()
        if not prefix:
            return all_terms[:50]   # show top 50 terms when nothing typed
        return [t for t in all_terms if prefix in t.lower()][:50]  # limit to 50 results


    ae_editor = st.data_editor(
        ae_template,
        num_rows="dynamic",
        key="ae_editor",
        use_container_width=True,

        column_config = {
            "ae_term": st.column_config.TextColumn(
                "Adverse effect term",
                help="Type an AE term. Suggestions will appear below.",
            ),
            "ae_group": st.column_config.SelectboxColumn(
                "AE group",
                options=ae_groups,
            ),
            "severity": st.column_config.SelectboxColumn(
                "Severity",
                options=severities,
            ),
            "source_type": st.column_config.SelectboxColumn(
                "Source type",
                options=source_types,
            ),
        }
    )

    st.markdown("### 🔍 AE Term Suggestions")

    invalid_ae_rows = []
    suggestions = {}

    for idx, row in ae_editor.iterrows():
        term = str(row.get("ae_term", "")).strip()
        if term and term not in ae_terms:
            invalid_ae_rows.append(idx)
            prefix = term.lower()
            matches = [t for t in ae_terms if prefix in t.lower()][:10]
            suggestions[idx] = matches

    if invalid_ae_rows:
        st.warning(f"Found **{len(invalid_ae_rows)}** AE terms not in database.")
        for idx in invalid_ae_rows:
            st.write(f"Row {idx} — `{ae_editor.loc[idx, 'ae_term']}`")
            if suggestions[idx]:
                st.write("Suggestions:", ", ".join(suggestions[idx]))
            else:
                st.write("No close matches found.")
    else:
        st.success("All AE terms match known values.")

    st.markdown("---")
    submit = st.button("💾 Save All Data", type="primary")

    if submit:
        if not t_id or not t_name:
            st.error("Treatment ID and Generic name are required.")
            st.stop()

        # Check collisions
        if t_id in existing_t_ids:
            st.error(f"Treatment ID `{t_id}` already exists.")
            st.stop()

        if t_name in existing_names:
            st.error(f"Generic name `{t_name}` already exists.")
            st.stop()

        # Perform SQL inserts
        try:
            with sqlite3.connect(DBP) as con:
                cur = con.cursor()

                # Insert treatment
                cur.execute("""
                    INSERT INTO treatments (
                        treatment_id, generic_name, "Target gene", mechanism_summary,
                        route, backbone, sugar, "Structure ", gapmer_notes,
                        conjugate , chem_length_nt, indication_primary, treatment_group
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    t_id, t_name, t_gene, t_mech,
                    t_route, t_backbone, t_sugar, None,
                    t_gapmer, t_conj, t_nt, t_indication, t_group
                ))

                # Insert refs
                for _, r in ref_editor.iterrows():
                    if any(str(v).strip() for v in r):
                        cur.execute("""
                            INSERT INTO refs (ref_id, treatment_id, ref_type, ref_value, year, title_or_note)
                            VALUES (NULL, ?, ?, ?, ?, ?)
                        """, (t_id, r["ref_type"], r["ref_value"], r["year"], r["title_or_note"]))

                # Insert approvals
                for _, a in app_editor.iterrows():
                    if any(str(v).strip() for v in a):
                        cur.execute("""
                            INSERT INTO approvals (
                                approval_id, treatment_id, region, pathway, decision_date,
                                decision_date_precision, label_ref, status_note, decision_date_full
                            ) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            t_id, a["region"], a["pathway"], a["decision_date"],
                            a["decision_date_precision"], a["label_ref"], a["status_note"],
                            a["decision_date_full"]
                        ))

                # Insert FAERS
                for _, f in faers_editor.iterrows():
                    if any(str(v).strip() for v in f):
                        cur.execute("""
                            INSERT INTO faers_summary (
                                faers_id, treatment_id, data_window, total_reports,
                                serious_reports, top_terms_json, notes
                            ) VALUES (NULL, ?, ?, ?, ?, ?, ?)
                        """, (
                            t_id, f["data_window"], f["total_reports"], f["serious_reports"],
                            f["top_terms_json"], f["notes"]
                        ))

                # Insert AE rows
                for _, ae in ae_editor.iterrows():
                    if any(str(v).strip() for v in ae):
                        cur.execute(f"""
                            INSERT INTO adverse_events_13_11 (
                                ae_row_id, treatment_id, source_type, source_id, ae_term,
                                severity, total_treated, pts_observed_n, pts_observed_percent,
                                ae_comments, n_events, time_window, notes, ae_group
                            ) VALUES (
                                NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                            )
                        """, (
                            t_id,
                            ae["source_type"], ae["source_id"],
                            ae["ae_term"], ae["severity"],
                            ae["total_treated"], ae["pts_observed_n"],
                            ae["pts_observed_percent"], ae["ae_comments"],
                            ae["n_events"], ae["time_window"],
                            ae["notes"], ae["ae_group"]
                        ))

                con.commit()

            st.success("🎉 All data saved successfully!")

        except Exception as e:
            st.error(f"Error saving data: {e}")
            st.stop()

    st.markdown('</div>', unsafe_allow_html=True)
