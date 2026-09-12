"""
app.py
------
MiniVecDB — Streamlit dashboard.

Layout
~~~~~~
  Sidebar   : all settings + Generate Dataset + quick presets
  Tab 1 🔍  : Search (Exact / LSH / Both)
  Tab 2 📊  : Benchmark (run + scatter plot)
  Tab 3 ✏️  : CRUD  (insert + delete + log)
  Tab 4 ℹ️  : About (algorithm explanation)

State
~~~~~
  st.session_state.vectors   — np.ndarray (N, dim)
  st.session_state.ids       — np.ndarray (N,)
  st.session_state.store     — VectorStore
  st.session_state.lsh       — LSHIndex (current settings)
  st.session_state.queries   — np.ndarray (Q, dim) query set
  st.session_state.bench_df  — pd.DataFrame | None
  st.session_state.crud_log  — list of strings
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st

from data.generate_data import (
    generate_random_vectors,
    generate_clustered_vectors,
    generate_query_set,
)
from src.vector_store import VectorStore
from src.exact_search import exact_search
from src.lsh_index import LSHIndex
from src.benchmark import run_benchmark, plot_benchmark, BENCHMARK_CONFIGS

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="MiniVecDB",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark premium look
# ---------------------------------------------------------------------------
# Detect Streamlit theme for chart coloring
_theme = st.get_option("theme.base") or "dark"
IS_DARK = (_theme == "dark")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── HIDE STREAMLIT TOOLBAR (Deploy button + main menu) ── */
    #MainMenu { visibility: hidden; display: none; }
    header[data-testid="stHeader"] { display: none; }
    div[data-testid="stToolbar"] { display: none; }
    div[data-testid="stDecoration"] { display: none; }
    footer { visibility: hidden; display: none; }

    /* ── DARK THEME ── */
    [data-theme="dark"] .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        color: #f1f5f9;
    }
    [data-theme="dark"] .metric-card .label { color: #94a3b8; }
    [data-theme="dark"] .metric-card .value { color: #38bdf8; }
    [data-theme="dark"] .result-row { background: #1e293b; color: #e2e8f0; }
    [data-theme="dark"] .log-entry { color: #94a3b8; border-bottom-color: #1e293b; }
    [data-theme="dark"] .subtitle-text { color: #94a3b8; }

    /* ── LIGHT THEME ── */
    [data-theme="light"] .metric-card {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        color: #0f172a;
    }
    [data-theme="light"] .metric-card .label { color: #64748b; }
    [data-theme="light"] .metric-card .value { color: #2563eb; }
    [data-theme="light"] .result-row { background: #f1f5f9; color: #0f172a; }
    [data-theme="light"] .log-entry { color: #475569; border-bottom-color: #e2e8f0; }
    [data-theme="light"] .subtitle-text { color: #475569; }

    /* ── SHARED ── */
    .metric-card {
        border-radius: 12px;
        padding: 0.6rem 1rem;
        text-align: center;
    }
    .metric-card .label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-card .value {
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 2px;
    }
    .result-row {
        border-left: 3px solid #38bdf8;
        border-radius: 6px;
        padding: 0.4rem 0.65rem;
        margin: 3px 0;
        font-size: 0.85rem;
    }
    .result-row.lsh { border-left-color: #34d399; }
    .result-row.exact { border-left-color: #f59e0b; }
    .log-entry {
        font-size: 0.78rem;
        padding: 2px 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85; }

    /* ── COMPACT SPACING (all tabs except About) ── */

    /* Reduce top padding of entire main area */
    .block-container {
        padding-top: 1.2rem !important;
        padding-bottom: 0.8rem !important;
    }

    /* Shrink h1/h2/h3 margins globally */
    h1 { margin-bottom: 0 !important; margin-top: 0 !important; }
    h2 { margin-top: 0.3rem !important; margin-bottom: 0.2rem !important; }
    h3 { margin-top: 0.3rem !important; margin-bottom: 0.2rem !important; }

    /* Reduce spacing between stMarkdown paragraphs outside About */
    div[data-testid="stMarkdownContainer"] > p {
        margin-bottom: 0.3rem !important;
        margin-top: 0 !important;
    }

    /* Tighten tab content top padding */
    div[data-testid="stTabsContent"] > div {
        padding-top: 0.5rem !important;
    }

    /* Reduce vertical gap between Streamlit element blocks */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0.35rem !important;
    }

    /* Compact subheader/heading block */
    div[data-testid="stHeading"] {
        padding-bottom: 0.15rem !important;
    }

    /* Compact divider margin */
    hr {
        margin-top: 0.4rem !important;
        margin-bottom: 0.4rem !important;
    }

    /* Sidebar — tighten internal spacing */
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] > div {
        gap: 0.25rem !important;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1rem !important;
    }

    /* ── RESTORE normal spacing inside About tab ── */
    /* About tab is the 4th tab — we use the aria-label to scope */
    div[aria-label="ℹ️ About"] div[data-testid="stMarkdownContainer"] > p {
        margin-bottom: 1rem !important;
        margin-top: 0.5rem !important;
    }
    div[aria-label="ℹ️ About"] div[data-testid="stVerticalBlock"] > div {
        gap: 0.8rem !important;
    }
    div[aria-label="ℹ️ About"] h3 {
        margin-top: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
defaults = {
    "vectors": None,
    "ids": None,
    "store": None,
    "lsh": None,
    "queries": None,
    "bench_df": None,
    "crud_log": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def _rebuild_lsh(num_tables: int, num_bits: int) -> None:
    """Rebuild LSH index with current store contents and given settings."""
    if st.session_state.store is None:
        return
    vecs, ids = st.session_state.store.all_vectors()
    if len(vecs) == 0:
        return
    dim = vecs.shape[1]
    lsh = LSHIndex(dimension=dim, num_tables=num_tables, num_bits=num_bits)
    lsh.build(vecs, ids)
    st.session_state.lsh = lsh


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    st.markdown("**Dataset**")
    n_vectors = st.select_slider(
        "Number of Vectors",
        options=[1_000, 5_000, 10_000, 25_000, 50_000],
        value=10_000,
        key="n_vectors",
    )
    dim_choice = st.selectbox("Dimension", [32, 64, 128], index=1)
    data_mode = st.radio("Data Mode", ["Clustered (recommended)", "Random"], index=0)
    n_queries_slider = st.slider("Query Set Size", 50, 500, 100, step=50)

    st.divider()
    st.markdown("**LSH Settings**")
    num_tables = st.slider("Hash Tables (T)", 1, 16, 4)
    num_bits = st.slider("Hash Bits (B)", 2, 16, 8)
    top_k = st.number_input("Top K", min_value=1, max_value=20, value=5)

    st.divider()
    st.markdown("**Quick Presets**")
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        if st.button("⚡ Fast"):
            st.session_state["num_tables_override"] = 2
            st.session_state["num_bits_override"] = 10
            st.rerun()
    with col_p2:
        if st.button("⚖️ Bal"):
            st.session_state["num_tables_override"] = 4
            st.session_state["num_bits_override"] = 8
            st.rerun()
    with col_p3:
        if st.button("🎯 Acc"):
            st.session_state["num_tables_override"] = 8
            st.session_state["num_bits_override"] = 6
            st.rerun()

    # Apply preset overrides
    if "num_tables_override" in st.session_state:
        num_tables = st.session_state.pop("num_tables_override")
    if "num_bits_override" in st.session_state:
        num_bits = st.session_state.pop("num_bits_override")

    st.divider()
    if st.button("🚀 Generate Dataset", use_container_width=True):
        with st.spinner(f"Generating {n_vectors:,} vectors (dim={dim_choice}) …"):
            if "Clustered" in data_mode:
                vecs, ids = generate_clustered_vectors(n=n_vectors, dim=dim_choice)
            else:
                vecs, ids = generate_random_vectors(n=n_vectors, dim=dim_choice)

            store = VectorStore(dimension=dim_choice)
            store.bulk_load(vecs, ids)

            queries = generate_query_set(vecs, n_queries=n_queries_slider)

            st.session_state.vectors = vecs
            st.session_state.ids = ids
            st.session_state.store = store
            st.session_state.queries = queries
            st.session_state.bench_df = None

        with st.spinner("Building LSH index …"):
            _rebuild_lsh(num_tables, num_bits)

        st.success(f"✅ {n_vectors:,} vectors loaded, LSH built!")

    # Dataset status in sidebar
    if st.session_state.store is not None:
        st.caption(
            f"📦 {st.session_state.store.size:,} active vectors · "
            f"dim={st.session_state.store.dimension}"
        )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; font-size:2.2rem; margin-bottom:0; margin-top:0;'>
        🗄️ MiniVecDB
    </h1>
    <p class='subtitle-text' style='text-align:center; font-size:0.95rem; margin-top:2px; margin-bottom:0.5rem;'>
        Vector database from scratch &middot; Exact + LSH &middot; Pure NumPy
    </p>
    """,
    unsafe_allow_html=True,
)

# Dataset summary metrics
if st.session_state.store is not None:
    store = st.session_state.store
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Active Vectors</div>"
            f"<div class='value'>{store.size:,}</div></div>",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"<div class='metric-card'><div class='label'>Dimensions</div>"
            f"<div class='value'>{store.dimension}</div></div>",
            unsafe_allow_html=True,
        )
    with m3:
        q_count = len(st.session_state.queries) if st.session_state.queries is not None else 0
        st.markdown(
            f"<div class='metric-card'><div class='label'>Query Set</div>"
            f"<div class='value'>{q_count:,}</div></div>",
            unsafe_allow_html=True,
        )
    with m4:
        lsh_info = (
            f"{num_tables}T/{num_bits}B"
            if st.session_state.lsh
            else "—"
        )
        st.markdown(
            f"<div class='metric-card'><div class='label'>LSH Config</div>"
            f"<div class='value' style='font-size:1.3rem'>{lsh_info}</div></div>",
            unsafe_allow_html=True,
        )
else:
    st.info("👈 Generate a dataset from the sidebar to get started.")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_search, tab_bench, tab_crud, tab_about = st.tabs(
    ["🔍 Search", "📊 Benchmark", "✏️ CRUD", "ℹ️ About"]
)

# ============================================================
# TAB 1 — SEARCH
# ============================================================
with tab_search:
    if st.session_state.store is None:
        st.warning("Generate a dataset first.")
    else:
        st.subheader("Vector Search")
        col_left, col_right = st.columns([1, 2])

        with col_left:
            query_id = st.number_input(
                "Query Vector ID",
                min_value=0,
                max_value=max(0, st.session_state.store.size - 1),
                value=0,
                step=1,
                key="query_id_input",
            )
            method = st.radio(
                "Search Method",
                ["Exact", "LSH", "Both (compare)"],
                index=2,
            )
            search_btn = st.button("🔍 Search", use_container_width=True)

        with col_right:
            if search_btn:
                try:
                    query_vec = st.session_state.store.get(int(query_id))
                except KeyError:
                    st.error(f"Vector ID {query_id} not found.")
                    st.stop()

                vecs_active, ids_active = st.session_state.store.all_vectors()

                results_data = []

                if method in ["Exact", "Both (compare)"]:
                    exact_res = exact_search(
                        vecs_active, ids_active, query_vec, k=int(top_k)
                    )
                    for rank, (vid, dist) in enumerate(exact_res, 1):
                        results_data.append(
                            {"Rank": rank, "Vector ID": vid,
                             "Sq. Distance": f"{dist:.5f}", "Method": "Exact"}
                        )

                if method in ["LSH", "Both (compare)"]:
                    if st.session_state.lsh is None:
                        _rebuild_lsh(num_tables, num_bits)
                    lsh_res = st.session_state.lsh.search(query_vec, k=int(top_k))
                    cand_count = st.session_state.lsh.candidate_count(query_vec)

                    for rank, (vid, dist) in enumerate(lsh_res, 1):
                        results_data.append(
                            {"Rank": rank, "Vector ID": vid,
                             "Sq. Distance": f"{dist:.5f}", "Method": "LSH"}
                        )
                    st.caption(
                        f"LSH examined **{cand_count}** / "
                        f"{len(vecs_active):,} candidates "
                        f"({100*cand_count/len(vecs_active):.1f}%)"
                    )

                if results_data:
                    df_res = pd.DataFrame(results_data)
                    # Colour-code by method
                    def highlight_method(row):
                        if row["Method"] == "Exact":
                            return ["background-color: #422006; color: #fbbf24"] * len(row)
                        return ["background-color: #052e16; color: #86efac"] * len(row)

                    st.dataframe(
                        df_res.style.apply(highlight_method, axis=1),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if method == "Both (compare)":
                        exact_ids = {
                            r["Vector ID"]
                            for r in results_data if r["Method"] == "Exact"
                        }
                        lsh_ids = {
                            r["Vector ID"]
                            for r in results_data if r["Method"] == "LSH"
                        }
                        recall = len(exact_ids & lsh_ids) / int(top_k)
                        st.metric(
                            f"Recall@{top_k} (this query)",
                            f"{recall:.0%}",
                        )

# ============================================================
# TAB 2 — BENCHMARK
# ============================================================
with tab_bench:
    if st.session_state.store is None:
        st.warning("Generate a dataset first.")
    else:
        st.subheader("Speed vs Accuracy Benchmark")
        st.markdown(
            "Runs **4 LSH configurations** against exact ground truth and "
            "plots the Recall@K vs QPS tradeoff curve."
        )

        run_bench_btn = st.button("▶ Run Full Benchmark", use_container_width=False)

        if run_bench_btn:
            vecs_b, ids_b = st.session_state.store.all_vectors()
            queries_b = st.session_state.queries

            progress_bar = st.progress(0, text="Building indexes …")

            def _progress(step, total):
                progress_bar.progress(step / total, text=f"Config {step}/{total} …")

            with st.spinner("Benchmarking …"):
                df_bench = run_benchmark(
                    vectors=vecs_b,
                    ids=ids_b,
                    queries=queries_b,
                    configs=BENCHMARK_CONFIGS,
                    k=int(top_k),
                    progress_callback=_progress,
                )
            progress_bar.empty()
            st.session_state.bench_df = df_bench

        if st.session_state.bench_df is not None:
            df_b = st.session_state.bench_df

            col_tbl, col_plot = st.columns([1, 1])

            with col_tbl:
                st.markdown("**Results Table**")
                display_df = df_b[
                    ["name", "num_tables", "num_bits", "recall", "qps", "avg_candidates"]
                ].rename(
                    columns={
                        "name": "Config",
                        "num_tables": "Tables",
                        "num_bits": "Bits",
                        "recall": f"Recall@{top_k}",
                        "qps": "QPS",
                        "avg_candidates": "Avg Candidates",
                    }
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True)

            with col_plot:
                st.markdown("**Tradeoff Curve**")
                os.makedirs("results", exist_ok=True)
                fig = plot_benchmark(
                    df_b,
                    output_path="results/benchmark.png",
                    k=int(top_k),
                    dark_mode=IS_DARK,
                )
                st.pyplot(fig, use_container_width=True)

            # Download button
            if os.path.exists("results/benchmark.png"):
                with open("results/benchmark.png", "rb") as f:
                    st.download_button(
                        "⬇️ Download Plot",
                        data=f,
                        file_name="benchmark.png",
                        mime="image/png",
                    )

# ============================================================
# TAB 3 — CRUD
# ============================================================
with tab_crud:
    if st.session_state.store is None:
        st.warning("Generate a dataset first.")
    else:
        st.subheader("Insert & Delete Vectors")
        col_ins, col_del = st.columns(2)

        # ---- INSERT ----
        with col_ins:
            st.markdown("### ➕ Insert Vector")
            new_id = st.number_input(
                "New Vector ID",
                min_value=0,
                max_value=9_999_999,
                value=99_999,
                step=1,
                key="insert_id",
            )
            insert_mode = st.radio(
                "Vector values",
                ["Auto-generate (random)", "Enter manually"],
                key="insert_mode",
            )

            if insert_mode == "Enter manually":
                dim = st.session_state.store.dimension
                raw = st.text_area(
                    f"Values ({dim} floats, comma-separated)",
                    value=", ".join(["0.0"] * dim),
                    key="manual_vector",
                )

            if st.button("➕ Insert", use_container_width=True, key="btn_insert"):
                try:
                    dim = st.session_state.store.dimension
                    if insert_mode == "Auto-generate (random)":
                        rng = np.random.default_rng()
                        vec = rng.standard_normal(dim).astype(np.float32)
                    else:
                        try:
                            vec = np.array(
                                [float(x) for x in raw.split(",")], dtype=np.float32
                            )
                        except Exception:
                            st.error("Invalid values — enter comma-separated floats.")
                            st.stop()

                    st.session_state.store.insert(int(new_id), vec)
                    _rebuild_lsh(num_tables, num_bits)
                    msg = f"✅ Inserted ID {new_id} · store size: {st.session_state.store.size:,}"
                    st.success(msg)
                    st.session_state.crud_log.insert(0, msg)
                except ValueError as e:
                    st.error(str(e))

        # ---- DELETE ----
        with col_del:
            st.markdown("### 🗑️ Delete Vector")
            del_id = st.number_input(
                "Vector ID to Delete",
                min_value=0,
                max_value=9_999_999,
                value=0,
                step=1,
                key="delete_id",
            )

            if st.button("🗑️ Delete", use_container_width=True, key="btn_delete"):
                try:
                    st.session_state.store.delete(int(del_id))
                    _rebuild_lsh(num_tables, num_bits)
                    msg = f"🗑️ Deleted ID {del_id} · store size: {st.session_state.store.size:,}"
                    st.success(msg)
                    st.session_state.crud_log.insert(0, msg)
                except KeyError as e:
                    st.error(str(e))

        # ---- Operation Log ----
        st.divider()
        st.markdown("### 📋 Operation Log")
        if st.session_state.crud_log:
            for entry in st.session_state.crud_log[:10]:
                st.markdown(
                    f"<div class='log-entry'>{entry}</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No operations yet.")

# ============================================================
# TAB 4 — ABOUT
# ============================================================
with tab_about:
    st.subheader("How MiniVecDB Works")
    st.markdown(
        """
        MiniVecDB is a vector database built **from scratch** using Python and NumPy.
        No Pinecone, FAISS, Chroma, or sklearn.

        ---

        ### Exact Search (Ground Truth)
        Compares the query against **every** vector using squared Euclidean distance:

        ```
        d²(q, x) = Σ (q_i - x_i)²
        ```

        Guarantees the true nearest neighbours. Scales as **O(N · dim)** per query.

        ---

        ### Approximate Search — Random Hyperplane LSH
        Locality-Sensitive Hashing (LSH) works by projecting vectors onto random
        hyperplanes and grouping similar vectors into the same hash bucket.

        **Build phase**
        1. Generate `T` tables, each with `B` random hyperplanes `H ∈ R^(dim × B)`
        2. For each vector `x`: hash = `sign(x @ H) > 0` → binary tuple key
        3. Store each vector's index in `tables[t][key]`

        **Search phase**
        1. Hash the query in each of the `T` tables
        2. Collect all candidate indices from matching buckets (union, deduplicated)
        3. Compute exact distance **only** on this candidate set
        4. Return the top-K

        ---

        ### Speed vs Accuracy Knob

        | Parameter | Effect |
        |-----------|--------|
        | More tables (`T↑`) | More candidate coverage → higher Recall, lower QPS |
        | More bits (`B↑`) | Smaller buckets → fewer candidates → lower Recall, higher QPS |
        | Fewer bits (`B↓`) | Larger buckets → more candidates → higher Recall, lower QPS |

        The Benchmark tab plots this tradeoff as a **curve** across 4 configurations.

        ---

        ### Recall@K
        ```
        Recall@K = |exact_top_K ∩ approx_top_K| / K
        ```
        A single number is a claim. A curve is a result.

        ---

        ### CRUD
        - **Insert**: Adds a new vector, rebuilds LSH index.
        - **Delete**: Soft-deletes (masks row from `all_vectors()`), rebuilds LSH.
        - Deleted vectors **never appear** in search results.
        """
    )
