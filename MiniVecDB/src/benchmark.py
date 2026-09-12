"""
src/benchmark.py
----------------
Measures Recall@K and Queries Per Second (QPS) for each LSH configuration.

Recall@K
~~~~~~~~
    Recall@K = |exact_top_k ∩ approx_top_k| / K

  A single number is a claim.  A curve across configurations is a result.
  We sweep at least 4 configurations and plot the speed-vs-accuracy tradeoff.

QPS
~~~
    QPS = num_queries / total_lsh_search_time_seconds

Avg Candidates
~~~~~~~~~~~~~~
  The mean number of candidate vectors examined per query.
  This directly explains *why* QPS differs between configs.
"""

from __future__ import annotations

import time
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for Streamlit
import matplotlib.pyplot as plt

try:
    from src.lsh_index import LSHIndex
    from src.exact_search import batch_exact_search
except ModuleNotFoundError:
    from lsh_index import LSHIndex
    from exact_search import batch_exact_search


# ---------------------------------------------------------------------------
# Configuration presets
# ---------------------------------------------------------------------------

BENCHMARK_CONFIGS: list[dict] = [
    {"name": "Fast",     "num_tables": 2,  "num_bits": 10},
    {"name": "Balanced", "num_tables": 4,  "num_bits": 8},
    {"name": "Accurate", "num_tables": 8,  "num_bits": 6},
    {"name": "Max",      "num_tables": 16, "num_bits": 4},
]


# ---------------------------------------------------------------------------
# Core benchmark function
# ---------------------------------------------------------------------------

def run_benchmark(
    vectors: np.ndarray,
    ids: np.ndarray,
    queries: np.ndarray,
    configs: list[dict] | None = None,
    k: int = 5,
    dimension: int | None = None,
    progress_callback=None,
) -> pd.DataFrame:
    """
    Run LSH benchmark across multiple configurations.

    Parameters
    ----------
    vectors          : (N, dim) float32 — all active database vectors
    ids              : (N,) int64
    queries          : (Q, dim) float32 — query set
    configs          : list of dicts with keys name/num_tables/num_bits
                       (defaults to BENCHMARK_CONFIGS)
    k                : top-K neighbours
    dimension        : overrides auto-detect from vectors.shape[1]
    progress_callback: optional callable(step, total) for UI progress bar

    Returns
    -------
    pd.DataFrame with columns:
        name, num_tables, num_bits, recall, qps, avg_candidates, build_time_s
    """
    if configs is None:
        configs = BENCHMARK_CONFIGS

    dim = dimension or vectors.shape[1]
    n_queries = len(queries)

    # Ground truth (exact) — computed once, reused for all configs
    gt_ids: list[list[int]] = batch_exact_search(vectors, ids, queries, k)

    rows = []
    for step, cfg in enumerate(configs):
        name = cfg["name"]
        num_tables = cfg["num_tables"]
        num_bits = cfg["num_bits"]

        # Build index
        t0 = time.perf_counter()
        lsh = LSHIndex(
            dimension=dim,
            num_tables=num_tables,
            num_bits=num_bits,
        )
        lsh.build(vectors, ids)
        build_time = time.perf_counter() - t0

        # Search all queries
        candidate_counts = []
        approx_id_lists = []

        t_search_start = time.perf_counter()
        for q in queries:
            res = lsh.search(q, k=k)
            approx_id_lists.append([vid for vid, _ in res])
            candidate_counts.append(lsh.candidate_count(q))
        t_search_end = time.perf_counter()

        total_search_time = t_search_end - t_search_start
        qps = n_queries / total_search_time if total_search_time > 0 else 0.0

        # Recall@K
        recalls = []
        for gt, approx in zip(gt_ids, approx_id_lists):
            gt_set = set(gt)
            approx_set = set(approx)
            recalls.append(len(gt_set & approx_set) / k)
        mean_recall = float(np.mean(recalls))
        avg_candidates = float(np.mean(candidate_counts))

        rows.append(
            {
                "name": name,
                "num_tables": num_tables,
                "num_bits": num_bits,
                "recall": round(mean_recall, 4),
                "qps": round(qps, 1),
                "avg_candidates": round(avg_candidates, 1),
                "build_time_s": round(build_time, 3),
            }
        )

        if progress_callback:
            progress_callback(step + 1, len(configs))

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Plot function
# ---------------------------------------------------------------------------

def plot_benchmark(
    df: pd.DataFrame,
    output_path: str = "results/benchmark.png",
    k: int = 5,
    dark_mode: bool = True,
) -> plt.Figure:
    """
    Scatter plot: x = QPS, y = Recall@K.
    Each point is annotated with its config name.

    Saves the figure to output_path AND returns the Figure object
    so Streamlit can display it with st.pyplot().

    Parameters
    ----------
    df          : output of run_benchmark()
    output_path : file path to save PNG
    k           : used in y-axis label
    dark_mode   : if True use dark theme, else light theme

    Returns
    -------
    matplotlib.figure.Figure
    """
    if dark_mode:
        fig_bg    = "#0f172a"
        ax_bg     = "#1e293b"
        text_col  = "white"
        label_col = "#cbd5e1"
        tick_col  = "#94a3b8"
        grid_col  = "#334155"
        spine_col = "#334155"
        line_col  = "#64748b"
    else:
        fig_bg    = "#ffffff"
        ax_bg     = "#f8fafc"
        text_col  = "#0f172a"
        label_col = "#334155"
        tick_col  = "#475569"
        grid_col  = "#cbd5e1"
        spine_col = "#94a3b8"
        line_col  = "#94a3b8"

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(fig_bg)
    ax.set_facecolor(ax_bg)

    colours = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

    for i, (_, row) in enumerate(df.iterrows()):
        colour = colours[i % len(colours)]
        ax.scatter(
            row["qps"], row["recall"],
            s=160, zorder=5,
            color=colour,
            edgecolors=spine_col,
            linewidths=0.8,
        )
        ax.annotate(
            row["name"],
            xy=(row["qps"], row["recall"]),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=10,
            color=colour,
            fontweight="bold",
        )

    # Connect points with a faint line to show the tradeoff curve
    sorted_df = df.sort_values("qps")
    ax.plot(
        sorted_df["qps"], sorted_df["recall"],
        linestyle="--", linewidth=1, color=line_col, zorder=3,
    )

    ax.set_xlabel("Queries per Second (QPS)", color=label_col, fontsize=12)
    ax.set_ylabel(f"Recall@{k}", color=label_col, fontsize=12)
    ax.set_title(
        "LSH Speed vs Accuracy Tradeoff", color=text_col, fontsize=14, pad=12
    )
    ax.tick_params(colors=tick_col)
    for spine in ax.spines.values():
        spine.set_edgecolor(spine_col)
    ax.grid(True, color=grid_col, linestyle="--", alpha=0.5)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())

    return fig


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    # Allow running directly from the MiniVecDB root
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.generate_data import generate_clustered_vectors, generate_query_set

    print("Generating dataset …")
    vecs, ids = generate_clustered_vectors(n=5_000, dim=64)
    queries = generate_query_set(vecs, n_queries=50)

    print("Running benchmark …")
    df = run_benchmark(vecs, ids, queries, configs=BENCHMARK_CONFIGS, k=5)
    print(df.to_string(index=False))

    assert len(df) == len(BENCHMARK_CONFIGS), "Row count mismatch"
    assert "recall" in df.columns
    assert "qps" in df.columns

    fig = plot_benchmark(df, output_path="results/benchmark.png")
    print("Plot saved -> results/benchmark.png")
    print("benchmark: PASS")
