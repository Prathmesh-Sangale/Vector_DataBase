"""
src/exact_search.py
-------------------
Brute-force exact nearest-neighbour search.

This is the ground truth used to:
  1. Return provably correct results when the user picks "Exact" mode.
  2. Compute the reference answer for every benchmark query so recall can
     be measured honestly.

Algorithm
~~~~~~~~~
Squared Euclidean distance:
    d²(q, x) = Σ (q_i - x_i)²

We use squared distance because the square root is monotone and does not
change the ranking.  This avoids an expensive sqrt on 50 000 rows.

All arithmetic is pure NumPy — no scipy, no sklearn.
"""

from __future__ import annotations

import numpy as np


def exact_search(
    vectors: np.ndarray,
    ids: np.ndarray,
    query: np.ndarray,
    k: int = 5,
) -> list[tuple[int, float]]:
    """
    Find the k nearest neighbours of *query* in *vectors*.

    Parameters
    ----------
    vectors : (N, dim) float32  — active database vectors
    ids     : (N,) int64        — corresponding IDs
    query   : (dim,) float32    — query vector
    k       : int               — number of neighbours to return

    Returns
    -------
    list of (vector_id, squared_distance) sorted ascending by distance
    """
    if len(vectors) == 0:
        return []

    query = np.asarray(query, dtype=np.float32)

    # Vectorised squared Euclidean distance — shape (N,)
    diff = vectors - query          # broadcast: (N, dim) - (dim,)
    sq_dists = np.einsum("ij,ij->i", diff, diff)   # faster than sum(axis=1)

    # Partial sort: only the k smallest indices matter
    k = min(k, len(sq_dists))
    nearest_indices = np.argpartition(sq_dists, k - 1)[:k]

    # Sort the k candidates by distance
    nearest_indices = nearest_indices[np.argsort(sq_dists[nearest_indices])]

    return [
        (int(ids[i]), float(sq_dists[i]))
        for i in nearest_indices
    ]


def batch_exact_search(
    vectors: np.ndarray,
    ids: np.ndarray,
    queries: np.ndarray,
    k: int = 5,
) -> list[list[int]]:
    """
    Run exact_search for every query and return only the ID lists.
    Used by the benchmark to build ground-truth answer sets.

    Parameters
    ----------
    queries : (Q, dim) float32

    Returns
    -------
    list of Q lists, each containing k int IDs in order
    """
    return [
        [vid for vid, _ in exact_search(vectors, ids, q, k)]
        for q in queries
    ]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    vectors = np.array(
        [[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]], dtype=np.float32
    )
    ids = np.array([1, 2, 3], dtype=np.int64)
    query = np.array([0.1, 0.1], dtype=np.float32)

    results = exact_search(vectors, ids, query, k=2)
    print("Results:", results)
    assert results[0][0] == 1, "Closest should be ID 1"
    assert results[1][0] == 2, "Second closest should be ID 2"
    print("exact_search: PASS")
