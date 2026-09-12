"""
src/lsh_index.py
----------------
Random Hyperplane Locality-Sensitive Hashing (LSH) index.

Algorithm overview
~~~~~~~~~~~~~~~~~~
Given dimension d and num_bits B, for each of T hash tables we:

  1. Sample B random hyperplanes: H ∈ R^(d × B)
  2. For a vector x ∈ R^d compute:
         projection = x @ H           → shape (B,)
         hash_bits  = projection > 0  → shape (B,) bool
         bucket_key = tuple(hash_bits)
  3. Store the vector's index in tables[t][bucket_key]

At query time:
  1. Hash the query in each table → bucket_key_t
  2. Collect all candidate indices from tables[t][bucket_key_t]
  3. Union across all tables, deduplicate
  4. Run exact squared-distance only on that candidate set
  5. Return top-k

Speed-accuracy knob
~~~~~~~~~~~~~~~~~~~
  num_tables ↑  →  more candidate coverage  →  recall ↑,  QPS ↓
  num_bits   ↑  →  smaller buckets          →  fewer candidates, recall ↓, QPS ↑
  num_bits   ↓  →  larger buckets           →  more candidates,  recall ↑, QPS ↓

The benchmark sweeps at least 3 configurations and plots the curve.
"""

from __future__ import annotations

import numpy as np


class LSHIndex:
    """Random Hyperplane LSH index. Pure NumPy, no external libraries."""

    def __init__(
        self,
        dimension: int,
        num_tables: int = 4,
        num_bits: int = 8,
        seed: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        dimension  : vector dimensionality
        num_tables : number of independent hash tables (T)
        num_bits   : number of hash bits per table (B)
        seed       : random seed for reproducibility
        """
        self.dimension = dimension
        self.num_tables = num_tables
        self.num_bits = num_bits
        self.seed = seed

        # Generated during build()
        self._hyperplanes: list[np.ndarray] = []   # T arrays of shape (dim, B)
        self._tables: list[dict] = []               # T dicts: bucket_key → [indices]

        # Stored references for use in search()
        self._vectors: np.ndarray | None = None     # (N, dim) float32
        self._ids: np.ndarray | None = None         # (N,) int64

        self._built: bool = False

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Index all vectors.

        Parameters
        ----------
        vectors : (N, dim) float32
        ids     : (N,) int64
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected (N, {self.dimension}), got {vectors.shape}"
            )

        self._vectors = np.asarray(vectors, dtype=np.float32)
        self._ids = np.asarray(ids, dtype=np.int64)

        rng = np.random.default_rng(self.seed)
        self._hyperplanes = []
        self._tables = []

        n = len(vectors)

        for _ in range(self.num_tables):
            # Random hyperplanes for this table: shape (dim, num_bits)
            H = rng.standard_normal(
                size=(self.dimension, self.num_bits)
            ).astype(np.float32)
            self._hyperplanes.append(H)

            # Project all vectors: (N, dim) @ (dim, B) → (N, B)
            projections = vectors @ H               # (N, B)
            bits = projections > 0                  # (N, B) bool

            # Build bucket dict
            table: dict = {}
            for idx in range(n):
                key = bits[idx].tobytes()           # compact binary key
                if key not in table:
                    table[key] = []
                table[key].append(idx)

            self._tables.append(table)

        self._built = True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query: np.ndarray,
        k: int = 5,
    ) -> list[tuple[int, float]]:
        """
        Approximate nearest-neighbour search.

        Returns
        -------
        list of (vector_id, squared_distance) sorted ascending, length ≤ k
        """
        if not self._built:
            raise RuntimeError("Call build() before search()")

        query = np.asarray(query, dtype=np.float32)
        candidates = self._collect_candidates(query)

        if not candidates:
            return []

        # Exact distance on candidates only (NOT all N vectors)
        cand_indices = np.fromiter(candidates, dtype=np.int32)
        cand_vectors = self._vectors[cand_indices]          # (C, dim)
        diff = cand_vectors - query
        sq_dists = np.einsum("ij,ij->i", diff, diff)       # (C,)

        k = min(k, len(sq_dists))
        top_k_local = np.argpartition(sq_dists, k - 1)[:k]
        top_k_local = top_k_local[np.argsort(sq_dists[top_k_local])]

        return [
            (int(self._ids[cand_indices[i]]), float(sq_dists[i]))
            for i in top_k_local
        ]

    def candidate_count(self, query: np.ndarray) -> int:
        """Return the number of candidates examined for a query."""
        if not self._built:
            return 0
        query = np.asarray(query, dtype=np.float32)
        return len(self._collect_candidates(query))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _collect_candidates(self, query: np.ndarray) -> set:
        """
        Hash the query in every table and return the union of matching
        bucket members as a set of vector indices.
        """
        candidates: set = set()
        for t_idx, (H, table) in enumerate(
            zip(self._hyperplanes, self._tables)
        ):
            proj = query @ H                    # (num_bits,)
            key = (proj > 0).tobytes()
            if key in table:
                candidates.update(table[key])
        return candidates

    def rebuild(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """Rebuild index (called after insert/delete in the UI)."""
        self.build(vectors, ids)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = "built" if self._built else "not built"
        return (
            f"LSHIndex(dim={self.dimension}, tables={self.num_tables}, "
            f"bits={self.num_bits}, status={status})"
        )


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from data.generate_data import generate_clustered_vectors, generate_query_set
    from src.exact_search import exact_search

    print("Generating 5 000-vector clustered dataset …")
    vecs, ids = generate_clustered_vectors(n=5_000, dim=64)

    print("Building LSH index (4 tables, 8 bits) …")
    lsh = LSHIndex(dimension=64, num_tables=4, num_bits=8)
    lsh.build(vecs, ids)

    query = vecs[0]
    results = lsh.search(query, k=5)
    c_count = lsh.candidate_count(query)

    print(f"Candidate count : {c_count} / {len(vecs)}")
    assert c_count < len(vecs), "Candidate count must be < N for approximate search"

    # Check recall against exact
    exact = exact_search(vecs, ids, query, k=5)
    exact_ids = {r[0] for r in exact}
    approx_ids = {r[0] for r in results}
    recall = len(exact_ids & approx_ids) / 5
    print(f"Recall@5        : {recall:.2f}")
    assert recall > 0, "Recall must be > 0"

    print("lsh_index: PASS")
