"""
data/generate_data.py
---------------------
Reproducible vector dataset generation.
Two modes:
  - random:    Pure Gaussian noise (fast baseline)
  - clustered: Vectors drawn from K Gaussian clusters (more realistic geometry,
               better for demonstrating LSH tradeoffs)
"""

import numpy as np


def generate_random_vectors(
    n: int = 50_000,
    dim: int = 64,
    seed: int = 42,
) -> tuple:
    """
    Generate n random float32 vectors of dimension dim.

    Returns
    -------
    vectors : np.ndarray, shape (n, dim), float32
    ids     : np.ndarray, shape (n,),    int64
    """
    rng = np.random.default_rng(seed)
    vectors = rng.standard_normal(size=(n, dim)).astype(np.float32)
    ids = np.arange(n, dtype=np.int64)
    return vectors, ids


def generate_clustered_vectors(
    n: int = 50_000,
    dim: int = 64,
    n_clusters: int = 20,
    cluster_std: float = 1.0,
    seed: int = 42,
) -> tuple:
    """
    Generate n clustered float32 vectors drawn from n_clusters Gaussian centres.

    Clustered data has lumpy geometry similar to real text embeddings and shows
    a more interesting speed-vs-accuracy tradeoff than pure random vectors.

    Returns
    -------
    vectors : np.ndarray, shape (n, dim), float32
    ids     : np.ndarray, shape (n,),    int64
    """
    rng = np.random.default_rng(seed)

    # Sample cluster centres uniformly in [-10, 10]^dim
    centres = rng.uniform(-10.0, 10.0, size=(n_clusters, dim)).astype(np.float32)

    # Assign each vector to a random cluster
    cluster_assignments = rng.integers(0, n_clusters, size=n)

    # Draw each vector from its cluster centre + Gaussian noise
    vectors = (
        centres[cluster_assignments]
        + rng.standard_normal(size=(n, dim)).astype(np.float32) * cluster_std
    )

    ids = np.arange(n, dtype=np.int64)
    return vectors, ids


def generate_query_set(
    vectors: np.ndarray,
    n_queries: int = 500,
    seed: int = 99,
) -> np.ndarray:
    """
    Sample n_queries rows from the dataset to use as a query set.

    Using actual database vectors as queries guarantees that exact ground-truth
    answers exist (the vector itself and its genuine neighbours).

    Returns
    -------
    queries : np.ndarray, shape (n_queries, dim), float32
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(vectors), size=n_queries, replace=False)
    return vectors[indices].copy()


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating clustered dataset (50 000 vectors, dim=64) …")
    vecs, ids = generate_clustered_vectors()
    print(f"  vectors shape : {vecs.shape}")
    print(f"  ids shape     : {ids.shape}")
    print(f"  dtype         : {vecs.dtype}")

    queries = generate_query_set(vecs, n_queries=500)
    print(f"  queries shape : {queries.shape}")
    print("Done.")
