"""
src/vector_store.py
-------------------
In-memory vector storage with CRUD operations.

Design decisions
~~~~~~~~~~~~~~~~
* Vectors are stored in a pre-allocated float32 NumPy array for fast slicing.
* IDs are stored in a parallel int64 array.
* Deletion uses a boolean mask (_active) instead of compacting the array on
  every delete.  This keeps delete O(1) while all_vectors() is O(N).
* id_to_idx maps vector_id -> row index for O(1) lookup.
* Rebuild the LSH index after insert/delete — the caller is responsible for
  triggering that (see app.py).
"""

from __future__ import annotations

import numpy as np


class VectorStore:
    """Simple in-memory vector store supporting insert, delete, and bulk load."""

    def __init__(self, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError(f"dimension must be positive, got {dimension}")

        self.dimension: int = dimension

        # Internal storage — grown dynamically via _grow()
        self._capacity: int = 0
        self._size: int = 0  # number of rows allocated (including deleted)
        self._vectors: np.ndarray = np.empty((0, dimension), dtype=np.float32)
        self._ids: np.ndarray = np.empty(0, dtype=np.int64)
        self._active: np.ndarray = np.empty(0, dtype=bool)

        # Fast lookup: vector_id -> row index
        self._id_to_idx: dict[int, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def bulk_load(self, vectors: np.ndarray, ids: np.ndarray) -> None:
        """
        Load an entire dataset at once (replaces current contents).

        Parameters
        ----------
        vectors : (N, dim) float32
        ids     : (N,) int64
        """
        if vectors.ndim != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(
                f"Expected shape (N, {self.dimension}), got {vectors.shape}"
            )
        if len(vectors) != len(ids):
            raise ValueError("vectors and ids must have the same length")

        n = len(vectors)
        self._vectors = np.array(vectors, dtype=np.float32)
        self._ids = np.array(ids, dtype=np.int64)
        self._active = np.ones(n, dtype=bool)
        self._size = n
        self._capacity = n
        self._id_to_idx = {int(vid): i for i, vid in enumerate(self._ids)}

    def insert(self, vector_id: int, vector: np.ndarray) -> None:
        """
        Insert a single vector.

        Raises
        ------
        ValueError  if vector_id already exists or vector has wrong dimension.
        """
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Expected dimension {self.dimension}, got {vector.shape}"
            )
        if vector_id in self._id_to_idx:
            raise ValueError(f"Vector ID {vector_id} already exists")

        row = self._size
        if row >= self._capacity:
            self._grow()

        self._vectors[row] = vector
        self._ids[row] = vector_id
        self._active[row] = True
        self._id_to_idx[vector_id] = row
        self._size += 1

    def delete(self, vector_id: int) -> None:
        """
        Soft-delete a vector by ID.

        The row is kept in memory but masked out from all_vectors().
        Raises KeyError if the ID does not exist or was already deleted.
        """
        if vector_id not in self._id_to_idx:
            raise KeyError(f"Vector ID {vector_id} not found")

        idx = self._id_to_idx[vector_id]
        if not self._active[idx]:
            raise KeyError(f"Vector ID {vector_id} already deleted")

        self._active[idx] = False
        del self._id_to_idx[vector_id]

    def get(self, vector_id: int) -> np.ndarray:
        """Return the vector for a given ID. Raises KeyError if not found."""
        if vector_id not in self._id_to_idx:
            raise KeyError(f"Vector ID {vector_id} not found")
        return self._vectors[self._id_to_idx[vector_id]].copy()

    def all_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (vectors, ids) for all **active** (non-deleted) entries.

        Returns
        -------
        vectors : (M, dim) float32  where M = number of active vectors
        ids     : (M,) int64
        """
        mask = self._active[: self._size]
        return self._vectors[: self._size][mask], self._ids[: self._size][mask]

    @property
    def size(self) -> int:
        """Number of active (non-deleted) vectors."""
        return int(self._active[: self._size].sum())

    @property
    def total_allocated(self) -> int:
        """Total rows allocated including soft-deleted rows."""
        return self._size

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _grow(self) -> None:
        """Double the internal capacity."""
        new_cap = max(16, self._capacity * 2)
        new_vecs = np.empty((new_cap, self.dimension), dtype=np.float32)
        new_ids = np.empty(new_cap, dtype=np.int64)
        new_active = np.zeros(new_cap, dtype=bool)

        if self._size > 0:
            new_vecs[: self._size] = self._vectors[: self._size]
            new_ids[: self._size] = self._ids[: self._size]
            new_active[: self._size] = self._active[: self._size]

        self._vectors = new_vecs
        self._ids = new_ids
        self._active = new_active
        self._capacity = new_cap


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    store = VectorStore(dimension=4)

    # Insert
    store.insert(1, np.array([0.1, 0.2, 0.3, 0.4]))
    store.insert(2, np.array([0.5, 0.6, 0.7, 0.8]))
    store.insert(3, np.array([1.0, 2.0, 3.0, 4.0]))
    print(f"Size after 3 inserts: {store.size}")  # 3

    # Get
    print(f"Vector 2: {store.get(2)}")

    # Delete
    store.delete(2)
    print(f"Size after delete: {store.size}")  # 2

    # all_vectors should not contain deleted row
    vecs, ids = store.all_vectors()
    print(f"Active IDs: {ids}")  # [1, 3]

    # Bulk load
    big = np.random.rand(1000, 4).astype(np.float32)
    big_ids = np.arange(1000, dtype=np.int64)
    store.bulk_load(big, big_ids)
    print(f"After bulk load: {store.size}")  # 1000
