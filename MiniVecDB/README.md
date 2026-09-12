# MiniVecDB

> **A fully-functional vector database built from scratch — using only Python and NumPy.**
> No FAISS. No Pinecone. No Chroma. No sklearn. Just pure math.

[![Python 3.11+](https://img.shields.io/badge/Python-3.11+-blue?logo=python)](https://www.python.org)
[![NumPy](https://img.shields.io/badge/NumPy-1.26+-orange?logo=numpy)](https://numpy.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What is MiniVecDB?

MiniVecDB is an educational, end-to-end vector search engine built from first principles. It demonstrates exactly how modern vector databases work under the hood — covering exact brute-force search, approximate Locality-Sensitive Hashing (LSH) indexing, dynamic CRUD operations, and empirical benchmarking — all packed into an interactive Streamlit dashboard.

**Why build it from scratch?**
Libraries like FAISS abstract away the most interesting parts. MiniVecDB forces you to confront the real math: hyperplane projections, hash tables, candidate filtering, recall measurement, and the fundamental speed-vs-accuracy tradeoff.

---

## Features

| Feature | Description |
|---|---|
| **Exact Nearest Neighbor Search** | Vectorized squared Euclidean distance over all active vectors — guaranteed 100% recall |
| **Approximate LSH Search** | Random Hyperplane projection with multi-table hashing — sub-linear query time |
| **Speed vs Accuracy Control** | Tune `num_tables` and `num_bits` to control the recall/QPS tradeoff |
| **CRUD Operations** | Insert, delete, and search vectors dynamically with automatic index updates |
| **Benchmark Suite** | Sweeps 4 index configurations, measures Recall@K and QPS, plots a Pareto curve |
| **Dual Dataset Support** | 50D synthetic clustered vectors *and* real 384D text embeddings (all-MiniLM-L6-v2) |
| **Interactive Dashboard** | Streamlit web app with 4 tabs: Search, CRUD, Benchmark, About |
| **Dark / Light Mode** | Full theme-aware UI via Streamlit |

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Prathmesh-Sangale/Vector_DataBase.git
cd Vector_DataBase
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Requirements:** `numpy>=1.26`, `streamlit>=1.35`, `matplotlib>=3.8`, `pandas>=2.1`

### 3. Run the App

```bash
streamlit run app.py
```

Then open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## Project Structure

```
MiniVecDB/
├── app.py                    # Streamlit dashboard — UI and tab routing
├── requirements.txt          # Python dependencies
├── README.md
├── .streamlit/
│   └── config.toml           # Streamlit theme config (hides toolbar)
├── src/
│   ├── vector_store.py       # In-memory CRUD storage (float32 NumPy array)
│   ├── exact_search.py       # Brute-force ground-truth search (vectorized L2)
│   ├── lsh_index.py          # Random Hyperplane LSH index (pure NumPy)
│   └── benchmark.py          # Recall@K and QPS measurement engine
├── data/
│   └── generate_data.py      # Synthetic clustered + random vector generation
└── results/
    └── benchmark.png         # Auto-generated benchmark plot
```

---

## How It Works

### 1. VectorStore (`src/vector_store.py`)

The central storage layer. Vectors are held in a pre-allocated `float32` NumPy array with a parallel boolean mask (`_active`) for O(1) soft deletes. An `id_to_idx` dictionary provides O(1) lookup by vector ID.

```python
store = VectorStore(dimension=128)
store.insert(vector_id=42, vector=np.random.rand(128).astype(np.float32))
store.delete(vector_id=42)
vectors, ids = store.all_vectors()  # returns only active (non-deleted) vectors
```

**Key design decisions:**
- **Soft deletes**: Deletion flips a boolean flag instead of compacting the array — O(1) delete, O(N) `all_vectors()`
- **Dynamic capacity growth**: Internal buffer doubles when capacity is exhausted
- **Automatic index rebuild**: After insert or delete, the LSH index is invalidated and rebuilt from the active set

---

### 2. Exact Search (`src/exact_search.py`)

Computes the squared Euclidean distance between a query vector and **every** stored vector using NumPy broadcasting:

```
D²(q, xᵢ) = Σⱼ (xᵢⱼ - qⱼ)²
```

Top-K results are extracted with `np.argpartition` (O(N)) followed by a sort of K candidates. This is the **ground-truth baseline** — it always returns the mathematically correct answer, which is used to compute Recall@K for the LSH index.

---

### 3. LSH Index (`src/lsh_index.py`)

Random Hyperplane Locality-Sensitive Hashing — the core approximate index.

**Build phase** (for each of `T` independent hash tables):
1. Sample `B` random hyperplane normals: `H ∈ ℝ^(dim × B)` from `N(0, 1)`
2. Project all `N` vectors: `projections = vectors @ H` → shape `(N, B)`
3. Convert to binary signature: `bits = projections > 0` → shape `(N, B)` bool
4. Store each vector's index in `table[bits.tobytes()]`

**Query phase:**
1. Hash the query using each of the `T` tables → collect candidate indices from matching buckets
2. Take the **union** across all `T` tables (deduplicating collisions)
3. Run **exact** squared-distance on the small candidate set only
4. Return top-K from candidates

**Tuning the tradeoff:**

| Parameter | Effect |
|---|---|
| `num_tables` ↑ | More candidates, higher recall, lower QPS |
| `num_bits` ↑ | Smaller buckets, fewer candidates, lower recall, higher QPS |
| `num_bits` ↓ | Larger buckets, more candidates, higher recall, lower QPS |

```python
lsh = LSHIndex(dimension=50, num_tables=8, num_bits=12)
lsh.build(vectors, ids)
results = lsh.search(query, k=10)  # returns [(id, sq_distance), ...]
```

---

### 4. Benchmark Engine (`src/benchmark.py`)

Automatically sweeps 4 LSH configurations, runs 100 random test queries per config, and computes:

- **Recall@K**: Fraction of true top-K neighbors (from exact search) found by LSH
- **QPS (Queries Per Second)**: Throughput measured with Python's `time.perf_counter`
- **Avg Candidates**: Average candidate pool size per query (measures filtering efficiency)

**Preset configurations:**

| Config | Tables | Bits | Expected Recall@5 | Expected QPS |
|---|---|---|---|---|
| Fast | 4 | 8 | ~65% | ~4,200 |
| Balanced | 8 | 12 | ~82% | ~2,500 |
| Accurate | 16 | 16 | ~94% | ~1,200 |
| Max | 24 | 16 | ~98% | ~850 |

> *Results vary by dataset size, dimensionality, and hardware. Run the benchmark tab to see real numbers.*

---

## Dashboard Tabs

### 🔍 Vector Search
- Load either the **50D synthetic dataset** (20-cluster Gaussian, 10,000 vectors) or the **384D real text corpus** (all-MiniLM-L6-v2 embeddings)
- Enter a query ID or free-text query sentence
- Compare **Exact L2** results vs **Approximate LSH** results side-by-side
- View per-query latency (ms), Euclidean distances, matched metadata, and candidate count

### ✏️ CRUD Operations
- **Insert**: Add a new vector to the live database — the LSH index auto-rebuilds
- **Delete**: Remove any vector by ID — soft-deleted, index auto-rebuilds
- View live database statistics (total active vectors, dimensionality, index status)

### 📊 Benchmark
- Run the full 4-config benchmark suite with progress bar
- View the dynamic **Pareto tradeoff scatter plot** (Recall@K vs QPS)
- Review the detailed comparison table for all configurations

### ℹ️ About
- Project overview, mathematical formulations, and architecture explanation
- LSH hyperplane projection diagrams and parameter descriptions

---

## Datasets

### Synthetic Clustered Vectors (50D)
Generated in `data/generate_data.py`. Creates 10,000 vectors drawn from 20 Gaussian cluster centres in 50D space. The cluster geometry produces realistic nearest-neighbor structure for demonstrating the LSH tradeoff.

### Real Text Embeddings (384D)
Encodes a curated English text corpus using the `all-MiniLM-L6-v2` sentence transformer model. These are real semantic embeddings — searching for "machine learning" will return semantically similar sentences, not just string matches.

---

## Design Choices & Limitations

| Decision | Reason |
|---|---|
| **Soft deletes** (boolean mask) | O(1) delete; trade-off: `all_vectors()` scans masked rows |
| **Full index rebuild on mutation** | Safe and correct for an MVP; a production system would use dynamic deletion or an HNSW structure |
| **In-memory only** | Data is lost on restart; persistence via `.npy` save/load is a straightforward extension |
| **No external vector libs** | Educational goal — every line of search logic is visible and readable |

---

## Future Extensions

- [ ] **HNSW Index** — Hierarchical Navigable Small World graphs for even better recall/speed tradeoffs
- [ ] **Persistent storage** — Save/load vectors and index to disk with NumPy `.npy` files
- [ ] **2D PCA visualization** — Plot vectors and highlight search results spatially
- [ ] **Dynamic LSH deletion** — Remove vectors without full index rebuild
- [ ] **Batch query support** — Process multiple queries simultaneously with matrix operations

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Prathmesh Sangale**
[GitHub →](https://github.com/Prathmesh-Sangale/Vector_DataBase)
