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

---

## What is Implemented

Every feature listed below is **fully working** in the current codebase.

### Storage — `src/vector_store.py`
- In-memory vector store backed by a pre-allocated `float32` NumPy array
- Dynamic capacity doubling (no fixed size limit)
- Soft delete via boolean mask — O(1) delete, active vectors always filtered correctly
- O(1) lookup by Vector ID via `id_to_idx` dictionary
- `bulk_load()` for loading an entire dataset at once
- `insert()`, `delete()`, `get()`, `all_vectors()`, `size` property

### Exact Search — `src/exact_search.py`
- Vectorized squared Euclidean distance: `D²(q, x) = Σ(q_i − x_i)²`
- Uses `np.einsum` for fast distance reduction across all active vectors
- `np.argpartition` for O(N) partial sort then exact sort of top-K
- `batch_exact_search()` for running a full query set — used by the benchmark
- **100% recall guaranteed** — exhaustively checks every active vector

### Approximate Search — `src/lsh_index.py`
- Random Hyperplane Locality-Sensitive Hashing (pure NumPy, no external libs)
- Configurable `num_tables` (T) and `num_bits` (B) per table
- Build phase: projects all vectors against random Gaussian hyperplanes `H ∈ ℝ^(dim × B)`, stores binary hash keys
- Query phase: hashes query in each of T tables, takes union of candidate indices, runs exact distance on candidates only
- `candidate_count()` method — reports how many vectors were actually examined
- `rebuild()` — fast rebuild after insert/delete
- Tunable speed-vs-accuracy knob: `num_tables ↑ → recall ↑, QPS ↓` / `num_bits ↑ → QPS ↑, recall ↓`

### Benchmark Engine — `src/benchmark.py`
- Sweeps **4 preset configurations**: Fast (2T/10B), Balanced (4T/8B), Accurate (8T/6B), Max (16T/4B)
- Computes **Recall@K**: `|exact_top_K ∩ approx_top_K| / K` per query, averaged over the full query set
- Computes **QPS** (Queries Per Second) using `time.perf_counter`
- Tracks **Avg Candidates** examined per query — explains *why* QPS varies
- Tracks **build_time_s** for each index configuration
- Produces a **Pareto scatter plot** (Recall@K vs QPS) with config labels and a tradeoff curve
- Plot supports **dark and light mode** rendering
- Saves plot to `results/benchmark.png`; also returns `matplotlib.Figure` for Streamlit display
- **Download button** for the benchmark plot PNG

### Datasets — `data/generate_data.py`
- `generate_clustered_vectors(n, dim, n_clusters)` — Gaussian clusters drawn from random centroids; realistic ANN geometry
- `generate_random_vectors(n, dim)` — pure Gaussian noise; worst case for LSH
- `generate_query_set(vectors, n_queries)` — samples rows from the dataset as held-out queries; guarantees exact ground-truth answers exist
- Configurable: 1,000 – 50,000 vectors, dimensions 32 / 64 / 128

### Streamlit Dashboard — `app.py`

**Sidebar:**
- Number of vectors slider (1K – 50K)
- Dimension selector (32, 64, 128)
- Data mode: Clustered or Random
- Query set size slider
- Hash Tables (T) and Hash Bits (B) sliders
- 3 Quick Preset buttons: ⚡ Fast, ⚖️ Balanced, 🎯 Accurate
- "Generate Dataset" button with spinner feedback

**🔍 Search Tab:**
- Query by Vector ID
- Choose method: `Exact`, `LSH`, or `Both (compare)`
- Color-coded results table (amber = Exact, green = LSH)
- Shows squared distances and rank for each result
- In `Both` mode: shows live **Recall@K for this single query**
- Shows candidate count examined by LSH (`764 / 10,000` style)

**📊 Benchmark Tab:**
- "Run Full Benchmark" button with live progress bar
- Side-by-side: results table + Pareto tradeoff scatter plot
- Table shows: Config, Tables, Bits, Recall@K, QPS, Avg Candidates
- Download PNG button

**✏️ CRUD Tab:**
- Insert: enter a Vector ID, auto-generate a random vector or enter values manually (comma-separated floats)
- Delete: enter a Vector ID to soft-delete
- Both operations auto-rebuild the LSH index after completion
- Operation log showing the last 10 inserts/deletes with timestamps

**ℹ️ About Tab:**
- Explains exact search, LSH build/search algorithm, speed-vs-accuracy knob, Recall@K formula, and CRUD mechanics

---

## What is NOT Implemented (Simplified for MVP)

| Feature | Current Behavior | What a Production System Would Do |
|---|---|---|
| **Data persistence** | All vectors are in-memory; lost on restart | Persist to `.npy` or a database |
| **Text / semantic search** | Query by Vector ID only; no text-to-embedding pipeline | Use `sentence-transformers` to embed queries |
| **Metadata per vector** | Not stored | Attach document text, tags, timestamps per vector |
| **Dynamic index delete** | Full LSH index rebuilt after every delete | Tombstoning or HNSW dynamic deletion |
| **Concurrent access** | Single-user Streamlit session | Thread-safe connection pooling |
| **Real embeddings dataset** | Synthetic Gaussian vectors only | Load real text/image embeddings |

---

## Quick Start

```bash
git clone https://github.com/Prathmesh-Sangale/Vector_DataBase.git
cd Vector_DataBase
pip install -r requirements.txt
streamlit run app.py
```

Open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## Project Structure

```
MiniVecDB/
├── app.py                    # Streamlit dashboard (all 4 tabs + sidebar)
├── requirements.txt          # numpy, streamlit, matplotlib, pandas
├── README.md
├── .streamlit/
│   └── config.toml           # Hides deploy toolbar
├── src/
│   ├── vector_store.py       # In-memory CRUD: float32 array + soft-delete mask
│   ├── exact_search.py       # Brute-force L2 search + batch variant
│   ├── lsh_index.py          # Random Hyperplane LSH: build, search, rebuild
│   └── benchmark.py          # Recall@K + QPS sweep + Pareto plot
├── data/
│   └── generate_data.py      # Clustered, random, and query-set generators
└── results/
    └── benchmark.png         # Auto-generated by benchmark tab
```

---

## How to Use It

### 1. Generate a Dataset
Open the sidebar, pick your vector count and dimension, choose **Clustered** or **Random**, and click **Generate Dataset**. The VectorStore loads instantly and the LSH index is built.

### 2. Search
Go to the **🔍 Search** tab, enter any Vector ID (0 to N−1), pick a method, and click Search. Use `Both (compare)` to see exact vs LSH side-by-side and the per-query Recall@K.

### 3. Benchmark
Click **▶ Run Full Benchmark** in the **📊 Benchmark** tab. The engine sweeps all 4 configs over your full query set and renders the Pareto curve.

### 4. CRUD
In the **✏️ CRUD** tab, insert a new vector (auto-generated or manually entered) or delete any existing one by ID. The operation log tracks every change.

---

## Architecture — How It Works

### Exact Search
```
D²(q, xᵢ) = Σⱼ (xᵢⱼ − qⱼ)²
top-K = argsort(D²)[:k]
```
Scans all N vectors. Guaranteed correct. Used as ground truth for Recall@K.

### LSH Index — Build
```
For each table t in T:
    H ~ N(0,1)^(dim × B)         # random hyperplanes
    bits = (vectors @ H) > 0      # (N, B) bool
    key  = bits.tobytes()          # compact hash key
    tables[t][key].append(idx)
```

### LSH Index — Query
```
candidates = ∅
For each table t:
    key = ((query @ H_t) > 0).tobytes()
    candidates ∪= tables[t][key]

Compute exact D² for candidates only → return top-K
```

### Recall@K
```
Recall@K = |exact_top_K ∩ approx_top_K| / K
```
Averaged over the full query set. A curve across configs, not just a single number.

---

## Tech Stack

| Component | Library |
|---|---|
| Linear algebra / storage | NumPy |
| Dashboard | Streamlit |
| Benchmark plots | Matplotlib |
| Results tables | Pandas |
| Language | Python 3.11+ |

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Author

**Prathmesh Sangale**  
[GitHub →](https://github.com/Prathmesh-Sangale/Vector_DataBase)
