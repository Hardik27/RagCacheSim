# RAGCacheSim

**RAGCacheSim** is a lightweight, fully-parameterisable **discrete-event simulator** for studying cache strategies in distributed Retrieval-Augmented Generation (RAG) clusters.  
It lets you compare three built-in designs

| Abbrev. | Strategy                                    |
|---------|---------------------------------------------|
| `CEC`   | *Centralised Exact-match Cache*             |
| `IC`    | *Independent Semantic Caches* (no sharing) |
| `DSC`   | *Distributed Semantic Cache Coordination*   |

and reports metrics such as hit-rate, average latency, remote-check count, and Bloom-filter false-positive rate.

---

## 1. Quick start

git clone https://github.com/Hardik27/RagCacheSim.git
cd ragcachesim

### install in editable mode (adds `ragcachesim` to PATH)
pip install -e .

### run with defaults
ragcachesim

### run an 8-node, 100k-query experiment with larger caches
ragcachesim --nodes 8 --queries 100000 --cache-size 5

### export summary as CSV
ragcachesim --csv results.csv

### Single-file Binary
Build a standalone executable with:

pyinstaller -F -n ragcachesim ragcachesim/simulator.py

Grab the executable from `dist/ragcachesim`.

## 2. Command-Line Options

| Flag           | Meaning                                      | Default     |
|----------------|----------------------------------------------|-------------|
| `--nodes`      | Number of RAG nodes in cluster               | 4           |
| `--queries`    | Total queries to simulate                    | 250000      |
| `--print-every`| Progress log interval                        | 50000       |
| `--dup-rate`   | Fraction of semantically duplicated queries  | 0.35        |
| `--cache-size` | Per-node semantic cache size (entries)       | 30           |
| `--emb-dim`    | Embedding dimensionality                     | 128         |
| `--thresh`     | Cosine-similarity threshold for semantic hit | 0.65        |
| `--sync-int`   | Bloom summary push period (seconds)          | 0.5         |
| `--inter-gap`  | Mean inter-arrival time between queries (s)  | 0.02        |
| `--csv FILE`   | Write summary table to FILE                  | –           |

Run `ragcachesim --help` for full auto-generated help.

## 3. How It Works (Two-Minute Tour)

- **Workload Generator**  
  Synthesizes prompts with configurable:
  - Semantic similarity distribution
  - Duplicate rate (`--dup-rate`)
  - Query arrival patterns (`--inter-gap`)

- **Node Processing**  
  Each RAG node:
  1. Embeds queries using FastEmbed
  2. Checks local semantic cache (LRU policy)
  3. On cache miss:
     - Forwards to peers (DSC strategy)
     - Falls back to Retriever+LLM

- **Cache Implementations**:
  - **CEC (Centralized Exact-match)**: Global key-value store for identical queries
  - **IC (Independent Semantic)**: Node-local vector similarity matching (`--thresh`)
  - **DSC (Distributed Coordination)**: Combines IC with Bloom filter sync (`--sync-int`)

- **Metrics Collection**:
  - Cache hit rates (local/remote)
  - Latency decomposition
  - False positive rates
  - Network overhead


## 4. Contributing

Pull requests welcome! Open issues for:
- New cache policies (LRU-PQ, LFU, TinyLFU)
- Alternative coordination schemes
- Additional metrics/visualizations


## 5. License
Released under the GNU GPL v3. See LICENSE for details.