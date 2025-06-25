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

git clone https://github.com/Hardik27/Distributed-Semantic-Cache-DSC-Simulator.git
cd Distributed-Semantic-Cache-DSC-Simulator

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
| `--cache-size` | Per-node semantic cache size (entries)       | 3           |
| `--emb-dim`    | Embedding dimensionality                     | 128         |
| `--thresh`     | Cosine-similarity threshold for semantic hit | 0.65        |
| `--sync-int`   | Bloom summary push period (seconds)          | 0.5         |
| `--inter-gap`  | Mean inter-arrival time between queries (s)  | 0.02        |
| `--csv FILE`   | Write summary table to FILE                  | –           |

Run `ragcachesim --help` for full auto-generated help.

## 3. How It Works (Two-Minute Tour)

- **Workload Generator**:  
  Emits synthetic prompts with `dup_rate` paraphrases to mimic real traffic
- **Node Model**:  
  Each RAG node:
  1. Embeds queries
  2. Performs local cache lookup
  3. Optionally forwards to peers on cache miss
- **Cache Implementations**:
  - LRU-ordered semantic cache (vector similarity) per node
  - Bloom filters summarize keys (broadcast every `sync_int` seconds in DSC)
- **Metrics Collection**:  
  Outputs GitHub-table summary or CSV
- **Simulation Engine**:  
  Built on SimPy 4 (time advances via `yield env.timeout(...)`; no threads)

## 4. Contributing

Pull requests welcome! Open issues for:
- New cache policies (LRU-PQ, LFU, TinyLFU)
- Alternative coordination schemes
- Additional metrics/visualizations


## 5. License
Released under the GNU GPL v3. See LICENSE for details.