#!/usr/bin/env python3
"""
ragcachesim.simulator
Discrete-event simulator for evaluating cache strategies in
distributed Retrieval-Augmented Generation (RAG) clusters.

Author : Hardik Ruparel · May-2025
"""

from __future__ import annotations
import simpy, numpy as np, argparse, sys, logging          # ← logging & sys
from faker import Faker
from pybloom_live import ScalableBloomFilter
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from tabulate import tabulate
from hashlib import blake2b

# ─────────────────────────────────────────  GLOBAL LOGGER  ──────────────────
log = logging.getLogger("ragcachesim")
log.setLevel(logging.INFO)
_hdlr = logging.StreamHandler(sys.stdout)
_hdlr.setFormatter(logging.Formatter("%(asctime)s  %(message)s",
                                     datefmt="%H:%M:%S"))
log.addHandler(_hdlr)
# ─────────────────────────────────────────────────────────────────────────────

# ---------------- default parameters ----------------------------------------
DEFAULTS = dict(
    num_nodes   = 4,
    num_queries = 250_000,
    print_every = 50_000,
    dup_rate    = 0.35,
    cache_size  = 30,
    emb_dim     = 128,
    thresh      = 0.65,
    sync_int    = 0.5,
    inter_gap   = 0.02,
)

# latency (seconds) ----------------------------------------------------------
T_EMB, T_LOC, T_NET, T_RET, T_LLM = 0.002, 0.001, 0.005, 0.050, 0.100

faker = Faker()
rng   = np.random.default_rng(42)

# ---------------- helpers ---------------------------------------------------
def fake_q() -> str:
    return faker.sentence(nb_words=8)

def embed(txt: str, dim: int) -> np.ndarray:
    h   = blake2b(txt.encode(), digest_size=8).hexdigest()
    vec = np.random.default_rng(int(h, 16)).random(dim)
    return vec / np.linalg.norm(vec)

# ---------------- cache -----------------------------------------------------
class SemanticCache:
    def __init__(self, cap: int, thresh: float):
        self.cap    = cap
        self.thresh = thresh
        self.store  : list[tuple[np.ndarray, str]] = []

    def lookup(self, v: np.ndarray) -> str | None:
        if not self.store:
            return None
        sims = cosine_similarity([v], [e[0] for e in self.store])[0]
        i    = int(np.argmax(sims))
        return self.store[i][1] if sims[i] >= self.thresh else None

    def insert(self, v: np.ndarray, ans: str) -> None:
        if len(self.store) >= self.cap:
            self.store.pop(0)
        self.store.append((v, ans))

# ---------------- node ------------------------------------------------------
class Node:
    def __init__(self, env: simpy.Environment, name: str,
                 args: argparse.Namespace):
        self.e   = env
        self.n   = name
        self.a   = args
        self.cache = SemanticCache(args.cache_size, args.thresh)
        self.bloom = ScalableBloomFilter(
            initial_capacity=args.cache_size, error_rate=0.01
        )
        self.peers : list[Node] = []

        # metrics
        self.h_loc = self.h_rem = self.mis = 0
        self.lat   = 0.0
        self.rc = self.fp = self.syncs = 0
        env.process(self._sync())

    # periodic Bloom push
    def _sync(self):
        while True:
            yield self.e.timeout(self.a.sync_int)
            self.syncs += 1

    # one query
    def handle(self, qid: int, txt: str):
        st = self.e.now
        yield self.e.timeout(T_EMB)
        v   = embed(txt, self.a.emb_dim)

        ans = self.cache.lookup(v)
        if ans:
            self.h_loc += 1
            yield self.e.timeout(T_LOC)
        else:
            found = False
            for p in self.peers:
                if str(hash(v.tobytes())) in p.bloom:
                    self.rc += 1
                    yield self.e.timeout(2*T_NET)
                    a = p.cache.lookup(v)
                    if a:
                        found = True
                        self.h_rem += 1
                        self.cache.insert(v, a)
                        break
                    else:
                        self.fp += 1
            if not found:
                self.mis += 1
                yield self.e.timeout(T_RET + T_LLM)
                ans = f"ans-{qid}"
                self.cache.insert(v, ans)
            self.bloom.add(str(hash(v.tobytes())))
        self.lat += self.e.now - st

# ---------------- simulation ------------------------------------------------
def run(cfg: str, args: argparse.Namespace) -> pd.DataFrame:
    env   = simpy.Environment()
    nodes = [Node(env, f"N{i}", args) for i in range(args.num_nodes)]

    if cfg == "DSC":
        for n in nodes: n.peers = [p for p in nodes if p is not n]

    exact_cache : dict[str, str] = {}
    dup_pool    = [fake_q() for _ in range(int(args.num_queries*args.dup_rate))]
    rr_ptr      = 0

    def workload(env):
        nonlocal rr_ptr
        for i in range(args.num_queries):
            if i % args.print_every == 0 and i > 0:
                log.info(f"{cfg}: generated {i} queries")

            dup  = rng.random() < args.dup_rate
            text = rng.choice(dup_pool) if dup else fake_q()
            node = nodes[rr_ptr % args.num_nodes] if dup else rng.choice(nodes)
            if dup: rr_ptr += 1

            if cfg == "CEC":
                if text in exact_cache:
                    node.h_loc += 1; node.lat += T_LOC
                else:
                    yield env.timeout(T_EMB + T_RET + T_LLM)
                    exact_cache[text] = "ans"
                    node.mis  += 1; node.lat += T_EMB + T_RET + T_LLM
            else:
                env.process(node.handle(i, text))

            yield env.timeout(args.inter_gap)

    sim_end = args.num_queries * (args.inter_gap + 0.002)
    env.process(workload(env))
    env.run(until=sim_end)

    rows = []
    for n in nodes:
        total = n.h_loc + n.h_rem + n.mis
        if total == 0: continue
        rows.append(dict(
            Config        = cfg,
            Node          = n.n,
            Queries       = total,
            HitRate       = round(100*(n.h_loc+n.h_rem)/total, 2),
            Latency_ms    = round(1000*n.lat/total, 2),
            RemoteChecks  = n.rc,
            FalsePosPct   = round(100*n.fp/n.rc, 2) if n.rc else 0,
            BloomKB_hour  = round(n.syncs*(3600/sim_end), 1),
        ))
    return pd.DataFrame(rows)

# ---------------- CLI -------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ragcachesim",
        description="Discrete-event simulator for caching in distributed RAG."
    )
    for k, v in DEFAULTS.items():
        arg = '--' + k.replace('_', '-')
        typ = int if isinstance(v, int) else float
        if isinstance(v, float) and v < 1: typ = float
        p.add_argument(arg, default=v, type=typ,
                       help=f"default={v}")
    p.add_argument('--csv',   metavar="FILE",
                   help="Write summary CSV to FILE instead of stdout")
    p.add_argument('--quiet', action='store_true',
                   help="Suppress progress logs")
    return p

def main() -> None:
    args = build_parser().parse_args()

    if args.quiet:
        log.setLevel(logging.ERROR)

    log.info("Launching RAGCacheSim …")
    dfs  = [run(c, args) for c in ("CEC", "IC", "DSC")]
    all_df = pd.concat(dfs, ignore_index=True)
    summary = all_df.groupby("Config").agg(
        Queries      = ('Queries','sum'),
        HitRate      = ('HitRate','mean'),
        Lat_ms       = ('Latency_ms','mean'),
        RemoteChecks = ('RemoteChecks','sum'),
        FalsePos     = ('FalsePosPct','mean')
    ).reset_index()

    if args.csv:
        summary.to_csv(args.csv, index=False)
        print(f"Summary written to {args.csv}")
    else:
        print(tabulate(summary, headers='keys', tablefmt='github'))

if __name__ == "__main__":
    main()
