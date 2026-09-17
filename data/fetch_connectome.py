"""Fetch the male-cns connectome and build a signed sparse adjacency matrix.

Two stages with two separate caches:

  fetch  (expensive, once)  neurons + raw edge list, cached verbatim: no sign,
                            no threshold, no normalization.
  build  (cheap, re-run)    signed, thresholded, optionally subset matrix built
                            only from the raw cache. Never talks to neuPrint.

    python -m data.fetch_connectome fetch [--force] [--batch-size N]
    python -m data.fetch_connectome build [--max-neurons N]
    python -m data.fetch_connectome check      # smoke test on the built cache

Matrix convention: W[i_post, j_pre] = sign(pre) * synapse count, so the
synaptic input to every neuron is W @ presynaptic_activity.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import textwrap
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from config import NEUPRINT_DATASET, NEUPRINT_SERVER, DataConfig, Paths

CACHE = ROOT / Paths().cache

# stage 1 (raw)
NEURONS_RAW = CACHE / "neurons_raw.parquet"
NEURON_ROIS_RAW = CACHE / "neuron_rois_raw.parquet"
EDGES_RAW = CACHE / "edges_raw.parquet"
EDGE_PARTS = CACHE / "edges_raw.parts"
EDGE_CHECKPOINT = CACHE / "edges_raw.checkpoint.json"
FETCH_META = CACHE / "fetch_meta.json"
# stage 2 (built)
ADJACENCY = CACHE / "adjacency.npz"
NEURONS = CACHE / "neurons.parquet"
INDICES = CACHE / "indices.json"
MODULATORY_EDGES = CACHE / "modulatory_edges.parquet"

NEURON_COLUMNS = [
    "bodyId", "type", "instance", "superclass", "class", "subclass", "supertype",
    "somaNeuromere", "entryNerve", "status", "predictedNt", "consensusNt",
    "celltypePredictedNt", "pre", "post",
    # not in the original column list, but needed for the sensory L/R split
    "somaSide", "rootSide",
]
EDGE_SCHEMA = pa.schema([("bodyId_pre", pa.int64()), ("bodyId_post", pa.int64()), ("weight", pa.int32())])

# Sanity check only: recon found these DNs under superclass descending_neuron.
# If a motor selection misses them, the selection is wrong.
EXPECTED_MOTOR_TYPES = ("MDN", "pIP1", "pIP10")


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, (tuple, set, np.ndarray)):
        return list(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


def cfg_dict(cfg):
    return {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")}


def fmt_s(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600:d}h{seconds % 3600 // 60:02d}m{seconds % 60:02d}s"


def banner(text):
    line = "!" * 78
    print(f"\n{line}\n{text}\n{line}\n", flush=True)


def header(text):
    print(f"\n=== {text} " + "=" * max(0, 72 - len(text)), flush=True)


# =============================================================================
# Stage 1: fetch
# =============================================================================

def raw_cache_complete():
    return all(p.exists() for p in (NEURONS_RAW, NEURON_ROIS_RAW, EDGES_RAW, FETCH_META))


def fetch(cfg, force=False, batch_size=None):
    if raw_cache_complete() and not force:
        print(f"raw cache exists ({EDGES_RAW.relative_to(ROOT)} etc.); skipping fetch. Use --force to re-fetch.")
        print_fetch_report(json.loads(FETCH_META.read_text()))
        return
    if force:
        for p in (NEURONS_RAW, NEURON_ROIS_RAW, EDGES_RAW, FETCH_META, EDGE_CHECKPOINT):
            p.unlink(missing_ok=True)
        shutil.rmtree(EDGE_PARTS, ignore_errors=True)

    from neuprint import Client, NeuronCriteria, fetch_neurons
    import neuprint

    CACHE.mkdir(exist_ok=True)
    batch_size = batch_size or cfg.fetch_batch_size
    c = Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, progress=False)
    ds_meta = c.fetch_datasets()[NEUPRINT_DATASET]
    started = now()

    # ---- neurons ------------------------------------------------------------
    # neurons are reused only once the edge checkpoint (which records their hash) exists
    if NEURONS_RAW.exists() and NEURON_ROIS_RAW.exists() and EDGE_CHECKPOINT.exists():
        resumed = True
        print(f"resuming: using cached {NEURONS_RAW.relative_to(ROOT)}")
        neurons = pd.read_parquet(NEURONS_RAW)
        status_counts = json.loads(EDGE_CHECKPOINT.read_text())["status_counts"]
    else:
        resumed = False
        header("fetching neuron table (all :Neuron, then status filter)")
        t0 = time.time()
        all_neurons, roi_counts = fetch_neurons(NeuronCriteria(), omit_rois=False, client=c)
        print(f"fetched {len(all_neurons):,} neurons and {len(roi_counts):,} per-ROI rows in {time.time() - t0:.1f}s")
        missing = [col for col in NEURON_COLUMNS if col not in all_neurons.columns]
        if missing:
            raise KeyError(f"fetch_neurons did not return columns {missing}")
        status_counts = {("<null>" if pd.isna(k) else k): int(v)
                         for k, v in all_neurons["status"].value_counts(dropna=False).items()}
        if cfg.fetch_status not in status_counts:
            raise ValueError(f"status {cfg.fetch_status!r} not present; have {status_counts}")
        neurons = (all_neurons.loc[all_neurons["status"] == cfg.fetch_status, NEURON_COLUMNS]
                   .sort_values("bodyId").reset_index(drop=True))
        if neurons["bodyId"].duplicated().any():
            raise ValueError("duplicate bodyIds in neuron table")
        roi_counts = roi_counts[roi_counts["bodyId"].isin(neurons["bodyId"])].reset_index(drop=True)
        print(f"status breakdown: {status_counts}")
        print(f"kept status == {cfg.fetch_status!r}: {len(neurons):,}; "
              f"dropped {len(all_neurons) - len(neurons):,} ({1 - len(neurons) / len(all_neurons):.2%})")
        roi_counts.to_parquet(NEURON_ROIS_RAW, index=False)
        neurons.to_parquet(NEURONS_RAW, index=False)

    source_ids = neurons["bodyId"].to_numpy()
    ids_sha = hashlib.sha1(source_ids.tobytes()).hexdigest()

    # ---- edges, paged over source bodyIds with a checkpoint -----------------
    if resumed:
        ckpt = json.loads(EDGE_CHECKPOINT.read_text())
        if ckpt["source_ids_sha1"] != ids_sha or ckpt["fetch_min_weight"] != cfg.fetch_min_weight:
            raise RuntimeError(f"{EDGE_CHECKPOINT} does not match the current neuron list / fetch_min_weight; "
                               "re-run with --force")
    else:
        ckpt = {"source_ids_sha1": ids_sha, "n_sources": len(source_ids), "next_index": 0,
                "fetch_min_weight": cfg.fetch_min_weight, "status_counts": status_counts,
                "started": started, "batches": []}
        shutil.rmtree(EDGE_PARTS, ignore_errors=True)
        write_json(EDGE_CHECKPOINT, ckpt)
    EDGE_PARTS.mkdir(exist_ok=True)
    # a part written after the last checkpoint update is stale: drop it
    for part in EDGE_PARTS.glob("part_*.parquet"):
        if int(part.stem.split("_")[1]) >= ckpt["next_index"]:
            part.unlink()

    header(f"fetching edges: {len(source_ids):,} source neurons, weight >= {cfg.fetch_min_weight}, "
           f"batch {batch_size}")
    if ckpt["next_index"]:
        print(f"resuming at source index {ckpt['next_index']:,}")
    targets = NeuronCriteria(status=cfg.fetch_status)
    t_run = time.time()
    done_this_run = 0
    n_edges = sum(b["edges"] for b in ckpt["batches"])
    while ckpt["next_index"] < len(source_ids):
        start = ckpt["next_index"]
        stop = min(start + batch_size, len(source_ids))
        t0 = time.time()
        conn = fetch_batch_with_retry(cfg, c, source_ids[start:stop].tolist(), targets, start, stop)
        dt = time.time() - t0
        if list(conn.columns) != EDGE_SCHEMA.names:
            raise ValueError(f"unexpected adjacency columns {list(conn.columns)}")
        table = pa.Table.from_pandas(conn, schema=EDGE_SCHEMA, preserve_index=False)
        part = EDGE_PARTS / f"part_{start:07d}_{stop:07d}.parquet"
        tmp = part.with_suffix(".tmp")
        pq.write_table(table, tmp)
        os.replace(tmp, part)
        ckpt["next_index"] = stop
        ckpt["batches"].append({"start": start, "stop": stop, "edges": len(conn), "seconds": round(dt, 2)})
        write_json(EDGE_CHECKPOINT, ckpt)

        n_edges += len(conn)
        done_this_run += stop - start
        rate = (time.time() - t_run) / done_this_run
        eta = rate * (len(source_ids) - stop)
        print(f"  [{stop:>7,}/{len(source_ids):,} {stop / len(source_ids):6.1%}] "
              f"+{len(conn):>7,} edges in {dt:5.1f}s | total {n_edges:>11,} | ETA {fmt_s(eta)}", flush=True)

    # ---- consolidate parts into one parquet ---------------------------------
    header("consolidating edge parts")
    parts = sorted(EDGE_PARTS.glob("part_*.parquet"))
    covered = [(int(p.stem.split("_")[1]), int(p.stem.split("_")[2])) for p in parts]
    expected_start = 0
    for s, e in covered:
        if s != expected_start:
            raise RuntimeError(f"edge parts are not contiguous at source index {expected_start} (next part starts {s})")
        expected_start = e
    if expected_start != len(source_ids):
        raise RuntimeError(f"edge parts cover {expected_start} of {len(source_ids)} sources")
    tmp = EDGES_RAW.with_suffix(".tmp")
    rows = 0
    with pq.ParquetWriter(tmp, EDGE_SCHEMA) as writer:
        for p in parts:
            t = pq.read_table(p, schema=EDGE_SCHEMA)
            rows += t.num_rows
            writer.write_table(t)
    if rows != n_edges:
        raise RuntimeError(f"consolidated {rows} rows but checkpoint recorded {n_edges}")
    os.replace(tmp, EDGES_RAW)

    edges = pd.read_parquet(EDGES_RAW)
    if not edges["bodyId_pre"].isin(neurons["bodyId"]).all() or not edges["bodyId_post"].isin(neurons["bodyId"]).all():
        raise ValueError("edge endpoints outside the cached neuron table")
    w = edges["weight"].to_numpy()
    meta = {
        "server": NEUPRINT_SERVER,
        "dataset": NEUPRINT_DATASET,
        "dataset_uuid": ds_meta.get("uuid"),
        "dataset_last_mod": ds_meta.get("last-mod"),
        "neuprint_server_version": c.fetch_version(),
        "neuprint_python_version": neuprint.__version__,
        "started": ckpt["started"],
        "finished": now(),
        "params": {"fetch_status": cfg.fetch_status, "fetch_min_weight": cfg.fetch_min_weight,
                   "fetch_batch_size": cfg.fetch_batch_size, "batch_size_last_run": batch_size,
                   "neuron_columns": NEURON_COLUMNS, "edge_columns": EDGE_SCHEMA.names,
                   "edge_targets": f"status == {cfg.fetch_status!r}", "omit_rois": True},
        "all_rois": sorted(c.all_rois),
        "status_counts_all_neurons": ckpt["status_counts"],
        "rows": {"neurons": len(neurons), "neuron_rois": int(pq.read_metadata(NEURON_ROIS_RAW).num_rows),
                 "edges": len(edges)},
        "edge_seconds_total": round(sum(b["seconds"] for b in ckpt["batches"]), 1),
        "weight": {"min": int(w.min()), "median": float(np.median(w)), "p90": float(np.percentile(w, 90)),
                   "max": int(w.max()), "sum": int(w.sum()), "frac_edges_weight_1": float((w == 1).mean()),
                   "frac_weight_in_weight_1_edges": float(w[w == 1].sum() / w.sum())},
    }
    write_json(FETCH_META, meta)
    shutil.rmtree(EDGE_PARTS)
    EDGE_CHECKPOINT.unlink()
    print_fetch_report(meta)


def fetch_batch_with_retry(cfg, client, body_ids, targets, start, stop):
    """One edge page. HTTP 5xx is retried with exponential backoff; anything else raises at once."""
    import requests
    from neuprint import fetch_adjacencies

    for attempt in range(cfg.fetch_max_retries + 1):
        try:
            _, conn = fetch_adjacencies(body_ids, targets, min_total_weight=cfg.fetch_min_weight,
                                        omit_rois=True, properties=[], weight_props=["weight"], client=client)
            return conn
        except requests.HTTPError as ex:
            status = ex.response.status_code if ex.response is not None else None
            if status is None or not 500 <= status < 600:
                raise
            body = ex.response.text.strip().replace("\n", " ")[:300]
            if attempt == cfg.fetch_max_retries:
                print(f"  batch [{start}:{stop}] failed with HTTP {status} after {cfg.fetch_max_retries} retries; "
                      f"body: {body!r}", flush=True)
                raise
            delay = cfg.fetch_backoff_s * 2 ** attempt
            print(f"  batch [{start}:{stop}] HTTP {status} (body: {body!r}); "
                  f"retry {attempt + 1}/{cfg.fetch_max_retries} in {delay:g}s", flush=True)
            time.sleep(delay)
    raise AssertionError("unreachable")


def print_fetch_report(meta):
    header("fetch report")
    print(f"dataset {meta['dataset']} uuid={meta['dataset_uuid']} (server {meta['neuprint_server_version']}, "
          f"neuprint-python {meta['neuprint_python_version']}), fetched {meta['started']} -> {meta['finished']}")
    sc = meta["status_counts_all_neurons"]
    total = sum(sc.values())
    kept = meta["rows"]["neurons"]
    print(f"neurons: {total:,} total, {kept:,} with status {meta['params']['fetch_status']!r}, "
          f"{total - kept:,} dropped ({(total - kept) / total:.2%}) {sc}")
    print(f"per-ROI rows: {meta['rows']['neuron_rois']:,}")
    w = meta["weight"]
    print(f"edges: {meta['rows']['edges']:,} (weight >= {meta['params']['fetch_min_weight']}), "
          f"total synapses {w['sum']:,}, edge fetch time {fmt_s(meta['edge_seconds_total'])}")
    print(f"weight: min {w['min']}  median {w['median']:g}  p90 {w['p90']:g}  max {w['max']:,}")
    print(f"weight == 1: {w['frac_edges_weight_1']:.2%} of edges, {w['frac_weight_in_weight_1_edges']:.2%} of synapses")


# =============================================================================
# Stage 2: build
# =============================================================================

def select_sensory(cfg, neurons):
    """Return (candidates, inputs, rule).

    candidates: neurons matching cfg.sensory_mode.
    inputs:     candidates on cfg.sensory_sides with pre >= cfg.sensory_min_pre; the
                only neurons that receive external audio current. Every other
                candidate stays in the network as an ordinary neuron.
    """
    subclass = (neurons["subclass"] == cfg.sensory_subclass).fillna(False)
    prefix = neurons["type"].str.startswith(tuple(cfg.sensory_type_prefixes), na=False)
    if cfg.sensory_mode == "auditory_or_jo_ab":
        mask = subclass | prefix
        rule = f"subclass == {cfg.sensory_subclass!r} OR type startswith {tuple(cfg.sensory_type_prefixes)}"
    elif cfg.sensory_mode == "subclass_auditory":
        mask, rule = subclass, f"subclass == {cfg.sensory_subclass!r}"
    elif cfg.sensory_mode == "jo_ab_prefix":
        mask, rule = prefix, f"type startswith {tuple(cfg.sensory_type_prefixes)}"
    elif cfg.sensory_mode == "jo_all":
        mask = neurons["type"].str.startswith(cfg.sensory_jo_prefix, na=False)
        rule = f"type startswith {cfg.sensory_jo_prefix!r}"
    else:
        raise ValueError(f"unknown sensory_mode {cfg.sensory_mode!r}")
    candidates = mask.fillna(False).astype(bool).to_numpy()
    if not candidates.any():
        raise ValueError(f"sensory candidate selection ({rule}) is empty")
    sides_present = set(neurons.loc[candidates, "rootSide"].dropna())
    bad_sides = [x for x in cfg.sensory_sides if x not in sides_present]
    if bad_sides:
        raise ValueError(f"sensory_sides {bad_sides} not among candidate rootSide values {sorted(sides_present)}")
    on_side = neurons["rootSide"].isin(cfg.sensory_sides).to_numpy()
    enough_pre = (neurons["pre"] >= cfg.sensory_min_pre).to_numpy()
    inputs = candidates & on_side & enough_pre
    if not inputs.any():
        raise ValueError("sensory input set is empty after side / min_pre filters")
    rule += f"; input = rootSide in {tuple(cfg.sensory_sides)} AND pre >= {cfg.sensory_min_pre}"
    return candidates, inputs, rule


def report_sensory(cfg, neurons, candidates, inputs, rule, roi_counts, all_rois):
    header(f"sensory population: mode {cfg.sensory_mode!r}")
    print(rule)
    c = neurons[candidates]
    print(f"\ncandidates: {len(c)}")
    print("  L/R x pre > 0 (rootSide):")
    print(textwrap.indent(pd.crosstab(c["rootSide"].fillna("<null>"), c["pre"] > 0, margins=True)
                          .rename(columns={False: "pre==0", True: "pre>0"}).to_string(), "    "))
    off_side = candidates & ~neurons["rootSide"].isin(cfg.sensory_sides).to_numpy()
    low_pre = candidates & ~off_side & (neurons["pre"] < cfg.sensory_min_pre).to_numpy()
    print(f"  excluded from input by side (rootSide not in {tuple(cfg.sensory_sides)}): {int(off_side.sum())}")
    print(f"  excluded from input by pre < {cfg.sensory_min_pre} (on-side): {int(low_pre.sum())} "
          f"{neurons.loc[low_pre, 'type'].fillna('<null>').value_counts().to_dict()}")
    print("  (excluded neurons stay in the network as ordinary neurons; they get no external current)")

    s = neurons[inputs]
    print(f"\nINPUT SET: {len(s)} neurons")
    print("per type:")
    print(textwrap.indent(s["type"].fillna("<null>").value_counts().sort_index().to_string(), "  "))
    print(f"subclass: {s['subclass'].fillna('<null>').value_counts().to_dict()}")
    print(f"rootSide: {s['rootSide'].fillna('<null>').value_counts().to_dict()}")
    print(f"presynapses: median {s['pre'].median():g}, range {s['pre'].min()} .. {s['pre'].max()}; "
          f"quartiles {s['pre'].quantile(0.25):g} / {s['pre'].quantile(0.75):g}")

    missing = [r for r in cfg.sensory_roi_check if r not in all_rois]
    if missing:
        raise KeyError(f"sensory_roi_check ROIs not in dataset: {missing}")
    rc = roi_counts[roi_counts["bodyId"].isin(s["bodyId"]) & roi_counts["roi"].isin(cfg.sensory_roi_check)]
    print(f"cross-check (not a filter): {rc['bodyId'].nunique()} / {len(s)} have {list(cfg.sensory_roi_check)} "
          f"in roiInfo; {rc.loc[rc['pre'] > 0, 'bodyId'].nunique()} have pre > 0 there "
          f"({', '.join(f'{r}: {rc.loc[rc.roi == r, 'bodyId'].nunique()}' for r in cfg.sensory_roi_check)})")


def select_motor(cfg, neurons):
    header(f"motor population: superclass in {tuple(cfg.motor_superclasses)}")
    mask = neurons["superclass"].isin(cfg.motor_superclasses).to_numpy()
    print(f"by superclass: {neurons.loc[mask, 'superclass'].value_counts().to_dict()}")
    untyped = mask & neurons["type"].isna().to_numpy()
    if cfg.motor_require_type:
        mask = mask & ~untyped
        print(f"motor_require_type: dropped {int(untyped.sum())} untyped")
    if not mask.any():
        raise ValueError("motor selection is empty")
    m = neurons[mask]
    types = m["type"].fillna("<null>").value_counts()
    print(f"count: {mask.sum()} neurons, {len(types)} types")
    print("type breakdown (type:count):")
    items = [f"{t}:{n}" for t, n in sorted(types.items())]
    print(textwrap.indent(textwrap.fill("  ".join(items), width=110), "  "))
    absent = [t for t in EXPECTED_MOTOR_TYPES if t not in types.index]
    if absent:
        raise ValueError(f"motor selection is wrong: expected types {absent} are missing")
    print("expected types present: " + ", ".join(f"{t}={types[t]}" for t in EXPECTED_MOTOR_TYPES))
    return mask


def resolve_nt(cfg, neurons):
    for f in (cfg.nt_field, cfg.nt_fallback_field):
        if f not in neurons.columns:
            raise KeyError(f"NT field {f!r} not in cached neuron table")
    sign_keys, mod, unclear = set(cfg.nt_sign), set(cfg.nt_modulatory), set(cfg.nt_unclear_values)
    if sign_keys & mod or sign_keys & unclear or mod & unclear:
        raise ValueError("nt_sign / nt_modulatory / nt_unclear_values overlap in config")

    primary = neurons[cfg.nt_field]
    fallback = neurons[cfg.nt_fallback_field]
    use_fb = primary.isna() & fallback.notna()
    nt = primary.where(~use_fb, fallback)
    source = np.where(primary.notna(), cfg.nt_field, np.where(use_fb, cfg.nt_fallback_field, "none"))

    role = np.empty(len(nt), dtype=object)
    sign = np.zeros(len(nt), dtype=np.int8)
    nt_obj = np.array([None if pd.isna(v) else v for v in nt], dtype=object)
    for i, v in enumerate(nt_obj):
        if v in mod:
            role[i] = "modulatory"
        elif v in cfg.nt_sign:
            role[i] = "signed"
            sign[i] = cfg.nt_sign[v]
        elif v is None or v in unclear:
            role[i] = "unknown"
            sign[i] = cfg.nt_unknown_sign
        else:
            raise ValueError(f"NT value {v!r} is not in nt_sign, nt_modulatory or nt_unclear_values")
    return nt_obj, source, role, sign


def bfs_levels(G, sources):
    """Hop distance from any source along G[pre, post] edges; -1 = unreachable."""
    dist = np.full(G.shape[0], -1, dtype=np.int32)
    frontier = np.unique(sources)
    dist[frontier] = 0
    d = 0
    while frontier.size:
        d += 1
        nb = np.unique(G[frontier].indices)
        nb = nb[dist[nb] < 0]
        dist[nb] = d
        frontier = nb
    return dist


def subset(cfg, W_counts, sensory, motor, body_ids):
    """Budgeted breadth-first expansion. W_counts[post, pre] >= 0 (unsigned, thresholded).

    Sensory and motor are included from the start. Each hop considers every
    neighbour (either direction) of the included set and admits them in
    descending order of total synaptic weight to/from the included set
    (ties by bodyId), until the budget is reached.
    """
    budget = cfg.max_neurons
    included = sensory | motor
    header(f"subsetting to max_neurons={budget:,}")
    if included.sum() > budget:
        raise ValueError(f"sensory+motor ({included.sum()}) exceeds max_neurons={budget}")
    U = (W_counts + W_counts.T).tocsr()
    hop = 0
    while included.sum() < budget:
        score = U @ included.astype(np.float64)
        score[included] = 0
        cand = np.flatnonzero(score > 0)
        if cand.size == 0:
            print(f"  hop {hop + 1}: no connected candidates left; stopping at {included.sum():,}")
            break
        order = cand[np.lexsort((body_ids[cand], -score[cand]))]
        take = order[: budget - included.sum()]
        included[take] = True
        hop += 1
        print(f"  hop {hop}: {cand.size:,} candidates, admitted {take.size:,} "
              f"(score {score[take[0]]:g} .. {score[take[-1]]:g}), included {included.sum():,}")
    return included


def diagnostics(W, sensory_idx, motor_idx):
    header("connectivity diagnostics")
    G = W.T.tocsr()  # G[pre, post]
    Gpos = G.copy()
    Gpos.data[Gpos.data < 0] = 0
    Gpos.eliminate_zeros()
    out = {}
    for name, graph in (("unsigned (all edges)", G), ("positive edges only", Gpos)):
        dist = bfs_levels(graph, sensory_idx)
        md = dist[motor_idx]
        reach = md[md >= 0]
        counts = pd.Series(md).replace(-1, np.nan).value_counts(dropna=False).sort_index()
        print(f"\n{name}: {graph.nnz:,} edges; {int((dist >= 0).sum()):,} / {len(dist):,} neurons reachable from sensory")
        print(f"  motor neurons reachable: {reach.size} / {md.size}; unreachable: {int((md < 0).sum())}")
        if reach.size:
            print(f"  hops to motor: min {reach.min()}  median {np.median(reach):g}  max {reach.max()}")
        print("  distribution (hops: motor neurons): "
              + ", ".join(f"{'unreachable' if pd.isna(k) else int(k)}: {v}" for k, v in counts.items()))
        n_comp, labels = connected_components(graph, directed=True, connection="strong")
        largest = int(np.bincount(labels).max())
        print(f"  strongly connected components: {n_comp:,}; largest has {largest:,} neurons "
              f"({largest / graph.shape[0]:.1%})")
        out[name] = {"motor_unreachable": int((md < 0).sum()), "largest_scc": largest,
                     "hop_counts": {("unreachable" if pd.isna(k) else int(k)): int(v) for k, v in counts.items()}}
    return out


def build(cfg):
    if not raw_cache_complete():
        raise FileNotFoundError("raw cache incomplete; run `python -m data.fetch_connectome fetch` first")
    t_build = time.time()
    fetch_meta = json.loads(FETCH_META.read_text())
    neurons = pd.read_parquet(NEURONS_RAW)
    roi_counts = pd.read_parquet(NEURON_ROIS_RAW)
    edges = pd.read_parquet(EDGES_RAW)
    header("build inputs")
    print(f"{len(neurons):,} neurons, {len(edges):,} raw edges, {int(edges['weight'].sum()):,} synapses "
          f"(dataset {fetch_meta['dataset']}, fetched {fetch_meta['finished']})")
    if (neurons["status"] != fetch_meta["params"]["fetch_status"]).any():
        raise ValueError("cached neuron table contains unexpected statuses")

    N = len(neurons)
    body_ids = neurons["bodyId"].to_numpy()
    idx = pd.Index(body_ids)
    pre = idx.get_indexer(edges["bodyId_pre"])
    post = idx.get_indexer(edges["bodyId_post"])
    if (pre < 0).any() or (post < 0).any():
        raise ValueError("edge endpoints missing from neuron table")
    w = edges["weight"].to_numpy().astype(np.int64)
    if np.unique(pre.astype(np.int64) * N + post).size != len(pre):
        raise ValueError("duplicate (pre, post) pairs in raw edge list")
    print(f"autapses (pre == post): {int((pre == post).sum()):,}")

    # ---- populations --------------------------------------------------------
    sensory_candidates, sensory, rule = select_sensory(cfg, neurons)
    report_sensory(cfg, neurons, sensory_candidates, sensory, rule, roi_counts, set(fetch_meta["all_rois"]))
    motor = select_motor(cfg, neurons)
    if (sensory & motor).any():
        print(f"note: {int((sensory & motor).sum())} neurons are both sensory and motor")

    # ---- signs ----------------------------------------------------------------
    header(f"signing by presynaptic NT: {cfg.nt_field} (fallback {cfg.nt_fallback_field})")
    nt, nt_source, role, sign = resolve_nt(cfg, neurons)
    print(f"NT source: {pd.Series(nt_source).value_counts().to_dict()}")
    thr = w >= cfg.matrix_weight_threshold
    total_w = w.sum()
    pre_nt = pd.Series(nt, dtype=object).fillna("<null>").to_numpy()[pre]
    rows = []
    nt_label = pd.Series(nt, dtype=object).fillna("<null>")
    for v in sorted(nt_label.unique(), key=lambda x: -int((nt_label == x).sum())):
        nmask = (nt_label == v).to_numpy()
        emask = pre_nt == v
        r = role[nmask][0]
        rows.append({
            "nt": v,
            "neurons": int(nmask.sum()),
            f"via_{cfg.nt_fallback_field}": int((nt_source[nmask] == cfg.nt_fallback_field).sum()),
            "out_edges": int(emask.sum()),
            "out_synapses": int(w[emask].sum()),
            "frac_synapses": w[emask].sum() / total_w,
            f"out_edges_w>={cfg.matrix_weight_threshold}": int((emask & thr).sum()),
            "assigned": ("modulatory -> 0" if r == "modulatory"
                         else f"unknown -> {cfg.nt_unknown_sign:+d}" if r == "unknown"
                         else f"{int(sign[nmask][0]):+d}"),
        })
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, formatters={"frac_synapses": "{:.2%}".format}))

    pre_role = role[pre]
    mod_e = pre_role == "modulatory"
    unk_e = pre_role == "unknown"
    print(f"\nmodulatory: {int((role == 'modulatory').sum()):,} neurons silenced; "
          f"{int(mod_e.sum()):,} outgoing edges ({w[mod_e].sum() / total_w:.2%} of synapses) "
          f"moved to {MODULATORY_EDGES.relative_to(ROOT)}")
    print(f"unknown sign ({cfg.nt_unknown_sign:+d}): {int((role == 'unknown').sum()):,} neurons "
          f"({(role == 'unknown').mean():.2%}); {int(unk_e.sum()):,} edges; "
          f"{w[unk_e].sum() / total_w:.2%} of ALL synaptic weight")

    mod_df = edges.loc[mod_e, ["bodyId_pre", "bodyId_post", "weight"]].copy()
    mod_df["nt"] = pre_nt[mod_e]
    mod_df.to_parquet(MODULATORY_EDGES, index=False)

    # ---- threshold -------------------------------------------------------------
    header(f"threshold: drop weight < {cfg.matrix_weight_threshold}")
    q = np.percentile(w, [50, 75, 90, 95, 99])
    print(f"raw weight distribution ({len(w):,} edges, {total_w:,} synapses): min {w.min()}  p50 {q[0]:g}  "
          f"p75 {q[1]:g}  p90 {q[2]:g}  p95 {q[3]:g}  p99 {q[4]:g}  max {w.max():,}")
    for v in range(1, max(cfg.threshold_report)):
        print(f"  weight == {v}: {(w == v).mean():6.2%} of edges, {w[w == v].sum() / total_w:6.2%} of synapses")
    print("threshold sweep over the raw edge list (removes weight < t):")
    for t in sorted(set(cfg.threshold_report) | {cfg.matrix_weight_threshold}):
        cut = w < t
        mark = "  <- matrix_weight_threshold" if t == cfg.matrix_weight_threshold else ""
        print(f"  t={t}: removes {cut.mean():6.2%} of EDGES, {w[cut].sum() / total_w:6.2%} of SYNAPTIC WEIGHT{mark}")
    keep_signed = ~mod_e
    dropped = keep_signed & ~thr
    signed_w = w[keep_signed].sum()
    print(f"of {int(keep_signed.sum()):,} non-modulatory edges ({signed_w:,} synapses): "
          f"dropped {int(dropped.sum()):,} edges ({dropped.sum() / keep_signed.sum():.2%}), "
          f"{w[dropped].sum():,} synapses ({w[dropped].sum() / signed_w:.2%} of weight)")
    keep = keep_signed & thr
    print(f"kept {int(keep.sum()):,} edges, {w[keep].sum():,} synapses")

    counts = sp.csr_matrix((w[keep].astype(np.float64), (post[keep], pre[keep])), shape=(N, N))

    # ---- optional subset --------------------------------------------------------
    if cfg.max_neurons > 0:
        included = subset(cfg, counts, sensory, motor, body_ids)
    else:
        included = np.ones(N, dtype=bool)
        header("subsetting: off (max_neurons == 0, whole connectome)")
    sel = np.flatnonzero(included)

    signed = sp.csr_matrix(((w[keep] * sign[pre[keep]]).astype(np.float32), (post[keep], pre[keep])),
                           shape=(N, N))
    W = signed[sel][:, sel].tocsr()
    W.sort_indices()
    out_neurons = neurons.iloc[sel].reset_index(drop=True)
    out_neurons.insert(0, "matrix_index", np.arange(len(sel), dtype=np.int64))
    out_neurons["nt"] = pd.Series(nt, dtype=object).iloc[sel].to_numpy()
    out_neurons["nt_source"] = nt_source[sel]
    out_neurons["nt_role"] = role[sel]
    out_neurons["sign"] = sign[sel]
    out_neurons["is_sensory_candidate"] = sensory_candidates[sel]
    out_neurons["is_sensory"] = sensory[sel]
    out_neurons["is_motor"] = motor[sel]
    sensory_idx = np.flatnonzero(sensory[sel])
    motor_idx = np.flatnonzero(motor[sel])
    print(f"\nmatrix: {W.shape[0]:,} x {W.shape[1]:,}, nnz {W.nnz:,} "
          f"(+{int((W.data > 0).sum()):,} / -{int((W.data < 0).sum()):,}), "
          f"sensory {len(sensory_idx)}, motor {len(motor_idx)}")
    unk_in = out_neurons["nt_role"].to_numpy() == "unknown"
    absw = abs(W).tocsc()
    unk_frac = absw[:, np.flatnonzero(unk_in)].sum() / absw.sum()
    print(f"unknown-sign fraction of synaptic weight in the final matrix: {unk_frac:.2%}")
    pos_w, neg_w = W.data[W.data > 0].sum(), -W.data[W.data < 0].sum()
    known = ~unk_in
    kpos = absw[:, np.flatnonzero(known & (out_neurons["sign"].to_numpy() > 0))].sum()
    kneg = absw[:, np.flatnonzero(known & (out_neurons["sign"].to_numpy() < 0))].sum()
    print(f"E/I balance by weight: excitatory {pos_w / (pos_w + neg_w):.2%}, inhibitory {neg_w / (pos_w + neg_w):.2%} "
          f"(known-sign neurons only: excitatory {kpos / (kpos + kneg):.2%})")
    print(f"density: {W.nnz / (W.shape[0] * W.shape[1]):.3e}")

    excluded = np.flatnonzero((sensory_candidates & ~sensory)[sel])
    in_deg, out_deg = np.diff(W.indptr), np.diff(W.tocsc().indptr)
    print(f"sensory candidates excluded from input but present in matrix: {excluded.size} / "
          f"{int((sensory_candidates & ~sensory).sum())}; of these {int((out_deg[excluded] > 0).sum())} have "
          f"outgoing and {int((in_deg[excluded] > 0).sum())} incoming matrix edges "
          f"({int(out_deg[excluded].sum()):,} out / {int(in_deg[excluded].sum()):,} in)")

    diag = diagnostics(W, sensory_idx, motor_idx)

    # ---- write --------------------------------------------------------------------
    header("writing")
    sp.save_npz(ADJACENCY, W, compressed=True)
    out_neurons.to_parquet(NEURONS, index=False)
    write_json(INDICES, {
        "sensory_idx": sensory_idx.tolist(),
        "motor_idx": motor_idx.tolist(),
        "matrix_convention": "W[i_post, j_pre] = sign(pre) * synapse_count; input = W @ presynaptic",
        "shape": list(W.shape),
        "nnz": int(W.nnz),
        "dataset": fetch_meta["dataset"],
        "dataset_uuid": fetch_meta["dataset_uuid"],
        "fetch_finished": fetch_meta["finished"],
        "build_timestamp": now(),
        "sensory_rule": rule,
        "sensory_candidate_idx": np.flatnonzero(sensory_candidates[sel]).tolist(),
        "unknown_sign_weight_fraction_raw": float(w[unk_e].sum() / total_w),
        "unknown_sign_weight_fraction_matrix": float(unk_frac),
        "modulatory_edges": {"file": MODULATORY_EDGES.name, "rows": int(mod_e.sum()),
                             "note": "all modulatory outgoing edges, unthresholded, not subset, by bodyId"},
        "diagnostics": diag,
        "config": cfg_dict(cfg),
    })
    for p in (ADJACENCY, NEURONS, INDICES, MODULATORY_EDGES):
        print(f"  {p.relative_to(ROOT)}  {p.stat().st_size / 1e6:,.1f} MB")
    print(f"build finished in {time.time() - t_build:.1f}s")


def load():
    """Return (W csr float32, neurons DataFrame, indices dict) from the built cache."""
    for p in (ADJACENCY, NEURONS, INDICES):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run `python -m data.fetch_connectome build`")
    W = sp.load_npz(ADJACENCY).tocsr()
    neurons = pd.read_parquet(NEURONS)
    indices = json.loads(INDICES.read_text())
    return W, neurons, indices


def sensory_only():
    """Sensory report from the neuron cache alone (no edges needed)."""
    for p in (NEURONS_RAW, NEURON_ROIS_RAW):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing; run `python -m data.fetch_connectome fetch` first")
    cfg = DataConfig()
    neurons = pd.read_parquet(NEURONS_RAW)
    roi_counts = pd.read_parquet(NEURON_ROIS_RAW)
    if FETCH_META.exists():
        all_rois, src = set(json.loads(FETCH_META.read_text())["all_rois"]), "fetch_meta.json"
    else:
        all_rois, src = set(roi_counts["roi"].unique()), "ROIs seen in neuron_rois_raw (fetch not finished)"
    print(f"{len(neurons):,} cached neurons; ROI names from {src}")
    candidates, inputs, rule = select_sensory(cfg, neurons)
    report_sensory(cfg, neurons, candidates, inputs, rule, roi_counts, all_rois)


def check():
    W, neurons, indices = load()
    header("check: built cache")
    assert W.dtype == np.float32, W.dtype
    assert W.shape == (len(neurons), len(neurons)) == tuple(indices["shape"]), (W.shape, len(neurons))
    assert (neurons["matrix_index"].to_numpy() == np.arange(len(neurons))).all()
    s, m = np.array(indices["sensory_idx"]), np.array(indices["motor_idx"])
    assert neurons["is_sensory"].to_numpy()[s].all() and neurons["is_sensory"].sum() == len(s)
    assert neurons["is_motor"].to_numpy()[m].all() and neurons["is_motor"].sum() == len(m)
    col_sign = np.sign(W.tocsc().max(axis=0).toarray().ravel() + W.tocsc().min(axis=0).toarray().ravel())
    nz = np.diff(W.tocsc().indptr) > 0
    mismatch = int((col_sign[nz] != neurons["sign"].to_numpy()[nz]).sum())
    assert mismatch == 0, f"{mismatch} columns whose entry signs disagree with neuron sign"
    print(f"W {W.shape}, nnz {W.nnz:,}, float32, |W| sum {np.abs(W.data).sum():,.0f}")
    print(f"sensory {len(s)}, motor {len(m)}, columns with outgoing edges {int(nz.sum()):,}")
    print(f"in-degree (nonzeros per row): median {np.median(np.diff(W.indptr)):g}, max {np.diff(W.indptr).max():,}")
    print(f"sensory out-degree: median {np.median(np.diff(W.tocsc().indptr)[s]):g}; "
          f"motor in-degree: median {np.median(np.diff(W.indptr)[m]):g}")
    print(f"built {indices['build_timestamp']} from {indices['dataset']} ({indices['sensory_rule']})")
    print("check OK")


def main(argv=None):
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(prog="python -m data.fetch_connectome", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="stage 1: fetch raw neurons + edges from neuPrint")
    f.add_argument("--force", action="store_true", help="discard the raw cache and re-fetch")
    f.add_argument("--batch-size", type=int, default=None, help="override DataConfig.fetch_batch_size")
    b = sub.add_parser("build", help="stage 2: build signed matrix from the raw cache")
    b.add_argument("--max-neurons", type=int, default=None, help="override DataConfig.max_neurons")
    sub.add_parser("sensory", help="sensory population report from the neuron cache only")
    sub.add_parser("check", help="smoke test: load the built cache and verify it")
    args = ap.parse_args(argv)

    cfg = DataConfig()
    if args.cmd == "fetch":
        fetch(cfg, force=args.force, batch_size=args.batch_size)
    elif args.cmd == "build":
        if args.max_neurons is not None:
            cfg.max_neurons = args.max_neurons
        build(cfg)
    elif args.cmd == "sensory":
        sensory_only()
    else:
        check()


if __name__ == "__main__":
    main()
