"""Fetch a real 3D position for every neuron of the anatomy set and cache it.

The neuron table has no coordinates, so the anatomical view of the web player needs a
second, much smaller fetch. The set is fixed by AnatomyConfig and chosen from the graph
alone - BFS hop distance from the sensory neurons - so it cannot be cherry-picked by
activity: every sensory neuron, every hop-1 neuron, and a fixed-seed uniform sample of
hops 2 and 3.

    .venv/bin/python -m data.fetch_positions [--force]

cache/positions.parquet
  bodyId        int64    neuPrint body id
  matrix_index  int64    row/column of that neuron in cache/adjacency.npz
  layer         str      sensory | motor | hop1 | hop2 | hop3  (motor = hop-1 motor,
                         i.e. a MIDI voice; hop1 = the other hop-1 neurons)
  hop           int32    BFS hop distance from the sensory set
  x, y, z       float64  neuPrint coordinates, voxels (8 nm), NOT normalized
  source        str      soma | synapse_centroid

Primary position is Neuron.somaLocation. Where that is null the fallback is the centroid
of every synapse of the body - a real measured position, recorded as such in `source`.
Neurons with neither are dropped and counted.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv

import numpy as np
import pandas as pd

from config import NEUPRINT_DATASET, NEUPRINT_SERVER, AnatomyConfig, Paths
from data.groups import hop_motor, load_neurons_and_indices, sensory_hop_distances

CACHE = ROOT / Paths().cache
POSITIONS = CACHE / "positions.parquet"
POSITIONS_META = CACHE / "positions_meta.json"


def anatomy_set(neurons, indices, dist, cfg):
    """The anatomy neuron set, as a frame of matrix_index / bodyId / layer / hop.

    Decided before any activity is loaded: all sensory (hop 0), all hop-1 (the motor ones
    labelled `motor`), and rng(cfg.sample_seed) samples of hop 2 and hop 3 drawn uniformly
    without replacement, with no reference to firing rate.
    """
    sensory = np.asarray(indices["sensory_idx"], dtype=np.int64)
    if not (dist[sensory] == 0).all():
        raise ValueError("sensory neurons are not all at hop 0")
    motor1 = np.asarray(hop_motor(indices, dist, 1), dtype=np.int64)
    hop1 = np.flatnonzero(dist == 1).astype(np.int64)
    if not np.isin(motor1, hop1).all():
        raise ValueError("hop-1 motor neurons are not a subset of the hop-1 set")
    hop1_rest = np.setdiff1d(hop1, motor1)

    rng = np.random.default_rng(cfg.sample_seed)
    picked = {}
    for hop, n in ((2, cfg.hop2_sample), (3, cfg.hop3_sample)):
        pool = np.flatnonzero(dist == hop).astype(np.int64)
        if pool.size < n:
            raise ValueError(f"hop {hop} has only {pool.size} neurons, cannot sample {n}")
        picked[hop] = np.sort(rng.choice(pool, size=n, replace=False))

    parts = [("sensory", sensory, 0), ("motor", motor1, 1), ("hop1", hop1_rest, 1),
             ("hop2", picked[2], 2), ("hop3", picked[3], 3)]
    frames = [pd.DataFrame({"matrix_index": idx, "layer": layer, "hop": np.int32(hop)})
              for layer, idx, hop in parts]
    out = pd.concat(frames, ignore_index=True)
    if out["matrix_index"].duplicated().any():
        raise ValueError("the anatomy layers overlap")
    out["bodyId"] = neurons["bodyId"].to_numpy()[out["matrix_index"].to_numpy()]
    return out[["matrix_index", "bodyId", "layer", "hop"]]


def client():
    load_dotenv(ROOT / ".env")
    from neuprint import Client
    return Client(NEUPRINT_SERVER, dataset=NEUPRINT_DATASET, progress=False)


def _batches(ids, size):
    for i in range(0, len(ids), size):
        yield [int(b) for b in ids[i:i + size]]


def fetch_soma(c, body_ids, cfg):
    """somaLocation per bodyId; rows whose soma is null are dropped here."""
    frames = []
    for batch in _batches(body_ids, cfg.soma_batch):
        df = c.fetch_custom(
            f"MATCH (n:Neuron) WHERE n.bodyId IN {batch} AND n.somaLocation IS NOT NULL "
            "RETURN n.bodyId AS bodyId, n.somaLocation.x AS x, n.somaLocation.y AS y, "
            "n.somaLocation.z AS z")
        frames.append(df)
        print(f"  soma: {sum(len(f) for f in frames):,} / {len(body_ids):,} asked", flush=True)
    out = pd.concat(frames, ignore_index=True)
    if out[["x", "y", "z"]].isna().any().any():
        raise ValueError("somaLocation query returned a null coordinate")
    out["source"] = "soma"
    return out


def fetch_synapse_centroid(c, body_ids, cfg):
    """Centroid of every synapse of each body; bodies with no synapse simply do not come back."""
    if len(body_ids) == 0:
        return pd.DataFrame(columns=["bodyId", "x", "y", "z", "n_syn", "source"])
    frames = []
    for batch in _batches(body_ids, cfg.synapse_batch):
        df = c.fetch_custom(
            f"MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(s:Synapse) "
            f"WHERE n.bodyId IN {batch} "
            "RETURN n.bodyId AS bodyId, count(s) AS n_syn, avg(s.location.x) AS x, "
            "avg(s.location.y) AS y, avg(s.location.z) AS z")
        frames.append(df)
        print(f"  synapse centroid: {sum(len(f) for f in frames):,} / {len(body_ids):,} asked", flush=True)
    out = pd.concat(frames, ignore_index=True)
    if out[["x", "y", "z"]].isna().any().any():
        raise ValueError("synapse centroid query returned a null coordinate")
    out["source"] = "synapse_centroid"
    return out


def fetch(cfg, force=False):
    if POSITIONS.exists() and not force:
        print(f"{POSITIONS.relative_to(ROOT)} exists; skipping fetch. Use --force to re-fetch.")
        return pd.read_parquet(POSITIONS), json.loads(POSITIONS_META.read_text())

    neurons, indices = load_neurons_and_indices()
    dist = sensory_hop_distances(indices)
    wanted = anatomy_set(neurons, indices, dist, cfg)
    print(f"anatomy set: {len(wanted):,} neurons")
    for layer in cfg.layers:
        print(f"  {layer:<8} {int((wanted['layer'] == layer).sum()):>5}")

    c = client()
    print(f"querying {NEUPRINT_DATASET} on {NEUPRINT_SERVER} (neuprint-python {c.fetch_version()})")
    soma = fetch_soma(c, wanted["bodyId"].to_numpy(), cfg)
    missing = np.setdiff1d(wanted["bodyId"].to_numpy(), soma["bodyId"].to_numpy())
    print(f"somaLocation: {len(soma):,} of {len(wanted):,}; {len(missing):,} null -> synapse centroid")
    centroid = fetch_synapse_centroid(c, missing, cfg)

    found = pd.concat([soma[["bodyId", "x", "y", "z", "source"]],
                       centroid[["bodyId", "x", "y", "z", "source"]]], ignore_index=True)
    out = wanted.merge(found, on="bodyId", how="left")
    lost = out["x"].isna()
    meta = {
        "dataset": NEUPRINT_DATASET,
        "n_wanted": int(len(wanted)),
        "n_kept": int((~lost).sum()),
        "n_dropped": int(lost.sum()),
        "dropped_bodyIds": [int(b) for b in out.loc[lost, "bodyId"]],
        "by_source": {k: int(v) for k, v in out.loc[~lost, "source"].value_counts().items()},
        "by_layer": {layer: {"wanted": int((wanted["layer"] == layer).sum()),
                             "kept": int(((out["layer"] == layer) & ~lost).sum()),
                             "soma": int(((out["layer"] == layer) & (out["source"] == "soma")).sum()),
                             "synapse_centroid": int(((out["layer"] == layer)
                                                      & (out["source"] == "synapse_centroid")).sum())}
                     for layer in cfg.layers},
        "config": {k: getattr(cfg, k) for k in dir(cfg) if not k.startswith("_")},
    }
    if lost.any():
        print(f"DROPPED {int(lost.sum())} neurons with neither soma nor any synapse: "
              f"{meta['dropped_bodyIds'][:10]}{' ...' if lost.sum() > 10 else ''}")
    out = out[~lost].reset_index(drop=True)
    out["matrix_index"] = out["matrix_index"].astype("int64")
    out["bodyId"] = out["bodyId"].astype("int64")
    out["hop"] = out["hop"].astype("int32")
    CACHE.mkdir(exist_ok=True)
    out.to_parquet(POSITIONS, index=False)
    POSITIONS_META.write_text(json.dumps(meta, indent=2, default=lambda o: list(o)))
    print(f"wrote {POSITIONS.relative_to(ROOT)} ({len(out):,} rows) "
          f"and {POSITIONS_META.relative_to(ROOT)}")
    return out, meta


def report(pos, meta, cfg):
    print(f"\n=== cache/positions.parquet: {len(pos):,} of {meta['n_wanted']:,} wanted "
          f"({meta['n_dropped']} dropped) ===")
    print(f"{'layer':<8} {'wanted':>7} {'kept':>6} {'soma':>6} {'centroid':>9} {'lost':>5}")
    for layer in cfg.layers:
        b = meta["by_layer"][layer]
        print(f"{layer:<8} {b['wanted']:>7} {b['kept']:>6} {b['soma']:>6} "
              f"{b['synapse_centroid']:>9} {b['wanted'] - b['kept']:>5}")
    for ax in "xyz":
        v = pos[ax].to_numpy()
        print(f"{ax}: {v.min():,.0f} .. {v.max():,.0f} voxels (median {np.median(v):,.0f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-fetch even if the cache exists")
    args = ap.parse_args()
    cfg = AnatomyConfig()
    pos, meta = fetch(cfg, force=args.force)
    report(pos, meta, cfg)
