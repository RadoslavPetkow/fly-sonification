"""Ordered neuron groups shared by the experiments, audio/ and midi/.

Rule: take the given matrix indices, sort by (type, bodyId) with untyped neurons
last, and split into n contiguous, near-equal groups (numpy.array_split). Groups
therefore collect neighbouring types / type families.

    .venv/bin/python -m data.groups      # prints the sensory and motor groups
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from config import AudioConfig, MidiConfig, Paths

CACHE = ROOT / Paths().cache


def load_neurons_and_indices(cache=CACHE):
    neurons = pd.read_parquet(Path(cache) / "neurons.parquet")
    indices = json.loads((Path(cache) / "indices.json").read_text())
    if not (neurons["matrix_index"].to_numpy() == np.arange(len(neurons))).all():
        raise ValueError("neurons.parquet is not in matrix order")
    return neurons, indices


def ordered_groups(neurons, idx, n_groups):
    """Split matrix indices `idx` into n_groups contiguous groups ordered by (type, bodyId)."""
    idx = np.asarray(idx)
    if n_groups < 1 or n_groups > idx.size:
        raise ValueError(f"cannot split {idx.size} neurons into {n_groups} groups")
    sub = neurons.iloc[idx]
    order = np.lexsort((sub["bodyId"].to_numpy(), sub["type"].fillna("￿").to_numpy(),
                        sub["type"].isna().to_numpy()))
    return [idx[g] for g in np.array_split(order, n_groups)]


def sensory_hop_distances(indices, cache=CACHE):
    """BFS hop distance of every neuron from the sensory input set over the built (unsigned) matrix; -1 = unreachable."""
    import scipy.sparse as sp
    from data.fetch_connectome import bfs_levels

    W = sp.load_npz(Path(cache) / "adjacency.npz").tocsr()
    return bfs_levels(W.T.tocsr(), np.asarray(indices["sensory_idx"]))


def hop_motor(indices, dist, hop):
    motor = np.asarray(indices["motor_idx"])
    return motor[dist[motor] == hop]


def describe(neurons, groups):
    rows = []
    for k, g in enumerate(groups):
        types = neurons.iloc[g]["type"].fillna("<untyped>")
        rows.append({"group": k, "n": len(g), "first_type": types.iloc[0], "last_type": types.iloc[-1],
                     "types": dict(types.value_counts().sort_index())})
    return rows


if __name__ == "__main__":
    neurons, indices = load_neurons_and_indices()
    for name, idx, n in (("sensory", indices["sensory_idx"], AudioConfig().n_bins),
                         ("motor", indices["motor_idx"], MidiConfig().n_groups)):
        groups = ordered_groups(neurons, idx, n)
        print(f"{name}: {len(idx)} neurons -> {n} groups")
        for r in describe(neurons, groups):
            t = r["types"]
            shown = ", ".join(f"{k}:{v}" for k, v in list(t.items())[:6]) + (" ..." if len(t) > 6 else "")
            print(f"  {r['group']:>2} n={r['n']:<4} {r['first_type']} .. {r['last_type']}  ({len(t)} types: {shown})")
