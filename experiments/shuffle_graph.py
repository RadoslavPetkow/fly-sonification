"""Degree-, weight- and sign-preserving shuffles of the signed adjacency matrix.

    .venv/bin/python -m experiments.shuffle_graph CONDITION SEED [--force]
        CONDITION: full | sensory_only | downstream_only

W[i_post, j_pre] (cache/adjacency.npz). A shuffle re-assigns the TARGETS of the selected presynaptic
neurons' outgoing edges; every edge keeps its presynaptic neuron and its weight, so each neuron's
out-degree, outgoing weight multiset and sign (set by its neurotransmitter) are unchanged. Targets are a
permutation of the selected edges' target stubs, so every neuron's in-degree is unchanged too
(configuration model):
  1. collect the selected edges in presynaptic (CSC) order, i.e. dealt out by out-degree;
  2. permute their target stubs;
  3. repair self-loops and duplicate (pre, post) pairs by swapping the offending edge's target with
     that of a random non-offending selected edge (a swap keeps both degree sequences), until none remain.
Unselected edges are untouched. Selection:
  full             every presynaptic neuron
  sensory_only     the sensory input set (indices.json sensory_idx)
  downstream_only  every presynaptic neuron except the sensory input set
Each shuffle is verified (in/out-degrees, per-neuron weight multisets, one sign per presynaptic neuron,
no duplicates, untouched edges identical) and cached as cache/shuffles/<condition>_seed<seed>.npz + .json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp

from config import Paths, ShuffleConfig

CACHE = ROOT / Paths().cache
SHUFFLES = CACHE / "shuffles"
CONDITIONS = ("full", "sensory_only", "downstream_only")


def shuffle_path(condition, seed):
    return SHUFFLES / f"{condition}_seed{seed}.npz"


def selection_mask(condition, n, sensory_idx):
    sens = np.zeros(n, dtype=bool)
    sens[np.asarray(sensory_idx)] = True
    if condition == "full":
        return np.ones(n, dtype=bool)
    if condition == "sensory_only":
        return sens
    if condition == "downstream_only":
        return ~sens
    raise ValueError(f"unknown condition {condition!r}")


def conflicts(e_pre, new_post, n):
    """Boolean mask of edges that are self-loops or a repeated (pre, post) pair (all but the first)."""
    key = e_pre * n + new_post
    order = np.argsort(key, kind="stable")
    ks = key[order]
    dup = np.zeros(key.size, dtype=bool)
    dup[order[1:][ks[1:] == ks[:-1]]] = True
    return dup | (e_pre == new_post)


def retarget(e_pre, targets, n, rng, max_iter):
    """Core of the shuffle: permute the target stubs of edges (e_pre grouped by presynaptic neuron), then repair
    self-loops and duplicate pairs by swapping with random non-offending edges. Returns (new targets, conflict history)."""
    new_post = rng.permutation(targets)
    history = []
    for it in range(max_iter):
        bad = conflicts(e_pre, new_post, n)
        nb = int(bad.sum())
        history.append(nb)
        if nb == 0:
            return new_post, history
        bi = rng.permutation(np.flatnonzero(bad))
        cand = np.unique(rng.integers(0, e_pre.size, size=2 * nb + 64))
        cand = rng.permutation(cand[~bad[cand]])
        k = min(bi.size, cand.size)
        a, b = bi[:k], cand[:k]
        new_post[a], new_post[b] = new_post[b], new_post[a]
    raise RuntimeError(f"shuffle did not converge in {max_iter} repair iterations (conflicts {history[-5:]})")


def shuffle_targets(W, mask_pre, seed, max_iter):
    n = W.shape[0]
    C = W.tocsc()
    C.sort_indices()
    outdeg = np.diff(C.indptr)
    pre = np.repeat(np.arange(n, dtype=np.int64), outdeg)
    post = C.indices.astype(np.int64)
    sel = np.flatnonzero(mask_pre[pre])
    e_pre = pre[sel]
    new_post, history = retarget(e_pre, post[sel], n, np.random.default_rng(seed), max_iter)
    new_all = post.copy()
    new_all[sel] = new_post
    S = sp.csr_matrix((C.data, (new_all, pre)), shape=W.shape, dtype=np.float32)
    S.sort_indices()
    return S, {"selected_edges": int(sel.size), "repair_iterations": len(history) - 1,
               "conflicts_initial": history[0], "conflicts_history_head": history[:10],
               "frac_targets_unchanged": float((new_post == post[sel]).mean())}


def verify(W, S, mask_pre):
    """Raise unless S preserves degrees, weight multisets, per-neuron sign and the unselected edges of W."""
    if S.nnz != W.nnz:
        raise AssertionError(f"nnz {S.nnz} != {W.nnz} (duplicate pairs were merged)")
    Wr, Sr = W.tocsr(), S.tocsr()
    if not np.array_equal(np.diff(Wr.indptr), np.diff(Sr.indptr)):
        raise AssertionError("in-degree sequence changed")
    Wc, Sc = W.tocsc(), S.tocsc()
    Wc.sort_indices()
    Sc.sort_indices()
    if not np.array_equal(np.diff(Wc.indptr), np.diff(Sc.indptr)):
        raise AssertionError("out-degree sequence changed")
    col = np.repeat(np.arange(W.shape[1]), np.diff(Wc.indptr))
    o_w = np.lexsort((Wc.data, col))
    o_s = np.lexsort((Sc.data, col))
    if not np.array_equal(Wc.data[o_w], Sc.data[o_s]):
        raise AssertionError("a presynaptic neuron's outgoing weight multiset changed")
    pos = np.zeros(W.shape[1], dtype=bool)
    neg = np.zeros(W.shape[1], dtype=bool)
    np.logical_or.at(pos, col, Sc.data > 0)
    np.logical_or.at(neg, col, Sc.data < 0)
    if (pos & neg).any():
        raise AssertionError(f"{int((pos & neg).sum())} presynaptic neurons have outgoing weights of both signs")
    if not np.array_equal(np.sort(W.data), np.sort(S.data)):
        raise AssertionError("weight distribution changed")
    keep = ~mask_pre[col]
    if not (np.array_equal(Wc.indices[keep], Sc.indices[keep]) and np.array_equal(Wc.data[keep], Sc.data[keep])):
        raise AssertionError("an unselected presynaptic neuron's edges changed")
    return {"nnz": int(S.nnz), "self_loops_real": int(W.diagonal().astype(bool).sum()),
            "self_loops_shuffled": int(S.diagonal().astype(bool).sum()), "one_sign_per_presynaptic_neuron": True}


def build_shuffle(condition, seed, force=False, W=None, sensory_idx=None):
    path = shuffle_path(condition, seed)
    meta_path = path.with_suffix(".json")
    if path.exists() and meta_path.exists() and not force:
        return path, json.loads(meta_path.read_text())
    cfg = ShuffleConfig()
    if W is None:
        W = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    if sensory_idx is None:
        sensory_idx = json.loads((CACHE / "indices.json").read_text())["sensory_idx"]
    mask = selection_mask(condition, W.shape[0], sensory_idx)
    t0 = time.perf_counter()
    S, info = shuffle_targets(W, mask, seed, cfg.max_repair_iterations)
    t_shuffle = time.perf_counter() - t0
    checks = verify(W, S, mask)
    SHUFFLES.mkdir(parents=True, exist_ok=True)
    sp.save_npz(path, S, compressed=True)
    meta = {"condition": condition, "seed": seed, **info, **checks, "shuffle_seconds": round(t_shuffle, 1),
            "total_seconds": round(time.perf_counter() - t0, 1), "verified": True}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"  shuffle {condition} seed {seed}: {info['selected_edges']:,} edges re-targeted, {info['conflicts_initial']:,} "
          f"initial conflicts repaired in {info['repair_iterations']} iterations, {info['frac_targets_unchanged']:.2%} "
          f"targets unchanged; verified (degrees, weight multisets, one sign per neuron, untouched edges); "
          f"self-loops {checks['self_loops_real']} -> {checks['self_loops_shuffled']}; {meta['total_seconds']:.0f} s", flush=True)
    return path, meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("condition", choices=CONDITIONS)
    ap.add_argument("seed", type=int)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    build_shuffle(a.condition, a.seed, a.force)
