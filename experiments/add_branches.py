"""Graph surgery: add synapses from the hop-1 population onto the hop-2 descending neurons.

    .venv/bin/python -m experiments.add_branches --k K --seed S
        [--policy spread|concentrated] [--targets-frac F] [--force]

The hop-2 descending neurons are already connected to the hop-1 population, yet
experiments/transmission.py finds them at chance. This module builds the modified matrices that
experiments/branch_sweep.py measures: it adds K synapses and changes nothing else.

Hop distances are computed ONCE from the UNMODIFIED cache/adjacency.npz with
data.fetch_connectome.bfs_levels over G = W.T (pre -> post), sources = indices.json sensory_idx:

    SOURCES = the BranchConfig.expect_sources neurons at hop BranchConfig.source_hop
    TARGETS = the BranchConfig.expect_targets motor_idx neurons at hop BranchConfig.target_hop

Both counts are asserted. Sampling rules, all mandatory:
  * pre uniform over SOURCES; post uniform over TARGETS ("spread") or over a random
    ceil(targets_frac * |TARGETS|)-subset of them fixed by the seed ("concentrated");
  * a draw is rejected and redrawn if W[post, pre] is already stored or pre == post;
  * the weight MAGNITUDE is drawn with replacement from the empirical np.abs(W.data) of the whole
    matrix - no fitted distribution, no constant;
  * the SIGN is inherited from the presynaptic neuron: the sign of the first stored value of that
    column of W.tocsc(), after asserting that every stored value of that column shares it.

cache/adjacency.npz is never modified. Output: cache/branches/<policy>_k<K>_seed<S>.npz plus a
sibling .json manifest.
"""
import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp

from config import BranchConfig, Paths
from data.fetch_connectome import bfs_levels

CACHE = ROOT / Paths().cache
ADJ = CACHE / "adjacency.npz"


def branch_dir(bc=BranchConfig()):
    d = CACHE / bc.branch_subdir
    d.mkdir(parents=True, exist_ok=True)
    return d


def branch_path(policy, k, seed, bc=BranchConfig(), targets_frac=None):
    """One path per (policy, targets_frac, k, seed). targets_frac is part of the name for the
    concentrated arm: without it, two Stage-3 conditions that differ only in targets_frac would
    map to the same file and silently overwrite each other."""
    if policy == "concentrated":
        f = bc.targets_frac if targets_frac is None else targets_frac
        return branch_dir(bc) / f"{policy}_f{f:g}_k{k}_seed{seed}.npz"
    return branch_dir(bc) / f"{policy}_k{k}_seed{seed}.npz"


def matched_arm_weights(k, seed, bc=BranchConfig()):
    """The excitatory and inhibitory magnitude multisets actually used by the spread arm at this
    (k, seed), read back from the matrix it produced. The displaced arm reuses them exactly, so the
    two arms match in count, in sign split and in the joint (sign, |w|) distribution, and differ in
    PLACEMENT and in nothing else."""
    real = branch_path("spread", k, seed, bc)
    if not real.exists():
        raise FileNotFoundError(f"{real.name} must be built before the displaced arm can match it")
    W0 = sp.load_npz(ADJ).tocsr()
    diff = (sp.load_npz(real).tocsr() - W0).tocoo()
    diff.eliminate_zeros()
    if diff.nnz != k:
        raise AssertionError(f"matched arm has {diff.nnz} added edges, expected {k}")
    return np.abs(diff.data[diff.data > 0]), np.abs(diff.data[diff.data < 0])


def load_original():
    """The unmodified signed matrix W[i_post, j_pre] and indices.json."""
    W = sp.load_npz(ADJ).tocsr()
    W.sort_indices()
    indices = json.loads((CACHE / "indices.json").read_text())
    if list(W.shape) != indices["shape"]:
        raise ValueError(f"adjacency shape {W.shape} does not match indices.json {indices['shape']}")
    return W, indices


def hop_distances(W, indices):
    """BFS hop distance from the sensory set over the UNMODIFIED matrix (pre -> post edges)."""
    return bfs_levels(W.T.tocsr(), np.asarray(indices["sensory_idx"]))


def wrong_target_sets(W, indices, dist, sources, targets, seed, bc=BranchConfig()):
    """Arm D's populations: the REAL sources (hop-1), and an equal-sized random draw from the hop-2
    neurons that are NOT descending. Matching the target-set size to |TARGETS| keeps edges-per-target
    identical to the real arm, so the two differ only in WHICH hop-2 cells receive the edges."""
    motor = np.asarray(indices["motor_idx"])
    pool = np.flatnonzero(dist == bc.wrong_targets_hop)
    pool = np.setdiff1d(pool, motor)
    pool = np.setdiff1d(pool, np.concatenate([sources, targets]))
    if pool.size < targets.size:
        raise AssertionError(f"only {pool.size} non-descending hop-{bc.wrong_targets_hop} neurons, "
                             f"need {targets.size}")
    rng = np.random.default_rng(seed)
    return sources, np.sort(rng.choice(pool, size=targets.size, replace=False)), pool.size


def displaced_sets(W, indices, dist, sources, targets, bc=BranchConfig()):
    """The placement control's populations: hop-BranchConfig.displaced_hop neurons that are NOT motor
    neurons, with every neuron of SOURCES or TARGETS excluded. Nothing here is on the path from the
    auditory input to the descending layer that the real arm wires into."""
    motor = np.asarray(indices["motor_idx"])
    pool = np.flatnonzero(dist == bc.displaced_hop)
    pool = np.setdiff1d(pool, motor)
    pool = np.setdiff1d(pool, np.concatenate([sources, targets]))
    if pool.size < 2:
        raise AssertionError(f"displaced pool has {pool.size} neurons")
    return pool, pool


def source_target_sets(W, indices, dist, bc=BranchConfig()):
    """(SOURCES, TARGETS) with their expected sizes asserted."""
    sources = np.flatnonzero(dist == bc.source_hop)
    motor = np.asarray(indices["motor_idx"])
    targets = np.sort(motor[dist[motor] == bc.target_hop])
    if sources.size != bc.expect_sources:
        raise AssertionError(f"hop-{bc.source_hop} population is {sources.size}, expected {bc.expect_sources}")
    if targets.size != bc.expect_targets:
        raise AssertionError(f"hop-{bc.target_hop} descending neurons are {targets.size}, "
                             f"expected {bc.expect_targets}")
    return sources, targets


def column_signs(W):
    """Sign of every presynaptic column of W, asserting that no column mixes signs.

    Returns (signs int8 [N], n_empty). An empty column has sign 0: that neuron has no stored
    outgoing synapse in this matrix, so there is no sign to inherit and it CANNOT be used as a
    presynaptic source. Callers must exclude those neurons explicitly rather than assume a sign."""
    Wc = W.tocsc()
    starts, ends = Wc.indptr[:-1], Wc.indptr[1:]
    nonempty = ends > starts
    signs = np.zeros(W.shape[1], dtype=np.int8)
    signs[nonempty] = np.sign(Wc.data[starts[nonempty]]).astype(np.int8)
    col_of = np.repeat(np.arange(W.shape[1]), np.diff(Wc.indptr))
    mixed = np.sign(Wc.data).astype(np.int8) != signs[col_of]
    if mixed.any():
        bad = np.unique(col_of[mixed])
        raise AssertionError(f"{bad.size} presynaptic columns mix signs (first 10: {bad[:10].tolist()}); "
                             "aborting - the sign rule is not repairable here")
    if (Wc.data == 0).any():
        raise AssertionError(f"{int((Wc.data == 0).sum())} stored entries are exactly zero")
    return signs, int((~nonempty).sum())


def existing_keys(W):
    """Sorted int64 keys row * N + col of every stored entry (CSR with sorted indices is already
    in ascending key order; asserted)."""
    n = W.shape[0]
    rows = np.repeat(np.arange(n, dtype=np.int64), np.diff(W.indptr))
    keys = rows * n + W.indices.astype(np.int64)
    if not (np.diff(keys) > 0).all():
        raise AssertionError("CSR keys are not strictly ascending (duplicate or unsorted indices)")
    return keys


def is_stored(keys, cand):
    pos = np.searchsorted(keys, cand)
    pos_c = np.clip(pos, 0, keys.size - 1)
    return keys[pos_c] == cand


def sample_edges(n, k, sources, target_pool, keys, rng, batch):
    """k distinct (post, pre) positions, pre uniform over sources, post uniform over target_pool,
    never a self-loop and never a position already stored in W. Returns (post, pre) int64 arrays."""
    acc = np.empty(0, dtype=np.int64)
    rounds, drawn = 0, 0
    while acc.size < k:
        need = k - acc.size
        m = min(batch, max(need * 4, 10000))
        pre = rng.choice(sources, size=m)
        post = rng.choice(target_pool, size=m)
        drawn += m
        rounds += 1
        cand = post.astype(np.int64) * n + pre.astype(np.int64)
        cand = cand[pre != post]
        cand = np.unique(cand)
        cand = cand[~is_stored(keys, cand)]
        if acc.size:
            cand = cand[~is_stored(acc, cand)]
        if cand.size > need:
            cand = rng.permutation(cand)[:need]      # never truncate a sorted array: that would bias post
        acc = np.union1d(acc, cand)
        if rounds > 1000:
            raise RuntimeError(f"edge sampling did not converge: {acc.size}/{k} after {rounds} rounds")
    if acc.size != k:
        raise AssertionError(f"sampled {acc.size} edges, wanted {k}")
    return acc // n, acc % n, {"rounds": rounds, "candidates_drawn": int(drawn),
                               "acceptance": float(k / drawn)}


def verify(W, W_new, post, pre, vals, post_pool, k, signs_orig, pool_label="TARGETS"):
    """Every check printed as an assertion before anything is written."""
    n = W.shape[0]
    print("verification:")
    assert W_new.shape == W.shape, f"shape {W_new.shape} != {W.shape}"
    print(f"  shape unchanged                                  {W_new.shape}  OK")
    assert W_new.dtype == W.dtype, f"dtype {W_new.dtype} != {W.dtype}"
    print(f"  dtype unchanged                                  {W_new.dtype}  OK")
    assert W_new.nnz == W.nnz + k, f"nnz {W_new.nnz} != {W.nnz} + {k}"
    print(f"  nnz == original + k                              {W.nnz:,} + {k:,} = {W_new.nnz:,}  OK")

    diff = (W_new - W).tocoo()
    diff.eliminate_zeros()
    assert diff.nnz == k, f"(W_new - W) has {diff.nnz} stored entries, expected exactly {k}"
    dk = np.sort(diff.row.astype(np.int64) * n + diff.col.astype(np.int64))
    ak_unsorted = post * n + pre
    ak = np.sort(ak_unsorted)
    assert np.array_equal(dk, ak), "the changed positions are not the sampled positions"
    assert not is_stored(existing_keys(W), dk).any(), "a changed position was already stored in W_orig"
    print(f"  (W_new - W_orig) has exactly k entries, all at positions that were zero  OK")
    order = np.argsort(diff.row.astype(np.int64) * n + diff.col.astype(np.int64))
    assert np.allclose(diff.data[order], vals[np.argsort(ak_unsorted)]), \
        "changed values differ from the sampled values"
    print(f"  no original entry changed by any amount (diff values == added values)   OK")

    signs_new, _ = column_signs(W_new)
    assert np.array_equal(signs_new[signs_orig != 0], signs_orig[signs_orig != 0]), \
        "a presynaptic column changed sign"
    print(f"  still one sign per presynaptic column, unchanged where it existed       OK")

    assert (post != pre).all(), "self-loop among the added edges"
    assert np.unique(ak).size == k, "duplicate (post, pre) among the added edges"
    print(f"  no self-loops, no duplicate (post, pre)          {k:,} distinct positions  OK")

    in_old, in_new = np.diff(W.indptr), np.diff(W_new.indptr)
    mask = np.ones(n, dtype=bool)
    mask[post_pool] = False
    assert np.array_equal(in_old[mask], in_new[mask]), f"in-degree changed outside {pool_label}"
    changed = np.flatnonzero(in_old != in_new)
    assert np.isin(changed, post_pool).all(), f"in-degree changed outside {pool_label}"
    print(f"  in-degree unchanged for all {int(mask.sum()):,} neurons outside {pool_label}  OK")
    return in_old, in_new


def build(k, seed, policy="spread", targets_frac=None, force=False, bc=BranchConfig(), verbose=True):
    """Write cache/branches/<policy>_k<K>_seed<S>.npz (+ .json). Returns (path, manifest)."""
    if policy not in bc.policies:
        raise ValueError(f"policy must be one of {bc.policies}, got {policy!r}")
    if k < 0:
        raise ValueError(f"k must be >= 0, got {k}")
    targets_frac = bc.targets_frac if targets_frac is None else targets_frac
    path = branch_path(policy, k, seed, bc, targets_frac)
    man_path = path.with_suffix(".json")
    if k == 0:
        raise ValueError("k = 0 is the unmodified cache/adjacency.npz; branch_sweep uses it directly")
    if path.exists() and man_path.exists() and not force:
        man = json.loads(man_path.read_text())
        want = {"k": int(k), "policy": policy, "seed": int(seed),
                "targets_frac": float(targets_frac) if policy == "concentrated" else 1.0}
        bad = {f: (man.get(f), v) for f, v in want.items() if man.get(f) != v}
        if bad:
            print(f"cached {path.name} does NOT describe this condition ({bad}); rebuilding")
        else:
            if verbose:
                print(f"using cached {path.relative_to(ROOT)} (--force to rebuild)")
            return path, man

    t0 = time.perf_counter()
    W, indices = load_original()
    n = W.shape[0]
    dist = hop_distances(W, indices)
    sources, targets = source_target_sets(W, indices, dist, bc)
    signs, n_empty_cols = column_signs(W)
    if verbose:
        print(f"W {W.shape} nnz {W.nnz:,} dtype {W.dtype}; SOURCES (hop {bc.source_hop}) {sources.size}, "
              f"TARGETS (motor at hop {bc.target_hop}) {targets.size}; {n_empty_cols:,} neurons have an "
              f"empty presynaptic column network-wide")

    usable = sources[signs[sources] != 0]
    n_sourceless = int(sources.size - usable.size)
    if n_sourceless and verbose:
        print(f"  NOTE: {n_sourceless} of the {sources.size} SOURCES have NO stored outgoing synapse, so the "
              f"sign rule has no value to inherit; they are excluded from the pre draw "
              f"({usable.size} usable sources). Indices: {sources[signs[sources] == 0].tolist()}")
    if usable.size == 0:
        raise AssertionError("no usable presynaptic sources")

    rng = np.random.default_rng(seed)
    if policy in ("displaced", "wrong_targets"):
        return _build_matched(W, indices, dist, sources, targets, signs, k, seed, rng, path,
                              man_path, bc, verbose, policy)
    if policy == "concentrated":
        n_sub = math.ceil(targets_frac * targets.size)
        if not 1 <= n_sub <= targets.size:
            raise ValueError(f"targets_frac {targets_frac} gives a subset of {n_sub} of {targets.size}")
        target_pool = np.sort(rng.choice(targets, size=n_sub, replace=False))
    else:
        n_sub = targets.size
        target_pool = targets
    max_edges = int(usable.size) * int(target_pool.size)
    already = int(W[target_pool][:, usable].nnz)
    if k > max_edges - already:
        raise ValueError(f"k = {k:,} exceeds the {max_edges - already:,} free (post, pre) positions "
                         f"between {usable.size} sources and {target_pool.size} targets")
    if verbose:
        print(f"  policy {policy}: target pool {target_pool.size} of {targets.size} "
              f"(targets_frac {targets_frac if policy == 'concentrated' else 1.0:g}); "
              f"{already:,} of the {max_edges:,} source->target positions are already stored")

    keys = existing_keys(W)
    post, pre, samp = sample_edges(n, k, usable, target_pool, keys, rng, bc.sample_batch)
    abs_data = np.abs(W.data)
    mag = abs_data[rng.integers(0, abs_data.size, size=k)]
    vals = (mag * signs[pre].astype(np.float32)).astype(W.dtype)
    assert (vals != 0).all(), "an added weight is zero"
    assert np.array_equal(np.sign(vals).astype(np.int8), signs[pre]), "an added weight lost its column sign"

    add = sp.coo_matrix((vals, (post, pre)), shape=W.shape, dtype=W.dtype)
    W_new = (W + add).tocsr()
    W_new.sort_indices()
    in_old, in_new = verify(W, W_new, post, pre, vals, targets, k, signs)

    row_abs_old = np.asarray(abs(W).sum(axis=1)).ravel()
    row_abs_new = np.asarray(abs(W_new).sum(axis=1)).ravel()
    delta = (in_new - in_old)[targets]
    n_exc = int((vals > 0).sum())
    manifest = {
        "k": int(k), "policy": policy, "seed": int(seed),
        "targets_frac": float(targets_frac) if policy == "concentrated" else 1.0,
        "target_subset_size": int(n_sub),
        "n_sources": int(sources.size), "n_sources_usable": int(usable.size),
        "n_sources_excluded_no_outgoing": n_sourceless,
        "n_targets": int(targets.size),
        "nnz_original": int(W.nnz), "nnz_new": int(W_new.nnz),
        "sign_split": {"excitatory": n_exc, "inhibitory": int(k - n_exc),
                       "exc_fraction": float(n_exc / k)},
        "added_weight_percentiles": {str(p): float(np.percentile(mag, p)) for p in (0, 10, 25, 50, 75, 90, 100)},
        "added_weight_mean": float(mag.mean()),
        "matrix_abs_weight_percentiles": {str(p): float(np.percentile(abs_data, p))
                                          for p in (0, 10, 25, 50, 75, 90, 100)},
        "in_degree_change": {
            "targets_touched": int((delta > 0).sum()),
            "added_per_target_min": int(delta.min()), "added_per_target_median": float(np.median(delta)),
            "added_per_target_mean": float(delta.mean()), "added_per_target_max": int(delta.max()),
            "in_degree_median_before": float(np.median(in_old[targets])),
            "in_degree_median_after": float(np.median(in_new[targets])),
            "per_target": {int(t): int(d) for t, d in zip(targets, delta) if d > 0},
        },
        "row_abs_sum_targets": {
            "median_before": float(np.median(row_abs_old[targets])),
            "median_after": float(np.median(row_abs_new[targets])),
            "mean_before": float(row_abs_old[targets].mean()),
            "mean_after": float(row_abs_new[targets].mean()),
            "sqrt_in_scale_median_before": float(np.median(1.0 / np.sqrt(row_abs_old[targets]))),
            "sqrt_in_scale_median_after": float(np.median(1.0 / np.sqrt(row_abs_new[targets]))),
        },
        "sampling": samp,
        "branch_config": {k2: v for k2, v in asdict(bc).items()},
        "source_path": str(ADJ.relative_to(ROOT)), "wall_s": None,
    }
    sp.save_npz(path, W_new)
    manifest["wall_s"] = round(time.perf_counter() - t0, 2)
    man_path.write_text(json.dumps(manifest, indent=2))
    if verbose:
        m = manifest
        print(f"  added edges: {n_exc:,} excitatory / {k - n_exc:,} inhibitory "
              f"({m['sign_split']['exc_fraction']:.1%} exc); |w| median {m['added_weight_percentiles']['50']:g}, "
              f"mean {m['added_weight_mean']:.3g}, p90 {m['added_weight_percentiles']['90']:g}, "
              f"max {m['added_weight_percentiles']['100']:g}")
        d = m["in_degree_change"]
        print(f"  in-degree of TARGETS: {d['targets_touched']:,} of {targets.size:,} touched, "
              f"+{d['added_per_target_min']}..{d['added_per_target_max']} per target "
              f"(median +{d['added_per_target_median']:g}); median in-degree "
              f"{d['in_degree_median_before']:g} -> {d['in_degree_median_after']:g}")
        r = m["row_abs_sum_targets"]
        print(f"  row |W| sum of TARGETS: median {r['median_before']:g} -> {r['median_after']:g} "
              f"(sqrt_in row scale 1/sqrt(sum) median {r['sqrt_in_scale_median_before']:.4g} -> "
              f"{r['sqrt_in_scale_median_after']:.4g})")
        print(f"  sampling: {samp['rounds']} rounds, {samp['candidates_drawn']:,} candidates, "
              f"acceptance {samp['acceptance']:.1%}")
        print(f"wrote {path.relative_to(ROOT)} and {man_path.relative_to(ROOT)} "
              f"in {manifest['wall_s']:.1f} s")
    return path, manifest


def _build_matched(W, indices, dist, sources, targets, signs, k, seed, rng, path, man_path, bc,
                   verbose, policy):
    """The two matched control arms. Same k, same sign rule, and the EXACT magnitudes and
    excitatory / inhibitory counts of the spread arm at this (k, seed); only the placement differs.

        displaced      pre and post both from hop-3 non-motor neurons: controls for adding edges
                       at all, but its sources carry no signal.
        wrong_targets  pre from the REAL hop-1 sources, post from an equal-sized random draw of
                       NON-descending hop-2 neurons: the sources do carry signal, so what changes
                       is only which hop-2 cells receive it.
    """
    n = W.shape[0]
    pool_size = None
    if policy == "displaced":
        pre_pool_src, post_pool = displaced_sets(W, indices, dist, sources, targets, bc)
    else:
        pre_pool_src, post_pool, pool_size = wrong_target_sets(W, indices, dist, sources, targets,
                                                               seed, bc)
    pool = post_pool
    mags_exc, mags_inh = matched_arm_weights(k, seed, bc)
    n_exc = int(mags_exc.size)
    pre_pool_exc = pre_pool_src[signs[pre_pool_src] > 0]
    pre_pool_inh = pre_pool_src[signs[pre_pool_src] < 0]
    if verbose:
        if policy == "displaced":
            print(f"  policy displaced: pre and post both from {post_pool.size:,} "
                  f"hop-{bc.displaced_hop} non-motor neurons with SOURCES and TARGETS removed")
        else:
            print(f"  policy wrong_targets: pre = the same {pre_pool_src.size:,} hop-1 SOURCES as "
                  f"the real arm; post = {post_pool.size:,} neurons drawn from the {pool_size:,} "
                  f"NON-descending hop-{bc.wrong_targets_hop} neurons (target-set size matched to "
                  f"the {targets.size:,} real TARGETS, so edges per target match exactly)")
        print(f"  presynaptic pool: {pre_pool_exc.size:,} excitatory, {pre_pool_inh.size:,} "
              f"inhibitory columns")
        print(f"  matching the spread arm at k={k} seed={seed} exactly: {n_exc} excitatory / "
              f"{k - n_exc} inhibitory edges, and its own multiset of |w| per sign group "
              f"(exc median {np.median(mags_exc):g}, inh median {np.median(mags_inh):g}, "
              f"max {max(mags_exc.max(), mags_inh.max()):g})")
    keys = existing_keys(W)
    post_all, pre_all, val_all, samp_all = [], [], [], []
    for want_n, pre_pool, mag, sign in ((n_exc, pre_pool_exc, mags_exc, +1),
                                        (k - n_exc, pre_pool_inh, mags_inh, -1)):
        if want_n == 0:
            continue
        if pre_pool.size == 0:
            raise AssertionError(f"no {'excitatory' if sign > 0 else 'inhibitory'} sources in the "
                                 f"displaced pool")
        po, pr, samp = sample_edges(n, want_n, pre_pool, post_pool, keys, rng, bc.sample_batch)
        post_all.append(po)
        pre_all.append(pr)
        val_all.append((rng.permutation(mag) * sign).astype(W.dtype))
        samp_all.append(samp)
    post = np.concatenate(post_all)
    pre = np.concatenate(pre_all)
    vals = np.concatenate(val_all)
    assert np.unique(post * n + pre).size == k, "duplicate position across the two sign groups"
    assert np.array_equal(np.sign(vals).astype(np.int8), signs[pre]), "an added weight lost its column sign"

    add = sp.coo_matrix((vals, (post, pre)), shape=W.shape, dtype=W.dtype)
    W_new = (W + add).tocsr()
    W_new.sort_indices()
    in_old, in_new = verify(W, W_new, post, pre, vals, post_pool, k, signs,
                            pool_label=f"the {policy} target pool")
    assert np.array_equal(in_old[targets], in_new[targets]), \
        f"the {policy} arm changed the in-degree of a hop-2 descending TARGET"
    print(f"  in-degree of all {targets.size:,} real TARGETS unchanged by the {policy} arm  OK")

    row_abs_old = np.asarray(abs(W).sum(axis=1)).ravel()
    row_abs_new = np.asarray(abs(W_new).sum(axis=1)).ravel()
    delta = (in_new - in_old)[pool]
    manifest = {
        "k": int(k), "policy": policy, "seed": int(seed), "targets_frac": 1.0,
        "target_subset_size": int(post_pool.size), "pool_size": int(pool_size or post_pool.size),
        "hop": bc.displaced_hop if policy == "displaced" else bc.wrong_targets_hop,
        "matched_arm": branch_path("spread", k, seed, bc).name,
        "n_sources": int(sources.size), "n_targets": int(targets.size),
        "nnz_original": int(W.nnz), "nnz_new": int(W_new.nnz),
        "sign_split": {"excitatory": int(n_exc), "inhibitory": int(k - n_exc),
                       "exc_fraction": float(n_exc / k)},
        "added_weight_percentiles": {str(p): float(np.percentile(np.abs(vals), p))
                                     for p in (0, 10, 25, 50, 75, 90, 100)},
        "added_weight_mean": float(np.abs(vals).mean()),
        "in_degree_change": {"targets_touched": int((delta > 0).sum()),
                             "added_per_target_min": int(delta.min()),
                             "added_per_target_median": float(np.median(delta)),
                             "added_per_target_mean": float(delta.mean()),
                             "added_per_target_max": int(delta.max()),
                             "in_degree_median_before": float(np.median(in_old[pool])),
                             "in_degree_median_after": float(np.median(in_new[pool])),
                             "per_target": {int(t): int(d) for t, d in zip(pool, delta) if d > 0}},
        "row_abs_sum_targets": {"median_before": float(np.median(row_abs_old[targets])),
                                "median_after": float(np.median(row_abs_new[targets])),
                                "mean_before": float(row_abs_old[targets].mean()),
                                "mean_after": float(row_abs_new[targets].mean()),
                                "sqrt_in_scale_median_before":
                                    float(np.median(1.0 / np.sqrt(row_abs_old[targets]))),
                                "sqrt_in_scale_median_after":
                                    float(np.median(1.0 / np.sqrt(row_abs_new[targets])))},
        "sampling": {"rounds": sum(x["rounds"] for x in samp_all),
                     "candidates_drawn": sum(x["candidates_drawn"] for x in samp_all),
                     "acceptance": float(k / sum(x["candidates_drawn"] for x in samp_all))},
        "branch_config": {k2: v for k2, v in asdict(bc).items()},
        "source_path": str(ADJ.relative_to(ROOT)), "wall_s": None,
    }
    sp.save_npz(path, W_new)
    man_path.write_text(json.dumps(manifest, indent=2))
    if verbose:
        print(f"  {policy} edges land on {int((delta > 0).sum()):,} of the {post_pool.size:,} pool "
              f"neurons; the hop-2 descending row sums are untouched "
              f"(median {np.median(row_abs_old[targets]):g} -> {np.median(row_abs_new[targets]):g})")
        print(f"wrote {path.relative_to(ROOT)} and {man_path.relative_to(ROOT)}")
    return path, manifest


if __name__ == "__main__":
    bc = BranchConfig()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=1000, help="number of synapses to add")
    ap.add_argument("--seed", type=int, default=bc.stage1_seed)
    ap.add_argument("--policy", choices=list(bc.policies), default="spread")
    ap.add_argument("--targets-frac", type=float, default=bc.targets_frac)
    ap.add_argument("--force", action="store_true", help="rebuild even if the output exists")
    a = ap.parse_args()
    build(a.k, a.seed, a.policy, a.targets_frac, a.force, bc)
