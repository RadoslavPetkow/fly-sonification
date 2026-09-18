"""Three checks on the k = 100 crossing, before any of it is believed.

    .venv/bin/python -m experiments.branch_checks [--k 100] [--seed 1] [--shuffles 200]

Stage 1 found the hop-2 descending layer above the joint circular-shift null at k = 100 added
synapses (z = 5.44). Three things have to be true before that is a result. All three run off the
simulations already cached by experiments/branch_sweep.py - nothing is re-simulated.

CHECK 1 (decisive) - touched vs untouched targets. The added edges land on a minority of the 1,057
  hop-2 descending neurons. Split them into the ones that received at least one added edge and the
  ones that received none, and measure each group at k = 0 and at k = this k. The k = 0 values are
  THE SAME NEURONS under the same split, so this is a difference-in-differences. If only the touched
  group rises, the effect is local and caused by the added edges. If both rise together, the network
  moved globally and the headline k is meaningless.

CHECK 2 - frozen unit set. A layer's statistic is the mean over units passing
  TransmissionConfig.min_spikes, and that set moves with k (357 units at k = 0, 368 at k = 100, 315
  at k = 1000). Re-run the identical analysis on the set frozen at k = 0 so every condition scores
  the same neurons, and score the newly-qualifying units separately.

CHECK 3 - a 20-shuffle null cannot carry a search. "Above every one of 20 shuffles" is p < 1/21;
  the sweep makes ~18 such tests, so one spurious crossing is the EXPECTED outcome and the bottom of
  the ladder is where it would show up. Re-run every crossing with n_shuffles = 200 and report the
  exact empirical p = (1 + #(null >= real)) / (1 + n_shuffles). z is reported as a descriptive
  statistic only.

The added edges are recovered by differencing the branched matrix against the unmodified one, not by
replaying the sampler's RNG, so the per-target counts and signs used here are ground truth.
"""
import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp

from config import BranchConfig, Paths, TransmissionConfig
from experiments.add_branches import branch_path
from experiments.branch_sweep import Context, condition_id, descriptor_mismatch, sha12, sim_cache_path
from experiments.transmission import analyze
from sim.calibrate import banner, header

CACHE = ROOT / Paths().cache
RESULTS = CACHE / "branch_checks.json"


# =============================================================================
# cached simulations
# =============================================================================

def load_sim(ctx, k, policy="spread", seed=None, targets_frac=None):
    """counts, drive_binned from a cached simulation, after re-checking its stamped descriptor."""
    cid = condition_id(k, policy, seed, targets_frac)
    path = sim_cache_path(ctx.bc, cid)
    if not path.exists():
        raise FileNotFoundError(f"{path.relative_to(ROOT)} does not exist; run the sweep stage first")
    d = np.load(path)
    meta = json.loads(str(d["meta"]))
    adj = ROOT / "cache" / "adjacency.npz" if k == 0 else branch_path(policy, k, seed, ctx.bc, targets_frac)
    want = {"k": int(k), "duration_ms": float(ctx.bc.duration_ms), "adjacency": str(adj),
            "adjacency_sha12": sha12(adj), "w_scale": float(ctx.sc.w_scale),
            "g_inh": float(ctx.sc.g_inh), "normalization": ctx.sc.normalization}
    bad = descriptor_mismatch(meta, want)
    if bad:
        raise ValueError(f"{path.name} does not describe k={k} {policy} seed={seed}: {bad}")
    counts = d["counts"]
    print(f"  {cid}: {path.name}, adjacency sha {want['adjacency_sha12']}, "
          f"{int(counts.sum(dtype=np.int64)):,} spikes over {counts.shape[0]} bins")
    return counts, d["drive_binned"]


def added_edges(ctx, k, policy="spread", seed=1, targets_frac=None):
    """Ground truth about what was added: (post, pre, value) from (W_branched - W_original)."""
    W0 = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    W1 = sp.load_npz(branch_path(policy, k, seed, ctx.bc, targets_frac)).tocsr()
    diff = (W1 - W0).tocoo()
    diff.eliminate_zeros()
    if diff.nnz != k:
        raise AssertionError(f"the branched matrix differs from the original in {diff.nnz} entries, not {k}")
    return diff.row.astype(np.int64), diff.col.astype(np.int64), diff.data


# =============================================================================
# the statistic, over explicit unit sets
# =============================================================================

def score_sets(counts, drive_binned, sets, tc, stat="mi"):
    """Run the SHARED analyze() over named index sets, keeping each unit's own score. Empty sets
    are skipped. Returns {name: stat-dict + n_units + members + the per-unit distributions}."""
    layers, order, members = {}, [], {}
    for name, idx in sets.items():
        idx = np.asarray(idx, dtype=np.int64)
        if idx.size == 0:
            continue
        layers[name] = counts[:, idx].T
        members[name] = idx
        order.append(name)
    if not layers:
        return {}
    res, _ = analyze(counts, drive_binned, layers, tc, np.random.default_rng(tc.seed + 2),
                     per_unit=True)
    out = {}
    for name in order:
        d = res[name][stat] | {"n_units": res[name]["n_units"], "members": members[name]}
        real = np.asarray(res[name]["unit_scores"][stat])
        null = np.asarray(res[name]["null_unit_scores"][stat])
        d["unit_scores"] = real
        d["null_unit_scores"] = null
        d["dist"] = distribution(real, null)
        out[name] = d
    return out


def spearman(x, y):
    """Spearman rank correlation (average ranks for ties)."""
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def ks_two_sample(a, b):
    """Two-sample Kolmogorov-Smirnov statistic between two 1-D samples (no p-value: the pooled null
    units are not independent of each other, so only the statistic is reported)."""
    a, b = np.sort(np.asarray(a, dtype=np.float64)), np.sort(np.asarray(b, dtype=np.float64))
    if a.size == 0 or b.size == 0:
        return float("nan")
    grid = np.concatenate([a, b])
    return float(np.max(np.abs(np.searchsorted(a, grid, "right") / a.size
                               - np.searchsorted(b, grid, "right") / b.size)))


def distribution(real, null):
    """Mean / median / IQR / top-5 of the per-unit real scores and of the pooled per-unit nulls,
    plus the KS statistic between the two. A mean that moves while the median does not is a few
    outliers, not a layer that started transmitting."""
    def d(x):
        q1, q3 = np.percentile(x, [25, 75])
        return {"mean": float(x.mean()), "median": float(np.median(x)), "q1": float(q1),
                "q3": float(q3), "iqr": float(q3 - q1),
                "top5": [float(v) for v in np.sort(x)[::-1][:5]]}
    return {"real": d(real), "null": d(null), "ks": ks_two_sample(real, null),
            "n_real": int(real.size), "n_null": int(null.size)}


def print_distribution(label, s):
    if s is None:
        return
    r, n = s["dist"]["real"], s["dist"]["null"]
    print(f"  {label:<30} real  mean {r['mean']:.5f}  median {r['median']:.5f}  "
          f"IQR [{r['q1']:.5f}, {r['q3']:.5f}]  top5 " + " ".join(f"{v:.4f}" for v in r["top5"]))
    print(f"  {'':<30} null  mean {n['mean']:.5f}  median {n['median']:.5f}  "
          f"IQR [{n['q1']:.5f}, {n['q3']:.5f}]  top5 " + " ".join(f"{v:.4f}" for v in n["top5"]))
    print(f"  {'':<30} KS(real, pooled null) = {s['dist']['ks']:.4f}  "
          f"(median shift {r['median'] - n['median']:+.5f}, mean shift {r['mean'] - n['mean']:+.5f})")


def row(label, s, extra=""):
    if s is None:
        return f"  {label:<34} {'-':>6}  (no units)"
    z = "n/a" if s["z"] is None else f"{s['z']:.2f}"
    return (f"  {label:<34} {s['n_units']:>5} {s['mean_best']:>9.5f} {s['null_mean']:>9.5f} "
            f"{s['null_sd']:>8.5f} {z:>7} {s['p_empirical']:>8.4f} "
            f"{s['n_units_above_own_null_max']:>5}/{s['n_units']:<5}{extra}")


HEAD = (f"  {'set':<34} {'units':>5} {'MI':>9} {'null mean':>9} {'null sd':>8} {'z':>7} "
        f"{'p':>8} {'units>null':>11}")

HEAVY = ("unit_scores", "null_unit_scores", "members", "null_values")


def strip_arrays(o):
    """The per-unit arrays stay in memory for the analysis; only summaries go to json."""
    if isinstance(o, dict):
        return {k: strip_arrays(v) for k, v in o.items() if k not in HEAVY}
    if isinstance(o, list):
        return [strip_arrays(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


# =============================================================================
# checks
# =============================================================================

def check1(ctx, k, seed, tc, out):
    header(f"CHECK 1 - touched vs untouched hop-2 descending targets, k = 0 vs k = {k}")
    post, pre, val = added_edges(ctx, k, "spread", seed)
    targets = ctx.targets
    n_added = np.zeros(ctx.dist.size, dtype=np.int64)
    n_exc = np.zeros(ctx.dist.size, dtype=np.int64)
    np.add.at(n_added, post, 1)
    np.add.at(n_exc, post, (val > 0).astype(np.int64))
    if not np.isin(post, targets).all():
        raise AssertionError("an added edge landed outside TARGETS")
    touched = targets[n_added[targets] > 0]
    untouched = targets[n_added[targets] == 0]
    print(f"  {k} added edges onto {touched.size} of {targets.size} targets "
          f"({untouched.size} untouched); per touched target "
          f"{np.bincount(n_added[touched]).tolist()[1:]} (1, 2, ... edges); "
          f"{int((val > 0).sum())} excitatory / {int((val < 0).sum())} inhibitory overall")

    c0, d0 = load_sim(ctx, 0)
    c1, d1 = load_sim(ctx, k, "spread", seed)
    active0 = c0.sum(axis=0, dtype=np.int64) >= tc.min_spikes      # the FROZEN set: k = 0 qualifiers

    def sets_for(idx_all):
        return {"variable": idx_all, "frozen": idx_all[active0[idx_all]]}

    groups = {"touched": touched, "untouched": untouched}
    for n in sorted(set(n_added[touched].tolist())):
        groups[f"touched n_added={n}"] = touched[n_added[touched] == n]
    allexc = touched[n_exc[touched] == n_added[touched]]
    allinh = touched[n_exc[touched] == 0]
    mixed = touched[(n_exc[touched] > 0) & (n_exc[touched] < n_added[touched])]
    groups["touched all-excitatory"] = allexc
    groups["touched all-inhibitory"] = allinh
    if mixed.size:
        groups["touched mixed sign"] = mixed

    res = {}
    for kind in ("variable", "frozen"):
        print(f"\n  unit set: {kind.upper()}"
              + (" (each condition's own >= min_spikes units)" if kind == "variable"
                 else " (the k = 0 qualifiers, identical neurons in both conditions)"))
        print(HEAD)
        for gname, idx in groups.items():
            s0 = score_sets(c0, d0, {gname: sets_for(idx)[kind]}, tc).get(gname)
            s1 = score_sets(c1, d1, {gname: sets_for(idx)[kind]}, tc).get(gname)
            delta = "" if (s0 is None or s1 is None) else f"   dMI {s1['mean_best'] - s0['mean_best']:+.5f}"
            print(row(f"{gname} @ k=0", s0))
            print(row(f"{gname} @ k={k}", s1, delta))
            res.setdefault(kind, {})[gname] = {"k0": s0, f"k{k}": s1,
                                               "delta_mi": None if (s0 is None or s1 is None)
                                               else s1["mean_best"] - s0["mean_best"],
                                               "n_members": int(np.asarray(idx).size)}
    # ---- the difference in differences
    header("CHECK 1 VERDICT")
    verdict = {}
    for kind in ("variable", "frozen"):
        t = res[kind]["touched"]
        u = res[kind]["untouched"]
        if t["delta_mi"] is None or u["delta_mi"] is None:
            continue
        did = t["delta_mi"] - u["delta_mi"]
        # scale both changes by the null sd of that group at k = 0: "how many null widths did it move"
        t_sd = t["k0"]["null_sd"] or float("nan")
        u_sd = u["k0"]["null_sd"] or float("nan")
        print(f"  {kind:<9}: touched dMI {t['delta_mi']:+.5f} ({t['delta_mi'] / t_sd:+.2f} null sd), "
              f"untouched dMI {u['delta_mi']:+.5f} ({u['delta_mi'] / u_sd:+.2f} null sd), "
              f"difference-in-differences {did:+.5f}")
        verdict[kind] = {"touched_delta": t["delta_mi"], "untouched_delta": u["delta_mi"],
                         "touched_delta_in_null_sd": t["delta_mi"] / t_sd,
                         "untouched_delta_in_null_sd": u["delta_mi"] / u_sd,
                         "difference_in_differences": did}
    out["check1"] = {"k": k, "seed": seed, "n_touched": int(touched.size),
                     "n_untouched": int(untouched.size),
                     "edges_per_touched": np.bincount(n_added[touched]).tolist(),
                     "n_exc": int((val > 0).sum()), "n_inh": int((val < 0).sum()),
                     "groups": res, "verdict": verdict}
    return res


def check2(ctx, ks, tc, out):
    header("CHECK 2 - the whole hop-2 descending layer on a unit set frozen at k = 0")
    c0, d0 = load_sim(ctx, 0)
    total0 = c0.sum(axis=0, dtype=np.int64)
    frozen = ctx.targets[total0[ctx.targets] >= tc.min_spikes]
    print(f"  frozen set: {frozen.size} of {ctx.targets.size} hop-2 descending neurons pass "
          f"min_spikes = {tc.min_spikes} at k = 0; that same set is scored at every k")
    res = {}
    print(HEAD)
    for k in ks:
        counts, drive = (c0, d0) if k == 0 else load_sim(ctx, k, "spread", ctx.bc.stage1_seed)
        total = counts.sum(axis=0, dtype=np.int64)
        variable = ctx.targets[total[ctx.targets] >= tc.min_spikes]
        newly = np.setdiff1d(variable, frozen)
        lost = np.setdiff1d(frozen, variable)
        s = score_sets(counts, drive, {"frozen": frozen, "variable": variable, "newly": newly}, tc)
        print(row(f"k={k:<7} frozen", s.get("frozen")))
        print(row(f"k={k:<7} variable", s.get("variable"),
                  f"   ({newly.size} newly qualifying, {lost.size} dropped out)"))
        if newly.size:
            print(row(f"k={k:<7} newly-qualifying only", s.get("newly")))
        print_distribution(f"k={k} variable", s.get("variable"))
        print_distribution(f"k={k} frozen", s.get("frozen"))
        res[str(k)] = {"frozen": s.get("frozen"), "variable": s.get("variable"),
                       "newly": s.get("newly"), "n_newly": int(newly.size), "n_lost": int(lost.size)}
    out["check2"] = {"frozen_set_size": int(frozen.size), "by_k": res}
    return res


def check3(ctx, ks, tc_small, tc_big, out):
    header(f"CHECK 3 - the same layers against {tc_big.n_shuffles} shuffles instead of "
           f"{tc_small.n_shuffles}")
    c0, d0 = load_sim(ctx, 0)
    total0 = c0.sum(axis=0, dtype=np.int64)
    frozen = ctx.targets[total0[ctx.targets] >= tc_small.min_spikes]
    n_tests = len(ctx.bc.stage1_k) - 1
    print(f"  a {tc_small.n_shuffles}-shuffle null gives a best possible p of "
          f"1/{tc_small.n_shuffles + 1} = {1 / (tc_small.n_shuffles + 1):.4f}; Stage 1 alone makes "
          f"{n_tests} such tests, so ~{n_tests / (tc_small.n_shuffles + 1):.2f} spurious crossings "
          f"are expected by chance. Bonferroni threshold over {n_tests} tests: "
          f"p <= {0.05 / n_tests:.5f}, which needs at least "
          f"{int(np.ceil(1 / (0.05 / n_tests) - 1))} shuffles to be reachable at all.")
    res = {}
    print(HEAD)
    for k in ks:
        counts, drive = (c0, d0) if k == 0 else load_sim(ctx, k, "spread", ctx.bc.stage1_seed)
        total = counts.sum(axis=0, dtype=np.int64)
        variable = ctx.targets[total[ctx.targets] >= tc_small.min_spikes]
        t0 = time.perf_counter()
        s = score_sets(counts, drive, {"frozen": frozen, "variable": variable}, tc_big)
        for name in ("frozen", "variable"):
            st = s.get(name)
            print(row(f"k={k:<7} {name} x{tc_big.n_shuffles}", st))
            res.setdefault(str(k), {})[name] = st
        print(f"    ({time.perf_counter() - t0:.1f} s)")
    out["check3"] = {"n_shuffles": tc_big.n_shuffles, "n_tests_stage1": n_tests,
                     "bonferroni_p": 0.05 / n_tests, "by_k": res}
    return res


def percell_contrast(ctx, k, seed, tc, out, policy="spread", n_perm=10000, n_boot=10000):
    """Check 1 done on the DISTRIBUTIONS instead of the group means.

    Comparing a 94-cell mean against a 963-cell mean in units of each group's own null sd is not a
    fair effect-size comparison: the larger group's mean has a ~sqrt(963/94) = 3.2x tighter null
    purely from n, so the normalisation reverses the ranking for reasons that have nothing to do
    with the treatment. The right object is each CELL's own change from k = 0 to k = k, and the
    right question is whether the touched cells' change distribution differs from the untouched
    cells' - tested with Mann-Whitney U, a cell-label permutation test, and reported as an effect
    size with a confidence interval."""
    from scipy.stats import mannwhitneyu

    header(f"PER-CELL CONTRAST - dMI distribution of touched vs untouched cells, k = 0 -> {k} "
           f"({policy})")
    post, _, val = added_edges(ctx, k, policy, seed)
    n_added = np.zeros(ctx.dist.size, dtype=np.int64)
    np.add.at(n_added, post, 1)
    c0, d0 = load_sim(ctx, 0)
    c1, d1 = load_sim(ctx, k, policy, seed)
    total0 = c0.sum(axis=0, dtype=np.int64)
    frozen = ctx.targets[total0[ctx.targets] >= tc.min_spikes]

    res = {}
    for label, cells in (("all 1,057 targets", ctx.targets),
                         (f"frozen ({int(frozen.size)} qualifying at k=0)", frozen)):
        s0 = score_sets(c0, d0, {"x": cells}, tc)["x"]
        s1 = score_sets(c1, d1, {"x": cells}, tc)["x"]
        assert np.array_equal(s0["members"], s1["members"]), "cell order differs between conditions"
        delta = s1["unit_scores"] - s0["unit_scores"]
        touched = n_added[s0["members"]] > 0
        a, b = delta[touched], delta[~touched]
        if a.size < 2 or b.size < 2:
            continue
        u, p_mw = mannwhitneyu(a, b, alternative="two-sided")
        rbc = 2.0 * u / (a.size * b.size) - 1.0          # rank-biserial correlation
        obs = float(np.median(a) - np.median(b))
        rng = np.random.default_rng(tc.seed + 7)
        pool = np.concatenate([a, b])
        # ONE permutation of the cell labels per replicate, split into the two groups (chunked so
        # the 10,000 x n_cells index matrix never has to exist all at once)
        perm = np.empty(n_perm)
        done = 0
        while done < n_perm:
            m = min(512, n_perm - done)
            order = np.argsort(rng.random((m, pool.size)), axis=1)
            perm[done:done + m] = (np.median(pool[order[:, :a.size]], axis=1)
                                   - np.median(pool[order[:, a.size:]], axis=1))
            done += m
        p_perm = float((1 + int((np.abs(perm) >= abs(obs)).sum())) / (1 + n_perm))
        ba = np.median(rng.choice(a, size=(n_boot, a.size)), axis=1)
        bb = np.median(rng.choice(b, size=(n_boot, b.size)), axis=1)
        lo, hi = np.percentile(ba - bb, [2.5, 97.5])
        print(f"\n  {label}: {a.size} touched vs {b.size} untouched")
        print(f"    touched   dMI  mean {a.mean():+.5f}  median {np.median(a):+.5f}  "
              f"IQR [{np.percentile(a, 25):+.5f}, {np.percentile(a, 75):+.5f}]")
        print(f"    untouched dMI  mean {b.mean():+.5f}  median {np.median(b):+.5f}  "
              f"IQR [{np.percentile(b, 25):+.5f}, {np.percentile(b, 75):+.5f}]")
        print(f"    median difference {obs:+.5f}  95% CI [{lo:+.5f}, {hi:+.5f}]  "
              f"(CI excludes zero: {bool(lo > 0 or hi < 0)})")
        print(f"    Mann-Whitney U = {u:.0f}, p = {p_mw:.4f}; rank-biserial r = {rbc:+.4f}; "
              f"label permutation ({n_perm:,}) p = {p_perm:.4f}")
        verdict = ("a local component IS detectable on top of the global movement"
                   if (lo > 0 or hi < 0) and p_perm < 0.05 else
                   "NO detectable local component: the touched cells' change distribution is not "
                   "distinguishable from the untouched cells'")
        print(f"    -> {verdict}")
        res[label] = {"n_touched": int(a.size), "n_untouched": int(b.size),
                      "touched_mean": float(a.mean()), "touched_median": float(np.median(a)),
                      "untouched_mean": float(b.mean()), "untouched_median": float(np.median(b)),
                      "median_difference": obs, "ci95": [float(lo), float(hi)],
                      "mannwhitney_u": float(u), "p_mannwhitney": float(p_mw),
                      "rank_biserial_r": float(rbc), "p_permutation": p_perm,
                      "n_permutations": n_perm, "verdict": verdict}
    out.setdefault("percell", {})[f"{policy}_k{k}_seed{seed}"] = res
    return res


DOSE_BINS = ((0, 0, "0"), (1, 1, "1"), (2, 2, "2"), (3, 4, "3-4"), (5, None, "5+"))


def bin_of(n):
    for lo, hi, label in DOSE_BINS:
        if n >= lo and (hi is None or n <= hi):
            return label
    raise ValueError(n)


def dose_response(ctx, conditions, tc, out):
    """Pool every hop-2 descending cell from every condition and arm, bin it by how many added
    edges IT received, and look at the score as a function of that dose.

    Convergence predicts a monotone rise across bins, holding within each condition as well as
    pooled. Raw MI is reported as asked; because each condition has its own null level, the
    within-condition contrast (a cell's MI minus the mean MI of the 0-edge cells of the SAME
    simulation) is reported beside it - that contrast is immune to condition-level drift and is
    what actually separates convergence from a global shift."""
    header("DOSE-RESPONSE - every target cell of every condition, binned by edges received")
    c0, d0 = load_sim(ctx, 0)
    total0 = c0.sum(axis=0, dtype=np.int64)
    frozen = ctx.targets[total0[ctx.targets] >= tc.min_spikes]
    frozen_mask = np.zeros(ctx.dist.size, dtype=bool)
    frozen_mask[frozen] = True

    rows = []
    for (k, policy, seed, frac) in conditions:
        counts, drive = load_sim(ctx, k, policy, seed, frac)
        n_added = np.zeros(ctx.dist.size, dtype=np.int64)
        if k > 0:
            post, _, _ = added_edges(ctx, k, policy, seed, frac)
            np.add.at(n_added, post, 1)
        total = counts.sum(axis=0, dtype=np.int64)
        variable = ctx.targets[total[ctx.targets] >= tc.min_spikes]
        s = score_sets(counts, drive, {"all": variable}, tc)["all"]
        mem, sc = s["members"], s["unit_scores"]
        base = sc[n_added[mem] == 0]
        base_mean = float(base.mean()) if base.size else float("nan")
        arm = "unmodified" if k == 0 else (policy if policy == "spread" else f"{policy} f{frac:g}")
        for cell, score in zip(mem, sc):
            rows.append({"cid": condition_id(k, policy, seed, frac), "k": k, "arm": arm,
                         "cell": int(cell), "n_added": int(n_added[cell]), "mi": float(score),
                         "mi_vs_own_zero_bin": float(score - base_mean),
                         "frozen": bool(frozen_mask[cell])})
    out_rows = rows

    def table(sel, title):
        """The within-condition contrast only exists for a condition that HAS 0-edge cells: at
        k = 10,000 and k = 100,000 every target received edges, so those cells contribute to the
        raw columns and are excluded (n/a) from the contrast columns rather than silently poisoning
        them with NaN."""
        n_contrast = sum(1 for r in sel if not np.isnan(r["mi_vs_own_zero_bin"]))
        print(f"\n  {title}"
              + ("" if n_contrast == len(sel)
                 else f"   [{len(sel) - n_contrast} of {len(sel)} cells come from conditions with no "
                      f"0-edge cells: no within-condition contrast for them]"))
        print(f"    {'dose':>5} {'cells':>6} {'mean MI':>9} {'median MI':>10} {'n':>5} "
              f"{'mean vs own 0-bin':>18} {'median vs own 0-bin':>20}")
        res = {}
        for lo, hi, label in DOSE_BINS:
            g = [r for r in sel if r["n_added"] >= lo and (hi is None or r["n_added"] <= hi)]
            if not g:
                continue
            mi = np.array([r["mi"] for r in g])
            dv = np.array([r["mi_vs_own_zero_bin"] for r in g])
            ok = dv[~np.isnan(dv)]
            mstr = f"{ok.mean():>18.5f}" if ok.size else f"{'n/a':>18}"
            dstr = f"{np.median(ok):>20.5f}" if ok.size else f"{'n/a':>20}"
            print(f"    {label:>5} {len(g):>6,} {mi.mean():>9.5f} {np.median(mi):>10.5f} "
                  f"{ok.size:>5,} {mstr} {dstr}")
            res[label] = {"cells": len(g), "mean_mi": float(mi.mean()),
                          "median_mi": float(np.median(mi)), "n_with_contrast": int(ok.size),
                          "mean_vs_zero": float(ok.mean()) if ok.size else None,
                          "median_vs_zero": float(np.median(ok)) if ok.size else None}
        # rank correlation between dose and score, within this selection
        doses = np.array([r["n_added"] for r in sel], dtype=np.float64)
        mis = np.array([r["mi"] for r in sel], dtype=np.float64)
        rho = spearman(doses, mis) if np.unique(doses).size > 1 else float("nan")
        print(f"    Spearman rho(edges received, MI) = {rho:.4f} over {len(sel):,} cells"
              if np.isfinite(rho) else "    Spearman rho: n/a (every cell received the same dose)")
        res["spearman_rho"] = None if not np.isfinite(rho) else float(rho)
        return res

    pooled = table(rows, "POOLED over every condition and arm (variable unit set)")
    pooled_frozen = table([r for r in rows if r["frozen"]],
                          "POOLED, frozen unit set (only cells qualifying at k = 0)")
    per_cond = {}
    for cid in dict.fromkeys(r["cid"] for r in rows):
        sel = [r for r in rows if r["cid"] == cid]
        if len({r["n_added"] for r in sel}) > 1:
            per_cond[cid] = table(sel, f"within {cid}")
    mono = None
    labels = [l for _, _, l in DOSE_BINS if l in pooled and pooled[l]["mean_vs_zero"] is not None]
    if len(labels) > 1:
        vals = [pooled[l]["mean_vs_zero"] for l in labels]
        mono = all(b >= a for a, b in zip(vals, vals[1:]))
        print(f"\n  pooled mean-vs-own-0-bin across the doses that have a contrast {labels}: "
              + ", ".join(f"{v:+.5f}" for v in vals)
              + f"  -> monotone rise: {mono}")
    raw = [l for _, _, l in DOSE_BINS if l in pooled]
    print(f"  pooled RAW mean MI across doses {raw}: "
          + ", ".join(f"{pooled[l]['mean_mi']:.5f}" for l in raw)
          + f"  -> monotone rise: {all(pooled[b]['mean_mi'] >= pooled[a]['mean_mi'] for a, b in zip(raw, raw[1:]))}"
          + "  (raw pooling mixes conditions with different null levels; the contrast column is the "
            "one that isolates dose)")
    out["dose_response"] = {"pooled": pooled, "pooled_frozen": pooled_frozen,
                            "per_condition": per_cond, "monotone_pooled": mono,
                            "bins": [l for _, _, l in DOSE_BINS]}
    return out_rows


def regime_table(ctx, out):
    """rate_hz, active_frac, rate_per_active_hz, sensory_rate_hz, motor_rate_hz for k = 0 and every
    k, adjacent, so a step in the MI can be compared against a step in the regime."""
    header("REGIME METRICS, k = 0 AND EVERY k SIDE BY SIDE")
    store = json.loads((CACHE / ctx.bc.results_json).read_text())
    conds = sorted(store["conditions"].values(), key=lambda c: (c["k"], c["policy"]))
    print(f"  {'k':>8} {'policy':>13} {'seed':>4} | {'rate_hz':>8} {'active_frac':>11} "
          f"{'per_active_hz':>13} {'sensory_hz':>10} {'motor_hz':>9} | {'vs k=0':>28}")
    base = next((c for c in conds if c["k"] == 0), None)
    rows = {}
    for c in conds:
        m = c["regime"]
        d = ""
        if base is not None and c["k"] != 0:
            b = base["regime"]
            d = (f"rate {m['rate_hz'] / b['rate_hz'] - 1:+.1%}, act "
                 f"{m['active_frac'] / b['active_frac'] - 1:+.1%}, motor "
                 f"{m['motor_rate_hz'] / b['motor_rate_hz'] - 1:+.1%}")
        print(f"  {c['k']:>8,} {c['policy']:>13} {str(c['seed']):>4} | {m['rate_hz']:>8.4g} "
              f"{m['active_frac']:>10.2%} {m['rate_per_active_hz']:>13.4g} "
              f"{m['sensory_rate_hz']:>10.4g} {m['motor_rate_hz']:>9.4g} | {d:>28}")
        rows[c["cid"]] = m
    out["regime_side_by_side"] = rows
    return rows


def run_to_run_context(ctx, out):
    """The run-to-run distribution of z, and every sweep claim read against it."""
    path = CACHE / "run_to_run.json"
    if not path.exists():
        print("  cache/run_to_run.json not present; run experiments.run_to_run first")
        return None
    rtr = json.loads(path.read_text())
    header(f"THE RUN-TO-RUN NULL ({len(rtr['seeds'])} LIF noise seeds, identical graph and drive)")
    print(f"  {'layer':<18} {'min':>7} {'max':>7} {'mean':>7} {'sd':>7} {'range':>7}   per-seed z")
    for name, sm in rtr["summary"].items():
        print(f"  {name:<18} {sm['min']:>7.2f} {sm['max']:>7.2f} {sm['mean']:>7.2f} {sm['sd']:>7.2f} "
              f"{sm['range']:>7.2f}   " + " ".join(f"{v:6.2f}" for v in sm["z"]))
    out["run_to_run"] = rtr["summary"]
    out["run_to_run_comparison"] = rtr.get("branch_sweep_comparison")
    return rtr


def displaced_arm(ctx, k, seeds, tc, out, rtr=None):
    """The placement control: identical k, sign split and weight multiset, placed between hop-3
    non-motor neurons instead of into the hop-1 -> hop-2-descending block."""
    header(f"PLACEMENT CONTROL - k = {k} displaced edges vs k = {k} in the real block")
    store = json.loads((CACHE / ctx.bc.results_json).read_text())
    rows = {"spread": [], "displaced": []}
    for cid, c in store["conditions"].items():
        if c["k"] == k and c["policy"] in rows and c.get("sim_seed") is None:
            rows[c["policy"]].append(c)
    base = store["conditions"].get("k0_none_fnone_seednone")
    if not rows["displaced"]:
        print("  no displaced conditions in the store yet")
        return None
    print(f"  {'arm':>11} {'seed':>4} {'units':>6} {'MI':>9} {'null mean':>9} {'z':>7} {'crossed':>8} "
          f"{'rate_hz':>8} {'motor_hz':>9}")
    if base:
        r, m = base["readout"], base["regime"]
        print(f"  {'k=0':>11} {'-':>4} {r['n_units']:>6} {r['mi']:>9.5f} {r['null_mean']:>9.5f} "
              f"{r['z']:>7.2f} {str(r['crossed']):>8} {m['rate_hz']:>8.4g} {m['motor_rate_hz']:>9.4g}")
    summary = {}
    for arm in ("spread", "displaced"):
        zs = []
        for c in sorted(rows[arm], key=lambda c: c["seed"]):
            r, m = c["readout"], c["regime"]
            zs.append(r["z"])
            print(f"  {arm:>11} {c['seed']:>4} {r['n_units']:>6} {r['mi']:>9.5f} "
                  f"{r['null_mean']:>9.5f} {r['z']:>7.2f} {str(r['crossed']):>8} "
                  f"{m['rate_hz']:>8.4g} {m['motor_rate_hz']:>9.4g}")
        if zs:
            summary[arm] = {"z": zs, "mean": float(np.mean(zs)),
                            "sd": float(np.std(zs, ddof=1)) if len(zs) > 1 else 0.0,
                            "min": float(np.min(zs)), "max": float(np.max(zs)), "n": len(zs)}
    verdict = None
    if "displaced" in summary and rtr is not None:
        null_z = np.array(rtr["summary"]["motor at hop 2"]["z"])
        d = summary["displaced"]
        pct = float((null_z < d["mean"]).mean() * 100)
        inside = d["mean"] <= null_z.max()
        real_mean = summary.get("spread", {}).get("mean")
        print(f"\n  displaced mean z {d['mean']:.2f} (range {d['min']:.2f}..{d['max']:.2f}) sits at "
              f"the {pct:.0f}th percentile of the run-to-run distribution "
              f"({null_z.min():.2f}..{null_z.max():.2f}); "
              + ("INSIDE it." if inside else "OUTSIDE it."))
        if real_mean is not None:
            near_real = abs(d["mean"] - real_mean) < abs(real_mean - float(null_z.mean())) / 2
            verdict = ("displaced z is comparable to the real arm's: a perturbation of this size "
                       "moves the layer wherever it is placed, so the ladder measures perturbation "
                       "MAGNITUDE, not wiring into this block"
                       if near_real else
                       "displaced z is well below the real arm's: PLACEMENT matters, and the k = 100 "
                       "effect is a property of this block even though it is not carried by the "
                       "cells that received the edges")
            print(f"  real-arm mean z {real_mean:.2f} vs displaced {d['mean']:.2f} -> {verdict}")
    out["displaced"] = {"summary": summary, "verdict": verdict}
    return summary


def resolution(ctx, out, rtr, layer="motor at hop 2"):
    """What resolution does this design actually have, given the measured run-to-run variance?

    Two-sample comparison of layer means between two conditions, each run n times, 5% two-sided,
    80% power:  n per condition = 2 * (sd * (1.96 + 0.8416) / delta)^2.
    The duration figure assumes the seed-to-seed variance is dominated by ESTIMATOR noise, which
    falls as 1/T. Any part of it that comes from genuine slow network dynamics does NOT fall that
    way, so the duration number is a LOWER BOUND on what would be needed, not a promise."""
    if rtr is None:
        return None
    header("RESOLUTION OF THIS DESIGN AT THE MEASURED RUN-TO-RUN VARIANCE")
    sm = rtr["summary"][layer]
    mis = np.array(sm["mi"], dtype=np.float64)
    sd_mi, sd_z = float(mis.std(ddof=1)), float(sm["sd"])
    store = json.loads((CACHE / ctx.bc.results_json).read_text())
    base = store["conditions"].get("k0_none_fnone_seednone")
    print(f"  {layer}: over {len(sm['z'])} noise seeds the layer-mean MI has sd {sd_mi:.5f} "
          f"(mean {mis.mean():.5f}) and z has sd {sd_z:.2f}, with NO change to the graph.")
    rows = {}
    C = (1.959964 + 0.841621) ** 2
    for cid, c in sorted(store["conditions"].items(), key=lambda kv: kv[1]["k"]):
        if c["k"] == 0 or c.get("sim_seed") is not None or base is None:
            continue
        delta = c["readout"]["mi"] - base["readout"]["mi"]
        if delta == 0:
            continue
        n = 2 * C * (sd_mi / abs(delta)) ** 2
        t_mult = (sd_mi / (abs(delta) / (2 * 2.801585))) ** 2
        rows[c["cid"]] = {"delta_mi_vs_k0": delta, "runs_per_condition_for_80pct_power": n,
                          "duration_multiplier_lower_bound": t_mult}
        print(f"    {c['cid']:<34} dMI vs k=0 {delta:+.5f} -> {n:>8.1f} runs per condition for 80% "
              f"power, or >= {t_mult * ctx.bc.duration_ms / 1e3:>8.0f} s per run "
              f"({t_mult:.1f}x the current {ctx.bc.duration_ms / 1e3:g} s) if the spread were pure "
              f"estimator noise")
    out["resolution"] = {"layer": layer, "sd_mi": sd_mi, "sd_z": sd_z, "n_seeds": len(sm["z"]),
                         "per_condition": rows,
                         "note": "runs-per-condition assumes 5% two-sided, 80% power; the duration "
                                 "multiplier additionally assumes the seed-to-seed spread is pure "
                                 "estimator noise falling as 1/T and is therefore a lower bound"}
    return rows


# =============================================================================
# main
# =============================================================================

def main(k=100, seed=1, shuffles=200):
    bc = BranchConfig()
    header("SETUP")
    ctx = Context(bc)
    tc = replace(TransmissionConfig(), duration_ms=bc.duration_ms)
    tc_big = replace(tc, n_shuffles=shuffles)
    out = {"k": k, "seed": seed, "n_shuffles_small": tc.n_shuffles, "n_shuffles_big": shuffles}
    store = json.loads((CACHE / bc.results_json).read_text()) if (CACHE / bc.results_json).exists() else {}
    available = sorted({c["k"] for c in store.get("conditions", {}).values()
                        if c["policy"] in ("none", "spread") and (c["seed"] in (None, seed))})
    print(f"cached spread/seed-{seed} conditions available: {available}")

    conditions = [(kk, "none" if kk == 0 else "spread", None if kk == 0 else seed, None)
                  for kk in available]
    check1(ctx, k, seed, tc, out)
    percell_contrast(ctx, k, seed, tc, out)
    check2(ctx, available, tc, out)
    check3(ctx, available, tc, tc_big, out)
    dose_response(ctx, conditions, tc, out)
    regime_table(ctx, out)
    rtr = run_to_run_context(ctx, out)
    displaced_arm(ctx, k, BranchConfig().seeds, tc, out, rtr)
    resolution(ctx, out, rtr)
    header("WHY THE NULL CANNOT MANUFACTURE A CROSSING BY ITSELF")
    print("  The joint circular-shift null shifts only the INPUT signals; every unit keeps its own\n"
          "  spike train intact, and all input signals are shifted by the same offset, so the\n"
          "  cross-signal correlation is kept too. A plug-in MI estimate is biased upward by an\n"
          "  amount that depends on the response distribution - firing rate, burstiness, how many\n"
          "  of the mi_response_levels a unit actually occupies - and because the real and the null\n"
          "  score the SAME spike trains, that bias is matched on both sides and cancels in the\n"
          "  comparison. A condition that merely fires faster therefore raises its real MI and its\n"
          "  own null by the same amount and does not cross. This is a genuine strength of the\n"
          "  design and it is why the regime table above is a sanity check rather than a rival\n"
          "  explanation - but it is only matched WITHIN a condition, which is exactly why the\n"
          "  dose-response contrast is taken within each simulation.")

    header("SUMMARY")
    v = out["check1"]["verdict"].get("frozen") or out["check1"]["verdict"].get("variable")
    c3 = out["check3"]["by_k"].get(str(k), {})
    lines = []
    if v:
        local = v["touched_delta"] > 0 and v["difference_in_differences"] > 0
        lines.append(f"CHECK 1: touched {v['touched_delta']:+.5f} MI, untouched "
                     f"{v['untouched_delta']:+.5f} MI, DiD {v['difference_in_differences']:+.5f} -> "
                     + ("the change is LOCAL to the touched cells."
                        if local and abs(v["untouched_delta"]) < abs(v["touched_delta"]) / 2
                        else "the untouched cells moved comparably: the layer moved GLOBALLY and the "
                             "headline k does not mean what it appears to mean."))
    f2 = out["check2"]["by_k"].get(str(k), {})
    if f2.get("frozen") and out["check2"]["by_k"].get("0", {}).get("frozen"):
        a = out["check2"]["by_k"]["0"]["frozen"]
        b = f2["frozen"]
        lines.append(f"CHECK 2: on the frozen set, MI {a['mean_best']:.5f} -> {b['mean_best']:.5f}, "
                     f"z {a['z']:.2f} -> {b['z']:.2f}, p {a['p_empirical']:.4f} -> {b['p_empirical']:.4f} "
                     f"({f2['n_newly']} units newly qualified at k = {k} and are excluded from it).")
    for name in ("frozen", "variable"):
        st = c3.get(name)
        if st:
            lines.append(f"CHECK 3: k = {k} {name} set over {shuffles} shuffles: MI {st['mean_best']:.5f}, "
                         f"null {st['null_mean']:.5f} +- {st['null_sd']:.5f}, z {st['z']:.2f}, "
                         f"{st['n_null_ge_real']} of {shuffles} shuffles >= real, exact p "
                         f"{st['p_empirical']:.4f} (Bonferroni threshold "
                         f"{out['check3']['bonferroni_p']:.5f}).")
    for ln in lines:
        print("  " + ln)
    out["summary"] = lines
    RESULTS.write_text(json.dumps(strip_arrays(out), indent=2, default=str))
    print(f"\nwrote {RESULTS.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=100, help="the crossing k to interrogate")
    ap.add_argument("--seed", type=int, default=BranchConfig().stage1_seed)
    ap.add_argument("--shuffles", type=int, default=200)
    a = ap.parse_args()
    main(a.k, a.seed, a.shuffles)
