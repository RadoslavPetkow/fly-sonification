"""The run-to-run null: one graph, one drive, many noise realizations.

    .venv/bin/python -m experiments.run_to_run [--seeds 8] [--adjacency PATH] [--force]

Every "above null" claim in this project so far has been made against the circular-shift null of
experiments/transmission.py. That null shifts the INPUT SIGNAL inside a single run; it answers "is
this layer's score higher than it would be if the signal were misaligned with these same spike
trains". It does NOT answer "how much would this score move if I simply ran the model again", and
nothing in the project measured that until now.

This module measures it. The graph, the drive, the duration and every parameter are held fixed; only
SimConfig.seed - the LIF membrane-noise realization - changes. The resulting spread of each layer's
z is the reference distribution against which any difference between two conditions has to be read,
including:

  * the published transmission result (hop-1 descending above null, hop-2 descending at chance),
    which rests on one run at noise seed 0;
  * every k in experiments/branch_sweep.py, which compares single runs of different graphs.

Seed 0 is measured alongside the others and flagged, because it is the run every other experiment in
this repo is built on. Results: cache/run_to_run.json, figures/run_to_run.png, runs/run_to_run.log.
"""
import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from config import (AudioConfig, CalibConfig, MidiConfig, Paths, RunToRunConfig, SimConfig,
                    TransmissionConfig)
from data.fetch_connectome import bfs_levels
from data.groups import load_neurons_and_indices, ordered_groups
from experiments.transmission import CALIBRATED_KEYS, analyze, build_layers, simulate
from sim.calibrate import CATEGORICAL, GRID, TEXT, TEXT_2, banner, header, save, style

CACHE = ROOT / Paths().cache

# Unambiguous short labels: "motor at hop 1".replace("motor at ", "") collides with the "hop 1" layer.
SHORT = {"hop 0 (sensory)": "sens", "hop 1": "hop1-all", "motor at hop 1": "DN-hop1",
         "motor at hop 2": "DN-hop2", "motor at hop 3": "DN-hop3", "motor (all)": "DN-all",
         "added-edge targets": "touched"}


def ci95(v):
    """95% confidence interval for the mean of a small sample (Student t, n-1 df)."""
    from scipy.stats import t as student
    v = np.asarray(v, dtype=np.float64)
    n = v.size
    if n < 2:
        return float("nan"), float("nan")
    h = student.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return float(v.mean() - h), float(v.mean() + h)


def sha12(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def variant_tag(normalization=None, w_scale=None):
    """Empty for the calibrated network, so existing caches stay valid; otherwise a suffix naming
    the normalisation and gain, so a raw run can never share a cache path with a sqrt_in run."""
    if normalization is None and w_scale is None:
        return ""
    return f"_{normalization or 'calib'}-w{w_scale:g}" if w_scale is not None else f"_{normalization}"


def results_path(rc, adj_sha, duration_ms, variant=""):
    d = CACHE / rc.subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"results_{adj_sha}_d{int(round(duration_ms))}{variant}.json"


def sim_path(rc, adj_sha, seed, variant=""):
    """Keyed by the matrix hash, the duration and the noise seed, so this module can be pointed at
    any matrix without the caches ever colliding."""
    d = CACHE / rc.subdir
    d.mkdir(parents=True, exist_ok=True)
    return d / f"sim_{adj_sha}_d{int(round(rc.duration_ms))}{variant}_ns{seed}.npz"


def adopt_existing(rc, adj_sha, seeds, sc, tc):
    """Re-use simulations already produced elsewhere in the repo under an IDENTICAL descriptor
    (experiments/branch_sweep.py's k = 0 conditions and its noise-seed replicates) instead of
    spending six minutes each to recompute them. The arrays are copied unchanged and their
    descriptor is rewritten to this module's; meta_migrated marks them so they are never mistaken
    for caches this module stamped itself. Every field that defines the computation - adjacency
    hash, duration, drive seed, noise seed, w_scale, g_inh, normalization - is checked first."""
    src = CACHE / "branches"
    if not src.exists():
        return []
    adopted = []
    for seed in seeds:
        dst = sim_path(rc, adj_sha, seed)
        if dst.exists():
            continue
        cand = src / (f"sim_k0_none_fnone_seednone_d{int(round(rc.duration_ms))}.npz" if seed == 0
                      else f"sim_k0_none_fnone_seednone_ns{seed}_d{int(round(rc.duration_ms))}.npz")
        if not cand.exists():
            continue
        d = np.load(cand)
        meta = json.loads(str(d["meta"]))
        checks = {"adjacency_sha12": adj_sha, "duration_ms": float(rc.duration_ms),
                  "w_scale": float(sc.w_scale), "g_inh": float(sc.g_inh),
                  "normalization": sc.normalization}
        bad = {f: (meta.get(f, "<missing>"), v) for f, v in checks.items()
               if (abs(float(meta.get(f, 0)) - v) > 1e-9 if isinstance(v, float)
                   else meta.get(f, "<missing>") != v)}
        cached_ns = meta.get("sim_seed", 0)
        if cached_ns != seed:
            bad["sim_seed"] = (cached_ns, seed)
        if bad:
            print(f"  NOT adopting {cand.name} for seed {seed}: {bad}")
            continue
        meta.update({"sim_seed": int(seed), "adjacency_sha12": adj_sha, "meta_migrated": True,
                     "drive_seed": int(tc.seed), "duration_ms": float(rc.duration_ms),
                     "adopted_from": str(cand.relative_to(ROOT))})
        np.savez_compressed(dst, counts=d["counts"], drive_binned=d["drive_binned"], z=d["z"],
                            meta=json.dumps(meta))
        adopted.append((cand.name, seed))
    if adopted:
        print(f"  adopted {len(adopted)} existing simulation(s) with an identical descriptor:")
        for name, seed in adopted:
            print(f"    seed {seed} <- cache/branches/{name}")
    return adopted


def run_seed(rc, sc, cc, tc, calib, groups, seed, adj, adj_sha, force, variant=""):
    path = sim_path(rc, adj_sha, seed, variant)
    want = {"sim_seed": int(seed), "adjacency_sha12": adj_sha, "duration_ms": float(rc.duration_ms),
            "w_scale": float(sc.w_scale), "g_inh": float(sc.g_inh),
            "normalization": sc.normalization, "drive_seed": int(tc.seed)}
    if path.exists() and not force:
        meta = json.loads(str(np.load(path)["meta"]))
        bad = {f: (meta.get(f, "<missing>"), v) for f, v in want.items()
               if (abs(float(meta.get(f, 0)) - v) > 1e-9 if isinstance(v, float)
                   else meta.get(f, "<missing>") != v)}
        if bad:
            print(f"  cache {path.name} does not describe seed {seed} ({bad}); re-running")
            force = True
    counts, drive, _, meta = simulate(replace(sc, seed=int(seed)), cc, tc, calib, groups, force,
                                      adjacency_path=adj, sim_cache=path, extra_meta=want)
    return counts, drive, meta


def main(n_seeds=None, adjacency=None, duration_ms=None, force=False, label=None,
         normalization=None, w_scale=None):
    rc = RunToRunConfig()
    if duration_ms is not None:
        rc = replace(rc, duration_ms=float(duration_ms))
    n_seeds = rc.n_seeds if n_seeds is None else n_seeds
    (ROOT / Paths().runs).mkdir(exist_ok=True)
    _adj = (Path(adjacency) if adjacency else CACHE / "adjacency.npz").resolve()
    if not _adj.exists():
        raise FileNotFoundError(_adj)
    _sha = sha12(_adj)
    _variant = variant_tag(normalization, w_scale)
    log = open(ROOT / Paths().runs
               / f"run_to_run_{_sha}_d{int(round(rc.duration_ms))}{_variant}.log", "a", buffering=1)

    class Tee:
        def write(self, s):
            sys.__stdout__.write(s)
            log.write(s)

        def flush(self):
            sys.__stdout__.flush()
            log.flush()

    sys.stdout = Tee()
    try:
        print(f"\n{'=' * 100}\nrun_to_run at {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'=' * 100}")
        header("SETUP")
        cc = CalibConfig()
        tc = replace(TransmissionConfig(), duration_ms=rc.duration_ms)
        tc_big = replace(tc, n_shuffles=rc.n_shuffles)
        calib = json.loads((CACHE / "calibration.json").read_text())
        sc = replace(SimConfig(), **{k: calib[k] for k in CALIBRATED_KEYS})
        if normalization is not None or w_scale is not None:
            sc = replace(sc, **({"normalization": normalization} if normalization else {}),
                         **({"w_scale": float(w_scale)} if w_scale is not None else {}))
            print(f"  NON-CALIBRATED VARIANT: normalization {sc.normalization!r}, w_scale "
                  f"{sc.w_scale:.6g} (calibrated: {calib['normalization']!r}, "
                  f"{calib['w_scale']:.6g}). This is a DIFFERENT OPERATING POINT - its numbers are "
                  f"not comparable with the calibrated ones, only with other runs of this variant. "
                  f"w_scale came from experiments/norm_regime.py, chosen on the regime gates alone "
                  f"before any MI was computed. The external drive is unchanged, so the input is "
                  f"identical.")
        adj, adj_sha = _adj, _sha
        neurons, indices = load_neurons_and_indices()
        sens = np.asarray(indices["sensory_idx"])
        motor = np.asarray(indices["motor_idx"])
        in_groups = ordered_groups(neurons, sens, AudioConfig().n_bins)
        motor_groups = ordered_groups(neurons, motor, MidiConfig().n_groups)
        # Layer membership ALWAYS comes from the unmodified matrix: adding edges can create
        # shortcuts that move neurons between hops, and conditions that do not score the same
        # neurons are not comparable.
        base_adj = CACHE / "adjacency.npz"
        W = sp.load_npz(base_adj).tocsr()
        dist = bfs_levels(W.T.tocsr(), sens)
        del W
        touched = None
        if adj.resolve() != base_adj.resolve():
            print(f"  layer membership taken from {base_adj.relative_to(ROOT)} "
                  f"(sha {sha12(base_adj)}), NOT from the analyzed matrix")
            # Each arm's OWN positive control: the cells whose in-degree the arm actually raised.
            # For the real arm those are the descending targets themselves; for a control arm they
            # are elsewhere, and if they do not respond, that arm is not testing what it should.
            W0 = sp.load_npz(base_adj).tocsr()
            W1 = sp.load_npz(adj).tocsr()
            touched = np.flatnonzero(np.diff(W1.indptr) != np.diff(W0.indptr))
            del W0, W1
            print(f"  this matrix raises the in-degree of {touched.size:,} neurons; they are "
                  f"reported as the layer 'added-edge targets' (this arm's own positive control)")
        layer_names = list(rc.layers) + (["added-edge targets"] if touched is not None else [])
        seeds = [0] + list(range(1, n_seeds + 1))
        print(f"matrix {adj.relative_to(ROOT)} sha256[:12] {adj_sha}; {len(seeds)} LIF noise seeds "
              f"{seeds} (seed 0 is the run every other experiment in this repo is built on)")
        print(f"held fixed: graph, drive (TransmissionConfig.seed {tc.seed}), duration "
              f"{rc.duration_ms / 1e3:g} s, " + ", ".join(f"{k}={calib[k]}" for k in CALIBRATED_KEYS))
        if not _variant:
            adopt_existing(rc, adj_sha, seeds, sc, tc)

        rows = {}
        for seed in seeds:
            header(f"NOISE SEED {seed}")
            counts, drive, meta = run_seed(rc, sc, cc, tc, calib, in_groups, seed, adj, adj_sha,
                                           force, _variant)
            total = counts.sum(axis=0, dtype=np.int64)
            T = counts.shape[0] * tc.bin_ms / 1e3
            rates = total / T
            layers, info = build_layers(counts, rates, total, sens, motor, dist, in_groups,
                                        motor_groups, tc, np.random.default_rng(tc.seed + 1),
                                        verbose=False)
            if touched is not None:
                idx = touched[total[touched] >= tc.min_spikes]
                if idx.size > tc.layer_sample:
                    idx = np.sort(np.random.default_rng(tc.seed + 3).choice(idx, tc.layer_sample,
                                                                           replace=False))
                layers["added-edge targets"] = counts[:, idx].T
            res, _ = analyze(counts, drive, layers, tc_big, np.random.default_rng(tc.seed + 2))
            row = {"total_spikes": int(total.sum()), "rate_hz": float(total.sum() / counts.shape[1] / T),
                   "active_frac": float((total > 0).mean()),
                   "motor_rate_hz": float(total[motor].mean() / T),
                   "sensory_rate_hz": float(total[sens].mean() / T), "layers": {}}
            for name in layer_names:
                s = res[name]["mi"]
                row["layers"][name] = {"n_units": res[name]["n_units"], "mi": s["mean_best"],
                                       "null_mean": s["null_mean"], "null_sd": s["null_sd"],
                                       "excess": s["mean_best"] - s["null_mean"],
                                       "z": s["z"], "p_empirical": s["p_empirical"],
                                       "n_null_ge_real": s["n_null_ge_real"],
                                       "n_units_above_own_null_max": s["n_units_above_own_null_max"],
                                       # excess normalised by the layer's OWN null mean, so layers
                                       # with different baseline MI are on one scale
                                       "excess_norm": ((s["mean_best"] - s["null_mean"]) / s["null_mean"]
                                                       if s["null_mean"] else float("nan")),
                                       # a per-CELL statistic: no layer mean, no unstable denominator
                                       "frac_units_above_null": (s["n_units_above_own_null_max"]
                                                                 / max(res[name]["n_units"], 1))}
            rows[seed] = row
            print(f"  {row['total_spikes']:,} spikes, rate {row['rate_hz']:.4g} Hz; excess (z): "
                  + ", ".join(f"{SHORT[n]} {row['layers'][n]['excess']:+.5f} "
                              + ("(n/a)" if row["layers"][n]["z"] is None
                                 else f"({row['layers'][n]['z']:.1f})")
                              for n in layer_names))

        # ------------------------------------------------------------- report
        header(f"RUN-TO-RUN SPREAD OVER {len(seeds)} NOISE SEEDS AT {rc.duration_ms / 1e3:g} s "
               f"(graph, drive and every parameter identical)")
        summary = {}
        for stat, fmt in (("mi", "{:8.5f}"), ("excess", "{:+8.5f}"),
                          ("excess_norm", "{:+8.4f}"), ("frac_units_above_null", "{:8.4f}"),
                          ("z", "{:8.2f}")):
            print(f"\n  {stat.upper()}"
                  + {"excess": "  (real - null mean: the primary quantity; z's denominator "
                                "is itself unstable)",
                     "excess_norm": "  (excess / the layer's own null mean: the three layers on one "
                                    "scale)",
                     "frac_units_above_null": "  (fraction of the layer's own units beating their "
                                              "own null max - a per-cell statistic with no layer "
                                              "mean and no unstable denominator)"}.get(stat, ""))
            print(f"  {'layer':<18} {'units':>6} {'min':>9} {'max':>9} {'mean':>9} {'sd':>9}   per seed")
            for name in layer_names:
                v = np.array([rows[s]["layers"][name][stat] for s in seeds], dtype=np.float64)
                n_units = int(np.median([rows[s]["layers"][name]["n_units"] for s in seeds]))
                d = summary.setdefault(name, {"seeds": seeds, "median_n_units": n_units})
                d[stat] = v.tolist()
                d[f"{stat}_min"], d[f"{stat}_max"] = float(v.min()), float(v.max())
                d[f"{stat}_mean"], d[f"{stat}_sd"] = float(v.mean()), float(v.std(ddof=1))
                lo, hi = ci95(v)
                d[f"{stat}_ci95"] = [lo, hi]
                print(f"  {name:<18} {n_units:>6,} " + " ".join(fmt.format(x) for x in
                      (v.min(), v.max(), v.mean(), v.std(ddof=1)))
                      + "   " + " ".join(fmt.format(x).strip().rjust(7) for x in v))
        print(f"\n  {'layer':<18} {'seeds called CROSSED at z >= ' + str(tc.null_z_min):>34} "
              f"{'seeds with exact p <= 0.05':>28}")
        for name in layer_names:
            zs = summary[name]["z"]
            ps = [rows[s]["layers"][name]["p_empirical"] for s in seeds]
            nz = sum(1 for v in zs if v >= tc.null_z_min)
            npv = sum(1 for v in ps if v <= 0.05)
            summary[name]["n_crossed_z"] = nz
            summary[name]["n_p_le_05"] = npv
            summary[name]["p_empirical"] = ps
            print(f"  {name:<18} {f'{nz}/{len(seeds)}':>34} {f'{npv}/{len(seeds)}':>28}")

        header("WHAT THIS MEANS FOR CLAIMS MADE AGAINST THE CIRCULAR-SHIFT NULL")
        lines = []
        for name in layer_names:
            s = summary[name]
            crosses = s["n_crossed_z"]
            lines.append(
                f"{name}: with NO change to the graph, excess spans {s['excess_min']:+.5f}.."
                f"{s['excess_max']:+.5f} (mean {s['excess_mean']:+.5f}, 95% CI "
                f"[{s['excess_ci95'][0]:+.5f}, {s['excess_ci95'][1]:+.5f}]) and z spans "
                f"{s['z_min']:.2f}..{s['z_max']:.2f}; {crosses}/{len(seeds)} seeds would be called "
                f"'above null' at z >= {tc.null_z_min:g}, {s['n_p_le_05']}/{len(seeds)} reach exact "
                f"p <= 0.05 over {rc.n_shuffles} shuffles.")
        # the branch sweep's k = 100 claim, read against this distribution
        bs = CACHE / "branch_sweep.json"
        comparison = None
        from config import BranchConfig as _BC
        if bs.exists() and abs(rc.duration_ms - _BC().duration_ms) > 1e-6:
            print(f"  (the branch sweep ran at {_BC().duration_ms / 1e3:g} s; its z values are NOT "
                  f"comparable to this {rc.duration_ms / 1e3:g} s distribution and are not shown "
                  f"here - see the {_BC().duration_ms / 1e3:g} s results file)")
        elif bs.exists():
            store = json.loads(bs.read_text())
            ref = {c["k"]: c["readout"]["z"] for c in store.get("conditions", {}).values()
                   if c.get("sim_seed") is None and c["policy"] in ("none", "spread")}
            null_z = np.array(summary["motor at hop 2"]["z"])
            comparison = {}
            for k, z in sorted(ref.items()):
                if z is None:
                    continue
                pct = float((null_z < z).mean() * 100)
                p = float((1 + (null_z >= z).sum()) / (1 + null_z.size))
                comparison[str(k)] = {"z": z, "percentile_in_run_to_run": pct, "p_vs_run_to_run": p,
                                      "inside_spread": bool(z <= null_z.max())}
                lines.append(f"branch sweep k = {k:,}: hop-2 z = {z:.2f} sits at the "
                             f"{pct:.0f}th percentile of the run-to-run distribution "
                             f"(p = {p:.3f} against it); "
                             + ("INSIDE the spread produced by noise alone."
                                if z <= null_z.max() else "OUTSIDE it."))
        for ln in lines:
            print("  " + ln)

        out = {"adjacency": str(adj.relative_to(ROOT)), "adjacency_sha12": adj_sha, "seeds": seeds,
               "duration_ms": rc.duration_ms, "drive_seed": tc.seed, "per_seed": rows,
               "summary": summary, "branch_sweep_comparison": comparison, "conclusions": lines,
               "calibration": {k: calib[k] for k in CALIBRATED_KEYS},
               "run_to_run_config": asdict(rc), "null_z_min": tc.null_z_min}
        out["label"] = label
        out["n_shuffles"] = rc.n_shuffles
        out["normalization"] = sc.normalization
        out["w_scale"] = sc.w_scale
        rp = results_path(rc, adj_sha, rc.duration_ms, _variant)
        rp.write_text(json.dumps(out, indent=2, default=str))
        print(f"\nwrote {rp.relative_to(ROOT)}")
        figure(rc, summary, seeds, tc, comparison, adj_sha)
        return out
    finally:
        sys.stdout = sys.__stdout__
        log.close()


def figure(rc, summary, seeds, tc, comparison, adj_sha):
    fig, ax = plt.subplots(figsize=(9, 5))
    names = list(rc.layers)
    for i, name in enumerate(names):
        zs = summary[name]["z"]
        ax.scatter([i] * len(zs), zs, s=34, color=CATEGORICAL[0], alpha=0.75, zorder=3,
                   label="one noise seed (graph unchanged)" if i == 0 else None)
        ax.plot([i - 0.25, i + 0.25], [np.mean(zs)] * 2, color=TEXT_2, linewidth=1.6,
                label="mean" if i == 0 else None)
    if comparison and "100" in comparison:
        j = names.index("motor at hop 2")
        ax.scatter([j], [comparison["100"]["z"]], marker="*", s=260, color=CATEGORICAL[1], zorder=4,
                   label="branch sweep k = 100 (one run, modified graph)")
    ax.axhline(tc.null_z_min, color=TEXT_2, linestyle="--", linewidth=1.1,
               label=f"'above null' threshold z = {tc.null_z_min:g}")
    ax.axhline(0, color=GRID, linewidth=1)
    ax.set_xticks(range(len(names)), [n.replace("motor at ", "motor\n") for n in names], fontsize=8)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.6, axis="y")
    style(ax, f"Run-to-run spread of the transmission z over {len(seeds)} LIF noise seeds "
              f"({rc.duration_ms / 1e3:g} s, identical graph and drive)",
          "layer", "z of layer mean MI vs its own circular-shift null")
    save(fig, f"run_to_run_{adj_sha}_d{int(round(rc.duration_ms))}.png")


DN_LAYERS = ("motor at hop 1", "motor at hop 2", "motor at hop 3")


def upgrade_summary(d):
    """Rebuild a results file's summary from its per_seed rows.

    The per-seed rows always carry everything; the summary is derived. This lets result files
    written before excess_norm and frac_units_above_null existed gain them without re-simulating or
    re-analysing anything. json turns the integer seed keys into strings, so both are accepted."""
    per = d["per_seed"]
    seeds = d["seeds"]
    def row(sd):
        return per[str(sd)] if str(sd) in per else per[sd]
    for name, sm in d["summary"].items():
        for stat in ("mi", "excess", "excess_norm", "frac_units_above_null", "z"):
            vals = []
            for sd in seeds:
                layer = row(sd)["layers"][name]
                if stat in layer:
                    vals.append(layer[stat])
                elif stat == "excess_norm":
                    vals.append(layer["excess"] / layer["null_mean"] if layer["null_mean"] else float("nan"))
                elif stat == "frac_units_above_null":
                    vals.append(layer["n_units_above_own_null_max"] / max(layer["n_units"], 1))
            v = np.array(vals, dtype=np.float64)
            lo, hi = ci95(v)
            sm[stat] = v.tolist()
            sm[f"{stat}_min"], sm[f"{stat}_max"] = float(v.min()), float(v.max())
            sm[f"{stat}_mean"], sm[f"{stat}_sd"] = float(v.mean()), float(v.std(ddof=1))
            sm[f"{stat}_ci95"] = [lo, hi]
    return d


def dprime(a, b):
    """Separation between two layers in units of their pooled run-to-run sd.

    Zero pooled spread with different means is perfect separation (inf), not an error; zero spread
    AND equal means is undefined (nan). The two must not be confused when picking a statistic."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    if pooled > 0:
        return float((a.mean() - b.mean()) / pooled)
    return float("inf") * np.sign(a.mean() - b.mean()) if a.mean() != b.mean() else float("nan")


def profile(paths, upgrade=True):
    """The three descending layers on one scale, and which statistic is actually stable.

    Addition 1: excess normalised by each layer's own null mean, so "does hop 2 cross a threshold"
    is replaced by "where does hop 2 sit between a layer that clearly transmits (hop 1) and one that
    clearly does not (hop 3)".

    Addition 2: the fraction of a layer's own units beating their own null max. It has no layer mean
    and no unstable denominator, so it may be far steadier across noise seeds than z. Whether it is
    is measured here, not assumed: the run-to-run sd of both is reported side by side, along with
    how far apart hop 1 and hop 3 are in units of each statistic's own run-to-run sd. The statistic
    that separates them by more, per unit of its own instability, is the one the published claim
    should rest on."""
    out = {}
    for path in paths:
        d = json.loads(Path(path).read_text())
        if upgrade:
            d = upgrade_summary(d)
            Path(path).write_text(json.dumps(d, indent=2, default=str))
        dur, seeds = d["duration_ms"], d["seeds"]
        label = d.get("label") or Path(path).stem
        header(f"LAYER PROFILE - {label}, {dur / 1e3:g} s, {len(seeds)} noise seeds")

        print("  per-seed EXCESS / null mean (the three descending layers on one scale)")
        print(f"  {'seed':>5} " + " ".join(f"{SHORT[n]:>12}" for n in DN_LAYERS)
              + f"   {'hop2 position':>14}")
        pos = []
        for sd in seeds:
            r = (d["per_seed"][str(sd)] if str(sd) in d["per_seed"] else d["per_seed"][sd])["layers"]
            vals = [r[n]["excess"] / r[n]["null_mean"] for n in DN_LAYERS]
            p = (vals[1] - vals[2]) / (vals[0] - vals[2]) if vals[0] != vals[2] else float("nan")
            pos.append(p)
            print(f"  {sd:>5} " + " ".join(f"{v:>+12.4f}" for v in vals) + f"   {p:>14.3f}")
        pos = np.array(pos, dtype=np.float64)
        lo, hi = ci95(pos)
        print(f"\n  hop-2 position on the hop3 -> hop1 axis (0 = indistinguishable from the layer "
              f"that does not transmit, 1 = as strong as the layer that does):")
        print(f"    mean {pos.mean():.3f}, sd {pos.std(ddof=1):.3f}, range {pos.min():.3f}.."
              f"{pos.max():.3f}, 95% CI [{lo:.3f}, {hi:.3f}]")

        print(f"\n  STABILITY of each statistic across the {len(seeds)} noise seeds "
              f"(graph and drive identical)")
        print(f"  {'statistic':<26} " + " ".join(f"{SHORT[n]:>22}" for n in DN_LAYERS)
              + f"   {'hop1 vs hop3 d-prime':>21}")
        stats = {}
        for stat in ("z", "excess", "excess_norm", "frac_units_above_null"):
            cells, vals = [], {}
            for n in DN_LAYERS:
                v = np.array(d["summary"][n][stat], dtype=np.float64)
                vals[n] = v
                cv = abs(v.std(ddof=1) / v.mean()) if v.mean() else float("nan")
                cells.append(f"{v.mean():>9.4f}+-{v.std(ddof=1):<7.4f} ({cv:>4.0%})")
            dp = dprime(vals[DN_LAYERS[0]], vals[DN_LAYERS[2]])
            stats[stat] = {"d_prime_hop1_vs_hop3": dp,
                           **{n: {"mean": float(vals[n].mean()), "sd": float(vals[n].std(ddof=1))}
                              for n in DN_LAYERS}}
            print(f"  {stat:<26} " + " ".join(f"{c:>22}" for c in cells) + f"   {dp:>21.1f}")
        print("    (mean +- run-to-run sd, with the coefficient of variation in brackets; d-prime is "
              "the hop1/hop3 gap\n     divided by their pooled run-to-run sd - higher means the "
              "statistic separates transmitting from\n     non-transmitting layers more reliably "
              "against its own instability)")
        usable = {k: v for k, v in stats.items()
                  if not np.isnan(v["d_prime_hop1_vs_hop3"])}
        if usable:
            best = max(usable, key=lambda k: abs(usable[k]["d_prime_hop1_vs_hop3"]))
            rec = (f"RECOMMENDED STATISTIC for the published claim: {best!r} "
                   f"(d-prime {stats[best]['d_prime_hop1_vs_hop3']:.1f} vs "
                   f"{stats['z']['d_prime_hop1_vs_hop3']:.1f} for z)")
            if len(usable) < len(stats):
                rec += (" [" + ", ".join(sorted(set(stats) - set(usable))
                                         ) + ": d-prime undefined, zero spread and equal means]")
        else:
            best, rec = None, "no statistic had a defined d-prime"
        print(f"\n  {rec}")
        out[label] = {"path": str(path), "duration_ms": dur, "seeds": seeds,
                      "hop2_position": {"per_seed": pos.tolist(), "mean": float(pos.mean()),
                                        "sd": float(pos.std(ddof=1)), "ci95": [lo, hi]},
                      "stability": stats, "recommended": best, "recommendation": rec}
        profile_figure(d, label, pos)
    # merge rather than overwrite: profiling one condition must not erase the others' record
    dest = CACHE / RunToRunConfig().subdir / "layer_profile.json"
    merged = json.loads(dest.read_text()) if dest.exists() else {}
    merged.update(out)
    dest.write_text(json.dumps(merged, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(ROOT)}")
    return out


def profile_figure(d, label, pos):
    """Three point-clouds with their run-to-run spreads: where hop 2 sits between hop 1 and hop 3."""
    seeds, dur = d["seeds"], d["duration_ms"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, stat, ylab in ((axes[0], "excess_norm", "excess / the layer's own null mean"),
                           (axes[1], "frac_units_above_null",
                            "fraction of units above their own null max")):
        for i, name in enumerate(DN_LAYERS):
            v = np.array(d["summary"][name][stat], dtype=np.float64)
            jitter = (np.arange(v.size) - v.size / 2) / max(v.size * 6, 1)
            ax.scatter(i + jitter, v, s=30, color=CATEGORICAL[i], alpha=0.8, zorder=3)
            ax.plot([i - 0.3, i + 0.3], [v.mean()] * 2, color=TEXT_2, linewidth=1.8, zorder=4)
            lo, hi = ci95(v)
            ax.add_patch(plt.Rectangle((i - 0.3, lo), 0.6, hi - lo, color=TEXT_2, alpha=0.12,
                                       zorder=2))
        ax.set_xticks(range(len(DN_LAYERS)),
                      [SHORT[n] + "\n(" + str(d["summary"][n]["median_n_units"]) + " units)"
                       for n in DN_LAYERS],
                      fontsize=8)
        ax.grid(True, color=GRID, linewidth=0.6, axis="y")
        ax.axhline(0, color=GRID, linewidth=1)
        style(ax, "", "descending layer", ylab)
    fig.suptitle(f"Descending layers on one scale - {label}, {dur / 1e3:g} s, {len(seeds)} noise "
                 f"seeds (identical graph and drive)\n"
                 f"hop-2 position on the hop3->hop1 axis: {pos.mean():.2f} +- {pos.std(ddof=1):.2f}",
                 fontsize=10, color=TEXT_2, x=0.01, ha="left")
    save(fig, f"layer_profile_{d['adjacency_sha12']}_d{int(round(dur))}.png")


def compare(paths, layer="motor at hop 2", alpha=0.05, power=0.80):
    """Compare conditions measured with the SAME noise seeds: A (unmodified), B (edges added to the
    real block), C (the same edges displaced).

    The comparison is PAIRED by noise seed - each condition was run at seeds 0..n, so the difference
    is taken seed by seed and the confidence interval is on the mean of those differences. Paired is
    the right design here: the seed fixes the noise realization, so any component shared between
    conditions cancels. The unpaired interval is printed beside it.

    Reported on the EXCESS (real - null mean), not on z: z's denominator is the null sd, which
    itself moves between runs, so a difference in z can come from either term."""
    from scipy.stats import t as student

    conds = []
    for path in paths:
        d = json.loads(Path(path).read_text())
        conds.append({"path": str(path), "label": d.get("label") or Path(path).stem,
                      "sha": d["adjacency_sha12"], "adjacency": d["adjacency"],
                      "duration_ms": d["duration_ms"], "seeds": d["seeds"],
                      "excess": d["summary"][layer]["excess"], "z": d["summary"][layer]["z"],
                      "mi": d["summary"][layer]["mi"],
                      "p": d["summary"][layer].get("p_empirical"),
                      "normalization": d.get("normalization"), "w_scale": d.get("w_scale"),
                      "n_crossed": d["summary"][layer].get("n_crossed_z"),
                      "n_shuffles": d.get("n_shuffles")})
    header(f"CONDITION COMPARISON - {layer}, excess (real - null mean), paired by noise seed")
    durations = {c["duration_ms"] for c in conds}
    if len(durations) > 1:
        raise ValueError(f"conditions were measured at different durations: {durations}")
    points = {(c.get("normalization"), c.get("w_scale")) for c in conds}
    if len(points) > 1:
        raise ValueError(f"conditions were measured at different operating points: {points}; a "
                         f"normalisation variant may only be compared with other runs of that "
                         f"same variant")
    print(f"  {len(conds)} conditions at {conds[0]['duration_ms'] / 1e3:g} s, "
          f"{conds[0]['n_shuffles']} shuffles per run")
    print(f"  {'condition':<28} {'seeds':>5} {'mean excess':>12} {'95% CI':>26} {'sd':>9} "
          f"{'crossed z>=3':>12}")
    for c in conds:
        v = np.array(c["excess"], dtype=np.float64)
        lo, hi = ci95(v)
        c["mean"], c["sd"], c["ci"] = float(v.mean()), float(v.std(ddof=1)), (lo, hi)
        crossed = f"{c['n_crossed']}/{len(v)}"
        print(f"  {c['label']:<28} {len(v):>5} {v.mean():>+12.5f} "
              + f"[{lo:+.5f}, {hi:+.5f}]".rjust(26) + f" {v.std(ddof=1):>9.5f} {crossed:>12}")

    base = conds[0]
    out = {"layer": layer, "duration_ms": base["duration_ms"], "conditions": conds, "pairwise": {}}
    print(f"\n  pairwise differences against {base['label']!r} (paired by seed):")
    print(f"  {'difference':<28} {'mean':>12} {'95% CI (paired)':>28} {'p':>8} "
          f"{'95% CI (unpaired)':>28}")
    for c in conds[1:]:
        if c["seeds"] != base["seeds"]:
            raise ValueError(f"{c['label']} used seeds {c['seeds']}, {base['label']} used "
                             f"{base['seeds']}; a paired comparison needs the same seeds")
        a = np.array(base["excess"], dtype=np.float64)
        b = np.array(c["excess"], dtype=np.float64)
        d = b - a
        n = d.size
        lo, hi = ci95(d)
        tstat = float(d.mean() / (d.std(ddof=1) / np.sqrt(n))) if d.std(ddof=1) > 0 else float("nan")
        pval = float(2 * (1 - student.cdf(abs(tstat), n - 1))) if np.isfinite(tstat) else float("nan")
        se_un = np.sqrt(a.var(ddof=1) / n + b.var(ddof=1) / n)
        h_un = student.ppf(0.975, 2 * n - 2) * se_un
        print(f"  {c['label'] + ' - ' + base['label']:<28} {d.mean():>+12.5f} "
              + f"[{lo:>+.5f}, {hi:>+.5f}]".rjust(28)
              + f" {pval:>8.4f} " + f"[{d.mean() - h_un:>+.5f}, {d.mean() + h_un:>+.5f}]".rjust(28))
        out["pairwise"][f"{c['label']} - {base['label']}"] = {
            "mean_difference": float(d.mean()), "ci95_paired": [lo, hi], "p_paired_t": pval,
            "ci95_unpaired": [float(d.mean() - h_un), float(d.mean() + h_un)],
            "per_seed": d.tolist(),
            "distinguishable_from_zero": bool(lo > 0 or hi < 0)}

    # ---------------------------------------------------------------- power
    header("WHAT THIS DESIGN CAN DETECT")
    sds = {c["label"]: c["sd"] for c in conds}
    n = len(base["excess"])
    tcrit = student.ppf(1 - alpha / 2, n - 1)
    tpow = student.ppf(power, n - 1)
    print(f"  run-to-run sd of the excess, per condition: "
          + ", ".join(f"{k} {v:.5f}" for k, v in sds.items()))
    for c in conds[1:]:
        d = np.array(c["excess"]) - np.array(base["excess"])
        sd_d = float(d.std(ddof=1))
        mde = (tcrit + tpow) * sd_d / np.sqrt(n)
        rel = mde / abs(np.mean(base["excess"])) if np.mean(base["excess"]) else float("nan")
        print(f"  {c['label']} vs {base['label']}: paired sd {sd_d:.5f} over {n} seeds -> minimum "
              f"detectable difference at {power:.0%} power, alpha {alpha} = {mde:+.5f} excess "
              f"({rel:.0%} of the unmodified layer's own excess of "
              f"{np.mean(base['excess']):.5f})")
        out.setdefault("power", {})[c["label"]] = {"paired_sd": sd_d, "n_seeds": n,
                                                   "min_detectable_difference": float(mde),
                                                   "as_fraction_of_baseline_excess": float(rel)}
    dest = CACHE / RunToRunConfig().subdir / f"comparison_d{int(round(base['duration_ms']))}.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=None, help="number of noise seeds besides seed 0")
    ap.add_argument("--adjacency", default=None)
    ap.add_argument("--duration-ms", type=float, default=None)
    ap.add_argument("--label", default=None, help="name for this condition in the comparison report")
    ap.add_argument("--compare", nargs="+", default=None,
                    help="result json files to compare instead of running")
    ap.add_argument("--profile", nargs="+", default=None,
                    help="result json files to profile (three descending layers on one scale)")
    ap.add_argument("--normalization", default=None, choices=("raw", "in_degree", "sqrt_in"))
    ap.add_argument("--w-scale", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.profile:
        profile(a.profile)
    elif a.compare:
        compare(a.compare)
    else:
        main(a.seeds, a.adjacency, a.duration_ms, a.force, a.label, a.normalization, a.w_scale)
