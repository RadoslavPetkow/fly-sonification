"""Does structured sensory input reach the motor layer, and if not, where does it die?

    .venv/bin/python -m experiments.transmission [--force]

1. Structured drive: the sensory input set is split into AudioConfig.n_bins groups
   (data/groups.py). Each group gets its own OU signal (tau ou_tau_ms), the signals are
   ZCA-whitened so their sample correlation is zero, then mapped log-uniformly into
   [drive_low_rheobase_mult x rheobase, i_ext_max] through the normal CDF.
2. duration_ms of simulation at the parameters in cache/calibration.json; spike counts
   of every neuron binned at bin_ms (cached in cache/transmission_sim.npz; --force re-runs).
3. Per layer (input groups, sensory neurons = hop 0, hop 1..max_hop from the BFS over
   the built matrix, motor neurons, motor by hop, motor groups) the mutual information
   and |Pearson r| between each input group's binned drive and each unit's binned count,
   at zero lag. A unit's score is its best input group. Layer statistic = mean best score
   over units with >= min_spikes spikes.
   Null (TransmissionConfig.null_mode, default "joint"): all input signals circularly shifted by
   the SAME random offset (>= min_shift_ms from zero lag), n_shuffles times, same analysis; this
   keeps each signal's autocorrelation and the cross-signal correlation. "independent" (v1)
   shifts each signal separately.
4. Verdict from the per-layer z-scores and percentiles (thresholds in TransmissionConfig).

Nothing is tuned here: the network parameters come from calibration.json unchanged.
"""
import argparse
import json
import math
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
import torch
from scipy.special import ndtr
from scipy.stats import rankdata

from config import AudioConfig, CalibConfig, MidiConfig, Paths, SimConfig, TransmissionConfig
from data.fetch_connectome import bfs_levels
from data.groups import load_neurons_and_indices, ordered_groups
from sim.calibrate import CATEGORICAL, GRID, SURFACE, TEXT, TEXT_2, banner, header, rheobase, save, style
from sim.lif_network import LIFNetwork

CACHE = ROOT / Paths().cache
SIM_CACHE = CACHE / "transmission_sim.npz"
RESULTS = CACHE / "transmission_results.json"
CALIBRATED_KEYS = ("normalization", "w_scale", "g_inh", "b_adapt", "tau_adapt_ms", "sensory_adapt_exempt",
                   "sigma_noise", "refrac_steps")


# =============================================================================
# drive
# =============================================================================

def make_signals(tc, n_groups, n_steps, dt_ms, rng, whiten_from=0):
    """n_groups independent unit-variance OU series (n_steps x n_groups), ZCA-whitened so that the
    segment [whiten_from:] (the analyzed window) has exactly zero sample correlation and unit variance.
    A linear mix of equal-tau OU processes is again OU with that tau, so autocorrelation is kept."""
    e = math.exp(-dt_ms / tc.ou_tau_ms)
    s = math.sqrt(1 - e * e)
    z = np.empty((n_steps, n_groups))
    z[0] = rng.standard_normal(n_groups)
    xi = rng.standard_normal((n_steps, n_groups))
    for t in range(1, n_steps):
        z[t] = e * z[t - 1] + s * xi[t]
    seg = z[whiten_from:]
    mu = seg.mean(axis=0)
    evals, evecs = np.linalg.eigh(np.cov(seg, rowvar=False))
    zca = evecs @ np.diag(1.0 / np.sqrt(evals)) @ evecs.T
    z = (z - mu) @ zca
    return z / z[whiten_from:].std(axis=0)


def offdiag_abs_max(x):
    c = np.corrcoef(x, rowvar=False)
    return float(np.abs(c[~np.eye(c.shape[0], dtype=bool)]).max()), c


# =============================================================================
# simulation
# =============================================================================

@torch.no_grad()
def simulate(sc, cc, tc, calib, groups, force):
    if SIM_CACHE.exists() and not force:
        print(f"using cached simulation {SIM_CACHE.relative_to(ROOT)} (--force to re-run)")
        d = np.load(SIM_CACHE)
        return d["counts"], d["drive_binned"], d["z"], json.loads(str(d["meta"]))

    net = LIFNetwork(CACHE, cfg=sc)
    n_groups = len(groups)
    n_warm = round(cc.warmup_ms / sc.dt_ms)
    n_meas = round(tc.duration_ms / sc.dt_ms)
    bin_steps = round(tc.bin_ms / sc.dt_ms)
    n_bins = n_meas // bin_steps
    n_meas = n_bins * bin_steps
    rng = np.random.default_rng(tc.seed)
    z = make_signals(tc, n_groups, n_warm + n_meas, sc.dt_ms, rng, whiten_from=n_warm)
    i_lo = tc.drive_low_rheobase_mult * rheobase(sc)
    i_hi = calib["i_ext_max"]
    current = np.exp(np.log(i_lo) + (np.log(i_hi) - np.log(i_lo)) * ndtr(z)).astype(np.float32)

    sens = net.sensory_idx.cpu().numpy()
    pos = {int(n): k for k, n in enumerate(sens)}
    group_of = np.empty(sens.size, dtype=np.int64)
    for g, members in enumerate(groups):
        for n in members:
            group_of[pos[int(n)]] = g
    cur_sens = torch.from_numpy(current[:, group_of])      # (steps, 86)
    I = torch.zeros(net.N)

    net.reset()
    print(f"simulating {cc.warmup_ms:g} ms warm-up + {n_meas * sc.dt_ms / 1e3:g} s "
          f"({n_bins} x {tc.bin_ms:g} ms bins), {net.N:,} neurons", flush=True)
    for t in range(n_warm):
        I[net.sensory_idx] = cur_sens[t]
        net.step(I)
    counts = np.zeros((n_bins, net.N), dtype=np.uint8)
    acc = torch.zeros(net.N, dtype=torch.int16)
    t0 = time.perf_counter()
    for b in range(n_bins):
        acc.zero_()
        for k in range(bin_steps):
            t = n_warm + b * bin_steps + k
            I[net.sensory_idx] = cur_sens[t]
            acc += net.step(I)
        counts[b] = acc.numpy()
        if (b + 1) % 200 == 0:
            el = time.perf_counter() - t0
            print(f"  {(b + 1) * tc.bin_ms / 1e3:>5.0f} s simulated | {el / 60:.1f} min | "
                  f"ETA {el / (b + 1) * (n_bins - b - 1) / 60:.1f} min | last-bin population "
                  f"{counts[b].sum() / net.N / (tc.bin_ms / 1e3):.3g} Hz", flush=True)
    wall = time.perf_counter() - t0
    drive_binned = current[n_warm:n_warm + n_meas].reshape(n_bins, bin_steps, n_groups).mean(axis=1)
    meta = {"n_bins": n_bins, "bin_ms": tc.bin_ms, "wall_s": wall, "ms_per_step": wall * 1e3 / n_meas,
            "i_lo": i_lo, "i_hi": i_hi, "warmup_steps": n_warm}
    np.savez_compressed(SIM_CACHE, counts=counts, drive_binned=drive_binned, z=z[n_warm:].astype(np.float32),
                        meta=json.dumps(meta))
    print(f"simulated in {wall / 60:.1f} min ({meta['ms_per_step']:.2f} ms/step); cached {SIM_CACHE.relative_to(ROOT)}")
    return counts, drive_binned, z[n_warm:], meta


# =============================================================================
# information measures
# =============================================================================

def signal_levels(x, K):
    """(G, T) equiprobable levels by rank; ties share a level (identical to plain ranking for
    continuous signals such as the OU drives, which have no ties)."""
    ranks = rankdata(x, axis=1, method="average")
    return np.clip(((ranks - 1) * K / x.shape[1]).astype(np.int64), 0, K - 1)


def response_levels(R, M):
    """(N, T) rank-based levels; ties share a level; constant rows -> single level."""
    ranks = rankdata(R, axis=1, method="average")
    return np.clip(((ranks - 1) * M / R.shape[1]).astype(np.int64), 0, M - 1)


def mutual_info(Rlev, Slev, M, K):
    """(N, G) plug-in mutual information in bits."""
    N, T = Rlev.shape
    out = np.zeros((N, Slev.shape[0]))
    base = (np.arange(N) * M * K)[:, None]
    for g in range(Slev.shape[0]):
        flat = (base + Rlev * K + Slev[g][None, :]).ravel()
        p = np.bincount(flat, minlength=N * M * K).reshape(N, M, K) / T
        pr = p.sum(axis=2, keepdims=True)
        ps = p.sum(axis=1, keepdims=True)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = np.where(p > 0, p * np.log2(p / (pr * ps)), 0.0)
        out[:, g] = terms.sum(axis=(1, 2))
    return out


def abs_corr(R, S):
    """(N, G) |Pearson r|; zero-variance rows give 0."""
    Rc = R - R.mean(axis=1, keepdims=True)
    Sc = S - S.mean(axis=1, keepdims=True)
    rn = np.linalg.norm(Rc, axis=1, keepdims=True)
    sn = np.linalg.norm(Sc, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (Rc @ Sc.T) / (rn * sn.T)
    return np.abs(np.nan_to_num(r))


def analyze(counts, drive_binned, layers, tc, rng):
    T = counts.shape[0]
    K, M = tc.mi_signal_bins, tc.mi_response_levels
    S = drive_binned.T.astype(np.float64)                       # (G, T)
    G = S.shape[0]
    min_shift = round(tc.min_shift_ms / tc.bin_ms)
    if T - 2 * min_shift < 1:
        raise ValueError("recording too short for the minimum circular shift")
    if tc.null_mode == "joint":
        shifts = np.repeat(rng.integers(min_shift, T - min_shift + 1, size=(tc.n_shuffles, 1)), G, axis=1)
    elif tc.null_mode == "independent":
        shifts = rng.integers(min_shift, T - min_shift + 1, size=(tc.n_shuffles, G))
    else:
        raise ValueError(f"unknown null_mode {tc.null_mode!r}")
    Slev = signal_levels(S, K)

    results = {}
    for name, R in layers.items():
        R = R.astype(np.float64)
        Rlev = response_levels(R, M)
        mi_real = mutual_info(Rlev, Slev, M, K)
        r_real = abs_corr(R, S)
        best_mi, best_r = mi_real.max(axis=1), r_real.max(axis=1)
        null_mi_best = np.empty((tc.n_shuffles, R.shape[0]))
        null_r_best = np.empty((tc.n_shuffles, R.shape[0]))
        for i in range(tc.n_shuffles):
            S_sh = np.stack([np.roll(S[g], shifts[i, g]) for g in range(G)])
            L_sh = np.stack([np.roll(Slev[g], shifts[i, g]) for g in range(G)])
            null_mi_best[i] = mutual_info(Rlev, L_sh, M, K).max(axis=1)
            null_r_best[i] = abs_corr(R, S_sh).max(axis=1)
        res = {"n_units": int(R.shape[0])}
        for key, real, null in (("mi", best_mi, null_mi_best), ("abs_r", best_r, null_r_best)):
            stat, null_stat = float(real.mean()), null.mean(axis=1)
            sd = float(null_stat.std(ddof=1))
            res[key] = {
                "mean_best": stat, "median_best": float(np.median(real)), "p90_best": float(np.percentile(real, 90)),
                "null_mean": float(null_stat.mean()), "null_sd": sd, "null_min": float(null_stat.min()),
                "null_max": float(null_stat.max()),
                "z": (stat - float(null_stat.mean())) / sd if sd > 0 else None,
                "percentile": float((null_stat < stat).mean() * 100),
                "frac_units_above_own_null_max": float((real > null.max(axis=0)).mean()),
            }
        res["best_group_counts"] = np.bincount(mi_real.argmax(axis=1), minlength=G).tolist()
        results[name] = res
    return results, shifts


# =============================================================================
# layers
# =============================================================================

def build_layers(counts, rates, total, sens, motor, dist, in_groups, motor_groups, tc, rng, verbose=True):
    """The transmission experiment's layers; rng draws the hop-layer samples (seed TransmissionConfig.seed + 1)."""
    layers, layer_info = {}, {}

    def add(name, idx, sample=False):
        idx = np.asarray(idx)
        active = idx[total[idx] >= tc.min_spikes]
        chosen = np.sort(rng.choice(active, size=tc.layer_sample, replace=False)) if (
            sample and active.size > tc.layer_sample) else active
        layers[name] = counts[:, chosen].T
        layer_info[name] = {"neurons": int(idx.size), "active": int(active.size), "analyzed": int(chosen.size),
                            "mean_rate_hz": float(rates[idx].mean())}
        if verbose:
            print(f"  {name:<22} {idx.size:>7,} neurons, {active.size:>7,} with >= {tc.min_spikes} spikes, "
                  f"{chosen.size:>5,} analyzed, mean rate {rates[idx].mean():.3g} Hz")

    layers["input groups"] = np.stack([counts[:, g].sum(axis=1) for g in in_groups], axis=1).T
    layer_info["input groups"] = {"neurons": len(in_groups), "active": len(in_groups), "analyzed": len(in_groups)}
    if verbose:
        print(f"  {'input groups':<22} {len(in_groups)} summed groups")
    add("hop 0 (sensory)", sens)
    for h in range(1, tc.max_hop + 1):
        add(f"hop {h}", np.flatnonzero(dist == h), sample=True)
    add("motor (all)", motor)
    for h in sorted(set(dist[motor].tolist())):
        add(f"motor at hop {h}", motor[dist[motor] == h])
    layers["motor groups"] = np.stack([counts[:, g].sum(axis=1) for g in motor_groups], axis=1).T
    layer_info["motor groups"] = {"neurons": len(motor_groups), "active": len(motor_groups), "analyzed": len(motor_groups)}
    if verbose:
        print(f"  {'motor groups':<22} {len(motor_groups)} summed groups")
    return layers, layer_info


# =============================================================================
# main
# =============================================================================

def main(force=False):
    tc, cc = TransmissionConfig(), CalibConfig()
    calib = json.loads((CACHE / "calibration.json").read_text())
    sc = replace(SimConfig(), **{k: calib[k] for k in CALIBRATED_KEYS})
    for k, v in calib["sim_config"].items():
        if k not in CALIBRATED_KEYS and getattr(sc, k) != v:
            raise ValueError(f"SimConfig.{k} = {getattr(sc, k)} differs from the calibration's {v}")
    neurons, indices = load_neurons_and_indices()
    sens = np.array(indices["sensory_idx"])
    motor = np.array(indices["motor_idx"])
    n_in_groups, n_motor_groups = AudioConfig().n_bins, MidiConfig().n_groups
    in_groups = ordered_groups(neurons, sens, n_in_groups)
    motor_groups = ordered_groups(neurons, motor, n_motor_groups)

    header("SETUP")
    print(f"calibration.json (protocol {calib['protocol']}, validated {calib['validated']}): "
          + ", ".join(f"{k}={calib[k]}" for k in CALIBRATED_KEYS) + f", i_ext_max={calib['i_ext_max']:.4g}")
    fi = [p for p in calib["fi_curve"] if tc.drive_low_rheobase_mult * rheobase(sc) <= p["current"] <= calib["i_ext_max"]]
    fi_rates = [p["sensory_rate_hz"] for p in fi]
    print("f-I points inside the drive range: " + ", ".join(f"I {p['current']:g} -> {p['sensory_rate_hz']:.4g} Hz" for p in fi))
    if any(b <= a for a, b in zip(fi_rates, fi_rates[1:])):
        raise ValueError("the measured f-I curve is not monotonic inside the drive range")
    print(f"input groups ({n_in_groups}): sizes {[len(g) for g in in_groups]}; motor groups ({n_motor_groups}): "
          f"sizes {[len(g) for g in motor_groups]}")

    W = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    dist = bfs_levels(W.T.tocsr(), sens)
    del W
    hop_sizes = {h: int((dist == h).sum()) for h in range(tc.max_hop + 1)}
    print(f"BFS hop layers from the sensory set (unsigned, built matrix): {hop_sizes}; unreachable {int((dist < 0).sum())}")

    counts, drive_binned, z, meta = simulate(sc, cc, tc, calib, in_groups, force)
    T = counts.shape[0]

    header("DRIVE SIGNALS")
    max_fine, _ = offdiag_abs_max(z)
    max_bin, cbin = offdiag_abs_max(drive_binned)
    print(f"pairwise correlation of the {n_in_groups} OU signals: max |r| {max_fine:.2e} at {SimConfig().dt_ms:g} ms "
          f"resolution (ZCA-whitened), max |r| {max_bin:.4f} for the binned drive currents")
    print(np.array2string(cbin, precision=3, suppress_small=True, max_line_width=160))
    lag1 = np.mean([np.corrcoef(drive_binned[:-1, g], drive_binned[1:, g])[0, 1] for g in range(n_in_groups)])
    print(f"binned drive autocorrelation at one bin ({tc.bin_ms:g} ms): {lag1:.3f} "
          f"(OU expectation exp(-{tc.bin_ms:g}/{tc.ou_tau_ms:g}) = {math.exp(-tc.bin_ms / tc.ou_tau_ms):.3f})")
    print(f"drive current range {drive_binned.min():.3g} .. {drive_binned.max():.3g} (bounds {meta['i_lo']:.3g} .. "
          f"{meta['i_hi']:.3g}); sensory group rates: " + ", ".join(
              f"{counts[:, g].sum() / len(g) / (T * tc.bin_ms / 1e3):.3g}" for g in in_groups) + " Hz")

    # ---------------------------------------------------------------- layers
    header("LAYERS")
    rng = np.random.default_rng(tc.seed + 1)
    total = counts.sum(axis=0, dtype=np.int64)
    rates = total / (T * tc.bin_ms / 1e3)
    print(f"population {rates.mean():.4g} Hz over {T * tc.bin_ms / 1e3:g} s; sensory {rates[sens].mean():.4g} Hz; "
          f"motor {rates[motor].mean():.4g} Hz")
    layers, layer_info = build_layers(counts, rates, total, sens, motor, dist, in_groups, motor_groups, tc, rng)

    # ---------------------------------------------------------------- analysis
    header(f"TRANSMISSION - MI and |r| vs {n_in_groups} input signals, {tc.n_shuffles} {tc.null_mode} circular-shift nulls")
    t0 = time.perf_counter()
    results, shifts = analyze(counts, drive_binned, layers, tc, np.random.default_rng(tc.seed + 2))
    print(f"analysis {time.perf_counter() - t0:.1f} s; shifts {shifts.min()}..{shifts.max()} bins "
          f"(min {round(tc.min_shift_ms / tc.bin_ms)})")
    for key, label in (("mi", "MI (bits), best input group per unit"), ("abs_r", "|r|, best input group per unit")):
        print(f"\n{label}:")
        print(f"  {'layer':<22} {'units':>6} {'mean':>9} {'median':>9} {'p90':>9} {'null mean':>10} {'null max':>9} "
              f"{'z':>8} {'pctile':>7} {'units>own null max':>19}")
        for name, r in results.items():
            s = r[key]
            z = "n/a" if s["z"] is None else f"{s['z']:.2f}"
            print(f"  {name:<22} {r['n_units']:>6} {s['mean_best']:>9.4g} {s['median_best']:>9.4g} {s['p90_best']:>9.4g} "
                  f"{s['null_mean']:>10.4g} {s['null_max']:>9.4g} {z:>8} {s['percentile']:>6.0f}% "
                  f"{s['frac_units_above_own_null_max']:>18.1%}")
    own = results["hop 0 (sensory)"]
    print(f"\nexpected by chance: {1 / (tc.n_shuffles + 1):.1%} of units above their own null max")

    def above(name, key="mi"):
        s = results[name][key]
        return s["z"] is not None and s["z"] >= tc.null_z_min and s["mean_best"] > s["null_max"]

    # ---------------------------------------------------------------- figures
    header("FIGURES")
    hop_names = ["hop 0 (sensory)"] + [f"hop {h}" for h in range(1, tc.max_hop + 1)]
    for key, ylab, fname in (("mi", "mean best-group MI (bits)", "transmission_mi_vs_hop.png"),
                             ("abs_r", "mean best-group |r|", "transmission_corr_vs_hop.png")):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        xs = list(range(len(hop_names)))
        real = [results[n][key]["mean_best"] for n in hop_names]
        nmin = [results[n][key]["null_min"] for n in hop_names]
        nmax = [results[n][key]["null_max"] for n in hop_names]
        nmean = [results[n][key]["null_mean"] for n in hop_names]
        ax.fill_between(xs, nmin, nmax, color=GRID, alpha=0.9, label=f"null range ({tc.n_shuffles} shifts)")
        ax.plot(xs, nmean, color=TEXT_2, linewidth=1.2, linestyle="--", label="null mean")
        ax.plot(xs, real, color=CATEGORICAL[0], linewidth=2, marker="o", ms=6, label="real")
        mx = len(xs)
        mreal = results["motor (all)"][key]
        ax.errorbar([mx], [mreal["null_mean"]], yerr=[[mreal["null_mean"] - mreal["null_min"]],
                                                       [mreal["null_max"] - mreal["null_mean"]]],
                    color=TEXT_2, capsize=4, linewidth=1)
        ax.plot([mx], [mreal["mean_best"]], "o", color=CATEGORICAL[1], ms=8, label="motor neurons (real)")
        ax.set_xticks(xs + [mx], [n.replace(" (sensory)", "\n(sensory)") for n in hop_names] + ["motor\n(all)"])
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
        ax.grid(True, color=GRID, linewidth=0.6, axis="y")
        style(ax, f"Transmission vs hop distance: {ylab}, {T * tc.bin_ms / 1e3:g} s, {tc.bin_ms:g} ms bins",
              "layer", ylab)
        save(fig, fname)

    # ---------------------------------------------------------------- verdict
    header("VERDICT")
    for n in results:
        print(f"  {n:<22} MI above null: {above(n)}   |r| above null: {above(n, 'abs_r')}")
    motor_ok = above("motor (all)") or above("motor groups")
    hop_ok = {h: above(f"hop {h}") for h in range(1, tc.max_hop + 1)}
    if motor_ok:
        verdict = ("INFORMATION REACHES THE MOTOR LAYER: motor MI is above the circular-shift null. The mean-rate "
                   "test was the wrong instrument.")
    elif hop_ok[1]:
        dies = next((h for h in range(2, tc.max_hop + 1) if not hop_ok[h]), None)
        verdict = (f"SIGNAL IS SWALLOWED BY THE RECURRENT BULK: above null at hop 1, at null by hop {dies}"
                   if dies is not None else
                   f"above null through hop {tc.max_hop} but NOT in the motor layer")
    else:
        verdict = "MI IS AT NULL EVEN AT HOP 1: the problem is upstream of the recurrence (look at the drive)"
    if not above("hop 0 (sensory)"):
        verdict += " [WARNING: the sensory neurons themselves do not carry the drive above null]"
    banner(verdict)
    RESULTS.write_text(json.dumps({"verdict": verdict, "results": results, "layers": layer_info,
                                   "input_groups_bodyIds": [neurons.iloc[g]["bodyId"].tolist() for g in in_groups],
                                   "null_mode": tc.null_mode,
                                   "hop_sizes": hop_sizes, "drive": {"max_abs_r_fine": max_fine, "max_abs_r_binned": max_bin,
                                                                     "lag1_autocorr": lag1},
                                   "sim_meta": meta, "calibration": {k: calib[k] for k in CALIBRATED_KEYS},
                                   "transmission_config": asdict(tc)}, indent=2, default=str))
    print(f"wrote {RESULTS.relative_to(ROOT)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="re-run the simulation even if cached")
    main(force=ap.parse_args().force)
