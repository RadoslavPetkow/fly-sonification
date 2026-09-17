"""Calibrate the LIF network into a stable low-rate regime that still transmits input.

    .venv/bin/python -m sim.calibrate [--force] [--protocol v3|v2]

v3 (default): (1) one adaptation panel with the sensory set exempt vs plain b_adapt=0,
(2) refine g_inh x w_scale around the v2 regime with b_adapt 0 and the v3 gates
(drivable-active is a 10% sanity floor; preservation > 70%; per-active < 40 Hz),
then i_ext_max and Stage C. The v2 description follows.

v2 protocol (v1 found a tonic seizure; see config.CalibConfig,
runs/calibrate_v1_stageA_seizure.log and runs/calibrate_v1_source.py):

Stage 0  gain anchors w_anchor(f) = 1000 / (geom * S * f * dt), checked against the
         independently measured values (context for the sweep range).
Sweep    sqrt_in only. Panels b_adapt in {0, b(10%), b(30%)} x g_inh in sweep_g_inh x
         7 log-spaced w_scale over sweep_w_range; noise frac sweep_frac; 1 s each after
         warm-up, every point started from rest; constant weak sensory drive.
         Per panel a silent reference (w_scale = 0) gives the sensory rate without
         recurrent input.
Select   rate in target_rate_hz, >target_drivable_active_frac of the drivable set active,
         sensory rate >= target_sensory_preservation x silent reference, rate per active
         neuron < target_max_rate_per_active_hz. Closest to target_pick_rate_hz, then
         lowest si_net. None qualifying -> report the closest points and STOP.
i_ext_max  measured sensory f-I curve in the winning network.
Stage C  5 s validation + input-transmission test (drive removed / doubled vs baseline,
         replicate-seed noise floor, paired per-motor-neuron effect sizes).

Drivable set: neurons with incoming edges and signed input row sum >= 0 in the built
matrix (row scaling by any normalization is positive, so this set does not depend on
the normalization; it is fixed at g_inh = 1 so its size does not move with the sweep).

Sweep results are cached in cache/calibration_sweep.json keyed on the configs and the
matrix build; --force recomputes.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize

from config import CalibConfig, Paths, SimConfig
from sim.lif_network import LIFNetwork, adaptation_peak_per_b
from sim.normalize import normalize

CACHE = ROOT / Paths().cache
FIGS = ROOT / Paths().figures
CALIBRATION = CACHE / "calibration.json"
SWEEP = CACHE / "calibration_sweep.json"
CAL_V3 = CACHE / "calibration_v3.json"

# reference palette (dataviz skill, light mode)
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
CATEGORICAL = ("#2a78d6", "#eb6834", "#1baf7a")
BLUE_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 30)


def attrs(obj):
    return {k: getattr(obj, k) for k in dir(obj) if not k.startswith("_")}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def rel(p):
    return os.path.relpath(p, ROOT)


def header(text):
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}", flush=True)


def banner(text):
    line = "!" * 78
    print(f"\n{line}\n{text}\n{line}\n", flush=True)


def sigma_for(frac, sc):
    """sigma giving stationary std(V) = frac * v_thresh (OU: std = R * sigma * sqrt(dt / (2 tau_m)))."""
    return frac * sc.v_thresh / (sc.r_membrane * math.sqrt(sc.dt_ms / (2 * sc.tau_m_ms)))


def rheobase(sc):
    return sc.v_thresh / sc.r_membrane


def steps(ms, sc):
    return round(ms / sc.dt_ms)


def fmt_min(seconds):
    return f"{seconds / 60:.1f} min"


# =============================================================================
# Stage 0
# =============================================================================

def stage0(sc, cc, W):
    header("STAGE 0 - gain anchors from the matrix")
    has_in = np.diff(W.indptr) > 0
    geom = 1.0 / (1.0 - math.exp(-sc.dt_ms / sc.tau_syn_ms))
    out = {"geom": geom, "per_norm": {}}
    for norm in cc.normalizations:
        Wn, _ = normalize(W, norm)
        row_sum = np.asarray(Wn.sum(axis=1)).ravel()[has_in]
        S = float(np.median(row_sum))
        if not S > 0:
            raise ValueError(f"{norm}: median signed incoming row sum is {S}; the anchor formula needs S > 0")
        anchors = {str(f): 1000.0 / (geom * S * f * sc.dt_ms) for f in cc.anchor_rates_hz}
        out["per_norm"][norm] = {"S": S, "w_anchor": anchors}
        print(f"  {norm:>9}: S={S:.4g}  " + "  ".join(f"w_anchor({float(f):g} Hz)={a:.4g}" for f, a in anchors.items()))
        del Wn
    for norm, expected in cc.anchor_expected.items():
        got = [out["per_norm"][norm]["w_anchor"][str(f)] for f in cc.anchor_rates_hz]
        err = max(abs(g - e) / e for g, e in zip(got, expected))
        print(f"  check {norm}: max rel. difference from expected {err:.1%} (tolerance {cc.anchor_tolerance:.0%})")
        if err > cc.anchor_tolerance:
            raise RuntimeError(f"{norm} anchors differ from the measured values by {err:.1%}; stopping")
    return out


# =============================================================================
# simulation with on-the-fly metrics
# =============================================================================

class Runner:
    """Holds one LIFNetwork for the current (normalization, g_inh)."""

    def __init__(self, sc, drivable, adjacency_path=None):
        self.sc = sc
        self.key = None
        self.net = None
        self.drivable = drivable
        self.adjacency_path = adjacency_path

    def get(self, norm, g_inh):
        if (norm, g_inh) != self.key:
            self.net = None
            gc.collect()
            self.net = LIFNetwork(CACHE, cfg=replace(self.sc, normalization=norm, g_inh=g_inh),
                                  adjacency_path=self.adjacency_path)
            self.key = (norm, g_inh)
        return self.net

    @torch.no_grad()
    def measure(self, norm, g_inh, b_adapt, w_scale, sigma, drive, warm_ms, meas_ms, seed=None, record=False,
                exempt=False):
        net = self.get(norm, g_inh)
        sc = self.sc
        net.set_scale(w_scale, sigma)
        net.set_adaptation(b_adapt, sensory_exempt=exempt)
        old_seed = net.cfg.seed
        if seed is not None:
            net.cfg.seed = seed
        net.reset()
        net.cfg.seed = old_seed
        I = torch.zeros(net.N, device=net.device)
        I[net.sensory_idx] = drive
        n_warm, n_meas = steps(warm_ms, sc), steps(meas_ms, sc)
        t0 = time.perf_counter()
        for _ in range(n_warm):
            net.step(I)
        counts = torch.zeros(net.N, dtype=torch.int32, device=net.device)
        pop = torch.zeros(n_meas, dtype=torch.int64, device=net.device)
        pop_sens = torch.zeros(n_meas, dtype=torch.int64, device=net.device)
        pop_motor = torch.zeros(n_meas, dtype=torch.int64, device=net.device)
        rec_steps, rec_idx = [], []
        for s in range(n_meas):
            spk = net.step(I)
            counts += spk
            pop[s] = spk.sum()
            pop_sens[s] = spk[net.sensory_idx].sum()
            pop_motor[s] = spk[net.motor_idx].sum()
            if record:
                fired = spk.nonzero().flatten().to(torch.int32)
                if fired.numel():
                    rec_idx.append(fired.cpu())
                    rec_steps.append(torch.full_like(fired, s).cpu())
        wall = time.perf_counter() - t0

        T = n_meas * sc.dt_ms / 1e3
        dt_s = sc.dt_ms / 1e3
        counts = counts.cpu().numpy()
        pop, pop_sens, pop_motor = pop.cpu().numpy(), pop_sens.cpu().numpy(), pop_motor.cpu().numpy()
        sens = net.sensory_idx.cpu().numpy()
        motor = net.motor_idx.cpu().numpy()
        n_ns = net.N - sens.size
        active = counts > 0

        def si(series):
            m = series.mean()
            return float(series.var() / m) if m > 0 else None

        m = {
            "normalization": norm, "g_inh": float(g_inh), "b_adapt": float(b_adapt), "sensory_adapt_exempt": bool(exempt),
            "w_scale": float(w_scale),
            "sigma": float(sigma), "drive": float(drive),
            "rate_hz": float(counts.sum() / net.N / T),
            "active_frac": float(active.mean()),
            "drivable_active_frac": float(active[self.drivable].mean()),
            "rate_per_active_hz": float(counts.sum() / active.sum() / T) if active.any() else 0.0,
            "si_all": si(pop / net.N / dt_s),
            "si_net": si((pop - pop_sens) / n_ns / dt_s),
            "motor_rate_hz": float(counts[motor].mean() / T),
            "sensory_rate_hz": float(counts[sens].mean() / T),
            "exc_fraction": net.ei_stats["exc_fraction"],
            "wall_s": round(wall, 2),
            "ms_per_step": round(wall * 1e3 / (n_warm + n_meas), 2),
        }
        extra = {"motor_rates": counts[motor] / T, "pop": pop, "pop_motor": pop_motor, "rates": counts / T}
        if record:
            extra["rec_steps"] = torch.cat(rec_steps).numpy() if rec_steps else np.empty(0, np.int32)
            extra["rec_idx"] = torch.cat(rec_idx).numpy() if rec_idx else np.empty(0, np.int32)
        return m, extra


def print_row(m, prefix=""):
    si_net = "n/a" if m["si_net"] is None else f"{m['si_net']:.3g}"
    pres = m.get("sensory_preservation")
    pres = "" if pres is None else f" pres={pres:<6.3g}"
    print(f"  {prefix}g={m['g_inh']:<3g} b={m['b_adapt']:<6.4g} w={m['w_scale']:<7.4g} rate={m['rate_hz']:<8.4g} "
          f"act={m['active_frac']:<6.1%} drv_act={m['drivable_active_frac']:<6.1%} "
          f"per_act={m['rate_per_active_hz']:<7.4g} sens={m['sensory_rate_hz']:<7.4g}{pres} "
          f"motor={m['motor_rate_hz']:<8.4g} si_net={si_net:<7} ({m['ms_per_step']} ms/step)", flush=True)


def gate_failures(m, cc):
    tmin, tmax = cc.target_rate_hz
    f = []
    if not tmin <= m["rate_hz"] <= tmax:
        f.append(f"rate {m['rate_hz']:.3g}")
    if not m["drivable_active_frac"] > cc.target_drivable_active_frac:
        f.append(f"drv_act {m['drivable_active_frac']:.1%}")
    if not m["sensory_preservation"] >= cc.target_sensory_preservation:
        f.append(f"pres {m['sensory_preservation']:.2f}")
    if not m["rate_per_active_hz"] < cc.target_max_rate_per_active_hz:
        f.append(f"per_act {m['rate_per_active_hz']:.3g}")
    return f


def shortfall(m, cc):
    """Distance from qualifying: sum of |log| factors by which each failed gate misses."""
    tmin, tmax = cc.target_rate_hz
    d = 0.0
    r = max(m["rate_hz"], 1e-6)
    d += math.log(tmin / r) if r < tmin else (math.log(r / tmax) if r > tmax else 0.0)
    d += max(0.0, math.log(cc.target_drivable_active_frac / max(m["drivable_active_frac"], 1e-6)))
    d += max(0.0, math.log(cc.target_sensory_preservation / max(m["sensory_preservation"], 1e-6)))
    d += max(0.0, math.log(max(m["rate_per_active_hz"], 1e-6) / cc.target_max_rate_per_active_hz))
    return d


# =============================================================================
# figures
# =============================================================================

def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, loc="left", fontsize=10)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, name, tight=True):
    FIGS.mkdir(exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    if tight:
        fig.tight_layout()
    path = FIGS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {rel(path)}")


def heatmaps(rows, cc, winner, prefix="calib_v2"):
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    b_vals = sorted({r["b_adapt"] for r in rows})
    specs = (("rate_hz", "population rate (Hz)", "log", "rate.png"),
             ("drivable_active_frac", "active fraction of drivable set", "lin", "drivable_active.png"),
             ("sensory_preservation", "sensory rate / silent-network sensory rate", "lin", "sensory_preservation.png"),
             ("rate_per_active_hz", "rate per active neuron (Hz)", "log", "rate_per_active.png"))
    for key, label, scale, fname in specs:
        vals = np.array([r[key] for r in rows], dtype=float)
        pos = vals[vals > 0]
        normc = (LogNorm(vmin=pos.min(), vmax=pos.max()) if scale == "log" and pos.size
                 else Normalize(vmin=np.nanmin(vals), vmax=np.nanmax(vals)))
        fig, axes = plt.subplots(1, len(b_vals), figsize=(4.6 * len(b_vals), 3.8), squeeze=False)
        for ax, b in zip(axes[0], b_vals):
            panel = [r for r in rows if r["b_adapt"] == b]
            ws = sorted({r["w_scale"] for r in panel})
            gs = sorted({r["g_inh"] for r in panel})
            Z = np.full((len(gs), len(ws)), np.nan)
            for r in panel:
                v = r[key]
                Z[gs.index(r["g_inh"]), ws.index(r["w_scale"])] = np.nan if (scale == "log" and v <= 0) else v
            im = ax.imshow(Z, origin="lower", aspect="auto", cmap=cmap, norm=normc)
            ax.set_xticks(range(len(ws)), [f"{w:.3g}" for w in ws], rotation=45)
            ax.set_yticks(range(len(gs)), [f"{g:g}" for g in gs])
            for r in panel:
                if not r["fails"]:
                    ax.plot(ws.index(r["w_scale"]), gs.index(r["g_inh"]), "o", ms=4, mfc="white", mec=TEXT, mew=0.8)
            if winner is not None and winner["b_adapt"] == b:
                ax.plot(ws.index(winner["w_scale"]), gs.index(winner["g_inh"]), "o", ms=12, mfc="none", mec=TEXT, mew=1.5)
            style(ax, f"b_adapt {b:.3g}", "w_scale", "g_inh")
        fig.colorbar(im, ax=list(axes[0]), shrink=0.9).ax.tick_params(labelsize=8, colors=TEXT_2)
        fig.suptitle(f"{cc.sweep_normalization}: {label}. Dots = all gates pass, ring = winner, blank = zero.",
                     color=TEXT, fontsize=10, x=0.01, ha="left")
        save(fig, f"{prefix}_{fname}", tight=False)


# =============================================================================
# main
# =============================================================================

def effect(base, cond, rng, n_boot):
    d = cond - base
    boot = rng.choice(d, size=(n_boot, d.size), replace=True).mean(axis=1)
    sd = d.std(ddof=1)
    return {"mean_base_hz": float(base.mean()), "mean_cond_hz": float(cond.mean()),
            "delta_hz": float(d.mean()), "rel_change": float(d.mean() / base.mean()) if base.mean() > 0 else None,
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "d_z": float(d.mean() / sd) if sd > 0 else None,
            "neurons_up": int((d > 0).sum()), "neurons_down": int((d < 0).sum()), "neurons_same": int((d == 0).sum())}


def bkey(b):
    return f"{b:.8g}"


def measure_i_ext_max(runner, sc, cc, norm, g, b, w, sigma, exempt=False):
    """Constant sensory current that drives the sensory set to cc.sensory_full_scale_rate_hz, from a measured f-I curve."""
    header(f"i_ext_max - sensory f-I curve in the winning network (target sensory rate "
           f"{cc.sensory_full_scale_rate_hz:g} Hz)")
    fi = []
    for mult in cc.fi_rheobase_mults:
        current = mult * rheobase(sc)
        m, _ = runner.measure(norm, g, b, w, sigma, current, cc.warmup_ms, cc.fi_ms, exempt=exempt)
        fi.append({"current": current, "rheobase_mult": mult, "sensory_rate_hz": m["sensory_rate_hz"],
                   "rate_hz": m["rate_hz"], "motor_rate_hz": m["motor_rate_hz"]})
        print(f"  I_ext {current:<6.3g} ({mult:g} x rheobase): sensory {m['sensory_rate_hz']:.4g} Hz, "
              f"population {m['rate_hz']:.4g} Hz, motor {m['motor_rate_hz']:.4g} Hz", flush=True)
    target = cc.sensory_full_scale_rate_hz
    rates = [p["sensory_rate_hz"] for p in fi]
    if rates[0] >= target:
        raise RuntimeError(f"sensory rate already {rates[0]:.4g} Hz >= {target} Hz at the lowest tested current; "
                           "cannot bracket i_ext_max")
    reach = [k for k, r in enumerate(rates) if r >= target]
    if not reach:
        raise RuntimeError(f"sensory f-I curve {[round(r, 3) for r in rates]} never reaches {target} Hz; "
                           "cannot measure i_ext_max")
    # i_ext_max = smallest current that reaches the target (first upward crossing)
    i = reach[0]
    (i0, r0), (i1, r1) = (fi[i - 1]["current"], rates[i - 1]), (fi[i]["current"], rates[i])
    i_ext_max = float(math.exp(math.log(i0) + (target - r0) / (r1 - r0) * (math.log(i1) - math.log(i0))))
    print(f"  i_ext_max = {i_ext_max:.4g} (first crossing, log-interpolated between {i0:.3g} -> {r0:.4g} Hz and "
          f"{i1:.3g} -> {r1:.4g} Hz)")
    fall = [k for k in range(i + 1, len(rates)) if rates[k] < target]
    if fall:
        banner(f"NON-MONOTONIC sensory f-I curve: above i_ext_max the sensory rate falls back below {target:g} Hz at\n"
               + ", ".join(f"I {fi[k]['current']:.3g} -> {rates[k]:.4g} Hz" for k in fall)
               + "\nStronger input drive suppresses the input population in this network (recurrent feedback).")
    return i_ext_max, fi, not fall


def stage_c(runner, sc, cc, norm, g, b, w, sigma, drive, calibration, fig_prefix, exempt=False):
    header(f"STAGE C - validation over {cc.stage_c_ms:g} ms + input-transmission test")
    seed = sc.seed
    reps = [f"replicate (seed+{k})" for k in range(1, cc.stage_c_replicates + 1)]
    conds = ([("baseline", drive, seed, True)] + [(r, drive, seed + k, False) for k, r in enumerate(reps, 1)]
             + [("drive removed", 0.0, seed, False), ("drive doubled", 2 * drive, seed, False)])
    res = {}
    for name, dv, sd, rec in conds:
        m, extra = runner.measure(norm, g, b, w, sigma, dv, cc.warmup_ms, cc.stage_c_ms, seed=sd, record=rec,
                                  exempt=exempt)
        res[name] = (m, extra)
        print_row(m, prefix=f"{name:>18} (drive {dv:g}, seed {sd}): ")

    base_m, base_x = res["baseline"]
    dt = sc.dt_ms
    n_steps = steps(cc.stage_c_ms, sc)
    t_ms = np.arange(n_steps) * dt
    net = runner.get(norm, g)
    bin_steps = net.window_steps
    rng = np.random.default_rng(sc.seed)

    fig, ax = plt.subplots(figsize=(10, 3.4))
    ax.plot(t_ms, base_x["pop"] / net.N / (dt / 1e3), color=CATEGORICAL[0], linewidth=0.6)
    ax.set_xlim(0, cc.stage_c_ms)
    ax.grid(True, color=GRID, linewidth=0.6)
    style(ax, f"Stage C baseline: population rate ({dt:g} ms bins); {norm}, w {w:.4g}, g_inh {g:g}, b_adapt {b:.3g}",
          "time after warm-up (ms)", "Hz per neuron")
    save(fig, f"{fig_prefix}_population_rate.png")

    sample = np.sort(rng.choice(net.N, size=500, replace=False))
    row = np.full(net.N, -1)
    row[sample] = np.arange(500)
    r = row[base_x["rec_idx"]]
    keep = r >= 0
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.scatter(base_x["rec_steps"][keep] * dt, r[keep], s=1.5, c=CATEGORICAL[0], linewidths=0)
    ax.set_xlim(0, cc.stage_c_ms)
    ax.set_ylim(-1, 500)
    style(ax, f"Stage C baseline: raster of 500 random neurons ({int(keep.sum()):,} spikes)",
          "time after warm-up (ms)", "neuron (sample row)")
    save(fig, f"{fig_prefix}_raster.png")

    fig, ax = plt.subplots(figsize=(10, 3.6))
    kern = np.ones(bin_steps) / (bin_steps * dt / 1e3) / net.motor_idx.numel()
    for name, color in zip(("drive removed", "baseline", "drive doubled"), CATEGORICAL):
        trace = np.convolve(res[name][1]["pop_motor"], kern, mode="full")[:n_steps]
        ax.plot(t_ms, trace, color=color, linewidth=1.2, label=f"{name} ({res[name][0]['motor_rate_hz']:.3g} Hz)")
    ax.set_xlim(0, cc.stage_c_ms)
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    style(ax, f"Stage C: motor population rate ({net.motor_idx.numel():,} neurons, {bin_steps * dt:g} ms rolling)",
          "time after warm-up (ms)", "Hz per neuron")
    save(fig, f"{fig_prefix}_motor_conditions.png")

    brng = np.random.default_rng(sc.seed)
    base_rates = base_x["motor_rates"]
    eff = {name: effect(base_rates, res[name][1]["motor_rates"], brng, cc.bootstrap_n)
           for name in reps + ["drive removed", "drive doubled"]}
    floor = max(abs(eff[r]["delta_hz"]) for r in reps)
    print(f"\nmotor rate per neuron (n={base_rates.size}), paired against baseline; noise floor = largest |delta| of "
          f"{len(reps)} replicate seeds = {floor:.4g} Hz; measurable = 95% CI excludes 0 AND |delta| > "
          f"{cc.effect_noise_mult:g} x floor")
    for name, e in eff.items():
        relc = "n/a" if e["rel_change"] is None else f"{e['rel_change']:+.1%}"
        dz = "n/a" if e["d_z"] is None else f"{e['d_z']:+.3f}"
        print(f"  {name:>20}: mean {e['mean_base_hz']:.4g} -> {e['mean_cond_hz']:.4g} Hz, delta {e['delta_hz']:+.4g} Hz "
              f"({relc}), 95% CI [{e['ci95'][0]:+.4g}, {e['ci95'][1]:+.4g}], d_z {dz}, "
              f"neurons up/down/same {e['neurons_up']}/{e['neurons_down']}/{e['neurons_same']}")
    rem, dbl = eff["drive removed"], eff["drive doubled"]
    drop = rem["ci95"][1] < 0 and abs(rem["delta_hz"]) > cc.effect_noise_mult * floor
    rise = dbl["ci95"][0] > 0 and abs(dbl["delta_hz"]) > cc.effect_noise_mult * floor
    print(f"\n(a) drive removed -> motor rate drops measurably: {drop}")
    print(f"(b) drive doubled -> motor rate rises measurably: {rise}")
    print("\nper condition:        sensory Hz   population Hz   motor Hz   per-active Hz   si_net")
    for name in ["drive removed", "baseline"] + reps + ["drive doubled"]:
        m = res[name][0]
        si_net = "n/a" if m["si_net"] is None else f"{m['si_net']:.4g}"
        print(f"  {name:>18}   {m['sensory_rate_hz']:<12.4g} {m['rate_hz']:<15.4g} {m['motor_rate_hz']:<10.4g} "
              f"{m['rate_per_active_hz']:<15.4g} {si_net}")

    calibration["stage_c"] = {"metrics": {k: v[0] for k, v in res.items()}, "effects": eff,
                              "noise_floor_hz": floor, "drop_measurable": drop, "rise_measurable": rise}
    calibration["validated"] = bool(drop and rise)
    CALIBRATION.write_text(json.dumps(calibration, indent=2, default=str))
    print(f"updated {rel(CALIBRATION)}: validated = {calibration['validated']}")
    if not drop and not rise:
        banner("THE NETWORK IS NOT TRANSMITTING INPUT: removing the sensory drive does not lower the motor\n"
               "rate and doubling it does not raise it beyond the replicate-seed noise floor. The motor\n"
               "activity is self-generated and the audio would be decorative.\n"
               "calibration.json is marked validated=false. STOPPING.")
        raise SystemExit(3)
    if not (drop and rise):
        banner("Only one of the two transmission tests passed; calibration.json is marked validated=false.")
        raise SystemExit(3)
    return res, eff


def gate_failures_v3(m, cc):
    tmin, tmax = cc.target_rate_hz
    f = []
    if not tmin <= m["rate_hz"] <= tmax:
        f.append(f"rate {m['rate_hz']:.3g}")
    if not m["drivable_active_frac"] >= cc.target_drivable_active_floor:
        f.append(f"drv_act {m['drivable_active_frac']:.1%}")
    if not m["sensory_preservation"] > cc.refine_min_sensory_preservation:
        f.append(f"pres {m['sensory_preservation']:.2f}")
    if not m["rate_per_active_hz"] < cc.refine_max_rate_per_active_hz:
        f.append(f"per_act {m['rate_per_active_hz']:.3g}")
    return f


def shortfall_v3(m, cc):
    tmin, tmax = cc.target_rate_hz
    r = max(m["rate_hz"], 1e-6)
    d = math.log(tmin / r) if r < tmin else (math.log(r / tmax) if r > tmax else 0.0)
    d += max(0.0, math.log(cc.target_drivable_active_floor / max(m["drivable_active_frac"], 1e-6)))
    d += max(0.0, math.log(cc.refine_min_sensory_preservation / max(m["sensory_preservation"], 1e-6)))
    d += max(0.0, math.log(max(m["rate_per_active_hz"], 1e-6) / cc.refine_max_rate_per_active_hz))
    return d


def main_v3(force=False):
    sc, cc = SimConfig(), CalibConfig()
    indices = json.loads((CACHE / "indices.json").read_text())
    W = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    has_in = np.diff(W.indptr) > 0
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    drivable = has_in & (row_sum >= 0)
    anchors = stage0(sc, cc, W)
    del W
    gc.collect()

    norm = cc.sweep_normalization
    frac = cc.sweep_frac
    sigma = sigma_for(frac, sc)
    drive = cc.drive_rheobase_mult * rheobase(sc)
    b_ex = cc.exempt_adapt_peak_frac * sc.v_thresh / adaptation_peak_per_b(sc)
    ws_ex = np.geomspace(*cc.sweep_w_range, cc.sweep_w_points)
    ws_ref = np.geomspace(*cc.refine_w_range, cc.refine_w_points)
    tmin, tmax = cc.target_rate_hz

    header("SETUP (v3)")
    print(f"normalization {norm}; noise frac {frac} -> sigma {sigma:.4g}; drive {drive:g} "
          f"({cc.drive_rheobase_mult:g} x rheobase) into {len(indices['sensory_idx'])} sensory neurons; "
          f"refrac_steps {sc.refrac_steps}; warm-up {cc.warmup_ms:g} ms; every point starts from rest")
    print(f"drivable set: {int(drivable.sum()):,} neurons ({drivable.mean():.1%})")
    print(f"(1) exemption panel: g_inh {cc.exempt_g_inh:g}; variants b=0 | b={b_ex:.4g} | b={b_ex:.4g} sensory-exempt; "
          f"w_scale {', '.join(f'{w:.4g}' for w in ws_ex)}; {cc.sweep_ms:g} ms each. Beats b=0 if per-active rate is "
          f"lower at every w where b=0 has rate >= {tmin:g} Hz AND sensory rate >= {cc.exempt_sensory_tolerance:g} x b=0's there")
    print(f"(2) refine: g_inh {cc.refine_g_inh} x w_scale {', '.join(f'{w:.4g}' for w in ws_ref)}; b_adapt 0; "
          f"{cc.refine_ms:g} ms each")
    print(f"GATES (fixed before running): rate {tmin:g}-{tmax:g} Hz; drivable active >= {cc.target_drivable_active_floor:.0%} "
          f"(sanity floor); sensory preservation > {cc.refine_min_sensory_preservation:.0%} of the w_scale=0 sensory rate; "
          f"rate per active neuron < {cc.refine_max_rate_per_active_hz:g} Hz. Pick rate closest to "
          f"{cc.refine_pick_rate_hz:g} Hz, then lowest si_net. Synchrony reported, not gated.")

    runner = Runner(sc, drivable)
    key = hashlib.sha1(json.dumps({"sim": asdict(sc), "calib": attrs(cc), "build": indices["build_timestamp"],
                                   "protocol": "v3"}, sort_keys=True, default=str).encode()).hexdigest()
    cache = {}
    if CAL_V3.exists() and not force:
        cached = json.loads(CAL_V3.read_text())
        if cached.get("key") == key:
            cache = cached
            print(f"resuming from {rel(CAL_V3)}: {sum(len(v) for v in cache.get('points', {}).values())} points cached")
        else:
            print(f"{rel(CAL_V3)} was made with different configs or matrix; recomputing")
    cache["key"] = key
    cache.setdefault("points", {})

    def point(tag, g, b, w, ms, exempt=False):
        k = f"{tag}|{g:g}|{bkey(b)}|{bkey(w)}|{int(exempt)}"
        if k not in cache["points"]:
            m, _ = runner.measure(norm, g, b, w, sigma, drive, cc.warmup_ms, ms, exempt=exempt)
            m["frac"] = frac
            cache["points"][k] = m
            CAL_V3.write_text(json.dumps(cache, indent=1, default=str))
            print_row(m, prefix=f"{tag:<8} ex={int(exempt)} ")
        return cache["points"][k]

    # ---------------------------------------------------------------- budget
    header("BUDGET")
    ms_step = runner.get(norm, cc.exempt_g_inh).load_ms_per_step
    n_ex = 3 * (len(ws_ex) + 1)
    n_ref = len(cc.refine_g_inh) * len(ws_ref) + 1
    est = (n_ex * steps(cc.warmup_ms + cc.sweep_ms, sc) + n_ref * steps(cc.warmup_ms + cc.refine_ms, sc)) * ms_step / 1e3
    print(f"{ms_step:.2f} ms/step; exemption {n_ex} points x {cc.warmup_ms + cc.sweep_ms:g} ms + refine {n_ref} points x "
          f"{cc.warmup_ms + cc.refine_ms:g} ms = {fmt_min(est)} (+ f-I and Stage C) vs budget {cc.budget_minutes:g} min")
    if est / 60 > cc.budget_minutes:
        raise SystemExit("estimate exceeds budget; cut the grid in CalibConfig first")

    # ---------------------------------------------------------------- (1) exemption panel
    header(f"(1) ADAPTATION WITH SENSORY EXEMPTION - g_inh {cc.exempt_g_inh:g}")
    variants = (("b=0", 0.0, False), (f"b={b_ex:.3g}", b_ex, False), (f"b={b_ex:.3g} exempt", b_ex, True))
    panel = {}
    for label, b, ex in variants:
        ref = point("ex_ref", cc.exempt_g_inh, b, 0.0, cc.sweep_ms, ex)
        pts = [point("ex", cc.exempt_g_inh, b, w, cc.sweep_ms, ex) for w in ws_ex]
        for m in pts:
            m["sensory_preservation"] = m["sensory_rate_hz"] / ref["sensory_rate_hz"]
        panel[label] = (ref, pts)
    print(f"\n{'w_scale':>8} | " + " | ".join(f"{lab:^38}" for lab, _, _ in variants))
    print(f"{'':>8} | " + " | ".join(f"{'rate':>8} {'per_act':>8} {'sensory':>8} {'si_net':>9}" for _ in variants))
    for i, w in enumerate(ws_ex):
        cells = []
        for lab, _, _ in variants:
            m = panel[lab][1][i]
            si_net = "n/a" if m["si_net"] is None else f"{m['si_net']:.3g}"
            cells.append(f"{m['rate_hz']:>8.3g} {m['rate_per_active_hz']:>8.3g} {m['sensory_rate_hz']:>8.3g} {si_net:>9}")
        print(f"{w:>8.4g} | " + " | ".join(cells))
    print("silent (w=0) sensory rate: " + ", ".join(f"{lab} {panel[lab][0]['sensory_rate_hz']:.4g} Hz" for lab, _, _ in variants))
    plain, exem = panel["b=0"][1], panel[f"b={b_ex:.3g} exempt"][1]
    active = [i for i, m in enumerate(plain) if m["rate_hz"] >= tmin]
    lower_per_act = all(exem[i]["rate_per_active_hz"] < plain[i]["rate_per_active_hz"] for i in active)
    keeps_sensory = all(exem[i]["sensory_rate_hz"] >= cc.exempt_sensory_tolerance * plain[i]["sensory_rate_hz"] for i in active)
    ex_beats = bool(active) and lower_per_act and keeps_sensory
    print(f"\ncompared at w_scale {[round(float(ws_ex[i]), 4) for i in active]} (where b=0 rate >= {tmin:g} Hz):")
    for i in active:
        print(f"  w {ws_ex[i]:<7.4g}: per-active {plain[i]['rate_per_active_hz']:.3g} -> {exem[i]['rate_per_active_hz']:.3g} Hz "
              f"({exem[i]['rate_per_active_hz'] / plain[i]['rate_per_active_hz'] - 1:+.0%}); sensory "
              f"{plain[i]['sensory_rate_hz']:.3g} -> {exem[i]['sensory_rate_hz']:.3g} Hz "
              f"({exem[i]['sensory_rate_hz'] / plain[i]['sensory_rate_hz'] - 1:+.0%}); population "
              f"{plain[i]['rate_hz']:.3g} -> {exem[i]['rate_hz']:.3g} Hz")
    print(f"EXEMPTION VERDICT: lower per-active rate at every compared w: {lower_per_act}; sensory rate kept within "
          f"{cc.exempt_sensory_tolerance:g}x: {keeps_sensory} -> beats plain b_adapt=0: {ex_beats}")
    print("(refine below uses b_adapt = 0 as specified, regardless of this verdict)")

    # ---------------------------------------------------------------- (2) refine
    header(f"(2) REFINE - g_inh {cc.refine_g_inh} x {len(ws_ref)} w_scale, b_adapt 0, {cc.refine_ms:g} ms each")
    silent = point("ref_ref", cc.refine_g_inh[0], 0.0, 0.0, cc.refine_ms)
    base_sens = silent["sensory_rate_hz"]
    print(f"preservation baseline: sensory rate with w_scale = 0 (no recurrent input), {cc.refine_ms:g} ms: {base_sens:.4g} Hz")
    rows = []
    for g in cc.refine_g_inh:
        for w in ws_ref:
            m = dict(point("refine", g, 0.0, w, cc.refine_ms))
            m["silent_sensory_rate_hz"] = base_sens
            m["sensory_preservation"] = m["sensory_rate_hz"] / base_sens
            m["fails"] = gate_failures_v3(m, cc)
            rows.append(m)
    df = pd.DataFrame(rows)
    df["gates"] = ["PASS" if not r["fails"] else "; ".join(r["fails"]) for r in rows]
    print(df[["g_inh", "w_scale", "exc_fraction", "rate_hz", "active_frac", "drivable_active_frac", "rate_per_active_hz",
              "sensory_rate_hz", "sensory_preservation", "motor_rate_hz", "si_all", "si_net", "gates"]]
          .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
    qualifying = [r for r in rows if not r["fails"]]
    winner = (min(qualifying, key=lambda r: (abs(r["rate_hz"] - cc.refine_pick_rate_hz),
                                             r["si_net"] if r["si_net"] is not None else math.inf))
              if qualifying else None)
    header("REFINE - heatmaps")
    heatmaps(rows, cc, winner, prefix="calib_v3_refine")
    print(f"\nqualifying points: {len(qualifying)} / {len(rows)}")
    if winner is None:
        closest = sorted(rows, key=lambda r: shortfall_v3(r, cc))[:8]
        cdf = pd.DataFrame(closest)
        cdf["failed"] = ["; ".join(r["fails"]) for r in closest]
        print(cdf[["g_inh", "w_scale", "rate_hz", "drivable_active_frac", "sensory_preservation", "rate_per_active_hz",
                   "motor_rate_hz", "si_net", "failed"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        banner("NO refine point satisfies the v3 gates. calibration.json NOT written. STOPPING.")
        raise SystemExit(2)
    for q in sorted(qualifying, key=lambda r: abs(r["rate_hz"] - cc.refine_pick_rate_hz)):
        print_row(q, prefix="qualifying: ")
    print("WINNER:")
    print_row(winner)
    g, w = winner["g_inh"], winner["w_scale"]
    b, exempt = 0.0, False

    i_ext_max, fi, fi_monotone = measure_i_ext_max(runner, sc, cc, norm, g, b, w, sigma, exempt=exempt)
    calibration = {
        "normalization": norm, "w_scale": w, "g_inh": g, "b_adapt": b, "tau_adapt_ms": sc.tau_adapt_ms,
        "sensory_adapt_exempt": exempt, "sigma_noise": sigma, "frac": frac, "refrac_steps": sc.refrac_steps,
        "i_ext_max": i_ext_max,
        "silent_sensory_rate_hz": base_sens,
        "silent_sensory_rate_definition": (f"mean sensory rate with w_scale = 0 (no recurrent input), same drive "
                                           f"{drive:g}, noise and seed, {cc.refine_ms:g} ms after {cc.warmup_ms:g} ms warm-up"),
        "validated": False,
        "created": now(),
        "protocol": "v3",
        "winner_metrics": {k: winner[k] for k in ("rate_hz", "active_frac", "drivable_active_frac", "rate_per_active_hz",
                                                  "sensory_rate_hz", "sensory_preservation", "motor_rate_hz", "si_all",
                                                  "si_net", "exc_fraction")},
        "calibration_drive": drive,
        "exemption_test": {"b_adapt": b_ex, "beats_plain": ex_beats, "lower_per_active": lower_per_act,
                           "keeps_sensory": keeps_sensory},
        "fi_curve": fi,
        "fi_monotone_above_i_ext_max": fi_monotone,
        "anchors": anchors,
        "n_qualifying": len(qualifying),
        "drivable_neurons": int(drivable.sum()),
        "sim_config": asdict(sc),
        "calib_config": attrs(cc),
        "matrix_build": indices["build_timestamp"],
    }
    CALIBRATION.write_text(json.dumps(calibration, indent=2, default=str))
    print(f"wrote {rel(CALIBRATION)} (validated: false until Stage C passes)")
    stage_c(runner, sc, cc, norm, g, b, w, sigma, drive, calibration, "calib_v3", exempt=exempt)


def main_v2(force=False):
    sc, cc = SimConfig(), CalibConfig()
    indices = json.loads((CACHE / "indices.json").read_text())
    W = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    has_in = np.diff(W.indptr) > 0
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    drivable = has_in & (row_sum >= 0)
    anchors = stage0(sc, cc, W)
    del W
    gc.collect()

    norm = cc.sweep_normalization
    frac = cc.sweep_frac
    sigma = sigma_for(frac, sc)
    drive = cc.drive_rheobase_mult * rheobase(sc)
    per_b = adaptation_peak_per_b(sc)
    b_values = [f * sc.v_thresh / per_b for f in cc.sweep_adapt_peak_fracs]
    ws = np.geomspace(*cc.sweep_w_range, cc.sweep_w_points)
    anchor10 = anchors["per_norm"][norm]["w_anchor"]["10.0"]

    header("SETUP")
    print(f"normalization {norm}; noise frac {frac} -> sigma {sigma:.4g}; drive {drive:g} "
          f"({cc.drive_rheobase_mult:g} x rheobase) into {len(indices['sensory_idx'])} sensory neurons; "
          f"refrac_steps {sc.refrac_steps}; tau_adapt {sc.tau_adapt_ms:g} ms; warm-up {cc.warmup_ms:g} ms, "
          f"every point starts from rest")
    print(f"drivable set: {int(drivable.sum()):,} neurons ({drivable.mean():.1%} of {drivable.size:,}) with incoming "
          f"edges and signed input sum >= 0; {int((has_in & (row_sum < 0)).sum()):,} net-inhibited, "
          f"{int((~has_in).sum()):,} without incoming edges")
    print(f"b_adapt: one spike's adaptation peaks at {per_b:.4f} x b below rest -> "
          + ", ".join(f"{f:.0%} of V_thresh = b {b:.4g}" for f, b in zip(cc.sweep_adapt_peak_fracs, b_values)))
    print(f"g_inh {cc.sweep_g_inh}; w_scale {', '.join(f'{w:.4g}' for w in ws)} "
          f"(= {ws[0] / anchor10:.3g} .. {ws[-1] / anchor10:.3g} x w_anchor(10 Hz))")
    tmin, tmax = cc.target_rate_hz
    print(f"TARGET (fixed before running): rate {tmin:g}-{tmax:g} Hz; drivable active > {cc.target_drivable_active_frac:.0%}; "
          f"sensory rate >= {cc.target_sensory_preservation:g} x silent-network sensory rate; rate per active neuron "
          f"< {cc.target_max_rate_per_active_hz:g} Hz. Pick closest to {cc.target_pick_rate_hz:g} Hz, then lowest si_net. "
          f"Synchrony reported, not gated.")

    runner = Runner(sc, drivable)
    key = hashlib.sha1(json.dumps({"sim": asdict(sc), "calib": attrs(cc), "build": indices["build_timestamp"]},
                                  sort_keys=True, default=str).encode()).hexdigest()
    cache = {}
    if SWEEP.exists() and not force:
        cached = json.loads(SWEEP.read_text())
        if cached.get("key") == key:
            cache = cached
            print(f"resuming from {rel(SWEEP)}: {len(cache.get('rows', []))} sweep points cached")
        else:
            print(f"{rel(SWEEP)} was made with different configs or matrix; recomputing")
    cache["key"] = key
    cache.setdefault("rows", [])
    cache.setdefault("silent_ref", {})

    def persist():
        SWEEP.write_text(json.dumps(cache, indent=1, default=str))

    # ---------------------------------------------------------------- budget
    header("BUDGET")
    net = runner.get(norm, cc.sweep_g_inh[0])
    ms_step = net.load_ms_per_step
    n_points = len(cc.sweep_g_inh) * len(b_values) * len(ws) + len(b_values)
    est = n_points * steps(cc.warmup_ms + cc.sweep_ms, sc) * ms_step / 1e3
    print(f"{ms_step:.2f} ms/step x {n_points} points ({len(b_values)} silent references + "
          f"{len(cc.sweep_g_inh)}x{len(b_values)}x{len(ws)} sweep) x {cc.warmup_ms + cc.sweep_ms:g} ms = {fmt_min(est)} "
          f"(+ {len(cc.sweep_g_inh)} network loads) vs budget {cc.budget_minutes:g} min")
    if est / 60 > cc.budget_minutes:
        raise SystemExit("sweep estimate exceeds budget; cut the grid in CalibConfig first")

    # ---------------------------------------------------------------- sweep
    header("SWEEP")
    t0 = time.perf_counter()
    for b in b_values:
        if bkey(b) not in cache["silent_ref"]:
            m, _ = runner.measure(norm, cc.sweep_g_inh[0], b, 0.0, sigma, drive, cc.warmup_ms, cc.sweep_ms)
            cache["silent_ref"][bkey(b)] = m
            persist()
        m = cache["silent_ref"][bkey(b)]
        print(f"silent reference (w_scale 0), b_adapt {b:.4g}: sensory {m['sensory_rate_hz']:.4g} Hz, "
              f"population {m['rate_hz']:.4g} Hz, drivable active {m['drivable_active_frac']:.1%}")
    by_key = {(r["g_inh"], bkey(r["b_adapt"]), bkey(r["w_scale"])): r for r in cache["rows"]}
    for g in cc.sweep_g_inh:
        for b in b_values:
            print(f"\ng_inh {g:g}, b_adapt {b:.4g}:")
            for w in ws:
                k = (float(g), bkey(b), bkey(w))
                if k in by_key:
                    print_row(by_key[k], prefix="(cached) ")
                    continue
                m, _ = runner.measure(norm, g, b, w, sigma, drive, cc.warmup_ms, cc.sweep_ms)
                m["frac"] = frac
                print_row(m)
                cache["rows"].append(m)
                by_key[k] = m
                persist()
    print(f"\nsweep wall time this run {fmt_min(time.perf_counter() - t0)}")

    rows = cache["rows"]
    for r in rows:
        ref = cache["silent_ref"][bkey(r["b_adapt"])]["sensory_rate_hz"]
        r["silent_sensory_rate_hz"] = ref
        r["sensory_preservation"] = r["sensory_rate_hz"] / ref if ref > 0 else 0.0
        r["fails"] = gate_failures(r, cc)

    header("SWEEP - full tables, one panel per b_adapt")
    cols = ("g_inh", "w_scale", "exc_fraction", "rate_hz", "active_frac", "drivable_active_frac", "rate_per_active_hz",
            "sensory_rate_hz", "sensory_preservation", "motor_rate_hz", "si_net")
    for b, f in zip(b_values, cc.sweep_adapt_peak_fracs):
        panel = sorted((r for r in rows if bkey(r["b_adapt"]) == bkey(b)), key=lambda r: (r["g_inh"], r["w_scale"]))
        df = pd.DataFrame(panel)
        df["gates"] = ["PASS" if not r["fails"] else "; ".join(r["fails"]) for r in panel]
        print(f"\nPANEL b_adapt = {b:.4g} (single-spike adaptation {f:.0%} of V_thresh); silent-network sensory rate "
              f"{cache['silent_ref'][bkey(b)]['sensory_rate_hz']:.4g} Hz")
        print(df[list(cols) + ["gates"]].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\nE/I balance by weight (sqrt_in, after g_inh): " + ", ".join(
        f"g_inh {g:g} -> {next(r['exc_fraction'] for r in rows if r['g_inh'] == g):.1%} excitatory" for g in cc.sweep_g_inh))

    qualifying = [r for r in rows if not r["fails"]]
    winner = (min(qualifying, key=lambda r: (abs(r["rate_hz"] - cc.target_pick_rate_hz),
                                             r["si_net"] if r["si_net"] is not None else math.inf))
              if qualifying else None)

    header("SWEEP - heatmaps")
    heatmaps(rows, cc, winner)

    print(f"\nqualifying points: {len(qualifying)} / {len(rows)}")
    if winner is None:
        closest = sorted(rows, key=lambda r: shortfall(r, cc))[:10]
        print("closest points (ranked by summed |log| shortfall over the failed gates):")
        df = pd.DataFrame(closest)
        df["shortfall"] = [shortfall(r, cc) for r in closest]
        df["failed"] = ["; ".join(r["fails"]) for r in closest]
        print(df[["b_adapt", "g_inh", "w_scale", "rate_hz", "drivable_active_frac", "sensory_preservation",
                  "rate_per_active_hz", "motor_rate_hz", "si_net", "shortfall", "failed"]]
              .to_string(index=False, float_format=lambda x: f"{x:.4g}"))
        banner("NO (g_inh, w_scale, b_adapt) combination satisfies the revised target regime.\n"
               "calibration.json NOT written. Not relaxing the targets. STOPPING.")
        raise SystemExit(2)

    for q in sorted(qualifying, key=lambda r: abs(r["rate_hz"] - cc.target_pick_rate_hz)):
        print_row(q, prefix="qualifying: ")
    print("WINNER:")
    print_row(winner)
    g, b, w = winner["g_inh"], winner["b_adapt"], winner["w_scale"]

    i_ext_max, fi, fi_monotone = measure_i_ext_max(runner, sc, cc, norm, g, b, w, sigma)

    calibration = {
        "normalization": norm, "w_scale": w, "g_inh": g, "b_adapt": b, "tau_adapt_ms": sc.tau_adapt_ms,
        "sigma_noise": sigma, "frac": frac, "refrac_steps": sc.refrac_steps, "i_ext_max": i_ext_max,
        "validated": False,
        "created": now(),
        "protocol": "v2",
        "winner_metrics": {k: winner[k] for k in ("rate_hz", "active_frac", "drivable_active_frac", "rate_per_active_hz",
                                                  "sensory_rate_hz", "silent_sensory_rate_hz", "sensory_preservation",
                                                  "motor_rate_hz", "si_all", "si_net", "exc_fraction")},
        "calibration_drive": drive,
        "adaptation_peak_frac": cc.sweep_adapt_peak_fracs[[bkey(x) for x in b_values].index(bkey(b))],
        "fi_curve": fi,
        "anchors": anchors,
        "n_qualifying": len(qualifying),
        "drivable_neurons": int(drivable.sum()),
        "sim_config": asdict(sc),
        "calib_config": attrs(cc),
        "matrix_build": indices["build_timestamp"],
    }
    CALIBRATION.write_text(json.dumps(calibration, indent=2, default=str))
    print(f"wrote {rel(CALIBRATION)} (validated: false until Stage C passes)")

    stage_c(runner, sc, cc, norm, g, b, w, sigma, drive, calibration, "calib_v2")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="recompute instead of resuming from cache")
    ap.add_argument("--protocol", choices=("v2", "v3"), default="v3")
    args = ap.parse_args()
    (main_v3 if args.protocol == "v3" else main_v2)(force=args.force)
