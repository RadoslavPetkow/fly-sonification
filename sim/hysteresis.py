"""Is the silent -> active transition bistable? One continuous up/down w_scale ramp.

    .venv/bin/python -m sim.hysteresis

w_scale is ramped log-linearly from w_lo to w_hi over hysteresis_ramp_ms and back
over the same time, without resetting state. Population rate is binned; the two
branches are compared at the w_scale where each first crosses
hysteresis_cross_rate_hz. Uses SimConfig exactly as configured (recorded in the
output), constant sensory drive and noise as in calibration.
"""
import argparse
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
import torch

from config import CalibConfig, Paths, SimConfig
from sim.calibrate import CATEGORICAL, GRID, SURFACE, TEXT, TEXT_2, rheobase, save, sigma_for, steps, style
from sim.lif_network import LIFNetwork


def crossing(w, rate, level, direction):
    """w at the first bin where the rate crosses `level` in the ramp direction."""
    idx = np.flatnonzero(rate >= level) if direction == "up" else np.flatnonzero(rate < level)
    return (float(w[idx[0]]), int(idx[0])) if idx.size else (None, None)


@torch.no_grad()
def main(refrac_steps=None):
    sc, cc = SimConfig(), CalibConfig()
    if refrac_steps is not None:
        sc = replace(sc, refrac_steps=refrac_steps)
    net = LIFNetwork(ROOT / Paths().cache, cfg=replace(sc, normalization=cc.hysteresis_normalization))
    sigma = sigma_for(cc.hysteresis_frac, sc)
    drive = cc.drive_rheobase_mult * rheobase(sc)
    I = torch.zeros(net.N, device=net.device)
    I[net.sensory_idx] = drive
    w_lo, w_hi = cc.hysteresis_w_range
    n_ramp, n_bin, n_warm = steps(cc.hysteresis_ramp_ms, sc), steps(cc.hysteresis_bin_ms, sc), steps(cc.warmup_ms, sc)
    print(f"hysteresis: {cc.hysteresis_normalization}, w_scale {w_lo:g} -> {w_hi:g} -> {w_lo:g}, "
          f"{cc.hysteresis_ramp_ms:g} ms each way, frac {cc.hysteresis_frac} (sigma {sigma:.4g}), drive {drive:g}; "
          f"refrac_steps {sc.refrac_steps}")

    ramp_up = w_lo * (w_hi / w_lo) ** (np.arange(n_ramp) / (n_ramp - 1))
    schedule = np.concatenate([ramp_up, ramp_up[::-1]])
    net.set_scale(w_lo, sigma)
    net.reset()
    for _ in range(n_warm):
        net.step(I)
    t0 = time.perf_counter()
    pop = np.zeros(schedule.size)
    pop_sens = np.zeros(schedule.size)
    for s, w in enumerate(schedule):
        net.set_scale(w, sigma)
        spk = net.step(I)
        pop[s] = float(spk.sum())
        pop_sens[s] = float(spk[net.sensory_idx].sum())
        if s % 2000 == 1999:
            print(f"  t={s + 1:>5} ms  w_scale {w:.4g}  last-bin rate {pop[s - n_bin + 1:s + 1].mean() / net.N * 1e3:.4g} Hz",
                  flush=True)
    net.report_benchmark("hysteresis run", (time.perf_counter() - t0) * 1e3 / schedule.size)

    n_bins = n_ramp // n_bin
    dt_s = sc.dt_ms / 1e3
    branches = {}
    for name, sl in (("up", slice(0, n_bins * n_bin)), ("down", slice(n_ramp, n_ramp + n_bins * n_bin))):
        w = np.exp(np.log(schedule[sl]).reshape(n_bins, n_bin).mean(axis=1))
        rate = pop[sl].reshape(n_bins, n_bin).mean(axis=1) / net.N / dt_s
        sens = pop_sens[sl].reshape(n_bins, n_bin).mean(axis=1) / net.sensory_idx.numel() / dt_s
        branches[name] = (w, rate, sens)

    L = cc.hysteresis_cross_rate_hz
    w_up, i_up = crossing(branches["up"][0], branches["up"][1], L, "up")
    w_dn, i_dn = crossing(branches["down"][0], branches["down"][1], L, "down")
    print(f"\nup branch: rate first >= {L:g} Hz at w_scale {w_up} (bin {i_up})")
    print(f"down branch: rate first < {L:g} Hz at w_scale {w_dn} (bin {i_dn})")
    wu, ru, su = branches["up"]
    wd, rd, sd = branches["down"]
    print("\n  w_scale   rate_up(Hz)  rate_down(Hz)  sensory_up  sensory_down")
    for k in range(0, n_bins, 5):
        j = n_bins - 1 - k   # down branch runs high -> low; match the same w
        print(f"  {wu[k]:<9.4g} {ru[k]:<12.4g} {rd[j]:<14.4g} {su[k]:<11.4g} {sd[j]:.4g}")
    ratio_is_lower_bound = False
    if w_up is None:
        verdict = "NO IGNITION: the up ramp never reached the crossing rate; range does not span the transition"
        ratio = None
    elif w_dn is None:
        # the down branch stays above the crossing level all the way to w_lo: the loop is
        # at least as wide as w_up / w_lo
        ratio, ratio_is_lower_bound = w_up / branches["down"][0][-1], True
        print(f"\ndown branch never fell below {L:g} Hz; up-crossing / lowest w_scale reached = {ratio:.3f} "
              f"(a LOWER BOUND on the loop width; loop if > {cc.hysteresis_loop_min_ratio:g})")
        verdict = ("LOOP: genuinely bistable (down branch still active at the bottom of the ramp)"
                   if ratio > cc.hysteresis_loop_min_ratio else "OVERLAP: continuous, steep transition")
    else:
        ratio = w_up / w_dn
        verdict = ("LOOP: genuinely bistable" if ratio > cc.hysteresis_loop_min_ratio
                   else "OVERLAP: continuous, steep transition")
        print(f"\nup-crossing / down-crossing = {ratio:.3f} (loop if > {cc.hysteresis_loop_min_ratio:g})")
    print(f"VERDICT: {verdict}")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for (name, label), color in zip((("up", "ramp up"), ("down", "ramp down")), CATEGORICAL):
        w, rate, _ = branches[name]
        ax.plot(w, np.maximum(rate, 1e-3), color=color, linewidth=2, label=label, marker="o", ms=3)
    ax.axhline(L, color=TEXT_2, linewidth=0.8, linestyle=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    style(ax, f"Hysteresis ({cc.hysteresis_normalization}, refrac {sc.refrac_steps} steps): population rate vs w_scale, "
          f"{cc.hysteresis_bin_ms:g} ms bins. {verdict.split(':')[0]}",
          "w_scale (log)", "population rate, Hz per neuron (log)")
    save(fig, f"hysteresis_{cc.hysteresis_normalization}_refrac{sc.refrac_steps}.png")
    result = {"w_up_cross": w_up, "w_down_cross": w_dn, "ratio": ratio, "ratio_is_lower_bound": ratio_is_lower_bound,
              "verdict": verdict, "sim_config": asdict(sc),
              "branches": {k: {"w_scale": v[0].tolist(), "rate_hz": v[1].tolist(), "sensory_rate_hz": v[2].tolist()}
                           for k, v in branches.items()}}
    out = ROOT / Paths().runs / f"hysteresis_{cc.hysteresis_normalization}_refrac{sc.refrac_steps}.json"
    out.write_text(json.dumps(result, indent=1))
    print(f"  wrote {out.relative_to(ROOT)}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refrac-steps", type=int, default=None, help="override SimConfig.refrac_steps")
    main(ap.parse_args().refrac_steps)
