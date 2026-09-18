"""Does a given normalisation have a usable operating regime at all?

    .venv/bin/python -m experiments.norm_regime [--normalization raw] [--points 11]

Every transmission result in this project is measured under sqrt_in, which won the calibration.
sqrt_in is not neutral for the hop-2 descending readout: those cells carry 5.7x the total incoming
weight of a typical hop-2 neuron, so dividing each row by sqrt(sum_j |W[i, j]|) attenuates their
inputs 2.4x relative to their neighbours, purely for being large.

in_degree cannot test that concern - dividing by the sum itself would attenuate them 5.7x, harder
still, so no outcome of it could overturn the worry. `raw` can: with no row scaling there is no
differential attenuation at all, and the descending cells' measured 5.2x absolute advantage in
auditory input weight (experiments/input_share.py) applies undiminished.

This module asks the prior question - whether `raw` can be brought into the calibrated operating
band at all - BEFORE any mutual information is computed, so the choice of w_scale cannot be
influenced by the answer it produces. w_scale is re-derived from the anchor already in
cache/calibration.json (raw: S = 26, w_anchor(10 Hz) = 0.697) and swept around it; the pick rule is
the calibration's own: the in-band point whose rate is closest to CalibConfig.target_pick_rate_hz.

If no w_scale puts the network inside CalibConfig.target_rate_hz with rate_per_active below the
v3 ceiling, that is itself the answer, and it means every normalisation yielding a stable network
divides by S^alpha with alpha >= 0.5 - so the result cannot be separated from that family.
"""
import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp

from config import CalibConfig, Paths, SimConfig
from sim.calibrate import Runner, banner, header, rheobase

CACHE = ROOT / Paths().cache
CALIBRATED_KEYS = ("g_inh", "b_adapt", "tau_adapt_ms", "sensory_adapt_exempt", "sigma_noise",
                   "refrac_steps")


def main(normalization="raw", points=11, meas_ms=1000.0):
    cc = CalibConfig()
    calib = json.loads((CACHE / "calibration.json").read_text())
    # everything except the normalisation and w_scale is taken from the calibrated network
    sc = replace(SimConfig(), normalization=normalization,
                 **{k: calib[k] for k in CALIBRATED_KEYS})
    anchor = calib["anchors"]["per_norm"][normalization]
    w10 = anchor["w_anchor"]["10.0"]

    header(f"REGIME PROBE - normalization {normalization!r}")
    print(f"anchor from cache/calibration.json: S = {anchor['S']:.4g}, "
          f"w_anchor(10 Hz) = {w10:.4g} (formula 1000 / (geom * S * f * dt), "
          f"geom = {calib['anchors']['geom']:.4g})")
    print(f"held fixed from the calibrated network: "
          + ", ".join(f"{k}={getattr(sc, k)}" for k in CALIBRATED_KEYS))
    print(f"gates, fixed before this run: rate in {cc.target_rate_hz} Hz, rate_per_active < "
          f"{cc.target_max_rate_per_active_hz:g} Hz, drivable-active >= "
          f"{cc.target_drivable_active_floor:.0%}; pick = the in-band point closest to "
          f"{cc.target_pick_rate_hz:g} Hz")

    W = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    has_in = np.diff(W.indptr) > 0
    row_sum = np.asarray(W.sum(axis=1)).ravel()
    drivable = has_in & (row_sum >= 0)
    del W
    print(f"drivable set: {int(drivable.sum()):,} neurons ({drivable.mean():.1%})")

    ws = np.unique(np.round(w10 * np.logspace(-1.0, 0.7, points), 6))
    drive = cc.drive_rheobase_mult * rheobase(sc)
    runner = Runner(sc, drivable)
    print(f"\nsweeping w_scale over {ws[0]:.4g}..{ws[-1]:.4g} "
          f"({ws[0] / w10:.2f}x..{ws[-1] / w10:.2f}x the 10 Hz anchor), constant drive "
          f"{drive:g} into the sensory set, {cc.warmup_ms:g} ms warm-up + {meas_ms:g} ms measured\n")
    print(f"  {'w_scale':>9} {'x anchor':>9} {'rate_hz':>9} {'active':>8} {'drv_act':>8} "
          f"{'per_active':>11} {'sensory_hz':>11} {'motor_hz':>9}  gates")
    rows = []
    for w in ws:
        m, _ = runner.measure(normalization, sc.g_inh, sc.b_adapt, float(w), sc.sigma_noise, drive,
                              cc.warmup_ms, meas_ms)
        fails = []
        if not cc.target_rate_hz[0] <= m["rate_hz"] <= cc.target_rate_hz[1]:
            fails.append("rate")
        if not m["rate_per_active_hz"] < cc.target_max_rate_per_active_hz:
            fails.append("per_active")
        if not m["drivable_active_frac"] >= cc.target_drivable_active_floor:
            fails.append("drv_act")
        m["fails"] = fails
        m["x_anchor"] = float(w / w10)
        rows.append(m)
        print(f"  {w:>9.4g} {w / w10:>9.3g} {m['rate_hz']:>9.4g} {m['active_frac']:>7.2%} "
              f"{m['drivable_active_frac']:>7.2%} {m['rate_per_active_hz']:>11.4g} "
              f"{m['sensory_rate_hz']:>11.4g} {m['motor_rate_hz']:>9.4g}  "
              + ("PASS" if not fails else "fail: " + ",".join(fails)), flush=True)

    ok = [m for m in rows if not m["fails"]]
    header("VERDICT")
    out = {"normalization": normalization, "anchor_S": anchor["S"], "w_anchor_10hz": w10,
           "drive": drive, "meas_ms": meas_ms, "rows": rows, "usable": bool(ok)}
    if ok:
        pick = min(ok, key=lambda m: abs(m["rate_hz"] - cc.target_pick_rate_hz))
        out["pick"] = pick
        print(f"  {normalization!r} HAS a usable regime: {len(ok)} of {len(rows)} swept points pass "
              f"every gate.")
        print(f"  picked w_scale = {pick['w_scale']:.6g} ({pick['x_anchor']:.3g}x the 10 Hz anchor): "
              f"rate {pick['rate_hz']:.4g} Hz, active {pick['active_frac']:.2%}, per-active "
              f"{pick['rate_per_active_hz']:.4g} Hz, sensory {pick['sensory_rate_hz']:.4g} Hz, "
              f"motor {pick['motor_rate_hz']:.4g} Hz")
        print(f"  the pick rule is the calibration's own (closest to "
              f"{cc.target_pick_rate_hz:g} Hz) and was fixed before any MI was computed.")
        banner(f"{normalization}: usable, w_scale = {pick['w_scale']:.6g}")
    else:
        print(f"  {normalization!r} has NO usable regime: not one of the {len(rows)} swept points "
              f"lands inside rate {cc.target_rate_hz} Hz with rate_per_active < "
              f"{cc.target_max_rate_per_active_hz:g} Hz.")
        print("  That is itself the answer: every normalisation that yields a stable network here "
              "divides by S^alpha with alpha >= 0.5, so the transmission result cannot be separated "
              "from that family of choices.")
        banner(f"{normalization}: NO usable regime")
    dest = CACHE / f"norm_regime_{normalization}.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {dest.relative_to(ROOT)}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--normalization", default="raw", choices=("raw", "in_degree", "sqrt_in"))
    ap.add_argument("--points", type=int, default=11)
    ap.add_argument("--meas-ms", type=float, default=1000.0)
    a = ap.parse_args()
    main(a.normalization, a.points, a.meas_ms)
