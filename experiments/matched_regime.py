"""Control 1a: does a working window exist at all for degree-matched shuffled graphs?

    .venv/bin/python -m experiments.matched_regime

PREDICTION (stated before running): the real graph needed g_inh = 2.5 to open a low-rate, input-driven window that
did not exist at g_inh = 1. The question is whether degree-matched random graphs have such a window at the SAME
g_inh at all. If they do not, the connectome's topology admits a stable low-rate input-driven state that
degree-matched random graphs do not.

Protocol, identical for every graph (the real graph is included as a positive control; if the protocol cannot find
the real graph's window, a negative result for the shuffles means nothing):
  graphs     real; ShuffleConfig.regime_conditions x regime_graph_seeds shuffled matrices (cache/shuffles/)
  fixed      calibrated normalization, g_inh, b_adapt, noise (cache/calibration.json); calibration drive
             CalibConfig.drive_rheobase_mult x rheobase into the sensory set; warm-up CalibConfig.warmup_ms, every
             point from rest
  grid       CalibConfig.stage_a_points w_scale values: the graph's OWN anchor w_anchor(stage_a_center_rate_hz)
             (1000 / (geom * S * f * dt), S = median signed incoming row sum of the normalized matrix over neurons
             with incoming edges) x logspace(-stage_a_log10_span, +stage_a_log10_span); CalibConfig.stage_a_ms each
  target     (1) population rate in CalibConfig.target_rate_hz
             (2) sensory rate > CalibConfig.refine_min_sensory_preservation x the w_scale = 0 sensory rate
             (3) input-driven: with the drive removed (from rest, same noise) the non-sensory rate stays
                 <= AudioConfig.ignition_rate_hz (run only at points passing 1 and 2)
  window     a graph has a working window if any grid point passes all three.
Writes cache/matched_regime.json, figures/matched_regime.png, experiments/matched_regime.md.
"""
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from audio.audio_in import CALIBRATED_KEYS
from config import AudioConfig, CalibConfig, Paths, ShuffleConfig, SimConfig
from data.groups import load_neurons_and_indices
from experiments.shuffle_graph import shuffle_path
from sim.calibrate import CATEGORICAL, GRID, TEXT, TEXT_2, Runner, header, rheobase, save, style
from sim.normalize import normalize

CACHE = ROOT / Paths().cache
OUT_JSON = CACHE / "matched_regime.json"
OUT_MD = ROOT / "experiments" / "matched_regime.md"


def anchor(path, sc, cc):
    W = sp.load_npz(path).tocsr()
    has_in = np.diff(W.indptr) > 0
    Wn, _ = normalize(W, sc.normalization)
    S = float(np.median(np.asarray(Wn.sum(axis=1)).ravel()[has_in]))
    if not S > 0:
        raise ValueError(f"{path}: median signed incoming row sum {S} <= 0; the anchor formula does not apply")
    geom = 1.0 / (1.0 - math.exp(-sc.dt_ms / sc.tau_syn_ms))
    return S, 1000.0 / (geom * S * cc.stage_a_center_rate_hz * sc.dt_ms)


def main():
    cc, ac, shc = CalibConfig(), AudioConfig(), ShuffleConfig()
    cal = json.loads((CACHE / "calibration.json").read_text())
    sc = replace(SimConfig(), **{k: cal[k] for k in CALIBRATED_KEYS})
    neurons, indices = load_neurons_and_indices()
    N, n_sens = len(neurons), len(indices["sensory_idx"])
    drive = cc.drive_rheobase_mult * rheobase(sc)
    tmin, tmax = cc.target_rate_hz
    graphs = [("real", CACHE / "adjacency.npz")] + [(f"{c} seed {s}", shuffle_path(c, s))
                                                     for c in shc.regime_conditions for s in shc.regime_graph_seeds]
    header("MATCHED REGIME - does a working window exist at the calibrated g_inh?")
    print(f"PREDICTION: the question is whether degree-matched random graphs have a low-rate input-driven window at "
          f"g_inh {cal['g_inh']} at all (the real graph is the positive control)")
    print(f"fixed: {sc.normalization}, g_inh {cal['g_inh']}, b_adapt {cal['b_adapt']}, sigma {cal['sigma_noise']:.4g}, "
          f"refrac {sc.refrac_steps}; drive {drive:g}; {cc.stage_a_points} points x {cc.stage_a_ms:g} ms after "
          f"{cc.warmup_ms:g} ms warm-up; target rate {tmin:g}-{tmax:g} Hz, sensory > "
          f"{cc.refine_min_sensory_preservation:.0%} of w=0, input-driven: no-drive non-sensory rate <= {ac.ignition_rate_hz:g} Hz")

    dummy = np.ones(N, dtype=bool)
    ref_runner = Runner(sc, dummy)
    ref, _ = ref_runner.measure(cal["normalization"], cal["g_inh"], cal["b_adapt"], 0.0, cal["sigma_noise"], drive,
                                cc.warmup_ms, cc.stage_a_ms, exempt=cal["sensory_adapt_exempt"])
    ref_sens = ref["sensory_rate_hz"]
    del ref_runner
    print(f"sensory reference (w_scale 0, graph-independent): {ref_sens:.4g} Hz")

    out = {"prediction": "degree-matched random graphs may have no low-rate input-driven window at the calibrated g_inh",
           "fixed": {**{k: cal[k] for k in CALIBRATED_KEYS}, "drive": drive}, "sensory_reference_hz": ref_sens, "graphs": {}}
    for name, path in graphs:
        S, a10 = anchor(path, sc, cc)
        ws = a10 * np.logspace(-cc.stage_a_log10_span, cc.stage_a_log10_span, cc.stage_a_points)
        print(f"\n{name}: S {S:.4g}, w_anchor({cc.stage_a_center_rate_hz:g} Hz) {a10:.4g}; w_scale {ws[0]:.3g} .. {ws[-1]:.3g}", flush=True)
        runner = Runner(sc, dummy, adjacency_path=path)
        rows = []
        for w in ws:
            m, _ = runner.measure(cal["normalization"], cal["g_inh"], cal["b_adapt"], w, cal["sigma_noise"], drive,
                                  cc.warmup_ms, cc.stage_a_ms, exempt=cal["sensory_adapt_exempt"])
            pres = m["sensory_rate_hz"] / ref_sens
            row = {"w_scale": float(w), "w_over_anchor": float(w / a10), "rate_hz": m["rate_hz"], "motor_rate_hz": m["motor_rate_hz"],
                   "sensory_rate_hz": m["sensory_rate_hz"], "preservation": pres, "rate_per_active_hz": m["rate_per_active_hz"],
                   "active_frac": m["active_frac"], "rate_ok": tmin <= m["rate_hz"] <= tmax,
                   "preservation_ok": pres > cc.refine_min_sensory_preservation, "no_drive_nonsensory_hz": None,
                   "input_driven": None}
            if row["rate_ok"] and row["preservation_ok"]:
                m0, _ = runner.measure(cal["normalization"], cal["g_inh"], cal["b_adapt"], w, cal["sigma_noise"], 0.0,
                                       cc.warmup_ms, cc.stage_a_ms, exempt=cal["sensory_adapt_exempt"])
                ns = (m0["rate_hz"] * N - m0["sensory_rate_hz"] * n_sens) / (N - n_sens)
                row["no_drive_nonsensory_hz"] = ns
                row["input_driven"] = ns <= ac.ignition_rate_hz
            row["passes"] = bool(row["rate_ok"] and row["preservation_ok"] and row["input_driven"])
            rows.append(row)
            nd = "" if row["no_drive_nonsensory_hz"] is None else f" no-drive {row['no_drive_nonsensory_hz']:.3g} Hz"
            print(f"  w {w:<8.4g} ({w / a10:<6.3g}x) rate {m['rate_hz']:<8.4g} motor {m['motor_rate_hz']:<7.4g} sensory "
                  f"{m['sensory_rate_hz']:<6.4g} ({pres:.0%}) per-active {m['rate_per_active_hz']:<6.4g}{nd} -> "
                  f"{'PASS' if row['passes'] else 'fail: ' + ', '.join(k for k, ok in (('rate', row['rate_ok']), ('preservation', row['preservation_ok']), ('input-driven', row['input_driven'])) if ok is False)}" + ("" if row["input_driven"] is not None else " (input-driven not tested)"),
                  flush=True)
        del runner
        passing = [r for r in rows if r["passes"]]
        rates = [r["rate_hz"] for r in rows]
        in_band = [r for r in rows if r["rate_ok"]]
        out["graphs"][name] = {"adjacency": str(path), "S": S, "w_anchor_10hz": a10, "points": rows, "window": bool(passing),
                               "n_passing": len(passing), "n_rate_in_band": len(in_band),
                               "silent_to_active_jump": {"max_rate_below_band": max([x for x in rates if x < tmin], default=None),
                                                         "min_rate_above_band": min([x for x in rates if x > tmax], default=None)}}
        print(f"  {name}: {'WINDOW EXISTS' if passing else 'NO WINDOW'} ({len(passing)} passing points, {len(in_band)} in the rate band)")

    OUT_JSON.write_text(json.dumps(out, indent=2))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    colors = [TEXT] + list(CATEGORICAL) + ["#e87ba4"]
    for (name, _), color in zip(graphs, colors):
        pts = out["graphs"][name]["points"]
        x = [p["w_over_anchor"] for p in pts]
        y = [max(p["rate_hz"], 1e-3) for p in pts]
        ax.plot(x, y, marker="o", ms=4, linewidth=2 if name == "real" else 1.4, color=color, label=name)
        for p in pts:
            if p["passes"]:
                ax.plot(p["w_over_anchor"], p["rate_hz"], "o", ms=11, mfc="none", mec=color, mew=1.5)
    ax.axhspan(tmin, tmax, color=GRID, alpha=0.6, label=f"target rate {tmin:g}-{tmax:g} Hz")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    style(ax, f"Population rate vs w_scale / own anchor at g_inh {cal['g_inh']} (rings: all three target criteria pass)",
          "w_scale / w_anchor(10 Hz) of that graph", "population rate (Hz)")
    save(fig, "matched_regime.png")

    real_ok = out["graphs"]["real"]["window"]
    shuffled = {k: v for k, v in out["graphs"].items() if k != "real"}
    n_win = sum(v["window"] for v in shuffled.values())
    if not real_ok:
        verdict = ("INCONCLUSIVE: the protocol did not find a window for the REAL graph either, so the absence of a window in "
                   "shuffled graphs cannot be interpreted.")
    elif n_win == 0:
        verdict = (f"NO degree-matched shuffled graph ({len(shuffled)} tested) has a working window at g_inh {cal['g_inh']}, while the "
                   "real graph does under the identical protocol. The connectome's topology admits a stable low-rate, input-driven "
                   "state at this inhibitory gain that degree-, weight- and sign-matched random graphs do not.")
    elif n_win == len(shuffled):
        verdict = (f"Every shuffled graph ({n_win} of {len(shuffled)}) has a working window at g_inh {cal['g_inh']}: the low-rate regime "
                   "is not specific to the connectome's topology; the earlier comparison was a calibration mismatch. Control 1b applies.")
    else:
        verdict = (f"{n_win} of {len(shuffled)} shuffled graphs have a working window at g_inh {cal['g_inh']}: mixed result. Control 1b "
                   "applies to those that do.")
    out["verdict"] = verdict
    OUT_JSON.write_text(json.dumps(out, indent=2))

    md = ["## Control 1a: does a working window exist for degree-matched shuffled graphs?", "",
          f"**Prediction, stated before running:** the real graph needed g_inh = {cal['g_inh']} to open a window that did not exist "
          "at g_inh = 1; the question is whether degree-matched random graphs have such a window at the same g_inh at all.", "",
          "Method: `experiments/matched_regime.py`. Identical protocol for every graph, the real graph included as positive control: "
          f"g_inh {cal['g_inh']}, noise sigma {cal['sigma_noise']:.3g}, drive {drive:g}, {cc.stage_a_points} w_scale points over each "
          f"graph's own w_anchor(10 Hz) x 10^[-{cc.stage_a_log10_span:g}, +{cc.stage_a_log10_span:g}], {cc.stage_a_ms:g} ms after "
          f"{cc.warmup_ms:g} ms warm-up, every point from rest. A point passes if (1) population rate {tmin:g}-{tmax:g} Hz, "
          f"(2) sensory rate > {cc.refine_min_sensory_preservation:.0%} of the w_scale = 0 reference ({ref_sens:.3g} Hz), (3) input-driven: "
          f"with the drive removed the non-sensory rate stays <= {ac.ignition_rate_hz:g} Hz.", "",
          "| graph | anchor S | w_anchor(10 Hz) | points in rate band | passing points | window | highest rate below band | lowest rate above band |",
          "|---|---|---|---|---|---|---|---|"]
    for name, g in out["graphs"].items():
        j = g["silent_to_active_jump"]
        below = "n/a" if j["max_rate_below_band"] is None else f"{j['max_rate_below_band']:.3g} Hz"
        above = "n/a" if j["min_rate_above_band"] is None else f"{j['min_rate_above_band']:.3g} Hz"
        window = "**yes**" if g["window"] else "**no**"
        md.append(f"| {name} | {g['S']:.3g} | {g['w_anchor_10hz']:.3g} | {g['n_rate_in_band']} | {g['n_passing']} | "
                  f"{window} | {below} | {above} |")
    md += ["", "Per point (rate Hz / sensory preservation / no-drive non-sensory Hz where tested):", ""]
    for name, g in out["graphs"].items():
        md.append(f"- **{name}**: " + "; ".join(
            f"{p['w_over_anchor']:.3g}x: {p['rate_hz']:.3g} / {p['preservation']:.0%}"
            + ("" if p["no_drive_nonsensory_hz"] is None else f" / {p['no_drive_nonsensory_hz']:.3g}")
            + (" PASS" if p["passes"] else "") for p in g["points"]))
    md += ["", "![matched regime](../figures/matched_regime.png)", "", f"**Result:** {verdict}", ""]
    OUT_MD.write_text("\n".join(md))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")
    print("RESULT: " + verdict)


if __name__ == "__main__":
    main()
