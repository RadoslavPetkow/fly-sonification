"""Control 1a, refinement: is there a window narrower than one matched_regime grid step?

    .venv/bin/python -m experiments.transition_refine
    .venv/bin/python -m experiments.transition_refine --figure-only    # redraw from cache/transition_refine.json

The matched_regime grid steps w_scale by 10^(2 * stage_a_log10_span / (stage_a_points - 1)) = 1.78x; the shuffled
graphs jump from < 1 Hz to 24-27 Hz between two neighbouring points, so a narrower window was not excluded.
For every graph in cache/matched_regime.json (real graph = positive control) the TRANSITION STEP is the first pair
of neighbouring grid points whose population rate goes from below CalibConfig.target_rate_hz[0] to at or above it.
That step is refined into ShuffleConfig.transition_refine_points geometrically spaced w_scale values, endpoints
included (1.075x steps). Everything else is the matched_regime protocol, unchanged: calibrated normalization,
g_inh, noise, drive, warm-up, stage_a_ms per point, every point from rest, the same three pass conditions
(rate in band; sensory > refine_min_sensory_preservation of the w_scale = 0 reference stored in matched_regime.json;
no-drive non-sensory rate <= AudioConfig.ignition_rate_hz, tested only where the first two pass).
PREDICTION (stated before running): the shuffled graphs' transition is an ignition jump, not a graded rise; at
1.075x resolution each still goes from < 1 Hz to > 8 Hz between neighbouring points, so no shuffled point passes.
The real graph's transition is graded and its refined step contains passing points.
The re-measured endpoints are compared with the grid values (reproducibility check).
Writes cache/transition_refine.json, figures/transition_refine.png, experiments/transition_refine.md.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from audio.audio_in import CALIBRATED_KEYS
from config import AudioConfig, CalibConfig, Paths, ShuffleConfig, SimConfig
from data.groups import load_neurons_and_indices
from sim.calibrate import CATEGORICAL, GRID, TEXT, TEXT_2, Runner, header, save, style

CACHE = ROOT / Paths().cache
IN_JSON = CACHE / "matched_regime.json"
OUT_JSON = CACHE / "transition_refine.json"
OUT_MD = ROOT / "experiments" / "transition_refine.md"


def transition_step(points, tmin):
    for lo, hi in zip(points[:-1], points[1:]):
        if lo["rate_hz"] < tmin <= hi["rate_hz"]:
            return lo, hi
    raise ValueError(f"no grid step crosses {tmin} Hz from below")


def plot(out, tmin, tmax, k, step):
    names = list(out["graphs"])
    fig, axes = plt.subplots(1, len(names), figsize=(3.1 * len(names), 3.6), sharey=True)
    colors = [TEXT if n == "real" else CATEGORICAL[ShuffleConfig().regime_conditions.index(n.split(" seed")[0])] for n in names]
    for ax, name, color in zip(axes, names, colors):
        g = out["graphs"][name]
        x = [p["w_over_anchor"] for p in g["points"]]
        y = [max(p["rate_hz"], 1e-3) for p in g["points"]]
        ax.axhspan(tmin, tmax, color=GRID, alpha=0.6)
        ax.plot(x, y, marker="o", ms=4, linewidth=1.6, color=color)
        for p in g["points"]:
            if p["passes"]:
                ax.plot(p["w_over_anchor"], p["rate_hz"], "o", ms=11, mfc="none", mec=color, mew=1.5)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2g"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xticks(np.geomspace(x[0], x[-1], 3))
        ax.grid(True, color=GRID, linewidth=0.6)
        style(ax, f"{name}: {g['n_passing']} pass", "w_scale / own anchor", "population rate (Hz)" if ax is axes[0] else "")
    fig.suptitle(f"Refined transition step, {k} points ({step:.3f}x); band = target rate, rings = all criteria pass",
                 color=TEXT, fontsize=10, x=0.01, ha="left")
    save(fig, "transition_refine.png")


def main():
    cc, ac, shc = CalibConfig(), AudioConfig(), ShuffleConfig()
    cal = json.loads((CACHE / "calibration.json").read_text())
    sc = replace(SimConfig(), **{k: cal[k] for k in CALIBRATED_KEYS})
    mr = json.loads(IN_JSON.read_text())
    neurons, indices = load_neurons_and_indices()
    N, n_sens = len(neurons), len(indices["sensory_idx"])
    drive, ref_sens = mr["fixed"]["drive"], mr["sensory_reference_hz"]
    tmin, tmax = cc.target_rate_hz
    k = shc.transition_refine_points
    header("TRANSITION REFINEMENT - a window narrower than one grid step?")
    print(f"fixed: {sc.normalization}, g_inh {cal['g_inh']}, sigma {cal['sigma_noise']:.4g}, drive {drive:g}; {k} points per "
          f"transition step, {cc.stage_a_ms:g} ms after {cc.warmup_ms:g} ms warm-up, from rest; pass: rate {tmin:g}-{tmax:g} Hz, "
          f"sensory > {cc.refine_min_sensory_preservation:.0%} of {ref_sens:.4g} Hz, no-drive non-sensory <= {ac.ignition_rate_hz:g} Hz")
    print("PREDICTION: the shuffled graphs still jump from < 1 Hz to > 8 Hz between neighbouring refined points (no shuffled "
          "point passes); the real graph's transition is graded and contains passing points")
    dummy = np.ones(N, dtype=bool)
    out = {"prediction": "shuffled graphs still jump across the band at 1.075x resolution; real graph graded", "source": str(IN_JSON.relative_to(ROOT)), "points_per_step": k, "graphs": {}}
    for name, g in mr["graphs"].items():
        lo, hi = transition_step(g["points"], tmin)
        ws = np.geomspace(lo["w_scale"], hi["w_scale"], k)
        ratio = float((hi["w_scale"] / lo["w_scale"]) ** (1 / (k - 1)))
        a10 = g["w_anchor_10hz"]
        print(f"\n{name}: transition step {lo['w_over_anchor']:.3g}x ({lo['rate_hz']:.3g} Hz) -> {hi['w_over_anchor']:.3g}x "
              f"({hi['rate_hz']:.3g} Hz); w_scale {ws[0]:.4g} .. {ws[-1]:.4g}, step {ratio:.4f}x", flush=True)
        runner = Runner(sc, dummy, adjacency_path=Path(g["adjacency"]))
        rows = []
        for w in ws:
            m, _ = runner.measure(cal["normalization"], cal["g_inh"], cal["b_adapt"], w, cal["sigma_noise"], drive,
                                  cc.warmup_ms, cc.stage_a_ms, exempt=cal["sensory_adapt_exempt"])
            pres = m["sensory_rate_hz"] / ref_sens
            row = {"w_scale": float(w), "w_over_anchor": float(w / a10), "rate_hz": m["rate_hz"], "motor_rate_hz": m["motor_rate_hz"],
                   "sensory_rate_hz": m["sensory_rate_hz"], "preservation": pres, "active_frac": m["active_frac"],
                   "rate_ok": tmin <= m["rate_hz"] <= tmax, "preservation_ok": pres > cc.refine_min_sensory_preservation,
                   "no_drive_nonsensory_hz": None, "input_driven": None}
            if row["rate_ok"] and row["preservation_ok"]:
                m0, _ = runner.measure(cal["normalization"], cal["g_inh"], cal["b_adapt"], w, cal["sigma_noise"], 0.0,
                                       cc.warmup_ms, cc.stage_a_ms, exempt=cal["sensory_adapt_exempt"])
                ns = (m0["rate_hz"] * N - m0["sensory_rate_hz"] * n_sens) / (N - n_sens)
                row["no_drive_nonsensory_hz"] = ns
                row["input_driven"] = ns <= ac.ignition_rate_hz
            row["passes"] = bool(row["rate_ok"] and row["preservation_ok"] and row["input_driven"])
            rows.append(row)
            failed = [c for c, ok in (("rate", row["rate_ok"]), ("preservation", row["preservation_ok"]),
                                      ("input-driven", row["input_driven"])) if ok is False]
            nd = "" if row["no_drive_nonsensory_hz"] is None else f" no-drive {row['no_drive_nonsensory_hz']:.3g} Hz"
            print(f"  w {w:<8.4g} ({w / a10:<6.4g}x) rate {m['rate_hz']:<8.4g} motor {m['motor_rate_hz']:<7.4g} sensory "
                  f"{m['sensory_rate_hz']:<6.4g} ({pres:.0%}) active {m['active_frac']:.3f}{nd} -> "
                  + ("PASS" if row["passes"] else "fail: " + ", ".join(failed)
                     + ("" if row["input_driven"] is not None else " (input-driven not tested)")), flush=True)
        del runner
        rates = [r["rate_hz"] for r in rows]
        below = [x for x in rates if x < tmin]
        above = [x for x in rates if x > tmax]
        passing = [r for r in rows if r["passes"]]
        jump = None
        for a, b in zip(rows[:-1], rows[1:]):
            if a["rate_hz"] < tmin <= b["rate_hz"]:
                jump = {"from_w_over_anchor": a["w_over_anchor"], "from_rate_hz": a["rate_hz"],
                        "to_w_over_anchor": b["w_over_anchor"], "to_rate_hz": b["rate_hz"]}
                break
        out["graphs"][name] = {
            "grid_step": {"lo": lo, "hi": hi}, "step_ratio": ratio, "points": rows, "n_passing": len(passing),
            "n_rate_in_band": sum(r["rate_ok"] for r in rows), "window": bool(passing),
            "max_rate_below_band": max(below, default=None), "min_rate_above_band": min(above, default=None),
            "fine_jump": jump,
            "endpoint_reproduction": {"lo_grid_hz": lo["rate_hz"], "lo_refined_hz": rows[0]["rate_hz"],
                                      "hi_grid_hz": hi["rate_hz"], "hi_refined_hz": rows[-1]["rate_hz"]}}
        print(f"  {name}: {len(passing)} passing, {out['graphs'][name]['n_rate_in_band']} in the rate band; endpoints grid "
              f"{lo['rate_hz']:.4g} / {hi['rate_hz']:.4g} Hz, refined {rows[0]['rate_hz']:.4g} / {rows[-1]['rate_hz']:.4g} Hz")

    shuffled = {n: g for n, g in out["graphs"].items() if n != "real"}
    real = out["graphs"]["real"]
    n_win = sum(g["window"] for g in shuffled.values())
    step = max(g["step_ratio"] for g in out["graphs"].values())
    if not real["window"]:
        verdict = "INCONCLUSIVE: the refinement found no passing point for the real graph either."
    elif n_win == 0:
        verdict = (f"The claim SURVIVES at {step:.3f}x resolution: no point of any of the {len(shuffled)} shuffled graphs passes; "
                   f"the real graph has {real['n_passing']} passing points in its refined transition step.")
    else:
        names = ", ".join(f"{n} ({g['n_passing']} points)" for n, g in shuffled.items() if g["window"])
        verdict = (f"The claim does NOT survive as stated: {n_win} of {len(shuffled)} shuffled graphs have a passing point at "
                   f"{step:.3f}x resolution: {names}. At this resolution these are narrow windows, not absent ones.")
    out["verdict"] = verdict
    OUT_JSON.write_text(json.dumps(out, indent=2))

    plot(out, tmin, tmax, k, step)

    md = ["## Control 1a, refinement: a window narrower than one grid step?", "",
          f"`experiments/transition_refine.py`. For each graph the grid step in which the rate first reaches {tmin:g} Hz from below "
          f"was re-sampled at {k} geometrically spaced w_scale values, endpoints included ({step:.3f}x steps), with the "
          "matched_regime protocol and pass conditions unchanged (every point from rest). The real graph is the positive control; "
          "its refined step ends at a grid point that already passed, so its passing there is expected, and what the control adds "
          "is the shape of its transition.", "",
          "| graph | grid step (x own anchor) | points in rate band | passing points | highest rate below band | lowest rate above band | sharpest crossing (fine) | endpoints grid -> refined (Hz) |",
          "|---|---|---|---|---|---|---|---|"]
    for name, g in out["graphs"].items():
        j, e = g["fine_jump"], g["endpoint_reproduction"]
        fmt = lambda v: "n/a" if v is None else f"{v:.3g} Hz"
        cross = "n/a" if j is None else (f"{j['from_rate_hz']:.3g} Hz @ {j['from_w_over_anchor']:.3g}x -> "
                                          f"{j['to_rate_hz']:.3g} Hz @ {j['to_w_over_anchor']:.3g}x")
        md.append(f"| {name} | {g['grid_step']['lo']['w_over_anchor']:.3g} - {g['grid_step']['hi']['w_over_anchor']:.3g} | "
                  f"{g['n_rate_in_band']} | **{g['n_passing']}** | {fmt(g['max_rate_below_band'])} | {fmt(g['min_rate_above_band'])} | "
                  f"{cross} | {e['lo_grid_hz']:.3g} -> {e['lo_refined_hz']:.3g}; {e['hi_grid_hz']:.3g} -> {e['hi_refined_hz']:.3g} |")
    md += ["", "Rate curve across the step (x own anchor: rate Hz / sensory preservation / no-drive non-sensory Hz where tested):", ""]
    for name, g in out["graphs"].items():
        md.append(f"- **{name}**: " + "; ".join(
            f"{p['w_over_anchor']:.4g}x: {p['rate_hz']:.3g} / {p['preservation']:.0%}"
            + ("" if p["no_drive_nonsensory_hz"] is None else f" / {p['no_drive_nonsensory_hz']:.3g}")
            + (" PASS" if p["passes"] else "") for p in g["points"]))
    md += ["", "![transition refinement](../figures/transition_refine.png)", "", f"**Result:** {verdict}", ""]
    OUT_MD.write_text("\n".join(md))
    print(f"\nwrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")
    print("RESULT: " + verdict)


if __name__ == "__main__":
    if "--figure-only" in sys.argv[1:]:
        saved = json.loads(OUT_JSON.read_text())
        tmin, tmax = CalibConfig().target_rate_hz
        plot(saved, tmin, tmax, saved["points_per_step"], max(g["step_ratio"] for g in saved["graphs"].values()))
    else:
        main()
