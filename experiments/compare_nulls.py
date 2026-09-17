"""Independent-shift (v1) vs joint-shift null, side by side, on cached spike records.

    .venv/bin/python -m experiments.compare_nulls

1. OU transmission run (cache/transmission_sim.npz, 60 s, 8 independent OU drives), its full layer set.
2. Satie test (c) section as originally analysed (absolute dBFS mapping, 83 s, 20 s), re-simulated
   deterministically into cache/audio_runs/ if absent; its test (c) layers.
Same spike records, same layer samples, same analysis; only TransmissionConfig.null_mode differs.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from config import AudioConfig, MidiConfig, Paths, TransmissionConfig
from data.groups import load_neurons_and_indices, ordered_groups, sensory_hop_distances
from experiments.audio_run import audio_layers, run_clip
from experiments.transmission import SIM_CACHE, analyze, build_layers
from sim.calibrate import header

CACHE = ROOT / Paths().cache
SATIE = ROOT / "cache/test_audio/Gymnopedie_No_1.flac"
SATIE_OFFSET = 83.0      # the section test (c) used before the mapping change


def both_nulls(counts, signals, layers, tc):
    out = {}
    for mode in ("independent", "joint"):
        t = replace(tc, null_mode=mode)
        out[mode], _ = analyze(counts, signals, layers, t, np.random.default_rng(t.seed + 2))
    return out


def table(title, res, tc):
    header(title)
    print(f"{'layer':<22} {'units':>5} | {'|r| z indep':>11} {'joint':>7} | {'>null indep':>11} {'joint':>6} | "
          f"{'MI z indep':>10} {'joint':>7} | {'>null indep':>11} {'joint':>6} | verdict (|r| or MI) indep -> joint")

    def above(s):
        return s["z"] is not None and s["z"] >= tc.null_z_min and s["mean_best"] > s["null_max"]

    rows = {}
    for name in res["independent"]:
        a, b = res["independent"][name], res["joint"][name]
        z = lambda s: "n/a" if s["z"] is None else f"{s['z']:.2f}"
        va = above(a["abs_r"]) or above(a["mi"])
        vb = above(b["abs_r"]) or above(b["mi"])
        print(f"{name:<22} {a['n_units']:>5} | {z(a['abs_r']):>11} {z(b['abs_r']):>7} | "
              f"{a['abs_r']['frac_units_above_own_null_max']:>11.1%} {b['abs_r']['frac_units_above_own_null_max']:>6.1%} | "
              f"{z(a['mi']):>10} {z(b['mi']):>7} | {a['mi']['frac_units_above_own_null_max']:>11.1%} "
              f"{b['mi']['frac_units_above_own_null_max']:>6.1%} | {va} -> {vb}")
        rows[name] = {"independent": a, "joint": b, "above_independent": va, "above_joint": vb}
    return rows


def main():
    tc = TransmissionConfig()
    neurons, indices = load_neurons_and_indices()
    sens, motor = np.asarray(indices["sensory_idx"]), np.asarray(indices["motor_idx"])
    dist = sensory_hop_distances(indices)

    d = np.load(SIM_CACHE)
    counts, drive = d["counts"], d["drive_binned"]
    total = counts.sum(axis=0, dtype=np.int64)
    rates = total / (counts.shape[0] * tc.bin_ms / 1e3)
    in_groups = ordered_groups(neurons, sens, AudioConfig().n_bins)
    motor_groups = ordered_groups(neurons, motor, MidiConfig().n_groups)
    layers, _ = build_layers(counts, rates, total, sens, motor, dist, in_groups, motor_groups, tc,
                             np.random.default_rng(tc.seed + 1), verbose=False)
    ou = table("OU transmission run (60 s, independent OU drives): independent vs joint null",
               both_nulls(counts, drive, layers, tc), tc)

    run = run_clip(SATIE, SATIE_OFFSET, "absolute")
    sat_layers = audio_layers(run, tc, verbose=False)
    satie = table(f"Satie test (c), absolute mapping, {SATIE_OFFSET:g} s + {run['meta']['seconds']:g} s: independent vs joint null",
                  both_nulls(run["counts"], run["signals"], sat_layers, tc), tc)
    c = np.corrcoef(run["signals"], rowvar=False)
    print(f"Satie section band-band correlation: mean {c[~np.eye(8, dtype=bool)].mean():.3f}")
    out = CACHE / "null_comparison.json"
    out.write_text(json.dumps({"ou": ou, "satie_absolute_83s": satie}, indent=2, default=str))
    print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
