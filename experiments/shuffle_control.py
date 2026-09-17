"""Does the real connectome produce a different output from a degree-preserving random graph?

    .venv/bin/python -m experiments.shuffle_control [--report-only]

Design (config.ShuffleConfig), fixed before running:
  materials   ShuffleConfig.clips; per clip up to sections_per_clip replicate sections from
              experiments.audio_run.choose_sections (audio only, decided before simulating)
  conditions  full / sensory_only / downstream_only degree-, weight- and sign-preserving shuffles
              (experiments/shuffle_graph.py), seeds ShuffleConfig.seeds, one matrix per (condition, seed)
  runs        each section through the real matrix and through every shuffled matrix, identical drive,
              calibration and noise seed (paired); cached in cache/audio_runs/
MEASURES per run, identical for real and shuffled:
  transmission   hop-1 motor neurons OF THAT GRAPH (primary) and the fixed voice set (the real hop-1 motor
                 neurons): mean best-band |r| and MI minus their joint-shift null mean ("excess"), and z
  dynamics       on the 12 type-ordered groups of all motor neurons (MidiConfig.n_groups; fixed identity,
                 active in every graph): median autocorrelation timescale (first lag with ACF < 1/e), largest
                 eigenvalue fraction and participation ratio of the group correlation matrix, median spectral
                 flatness (Welch PSD without DC)
STATISTICS per measure and condition, over sections s and seeds k:
  effect D = mean_s (real_s - mean_k shuffled_{s,k})
  95% CI   bootstrap: sections resampled with replacement, seeds resampled within each section
  p        exact permutation test: within each section the "real" label is reassigned among its 1 + n_seeds
           runs, all (1 + n_seeds) ** n_sections assignments; two-sided on |D|
  size     D / pooled within-section SD of the shuffled runs
  verdict  "differs" only if p < alpha AND the CI excludes 0
Writes cache/shuffle_control_results.json, figures/control_*.png and experiments/control_report.md.
"""
import argparse
import itertools
import json
import math
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from scipy.signal import welch

from config import AudioConfig, MidiConfig, Paths, ShuffleConfig, TransmissionConfig
from data.fetch_connectome import bfs_levels
from data.groups import load_neurons_and_indices, ordered_groups
from experiments.audio_run import choose_sections, run_clip, run_path
from experiments.shuffle_graph import build_shuffle, shuffle_path
from experiments.transmission import analyze
from sim.calibrate import CATEGORICAL, GRID, SURFACE, TEXT, TEXT_2, header, save, style

CACHE = ROOT / Paths().cache
RESULTS = CACHE / "shuffle_control_results.json"
REPORT = ROOT / "experiments" / "control_report.md"

MEASURES = [
    ("excess_abs_r_own", "|r| excess over null, hop-1 motor of the graph (primary)"),
    ("excess_mi_own", "MI excess over null (bits), hop-1 motor of the graph (primary)"),
    ("z_abs_r_own", "|r| z, hop-1 motor of the graph"),
    ("z_mi_own", "MI z, hop-1 motor of the graph"),
    ("excess_abs_r_voices", "|r| excess, fixed voice set (real hop-1 motor)"),
    ("excess_mi_voices", "MI excess (bits), fixed voice set"),
    ("acf_tau_ms", "autocorrelation timescale of motor group rates (ms)"),
    ("eig_lambda1_frac", "largest eigenvalue fraction, motor group correlation"),
    ("eig_participation", "participation ratio, motor group correlation"),
    ("spectral_flatness", "spectral flatness of motor group rates"),
    ("population_rate_hz", "whole-network population rate (Hz)"),
    ("motor_rate_hz", "motor population rate (Hz)"),
]


# =============================================================================
# measures
# =============================================================================

def acf_timescale_ms(x, tick_ms, max_lag_ms):
    x = x - x.mean()
    if not np.any(x):
        return None
    max_lag = min(round(max_lag_ms / tick_ms), x.size - 1)
    denom = np.dot(x, x)
    acf = np.array([np.dot(x[:x.size - k], x[k:]) / denom for k in range(max_lag + 1)])
    below = np.flatnonzero(acf < 1 / math.e)
    if below.size == 0:
        return float(max_lag * tick_ms)
    k = below[0]
    frac = (acf[k - 1] - 1 / math.e) / (acf[k - 1] - acf[k])
    return float((k - 1 + frac) * tick_ms)


def spectral_flatness(x, tick_ms, nperseg):
    if not np.any(x - x.mean()):
        return None
    _, p = welch(x - x.mean(), fs=1e3 / tick_ms, nperseg=min(nperseg, x.size))
    p = p[1:]
    p = p[p > 0]
    return float(np.exp(np.mean(np.log(p))) / np.mean(p)) if p.size else None


def run_measures(run, own_hop1_motor, voices, motor_groups, n_neurons, tc, sc):
    counts, signals = run["counts"], run["signals"]
    T = counts.shape[0]
    tick_s = tc.bin_ms / 1e3
    total = counts.sum(axis=0)
    out = {"population_rate_hz": float(total.sum() / n_neurons / (T * tick_s))}
    layers = {}
    for name, idx in (("own", own_hop1_motor), ("voices", voices)):
        active = idx[total[idx] >= tc.min_spikes]
        out[f"n_active_{name}"] = int(active.size)
        out[f"n_{name}"] = int(idx.size)
        if active.size >= 2:
            layers[name] = counts[:, active].T
    res, _ = analyze(counts, signals, layers, tc, np.random.default_rng(tc.seed + 2)) if layers else ({}, None)
    for name in ("own", "voices"):
        for key, short in (("abs_r", "abs_r"), ("mi", "mi")):
            s = res.get(name, {}).get(key)
            out[f"excess_{short}_{name}"] = None if s is None else s["mean_best"] - s["null_mean"]
            out[f"z_{short}_{name}"] = None if s is None else s["z"]
            out[f"frac_above_{short}_{name}"] = None if s is None else s["frac_units_above_own_null_max"]
    motor_all = np.concatenate(motor_groups)
    out["motor_rate_hz"] = float(total[motor_all].sum() / motor_all.size / (T * tick_s))
    R = np.stack([counts[:, g].sum(axis=1) / len(g) / tick_s for g in motor_groups], axis=1)
    taus = [acf_timescale_ms(R[:, g], tc.bin_ms, sc.acf_max_lag_ms) for g in range(R.shape[1])]
    flats = [spectral_flatness(R[:, g], tc.bin_ms, sc.psd_nperseg) for g in range(R.shape[1])]
    taus, flats = [t for t in taus if t is not None], [f for f in flats if f is not None]
    out["acf_tau_ms"] = float(np.median(taus)) if taus else None
    out["spectral_flatness"] = float(np.median(flats)) if flats else None
    live = R[:, R.std(axis=0) > 0]
    out["n_live_motor_groups"] = int(live.shape[1])
    if live.shape[1] >= 2:
        ev = np.sort(np.linalg.eigvalsh(np.corrcoef(live, rowvar=False)))[::-1]
        out["eig_spectrum"] = ev.tolist()
        out["eig_lambda1_frac"] = float(ev[0] / ev.sum())
        out["eig_participation"] = float(ev.sum() ** 2 / np.sum(ev ** 2))
    else:
        out["eig_spectrum"], out["eig_lambda1_frac"], out["eig_participation"] = None, None, None
    return out


# =============================================================================
# statistics
# =============================================================================

def paired_stats(real, shuffled, n_boot, alpha, rng):
    """real: {section: value}; shuffled: {section: [values over seeds]} (None dropped)."""
    secs = [s for s in real if real[s] is not None and len([v for v in shuffled.get(s, []) if v is not None]) >= 1]
    if len(secs) < 2:
        return {"n_sections": len(secs), "D": None}
    r = np.array([real[s] for s in secs], dtype=float)
    sh = [np.array([v for v in shuffled[s] if v is not None], dtype=float) for s in secs]
    d = r - np.array([x.mean() for x in sh])
    D = float(d.mean())
    boots = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(secs), size=len(secs))
        boots[b] = np.mean([r[i] - rng.choice(sh[i], size=sh[i].size, replace=True).mean() for i in pick])
    ci = [float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))]
    pools = [np.concatenate([[r[i]], sh[i]]) for i in range(len(secs))]
    stats = []
    for choice in itertools.product(*[range(p.size) for p in pools]):
        stats.append(np.mean([p[c] - np.delete(p, c).mean() for p, c in zip(pools, choice)]))
    stats = np.abs(np.array(stats))
    p = float(np.mean(stats >= abs(D) - 1e-12))
    within = [x.var(ddof=1) for x in sh if x.size >= 2]
    sd = float(np.sqrt(np.mean(within))) if within else None
    return {"n_sections": len(secs), "sections": secs, "D": D, "ci95": ci, "p_perm": p,
            "n_permutations": int(stats.size), "effect_size": (D / sd) if sd else None, "pooled_shuffle_sd": sd,
            "per_section_diff": dict(zip(secs, d.tolist())), "real_mean": float(r.mean()),
            "shuffled_mean": float(np.mean([x.mean() for x in sh])),
            "differs": bool(p < alpha and (ci[0] > 0 or ci[1] < 0))}


# =============================================================================
# figures
# =============================================================================

def figure_effects(results, sc):
    keys = ["excess_abs_r_own", "excess_mi_own", "excess_abs_r_voices", "acf_tau_ms", "eig_participation", "spectral_flatness"]
    labels = dict(MEASURES)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5))
    for ax, key in zip(axes.flat, keys):
        for ci_, cond in enumerate(sc.conditions):
            st = results["stats"][cond][key]
            x0 = ci_
            for s_i, (sec, runs) in enumerate(results["runs"].items()):
                real = runs["real"].get(key)
                for seed in sc.seeds:
                    v = runs["shuffled"].get(cond, {}).get(str(seed), {}).get(key)
                    if real is None or v is None:
                        continue
                    marker = "o" if "Gymnopedie" in sec else "s"
                    ax.plot(x0 + (s_i - 2) * 0.06, real - v, marker, ms=4, color=CATEGORICAL[ci_], alpha=0.55)
            if st.get("D") is not None:
                ax.errorbar([x0 + 0.25], [st["D"]], yerr=[[st["D"] - st["ci95"][0]], [st["ci95"][1] - st["D"]]],
                            fmt="D", color=TEXT, ms=5, capsize=3, linewidth=1)
        ax.axhline(0, color=TEXT_2, linewidth=0.8)
        ax.set_xticks(range(len(sc.conditions)), [c.replace("_", "\n") for c in sc.conditions])
        ax.grid(True, color=GRID, linewidth=0.6, axis="y")
        import textwrap
        style(ax, textwrap.fill(labels[key], 42), "", "real minus shuffled")
    fig.suptitle("Real minus shuffled per section and seed (circles Satie, squares kyuchek); diamond = mean with 95% "
                 "bootstrap CI", color=TEXT, fontsize=10, x=0.01, ha="left")
    save(fig, "control_effects.png")


def figure_spectra(results, sc):
    fig, axes = plt.subplots(1, len(sc.conditions), figsize=(4.4 * len(sc.conditions), 3.8), squeeze=False)
    for ax, cond in zip(axes[0], sc.conditions):
        real, shuf = [], []
        for sec, runs in results["runs"].items():
            if runs["real"].get("eig_spectrum") and len(runs["real"]["eig_spectrum"]) == MidiConfig().n_groups:
                real.append(runs["real"]["eig_spectrum"])
            for seed in sc.seeds:
                e = runs["shuffled"].get(cond, {}).get(str(seed), {}).get("eig_spectrum")
                if e and len(e) == MidiConfig().n_groups:
                    shuf.append(e)
        k = np.arange(1, MidiConfig().n_groups + 1)
        for arr, color, label in ((real, CATEGORICAL[0], f"real (n={len(real)})"), (shuf, CATEGORICAL[1], f"shuffled (n={len(shuf)})")):
            if arr:
                a = np.array(arr)
                ax.fill_between(k, a.min(axis=0), a.max(axis=0), color=color, alpha=0.15, linewidth=0)
                ax.plot(k, a.mean(axis=0), color=color, linewidth=2, marker="o", ms=3, label=label)
        ax.set_yscale("log")
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
        ax.grid(True, color=GRID, linewidth=0.6)
        style(ax, cond, "eigenvalue rank", "eigenvalue")
    fig.suptitle("Eigenvalue spectra of the 12 motor-group correlation matrices (mean line, min-max band over runs)",
                 color=TEXT, fontsize=10, x=0.01, ha="left")
    save(fig, "control_spectra.png")


# =============================================================================
# report
# =============================================================================

def fmt(v, nd=3):
    return "n/a" if v is None else (f"{v:.{nd}g}" if isinstance(v, float) else str(v))


def write_report(results, sc):
    lines = [f"# Shuffle control: real connectome vs degree-preserving random graphs",
             "", f"Generated {results['created']} by `experiments/shuffle_control.py`; raw numbers in "
             "`cache/shuffle_control_results.json`.", "",
             "## Design (fixed before running)", "",
             f"- Materials and replicate sections (chosen from the audio alone): " + "; ".join(
                 f"{Path(c).name}: " + ", ".join(f"{o:g}-{o + AudioConfig().test_clip_seconds:g} s" for o in offs)
                 for c, offs in results["sections"].items()),
             f"- Conditions: {', '.join(sc.conditions)}; seeds {list(sc.seeds)}; one shuffled matrix per condition and seed, "
             "the same matrix for every section.",
             "- Shuffle: each presynaptic neuron keeps its out-degree, outgoing weights and sign; targets are a permutation "
             "of the selected edges' target stubs (in-degrees kept), self-loops and duplicate pairs repaired by swaps. "
             "Verified after every shuffle.",
             "- Identical drive, calibration and noise seed for real and shuffled runs (paired).",
             "- Primary transmission measure: hop-1 motor neurons of each graph (its own BFS from the sensory set). The "
             "fixed voice set (the real graph's hop-1 motor neurons) is also reported; under a shuffle that moves the "
             "sensory edges those neurons are no longer necessarily direct targets.",
             "- Dynamics measures on the 12 type-ordered groups of all 1,322 motor neurons.",
             f"- Effect = mean over sections of (real - mean over seeds); 95% CI by section/seed bootstrap "
             f"({sc.n_bootstrap} resamples); exact permutation p over reassignments of the real label within sections; "
             f"effect size = effect / pooled within-section SD of the shuffled runs. \"Differs\" requires p < {sc.alpha} "
             "and a CI excluding 0. No correction for the number of measures is applied in the flag; see the table.",
             "", "## Budget and cuts", "", results["budget_text"], "",
             "## Shuffle verification", "", "| condition | seed | edges re-targeted | initial conflicts | repair iterations | targets unchanged | self-loops real -> shuffled |",
             "|---|---|---|---|---|---|---|"]
    for m in results["shuffles"]:
        lines.append(f"| {m['condition']} | {m['seed']} | {m['selected_edges']:,} | {m['conflicts_initial']:,} | "
                     f"{m['repair_iterations']} | {m['frac_targets_unchanged']:.2%} | {m['self_loops_real']} -> {m['self_loops_shuffled']} |")
    lines += ["", "All shuffles passed: in- and out-degree sequences unchanged, per-neuron outgoing weight multisets unchanged, "
              "one sign per presynaptic neuron, no duplicate pairs, unselected edges identical.", "",
              "## Hop-1 motor sets", "", "| graph | hop-1 motor neurons (own) | voices still at hop 1 |", "|---|---|---|"]
    for g, h in results["hop1"].items():
        lines.append(f"| {g} | {h['n_own']} | {h['voices_still_hop1']} of {h['n_voices']} |")
    lines += ["", "## Per-run values", "", "| section | graph | pop Hz | motor Hz | own hop-1 active | |r| excess own | MI excess own | "
              "|r| z own | MI z own | |r| excess voices | ACF tau ms | lambda1 frac | participation | flatness |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sec, runs in results["runs"].items():
        rows = [("real", runs["real"])] + [(f"{c} s{sd}", runs["shuffled"].get(c, {}).get(str(sd), {}))
                                          for c in sc.conditions for sd in sc.seeds]
        for label, m in rows:
            if not m:
                continue
            lines.append(f"| {sec} | {label} | {fmt(m.get('population_rate_hz'))} | {fmt(m.get('motor_rate_hz'))} | "
                         f"{m.get('n_active_own')} | {fmt(m.get('excess_abs_r_own'))} | {fmt(m.get('excess_mi_own'))} | "
                         f"{fmt(m.get('z_abs_r_own'))} | {fmt(m.get('z_mi_own'))} | {fmt(m.get('excess_abs_r_voices'))} | "
                         f"{fmt(m.get('acf_tau_ms'))} | {fmt(m.get('eig_lambda1_frac'))} | {fmt(m.get('eig_participation'))} | "
                         f"{fmt(m.get('spectral_flatness'))} |")
    lines += ["", "## Effects: real minus shuffled", ""]
    n_tests = len(sc.conditions) * len(MEASURES)
    for cond in sc.conditions:
        lines += [f"### {cond}", "", "| measure | real mean | shuffled mean | effect (real - shuffled) | 95% CI | permutation p | "
                  "effect size | sections | differs |", "|---|---|---|---|---|---|---|---|---|"]
        for key, label in MEASURES:
            st = results["stats"][cond][key]
            if st.get("D") is None:
                lines.append(f"| {label} | n/a | n/a | n/a | n/a | n/a | n/a | {st['n_sections']} | n/a |")
                continue
            lines.append(f"| {label} | {fmt(st['real_mean'])} | {fmt(st['shuffled_mean'])} | {st['D']:+.3g} | "
                         f"[{st['ci95'][0]:+.3g}, {st['ci95'][1]:+.3g}] | {st['p_perm']:.3g} (of {st['n_permutations']}) | "
                         f"{fmt(st['effect_size'])} | {st['n_sections']} | {'**yes**' if st['differs'] else 'no'} |")
        lines.append("")
    lines += [f"Bonferroni reference: {n_tests} tests -> alpha {sc.alpha / n_tests:.2g}. The smallest attainable permutation "
              f"p with these sections and seeds is {1 / results['min_permutations']:.3g}.", "",
              "## Figures", "", "![effects](../figures/control_effects.png)", "", "![spectra](../figures/control_spectra.png)", "",
              "## Verdict (mechanical, from the rule above)", "", results["verdict_auto"], ""]
    for extra in ("control_interpretation.md", "matched_regime.md", "transition_refine.md", "structural_selectivity.md", "partition_control.md"):
        path = ROOT / "experiments" / extra
        if path.exists():
            lines += [path.read_text()]
    REPORT.write_text("\n".join(lines))
    print(f"wrote {os.path.relpath(REPORT, ROOT)}")


def auto_verdict(results, sc):
    out = []
    for cond in sc.conditions:
        diff = [label for key, label in MEASURES if results["stats"][cond][key].get("differs")]
        prim = [results["stats"][cond][k] for k in ("excess_abs_r_own", "excess_mi_own")]
        prim_txt = "; ".join(f"{k}: effect {s['D']:+.3g} [{s['ci95'][0]:+.3g}, {s['ci95'][1]:+.3g}], p {s['p_perm']:.3g}"
                             for k, s in zip(("|r| excess", "MI excess"), prim) if s.get("D") is not None)
        out.append(f"- **{cond}**: " + (f"differs on {len(diff)} measure(s): {', '.join(diff)}." if diff else
                                        "statistically indistinguishable from the real connectome on every measure.")
                   + f" Primary transmission: {prim_txt}.")
    return "\n".join(out)


# =============================================================================
# main
# =============================================================================

def main(report_only=False):
    sc, tc, ac = ShuffleConfig(), TransmissionConfig(), AudioConfig()
    neurons, indices = load_neurons_and_indices()
    n = len(neurons)
    sens = np.asarray(indices["sensory_idx"])
    motor = np.asarray(indices["motor_idx"])
    motor_groups = ordered_groups(neurons, motor, MidiConfig().n_groups)
    W = sp.load_npz(CACHE / "adjacency.npz").tocsr()

    header("SECTIONS (audio only, before simulating)")
    sections = {clip: choose_sections(clip, sc.sections_per_clip, sc.max_section_overlap) for clip in sc.clips}
    jobs = [(clip, off) for clip, offs in sections.items() for off in offs]

    header("SHUFFLES")
    shuffles = []
    for cond in sc.conditions:
        for seed in sc.seeds:
            _, meta = build_shuffle(cond, seed, W=W, sensory_idx=sens)
            shuffles.append(meta)

    header("HOP-1 MOTOR SETS")
    real_dist = bfs_levels(W.T.tocsr(), sens)
    voices = motor[real_dist[motor] == 1]
    hop1 = {"real": {"n_own": int(voices.size), "voices_still_hop1": int(voices.size), "n_voices": int(voices.size), "idx": voices}}
    for cond in sc.conditions:
        for seed in sc.seeds:
            S = sp.load_npz(shuffle_path(cond, seed)).tocsr()
            d = bfs_levels(S.T.tocsr(), sens)
            own = motor[d[motor] == 1]
            hop1[f"{cond} s{seed}"] = {"n_own": int(own.size), "voices_still_hop1": int((d[voices] == 1).sum()),
                                       "n_voices": int(voices.size), "idx": own}
            print(f"  {cond} seed {seed}: {own.size} hop-1 motor neurons; {int((d[voices] == 1).sum())} of {voices.size} real voices still at hop 1")
    del W

    header("BUDGET")
    todo = []
    for clip, off in jobs:
        if not run_path(clip, off, ac.level_reference).exists():
            todo.append((clip, off, None, None))
        for cond in sc.conditions:
            for seed in sc.seeds:
                tag = f"shuffle-{cond}-seed{seed}"
                if not run_path(clip, off, ac.level_reference, tag).exists():
                    todo.append((clip, off, shuffle_path(cond, seed), tag))
    n_runs = len(jobs) * (1 + len(sc.conditions) * len(sc.seeds))
    min_per_run = 3.35
    est = len(todo) * min_per_run
    budget_text = (f"{len(jobs)} sections ({', '.join(f'{Path(c).name}: {len(o)}' for c, o in sections.items())}) x "
                   f"(1 real + {len(sc.conditions)} conditions x {len(sc.seeds)} seeds) = {n_runs} runs, {n_runs - len(todo)} already "
                   f"cached, {len(todo)} to simulate at ~{min_per_run} min each (20.2 s at ~9.3 ms/step plus loading) = "
                   f"~{est:.0f} min against a {sc.budget_minutes:g} min budget. Cut: a third shuffle seed (would add "
                   f"{len(jobs) * len(sc.conditions)} runs, ~{len(jobs) * len(sc.conditions) * min_per_run:.0f} min); a third kyuchek "
                   f"section is not available (a 37.6 s clip with 2 s margins cannot hold three 20 s sections overlapping by "
                   f"<= {sc.max_section_overlap:.0%}).")
    log = ROOT / Paths().runs / "shuffle_control.log"
    if report_only and log.exists():
        import re
        text = log.read_text()
        planned = re.search(r"^\d+ sections .*?budget\. Cut:.*$", text, re.M)
        elapsed = re.findall(r"elapsed ([0-9.]+) min", text)
        if planned:
            budget_text = planned.group(0)
        if elapsed:
            budget_text += (f" Actual simulation time of the main run: {float(elapsed[-1]):.1f} min "
                            f"({float(elapsed[-1]) / max(1, len(elapsed)):.2f} min per run, above the {min_per_run} min estimate).")
    print(budget_text)
    if not report_only and est > sc.budget_minutes:
        raise SystemExit("estimate exceeds the budget; reduce sections or seeds in ShuffleConfig")

    if not report_only:
        header(f"SIMULATING {len(todo)} runs")
        t0 = time.perf_counter()
        for i, (clip, off, adj, tag) in enumerate(todo):
            label = "real" if tag is None else tag
            print(f"[{i + 1}/{len(todo)}] {Path(clip).name} {off:g} s {label}", flush=True)
            run_clip(clip, off, ac.level_reference, adjacency_path=adj, tag=tag)
            el = (time.perf_counter() - t0) / 60
            print(f"  elapsed {el:.1f} min, ETA {el / (i + 1) * (len(todo) - i - 1):.1f} min", flush=True)

    header("MEASURES")
    runs = {}
    for clip, off in jobs:
        sec = f"{Path(clip).stem} {off:g}s"
        real = run_clip(clip, off, ac.level_reference)
        entry = {"real": run_measures(real, hop1["real"]["idx"], voices, motor_groups, n, tc, sc), "shuffled": {}}
        for cond in sc.conditions:
            entry["shuffled"][cond] = {}
            for seed in sc.seeds:
                tag = f"shuffle-{cond}-seed{seed}"
                r = run_clip(clip, off, ac.level_reference, adjacency_path=shuffle_path(cond, seed), tag=tag)
                entry["shuffled"][cond][str(seed)] = run_measures(r, hop1[f"{cond} s{seed}"]["idx"], voices, motor_groups, n, tc, sc)
        runs[sec] = entry
        m = entry["real"]
        print(f"  {sec}: real |r| excess {fmt(m['excess_abs_r_own'])} (z {fmt(m['z_abs_r_own'])}), MI excess "
              f"{fmt(m['excess_mi_own'])}; " + "; ".join(
                  f"{c}: |r| excess " + "/".join(fmt(entry['shuffled'][c][str(s)]['excess_abs_r_own']) for s in sc.seeds)
                  for c in sc.conditions), flush=True)

    header("STATISTICS")
    rng = np.random.default_rng(tc.seed + 7)
    stats = {}
    for cond in sc.conditions:
        stats[cond] = {}
        for key, label in MEASURES:
            real = {sec: runs[sec]["real"].get(key) for sec in runs}
            shuf = {sec: [runs[sec]["shuffled"][cond][str(s)].get(key) for s in sc.seeds] for sec in runs}
            st = paired_stats(real, shuf, sc.n_bootstrap, sc.alpha, rng)
            stats[cond][key] = st
            if st.get("D") is not None:
                print(f"  {cond:<16} {label:<62} effect {st['D']:+.4g} [{st['ci95'][0]:+.3g}, {st['ci95'][1]:+.3g}] "
                      f"p {st['p_perm']:.3g} size {fmt(st['effect_size'])} {'DIFFERS' if st['differs'] else ''}")
    results = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "sections": {c: o for c, o in sections.items()}, "shuffles": shuffles,
               "hop1": {g: {k: v for k, v in h.items() if k != "idx"} for g, h in hop1.items()},
               "runs": runs, "stats": stats, "budget_text": budget_text,
               "min_permutations": (1 + len(sc.seeds)) ** len(jobs),
               "shuffle_config": {k: getattr(sc, k) for k in dir(sc) if not k.startswith("_")}}
    results["verdict_auto"] = auto_verdict(results, sc)
    RESULTS.write_text(json.dumps(results, indent=2, default=str))
    print(f"wrote {os.path.relpath(RESULTS, ROOT)}")
    header("FIGURES")
    figure_effects(results, sc)
    figure_spectra(results, sc)
    write_report(results, sc)
    print("\n" + results["verdict_auto"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report-only", action="store_true", help="skip simulation; every run must already be cached")
    main(report_only=ap.parse_args().report_only)
