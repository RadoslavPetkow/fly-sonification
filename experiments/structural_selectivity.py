"""Is the auditory input to the hop-1 descending neurons band-selective in the WIRING? (No simulation.)

    .venv/bin/python -m experiments.structural_selectivity

For each hop-1 motor neuron m (the 56 voices) and each of the 8 sensory groups g (data/groups.py split, band g
drives group g): w[m, g] = summed |synaptic weight| from the members of g onto m in the built matrix.
  p[m, g]        = w[m, g] / sum_g w[m, g]
  entropy        = -sum_g p log p / log 8        (normalized, 0 = all input from one group, 1 = uniform)
  selectivity    = 1 - entropy
  max share      = max_g p[m, g]
PREDICTION (stated before computing): if the real connectome gives descending neurons band-selective auditory
input, the real mean entropy is LOWER (selectivity and max share HIGHER) than under sensory-only shuffles.

Null: sensory-only shuffles, i.e. the sensory neurons' outgoing edges re-targeted with experiments.shuffle_graph's
procedure (targets' sensory in-degree, each sensory neuron's out-degree and weights preserved). Because target
in-degree within the sensory edges is preserved, the same 56 neurons remain the sensory set's motor targets with
the same number of sensory synapse contacts; only WHICH sensory group contacts them changes.
  built      the cached sensory-only matrices used by the shuffle control (cache/shuffles/sensory_only_seed*.npz)
  ensemble   ShuffleConfig.structural_n_shuffles further re-targetings of the sensory edges alone
             (seeds structural_seed0 + i), structure only
Statistics: difference real - shuffled mean; 95% interval over shuffle seeds; one-sided permutation p in the
predicted direction, (1 + #{shuffled >= real}) / (1 + n) for selectivity and max share; effect size in SDs of
the shuffle distribution. Also per voice family (MidiConfig family grouping) and per neuron.

Caveat: the 8 groups are consecutive slices of the sensory neurons sorted by TYPE, and JO types are partly
defined by projection pattern, so group-selective wiring is to some degree type-selective by construction.
The assignment of groups to frequency bands is this project's, not a measured tonotopy.
Writes cache/structural_selectivity.json, figures/structural_selectivity.png, experiments/structural_selectivity.md.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp

from config import AudioConfig, MidiConfig, Paths, ShuffleConfig
from data.fetch_connectome import bfs_levels
from data.groups import load_neurons_and_indices, ordered_groups
from experiments.shuffle_graph import retarget, shuffle_path
from midi.midi_out import group_key
from sim.calibrate import CATEGORICAL, GRID, TEXT, TEXT_2, header, save, style

CACHE = ROOT / Paths().cache
OUT_JSON = CACHE / "structural_selectivity.json"
OUT_MD = ROOT / "experiments" / "structural_selectivity.md"


def sensory_edges(W, sens):
    """Sensory neurons' outgoing edges in presynaptic (CSC) order: (pre, post, |weight|)."""
    C = W.tocsc()
    C.sort_indices()
    pre = np.repeat(np.arange(W.shape[1], dtype=np.int64), np.diff(C.indptr))
    mask = np.zeros(W.shape[1], dtype=bool)
    mask[sens] = True
    sel = mask[pre]
    return pre[sel], C.indices[sel].astype(np.int64), np.abs(C.data[sel]).astype(np.float64)


def input_matrix(e_pre, e_post, e_w, targets, group_of, n_groups):
    pos = np.full(group_of.size, -1)
    pos[targets] = np.arange(targets.size)
    keep = pos[e_post] >= 0
    M = np.zeros((targets.size, n_groups))
    np.add.at(M, (pos[e_post[keep]], group_of[e_pre[keep]]), e_w[keep])
    return M


def selectivity(M):
    tot = M.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(tot > 0, M / tot, 0.0)
        H = -np.where(p > 0, p * np.log(p), 0.0).sum(axis=1) / np.log(M.shape[1])
    has = tot[:, 0] > 0
    return {"entropy": np.where(has, H, np.nan), "selectivity": np.where(has, 1 - H, np.nan),
            "max_share": np.where(has, p.max(axis=1), np.nan), "strongest_group": np.where(has, p.argmax(axis=1), -1)}


def summary(M, subset):
    s = selectivity(M)
    return {k: float(np.nanmean(s[k][subset])) for k in ("entropy", "selectivity", "max_share")}


def compare(real, shuffled, key, direction):
    """direction +1: prediction real > shuffled; -1: real < shuffled."""
    x = np.array([v[key] for v in shuffled])
    r = real[key]
    D = r - x.mean()
    lo, hi = np.percentile(x, [2.5, 97.5])
    extreme = np.sum(x >= r) if direction > 0 else np.sum(x <= r)
    return {"real": r, "shuffled_mean": float(x.mean()), "shuffled_sd": float(x.std(ddof=1)), "D": float(D),
            "ci95_D": [float(r - hi), float(r - lo)], "p_one_sided": float((1 + extreme) / (1 + x.size)),
            "p_two_sided": float((1 + np.sum(np.abs(x - x.mean()) >= abs(D))) / (1 + x.size)),
            "effect_size_sd": float(D / x.std(ddof=1)) if x.std(ddof=1) > 0 else None, "n_shuffles": int(x.size)}


def main():
    sc, mc = ShuffleConfig(), MidiConfig()
    neurons, indices = load_neurons_and_indices()
    N = len(neurons)
    sens = np.asarray(indices["sensory_idx"])
    motor = np.asarray(indices["motor_idx"])
    n_groups = AudioConfig().n_bins
    groups = ordered_groups(neurons, sens, n_groups)
    group_of = np.full(N, -1)
    for g, members in enumerate(groups):
        group_of[members] = g
    W = sp.load_npz(CACHE / "adjacency.npz").tocsr()
    voices = motor[bfs_levels(W.T.tocsr(), sens)[motor] == 1]
    e_pre, e_post, e_w = sensory_edges(W, sens)
    header(f"STRUCTURAL SELECTIVITY - {voices.size} hop-1 motor neurons x {n_groups} sensory groups")
    print("PREDICTION: band-selective wiring -> real entropy LOWER, selectivity and max share HIGHER than sensory-only shuffles")

    M_real = input_matrix(e_pre, e_post, e_w, voices, group_of, n_groups)
    n_edges = input_matrix(e_pre, e_post, np.ones_like(e_w), voices, group_of, n_groups).sum(axis=1)
    all_v = np.ones(voices.size, dtype=bool)
    multi = n_edges >= 2
    fams = np.array([group_key(t, mc) for t in neurons.iloc[voices]["type"]])
    fam_names = sorted(set(fams))
    F = lambda M: np.stack([M[fams == f].sum(axis=0) for f in fam_names])
    real_sel = selectivity(M_real)
    print(f"sensory edges: {e_pre.size:,}; onto the voices: {int(n_edges.sum())} contacts; voices with >= 2 sensory "
          f"contacts: {int(multi.sum())}; sensory synapses onto voices per neuron median {np.median(M_real.sum(1)):g}")

    shuffled_built, shuffled_ens = [], []
    per_neuron_ens = []
    fam_ens = []
    for seed in sc.seeds:
        path = shuffle_path("sensory_only", seed)
        if not path.exists():
            continue
        S = sp.load_npz(path).tocsr()
        sp_pre, sp_post, sp_w = sensory_edges(S, sens)
        M = input_matrix(sp_pre, sp_post, sp_w, voices, group_of, n_groups)
        np_new, _ = retarget(e_pre, e_post, N, np.random.default_rng(seed), sc.max_repair_iterations)
        M_check = input_matrix(e_pre, np_new, e_w, voices, group_of, n_groups)
        if not np.allclose(M, M_check):
            raise AssertionError(f"ensemble procedure does not reproduce cached sensory_only seed {seed}")
        shuffled_built.append({"seed": seed, "all": summary(M, all_v), "multi": summary(M, multi)})
    print(f"built sensory-only matrices: {len(shuffled_built)} (reproduced exactly by the ensemble procedure)")

    for i in range(sc.structural_n_shuffles):
        new_post, _ = retarget(e_pre, e_post, N, np.random.default_rng(sc.structural_seed0 + i), sc.max_repair_iterations)
        M = input_matrix(e_pre, new_post, e_w, voices, group_of, n_groups)
        s = selectivity(M)
        shuffled_ens.append({"all": summary(M, all_v), "multi": summary(M, multi)})
        per_neuron_ens.append(s["selectivity"])
        fam_ens.append(selectivity(F(M))["selectivity"])
    per_neuron_ens = np.array(per_neuron_ens)
    fam_ens = np.array(fam_ens)

    res = {"n_voices": int(voices.size), "n_multi_contact": int(multi.sum()), "n_sensory_edges": int(e_pre.size),
           "built": shuffled_built, "ensemble_n": sc.structural_n_shuffles, "stats": {}}
    for subset in ("all", "multi"):
        real = summary(M_real, all_v if subset == "all" else multi)
        res["stats"][subset] = {
            "entropy": compare(real, [x[subset] for x in shuffled_ens], "entropy", -1),
            "selectivity": compare(real, [x[subset] for x in shuffled_ens], "selectivity", +1),
            "max_share": compare(real, [x[subset] for x in shuffled_ens], "max_share", +1),
        }
        for key, st in res["stats"][subset].items():
            built = [b[subset][key] for b in shuffled_built]
            print(f"  {subset:<5} {key:<11} real {st['real']:.4f} | built seeds {', '.join(f'{v:.4f}' for v in built)} | "
                  f"ensemble mean {st['shuffled_mean']:.4f} sd {st['shuffled_sd']:.4f} | D {st['D']:+.4f} "
                  f"[{st['ci95_D'][0]:+.4f}, {st['ci95_D'][1]:+.4f}] | p(one-sided, predicted) {st['p_one_sided']:.4g} "
                  f"p(two-sided) {st['p_two_sided']:.4g} | {st['effect_size_sd']:+.2f} SD")

    pct = (per_neuron_ens < real_sel["selectivity"][None, :]).mean(axis=0)
    above95 = int(np.nansum(pct >= 0.95))
    below5 = int(np.nansum(pct <= 0.05))
    res["per_neuron"] = [{"bodyId": int(neurons.iloc[v]["bodyId"]), "type": neurons.iloc[v]["type"], "family": fams[k],
                          "sensory_contacts": int(n_edges[k]), "sensory_synapses": float(M_real[k].sum()),
                          "selectivity": None if np.isnan(real_sel["selectivity"][k]) else float(real_sel["selectivity"][k]),
                          "max_share": float(real_sel["max_share"][k]), "strongest_group": int(real_sel["strongest_group"][k]),
                          "shuffled_selectivity_mean": float(np.nanmean(per_neuron_ens[:, k])),
                          "percentile_vs_shuffles": float(pct[k])} for k, v in enumerate(voices)]
    print(f"per neuron: {above95} of {voices.size} voices above the 95th percentile of their own shuffles (5% expected = "
          f"{0.05 * voices.size:.1f}); {below5} below the 5th")

    fam_real = selectivity(F(M_real))
    fam_rows = []
    for j, f in enumerate(fam_names):
        x = fam_ens[:, j]
        fam_rows.append({"family": f, "members": int((fams == f).sum()), "selectivity": float(fam_real["selectivity"][j]),
                         "max_share": float(fam_real["max_share"][j]), "strongest_group": int(fam_real["strongest_group"][j]),
                         "shuffled_mean": float(np.nanmean(x)), "p_one_sided": float((1 + np.sum(x >= fam_real["selectivity"][j])) / (1 + x.size)),
                         "input_share_by_group": (F(M_real)[j] / F(M_real)[j].sum()).round(4).tolist()})
        print(f"  family {f:<5} n={fam_rows[-1]['members']:<3} selectivity {fam_rows[-1]['selectivity']:.3f} vs shuffled "
              f"{fam_rows[-1]['shuffled_mean']:.3f} (p {fam_rows[-1]['p_one_sided']:.3g}); strongest group "
              f"{fam_rows[-1]['strongest_group']} ({fam_rows[-1]['max_share']:.1%}); shares {fam_rows[-1]['input_share_by_group']}")
    res["families"] = fam_rows

    renders = {}
    for p in sorted((ROOT / Paths().runs).glob("*_family_chromatic/render_summary.json")):
        d = json.loads(p.read_text())
        renders[p.parent.name] = {g["name"]: {"best_band": g["best_band"], "drives": g["band_drives_note"]} for g in d["groups"]}
    res["render_band_drivers"] = renders
    for name, groups_ in renders.items():
        rows = [f"{f}: render band {v['best_band']}{' (beats null)' if v['drives'] else ''} vs wiring strongest group "
                f"{next(r['strongest_group'] for r in fam_rows if r['family'] == f)}" for f, v in groups_.items()
                if f in fam_names and v["drives"]]
        print(f"  {name}: " + ("; ".join(rows) if rows else "no family beat its null"))

    OUT_JSON.write_text(json.dumps(res, indent=2, default=str))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
    ax = axes[0]
    x = np.array([s["all"]["selectivity"] for s in shuffled_ens])
    ax.hist(x, bins=40, color=CATEGORICAL[1], alpha=0.8, edgecolor="white", linewidth=0.4,
            label=f"{x.size} sensory-only shuffles")
    ax.axvline(res["stats"]["all"]["selectivity"]["real"], color=CATEGORICAL[0], linewidth=2.5, label="real connectome")
    for b in shuffled_built:
        ax.axvline(b["all"]["selectivity"], color=TEXT_2, linewidth=1, linestyle="--")
    ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_2)
    ax.grid(True, color=GRID, linewidth=0.6, axis="y")
    style(ax, "Mean selectivity (1 - normalized entropy) over the 56 voices", "mean selectivity", "shuffles")
    ax = axes[1]
    mean_sh = np.nanmean(per_neuron_ens, axis=0)
    ok = ~np.isnan(real_sel["selectivity"])
    ax.scatter(mean_sh[ok], real_sel["selectivity"][ok], s=np.clip(n_edges[ok] * 6, 12, 120), color=CATEGORICAL[0],
               alpha=0.7, edgecolor="white", linewidth=0.5)
    lim = [0, 1]
    ax.plot(lim, lim, color=TEXT_2, linewidth=0.8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.grid(True, color=GRID, linewidth=0.6)
    style(ax, "Per voice: real vs mean shuffled selectivity (dot size = sensory contacts)", "shuffled mean", "real")
    save(fig, "structural_selectivity.png")

    s_all, s_multi = res["stats"]["all"], res["stats"]["multi"]
    holds = s_all["entropy"]["p_one_sided"] < sc.alpha and s_all["entropy"]["ci95_D"][1] < 0
    md = ["## Structural analysis: band selectivity of the sensory input to the hop-1 motor neurons (no simulation)", "",
          "**Prediction, stated before computing:** if the real connectome gives descending neurons band-selective auditory "
          "input, the real mean normalized entropy of each hop-1 motor neuron's sensory input across the 8 band groups is "
          "LOWER than under sensory-only shuffles (selectivity = 1 - entropy and the strongest group's share HIGHER).", "",
          "Method: `experiments/structural_selectivity.py`. Input weight = summed |synapses| from each sensory group. Null = "
          f"sensory-only re-targeting (the procedure of `shuffle_graph.py`): the {len(shuffled_built)} matrices built for the "
          f"shuffle control, and an ensemble of {sc.structural_n_shuffles} further seeds of the same procedure for the "
          "interval and the permutation test. Because the sensory edges' target in-degrees are preserved, the same 56 neurons "
          "keep the same number of sensory contacts; only which group contacts them changes.", "",
          f"{voices.size} voices; {int(multi.sum())} have >= 2 sensory contacts (a single contact is maximally selective in "
          "real and shuffled alike).", "",
          "| subset | measure | real | built seeds | shuffled mean (sd) | real - shuffled [95% over seeds] | p one-sided (predicted) | p two-sided | effect (SD) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for subset, label in (("all", "all 56"), ("multi", f">= 2 contacts ({int(multi.sum())})")):
        for key in ("entropy", "selectivity", "max_share"):
            st = res["stats"][subset][key]
            built = ", ".join(f"{b[subset][key]:.3f}" for b in shuffled_built)
            md.append(f"| {label} | {key} | {st['real']:.3f} | {built} | {st['shuffled_mean']:.3f} ({st['shuffled_sd']:.3f}) | "
                      f"{st['D']:+.3f} [{st['ci95_D'][0]:+.3f}, {st['ci95_D'][1]:+.3f}] | {st['p_one_sided']:.3g} | "
                      f"{st['p_two_sided']:.3g} | {st['effect_size_sd']:+.2f} |")
    md += ["", f"Per neuron: {above95} of {voices.size} voices exceed the 95th percentile of their own shuffle distribution "
           f"(chance {0.05 * voices.size:.1f}); {below5} fall below the 5th.", "",
           "| voice family | members | selectivity | shuffled mean | p one-sided | strongest group (share) |", "|---|---|---|---|---|---|"]
    for r in fam_rows:
        md.append(f"| {r['family']} | {r['members']} | {r['selectivity']:.3f} | {r['shuffled_mean']:.3f} | {r['p_one_sided']:.3g} | "
                  f"{r['strongest_group']} ({r['max_share']:.1%}) |")
    md += ["", "Render comparison (families whose note beat its joint null in the chromatic family renders): "]
    for name, groups_ in renders.items():
        rows = [f"{f} band {v['best_band']} (wiring strongest group {next(r['strongest_group'] for r in fam_rows if r['family'] == f)})"
                for f, v in groups_.items() if f in fam_names and v["drives"]]
        md.append(f"- {name}: " + (", ".join(rows) if rows else "none"))
    md += ["", "![structural selectivity](../figures/structural_selectivity.png)", "",
           "Caveat: the 8 groups are consecutive slices of the sensory neurons sorted by type, and JO types are partly defined "
           "by projection pattern, so group-selective wiring is partly type-selective by construction; which group is called "
           "which frequency band is this project's assignment, not measured tonotopy.", "",
           "**Result:** " + ("the prediction HOLDS: real input entropy is lower than shuffled "
                             if holds else "the prediction does NOT hold at alpha {:.2g}: ".format(sc.alpha))
           + f"(all 56: entropy {s_all['entropy']['real']:.3f} vs {s_all['entropy']['shuffled_mean']:.3f}, difference "
           f"{s_all['entropy']['D']:+.3f} [{s_all['entropy']['ci95_D'][0]:+.3f}, {s_all['entropy']['ci95_D'][1]:+.3f}], "
           f"one-sided p {s_all['entropy']['p_one_sided']:.3g}; voices with >= 2 contacts: difference "
           f"{s_multi['entropy']['D']:+.3f}, p {s_multi['entropy']['p_one_sided']:.3g})."]
    OUT_MD.write_text("\n".join(md) + "\n")
    print(f"wrote {OUT_JSON.relative_to(ROOT)}, {OUT_MD.relative_to(ROOT)}")
    print(md[-1])


if __name__ == "__main__":
    main()
