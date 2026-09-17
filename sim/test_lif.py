"""Untuned LIF run on the real connectome: Poisson drive into the sensory set.

Every SimConfig value is used at its default; nothing here is tuned. The drive:
each sensory neuron receives Poisson events at SimConfig.test_input_rate_hz,
each event a one-step current pulse scaled so the MEAN input current equals
AudioConfig.i_ext_max.

    .venv/bin/python -m sim.test_lif

Writes figures/lif_raster.png, lif_population_rate.png, lif_rate_histogram.png,
lif_motor_rate.png.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import AudioConfig, Paths, SimConfig
from sim.lif_network import LIFNetwork

DURATION_MS = 2000.0
BENCH_STEPS = 500
RASTER_SAMPLE = 500
SUMMARY_BIN_MS = 250.0   # printed population-rate time course

# reference palette (dataviz skill, light mode)
SURFACE, TEXT, TEXT_2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6"


def poisson_drive(net, rate_hz, i_ext_max, seed):
    p = rate_hz * net.cfg.dt_ms / 1e3
    if not 0 < p <= 1:
        raise ValueError(f"test_input_rate_hz={rate_hz} gives per-step event probability {p}")
    amp = i_ext_max / p
    gen = torch.Generator(device=net.device).manual_seed(seed)
    buf = torch.zeros(net.N, device=net.device)
    n_s = net.sensory_idx.numel()

    def fn(_step):
        buf.zero_()
        events = torch.rand(n_s, generator=gen, device=net.device) < p
        buf[net.sensory_idx] = events.to(torch.float32) * amp
        return buf

    return fn, p, amp


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=TEXT, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=TEXT_2)
    ax.set_ylabel(ylabel, color=TEXT_2)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, path):
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def main():
    scfg, acfg = SimConfig(), AudioConfig()
    net = LIFNetwork(ROOT / Paths().cache, cfg=scfg)
    fn, p, amp = poisson_drive(net, scfg.test_input_rate_hz, acfg.i_ext_max, scfg.seed + 1)
    print(f"drive: {net.sensory_idx.numel()} sensory neurons, Poisson {scfg.test_input_rate_hz:g} Hz "
          f"(p={p:g}/step), pulse {amp:g} -> mean I_ext {acfg.i_ext_max:g}; one pulse raises V by "
          f"{amp * scfg.r_membrane * scfg.dt_ms / scfg.tau_m_ms:g} (v_thresh {scfg.v_thresh:g})")

    net.run(BENCH_STEPS, fn)
    net.report_benchmark(f"after {BENCH_STEPS}-step driven run", net.last_run_ms_per_step)

    net.reset()
    fn, _, _ = poisson_drive(net, scfg.test_input_rate_hz, acfg.i_ext_max, scfg.seed + 1)
    n_steps = round(DURATION_MS / scfg.dt_ms)
    rec = net.run(n_steps, fn)
    net.report_benchmark(f"{n_steps}-step driven run", net.last_run_ms_per_step)

    dur_s = n_steps * scfg.dt_ms / 1e3
    dt_s = scfg.dt_ms / 1e3
    rates = rec.counts_per_neuron().numpy() / dur_s
    sens = net.sensory_idx.cpu().numpy()
    motor = net.motor_idx.cpu().numpy()
    other = np.ones(net.N, dtype=bool)
    other[sens] = False
    pop_rate = rec.counts_per_step().numpy() / net.N / dt_s            # Hz, 1-step bins
    motor_counts = rec.counts_per_step(net.motor_idx.cpu()).numpy()
    k = net.window_steps
    motor_roll = np.convolve(motor_counts, np.ones(k), mode="full")[:n_steps] / len(motor) / (k * dt_s)

    # the ring-buffer readout must agree with the spike record over the last window
    last = rec.steps >= n_steps - k
    rec_motor = np.bincount(rec.idx[last].numpy(), minlength=net.N)[motor] / (k * dt_s)
    if not np.allclose(net.get_motor_rates().cpu().numpy(), rec_motor):
        raise AssertionError("get_motor_rates disagrees with the spike record")

    print(f"\n=== results: {DURATION_MS:g} ms, all SimConfig defaults, no tuning ===")
    print(f"spikes: {rec.n_spikes:,}")
    print(f"firing rate per neuron (Hz): mean {rates.mean():.4f}  median {np.median(rates):.4f}  max {rates.max():.2f}")
    print(f"never fired: {(rates == 0).mean():.2%} ({int((rates == 0).sum()):,} / {net.N:,})")
    print(f"sensory (driven) mean {rates[sens].mean():.2f} Hz, {int((rates[sens] > 0).sum())}/{len(sens)} fired | "
          f"non-sensory mean {rates[other].mean():.4f} Hz, {int((rates[other] > 0).sum()):,} fired | "
          f"motor mean {rates[motor].mean():.4f} Hz, {int((rates[motor] > 0).sum())}/{len(motor)} fired")
    mean_pop = pop_rate.mean()
    if mean_pop > 0:
        print(f"synchrony index var/mean of population rate ({scfg.dt_ms:g} ms bins, Hz): "
              f"{pop_rate.var() / mean_pop:.4f}  (mean {mean_pop:.4f} Hz, var {pop_rate.var():.4f})")
    else:
        print("synchrony index: undefined, population rate is zero throughout")
    b = round(SUMMARY_BIN_MS / scfg.dt_ms)
    course = pop_rate[: n_steps // b * b].reshape(-1, b).mean(axis=1)
    print(f"population rate per {SUMMARY_BIN_MS:g} ms (Hz): " + "  ".join(f"{x:.4f}" for x in course))
    print(f"final V: mean {float(net.V.mean()):.4f}, max {float(net.V.max()):.4f}; "
          f"I_syn mean {float(net.I_syn.mean()):.5f}, max {float(net.I_syn.max()):.4f}")

    fig_dir = ROOT / Paths().figures
    fig_dir.mkdir(exist_ok=True)
    t_ms = np.arange(n_steps) * scfg.dt_ms
    print("figures:")

    rng = np.random.default_rng(scfg.seed)
    sample = np.sort(rng.choice(net.N, size=RASTER_SAMPLE, replace=False))
    row = np.full(net.N, -1)
    row[sample] = np.arange(RASTER_SAMPLE)
    r = row[rec.idx.numpy()]
    keep = r >= 0
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.scatter(rec.steps.numpy()[keep] * scfg.dt_ms, r[keep], s=2, c=BLUE, linewidths=0)
    ax.set_xlim(0, DURATION_MS)
    ax.set_ylim(-1, RASTER_SAMPLE)
    style(ax, f"(a) Spike raster, random {RASTER_SAMPLE} of {net.N:,} neurons  ({int(keep.sum()):,} spikes)",
          "time (ms)", "neuron (sample row)")
    if not keep.any():
        ax.text(0.5, 0.5, "no spikes in this sample", transform=ax.transAxes, ha="center", color=TEXT_2)
    save(fig, fig_dir / "lif_raster.png")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t_ms, pop_rate, color=BLUE, linewidth=1)
    ax.set_xlim(0, DURATION_MS)
    style(ax, f"(b) Population firing rate, all {net.N:,} neurons ({scfg.dt_ms:g} ms bins)", "time (ms)", "Hz per neuron")
    save(fig, fig_dir / "lif_population_rate.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rates, bins=60, color=BLUE, edgecolor=SURFACE, linewidth=0.5)
    ax.set_yscale("log")
    style(ax, f"(c) Per-neuron firing rate over {DURATION_MS / 1e3:g} s  ({(rates == 0).mean():.2%} never fired)",
          "firing rate (Hz)", "neurons (log)")
    save(fig, fig_dir / "lif_rate_histogram.png")

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(t_ms, motor_roll, color=BLUE, linewidth=1.5)
    ax.set_xlim(0, DURATION_MS)
    style(ax, f"(d) Motor population rate, {len(motor):,} neurons ({scfg.rate_window_ms:g} ms rolling window)",
          "time (ms)", "Hz per neuron")
    save(fig, fig_dir / "lif_motor_rate.png")


if __name__ == "__main__":
    main()
