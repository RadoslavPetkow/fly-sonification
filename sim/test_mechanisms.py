"""Forced-spike verification of the stabilizing mechanisms on the real matrix.

    .venv/bin/python -m sim.test_mechanisms

1. g_inh      one inhibitory and one excitatory neuron are forced to spike; after the
              synaptic delay their targets' I_syn must equal w_scale * W column, scaled
              by g_inh for the inhibitory neuron only.
2. adaptation a forced spike must set a = b_adapt, decay it by exp(-dt/tau_adapt) per
              step, and hyperpolarize V exactly as a scalar reference of step() predicts.
3. refractory constant suprathreshold current -> inter-spike interval refrac_steps + 1.
5. exemption  with sensory_adapt_exempt, a forced sensory spike leaves a = 0 while a
              forced non-sensory spike still sets a = b_adapt.
4. regression with g_inh=1, b_adapt=0, refrac_steps=2 the Poisson-driven test_lif run
              must reproduce its recorded spike count exactly.
"""
import math
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import sim.lif_network as lif
from config import AudioConfig, Paths, SimConfig
from sim.lif_network import LIFNetwork, adaptation_peak_per_b

lif.LOAD_BENCHMARK_STEPS = 5
CACHE = ROOT / Paths().cache
KICK = 100.0                       # one-step current that crosses threshold from rest
TEST_LIF_SPIKES = 4431             # runs/test_lif.log: 2 s, in_degree, w 1, sigma 0.05, refrac 2


def kick(net, j):
    I = torch.zeros(net.N)
    I[j] = KICK
    return I


@torch.no_grad()
def forced_response(net, j, n_steps):
    net.reset()
    fired = [int(net.step(kick(net, j)).sum())]
    for _ in range(n_steps - 1):
        fired.append(int(net.step().sum()))
    return fired


@torch.no_grad()
def test_g_inh(base):
    print("\n=== 1. inhibitory gain ===")
    g = 3.0
    out = {}
    for gi in (1.0, g):
        net = LIFNetwork(CACHE, cfg=replace(base, g_inh=gi, w_scale=0.1))
        if gi == 1.0:
            W = net.W
            neg = torch.zeros(net.N)
            pos = torch.zeros(net.N)
            cols = W.to_sparse_coo().coalesce()
            c_idx, vals = cols.indices()[1], cols.values()
            neg.index_add_(0, c_idx, (vals < 0).float())
            pos.index_add_(0, c_idx, (vals > 0).float())
            j_inh, j_exc = int(neg.argmax()), int(pos.argmax())
            print(f"j_inh={j_inh} ({int(neg[j_inh])} inhibitory targets), j_exc={j_exc} ({int(pos[j_exc])} excitatory targets)")
            del cols, c_idx, vals
        for name, j in (("inh", j_inh), ("exc", j_exc)):
            fired = forced_response(net, j, net.delay_steps + 1)
            assert fired == [1] + [0] * net.delay_steps, f"unexpected spikes {fired}"
            col = torch.mv(net.W, torch.nn.functional.one_hot(torch.tensor(j), net.N).float())
            assert torch.allclose(net.I_syn, net.w_scale * col), "I_syn != w_scale * W column after the delay"
            out[(gi, name)] = net.I_syn.clone()
        print(f"  g_inh {gi:g}: E/I by weight {net.ei_stats['exc_fraction']:.2%} excitatory")
        del net
    r_inh = out[(g, "inh")][out[(1.0, "inh")] != 0] / out[(1.0, "inh")][out[(1.0, "inh")] != 0]
    r_exc = out[(g, "exc")][out[(1.0, "exc")] != 0] / out[(1.0, "exc")][out[(1.0, "exc")] != 0]
    print(f"  inhibitory targets: I_syn(g={g:g}) / I_syn(g=1) min {float(r_inh.min()):.6f} max {float(r_inh.max()):.6f} "
          f"(n={r_inh.numel()})")
    print(f"  excitatory targets: ratio min {float(r_exc.min()):.6f} max {float(r_exc.max()):.6f} (n={r_exc.numel()})")
    assert torch.allclose(r_inh, torch.full_like(r_inh, g)) and torch.allclose(r_exc, torch.ones_like(r_exc))
    print("  PASS")


@torch.no_grad()
def test_adaptation(base):
    print("\n=== 2. spike-frequency adaptation ===")
    per_b = adaptation_peak_per_b(base)
    b = 0.3 * base.v_thresh / per_b
    print(f"one spike's adaptation peaks at {per_b:.4f} * b_adapt below rest (free membrane); "
          f"b_adapt for 30% of V_thresh = {b:.4f}")
    net = LIFNetwork(CACHE, cfg=replace(base, w_scale=0.0, b_adapt=b))
    j = int(net.sensory_idx[0])
    n = 400
    a_trace, v_trace = [], []
    net.reset()
    for s in range(n):
        spk = net.step(kick(net, j) if s == 0 else None)
        assert int(spk.sum()) == (1 if s == 0 else 0)
        a_trace.append(float(net.a[j]))
        v_trace.append(float(net.V[j]))
    others = torch.ones(net.N, dtype=torch.bool)
    others[j] = False
    assert float(net.a[others].abs().max()) == 0.0 and float(net.V[others].abs().max()) == 0.0

    c = net.cfg
    d, k_m = math.exp(-c.dt_ms / c.tau_adapt_ms), c.dt_ms / c.tau_m_ms
    v, a, refrac = 0.0, 0.0, 0
    ref_a, ref_v = [], []
    for s in range(n):
        a *= d
        v += k_m * (-(v - c.v_rest) + c.r_membrane * ((KICK if s == 0 else 0.0) - a))
        if refrac > 0:
            v = c.v_reset
        refrac = max(refrac - 1, 0)
        if v > c.v_thresh:
            v, refrac = c.v_reset, c.refrac_steps
            a += b
        ref_a.append(a)
        ref_v.append(v)
    assert np.allclose(a_trace, ref_a, rtol=1e-5, atol=1e-6), "a trace differs from reference"
    assert np.allclose(v_trace, ref_v, rtol=1e-4, atol=1e-6), "V trace differs from reference"
    print(f"  a[j] after spike {a_trace[0]:.4f} (b {b:.4f}); after 150 steps {a_trace[150]:.4f} "
          f"(expected {b * d ** 150:.4f} = b/e)")
    k = int(np.argmin(v_trace))
    print(f"  V[j] minimum {v_trace[k]:.4f} at step {k} ({-v_trace[k] / c.v_thresh:.1%} of V_thresh, "
          f"with the {c.refrac_steps}-step refractory clamp)")
    print(f"  trace matches the scalar reference over {n} steps; no other neuron touched")
    print("  PASS")


@torch.no_grad()
def test_refractory(base):
    print("\n=== 3. refractory period ===")
    for r in (2, 5):
        net = LIFNetwork(CACHE, cfg=replace(base, w_scale=0.0, refrac_steps=r))
        j = int(net.sensory_idx[0])
        I = kick(net, j)
        net.reset()
        times = [s for s in range(60) if bool(net.step(I)[j])]
        isi = set(np.diff(times).tolist())
        rate = 1e3 / ((r + 1) * net.cfg.dt_ms)
        print(f"  refrac_steps {r}: spike steps {times[:6]}..., ISI {isi} -> max rate {rate:.1f} Hz")
        assert isi == {r + 1}
        del net
    print("  PASS")


@torch.no_grad()
def test_regression():
    print("\n=== 4. regression: mechanisms disabled reproduce test_lif ===")
    from sim.test_lif import poisson_drive

    cfg = replace(SimConfig(), refrac_steps=2, g_inh=1.0, b_adapt=0.0)
    net = LIFNetwork(CACHE, cfg=cfg)
    net.reset()
    fn, _, _ = poisson_drive(net, cfg.test_input_rate_hz, AudioConfig().i_ext_max, cfg.seed + 1)
    rec = net.run(round(2000 / cfg.dt_ms), fn)
    print(f"  spikes {rec.n_spikes:,} (test_lif recorded {TEST_LIF_SPIKES:,})")
    assert rec.n_spikes == TEST_LIF_SPIKES
    print("  PASS")


@torch.no_grad()
def test_sensory_exempt(base):
    print("\n=== 5. sensory adaptation exemption ===")
    b = 0.1 * base.v_thresh / adaptation_peak_per_b(base)
    for exempt in (False, True):
        net = LIFNetwork(CACHE, cfg=replace(base, w_scale=0.0, b_adapt=b, sensory_adapt_exempt=exempt))
        j_s = int(net.sensory_idx[0])
        is_sens = torch.zeros(net.N, dtype=torch.bool)
        is_sens[net.sensory_idx] = True
        j_o = int((~is_sens).nonzero()[0])
        I = torch.zeros(net.N)
        I[j_s] = KICK
        I[j_o] = KICK
        net.reset()
        spk = net.step(I)
        assert bool(spk[j_s]) and bool(spk[j_o]) and int(spk.sum()) == 2
        a_s, a_o = float(net.a[j_s]), float(net.a[j_o])
        print(f"  exempt={exempt}: sensory neuron {j_s} a={a_s:.4f}, non-sensory neuron {j_o} a={a_o:.4f} (b {b:.4f})")
        assert math.isclose(a_o, b, rel_tol=1e-6)
        assert (a_s == 0.0) if exempt else math.isclose(a_s, b, rel_tol=1e-6)
        del net
    print("  PASS")


if __name__ == "__main__":
    base = replace(SimConfig(), normalization="sqrt_in", sigma_noise=0.0)
    test_g_inh(base)
    test_adaptation(base)
    test_refractory(base)
    test_regression()
    test_sensory_exempt(base)
    print("\nall mechanism tests passed")
