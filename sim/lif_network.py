"""Vectorized leaky integrate-and-fire network over the cached connectome.

W[i_post, j_pre] (cache/adjacency.npz) is normalized once at load
(sim/normalize.py) and held as a torch sparse CSR tensor with int32 indices.
w_scale and sigma_noise are free scalars (set_scale) so a parameter sweep never
touches W.

Per step (dt = cfg.dt_ms):
  1. I_syn *= exp(-dt / tau_syn)
  2. I_syn += w_scale * (W @ spikes emitted round(delay_ms/dt) steps ago)
  3. a *= exp(-dt / tau_adapt)
     V += dt/tau_m * (-(V - v_rest) + R * (I_syn + I_ext + sigma_noise * randn - a))
  4. V[refrac > 0] = v_reset; refrac -= 1
  5. spikes = V > v_thresh; V[spikes] = v_reset; refrac[spikes] = refrac_steps; a[spikes] += b_adapt
     (b_adapt is 0 for the sensory input set when cfg.sensory_adapt_exempt)

W is normalized (cfg.normalization) and its negative entries multiplied by cfg.g_inh
once at load. Adaptation is skipped entirely while b_adapt == 0 (a stays 0).

    .venv/bin/python -m sim.lif_network      # smoke test: load, benchmark, 500 noise-only steps
"""
import json
import math
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.sparse as sp
import torch

from config import Paths, SimConfig
from sim.normalize import apply_inhibitory_gain, normalize

LOAD_BENCHMARK_STEPS = 50   # timed steps for the at-load benchmark (after a short warm-up)
LOAD_WARMUP_STEPS = 5


def resolve_device(device, force_mps):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if device.type == "mps" and not force_mps:
        warnings.warn("torch.sparse support on MPS is incomplete; falling back to cpu "
                      "(set SimConfig.force_mps=True to override)")
        device = torch.device("cpu")
    return device


def adaptation_peak_per_b(cfg, horizon_ms=2000.0):
    """Peak membrane hyperpolarization (as a positive number, per unit b_adapt) caused by
    one spike's adaptation current alone, from the discrete update used in step():
    a_k = b * exp(-dt/tau_adapt)^k,  V_k = V_{k-1} + dt/tau_m * (-V_{k-1} - R * a_k).
    The refractory clamp is ignored (it only truncates the first refrac_steps steps)."""
    d = math.exp(-cfg.dt_ms / cfg.tau_adapt_ms)
    k_m = cfg.dt_ms / cfg.tau_m_ms
    v, a, peak = 0.0, 1.0, 0.0
    for _ in range(round(horizon_ms / cfg.dt_ms)):
        a *= d
        v += k_m * (-v - cfg.r_membrane * a)
        peak = max(peak, -v)
    return peak


@dataclass
class SpikeRecord:
    """Sparse spike record: spike k happened at run step steps[k] in neuron idx[k]."""
    steps: torch.Tensor   # int32, run-relative step
    idx: torch.Tensor     # int32, neuron (matrix) index
    n_steps: int
    n_neurons: int
    dt_ms: float

    @property
    def n_spikes(self):
        return int(self.idx.numel())

    def counts_per_neuron(self):
        return torch.bincount(self.idx.long(), minlength=self.n_neurons)

    def counts_per_step(self, neurons=None):
        steps = self.steps
        if neurons is not None:
            mask = torch.zeros(self.n_neurons, dtype=torch.bool)
            mask[neurons] = True
            steps = steps[mask[self.idx.long()]]
        return torch.bincount(steps.long(), minlength=self.n_steps)

    def to_coo(self):
        return torch.sparse_coo_tensor(torch.stack([self.steps.long(), self.idx.long()]),
                                       torch.ones(self.n_spikes, dtype=torch.bool),
                                       size=(self.n_steps, self.n_neurons))


class LIFNetwork:
    def __init__(self, cache_dir=ROOT / Paths().cache, device=None, cfg=SimConfig, adjacency_path=None):
        self.cfg = cfg() if isinstance(cfg, type) else cfg
        c = self.cfg
        self.device = resolve_device(device, c.force_mps)
        cache_dir = Path(cache_dir)

        t0 = time.perf_counter()
        self.adjacency_path = Path(adjacency_path) if adjacency_path is not None else cache_dir / "adjacency.npz"
        W = sp.load_npz(self.adjacency_path)
        indices = json.loads((cache_dir / "indices.json").read_text())
        if W.shape[0] != W.shape[1] or list(W.shape) != indices["shape"]:
            raise ValueError(f"adjacency shape {W.shape} does not match indices.json {indices['shape']}")
        Wn, self.norm_stats = normalize(W, c.normalization)
        del W
        Wn, self.ei_stats = apply_inhibitory_gain(Wn, c.g_inh)
        if Wn.nnz >= 2**31 or Wn.shape[0] >= 2**31:
            raise OverflowError("matrix too large for int32 CSR indices")
        self.W = torch.sparse_csr_tensor(
            torch.from_numpy(Wn.indptr.astype(np.int32)),
            torch.from_numpy(Wn.indices.astype(np.int32)),
            torch.from_numpy(Wn.data.astype(np.float32)),
            size=Wn.shape, device=self.device)
        assert self.W.crow_indices().dtype == torch.int32 and self.W.col_indices().dtype == torch.int32
        del Wn
        self.N = self.W.shape[0]
        self.nnz = self.W.values().numel()
        self.sensory_idx = torch.tensor(indices["sensory_idx"], dtype=torch.long, device=self.device)
        self.motor_idx = torch.tensor(indices["motor_idx"], dtype=torch.long, device=self.device)
        if self.sensory_idx.numel() == 0 or self.motor_idx.numel() == 0:
            raise ValueError("indices.json has an empty sensory or motor set")

        self.delay_steps = round(c.delay_ms / c.dt_ms)
        if self.delay_steps < 1:
            raise ValueError(f"delay_ms={c.delay_ms} at dt_ms={c.dt_ms} rounds to {self.delay_steps} steps; "
                             "the minimum synaptic delay is one step")
        self.window_steps = round(c.rate_window_ms / c.dt_ms)
        if self.window_steps < 1:
            raise ValueError(f"rate_window_ms={c.rate_window_ms} is shorter than one step")
        self.decay_syn = math.exp(-c.dt_ms / c.tau_syn_ms)
        self.k_m = c.dt_ms / c.tau_m_ms
        self.gen = torch.Generator(device=self.device)
        self.set_scale(c.w_scale, c.sigma_noise)
        self.set_adaptation(c.b_adapt, c.tau_adapt_ms)

        dev, f32 = self.device, torch.float32
        self.V = torch.empty(self.N, dtype=f32, device=dev)
        self.I_syn = torch.empty(self.N, dtype=f32, device=dev)
        self.a = torch.empty(self.N, dtype=f32, device=dev)
        self.refrac = torch.empty(self.N, dtype=torch.int32, device=dev)
        self.spikes = torch.empty(self.N, dtype=torch.bool, device=dev)
        self._noise = torch.empty(self.N, dtype=f32, device=dev)
        self._drive = torch.empty(self.N, dtype=f32, device=dev)
        self._delay_buf = torch.empty(self.delay_steps, self.N, dtype=f32, device=dev)
        self._motor_buf = torch.empty(self.window_steps, self.motor_idx.numel(), dtype=torch.int32, device=dev)
        self._motor_count = torch.empty(self.motor_idx.numel(), dtype=torch.int32, device=dev)
        self._global_buf = torch.empty(self.window_steps, dtype=torch.int64, device=dev)
        self.reset()

        idx_mb = (self.W.crow_indices().numel() + self.W.col_indices().numel()) * 4 / 1e6
        print(f"LIFNetwork: N={self.N:,} nnz={self.nnz:,} on {self.device} "
              f"(torch {torch.__version__}, {torch.get_num_threads()} threads); "
              f"W values {self.nnz * 4 / 1e6:.1f} MB + int32 indices {idx_mb:.1f} MB; "
              f"loaded in {time.perf_counter() - t0:.1f}s")
        s = self.norm_stats
        print(f"  normalization {s['mode']!r}: {s['zero_in_degree']:,} zero-in-degree neurons; "
              f"row sum |W| median {s['in_abs_median_before']:g} -> {s['in_abs_median_after']:g}, "
              f"max {s['in_abs_max_before']:g} -> {s['in_abs_max_after']:g}")
        e = self.ei_stats
        print(f"  g_inh {e['g_inh']:g}: E/I by weight {e['exc_fraction']:.2%} excitatory "
              f"(E {e['exc_weight']:.4g}, I {e['inh_weight']:.4g}); refrac_steps {c.refrac_steps} "
              f"(max {1e3 / ((c.refrac_steps + 1) * c.dt_ms):.1f} Hz); adaptation b {self.b_adapt:g}, "
              f"tau {self.tau_adapt_ms:g} ms, sensory exempt {self.sensory_adapt_exempt}")
        print(f"  dt {c.dt_ms} ms, delay {self.delay_steps} steps, tau_m {c.tau_m_ms}, tau_syn {c.tau_syn_ms}, "
              f"w_scale {self.w_scale}, sigma_noise {self.sigma_noise}, sensory {self.sensory_idx.numel()}, "
              f"motor {self.motor_idx.numel()}")
        self.load_ms_per_step = self.benchmark(LOAD_BENCHMARK_STEPS, label="at load")
        self.reset()

    # ------------------------------------------------------------------ control
    def set_scale(self, w_scale, sigma_noise):
        self.w_scale = float(w_scale)
        self.sigma_noise = float(sigma_noise)

    def set_adaptation(self, b_adapt, tau_adapt_ms=None, sensory_exempt=None):
        tau = self.cfg.tau_adapt_ms if tau_adapt_ms is None else tau_adapt_ms
        exempt = self.cfg.sensory_adapt_exempt if sensory_exempt is None else sensory_exempt
        if not b_adapt >= 0 or not tau > 0:
            raise ValueError(f"need b_adapt >= 0 and tau_adapt_ms > 0, got {b_adapt}, {tau}")
        self.b_adapt = float(b_adapt)
        self.tau_adapt_ms = float(tau)
        self.sensory_adapt_exempt = bool(exempt)
        self.decay_adapt = math.exp(-self.cfg.dt_ms / tau)
        if self.sensory_adapt_exempt:
            self._b_vec = torch.full((self.N,), self.b_adapt, dtype=torch.float32, device=self.device)
            self._b_vec[self.sensory_idx] = 0.0
        else:
            self._b_vec = None

    def reset(self):
        c = self.cfg
        self.V.fill_(c.v_rest)
        self.I_syn.zero_()
        self.a.zero_()
        self.refrac.zero_()
        self.spikes.zero_()
        self._delay_buf.zero_()
        self._delay_ptr = 0
        self._motor_buf.zero_()
        self._motor_count.zero_()
        self._global_buf.zero_()
        self._win_ptr = 0
        self.t = 0
        self.gen.manual_seed(c.seed)

    # ------------------------------------------------------------------ dynamics
    @torch.no_grad()
    def step(self, I_ext=None):
        c = self.cfg
        # 1-2. decaying synaptic current plus delayed recurrent input; w_scale on the result
        delayed = self._delay_buf[self._delay_ptr]
        self.I_syn.mul_(self.decay_syn)
        self.I_syn.add_(torch.mv(self.W, delayed), alpha=self.w_scale)

        # 3. adaptation decay, membrane update
        adapt = self.b_adapt != 0.0
        drive = torch.add(self.I_syn, self._noise.normal_(generator=self.gen).mul_(self.sigma_noise),
                          out=self._drive)
        if I_ext is not None:
            if I_ext.shape != (self.N,):
                raise ValueError(f"I_ext must have shape ({self.N},), got {tuple(I_ext.shape)}")
            drive.add_(I_ext)
        if adapt:
            self.a.mul_(self.decay_adapt)
            drive.sub_(self.a)
        self.V.add_((c.v_rest - self.V).add_(drive, alpha=c.r_membrane).mul_(self.k_m))

        # 4. refractory clamp
        self.V.masked_fill_(self.refrac > 0, c.v_reset)
        self.refrac.sub_(1).clamp_(min=0)

        # 5. spike, reset, start refractory period
        spikes = self.V > c.v_thresh
        self.V.masked_fill_(spikes, c.v_reset)
        self.refrac.masked_fill_(spikes, c.refrac_steps)
        if adapt:
            if self._b_vec is None:
                self.a.add_(spikes.to(torch.float32), alpha=self.b_adapt)
            else:
                self.a.addcmul_(spikes.to(torch.float32), self._b_vec)
        self.spikes = spikes

        # delay line: this step's spikes are read back delay_steps steps from now
        self._delay_buf[self._delay_ptr].copy_(spikes)
        self._delay_ptr = (self._delay_ptr + 1) % self.delay_steps

        # rolling spike counts
        m = spikes[self.motor_idx].to(torch.int32)
        self._motor_count.add_(m).sub_(self._motor_buf[self._win_ptr])
        self._motor_buf[self._win_ptr] = m
        self._global_buf[self._win_ptr] = spikes.sum()
        self._win_ptr = (self._win_ptr + 1) % self.window_steps
        self.t += 1
        return spikes

    @torch.no_grad()
    def run(self, n_steps, I_ext_fn=None):
        """Run n_steps; I_ext_fn(run_step) -> I_ext tensor or None. Returns a SpikeRecord."""
        steps, idx = [], []
        t0 = time.perf_counter()
        for s in range(n_steps):
            spikes = self.step(None if I_ext_fn is None else I_ext_fn(s))
            fired = spikes.nonzero().flatten().to(torch.int32)
            if fired.numel():
                idx.append(fired)
                steps.append(torch.full_like(fired, s))
        self._sync()
        self.last_run_ms_per_step = (time.perf_counter() - t0) * 1e3 / max(n_steps, 1)
        empty = torch.empty(0, dtype=torch.int32)
        return SpikeRecord(steps=torch.cat(steps).cpu() if steps else empty,
                           idx=torch.cat(idx).cpu() if idx else empty,
                           n_steps=n_steps, n_neurons=self.N, dt_ms=self.cfg.dt_ms)

    # ------------------------------------------------------------------ readout
    def _window(self, window_ms):
        window_ms = self.cfg.rate_window_ms if window_ms is None else window_ms
        k = round(window_ms / self.cfg.dt_ms)
        if not 1 <= k <= self.window_steps:
            raise ValueError(f"window_ms={window_ms} must be between one step and "
                             f"rate_window_ms={self.cfg.rate_window_ms}")
        if self.t == 0:
            raise RuntimeError("no steps run since reset; rate is undefined")
        k = min(k, self.t)
        slots = (self._win_ptr - 1 - torch.arange(k, device=self.device)) % self.window_steps
        return k, slots

    def get_motor_rates(self, window_ms=None):
        """Hz per motor neuron (order of indices.json motor_idx) over the last window_ms."""
        k, slots = self._window(window_ms)
        counts = self._motor_count if k == self.window_steps else self._motor_buf[slots].sum(dim=0)
        return counts.to(torch.float32) / (k * self.cfg.dt_ms / 1e3)

    def get_global_rate(self, window_ms=None):
        """Mean Hz per neuron over the whole network over the last window_ms."""
        k, slots = self._window(window_ms)
        return float(self._global_buf[slots].sum()) / self.N / (k * self.cfg.dt_ms / 1e3)

    # ------------------------------------------------------------------ benchmark
    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def report_benchmark(self, label, ms_per_step):
        rtf = self.cfg.dt_ms / ms_per_step
        print(f"  BENCHMARK [{label}]: {ms_per_step:.2f} ms/step -> real-time factor {rtf:.3f} "
              f"at dt={self.cfg.dt_ms:g} ms ({1e3 / rtf / 1e3:.1f} s wall per simulated second)")
        return rtf

    @torch.no_grad()
    def benchmark(self, n_steps, label="benchmark"):
        for _ in range(LOAD_WARMUP_STEPS):
            self.step()
        self._sync()
        t0 = time.perf_counter()
        for _ in range(n_steps):
            self.step()
        self._sync()
        ms = (time.perf_counter() - t0) * 1e3 / n_steps
        self.report_benchmark(f"{label}, {n_steps} steps", ms)
        return ms


if __name__ == "__main__":
    net = LIFNetwork()
    n = 500
    rec = net.run(n)
    net.report_benchmark(f"{n}-step run, noise only", net.last_run_ms_per_step)
    counts = rec.counts_per_neuron()
    print(f"noise-only {n} steps: {rec.n_spikes:,} spikes, {int((counts > 0).sum()):,} neurons fired, "
          f"global rate (last {net.cfg.rate_window_ms:g} ms) {net.get_global_rate():.4f} Hz, "
          f"motor mean {float(net.get_motor_rates().mean()):.4f} Hz, "
          f"V mean {float(net.V.mean()):.4f} max {float(net.V.max()):.4f}")
