# config.py — single source of truth for every tunable constant.
# These are STARTING POINTS. calibrate.py writes the tuned sim/audio values
# to cache/calibration.json; at runtime calibration.json wins over these.

from dataclasses import dataclass

NEUPRINT_SERVER  = "https://neuprint.janelia.org"
NEUPRINT_DATASET = "male-cns:v1.0"
# token: NEUPRINT_APPLICATION_CREDENTIALS env var, read by neuprint-python itself

@dataclass
class DataConfig:
    max_neurons: int = 0            # 0 = whole connectome, no subsetting
    min_synapse_weight: int = 1
    nt_sign = {"acetylcholine": +1, "gaba": -1, "glutamate": -1}
    nt_modulatory = ("dopamine", "octopamine", "serotonin")
    nt_unknown_sign: int = +1
    sensory_roi: str = "AMMC"       # exact spelling confirmed by recon.py
    auditory_subtypes = ("JO-A", "JO-B")   # JO-C/D/E = gravity & wind, not sound
    include_nonauditory_jo: bool = False

@dataclass
class SimConfig:
    dt_ms: float = 1.0
    tau_m_ms: float = 20.0
    tau_syn_ms: float = 5.0
    delay_ms: float = 2.0
    v_rest: float = 0.0
    v_reset: float = 0.0
    v_thresh: float = 1.0
    r_membrane: float = 1.0
    refrac_steps: int = 2
    normalization: str = "in_degree"    # raw | in_degree | sqrt_in
    w_scale: float = 1.0                # SET BY calibrate.py
    sigma_noise: float = 0.05           # SET BY calibrate.py
    force_mps: bool = False

@dataclass
class AudioConfig:
    sr: int = 16000
    block_ms: float = 10.0
    window_mult: int = 4
    f_min_hz: float = 80.0              # Drosophila near-field hearing range
    f_max_hz: float = 1000.0
    n_bins: int = 16
    db_floor: float = -60.0
    db_ceil: float = 0.0
    i_ext_max: float = 1.0              # SET BY calibrate.py

@dataclass
class MidiConfig:
    tick_ms: float = 50.0
    n_groups: int = 12
    scale_mode: str = "chromatic"       # chromatic | pentatonic
    base_note: int = 48
    velocity_min: int = 40
    velocity_max: int = 127
    thresh_on_pct: float = 70.0         # percentile of the calibrated rate dist
    thresh_off_pct: float = 55.0
    cc_number: int = 74
    port_name: str = "FlyBrain"
    soundfont: str = ""                 # path to FluidR3_GM.sf2

@dataclass
class Paths:
    cache: str = "cache"
    figures: str = "figures"
    runs: str = "runs"