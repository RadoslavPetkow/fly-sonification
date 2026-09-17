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

    # --- fetch -------------------------------------------------------------
    fetch_status: str = "Traced"    # only neurons with this status are cached
    fetch_batch_size: int = 1000    # source bodyIds per edge-fetch page; lower it
                                    # if the server times out (resume keeps progress)

    # --- edges -------------------------------------------------------------
    fetch_min_weight: int = 1       # what we FETCH and cache as a raw edge list
    matrix_weight_threshold: int = 2  # edges below this are dropped when BUILDING
                                      # the matrix; re-thresholding must never
                                      # require re-fetching

    # --- neurotransmitter -> synaptic sign ---------------------------------
    # Coverage in male-cns:v1.0 (recon, Sep 2026): acetylcholine 95,409 /
    # glutamate 28,199 / unclear 22,902 / gaba 20,357 / dopamine 4,447 /
    # histamine 2,242 / serotonin 483 / octopamine 126; 2,257 neurons have no
    # value in any NT field. consensusNt differs from predictedNt for 20,169.
    nt_field: str = "consensusNt"          # consensusNt | predictedNt | celltypePredictedNt
    nt_fallback_field: str = "predictedNt" # used only where nt_field is null
    nt_sign = {
        "acetylcholine": +1,
        "gaba": -1,
        "glutamate": -1,   # GluCl is a chloride channel in Drosophila
        "histamine": -1,   # HisCl1 / ort are chloride channels
    }
    nt_modulatory = ("dopamine", "octopamine", "serotonin")
    nt_unclear_values = ("unclear", "", None)
    nt_unknown_sign: int = +1   # applies to ~14% of neurons; must be reported

    # --- sensory population ------------------------------------------------
    # The dataset's own functional labels disagree with the textbook
    # "JO-A/B = sound, C/D/E = wind & gravity" split: 34 of 88 JO-B neurons are
    # subclass wind_gravity, and 8 JO-C neurons are subclass auditory. We trust
    # the curators' subclass over the type prefix.
    sensory_mode: str = "subclass_auditory"   # subclass_auditory | jo_ab_prefix | jo_all
    sensory_subclass: str = "auditory"        # 115 neurons
    sensory_type_prefixes = ("JO-A", "JO-B")  # prefix match; no type is exactly "JO-A"
    sensory_jo_prefix: str = "JO-"            # sensory_mode == "jo_all"
    sensory_roi_check = ("AMMC(L)", "AMMC(R)")  # cross-check only, NOT a selector.
                                                # Non-primary ROIs, both under SAD.

    # --- motor population --------------------------------------------------
    # superclass is populated for 166,700 of 176,422 neurons; class for only
    # 26,513 - so superclass is the reliable field. Do NOT select DNs by '^DN'
    # regex: it drags in clock neurons (DN1a, DN1pA/B, DNd01) and endocrine
    # cells (DNES1/2/3), and misses MDN, pIP1, pIP10, pMP2, aSP22, LN-DN1/2.
    motor_superclasses = ("descending_neuron", "sensory_descending")
    motor_require_type: bool = True   # untyped neurons cannot be grouped for MIDI


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
    n_bins: int = 8                     # only 115 auditory sensory neurons exist,
                                        # so 8 bands ~= 14 neurons per band
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