# fly-sonification — agent rules

Standing rules for every session in this repo. Read before doing anything.

## Ground rules

- Python 3.12 in a local venv (`.venv/`). Pin exact versions in requirements.txt.
- NEVER fabricate, mock, or synthesize data. If an API call, query, or file
  read fails or returns empty, raise with the real error. Do not insert
  placeholder or fallback values to keep the code running.
- Do not invent neuPrint field names, dataset names, ROI names, or neuron
  type names. Query the API for what exists and assert on it.
- No bare `except`, no silently swallowed exceptions.
- config.py is provided and is the single source of truth for tunable
  constants. Extend it when a new constant is needed; never bypass it with a
  literal in a module, and never rename an existing field.
- Every module gets a `__main__` smoke test that runs on real cached data and
  prints real numbers.
- Caches go in cache/ (gitignored); never re-fetch if the cache exists unless
  --force is passed.
- Report what you actually ran and what it printed. Do not claim a step works
  without showing its output.
- Do not run `git commit` yourself. Leave the working tree for the user to
  review and commit.

## Connection details

- Server: https://neuprint.janelia.org — dataset male-cns:v1.0 (167k neurons,
  brain + ventral nerve cord).
- The auth token lives in the NEUPRINT_APPLICATION_CREDENTIALS env var, loaded
  from .env. neuprint-python reads that variable itself — never pass token=
  explicitly, never print the token, never commit .env.
- neuPrint switched to a new authorization system in 2026 and the released
  neuprint-python docs do not yet reflect it. On 401/403 with a valid token,
  the PyPI release may predate the change; installing from git master is the
  first thing to try.

## Build order

Each step is delivered as its own prompt by the user. Do not run ahead to a
later step, and do not start a step before its predecessor's output has been
reviewed.

0. explore/  — connectivity ping, then the recon report
1. data/     — connectome fetch, signed sparse adjacency, cache
2. sim/      — LIF network
3. sim/      — calibration sweep (the make-or-break step)
4. audio/    — offline audio -> sensory current
5. midi/     — motor rates -> .mid file
6. run_offline.py + experiments/shuffle_control.py
7. realtime.py (optional)
8. sim/plasticity.py (optional)
