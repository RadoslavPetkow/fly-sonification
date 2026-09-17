## Control 1a: does a working window exist for degree-matched shuffled graphs?

**Prediction, stated before running:** the real graph needed g_inh = 2.5 to open a window that did not exist at g_inh = 1; the question is whether degree-matched random graphs have such a window at the same g_inh at all.

Method: `experiments/matched_regime.py`. Identical protocol for every graph, the real graph included as positive control: g_inh 2.5, noise sigma 0.632, drive 1.1, 13 w_scale points over each graph's own w_anchor(10 Hz) x 10^[-1.5, +1.5], 1000 ms after 200 ms warm-up, every point from rest. A point passes if (1) population rate 1-8 Hz, (2) sensory rate > 70% of the w_scale = 0 reference (20.2 Hz), (3) input-driven: with the drive removed the non-sensory rate stays <= 1 Hz.

| graph | anchor S | w_anchor(10 Hz) | points in rate band | passing points | window | highest rate below band | lowest rate above band |
|---|---|---|---|---|---|---|---|
| real | 1.83 | 9.9 | 2 | 2 | **yes** | 0.0187 Hz | 11.7 Hz |
| full seed 1 | 5.15 | 3.52 | 0 | 0 | **no** | 0.0155 Hz | 24.2 Hz |
| full seed 2 | 5.12 | 3.54 | 0 | 0 | **no** | 0.0174 Hz | 24.8 Hz |
| downstream_only seed 1 | 5.16 | 3.51 | 0 | 0 | **no** | 0.688 Hz | 26.1 Hz |
| downstream_only seed 2 | 5.14 | 3.53 | 0 | 0 | **no** | 0.749 Hz | 27 Hz |

Per point (rate Hz / sensory preservation / no-drive non-sensory Hz where tested):

- **real**: 0.0316x: 0.0114 / 108%; 0.0562x: 0.0129 / 117%; 0.1x: 0.0187 / 118%; 0.178x: 0.0186 / 94%; 0.316x: 2.63 / 85% / 0 PASS; 0.562x: 4.43 / 77% / 0 PASS; 1x: 11.7 / 58%; 1.78x: 12 / 78%; 3.16x: 15.6 / 129%; 5.62x: 18.6 / 106%; 10x: 19.8 / 80%; 17.8x: 18.9 / 42%; 31.6x: 17.4 / 49%
- **full seed 1**: 0.0316x: 0.0105 / 100%; 0.0562x: 0.0105 / 100%; 0.1x: 0.0105 / 100%; 0.178x: 0.0106 / 100%; 0.316x: 0.0116 / 100%; 0.562x: 0.0155 / 101%; 1x: 24.2 / 175%; 1.78x: 41.1 / 231%; 3.16x: 52 / 301%; 5.62x: 58.9 / 356%; 10x: 62.9 / 387%; 17.8x: 65.7 / 409%; 31.6x: 67.1 / 400%
- **full seed 2**: 0.0316x: 0.0105 / 100%; 0.0562x: 0.0105 / 100%; 0.1x: 0.0105 / 100%; 0.178x: 0.0107 / 100%; 0.316x: 0.0113 / 101%; 0.562x: 0.0174 / 101%; 1x: 24.8 / 188%; 1.78x: 41.7 / 264%; 3.16x: 52 / 326%; 5.62x: 58.6 / 374%; 10x: 62.8 / 399%; 17.8x: 65.5 / 414%; 31.6x: 67.3 / 414%
- **downstream_only seed 1**: 0.0316x: 0.0108 / 103%; 0.0562x: 0.011 / 105%; 0.1x: 0.0116 / 109%; 0.178x: 0.0144 / 117%; 0.316x: 0.0456 / 133%; 0.562x: 0.688 / 168%; 1x: 26.1 / 214%; 1.78x: 42.6 / 297%; 3.16x: 52.8 / 351%; 5.62x: 59.8 / 389%; 10x: 64 / 418%; 17.8x: 66.4 / 428%; 31.6x: 67.9 / 439%
- **downstream_only seed 2**: 0.0316x: 0.0108 / 103%; 0.0562x: 0.011 / 105%; 0.1x: 0.0115 / 109%; 0.178x: 0.0142 / 116%; 0.316x: 0.0437 / 130%; 0.562x: 0.749 / 161%; 1x: 27 / 237%; 1.78x: 43.3 / 339%; 3.16x: 54 / 398%; 5.62x: 60 / 431%; 10x: 64.5 / 452%; 17.8x: 66.8 / 463%; 31.6x: 68.2 / 477%

![matched regime](../figures/matched_regime.png)

**Result:** NO degree-matched shuffled graph (4 tested) has a working window at g_inh 2.5, while the real graph does under the identical protocol. The connectome's topology admits a stable low-rate, input-driven state at this inhibitory gain that degree-, weight- and sign-matched random graphs do not.
