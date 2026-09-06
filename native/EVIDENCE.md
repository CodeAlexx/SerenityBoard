# Native backend — gates run on 2026-09-05 (me-AI, g++ 15.2, RTX 5080 box)

| gate | command | result |
|---|---|---|
| Core unit gates | `build/sb_core_tests` | pass: SHA-256/SHA-1/base64 vectors, schema v2 + V1 migration, NumPy-equal histogram edges/counts, PR-curve packing, WAV/PNG headers, reservoir capacity/determinism/drain, writer round trip (22 scalars, tensors, artifact blob on disk, text, trace, eval, PR, audio, plugin, layout, hparam metrics, session complete), resume purge above step 10 (+ second session row), `Provide resume_step=` error, rank gate no-op |
| Cross-language read | `tests/py/cross_read.py <dir>` | pass: Python `RunWatcher`/`RunDataProvider` read the C++-written run — status complete, tags, scalars, histogram bins == `np.histogram`, PNG blob decodes (2x2 RGB, pixel (1,2,3)), PR curve, audio analysis (peak_db −50.31), text, traces, eval, hparams, last, session id |
| Route parity vs Python server | `scripts/route_parity.py build/serenityboardd` | **0 mismatches** over 95 route variants across two runs (runs/tags/metrics/hparams/scalars ×5 axes+downsamples/last/histograms/distributions/text/traces/eval/artifacts/images/pr-curves/audio/custom-scalars/graphs/meshes/embeddings/notes/export JSON/blob 400+404/compare scalars/eval/hparams/plugins/missing run 404), blob bytes identical, CSV export identical, notes written by native read by Python, 404 envelope byte-exact |
| WebSocket | same script | native: 101 upgrade, accept key correct, subscribe → 46 scalar points + trace + eval messages while a Python writer appended (the Python server could not be probed: its venv lacks a WebSocket library for uvicorn) |
| Compiler ingest | `build/sbingest …` on real outputs | H3 runner log (19 H3_STEP + H3_PROFILE) → 87 scalars/19 traces; Krea 2 report + log → 51 scalars/57 traces; accepted FLUX.2 report → 16 scalars/15 traces; difbench document → 20 scalars/12 stage traces + summary text; all readable by the Python provider (denoiser_ms per step, stage walls) |
| Frontend headless | `scripts/ui_drive.mjs` + `ui_check*.js` | native server serves the unmodified UI: runs listed with status, "Connected" WebSocket badge, tags populate on run select, ECharts render on Scalars/Histograms/PR Curves/Audio tabs (screenshots in the scratchpad during the session) |

Measured, not claimed: compile of the native tree ~40 s cold (sqlite amalgamation dominates);
`serenityboardd` binary 2.6 MB; server answers the parity suite in well under a second
per route; no Python, no torch, no numpy at runtime.
