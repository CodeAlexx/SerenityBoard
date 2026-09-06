# SerenityBoard native backend (C++) — plan

Assignment (Alex, 2026-09-05): port the SerenityBoard backend to C++ for use with the
Diffusion Compiler. The JS frontend stays as-is; the Python package remains the
**oracle** for every contract (schema, routes, WebSocket, writer semantics).

## Layout

```
native/
  CMakeLists.txt             C++20, -Wall -Wextra -Wpedantic -Werror; targets below
  third_party/               sqlite 3.46 amalgamation (public domain), nlohmann/json 3.11.3 (MIT)
  include/serenityboard/     public headers (+ C ABI serenityboard.h for compiler tools)
  src/core/                  sbcore: db/schema v2 (+ migrations), blob storage, PNG (zlib), WAV,
                             reservoir, session guard, async SummaryWriter, system metrics, C API
  src/server/                sbserver: own HTTP/1.1 + WebSocket server, data provider, run manager,
                             live updates (1 s poll), all /api routes, static frontend
  src/ingest/                Diffusion Compiler adapter: telemetry JSON, --report files, step lines,
                             diftrain losses -> board.db rows
  tools/                     serenityboardd (serve), sbingest
  tests/                     C++ gates + Python cross-checks (Python reads C++ dbs and vice versa,
                             route-by-route JSON diff against the Python server on the same logdir)
```

## Contracts to reproduce exactly (from the recon of the Python package @ 048eac9)

- On-disk: `<logdir>/<run>/board.db` (+WAL) and `blobs/<sha256[:16]>.<ext>`; run discovery
  = any dir with `board.db`, depth <= 4, nested names joined with `__`.
- Schema v2 DDL verbatim (WITHOUT ROWID tables, indexes), pragmas WAL/synchronous NORMAL/
  busy_timeout 5000; metadata values JSON-encoded; `run_notes` created lazily by the server.
- Writer: async single writer thread, unbounded batch per drain, INSERT OR REPLACE, retries
  0.1/0.5/2.0 s on SQLITE_BUSY-class errors then sticky error; per-tag reservoirs
  (deterministic seed 0, always_keep_last) drained on flush/close only; session guard
  (resume purge, crashed/complete transitions); histogram = NumPy uniform binning
  `[left,right,count]` float64; PR curve `(6,n)` float64; audio int16 WAV; image PNG via zlib;
  rank gate (SB_RANK/RANK/LOCAL_RANK); system metrics thread (own step counter, >= 5 s).
- Server: every route in section 3 of the recon, error envelope
  `{"error":{"code","message","details":{}}}`, no-cache headers on `/`, `.js`, `.css`, `.html`,
  frontend served from `../serenityboard/frontend`, run status derivation (stopped/completed/
  empty after 300 s), export CSV/JSON formats, distributions basis points, audio analysis
  (waveform/spectrogram/peak/rms), meshes/embeddings single-object form, notes upsert.
- WebSocket `/ws/live`: `{"subscribe":{runs,tags,kinds}}`; pushes `scalar`/`trace`/`eval`/
  `session_changed` with the exact field sets; fnmatch tag globs; 1 s poll.
- LoRA inspector routes: need safetensors reading + SVD in C++ — Phase 2 (initially 503 with
  the same message the Python side uses when torch is missing).

## Compiler integration

- `serenityboard.h` C ABI so `difdittrain`/`diftrain` (or any tool) can log scalars/images
  live by linking `sbcore` (no Python, no torch).
- `sbingest`: reads what the compiler already writes without linking — telemetry documents,
  `--report` JSON, `H3_STEP`/`FLUX2_NATIVE_STEP`/`KREA2_NATIVE_STEP` lines from stdout logs,
  `DIF_TRACE_FILE` runtime traces, diftrain losses `.diftensor` — into scalars/trace_events/
  artifacts so the dashboard shows compiler runs today. (Details after the telemetry recon.)

## Gates

1. Schema: C++ writer db read by Python `RunDataProvider`; Python writer db read by C++ server.
2. Routes: Python server and C++ server on the same logdir; JSON diff of every route with
   query variants (`scripts/route_parity.py`). Byte-equal bodies where deterministic.
3. WebSocket: subscribe → scalar/session_changed messages match.
4. Writer semantics: C++ port of the relevant Python tests (session guard, reservoir,
   disk error, encoders) plus histogram/pr-curve numeric equality vs NumPy.
5. Compiler run: a real compiler job ingested and displayed; frontend driven headlessly.
