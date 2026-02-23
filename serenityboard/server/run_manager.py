"""Run discovery and lifecycle management via filesystem polling."""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time

from serenityboard.server.data_provider import RunDataProvider

# If a "running" run has no new data for this many seconds, mark it "completed".
STALE_TIMEOUT_S = 300  # 5 minutes

__all__ = ["RunWatcher"]

logger = logging.getLogger(__name__)


class RunWatcher:
    """Periodically scans logdir for new board.db files. Detects new/removed runs.

    Uses filesystem polling (5s default). Each subdirectory of *logdir* that
    contains a ``board.db`` file is treated as a run. The subdirectory name
    becomes the run name.
    """

    def __init__(self, logdir: str, poll_interval: float = 5.0) -> None:
        self.logdir = logdir
        self.poll_interval = poll_interval
        self.known_runs: dict[str, RunDataProvider] = {}  # run_name -> provider
        self._callbacks_add: list = []  # callbacks for new runs
        self._callbacks_remove: list = []  # callbacks for removed runs

    # ------------------------------------------------------------------
    # Callback registration
    # ------------------------------------------------------------------

    def on_add(self, callback) -> None:
        """Register callback for new run detection."""
        self._callbacks_add.append(callback)

    def on_remove(self, callback) -> None:
        """Register callback for run removal."""
        self._callbacks_remove.append(callback)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def _find_run_dbs(self) -> dict[str, str]:
        """Recursively find board.db files up to 4 levels deep.

        Returns {run_name: db_path} where run_name is a URL-safe key
        derived from the relative path (slashes replaced with ``__``).
        Falls back to basename when the db sits directly inside logdir.
        """
        found: dict[str, str] = {}
        if not os.path.isdir(self.logdir):
            return found
        logdir = self.logdir.rstrip(os.sep) + os.sep
        for root, dirs, files in os.walk(self.logdir):
            # Limit depth to 4 levels
            depth = root[len(self.logdir):].count(os.sep)
            if depth >= 4:
                dirs.clear()
                continue
            if "board.db" in files:
                db_path = os.path.join(root, "board.db")
                rel = root[len(logdir):]
                if rel:
                    # Replace path separators so run name is URL-safe.
                    run_name = rel.replace(os.sep, "__")
                else:
                    run_name = os.path.basename(root)
                found[run_name] = db_path
        return found

    def scan_once(self) -> tuple[list[str], list[str]]:
        """Synchronous scan. Returns (added, removed) run names."""
        added: list[str] = []
        removed: list[str] = []

        found = self._find_run_dbs()
        current = set(found.keys())

        for name, db_path in found.items():
            if name not in self.known_runs:
                try:
                    provider = RunDataProvider(db_path)
                    provider.get_run_info()
                    self.known_runs[name] = provider
                    added.append(name)
                except Exception:
                    logger.warning("Failed to open %s", db_path)

        gone = set(self.known_runs) - current
        for name in gone:
            provider = self.known_runs.pop(name)
            provider.close()
            removed.append(name)

        return added, removed

    async def scan_loop(self) -> None:
        """Background async scan loop."""
        while True:
            added, removed = self.scan_once()
            for name in added:
                for cb in self._callbacks_add:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(name)
                    else:
                        cb(name)
            for name in removed:
                for cb in self._callbacks_remove:
                    if asyncio.iscoroutinefunction(cb):
                        await cb(name)
                    else:
                        cb(name)
            await asyncio.sleep(self.poll_interval)

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_runs(self) -> list[dict]:
        """Return list of run info dicts for all known runs.

        Auto-detects stale runs and classifies as:
        - "completed": reached max_steps (or writer.close() was called)
        - "stopped":   no new data for 5min but didn't reach max_steps
        - "running":   actively logging
        """
        now = time.time()
        result: list[dict] = []
        for name, provider in sorted(self.known_runs.items()):
            try:
                info = provider.get_run_info()
                status = info.get("status", "unknown")
                last_wt, last_step = provider.get_last_activity()
                hparams = info.get("hparams", {})
                max_steps = hparams.get("max_steps")

                # Auto-detect finished/stopped runs that never called writer.close()
                if status == "running":
                    if last_wt is not None and now - last_wt > STALE_TIMEOUT_S:
                        if max_steps and last_step is not None and last_step >= max_steps:
                            status = "completed"
                        else:
                            status = "stopped"
                    elif last_wt is None:
                        # Never logged any data — check if start_time is stale
                        st = info.get("start_time")
                        if st is not None and now - st > STALE_TIMEOUT_S:
                            status = "empty"

                result.append({
                    "name": name,
                    "start_time": info.get("start_time"),
                    "status": status,
                    "last_activity": last_wt,
                    "last_step": last_step,
                    "max_steps": max_steps,
                    "active_session_id": info.get("active_session_id"),
                    "hparams": hparams,
                })
            except Exception:
                result.append({
                    "name": name,
                    "start_time": None,
                    "status": "error",
                    "last_activity": None,
                    "last_step": None,
                    "max_steps": None,
                    "active_session_id": None,
                    "hparams": {},
                })
        return result

    def get_provider(self, run_name: str) -> RunDataProvider | None:
        """Return the data provider for a specific run, or None."""
        return self.known_runs.get(run_name)

    def delete_run(self, run_name: str) -> bool:
        """Delete a run's data directory and remove from registry.

        Returns True if deleted, False if not found.
        """
        provider = self.known_runs.get(run_name)
        if provider is None:
            return False
        run_dir = provider.run_dir
        provider.close()
        del self.known_runs[run_name]
        if os.path.isdir(run_dir):
            shutil.rmtree(run_dir)
        return True

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close all providers and clear state."""
        for provider in self.known_runs.values():
            provider.close()
        self.known_runs.clear()
