"""Sweeps service layer for SerenityBoard ops."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import time
import uuid

from serenityboard.ops.db import OpsDB

__all__ = ["SweepsService"]

VALID_METHODS = frozenset(["random", "grid", "bayes_v2"])
VALID_SWEEP_STATUSES = frozenset(
    ["pending", "running", "paused", "complete", "failed", "cancelled"]
)
VALID_TRIAL_STATUSES = frozenset(
    ["queued", "assigned", "running", "success", "failed", "aborted"]
)
VALID_AGENT_STATUSES = frozenset(["online", "offline", "draining"])
VALID_OBJECTIVES = frozenset(["min", "max"])

DEFAULT_LEASE_TIMEOUT = 600  # 10 min
DEFAULT_MAX_ATTEMPTS = 2


class SweepsService:
    """Sweep lifecycle, agent protocol, trial scheduling."""

    def __init__(self, db: OpsDB) -> None:
        self._db = db

    # ── Sweep CRUD ──────────────────────────────────────────────────

    def create_sweep(
        self,
        name: str,
        method: str,
        objective_metric: str,
        objective_mode: str,
        parameter_space: dict,
        max_trials: int,
        parallelism: int = 1,
    ) -> dict:
        if method not in VALID_METHODS:
            raise ValueError(f"method must be one of {sorted(VALID_METHODS)}")
        if objective_mode not in VALID_OBJECTIVES:
            raise ValueError(f"objective_mode must be one of {sorted(VALID_OBJECTIVES)}")
        now = time.time()
        sid = uuid.uuid4().hex

        self._db.conn.execute(
            "INSERT INTO sweeps "
            "(sweep_id, name, method, objective_metric, objective_mode, "
            "parameter_space, max_trials, parallelism, status, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)",
            (
                sid,
                name,
                method,
                objective_metric,
                objective_mode,
                json.dumps(parameter_space),
                max_trials,
                parallelism,
                now,
                now,
            ),
        )
        self._db.conn.commit()

        # Pre-generate trials
        self._generate_trials(sid, method, parameter_space, max_trials)

        return self.get_sweep(sid)

    def _generate_trials(
        self,
        sweep_id: str,
        method: str,
        parameter_space: dict,
        max_trials: int,
    ) -> None:
        """Pre-generate trial parameter assignments."""
        now = time.time()
        configs = self._sample_configs(sweep_id, method, parameter_space, max_trials)
        with self._db.conn:
            for i, params in enumerate(configs):
                tid = uuid.uuid4().hex
                self._db.conn.execute(
                    "INSERT INTO sweep_trials "
                    "(trial_id, sweep_id, params_json, status, created_at, updated_at) "
                    "VALUES (?, ?, ?, 'queued', ?, ?)",
                    (tid, sweep_id, json.dumps(params), now, now),
                )

    def _sample_configs(
        self,
        sweep_id: str,
        method: str,
        parameter_space: dict,
        max_trials: int,
    ) -> list[dict]:
        if method == "grid":
            return self._grid_configs(parameter_space, max_trials)
        elif method == "random":
            return self._random_configs(sweep_id, parameter_space, max_trials)
        return []  # bayes_v2 is post-V1

    def _grid_configs(self, parameter_space: dict, max_trials: int) -> list[dict]:
        """Deterministic product order for grid search."""
        param_names = sorted(parameter_space.keys())
        param_values = []
        for name in param_names:
            spec = parameter_space[name]
            ptype = spec.get("type", "categorical")
            if ptype == "categorical":
                param_values.append(spec["values"])
            elif ptype in ("uniform", "log_uniform"):
                # For grid, discretize into reasonable steps
                steps = spec.get("steps", min(max_trials, 5))
                lo, hi = spec["min"], spec["max"]
                if ptype == "log_uniform":
                    vals = [
                        math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / max(steps - 1, 1))
                        for i in range(steps)
                    ]
                else:
                    vals = [lo + i * (hi - lo) / max(steps - 1, 1) for i in range(steps)]
                param_values.append(vals)
            else:
                param_values.append(spec.get("values", []))

        configs = []
        for combo in itertools.product(*param_values):
            if len(configs) >= max_trials:
                break
            configs.append(dict(zip(param_names, combo)))
        return configs

    def _random_configs(
        self, sweep_id: str, parameter_space: dict, max_trials: int
    ) -> list[dict]:
        """Seeded RNG per sweep for reproducibility."""
        seed = int(hashlib.sha256(sweep_id.encode()).hexdigest()[:8], 16)
        rng = random.Random(seed)
        configs = []
        for _ in range(max_trials):
            config = {}
            for name, spec in parameter_space.items():
                ptype = spec.get("type", "categorical")
                if ptype == "categorical":
                    config[name] = rng.choice(spec["values"])
                elif ptype == "uniform":
                    config[name] = rng.uniform(spec["min"], spec["max"])
                elif ptype == "log_uniform":
                    log_val = rng.uniform(math.log(spec["min"]), math.log(spec["max"]))
                    config[name] = math.exp(log_val)
                else:
                    config[name] = rng.choice(spec.get("values", [0]))
            configs.append(config)
        return configs

    def get_sweep(self, sweep_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM sweeps WHERE sweep_id = ?", (sweep_id,)
        ).fetchone()
        if not row:
            return None
        return {
            **{k: row[k] for k in row.keys() if k != "parameter_space"},
            "parameter_space": json.loads(row["parameter_space"]),
        }

    def list_sweeps(self) -> list[dict]:
        rows = self._db.execute("SELECT * FROM sweeps ORDER BY created_at").fetchall()
        return [
            {
                **{k: r[k] for k in r.keys() if k != "parameter_space"},
                "parameter_space": json.loads(r["parameter_space"]),
            }
            for r in rows
        ]

    def _update_sweep_status(self, sweep_id: str, status: str) -> dict | None:
        now = time.time()
        self._db.conn.execute(
            "UPDATE sweeps SET status = ?, updated_at = ? WHERE sweep_id = ?",
            (status, now, sweep_id),
        )
        self._db.conn.commit()
        return self.get_sweep(sweep_id)

    def pause_sweep(self, sweep_id: str) -> dict | None:
        return self._update_sweep_status(sweep_id, "paused")

    def resume_sweep(self, sweep_id: str) -> dict | None:
        return self._update_sweep_status(sweep_id, "running")

    def cancel_sweep(self, sweep_id: str) -> dict | None:
        return self._update_sweep_status(sweep_id, "cancelled")

    # ── Trials ──────────────────────────────────────────────────────

    def list_trials(
        self,
        sweep_id: str,
        status: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        sql = "SELECT * FROM sweep_trials WHERE sweep_id = ?"
        params: list = [sweep_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        rows = self._db.execute(sql, tuple(params)).fetchall()
        return [self._trial_to_dict(r) for r in rows]

    def get_best_trials(self, sweep_id: str, top_k: int = 1) -> list[dict]:
        sweep = self.get_sweep(sweep_id)
        if not sweep:
            return []
        direction = "ASC" if sweep["objective_mode"] == "min" else "DESC"
        rows = self._db.execute(
            f"SELECT * FROM sweep_trials WHERE sweep_id = ? "
            f"AND status = 'success' AND objective_value IS NOT NULL "
            f"ORDER BY objective_value {direction} LIMIT ?",
            (sweep_id, top_k),
        ).fetchall()
        return [self._trial_to_dict(r) for r in rows]

    # ── Agent Protocol ──────────────────────────────────────────────

    def register_agent(self, labels: list[str] | None = None) -> dict:
        now = time.time()
        agent_id = uuid.uuid4().hex
        self._db.conn.execute(
            "INSERT INTO sweep_agents (agent_id, labels_json, last_heartbeat, status) "
            "VALUES (?, ?, ?, 'online')",
            (agent_id, json.dumps(labels or []), now),
        )
        self._db.conn.commit()
        return {"agent_id": agent_id, "status": "online"}

    def heartbeat(self, agent_id: str) -> dict:
        now = time.time()
        self._db.conn.execute(
            "UPDATE sweep_agents SET last_heartbeat = ? WHERE agent_id = ?",
            (now, agent_id),
        )
        self._db.conn.commit()
        return {"agent_id": agent_id, "last_heartbeat": now}

    def next_trial(self, agent_id: str) -> dict | None:
        """Assign next queued trial to agent. Returns trial or None.

        Only assigns trials from sweeps in 'pending' or 'running' status.
        Respects parallelism limits per sweep. Uses atomic UPDATE with
        status guard to prevent double-assignment race conditions.
        """
        # Reclaim stale leases first
        self._reclaim_stale_leases()

        now = time.time()
        with self._db.conn:
            # Retry on races: another worker can claim the same candidate first.
            for _ in range(8):
                # Select the oldest queued trial from active sweeps where
                # parallelism has available capacity.
                row = self._db.conn.execute(
                    "SELECT t.*, s.name AS sweep_name, s.status AS sweep_status, "
                    "s.parallelism AS sweep_parallelism "
                    "FROM sweep_trials t "
                    "JOIN sweeps s ON t.sweep_id = s.sweep_id "
                    "WHERE t.status = 'queued' "
                    "AND s.status IN ('pending', 'running') "
                    "AND (s.parallelism <= 0 OR ("
                    "  SELECT COUNT(*) FROM sweep_trials a "
                    "  WHERE a.sweep_id = t.sweep_id "
                    "    AND a.status IN ('assigned', 'running')"
                    ") < s.parallelism) "
                    "ORDER BY t.created_at LIMIT 1"
                ).fetchone()
                if not row:
                    return None

                trial_id = row["trial_id"]
                sweep_id = row["sweep_id"]
                parallelism = row["sweep_parallelism"]
                run_name = f"sweep_{row['sweep_name']}_trial_{trial_id[:8]}"

                # Atomic UPDATE with status + parallelism guard to prevent
                # double-assignment and parallelism overflow under contention.
                cur = self._db.conn.execute(
                    "UPDATE sweep_trials SET status = 'assigned', assigned_agent = ?, "
                    "run_name = ?, started_at = ?, updated_at = ? "
                    "WHERE trial_id = ? AND status = 'queued' "
                    "AND (? <= 0 OR ("
                    "  SELECT COUNT(*) FROM sweep_trials a "
                    "  WHERE a.sweep_id = ? "
                    "    AND a.status IN ('assigned', 'running')"
                    ") < ?)",
                    (
                        agent_id,
                        run_name,
                        now,
                        now,
                        trial_id,
                        parallelism,
                        sweep_id,
                        parallelism,
                    ),
                )
                if cur.rowcount == 0:
                    continue

                # Update sweep status to running if pending.
                if row["sweep_status"] == "pending":
                    self._db.conn.execute(
                        "UPDATE sweeps SET status = 'running', updated_at = ? "
                        "WHERE sweep_id = ?",
                        (now, sweep_id),
                    )

                return {
                    "trial_id": trial_id,
                    "sweep_id": sweep_id,
                    "run_name": run_name,
                    "params": json.loads(row["params_json"]),
                }

        return None

    def start_trial(self, trial_id: str) -> dict | None:
        now = time.time()
        self._db.conn.execute(
            "UPDATE sweep_trials SET status = 'running', started_at = ?, updated_at = ? "
            "WHERE trial_id = ?",
            (now, now, trial_id),
        )
        self._db.conn.commit()
        return self._get_trial(trial_id)

    def report_trial(self, trial_id: str, objective_value: float) -> dict | None:
        now = time.time()
        self._db.conn.execute(
            "UPDATE sweep_trials SET objective_value = ?, updated_at = ? WHERE trial_id = ?",
            (objective_value, now, trial_id),
        )
        self._db.conn.commit()
        return self._get_trial(trial_id)

    def finish_trial(self, trial_id: str) -> dict | None:
        now = time.time()
        self._db.conn.execute(
            "UPDATE sweep_trials SET status = 'success', finished_at = ?, updated_at = ? "
            "WHERE trial_id = ?",
            (now, now, trial_id),
        )
        self._db.conn.commit()
        self._check_sweep_completion(trial_id)
        return self._get_trial(trial_id)

    def fail_trial(self, trial_id: str, error_text: str = "") -> dict | None:
        now = time.time()
        trial = self._get_trial(trial_id)
        if not trial:
            return None

        attempt = trial["attempt_count"] + 1
        if attempt >= DEFAULT_MAX_ATTEMPTS:
            new_status = "aborted"
        else:
            new_status = "queued"  # Re-queue for retry

        self._db.conn.execute(
            "UPDATE sweep_trials SET status = ?, error_text = ?, "
            "attempt_count = ?, assigned_agent = NULL, finished_at = ?, updated_at = ? "
            "WHERE trial_id = ?",
            (new_status, error_text, attempt, now, now, trial_id),
        )
        self._db.conn.commit()
        return self._get_trial(trial_id)

    def _reclaim_stale_leases(self) -> int:
        """Return stale assigned trials to queue."""
        cutoff = time.time() - DEFAULT_LEASE_TIMEOUT
        cur = self._db.conn.execute(
            "UPDATE sweep_trials SET status = 'queued', assigned_agent = NULL, updated_at = ? "
            "WHERE status = 'assigned' AND started_at IS NOT NULL AND started_at < ?",
            (time.time(), cutoff),
        )
        if cur.rowcount > 0:
            self._db.conn.commit()
        return cur.rowcount

    def _check_sweep_completion(self, trial_id: str) -> None:
        """Check if all trials in sweep are done."""
        trial = self._get_trial(trial_id)
        if not trial:
            return
        sweep_id = trial["sweep_id"]
        row = self._db.execute(
            "SELECT COUNT(*) as c FROM sweep_trials "
            "WHERE sweep_id = ? AND status NOT IN ('success', 'failed', 'aborted')",
            (sweep_id,),
        ).fetchone()
        if row["c"] == 0:
            now = time.time()
            self._db.conn.execute(
                "UPDATE sweeps SET status = 'complete', updated_at = ? WHERE sweep_id = ?",
                (now, sweep_id),
            )
            self._db.conn.commit()

    def _get_trial(self, trial_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM sweep_trials WHERE trial_id = ?", (trial_id,)
        ).fetchone()
        return self._trial_to_dict(row) if row else None

    @staticmethod
    def _trial_to_dict(row) -> dict:
        return {
            **{k: row[k] for k in row.keys() if k != "params_json"},
            "params": json.loads(row["params_json"]),
        }
