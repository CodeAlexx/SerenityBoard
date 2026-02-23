"""Automations engine service layer for SerenityBoard ops."""
from __future__ import annotations

import fnmatch
import json
import time
import uuid

from serenityboard.ops.db import OpsDB

__all__ = ["AutomationsService"]

VALID_TRIGGER_TYPES = frozenset(
    ["metric_threshold", "run_state", "artifact_alias", "sweep_status"]
)
VALID_ACTION_TYPES = frozenset(
    ["webhook", "slack", "email", "registry_alias_update"]
)
VALID_COMPARATORS = frozenset(["==", "!=", ">", ">=", "<", "<="])
VALID_DISPATCH_STATUSES = frozenset(["queued", "sent", "failed", "skipped"])

DEFAULT_MAX_DISPATCH_ATTEMPTS = 5
_BACKOFF_BASE = 2  # exponential base in seconds


class AutomationsService:
    """Rule CRUD, event evaluation, dispatch management."""

    def __init__(self, db: OpsDB) -> None:
        self._db = db

    # ── Rule CRUD ───────────────────────────────────────────────────

    def create_rule(
        self,
        name: str,
        trigger_type: str,
        trigger: dict,
        action_type: str,
        action: dict,
        enabled: bool = True,
        cooldown_seconds: int = 0,
        dedupe_window_sec: int = 300,
    ) -> dict:
        if trigger_type not in VALID_TRIGGER_TYPES:
            raise ValueError(f"trigger_type must be one of {sorted(VALID_TRIGGER_TYPES)}")
        if action_type not in VALID_ACTION_TYPES:
            raise ValueError(f"action_type must be one of {sorted(VALID_ACTION_TYPES)}")
        now = time.time()
        rid = uuid.uuid4().hex
        self._db.conn.execute(
            "INSERT INTO automation_rules "
            "(rule_id, name, enabled, trigger_type, trigger_json, "
            "action_type, action_json, cooldown_seconds, dedupe_window_sec, "
            "created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                rid,
                name,
                int(enabled),
                trigger_type,
                json.dumps(trigger),
                action_type,
                json.dumps(action),
                cooldown_seconds,
                dedupe_window_sec,
                now,
                now,
            ),
        )
        self._db.conn.commit()
        return self.get_rule(rid)

    def get_rule(self, rule_id: str) -> dict | None:
        row = self._db.execute(
            "SELECT * FROM automation_rules WHERE rule_id = ?", (rule_id,)
        ).fetchone()
        return self._rule_to_dict(row) if row else None

    def list_rules(self) -> list[dict]:
        rows = self._db.execute(
            "SELECT * FROM automation_rules ORDER BY created_at"
        ).fetchall()
        return [self._rule_to_dict(r) for r in rows]

    def update_rule(self, rule_id: str, **kwargs) -> dict | None:
        rule = self.get_rule(rule_id)
        if not rule:
            return None
        sets, params = [], []
        if "enabled" in kwargs:
            sets.append("enabled = ?")
            params.append(int(kwargs["enabled"]))
        if "trigger" in kwargs:
            sets.append("trigger_json = ?")
            params.append(json.dumps(kwargs["trigger"]))
        if "action" in kwargs:
            sets.append("action_json = ?")
            params.append(json.dumps(kwargs["action"]))
        if "cooldown_seconds" in kwargs:
            sets.append("cooldown_seconds = ?")
            params.append(kwargs["cooldown_seconds"])
        if "dedupe_window_sec" in kwargs:
            sets.append("dedupe_window_sec = ?")
            params.append(kwargs["dedupe_window_sec"])
        if not sets:
            return rule
        now = time.time()
        sets.append("updated_at = ?")
        params.append(now)
        params.append(rule_id)
        self._db.conn.execute(
            f"UPDATE automation_rules SET {', '.join(sets)} WHERE rule_id = ?",
            tuple(params),
        )
        self._db.conn.commit()
        return self.get_rule(rule_id)

    def delete_rule(self, rule_id: str) -> bool:
        cur = self._db.conn.execute(
            "DELETE FROM automation_rules WHERE rule_id = ?", (rule_id,)
        )
        self._db.conn.commit()
        return cur.rowcount > 0

    # ── Event Evaluation ────────────────────────────────────────────

    def ingest_event(self, event_type: str, event_key: str, payload: dict) -> dict:
        """Ingest an event and evaluate against all matching rules."""
        now = time.time()
        eid = uuid.uuid4().hex
        self._db.conn.execute(
            "INSERT INTO automation_events "
            "(event_id, event_type, event_key, payload_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (eid, event_type, event_key, json.dumps(payload), now),
        )
        self._db.conn.commit()

        event = {
            "event_id": eid,
            "event_type": event_type,
            "event_key": event_key,
            "payload": payload,
            "created_at": now,
        }

        # Evaluate all enabled rules
        rules = self._db.execute(
            "SELECT * FROM automation_rules WHERE enabled = 1"
        ).fetchall()

        dispatched = []
        for row in rules:
            rule = self._rule_to_dict(row)
            if self._matches_trigger(rule, event):
                result = self._try_dispatch(rule, event)
                if result:
                    dispatched.append(result)

        event["dispatched"] = dispatched
        return event

    def _matches_trigger(self, rule: dict, event: dict) -> bool:
        """Check if event matches rule's trigger spec."""
        trigger_type = rule["trigger_type"]
        trigger = rule["trigger"]
        payload = event["payload"]

        if trigger_type == "metric_threshold":
            if event["event_type"] != "metric_threshold_candidate":
                return False
            metric_tag = payload.get("metric_tag")
            if metric_tag != trigger.get("metric_tag"):
                return False
            # Run selector
            run_name = payload.get("run_name", "")
            run_selector = trigger.get("run_selector", {})
            includes = run_selector.get("include", ["*"])
            excludes = run_selector.get("exclude", [])
            if not any(fnmatch.fnmatch(run_name, p) for p in includes):
                return False
            if any(fnmatch.fnmatch(run_name, p) for p in excludes):
                return False
            # Compare value
            value = payload.get("value")
            threshold = trigger.get("threshold")
            comparator = trigger.get("comparator", "<=")
            if value is None or threshold is None:
                return False
            return self._compare(value, comparator, threshold)

        if trigger_type == "run_state":
            if event["event_type"] != "run_state_change":
                return False
            states = trigger.get("states", [])
            return payload.get("state") in states

        if trigger_type == "artifact_alias":
            if event["event_type"] != "artifact_alias_change":
                return False
            return (
                payload.get("collection") == trigger.get("collection")
                and payload.get("alias") == trigger.get("alias")
            )

        if trigger_type == "sweep_status":
            if event["event_type"] != "sweep_status_change":
                return False
            return payload.get("status") in trigger.get("statuses", [])

        return False

    @staticmethod
    def _compare(value: float, comparator: str, threshold: float) -> bool:
        if comparator == "<=":
            return value <= threshold
        if comparator == "<":
            return value < threshold
        if comparator == ">=":
            return value >= threshold
        if comparator == ">":
            return value > threshold
        if comparator == "==":
            return value == threshold
        if comparator == "!=":
            return value != threshold
        return False

    def _try_dispatch(self, rule: dict, event: dict) -> dict | None:
        """Attempt to dispatch action for rule+event. Handles dedupe and cooldown."""
        rule_id = rule["rule_id"]
        event_id = event["event_id"]
        now = time.time()

        # Dedupe check: same rule + event_key within window
        dedupe_window = rule.get("dedupe_window_sec", 300)
        cutoff = now - dedupe_window
        existing = self._db.execute(
            "SELECT dispatch_id FROM automation_dispatch d "
            "JOIN automation_events e ON d.event_id = e.event_id "
            "WHERE d.rule_id = ? AND e.event_key = ? AND d.created_at > ? "
            "AND d.status != 'failed'",
            (rule_id, event["event_key"], cutoff),
        ).fetchone()
        if existing:
            return self._create_dispatch(rule_id, event_id, "skipped", "dedupe window active")

        # Cooldown check: time since last successful dispatch for this rule
        cooldown = rule.get("cooldown_seconds", 0)
        if cooldown > 0:
            last_sent = self._db.execute(
                "SELECT MAX(created_at) as last FROM automation_dispatch "
                "WHERE rule_id = ? AND status = 'sent'",
                (rule_id,),
            ).fetchone()
            if last_sent and last_sent["last"] and (now - last_sent["last"]) < cooldown:
                return self._create_dispatch(rule_id, event_id, "skipped", "cooldown active")

        # Queue for dispatch
        return self._create_dispatch(rule_id, event_id, "queued")

    def _create_dispatch(
        self, rule_id: str, event_id: str, status: str, error_text: str = ""
    ) -> dict:
        now = time.time()
        did = uuid.uuid4().hex
        next_attempt = now if status == "queued" else None
        try:
            self._db.conn.execute(
                "INSERT INTO automation_dispatch "
                "(dispatch_id, rule_id, event_id, status, error_text, "
                "attempt_count, next_attempt_at, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?, ?, ?)",
                (did, rule_id, event_id, status, error_text, next_attempt, now, now),
            )
            self._db.conn.commit()
        except Exception:
            # UNIQUE constraint on (rule_id, event_id) — skip if already exists
            return {"dispatch_id": did, "status": "skipped", "error_text": "already dispatched"}
        return {
            "dispatch_id": did,
            "rule_id": rule_id,
            "event_id": event_id,
            "status": status,
            "error_text": error_text,
        }

    # ── Dispatch Queue ──────────────────────────────────────────────

    def list_dispatch(
        self, status: str | None = None, limit: int = 100
    ) -> list[dict]:
        sql = "SELECT * FROM automation_dispatch"
        params: list = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self._db.execute(sql, tuple(params)).fetchall()
        return [dict(r) for r in rows]

    def process_dispatch(self, dispatch_id: str, success: bool, error_text: str = "") -> dict | None:
        """Mark dispatch as sent or failed with retry backoff."""
        row = self._db.execute(
            "SELECT * FROM automation_dispatch WHERE dispatch_id = ?",
            (dispatch_id,),
        ).fetchone()
        if not row:
            return None

        now = time.time()
        attempt = row["attempt_count"] + 1

        if success:
            self._db.conn.execute(
                "UPDATE automation_dispatch SET status = 'sent', "
                "attempt_count = ?, updated_at = ? WHERE dispatch_id = ?",
                (attempt, now, dispatch_id),
            )
        else:
            if attempt >= DEFAULT_MAX_DISPATCH_ATTEMPTS:
                new_status = "failed"
                next_at = None
            else:
                new_status = "queued"
                # Exponential backoff
                next_at = now + (_BACKOFF_BASE ** attempt)

            self._db.conn.execute(
                "UPDATE automation_dispatch SET status = ?, error_text = ?, "
                "attempt_count = ?, next_attempt_at = ?, updated_at = ? "
                "WHERE dispatch_id = ?",
                (new_status, error_text, attempt, next_at, now, dispatch_id),
            )
        self._db.conn.commit()

        updated = self._db.execute(
            "SELECT * FROM automation_dispatch WHERE dispatch_id = ?",
            (dispatch_id,),
        ).fetchone()
        return dict(updated)

    @staticmethod
    def _rule_to_dict(row) -> dict:
        return {
            **{
                k: row[k]
                for k in row.keys()
                if k not in ("trigger_json", "action_json")
            },
            "trigger": json.loads(row["trigger_json"]),
            "action": json.loads(row["action_json"]),
            "enabled": bool(row["enabled"]),
        }
