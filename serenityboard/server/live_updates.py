"""WebSocket live-update manager with subscription-scoped polling."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from serenityboard.server.run_manager import RunWatcher

__all__ = ["LiveUpdateManager", "SubscriptionFilter"]

logger = logging.getLogger(__name__)


@dataclass
class SubscriptionFilter:
    """Holds the set of runs and tag patterns a single WebSocket cares about."""

    runs: list[str] = field(default_factory=list)
    tag_patterns: set[str] = field(default_factory=set)
    kinds: set[str] = field(default_factory=lambda: {"scalar"})


class LiveUpdateManager:
    """Manages WebSocket subscriptions and pushes incremental scalar data.

    Background poller tails ONLY runs/tags that have active subscribers.
    When the last subscriber for a run disconnects, polling stops for that run.
    """

    def __init__(self) -> None:
        self._subscribers: dict[Any, SubscriptionFilter] = {}
        self._watcher: RunWatcher | None = None
        self._known_sessions: dict[str, str | None] = {}  # run_name -> session_id

    def set_watcher(self, watcher: RunWatcher) -> None:
        """Set the RunWatcher reference for provider lookups."""
        self._watcher = watcher

    # ── subscription management ───────────────────────────────────────

    def subscribe(self, ws: Any, filt: SubscriptionFilter) -> None:
        """Register a WebSocket with its subscription filter."""
        self._subscribers[ws] = filt

    def unsubscribe(self, ws: Any) -> None:
        """Remove a WebSocket subscription."""
        self._subscribers.pop(ws, None)

    # ── active subscription aggregation ───────────────────────────────

    def _active_subscriptions(self) -> dict[str, set[str]]:
        """Build {run_name: {tag_patterns}} from all current subscribers."""
        result: dict[str, set[str]] = {}
        for filt in self._subscribers.values():
            for run in filt.runs:
                if run not in result:
                    result[run] = set()
                result[run].update(filt.tag_patterns)
        return result

    def _active_kinds(self) -> dict[str, set[str]]:
        """Build {run_name: {kinds}} from all current subscribers."""
        result: dict[str, set[str]] = {}
        for filt in self._subscribers.values():
            for run in filt.runs:
                if run not in result:
                    result[run] = set()
                result[run].update(filt.kinds)
        return result

    # ── polling engine ────────────────────────────────────────────────

    async def poll_and_push(self) -> None:
        """Poll subscribed runs for new data and push to matching clients.

        Called every ~1s by a background task. Only queries providers for
        runs that have at least one active subscriber.
        """
        if not self._subscribers:
            return
        if self._watcher is None:
            return

        active = self._active_subscriptions()
        if not active:
            return

        active_kinds = self._active_kinds()

        for run_name, patterns in active.items():
            provider = self._watcher.get_provider(run_name)
            if provider is None:
                continue

            # ── session change detection ──────────────────────────
            try:
                current_session = provider.get_active_session_id()
            except Exception:
                current_session = None

            old_session = self._known_sessions.get(run_name)
            if old_session is not None and current_session != old_session:
                # Session changed — notify subscribers with resume_step
                resume_step = None
                try:
                    resume_step = provider.get_resume_step(current_session)
                except Exception:
                    pass
                session_msg = {
                    "type": "session_changed",
                    "run": run_name,
                    "old_session_id": old_session,
                    "new_session_id": current_session,
                    "resume_step": resume_step,
                }
                await self._push_to_run_subscribers(run_name, session_msg)
            self._known_sessions[run_name] = current_session

            kinds_for_run = active_kinds.get(run_name, {"scalar"})

            # ── scalar polling ────────────────────────────────────
            if "scalar" in kinds_for_run:
                try:
                    all_tags = provider.get_tags()
                except Exception:
                    logger.debug("Failed to get tags for run %s", run_name)
                    all_tags = {}

                scalar_tags = all_tags.get("scalars", [])
                matched_tags = _match_tags(scalar_tags, patterns)

                for tag in matched_tags:
                    try:
                        rows = provider.read_scalars_incremental(tag)
                    except Exception:
                        logger.debug("Failed to read tag %s for run %s", tag, run_name)
                        continue
                    if not rows:
                        continue

                    points = [
                        {"step": r[0], "wall_time": r[1], "value": r[2]} for r in rows
                    ]
                    message = {
                        "type": "scalar",
                        "run": run_name,
                        "tag": tag,
                        "session_id": current_session,
                        "points": points,
                    }

                    # Push to every subscriber that cares about this run+tag+kind
                    for ws, filt in list(self._subscribers.items()):
                        if run_name not in filt.runs:
                            continue
                        if "scalar" not in filt.kinds:
                            continue
                        if filt.tag_patterns and not any(fnmatch(tag, p) for p in filt.tag_patterns):
                            continue
                        try:
                            await ws.send_json(message)
                        except Exception:
                            logger.debug("Failed to send to subscriber, removing.")
                            self._subscribers.pop(ws, None)

            # ── trace polling ─────────────────────────────────────
            if "trace" in kinds_for_run:
                try:
                    events = provider.read_trace_events_incremental()
                except Exception:
                    logger.debug("Failed to read traces for run %s", run_name)
                    events = []

                if events:
                    message = {
                        "type": "trace",
                        "run": run_name,
                        "session_id": current_session,
                        "events": events,
                    }
                    await self._push_to_kind_subscribers(run_name, "trace", message)

            # ── eval polling ──────────────────────────────────────
            if "eval" in kinds_for_run:
                try:
                    results = provider.read_eval_results_incremental()
                except Exception:
                    logger.debug("Failed to read evals for run %s", run_name)
                    results = []

                if results:
                    message = {
                        "type": "eval",
                        "run": run_name,
                        "session_id": current_session,
                        "results": results,
                    }
                    await self._push_to_kind_subscribers(run_name, "eval", message)

    # ── push helpers ──────────────────────────────────────────────────

    async def _push_to_run_subscribers(self, run_name: str, message: dict) -> None:
        """Push a message to all subscribers watching this run."""
        for ws, filt in list(self._subscribers.items()):
            if run_name not in filt.runs:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                logger.debug("Failed to send to subscriber, removing.")
                self._subscribers.pop(ws, None)

    async def _push_to_kind_subscribers(
        self, run_name: str, kind: str, message: dict
    ) -> None:
        """Push a message to subscribers watching this run+kind."""
        for ws, filt in list(self._subscribers.items()):
            if run_name not in filt.runs:
                continue
            if kind not in filt.kinds:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                logger.debug("Failed to send to subscriber, removing.")
                self._subscribers.pop(ws, None)


def _match_tags(tags: list[str], patterns: set[str]) -> list[str]:
    """Return tags that match at least one fnmatch pattern.

    If patterns is empty, all tags are returned (wildcard behavior).
    """
    if not patterns:
        return list(tags)
    matched: list[str] = []
    for tag in tags:
        for pattern in patterns:
            if fnmatch(tag, pattern):
                matched.append(tag)
                break
    return matched
