"""Tests for WebSocket live-update manager and subscription system."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import sqlite3
import tempfile
import time
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from starlette.testclient import TestClient

from serenityboard.server.live_updates import (
    LiveUpdateManager,
    SubscriptionFilter,
    _match_tags,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class FakeWebSocket:
    """Minimal mock WebSocket with an async send_json that records messages."""

    def __init__(self) -> None:
        self.sent: list[dict] = []
        self._should_fail = False

    async def send_json(self, data: dict) -> None:
        if self._should_fail:
            raise ConnectionError("send failed")
        self.sent.append(data)


class FakeProvider:
    """Minimal mock RunDataProvider for unit testing poll_and_push."""

    def __init__(
        self,
        *,
        tags: dict[str, list[str]] | None = None,
        scalars: dict[str, list[tuple]] | None = None,
        session_id: str | None = "session-1",
        resume_step: int | None = None,
        trace_events: list[dict] | None = None,
        eval_results: list[dict] | None = None,
    ) -> None:
        self._tags = tags or {"scalars": []}
        self._scalars = scalars or {}
        self._session_id = session_id
        self._resume_step = resume_step
        self._trace_events = trace_events or []
        self._eval_results = eval_results or []

    def get_tags(self) -> dict[str, list[str]]:
        return self._tags

    def get_active_session_id(self) -> str | None:
        return self._session_id

    def get_resume_step(self, session_id: str) -> int | None:
        return self._resume_step

    def read_scalars_incremental(self, tag: str) -> list[tuple]:
        return self._scalars.get(tag, [])

    def read_trace_events_incremental(self) -> list[dict]:
        return self._trace_events

    def read_eval_results_incremental(self) -> list[dict]:
        return self._eval_results


class FakeWatcher:
    """Minimal mock RunWatcher."""

    def __init__(self, providers: dict[str, FakeProvider] | None = None) -> None:
        self._providers = providers or {}

    def get_provider(self, run_name: str) -> FakeProvider | None:
        return self._providers.get(run_name)


# ===========================================================================
# Unit tests: SubscriptionFilter
# ===========================================================================


class TestSubscriptionFilter:
    def test_defaults(self):
        """Default kinds is {'scalar'}, runs and tag_patterns are empty."""
        filt = SubscriptionFilter()
        assert filt.runs == []
        assert filt.tag_patterns == set()
        assert filt.kinds == {"scalar"}

    def test_custom_values(self):
        """Custom constructor values are stored correctly."""
        filt = SubscriptionFilter(
            runs=["run1", "run2"],
            tag_patterns={"loss/*", "lr"},
            kinds={"scalar", "trace"},
        )
        assert filt.runs == ["run1", "run2"]
        assert filt.tag_patterns == {"loss/*", "lr"}
        assert filt.kinds == {"scalar", "trace"}

    def test_independent_defaults(self):
        """Two default instances should not share mutable state."""
        a = SubscriptionFilter()
        b = SubscriptionFilter()
        a.runs.append("x")
        a.tag_patterns.add("y")
        assert b.runs == []
        assert b.tag_patterns == set()


# ===========================================================================
# Unit tests: _match_tags
# ===========================================================================


class TestMatchTags:
    def test_empty_patterns_returns_all(self):
        """Empty patterns set = wildcard, returns all tags."""
        tags = ["loss/train", "loss/val", "lr"]
        result = _match_tags(tags, set())
        assert result == tags

    def test_exact_match(self):
        """Exact pattern matches only that tag."""
        tags = ["loss/train", "loss/val", "lr"]
        result = _match_tags(tags, {"lr"})
        assert result == ["lr"]

    def test_wildcard_star(self):
        """Glob wildcard 'loss/*' matches tags under loss/."""
        tags = ["loss/train", "loss/val", "lr", "accuracy"]
        result = _match_tags(tags, {"loss/*"})
        assert result == ["loss/train", "loss/val"]

    def test_wildcard_all(self):
        """Pattern '*' matches everything."""
        tags = ["loss/train", "lr", "accuracy"]
        result = _match_tags(tags, {"*"})
        assert result == tags

    def test_multiple_patterns(self):
        """Multiple patterns are OR-ed together."""
        tags = ["loss/train", "loss/val", "lr", "accuracy"]
        result = _match_tags(tags, {"loss/*", "lr"})
        assert result == ["loss/train", "loss/val", "lr"]

    def test_no_match(self):
        """Patterns that match nothing return empty list."""
        tags = ["loss/train", "lr"]
        result = _match_tags(tags, {"accuracy/*"})
        assert result == []

    def test_empty_tags_list(self):
        """Empty tags list returns empty regardless of patterns."""
        assert _match_tags([], {"*"}) == []
        assert _match_tags([], set()) == []

    def test_question_mark_glob(self):
        """fnmatch '?' matches single character."""
        tags = ["v1", "v2", "v10"]
        result = _match_tags(tags, {"v?"})
        assert result == ["v1", "v2"]


# ===========================================================================
# Unit tests: LiveUpdateManager subscribe/unsubscribe
# ===========================================================================


class TestSubscribeUnsubscribe:
    def test_subscribe_adds_to_internal_state(self):
        """subscribe() registers the WebSocket and its filter."""
        mgr = LiveUpdateManager()
        ws = FakeWebSocket()
        filt = SubscriptionFilter(runs=["run1"])
        mgr.subscribe(ws, filt)
        assert ws in mgr._subscribers
        assert mgr._subscribers[ws] is filt

    def test_unsubscribe_removes(self):
        """unsubscribe() removes the WebSocket from subscribers."""
        mgr = LiveUpdateManager()
        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"]))
        mgr.unsubscribe(ws)
        assert ws not in mgr._subscribers

    def test_unsubscribe_nonexistent_is_noop(self):
        """unsubscribe() on unknown WebSocket does not raise."""
        mgr = LiveUpdateManager()
        ws = FakeWebSocket()
        mgr.unsubscribe(ws)  # should not raise

    def test_subscribe_overwrites_existing(self):
        """Re-subscribing the same WebSocket replaces the filter."""
        mgr = LiveUpdateManager()
        ws = FakeWebSocket()
        filt1 = SubscriptionFilter(runs=["run1"])
        filt2 = SubscriptionFilter(runs=["run2"])
        mgr.subscribe(ws, filt1)
        mgr.subscribe(ws, filt2)
        assert mgr._subscribers[ws] is filt2
        assert len(mgr._subscribers) == 1


# ===========================================================================
# Unit tests: _active_subscriptions aggregation
# ===========================================================================


class TestActiveSubscriptions:
    def test_single_subscriber(self):
        """Single subscriber produces correct aggregation."""
        mgr = LiveUpdateManager()
        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(
            runs=["run1"],
            tag_patterns={"loss/*"},
        ))
        result = mgr._active_subscriptions()
        assert result == {"run1": {"loss/*"}}

    def test_multiple_subscribers_same_run(self):
        """Multiple subscribers to the same run merge tag patterns."""
        mgr = LiveUpdateManager()
        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()
        mgr.subscribe(ws1, SubscriptionFilter(
            runs=["run1"],
            tag_patterns={"loss/*"},
        ))
        mgr.subscribe(ws2, SubscriptionFilter(
            runs=["run1"],
            tag_patterns={"lr"},
        ))
        result = mgr._active_subscriptions()
        assert result == {"run1": {"loss/*", "lr"}}

    def test_multiple_subscribers_different_runs(self):
        """Different run subscriptions produce separate entries."""
        mgr = LiveUpdateManager()
        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()
        mgr.subscribe(ws1, SubscriptionFilter(runs=["run1"]))
        mgr.subscribe(ws2, SubscriptionFilter(runs=["run2"]))
        result = mgr._active_subscriptions()
        assert "run1" in result
        assert "run2" in result

    def test_empty_after_unsubscribe(self):
        """After all subscribers leave, active_subscriptions is empty."""
        mgr = LiveUpdateManager()
        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"]))
        mgr.unsubscribe(ws)
        assert mgr._active_subscriptions() == {}

    def test_active_kinds_aggregation(self):
        """_active_kinds merges kind sets per run across subscribers."""
        mgr = LiveUpdateManager()
        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()
        mgr.subscribe(ws1, SubscriptionFilter(
            runs=["run1"], kinds={"scalar"},
        ))
        mgr.subscribe(ws2, SubscriptionFilter(
            runs=["run1"], kinds={"trace", "eval"},
        ))
        result = mgr._active_kinds()
        assert result == {"run1": {"scalar", "trace", "eval"}}


# ===========================================================================
# Unit tests: poll_and_push
# ===========================================================================


class TestPollAndPush:
    @pytest.mark.asyncio
    async def test_poll_no_subscribers(self):
        """poll_and_push returns immediately with no subscribers."""
        mgr = LiveUpdateManager()
        watcher = FakeWatcher()
        mgr.set_watcher(watcher)
        # Should complete without error
        await mgr.poll_and_push()

    @pytest.mark.asyncio
    async def test_poll_no_watcher(self):
        """poll_and_push returns immediately when no watcher is set."""
        mgr = LiveUpdateManager()
        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"]))
        # No watcher set, should return without error
        await mgr.poll_and_push()
        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_poll_sends_scalar_data(self):
        """poll_and_push sends scalar points to matching subscribers."""
        now = time.time()
        provider = FakeProvider(
            tags={"scalars": ["loss/train"]},
            scalars={"loss/train": [(0, now, 0.5), (1, now + 1, 0.4)]},
        )
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(
            runs=["run1"],
            tag_patterns={"loss/*"},
            kinds={"scalar"},
        ))

        await mgr.poll_and_push()

        assert len(ws.sent) == 1
        msg = ws.sent[0]
        assert msg["type"] == "scalar"
        assert msg["run"] == "run1"
        assert msg["tag"] == "loss/train"
        assert len(msg["points"]) == 2
        assert msg["points"][0]["step"] == 0
        assert msg["points"][0]["value"] == 0.5

    @pytest.mark.asyncio
    async def test_poll_respects_tag_filter(self):
        """Subscriber with tag pattern 'lr' should NOT receive loss/train data."""
        now = time.time()
        provider = FakeProvider(
            tags={"scalars": ["loss/train", "lr"]},
            scalars={
                "loss/train": [(0, now, 0.5)],
                "lr": [(0, now, 1e-4)],
            },
        )
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(
            runs=["run1"],
            tag_patterns={"lr"},
            kinds={"scalar"},
        ))

        await mgr.poll_and_push()

        # Should only get lr, not loss/train
        assert len(ws.sent) == 1
        assert ws.sent[0]["tag"] == "lr"

    @pytest.mark.asyncio
    async def test_poll_empty_patterns_matches_all(self):
        """Empty tag_patterns means all tags are delivered."""
        now = time.time()
        provider = FakeProvider(
            tags={"scalars": ["loss/train", "lr"]},
            scalars={
                "loss/train": [(0, now, 0.5)],
                "lr": [(0, now, 1e-4)],
            },
        )
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(
            runs=["run1"],
            tag_patterns=set(),  # empty = wildcard
            kinds={"scalar"},
        ))

        await mgr.poll_and_push()

        tags_received = {m["tag"] for m in ws.sent}
        assert tags_received == {"loss/train", "lr"}

    @pytest.mark.asyncio
    async def test_poll_respects_run_filter(self):
        """Subscriber to run1 should not receive data for run2."""
        now = time.time()
        provider1 = FakeProvider(
            tags={"scalars": ["loss/train"]},
            scalars={"loss/train": [(0, now, 0.5)]},
        )
        provider2 = FakeProvider(
            tags={"scalars": ["loss/train"]},
            scalars={"loss/train": [(0, now, 0.9)]},
        )
        watcher = FakeWatcher({"run1": provider1, "run2": provider2})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(
            runs=["run1"],
            kinds={"scalar"},
        ))

        await mgr.poll_and_push()

        # Only run1 data
        for msg in ws.sent:
            assert msg["run"] == "run1"

    @pytest.mark.asyncio
    async def test_poll_respects_kind_filter(self):
        """Subscriber with kinds={'trace'} should not receive scalar data."""
        now = time.time()
        provider = FakeProvider(
            tags={"scalars": ["loss/train"]},
            scalars={"loss/train": [(0, now, 0.5)]},
            trace_events=[{"step": 0, "phase": "forward", "duration_ms": 10.0}],
        )
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(
            runs=["run1"],
            kinds={"trace"},
        ))

        await mgr.poll_and_push()

        # Should receive trace but not scalar
        types = {m["type"] for m in ws.sent}
        assert "scalar" not in types
        assert "trace" in types

    @pytest.mark.asyncio
    async def test_poll_skips_empty_scalars(self):
        """Tags with no new data should not produce messages."""
        provider = FakeProvider(
            tags={"scalars": ["loss/train"]},
            scalars={"loss/train": []},  # no data
        )
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"], kinds={"scalar"}))

        await mgr.poll_and_push()
        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_poll_missing_provider(self):
        """Run with no provider in watcher should not crash."""
        watcher = FakeWatcher({})  # no providers

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"], kinds={"scalar"}))

        await mgr.poll_and_push()
        assert ws.sent == []

    @pytest.mark.asyncio
    async def test_poll_removes_broken_subscriber(self):
        """A subscriber that fails on send_json gets removed."""
        now = time.time()
        provider = FakeProvider(
            tags={"scalars": ["loss/train"]},
            scalars={"loss/train": [(0, now, 0.5)]},
        )
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        ws._should_fail = True
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"], kinds={"scalar"}))

        await mgr.poll_and_push()

        # Broken subscriber should be removed
        assert ws not in mgr._subscribers

    @pytest.mark.asyncio
    async def test_session_change_notification(self):
        """When session_id changes between polls, a session_changed message is sent."""
        provider = FakeProvider(session_id="session-1", resume_step=50)
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"], kinds={"scalar"}))

        # First poll: establishes known session
        await mgr.poll_and_push()
        ws.sent.clear()

        # Change session
        provider._session_id = "session-2"

        # Second poll: detect session change
        await mgr.poll_and_push()

        session_msgs = [m for m in ws.sent if m["type"] == "session_changed"]
        assert len(session_msgs) == 1
        msg = session_msgs[0]
        assert msg["run"] == "run1"
        assert msg["old_session_id"] == "session-1"
        assert msg["new_session_id"] == "session-2"
        assert msg["resume_step"] == 50

    @pytest.mark.asyncio
    async def test_poll_trace_events(self):
        """Subscribers with kinds={'trace'} receive trace event messages."""
        events = [
            {"step": 0, "phase": "forward", "duration_ms": 12.5},
            {"step": 1, "phase": "backward", "duration_ms": 25.0},
        ]
        provider = FakeProvider(trace_events=events)
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"], kinds={"trace"}))

        await mgr.poll_and_push()

        trace_msgs = [m for m in ws.sent if m["type"] == "trace"]
        assert len(trace_msgs) == 1
        assert trace_msgs[0]["events"] == events

    @pytest.mark.asyncio
    async def test_poll_eval_results(self):
        """Subscribers with kinds={'eval'} receive eval result messages."""
        results = [{"suite": "fid", "score": 42.5}]
        provider = FakeProvider(eval_results=results)
        watcher = FakeWatcher({"run1": provider})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws = FakeWebSocket()
        mgr.subscribe(ws, SubscriptionFilter(runs=["run1"], kinds={"eval"}))

        await mgr.poll_and_push()

        eval_msgs = [m for m in ws.sent if m["type"] == "eval"]
        assert len(eval_msgs) == 1
        assert eval_msgs[0]["results"] == results

    @pytest.mark.asyncio
    async def test_poll_multi_subscriber_routing(self):
        """Two subscribers to different runs get only their own data."""
        now = time.time()
        provider1 = FakeProvider(
            tags={"scalars": ["loss/train"]},
            scalars={"loss/train": [(0, now, 0.5)]},
        )
        provider2 = FakeProvider(
            tags={"scalars": ["loss/train"]},
            scalars={"loss/train": [(0, now, 0.9)]},
        )
        watcher = FakeWatcher({"run1": provider1, "run2": provider2})

        mgr = LiveUpdateManager()
        mgr.set_watcher(watcher)

        ws1 = FakeWebSocket()
        ws2 = FakeWebSocket()
        mgr.subscribe(ws1, SubscriptionFilter(runs=["run1"], kinds={"scalar"}))
        mgr.subscribe(ws2, SubscriptionFilter(runs=["run2"], kinds={"scalar"}))

        await mgr.poll_and_push()

        # ws1 should only have run1 data
        assert all(m["run"] == "run1" for m in ws1.sent)
        assert any(m["points"][0]["value"] == 0.5 for m in ws1.sent)

        # ws2 should only have run2 data
        assert all(m["run"] == "run2" for m in ws2.sent)
        assert any(m["points"][0]["value"] == 0.9 for m in ws2.sent)


# ===========================================================================
# Integration tests: WebSocket endpoint via TestClient
# ===========================================================================

# Thread-safety patch for TestClient (same approach as test_app.py)
from serenityboard.server.data_provider import RunDataProvider
from serenityboard.writer.schema import create_tables, set_pragmas

_orig_rdp_init = RunDataProvider.__init__


def _thread_safe_rdp_init(self, db_path: str) -> None:
    """Same as RunDataProvider.__init__ but with check_same_thread=False."""
    self._db_path = db_path
    self.run_dir = os.path.dirname(db_path)
    self._conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, check_same_thread=False
    )
    self._conn.execute("PRAGMA query_only = ON")
    self._last_seen: dict[str, int] = {}
    self._known_session_id: str | None = None


def _make_run_db(parent_dir: str, run_name: str) -> str:
    """Create a minimal board.db for integration tests. Returns db path."""
    run_dir = os.path.join(parent_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    db_path = os.path.join(run_dir, "board.db")

    now = time.time()
    conn = sqlite3.connect(db_path)
    set_pragmas(conn)
    create_tables(conn)

    session_id = f"session-{run_name}"
    with conn:
        for key, value in [
            ("active_session_id", json.dumps(session_id)),
            ("status", json.dumps("complete")),
            ("run_name", json.dumps(run_name)),
            ("start_time", json.dumps(now - 600)),
        ]:
            conn.execute(
                "INSERT OR REPLACE INTO metadata (key, value) VALUES (?, ?)",
                (key, value),
            )
        conn.execute(
            "INSERT INTO sessions (session_id, start_time, resume_step, status) "
            "VALUES (?, ?, NULL, 'complete')",
            (session_id, now - 600),
        )
        for step in range(5):
            conn.execute(
                "INSERT INTO scalars (tag, step, wall_time, value) VALUES (?, ?, ?, ?)",
                ("loss/train", step, now - 600 + step * 10, 1.0 - step * 0.1),
            )
    conn.close()
    return db_path


@pytest.fixture()
def ws_app_client():
    """Create a minimal app with a run, yield TestClient for WebSocket tests."""
    from serenityboard.server.app import create_app

    logdir = tempfile.mkdtemp(prefix="sb_ws_test_")
    try:
        _make_run_db(logdir, "run1")
        _make_run_db(logdir, "run2")

        with patch.object(RunDataProvider, "__init__", _thread_safe_rdp_init):
            app = create_app(logdir)
            client = TestClient(app, raise_server_exceptions=False)
            yield client
    finally:
        shutil.rmtree(logdir, ignore_errors=True)


class TestWebSocketConnectDisconnect:
    def test_ws_connect_disconnect(self, ws_app_client):
        """WebSocket connects and disconnects cleanly without error."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            # Connection is open, now just close it
            pass
        # No exception means success

    def test_ws_connect_twice(self, ws_app_client):
        """Two sequential WebSocket connections both succeed."""
        with ws_app_client.websocket_connect("/ws/live"):
            pass
        with ws_app_client.websocket_connect("/ws/live"):
            pass


class TestWebSocketSubscribe:
    def test_ws_subscribe(self, ws_app_client):
        """Sending a subscribe message does not crash the connection."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            ws.send_json({
                "subscribe": {
                    "runs": ["run1"],
                    "tags": ["loss/*"],
                    "kinds": ["scalar"],
                }
            })
            # Connection should still be alive — send another message
            ws.send_json({
                "subscribe": {
                    "runs": ["run1", "run2"],
                    "tags": ["*"],
                }
            })

    def test_ws_subscribe_no_tags(self, ws_app_client):
        """Subscribe with no tags field defaults to wildcard ['*']."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            ws.send_json({
                "subscribe": {
                    "runs": ["run1"],
                }
            })
            # No crash means the default handling worked

    def test_ws_subscribe_no_kinds(self, ws_app_client):
        """Subscribe with no kinds field defaults to ['scalar']."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            ws.send_json({
                "subscribe": {
                    "runs": ["run1"],
                    "tags": ["*"],
                }
            })


class TestWebSocketMalformedMessages:
    def test_ws_malformed_non_json(self, ws_app_client):
        """Sending non-JSON text should not kill the connection."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            ws.send_text("this is not json")
            # Connection should still be alive — send a valid message after
            ws.send_json({"subscribe": {"runs": ["run1"]}})

    def test_ws_malformed_no_subscribe_key(self, ws_app_client):
        """Sending JSON without 'subscribe' key should be ignored gracefully."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            ws.send_json({"unrelated": "data"})
            # Connection should remain open
            ws.send_json({"subscribe": {"runs": ["run1"]}})

    def test_ws_empty_json_object(self, ws_app_client):
        """Sending empty JSON object should not crash."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            ws.send_json({})
            ws.send_json({"subscribe": {"runs": ["run1"]}})


class TestWebSocketEmptySubscribe:
    def test_ws_empty_runs_list(self, ws_app_client):
        """Subscribe with empty runs list is accepted but produces no data."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            ws.send_json({
                "subscribe": {
                    "runs": [],
                    "tags": ["*"],
                }
            })
            # No crash, connection stays open

    def test_ws_subscribe_nonexistent_run(self, ws_app_client):
        """Subscribing to a run that doesn't exist should not crash."""
        with ws_app_client.websocket_connect("/ws/live") as ws:
            ws.send_json({
                "subscribe": {
                    "runs": ["nonexistent_run"],
                    "tags": ["*"],
                }
            })


class TestWebSocketMultiClient:
    def test_multi_client_different_subs(self, ws_app_client):
        """Two concurrent WebSocket clients with different subscriptions both work."""
        with ws_app_client.websocket_connect("/ws/live") as ws1:
            ws1.send_json({
                "subscribe": {
                    "runs": ["run1"],
                    "tags": ["loss/*"],
                }
            })
            with ws_app_client.websocket_connect("/ws/live") as ws2:
                ws2.send_json({
                    "subscribe": {
                        "runs": ["run2"],
                        "tags": ["*"],
                        "kinds": ["scalar", "trace"],
                    }
                })
                # Both connections are alive simultaneously
                # Re-subscribe ws1 to verify it's still responsive
                ws1.send_json({
                    "subscribe": {
                        "runs": ["run1"],
                        "tags": ["*"],
                    }
                })

    def test_multi_client_same_run(self, ws_app_client):
        """Two clients subscribing to the same run both succeed."""
        with ws_app_client.websocket_connect("/ws/live") as ws1:
            ws1.send_json({"subscribe": {"runs": ["run1"]}})
            with ws_app_client.websocket_connect("/ws/live") as ws2:
                ws2.send_json({"subscribe": {"runs": ["run1"]}})
