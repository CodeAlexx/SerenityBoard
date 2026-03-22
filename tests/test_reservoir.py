"""Tests for Reservoir bounded sampling."""
from __future__ import annotations

import threading

import pytest

from serenityboard.writer.reservoir import Reservoir


# ---------------------------------------------------------------------------
# Basic operations
# ---------------------------------------------------------------------------

class TestBasicOperations:

    def test_basic_add_and_get(self) -> None:
        """Add items under a key and retrieve them."""
        r = Reservoir(max_size=10, seed=0)
        r.add("tag", "a")
        r.add("tag", "b")
        r.add("tag", "c")
        items = r.get_items("tag")
        assert items == ["a", "b", "c"]

    def test_empty_reservoir(self) -> None:
        """get_items on a nonexistent key returns an empty list."""
        r = Reservoir(max_size=10, seed=0)
        assert r.get_items("missing") == []

    def test_drain_items(self) -> None:
        """drain returns items and clears the bucket."""
        r = Reservoir(max_size=10, seed=0)
        r.add("k", 1)
        r.add("k", 2)
        r.add("k", 3)
        drained = r.drain_items("k")
        assert drained == [1, 2, 3]
        # After drain the bucket is empty
        assert r.get_items("k") == []

    def test_drain_nonexistent_key(self) -> None:
        """drain_items on a missing key returns empty list."""
        r = Reservoir(max_size=5, seed=0)
        assert r.drain_items("nope") == []

    def test_keys_property(self) -> None:
        """keys returns the list of all added keys."""
        r = Reservoir(max_size=10, seed=0)
        r.add("alpha", 1)
        r.add("beta", 2)
        r.add("alpha", 3)
        keys = sorted(r.keys)
        assert keys == ["alpha", "beta"]


# ---------------------------------------------------------------------------
# Capacity behaviour
# ---------------------------------------------------------------------------

class TestCapacity:

    def test_capacity_limit(self) -> None:
        """Adding N items with max_size=K < N keeps at most K items."""
        max_size = 5
        r = Reservoir(max_size=max_size, seed=42)
        for i in range(100):
            r.add("tag", i)
        items = r.get_items("tag")
        assert len(items) <= max_size

    def test_unbounded_mode(self) -> None:
        """max_size=0 keeps every item."""
        r = Reservoir(max_size=0, seed=0)
        n = 200
        for i in range(n):
            r.add("all", i)
        assert len(r.get_items("all")) == n

    def test_capacity_one(self) -> None:
        """max_size=1 always keeps exactly 1 item."""
        r = Reservoir(max_size=1, seed=7)
        for i in range(10):
            r.add("single", i)
        items = r.get_items("single")
        assert len(items) == 1

    def test_capacity_larger_than_items(self) -> None:
        """When fewer items than max_size, all are kept."""
        r = Reservoir(max_size=100, seed=0)
        for i in range(5):
            r.add("small", i)
        items = r.get_items("small")
        assert items == [0, 1, 2, 3, 4]


# ---------------------------------------------------------------------------
# Sampling properties
# ---------------------------------------------------------------------------

class TestSamplingProperties:

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces identical reservoir contents."""
        def run_once(seed: int) -> list:
            r = Reservoir(max_size=10, seed=seed)
            for i in range(200):
                r.add("det", i)
            return r.get_items("det")

        a = run_once(123)
        b = run_once(123)
        assert a == b

    def test_different_seeds_differ(self) -> None:
        """Different seeds should (almost certainly) produce different results."""
        def run_once(seed: int) -> list:
            r = Reservoir(max_size=10, seed=seed)
            for i in range(200):
                r.add("det", i)
            return r.get_items("det")

        a = run_once(0)
        b = run_once(999)
        # Extremely unlikely to be identical with different seeds
        assert a != b

    def test_uniform_distribution(self) -> None:
        """Reservoir sampling should not systematically favour any position.

        We run many independent reservoirs (each with a fresh seed) and count
        how often each source item lands in the reservoir.  For uniform
        sampling every item should appear roughly the same number of times.
        """
        max_size = 10
        n_items = 100
        n_trials = 2000
        counts = [0] * n_items

        for trial in range(n_trials):
            r = Reservoir(max_size=max_size, seed=trial, always_keep_last=False)
            for i in range(n_items):
                r.add("u", i)
            for item in r.get_items("u"):
                counts[item] += 1

        # Expected count per item: n_trials * max_size / n_items = 200
        expected = n_trials * max_size / n_items
        # Allow generous tolerance (factor of 2.5) -- we just want to detect
        # gross non-uniformity, not fine statistical deviation.
        low = expected * 0.4
        high = expected * 1.6
        for i, c in enumerate(counts):
            assert low <= c <= high, (
                f"Item {i} appeared {c} times (expected ~{expected:.0f}, "
                f"range [{low:.0f}, {high:.0f}])"
            )

    def test_always_keep_last(self) -> None:
        """With always_keep_last=True the final item is always in the reservoir."""
        max_size = 5
        last_item = "LAST"
        # Run many times to be sure (probabilistic guarantee)
        for seed in range(100):
            r = Reservoir(max_size=max_size, seed=seed, always_keep_last=True)
            for i in range(200):
                r.add("akl", i)
            r.add("akl", last_item)
            items = r.get_items("akl")
            assert last_item in items, (
                f"seed={seed}: last item not found in reservoir: {items}"
            )

    def test_always_keep_last_false(self) -> None:
        """With always_keep_last=False the last item is NOT guaranteed."""
        max_size = 2
        n_items = 10000
        # Over many seeds the last item should sometimes be missing
        missing_count = 0
        for seed in range(100):
            r = Reservoir(max_size=max_size, seed=seed, always_keep_last=False)
            for i in range(n_items):
                r.add("no_akl", i)
            items = r.get_items("no_akl")
            if (n_items - 1) not in items:
                missing_count += 1

        # With max_size=2 and 10k items, probability of keeping last ~= 2/10000
        # So across 100 seeds, it should almost always be missing.
        assert missing_count > 50, (
            f"Expected last item to be missing frequently, but only missing {missing_count}/100 times"
        )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

class TestTransform:

    def test_transform(self) -> None:
        """add() with a transform function applies it to stored items."""
        r = Reservoir(max_size=10, seed=0)
        r.add("t", 5, transform=lambda x: x * 2)
        r.add("t", 3, transform=lambda x: x + 100)
        items = r.get_items("t")
        assert items == [10, 103]

    def test_transform_none(self) -> None:
        """transform=None stores the raw item."""
        r = Reservoir(max_size=10, seed=0)
        r.add("raw", "hello", transform=None)
        assert r.get_items("raw") == ["hello"]


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------

class TestThreadSafety:

    def test_thread_safety(self) -> None:
        """Concurrent adds from multiple threads must not corrupt or crash."""
        max_size = 50
        r = Reservoir(max_size=max_size, seed=0)
        n_threads = 8
        items_per_thread = 500
        barrier = threading.Barrier(n_threads)

        def worker(thread_id: int) -> None:
            barrier.wait()
            for i in range(items_per_thread):
                r.add("shared", thread_id * items_per_thread + i)

        threads = [
            threading.Thread(target=worker, args=(tid,))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        items = r.get_items("shared")
        assert len(items) <= max_size
        # All items should be valid integers from our range
        total = n_threads * items_per_thread
        for item in items:
            assert 0 <= item < total

    def test_thread_safety_multiple_keys(self) -> None:
        """Concurrent adds to different keys must not interfere."""
        r = Reservoir(max_size=20, seed=0)
        n_threads = 4
        items_per_thread = 100

        def worker(key: str) -> None:
            for i in range(items_per_thread):
                r.add(key, i)

        threads = [
            threading.Thread(target=worker, args=(f"key_{tid}",))
            for tid in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert sorted(r.keys) == [f"key_{i}" for i in range(n_threads)]
        for tid in range(n_threads):
            items = r.get_items(f"key_{tid}")
            assert len(items) <= 20


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_invalid_max_size_negative(self) -> None:
        """Negative max_size raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Reservoir(max_size=-1)

    def test_invalid_max_size_float(self) -> None:
        """Non-integer max_size raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            Reservoir(max_size=3.5)  # type: ignore[arg-type]
