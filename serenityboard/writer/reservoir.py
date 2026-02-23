"""Bounded reservoir sampling per tag."""
from __future__ import annotations

import random
import threading

__all__ = ["Reservoir"]


class Reservoir:
    """Map of tag -> ReservoirBucket. Bounds total items per tag. Thread-safe."""

    def __init__(
        self, max_size: int, seed: int = 0, always_keep_last: bool = True
    ) -> None:
        if max_size < 0 or max_size != int(max_size):
            raise ValueError(f"max_size must be non-negative int, got {max_size}")
        self._max_size = max_size
        self._seed = seed
        self._always_keep_last = always_keep_last
        self._buckets: dict[str, _ReservoirBucket] = {}
        self._lock = threading.Lock()

    def add(self, key: str, item: object, transform: object = None) -> None:
        """Add *item* under *key*, optionally applying *transform*."""
        with self._lock:
            if key not in self._buckets:
                self._buckets[key] = _ReservoirBucket(
                    self._max_size,
                    random.Random(self._seed),
                    self._always_keep_last,
                )
            bucket = self._buckets[key]
        # Bucket has its own lock -- fine-grained
        bucket.add(item, transform)

    def get_items(self, key: str) -> list:
        """Return a snapshot of items for *key*, or empty list."""
        with self._lock:
            bucket = self._buckets.get(key)
        if bucket is None:
            return []
        return bucket.get_items()

    def drain_items(self, key: str) -> list:
        """Return and clear buffered items for *key*."""
        with self._lock:
            bucket = self._buckets.get(key)
        if bucket is None:
            return []
        return bucket.drain_items()

    @property
    def keys(self) -> list[str]:
        with self._lock:
            return list(self._buckets)


class _ReservoirBucket:
    """Bounded-size sample from a stream.

    max_size == 0 means unbounded (keep everything).
    max_size > 0 uses reservoir sampling with optional always_keep_last.
    """

    def __init__(
        self, max_size: int, rng: random.Random, always_keep_last: bool
    ) -> None:
        self.items: list = []
        self._max_size = max_size
        self._num_seen = 0
        self._rng = rng
        self._always_keep_last = always_keep_last
        self._lock = threading.Lock()

    def add(self, item: object, transform: object = None) -> None:
        """Add one item, applying reservoir eviction when at capacity."""
        f = transform or (lambda x: x)
        with self._lock:
            if self._max_size == 0 or len(self.items) < self._max_size:
                self.items.append(f(item))
            else:
                r = self._rng.randint(0, self._num_seen)
                if r < self._max_size:
                    self.items.pop(r)
                    self.items.append(f(item))
                elif self._always_keep_last:
                    self.items[-1] = f(item)
            self._num_seen += 1

    def get_items(self) -> list:
        """Return a snapshot copy of stored items."""
        with self._lock:
            return list(self.items)

    def drain_items(self) -> list:
        """Return buffered items and clear the bucket for the next interval."""
        with self._lock:
            drained = list(self.items)
            self.items = []
            # Reset stream counter after draining; this implementation samples
            # within each flush interval.
            self._num_seen = 0
            return drained

    def filter(self, predicate: object) -> int:
        """Remove items not matching *predicate*. Returns count removed."""
        with self._lock:
            before = len(self.items)
            self.items = [x for x in self.items if predicate(x)]
            removed = before - len(self.items)
            if before > 0:
                self._num_seen = int(
                    round(self._num_seen * len(self.items) / before)
                )
            return removed
