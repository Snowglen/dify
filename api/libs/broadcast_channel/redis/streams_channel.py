from __future__ import annotations

import logging
import threading
import time
from collections import deque
from collections.abc import Iterator
from typing import Self

from libs.broadcast_channel.channel import Producer, Subscriber, Subscription
from libs.broadcast_channel.exc import SubscriptionClosedError
from redis import Redis, RedisCluster

logger = logging.getLogger(__name__)


class StreamsBroadcastChannel:
    """
    Redis Streams based broadcast channel implementation.

    Characteristics:
    - At-least-once delivery for late subscribers within the stream retention window.
    - Each topic is stored as a dedicated Redis Stream key.
    - The stream key expires `retention_seconds` after the last event is published (to bound storage).
    """

    def __init__(self, redis_client: Redis | RedisCluster, *, retention_seconds: int = 600):
        self._client = redis_client
        self._retention_seconds = max(int(retention_seconds or 0), 0)

    def topic(self, topic: str) -> StreamsTopic:
        return StreamsTopic(self._client, topic, retention_seconds=self._retention_seconds)


class StreamsTopic:
    def __init__(self, redis_client: Redis | RedisCluster, topic: str, *, retention_seconds: int = 600):
        self._client = redis_client
        self._topic = topic
        self._key = f"stream:{topic}"
        self._retention_seconds = retention_seconds

    def as_producer(self) -> Producer:
        return self

    def publish(self, payload: bytes) -> None:
        # Append to the stream; trim is optional and not enabled by default.
        # Use a single field b'data' to store the raw event payload (bytes) for consistent binary handling.
        self._client.xadd(self._key, {b"data": payload})
        if self._retention_seconds > 0:
            # Refresh key expiry so the stream is cleaned up some time after the last event.
            try:
                self._client.expire(self._key, self._retention_seconds)
            except Exception as e:
                # Non-fatal; best-effort cleanup only. Log at debug level to avoid noise.
                logger.warning("Failed to set expire for stream key %s: %s", self._key, e, exc_info=True)

    def as_subscriber(self) -> Subscriber:
        return self

    def subscribe(self) -> Subscription:
        return _StreamsSubscription(self._client, self._key)


class _StreamsSubscription(Subscription):
    def __init__(self, client: Redis | RedisCluster, key: str):
        self._client = client
        self._key = key
        self._closed = threading.Event()
        self._last_id = "0-0"  # start from the beginning of the stream (XREAD returns entries with IDs > 0-0)
        self._queue: deque[bytes] = deque()
        self._lock = threading.Lock()
        self._listener: threading.Thread | None = None

    def _listen(self) -> None:
        # Use XREAD in a loop; block up to 1 second to allow timely shutdown.
        while not self._closed.is_set():
            try:
                streams = self._client.xread({self._key: self._last_id}, block=1000, count=100)
            except Exception:
                # Back off a bit on errors to avoid hot loops.
                time.sleep(0.2)
                continue

            if not streams:
                continue

            # streams is a list of (key, [(id, {field: value}), ...])
            for _key, entries in streams:
                for entry_id, fields in entries:
                    data = None
                    if isinstance(fields, dict):
                        # Support both new 'data' and legacy 'd' keys, in bytes or str forms.
                        data = fields.get(b"data")
                    if isinstance(data, (bytes, bytearray)):
                        with self._lock:
                            self._queue.append(bytes(data))
                    self._last_id = entry_id

    def _start_if_needed(self) -> None:
        if self._listener is not None:
            return
        self._listener = threading.Thread(target=self._listen, name=f"redis-streams-sub-{self._key}", daemon=True)
        self._listener.start()

    def __iter__(self) -> Iterator[bytes]:
        # NB: stream-based subscriptions are pull-based via receive(); iterator is optional
        # and implemented with a simple polling loop.
        self._start_if_needed()
        while not self._closed.is_set():
            item = self.receive(timeout=1)
            if item is not None:
                yield item

    def receive(self, timeout: float | None = 0.1) -> bytes | None:
        if self._closed.is_set():
            raise SubscriptionClosedError("The Redis streams subscription is closed")
        self._start_if_needed()

        if timeout is None:
            while not self._closed.is_set():
                with self._lock:
                    if self._queue:
                        return self._queue.popleft()
                time.sleep(0.05)
            raise SubscriptionClosedError("The Redis streams subscription is closed")

        deadline: float = time.monotonic() + timeout
        while not self._closed.is_set():
            with self._lock:
                if self._queue:
                    return self._queue.popleft()
            now = time.monotonic()
            if now >= deadline:
                return None
            time.sleep(min(0.05, max(0.0, deadline - now)))

        raise SubscriptionClosedError("The Redis streams subscription is closed")

    def close(self) -> None:
        if self._closed.is_set():
            return
        self._closed.set()
        if self._listener is not None:
            self._listener.join(timeout=1.0)
            self._listener = None

    # Context manager helpers
    def __enter__(self) -> Self:
        self._start_if_needed()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        self.close()
        return None
