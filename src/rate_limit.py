"""Process-local rolling-window reservation limits for benchmark requests."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from threading import Condition


class RollingTokenLimiter:
    """Reserve request tokens against a rolling 60-second process-wide budget.

    Reservations remain charged for the full window even when a request finishes
    earlier. This mirrors provider TPM accounting based on request admission rather
    than the number of requests currently in flight.
    """

    WINDOW_SECONDS = 60.0

    def __init__(
        self,
        tokens_per_minute: int,
        *,
        _clock: Callable[[], float] = time.monotonic,
        _condition: Condition | None = None,
    ) -> None:
        if (
            not isinstance(tokens_per_minute, int)
            or isinstance(tokens_per_minute, bool)
            or tokens_per_minute <= 0
        ):
            raise ValueError("tokens_per_minute must be a positive integer")
        self.tokens_per_minute = tokens_per_minute
        self._clock = _clock
        self._condition = _condition or Condition()
        self._reservations: deque[tuple[float, int]] = deque()
        self._reserved_tokens = 0

    def _expire(self, now: float) -> None:
        cutoff = now - self.WINDOW_SECONDS
        expired = False
        while self._reservations and self._reservations[0][0] <= cutoff:
            _, tokens = self._reservations.popleft()
            self._reserved_tokens -= tokens
            expired = True
        if expired:
            self._condition.notify_all()

    def reserve(self, tokens: int) -> None:
        """Block until ``tokens`` fit, then charge them for the full window."""
        if not isinstance(tokens, int) or isinstance(tokens, bool) or tokens <= 0:
            raise ValueError("request token reservation must be a positive integer")
        if tokens > self.tokens_per_minute:
            raise ValueError(
                f"request reserves {tokens} tokens, exceeding the "
                f"{self.tokens_per_minute} --tokens-per-minute budget"
            )

        with self._condition:
            while True:
                now = self._clock()
                self._expire(now)
                if self._reserved_tokens + tokens <= self.tokens_per_minute:
                    self._reservations.append((now, tokens))
                    self._reserved_tokens += tokens
                    return

                next_expiry = self._reservations[0][0] + self.WINDOW_SECONDS
                self._condition.wait(timeout=max(next_expiry - now, 0.0))


class RollingRequestLimiter(RollingTokenLimiter):
    """Reserve one request against a rolling process-wide RPM budget."""

    def __init__(
        self,
        requests_per_minute: int,
        *,
        _clock: Callable[[], float] = time.monotonic,
        _condition: Condition | None = None,
    ) -> None:
        super().__init__(
            requests_per_minute,
            _clock=_clock,
            _condition=_condition,
        )
        self.requests_per_minute = requests_per_minute

    def reserve(self) -> None:
        """Block until one request slot is available, then charge it."""
        super().reserve(1)
