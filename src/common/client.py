import logging
import time
from datetime import datetime, timedelta
from threading import Lock

import httpx
import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class RetryError(Exception):
    pass


def _is_retryable_error(exc: BaseException) -> bool:
    """Check if an exception should trigger a retry."""
    if isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.TimeoutException,
            requests.exceptions.ConnectionError,
            RetryError,
            requests.exceptions.ReadTimeout,
        ),
    ):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503, 504)
    return False


def retry_request():
    """Decorator for HTTP requests with exponential backoff.

    Retries on:
    - Connection errors
    - Timeouts
    - HTTP 429 (rate limit)
    - HTTP 5xx (server errors)

    Uses exponential backoff starting at 1s, max 60s, up to 5 attempts.
    """
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=60),
        retry=retry_if_exception(_is_retryable_error),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


class RateLimiter:
    def __init__(self, requests: int, time_period: timedelta = timedelta(seconds=1)) -> None:
        self._limit = requests
        self._time_period = time_period
        self._requests: list[datetime] = []
        self._lock = Lock()

    def __enter__(self) -> None:
        with self._lock:
            now = datetime.now()
            while self._requests and self._requests[0] + self._time_period <= now:
                self._requests.pop(0)

            assert len(self._requests) <= self._limit
            assert sorted(self._requests) == self._requests
            if len(self._requests) == self._limit:
                free_at = self._requests[0] + self._time_period
                sleep_time = (free_at - now).total_seconds()
                time.sleep(sleep_time)

            self._requests.append(datetime.now())

    def __exit__(self, *_: object) -> None:
        pass
