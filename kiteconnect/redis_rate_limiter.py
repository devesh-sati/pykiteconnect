# -*- coding: utf-8 -*-
"""
    redis_rate_limiter.py

    Cross-process circuit breaker + distributed token-bucket rate limiter
    for the Kite Connect API, backed by Redis.

    Problem
    -------
    When multiple Python processes share the same Kite API key, a per-process
    in-memory limiter is blind to calls made by sibling processes.  Any one
    process can exhaust the shared quota and cause 429s for all others.
    Redis solves this by providing a single shared state store that every
    process reads and writes atomically.

    Circuit breaker states
    ----------------------

        CLOSED  ──► normal operation; token bucket enforces rate
           │
           │  429 received  OR  failure_threshold consecutive errors
           ▼
        OPEN    ──► ALL processes block for `open_ttl` seconds
           │        (calling thread sleeps on the Redis TTL)
           │
           │  cooldown expires (Redis key auto-deleted)
           ▼
      HALF_OPEN ──► exactly ONE probe request is allowed through (Redis SETNX)
           │
           ├── probe succeeds ──► CLOSED  (failures reset)
           └── probe fails    ──► OPEN    (cooldown restarts)

    Distributed token bucket
    ------------------------
    Implemented as an atomic Lua script executed on the Redis server.
    This guarantees correctness even when dozens of processes call
    `acquire()` simultaneously – there are no TOCTOU races.

    Usage
    -----
        import redis
        from kiteconnect import KiteConnect
        from kiteconnect.redis_rate_limiter import RedisRateLimiter

        r = redis.Redis(host="localhost", port=6379, db=0)
        limiter = RedisRateLimiter(r, api_key="your_api_key")

        kite = KiteConnect(api_key="your_api_key", rate_limiter=limiter)

    Redis key layout
    ----------------
        kite:cb:{api_key}:state     – current circuit state (string)
        kite:cb:{api_key}:open      – exists while OPEN; TTL = cooldown seconds
        kite:cb:{api_key}:probe     – SETNX lock for the single HALF_OPEN probe
        kite:cb:{api_key}:failures  – consecutive failure counter
        kite:tb:{api_key}:tokens    – current token bucket level (float)
        kite:tb:{api_key}:ts        – last refill timestamp (Unix float)

    Dependencies
    ------------
        pip install redis>=4.0.0
"""
import time
import logging
import threading

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Circuit breaker state constants
# ---------------------------------------------------------------------------
CB_CLOSED    = "closed"
CB_OPEN      = "open"
CB_HALF_OPEN = "half_open"

# ---------------------------------------------------------------------------
# Lua script: atomic distributed token bucket
#
# Refills tokens proportional to elapsed time, then attempts to consume one.
# Returns a two-element array: { acquired (0|1), remaining_tokens (float) }
# All reads and writes happen inside a single EVAL call – no race conditions.
# ---------------------------------------------------------------------------
_ACQUIRE_TOKEN_SCRIPT = """
local tokens_key = KEYS[1]
local ts_key     = KEYS[2]
local rate       = tonumber(ARGV[1])
local capacity   = tonumber(ARGV[2])
local now        = tonumber(ARGV[3])
local ttl        = tonumber(ARGV[4])

local last_tokens = tonumber(redis.call('GET', tokens_key))
if last_tokens == nil then
    last_tokens = capacity
end

local last_ts = tonumber(redis.call('GET', ts_key))
if last_ts == nil then
    last_ts = now
end

local delta  = math.max(0, now - last_ts)
local filled = math.min(capacity, last_tokens + delta * rate)

if filled >= 1 then
    redis.call('SETEX', tokens_key, ttl, filled - 1)
    redis.call('SETEX', ts_key,     ttl, now)
    return {1, filled - 1}
else
    redis.call('SETEX', tokens_key, ttl, filled)
    redis.call('SETEX', ts_key,     ttl, now)
    return {0, filled}
end
"""


class RedisRateLimiter:
    """
    Cross-process circuit breaker + distributed token-bucket rate limiter.

    Thread-safe: a single instance can be shared across threads within one
    process; Redis handles coordination across processes.
    """

    def __init__(
        self,
        redis_client,
        api_key,
        rate=3,
        capacity=5,
        open_ttl=60,
        failure_threshold=5,
        half_open_probe_ttl=10,
    ):
        """
        Parameters
        ----------
        redis_client        : ``redis.Redis`` instance (already connected)
        api_key             : Kite API key – used to namespace all Redis keys
                              so multiple API keys can coexist on one Redis DB
        rate                : sustained token refill rate in tokens/second (default 3)
        capacity            : token bucket burst ceiling (default 5)
        open_ttl            : seconds the circuit stays OPEN after a 429 or
                              threshold breach (default 60)
        failure_threshold   : consecutive non-429 failures (5xx / timeout) that
                              trip the circuit (default 5)
        half_open_probe_ttl : seconds the probe lock is held in HALF_OPEN state
                              before another process may claim it (default 10)
        """
        self._redis              = redis_client
        self._api_key            = api_key
        self._rate               = float(rate)
        self._capacity           = float(capacity)
        self._open_ttl           = int(open_ttl)
        self._failure_threshold  = int(failure_threshold)
        self._half_open_probe_ttl = int(half_open_probe_ttl)

        # ── Redis key names ────────────────────────────────────────────────
        _ns = "kite:cb:{}".format(api_key)
        self._state_key    = _ns + ":state"     # CB_CLOSED / CB_OPEN / CB_HALF_OPEN
        self._open_key     = _ns + ":open"      # present while OPEN; auto-expires
        self._probe_key    = _ns + ":probe"     # SETNX probe lock in HALF_OPEN
        self._failures_key = _ns + ":failures"  # consecutive failure counter

        _tb = "kite:tb:{}".format(api_key)
        self._tokens_key   = _tb + ":tokens"
        self._ts_key       = _tb + ":ts"

        # Register Lua script (returns a callable Script object)
        self._acquire_script = self._redis.register_script(_ACQUIRE_TOKEN_SCRIPT)

        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public interface consumed by connect.py
    # ------------------------------------------------------------------

    def acquire(self):
        """
        Block until the circuit permits a request and a token is available.

        - OPEN      → sleeps for remaining TTL of the open key, then
                      transitions this process to HALF_OPEN
        - HALF_OPEN → exactly one process claims the probe via SETNX;
                      all others sleep until the probe resolves
        - CLOSED    → consumes one token from the distributed bucket;
                      spins (with micro-sleep) if the bucket is empty
        """
        while True:
            state = self._get_state()

            # ── OPEN: block for the remaining cooldown ─────────────────
            if state == CB_OPEN:
                ttl = self._redis.ttl(self._open_key)
                if isinstance(ttl, (int, float)) and ttl > 0:
                    log.warning(
                        "Circuit OPEN (api_key=%s). All processes blocked "
                        "for %.0fs.",
                        self._api_key, ttl,
                    )
                    time.sleep(ttl)
                # Key has expired → move to HALF_OPEN
                self._transition_to_half_open()
                continue

            # ── HALF_OPEN: only one probe request gets through ─────────
            if state == CB_HALF_OPEN:
                claimed = self._redis.set(
                    self._probe_key, "1",
                    ex=self._half_open_probe_ttl,
                    nx=True,        # SET if Not eXists
                )
                if claimed:
                    log.info(
                        "Circuit HALF_OPEN (api_key=%s): probe slot claimed.",
                        self._api_key,
                    )
                    return  # this process is the probe; bypass token bucket

                # Another process holds the probe – wait for it to resolve
                wait = max(0.2, self._redis.ttl(self._probe_key) or 0.2)
                log.debug(
                    "HALF_OPEN probe held by peer. Waiting %.1fs.", wait
                )
                time.sleep(wait)
                continue

            # ── CLOSED: consume from distributed token bucket ──────────
            if self._try_consume_token():
                return
            # Bucket empty: back off a fraction of the inter-token interval
            time.sleep(1.0 / self._rate * 0.1)

    def record_429(self):
        """
        Open the circuit immediately.

        Call this whenever any process receives an HTTP 429 response.
        Sets the open key with TTL so *all* processes will block on their
        next :meth:`acquire` call.
        """
        log.error(
            "429 rate-limit received (api_key=%s). Opening circuit for %ds.",
            self._api_key, self._open_ttl,
        )
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(self._open_key,     "1", ex=self._open_ttl)
        pipe.set(self._state_key,    CB_OPEN)
        pipe.delete(self._probe_key)
        pipe.set(self._failures_key, 0)
        pipe.execute()

    def record_failure(self):
        """
        Increment the consecutive-failure counter.

        - In HALF_OPEN state: probe failed → re-open the circuit immediately.
        - In CLOSED state: open the circuit once ``failure_threshold`` is reached.

        Use this for transient errors (5xx, connection timeout) – **not** for
        429 (use :meth:`record_429`) or 403 (auth errors should not trip the
        circuit).
        """
        state = self._get_state()

        if state == CB_HALF_OPEN:
            log.warning(
                "HALF_OPEN probe failed (api_key=%s). Re-opening circuit.",
                self._api_key,
            )
            self.record_429()
            return

        count = self._redis.incr(self._failures_key)
        self._redis.expire(self._failures_key, self._open_ttl * 2)

        if int(count) >= self._failure_threshold:
            log.warning(
                "Failure threshold (%d) breached (api_key=%s). Opening circuit.",
                self._failure_threshold, self._api_key,
            )
            self.record_429()

    def record_success(self):
        """
        Register a successful API response.

        - In HALF_OPEN state: probe succeeded → close the circuit.
        - In CLOSED state: reset the failure counter.
        """
        state = self._get_state()
        if state == CB_HALF_OPEN:
            log.info(
                "HALF_OPEN probe succeeded (api_key=%s). Closing circuit.",
                self._api_key,
            )
            self._transition_to_closed()
        elif state == CB_CLOSED:
            self._redis.set(self._failures_key, 0)

    def get_state(self):
        """Return the current circuit state string (for monitoring / health checks)."""
        return self._get_state()

    def reset(self):
        """Force-close the circuit and clear all state (useful in tests)."""
        pipe = self._redis.pipeline(transaction=True)
        pipe.delete(self._state_key)
        pipe.delete(self._open_key)
        pipe.delete(self._probe_key)
        pipe.delete(self._failures_key)
        pipe.delete(self._tokens_key)
        pipe.delete(self._ts_key)
        pipe.execute()
        log.info("Circuit reset (api_key=%s).", self._api_key)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self):
        raw = self._redis.get(self._state_key)
        if raw is None:
            return CB_CLOSED
        return raw.decode() if isinstance(raw, bytes) else raw

    def _transition_to_half_open(self):
        # SETNX: only the first process to reach this wins; others find it
        # already set to CB_HALF_OPEN, which is fine.
        self._redis.setnx(self._state_key, CB_HALF_OPEN)
        log.info(
            "Circuit → HALF_OPEN (api_key=%s).", self._api_key
        )

    def _transition_to_closed(self):
        pipe = self._redis.pipeline(transaction=True)
        pipe.set(self._state_key,    CB_CLOSED)
        pipe.delete(self._open_key)
        pipe.delete(self._probe_key)
        pipe.set(self._failures_key, 0)
        pipe.execute()
        log.info("Circuit → CLOSED (api_key=%s).", self._api_key)

    def _try_consume_token(self):
        """
        Attempt to consume one token from the distributed bucket.

        Returns True if a token was acquired, False if the bucket is empty.
        The Lua script executes atomically on the Redis server.
        """
        now = time.time()
        # TTL for the bucket keys: 2× fill-time so they survive brief gaps
        ttl = int(self._capacity / self._rate * 2) + 10
        result = self._acquire_script(
            keys=[self._tokens_key, self._ts_key],
            args=[self._rate, self._capacity, now, ttl],
        )
        return bool(result[0])
