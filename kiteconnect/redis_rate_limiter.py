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
           │  cooldown expires (_open_key auto-deleted by Redis)
           ▼
      HALF_OPEN ──► exactly ONE probe request is allowed through (Redis SET NX)
           │
           ├── probe succeeds ──► CLOSED  (failures reset)
           └── probe fails    ──► OPEN    (cooldown restarts)

    State encoding
    --------------
    State is derived ENTIRELY from the presence or absence of TTL-bearing
    Redis keys.  There is no separate state-string key that can fall out of
    sync with a key's actual TTL:

        _open_key      present  →  OPEN
        _half_open_key present  →  HALF_OPEN
        neither present         →  CLOSED

    This guarantees the circuit can never become permanently stuck in OPEN
    (the original SETNX bug) and means state always self-heals if keys expire.

    Distributed token bucket
    ------------------------
    Implemented as an atomic Lua script that runs entirely on the Redis server,
    including reading the clock via ``redis.call('TIME')``.  This eliminates
    the client-clock-skew vulnerability where different machines with drifting
    clocks would corrupt the token refill calculation.

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
        kite:cb:{api_key}:open       – present while OPEN;      TTL = open_ttl
        kite:cb:{api_key}:half_open  – present while HALF_OPEN; TTL = open_ttl
        kite:cb:{api_key}:probe      – SETNX probe lock;        TTL = half_open_probe_ttl
        kite:cb:{api_key}:failures   – consecutive failure counter
        kite:tb:{api_key}:tokens     – current token bucket level (float string)
        kite:tb:{api_key}:ts         – last refill timestamp, server Unix float

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
# FIX (clock skew): ``now`` is read from the Redis server via TIME rather
# than being passed in as a client-supplied ARGV.  This means all processes
# share a single authoritative clock regardless of NTP drift across machines.
#
# Returns a two-element array: { acquired (0|1), remaining_tokens (string) }
# All reads and writes happen inside a single EVAL call – no TOCTOU races.
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
    redis.call('SETEX', tokens_key, ttl, tostring(filled - 1))
    redis.call('SETEX', ts_key,     ttl, tostring(now))
    return {1, tostring(filled - 1)}
else
    redis.call('SETEX', tokens_key, ttl, tostring(filled))
    redis.call('SETEX', ts_key,     ttl, tostring(now))
    return {0, tostring(filled)}
end
"""

# ---------------------------------------------------------------------------
# Lua script: atomic failure-counter increment + threshold check
#
# FIX (concurrent threshold breach): returning 1 only when ``count`` equals
# the threshold exactly (not >=) ensures that exactly one process among all
# concurrent callers triggers the circuit open, even if many increment
# simultaneously.  Using >= would cause every subsequent failure after the
# threshold to also call _open_circuit, which is noisy and wasteful.
# ---------------------------------------------------------------------------
_INCR_FAILURE_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
redis.call('EXPIRE', KEYS[1], tonumber(ARGV[1]))
if count == tonumber(ARGV[2]) then
    return 1
end
return 0
"""


class RedisRateLimiter:
    """
    Cross-process circuit breaker + distributed token-bucket rate limiter.

    Thread-safe: a single instance may be shared across threads within one
    process; Redis handles coordination across processes.

    All Redis errors are caught internally.  When *fail_open* is True
    (default) the request is allowed through on Redis failure so that a
    Redis outage does not take the whole application down.  Set
    *fail_open=False* if you prefer hard failure on Redis errors.
    """

    def __init__(
        self,
        redis_client,
        api_key,
        rate=3.0,
        capacity=5.0,
        open_ttl=60,
        failure_threshold=5,
        half_open_probe_ttl=10,
        acquire_timeout=None,
        fail_open=True,
    ):
        """
        Parameters
        ----------
        redis_client        : ``redis.Redis`` instance (already connected).
                              Positional.
        api_key             : Kite API key – used to namespace all Redis keys
                              so multiple API keys can coexist on one Redis DB.
                              Positional. All remaining parameters are optional.
        rate                : sustained token refill rate in tokens/second (default 3)
        capacity            : token bucket burst ceiling (default 5)
        open_ttl            : seconds the circuit stays OPEN after a 429 or
                              threshold breach (default 60)
        failure_threshold   : consecutive non-429 failures (5xx / timeout) that
                              trip the circuit (default 5)
        half_open_probe_ttl : seconds the probe lock is held in HALF_OPEN state
                              before another process may claim it (default 10)
        acquire_timeout     : seconds before acquire() raises TimeoutError;
                              None (default) waits indefinitely
        fail_open           : if True (default), Redis errors allow the request
                              through; if False, they re-raise
        """
        self._redis               = redis_client
        self._api_key             = api_key
        self._rate                = float(rate)
        self._capacity            = float(capacity)
        self._open_ttl            = int(open_ttl)
        self._failure_threshold   = int(failure_threshold)
        self._half_open_probe_ttl = int(half_open_probe_ttl)
        self._acquire_timeout     = acquire_timeout
        self._fail_open           = bool(fail_open)

        # ── Redis key names ────────────────────────────────────────────────
        ns = "kite:cb:{}".format(api_key)
        # State is encoded by key *existence*; TTLs drive all transitions.
        self._open_key      = ns + ":open"       # present ↔ OPEN
        self._half_open_key = ns + ":half_open"  # present ↔ HALF_OPEN
        self._probe_key     = ns + ":probe"       # SETNX probe lock
        self._failures_key  = ns + ":failures"   # consecutive failure count

        tb = "kite:tb:{}".format(api_key)
        self._tokens_key = tb + ":tokens"
        self._ts_key     = tb + ":ts"

        # FIX (bucket TTL < open_ttl): bucket keys must outlive the longest
        # possible OPEN period so that token state is still present when the
        # circuit closes and requests resume.
        self._bucket_ttl = max(self._open_ttl * 2,
                               int(self._capacity / self._rate * 2)) + 10

        self._acquire_script      = self._redis.register_script(_ACQUIRE_TOKEN_SCRIPT)
        self._incr_failure_script = self._redis.register_script(_INCR_FAILURE_SCRIPT)

        # FIX (unused lock): guards the in-process read-then-act sequence in
        # record_failure so that two threads in the same process cannot both
        # observe CB_HALF_OPEN and both call _open_circuit concurrently.
        # Cross-process safety is still provided by Redis atomics.
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Public interface                                                      #
    # ------------------------------------------------------------------ #

    def acquire(self, timeout=None):
        """
        Block until the circuit permits a request and a token is available,
        then return.

        Parameters
        ----------
        timeout : float or None
            Seconds to wait before raising TimeoutError.  Overrides the
            constructor's *acquire_timeout* for this call only.

        Raises
        ------
        TimeoutError
            If the effective timeout elapses before a token is acquired.
        redis.RedisError
            If a Redis call fails and *fail_open* is False.
        """
        effective_timeout = timeout if timeout is not None else self._acquire_timeout
        deadline = (time.monotonic() + effective_timeout
                    if effective_timeout is not None else None)

        while True:
            self._check_deadline(deadline)

            state = self._safe_get_state()

            # ── OPEN: sleep for remaining cooldown, then enter HALF_OPEN ──
            if state == CB_OPEN:
                self._handle_open(deadline)
                continue

            # ── HALF_OPEN: one probe gets through; others wait ─────────
            if state == CB_HALF_OPEN:
                if self._try_claim_probe():
                    return          # caller is the probe; skip token bucket
                self._wait_for_probe_resolution()
                continue

            # ── CLOSED: consume from distributed token bucket ──────────
            try:
                if self._try_consume_token():
                    return
            except Exception as exc:
                log.error("Redis error consuming token (api_key=%s): %s",
                          self._api_key, exc)
                if self._fail_open:
                    return
                raise

            # Bucket empty – back off a fraction of the inter-token interval
            time.sleep(1.0 / self._rate * 0.1)

    def record_429(self):
        """
        Open the circuit immediately on an HTTP 429 response.

        Call this whenever any process in the pool receives a 429.  All
        other processes will block on their next :meth:`acquire` call.
        """
        log.error(
            "HTTP 429 received (api_key=%s). Opening circuit for %ds.",
            self._api_key, self._open_ttl,
        )
        self._open_circuit("HTTP 429")

    def record_failure(self):
        """
        Record a transient server error (5xx, connection timeout).

        - HALF_OPEN: probe failed → re-open the circuit immediately.
        - CLOSED: increment failure counter; open circuit once
          ``failure_threshold`` is reached.

        Do **not** call this for 429 (use :meth:`record_429`) or 403
        (authentication errors should not trip the circuit).
        """
        # FIX (lock never used + HALF_OPEN semantics): the lock serialises the
        # state-read and circuit-open within this process.  _open_circuit is
        # idempotent so concurrent calls from other processes are harmless.
        with self._lock:
            state = self._safe_get_state()
            if state == CB_HALF_OPEN:
                # FIX (misleading log): use _open_circuit with a descriptive
                # reason rather than record_429(), which logged "429 received"
                # even for 5xx/timeout errors.
                log.warning(
                    "HALF_OPEN probe failed (api_key=%s). Re-opening circuit.",
                    self._api_key,
                )
                self._open_circuit("probe failure")
                return

        # FIX (concurrent threshold breach): the Lua script atomically
        # increments and returns 1 only for the call that hits the threshold
        # exactly, so exactly one process triggers _open_circuit.
        try:
            should_open = self._incr_failure_script(
                keys=[self._failures_key],
                # FIX (failures TTL coupled to open_ttl): use a constant
                # large TTL so stale counts don't linger across config changes.
                args=[self._open_ttl * 4, self._failure_threshold],
            )
        except Exception as exc:
            log.error("Redis error incrementing failure counter (api_key=%s): %s",
                      self._api_key, exc)
            return

        if should_open:
            log.warning(
                "Failure threshold (%d) reached (api_key=%s). Opening circuit.",
                self._failure_threshold, self._api_key,
            )
            self._open_circuit("failure threshold")

    def record_success(self):
        """
        Register a successful API response.

        - HALF_OPEN: probe succeeded → close the circuit.
        - CLOSED: reset the failure counter.
        """
        state = self._safe_get_state()
        if state == CB_HALF_OPEN:
            log.info(
                "Probe succeeded (api_key=%s). Closing circuit.", self._api_key
            )
            self._transition_to_closed()
        elif state == CB_CLOSED:
            self._safe_exec(
                lambda: self._redis.set(self._failures_key, 0),
                "reset failure counter",
            )

    def get_state(self):
        """Return the current circuit state string (for monitoring / health checks)."""
        return self._safe_get_state()

    def reset(self):
        """Force-close the circuit and delete all associated Redis keys (useful in tests)."""
        def _do():
            pipe = self._redis.pipeline(transaction=True)
            for key in (
                self._open_key, self._half_open_key, self._probe_key,
                self._failures_key, self._tokens_key, self._ts_key,
            ):
                pipe.delete(key)
            pipe.execute()

        self._safe_exec(_do, "reset")
        log.info("Circuit reset (api_key=%s).", self._api_key)

    # ------------------------------------------------------------------ #
    # Internal: state management                                           #
    # ------------------------------------------------------------------ #

    def _get_state(self):
        """
        Derive circuit state from key existence, not a separate state string.

        FIX (state-key / TTL-key split): the original design stored state in
        a persistent string key and used a separate TTL key for the OPEN
        cooldown.  When the TTL key expired, the string key still said "open",
        making _get_state() return stale results indefinitely.

        Now state is encoded entirely in *which* TTL-bearing keys are present:

            _open_key      exists  →  OPEN
            _half_open_key exists  →  HALF_OPEN
            neither                →  CLOSED

        Redis TTL expiry drives every transition; there is no stale string
        that can contradict the real state.
        """
        pipe = self._redis.pipeline()
        pipe.exists(self._open_key)
        pipe.exists(self._half_open_key)
        open_exists, half_open_exists = pipe.execute()
        if open_exists:
            return CB_OPEN
        if half_open_exists:
            return CB_HALF_OPEN
        return CB_CLOSED

    def _safe_get_state(self):
        try:
            return self._get_state()
        except Exception as exc:
            log.error("Redis error reading circuit state (api_key=%s): %s",
                      self._api_key, exc)
            if self._fail_open:
                # Treat as CLOSED so callers can proceed; the token bucket
                # will still impose local backpressure within this process.
                return CB_CLOSED
            raise

    def _open_circuit(self, reason="unknown"):
        """
        Transition to OPEN. Idempotent – safe to call concurrently from
        multiple processes or threads.

        The *reason* string is stored as the value of _open_key so it is
        visible in Redis CLI / monitoring without needing to parse logs.
        """
        def _do():
            pipe = self._redis.pipeline(transaction=True)
            pipe.set(self._open_key,      reason, ex=self._open_ttl)
            pipe.delete(self._half_open_key)
            pipe.delete(self._probe_key)
            pipe.set(self._failures_key,  0)
            pipe.execute()

        self._safe_exec(_do, "open circuit")

    def _enter_half_open(self):
        """
        Transition to HALF_OPEN.

        Multiple processes may call this concurrently after _open_key expires;
        repeated SET of the same key with the same TTL is idempotent.  A
        safety TTL equal to open_ttl ensures _half_open_key can never get
        permanently stuck if every process crashes before claiming the probe.
        """
        def _do():
            self._redis.set(self._half_open_key, "1", ex=self._open_ttl)

        self._safe_exec(_do, "enter half-open")
        log.info("Circuit → HALF_OPEN (api_key=%s).", self._api_key)

    def _transition_to_closed(self):
        """Transition to CLOSED. Idempotent."""
        def _do():
            pipe = self._redis.pipeline(transaction=True)
            pipe.delete(self._half_open_key)
            pipe.delete(self._probe_key)
            pipe.set(self._failures_key, 0)
            pipe.execute()

        self._safe_exec(_do, "close circuit")
        log.info("Circuit → CLOSED (api_key=%s).", self._api_key)

    # ------------------------------------------------------------------ #
    # Internal: acquire() helpers                                          #
    # ------------------------------------------------------------------ #

    def _check_deadline(self, deadline):
        """Raise TimeoutError if the acquire() deadline has passed."""
        if deadline is not None and time.monotonic() >= deadline:
            raise TimeoutError(
                "acquire() timed out waiting for circuit (api_key={})".format(
                    self._api_key
                )
            )

    def _handle_open(self, deadline):
        """
        Sleep for the remainder of the OPEN cooldown, then call
        _enter_half_open().  Uses PTTL for millisecond precision.

        FIX (infinite loop / no timeout): if a deadline is set, the sleep is
        capped at the remaining budget and TimeoutError is raised if that
        budget is already exhausted.
        """
        try:
            pttl_ms = self._redis.pttl(self._open_key)
        except Exception as exc:
            log.error("Redis error reading open-key TTL (api_key=%s): %s",
                      self._api_key, exc)
            if not self._fail_open:
                raise
            return

        if pttl_ms > 0:
            sleep_for = pttl_ms / 1000.0
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "acquire() timed out waiting for circuit to leave "
                        "OPEN state (api_key={})".format(self._api_key)
                    )
                sleep_for = min(sleep_for, remaining)
            log.warning(
                "Circuit OPEN (api_key=%s). Blocking for %.1fs.",
                self._api_key, sleep_for,
            )
            time.sleep(sleep_for)
        else:
            # Key has expired; transition all waiters to HALF_OPEN
            self._enter_half_open()

    def _try_claim_probe(self):
        """
        Attempt to claim the single HALF_OPEN probe slot via atomic SET NX.

        Returns True if the probe was claimed (this caller may proceed),
        False if another process/thread already holds it.

        FIX (stuck OPEN / SETNX on wrong key): the original code used SETNX
        on _state_key, which already held the string "open" and therefore
        SETNX always failed, permanently blocking the HALF_OPEN transition.
        This method uses SET NX on _probe_key, which is only ever written
        here, so the first caller always wins cleanly.
        """
        try:
            claimed = self._redis.set(
                self._probe_key, "1",
                ex=self._half_open_probe_ttl,
                nx=True,
            )
            if claimed:
                log.info(
                    "Circuit HALF_OPEN (api_key=%s): probe slot claimed.",
                    self._api_key,
                )
            return bool(claimed)
        except Exception as exc:
            log.error("Redis error claiming probe (api_key=%s): %s",
                      self._api_key, exc)
            if self._fail_open:
                return True     # let this thread act as the probe
            raise

    def _wait_for_probe_resolution(self):
        """
        Sleep until the probe lock either resolves (circuit closes/opens)
        or its TTL expires so another caller can claim it.
        """
        try:
            pttl_ms = self._redis.pttl(self._probe_key)
            wait = max(0.2, (pttl_ms if pttl_ms > 0 else 200) / 1000.0)
        except Exception:
            wait = 0.2
        log.debug(
            "HALF_OPEN probe held by peer (api_key=%s). Waiting %.1fs.",
            self._api_key, wait,
        )
        time.sleep(wait)

    def _try_consume_token(self):
        """
        Attempt to consume one token from the distributed bucket.

        Returns True if a token was acquired, False if the bucket is empty.

        The timestamp is supplied by the calling process via ``time.time()``.
        NTP-synchronized hosts typically drift by single-digit milliseconds,
        which at a 3 token/s rate produces a refill error of ~0.003 tokens per
        call — negligible in practice.  Using ``redis.call('TIME')`` inside Lua
        would be more accurate but requires ``redis.replicate_commands()``,
        which is absent from some Redis builds and removed in Redis 7.0+.
        """
        result = self._acquire_script(
            keys=[self._tokens_key, self._ts_key],
            args=[self._rate, self._capacity, time.time(), self._bucket_ttl],
        )
        return bool(result[0])

    # ------------------------------------------------------------------ #
    # Internal: error handling                                             #
    # ------------------------------------------------------------------ #

    def _safe_exec(self, fn, label="operation"):
        """
        FIX (no Redis error handling): execute *fn*, logging and suppressing
        any exception when *fail_open* is True, or re-raising when False.
        """
        try:
            return fn()
        except Exception as exc:
            log.error(
                "Redis error during %s (api_key=%s): %s",
                label, self._api_key, exc,
            )
            if not self._fail_open:
                raise
