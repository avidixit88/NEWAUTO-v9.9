"""Auto-execution manager (E*TRADE).

Design goals:
 - Zero impact on existing signal logic.
 - State survives Streamlit reruns via st.session_state.
 - No cached functions or unstable return shapes.
 - Conservative defaults: LONG-only, confirm-only optional.

This module owns:
 - eligibility gating (time windows, min score, engine selection)
 - lifecycle state machine per symbol
 - order placement + reconciliation (entry -> stop + TP0)
 - end-of-day liquidation (hard close by 15:55 ET)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field, fields, MISSING
from payload_utils import normalize_alert_payload
from datetime import datetime, time
from typing import Any, Dict, Optional, Tuple

import math
import re
import hashlib

import streamlit as st

from email_utils import send_email_alert
from etrade_client import ETradeClient
from sessions import classify_session
import pandas as pd


ET_TZ = "America/New_York"
ENTRY_TIMEOUT_MINUTES = 20  # default; can be overridden via AutoExecConfig.timeout_minutes


def _log(msg: str) -> None:
    """Lightweight logger.

    We intentionally keep logging simple (print) because Streamlit captures stdout
    and because we avoid introducing logging handlers that might interact with
    Streamlit reruns.
    """
    try:
        print(str(msg))
    except Exception:
        pass


def _append_note(lifecycle: "TradeLifecycle", note: str) -> None:
    """Append a short breadcrumb into lifecycle.notes (human-readable)."""
    try:
        n = str(note or "").strip()
        if not n:
            return
        cur = str(lifecycle.notes or "")
        if not cur:
            lifecycle.notes = n
        else:
            lifecycle.notes = f"{cur} | {n}"
    except Exception:
        # Never let observability break the workflow
        pass


def _tick_round(price: Optional[float]) -> Optional[float]:
    """Round prices to a safe equity tick.

    Conservative rule:
      - price < 1.00 -> 4 decimals (0.0001)
      - price >= 1.00 -> 2 decimals (0.01)

    Keeps email display + broker order prices coherent and avoids ultra-granular floats.
    """
    if price is None:
        return None
    try:
        p = float(price)
    except Exception:
        return None
    return round(p, 4) if p < 1.0 else round(p, 2)



def _mk_client_order_id(base_id: str, leg: str) -> str:
    """Deterministic E*TRADE clientOrderId (<=20 chars).

    base_id is typically lifecycle_id; leg is EN/ST/TP/MK.
    """
    base = ''.join(ch for ch in str(base_id) if ch.isalnum())
    leg = ''.join(ch for ch in str(leg) if ch.isalnum()).upper() or 'X'
    max_base = max(1, 20 - len(leg))
    return (base[:max_base] + leg)[:20]

def _fmt_price(price: Optional[float]) -> str:
    p = _tick_round(price)
    if p is None:
        return "—"
    try:
        return f"{p:.4f}" if p < 1.0 else f"{p:.2f}"
    except Exception:
        return str(p)


def _format_realized_today(state: Dict[str, Any]) -> str:
    try:
        today = _now_et().date().isoformat()
    except Exception:
        today = ""
    rows = []
    for r in (state.get("realized_trades") or []):
        if not isinstance(r, dict):
            continue
        ts = str(r.get("closed_ts", "") or "")
        if today and not ts.startswith(today):
            continue
        rows.append(r)
    if not rows:
        return "—"
    vals = [r.get("realized") for r in rows if isinstance(r.get("realized"), (int, float))]
    if vals:
        return f"${float(sum(vals)):,.2f} ({len(vals)} trades)"
    return "N/A (missing fill prices)"


def _record_activity(state: Dict[str, Any], kind: str, lifecycle: Optional["TradeLifecycle"]=None, details: str = "") -> None:
    """Append a lightweight activity event for hourly reporting (does not affect execution)."""
    try:
        log = state.setdefault("activity_log", [])
        if not isinstance(log, list):
            log = []
            state["activity_log"] = log
        evt = {
            "ts": _now_et().isoformat(),
            "kind": str(kind or "").upper().strip()[:40],
        }
        if lifecycle is not None:
            try:
                evt["symbol"] = str(getattr(lifecycle, "symbol", "") or "")
                evt["engine"] = str(getattr(lifecycle, "engine", "") or "")
                evt["lifecycle_id"] = str(getattr(lifecycle, "lifecycle_id", "") or "")
            except Exception:
                pass
        if details:
            evt["details"] = str(details)[:200]
        log.append(evt)
        # Keep bounded (Streamlit session_state)
        if len(log) > 500:
            del log[:-300]
    except Exception:
        pass


def _activity_since_last_report(state: Dict[str, Any]) -> Tuple[list[dict], str]:
    """Return (events, cutoff_ts_iso). cutoff is stored in state."""
    cutoff = str(state.get("activity_cutoff_ts") or "")
    events = state.get("activity_log") or []
    if not isinstance(events, list):
        return ([], cutoff)
    if not cutoff:
        # default cutoff = start of today ET
        try:
            now = _now_et()
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        except Exception:
            cutoff = ""
    out = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = str(e.get("ts") or "")
        if cutoff and ts and ts <= cutoff:
            continue
        out.append(e)
    return (out, cutoff)



@dataclass
class AutoExecConfig:
    enabled: bool
    sandbox: bool
    engines: Tuple[str, ...]
    min_score: float
    max_dollars_per_trade: float
    max_pool_dollars: float
    max_concurrent_symbols: int
    lifecycles_per_symbol_per_day: int
    timeout_minutes: int
    tp0_deviation: float
    confirm_only: bool
    status_emails: bool
    hourly_pnl_emails: bool



    entry_mode: str  # 'touch_required' | 'early_band' | 'immediate_on_stage'

    early_entry_limit_orders: bool
    entry_distance_guard_bps: float

    enforce_entry_windows: bool
    entry_grace_minutes: int

    # Optional marketable-limit buffer for ENTRY (immediate_on_stage only).
    # When enabled, we nudge the entry limit upward by a small, tick-rounded amount to improve fills.
    use_entry_buffer: bool = False
    entry_buffer_max: float = 0.01

    # Execution-window controls for order submission.
    # IMPORTANT: these are intentionally DECOUPLED from the scanner session toggles.
    # If these fields are missing (older saved config), auto-exec will default to
    # allowing entries in both windows.
    exec_allow_opening: bool = True
    exec_allow_midday: bool = True
    exec_allow_power: bool = True

    # Broker ping / token validity check.
    # When enabled, auto-exec will periodically call a lightweight broker endpoint
    # to verify OAuth tokens are not just present, but actually working.
    broker_ping_enabled: bool = True
    broker_ping_interval_sec: int = 60

    # Entry price source: for entry placement we can either fetch a fresh quote (GLOBAL_QUOTE)
    # or strictly use the last cached price observed by the scanners.
    entry_use_last_price_cache_only: bool = False

    # If a lifecycle is STAGED but entry isn't sent, optionally email the skip reason once per lifecycle.
    email_on_entry_skip: bool = True

    # For reconciliation checks that may need last price (e.g., stop-breach cancel on unfilled entry),
    # prefer using the scanner-cached LAST to avoid additional quote requests.
    reconcile_use_last_price_cache_only: bool = True

    # Periodic digest email (observability) — does not affect execution.
    digest_emails_enabled: bool = False
    digest_interval_minutes: int = 15
    digest_rth_only: bool = True

@dataclass
class TradeLifecycle:
    symbol: str
    engine: str
    created_ts: str
    # Lifecycle stage contract:
    #   PRESTAGED: signal captured but NOT executable because broker is not "armed" (OAuth/account).
    #   STAGED: executable (subject to execution window + price gates).
    #   ENTRY_SENT, IN_POSITION, CLOSED, CANCELED, CANCEL_PENDING
    stage: str
    desired_entry: float
    stop: float
    tp0: float  # already adjusted by cfg.tp0_deviation (exit limit)
    qty: int
    reserved_dollars: float
    # Order IDs stored as strings because state persists in st.session_state (JSON-ish)
    entry_order_id: Optional[str] = None
    stop_order_id: Optional[str] = None
    tp0_order_id: Optional[str] = None
    market_exit_order_id: Optional[str] = None

    filled_qty: int = 0
    entry_sent_ts: Optional[str] = None
    bracket_qty: int = 0
    emailed_events: Dict[str, str] = field(default_factory=dict)
    notes: str = ""
    # Last evaluation breadcrumb for debugging why an entry did/didn't send.
    last_entry_eval: str = ""

    # How many times this lifecycle has been evaluated for entry while STAGED.
    # Stored as an int in session_state for observability (digest emails).
    entry_eval_count: int = 0

    @property
    def lifecycle_id(self) -> str:
        """Canonical lifecycle identifier.

        Some downstream components (e.g., clientOrderId generation, order bookkeeping)
        expect a stable `lifecycle_id`. Older lifecycle objects did not include an
        explicit id field, so we derive a deterministic identifier from immutable
        attributes.

        This is intentionally a *property* (not a dataclass field) to avoid changing
        persisted session-state schema.
        """
        try:
            sym = ''.join(ch for ch in str(self.symbol) if ch.isalnum()).upper()[:6] or 'X'
            raw = f"{self.symbol}|{self.engine}|{self.created_ts}"
            digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
            return f"{sym}{digest}"
        except Exception:
            # Fallback: best-effort stable string
            return ''.join(ch for ch in f"{self.symbol}{self.engine}{self.created_ts}" if ch.isalnum())[:32]


# ----------------------------
# Session-state schema hardening
# ----------------------------


def _coerce_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off", ""}:
            return False
    return default


def _coerce_int(v: Any, default: int = 0) -> int:
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return int(float(v))
    except Exception:
        return default


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        if v is None or (isinstance(v, str) and not v.strip()):
            return default
        return float(v)
    except Exception:
        return default


def _safe_dataclass_from_dict(cls, raw: Any, *, required_defaults: Optional[Dict[str, Any]] = None,
                              coercers: Optional[Dict[str, Any]] = None) -> Any:
    """Safely hydrate a dataclass from a dict.

    Goals:
      - ignore unknown keys (old/new versions)
      - backfill missing keys using dataclass defaults or required_defaults
      - optionally coerce types for stability across reruns and deployments
    """
    if not isinstance(raw, dict):
        raw = {} if raw is None else dict(getattr(raw, "__dict__", {}) or {})

    required_defaults = required_defaults or {}
    coercers = coercers or {}

    out: Dict[str, Any] = {}
    flds = list(fields(cls))
    allowed = {f.name for f in flds}

    # Filter unknown keys first (prevents unexpected keyword errors).
    filtered = {k: raw.get(k) for k in allowed if k in raw}

    for f in flds:
        name = f.name
        if name in filtered:
            val = filtered[name]
        else:
            if f.default is not MISSING:
                val = f.default
            elif getattr(f, "default_factory", MISSING) is not MISSING:  # type: ignore
                val = f.default_factory()  # type: ignore
            elif name in required_defaults:
                val = required_defaults[name]
            else:
                # Last-resort: keep None for missing required fields.
                val = None

        if name in coercers:
            try:
                val = coercers[name](val)
            except Exception:
                pass
        out[name] = val

    return cls(**out)


def autoexec_cfg_from_raw(raw: Any) -> AutoExecConfig:
    """Hydrate AutoExecConfig from session_state safely (schema tolerant)."""
    required = {
        # Conservative defaults so missing keys don't arm trading accidentally.
        "enabled": False,
        "sandbox": True,
        "engines": tuple(),
        "min_score": 0.0,
        "max_dollars_per_trade": 0.0,
        "max_pool_dollars": 0.0,
        "max_concurrent_symbols": 1,
        "lifecycles_per_symbol_per_day": 1,
        "timeout_minutes": ENTRY_TIMEOUT_MINUTES,
        "tp0_deviation": 0.0,
        "confirm_only": False,
        "status_emails": False,
        "hourly_pnl_emails": False,
        "entry_mode": "touch_required",
        "early_entry_limit_orders": False,
        "entry_distance_guard_bps": 0.0,
        "enforce_entry_windows": True,
        "entry_grace_minutes": 0,
        "reconcile_use_last_price_cache_only": True,
        "digest_emails_enabled": False,
        "digest_interval_minutes": 15,
        "digest_rth_only": True,
    }
    coercers = {
        "enabled": lambda v: _coerce_bool(v, False),
        "sandbox": lambda v: _coerce_bool(v, True),
        "min_score": lambda v: _coerce_float(v, 0.0),
        "max_dollars_per_trade": lambda v: _coerce_float(v, 0.0),
        "max_pool_dollars": lambda v: _coerce_float(v, 0.0),
        "max_concurrent_symbols": lambda v: _coerce_int(v, 1),
        "lifecycles_per_symbol_per_day": lambda v: _coerce_int(v, 1),
        "timeout_minutes": lambda v: _coerce_int(v, ENTRY_TIMEOUT_MINUTES),
        "tp0_deviation": lambda v: _coerce_float(v, 0.0),
        "confirm_only": lambda v: _coerce_bool(v, False),
        "status_emails": lambda v: _coerce_bool(v, False),
        "hourly_pnl_emails": lambda v: _coerce_bool(v, False),
        "early_entry_limit_orders": lambda v: _coerce_bool(v, False),
        "entry_distance_guard_bps": lambda v: _coerce_float(v, 0.0),
        "enforce_entry_windows": lambda v: _coerce_bool(v, True),
        "entry_grace_minutes": lambda v: _coerce_int(v, 0),
        "exec_allow_opening": lambda v: _coerce_bool(v, True),
        "exec_allow_midday": lambda v: _coerce_bool(v, True),
        "exec_allow_power": lambda v: _coerce_bool(v, True),
        "broker_ping_enabled": lambda v: _coerce_bool(v, True),
        "broker_ping_interval_sec": lambda v: max(15, min(300, _coerce_int(v, 60))),
        "reconcile_use_last_price_cache_only": lambda v: _coerce_bool(v, True),
        "digest_emails_enabled": lambda v: _coerce_bool(v, False),
        "digest_interval_minutes": lambda v: max(5, min(60, _coerce_int(v, 15))),
        "digest_rth_only": lambda v: _coerce_bool(v, True),
    }
    cfg = _safe_dataclass_from_dict(AutoExecConfig, raw, required_defaults=required, coercers=coercers)
    # Ensure engines is a tuple[str,...]
    try:
        if isinstance(cfg.engines, list):
            cfg.engines = tuple(str(x) for x in cfg.engines)  # type: ignore
        elif isinstance(cfg.engines, str):
            cfg.engines = tuple([cfg.engines])  # type: ignore
    except Exception:
        pass
    return cfg


def lifecycle_from_raw(raw: Any) -> TradeLifecycle:
    """Hydrate TradeLifecycle from session_state safely (schema tolerant)."""
    now_ts = _now_et().isoformat()
    required = {
        "symbol": str(getattr(raw, "get", lambda k, d=None: d)("symbol", "UNKNOWN")) if isinstance(raw, dict) else "UNKNOWN",
        "engine": (raw.get("engine") if isinstance(raw, dict) else "") or "",
        "created_ts": (raw.get("created_ts") if isinstance(raw, dict) else None) or now_ts,
        "stage": (raw.get("stage") if isinstance(raw, dict) else None) or "CANCELED",
        "desired_entry": 0.0,
        "stop": 0.0,
        "tp0": 0.0,
        "qty": 0,
        "reserved_dollars": 0.0,
    }
    coercers = {
        "symbol": lambda v: str(v or "UNKNOWN"),
        "engine": lambda v: str(v or ""),
        "created_ts": lambda v: str(v or now_ts),
        "stage": lambda v: str(v or "CANCELED"),
        "desired_entry": lambda v: _coerce_float(v, 0.0),
        "stop": lambda v: _coerce_float(v, 0.0),
        "tp0": lambda v: _coerce_float(v, 0.0),
        "qty": lambda v: max(0, _coerce_int(v, 0)),
        "reserved_dollars": lambda v: max(0.0, _coerce_float(v, 0.0)),
        "filled_qty": lambda v: max(0, _coerce_int(v, 0)),
        "bracket_qty": lambda v: max(0, _coerce_int(v, 0)),
        "emailed_events": lambda v: v if isinstance(v, dict) else {},
        "notes": lambda v: str(v or ""),
        "last_entry_eval": lambda v: str(v or ""),
        "entry_eval_count": lambda v: max(0, _coerce_int(v, 0)),
        "entry_order_id": lambda v: (str(v) if v else None),
        "stop_order_id": lambda v: (str(v) if v else None),
        "tp0_order_id": lambda v: (str(v) if v else None),
        "market_exit_order_id": lambda v: (str(v) if v else None),
        "entry_sent_ts": lambda v: (str(v) if v else None),
    }
    lc = _safe_dataclass_from_dict(TradeLifecycle, raw, required_defaults=required, coercers=coercers)
    # Ensure stage is a known lifecycle stage string
    try:
        if lc.stage not in {"PRESTAGED", "STAGED", "ENTRY_SENT", "IN_POSITION", "CLOSED", "CANCELED", "CANCEL_PENDING"}:
            lc.notes = (lc.notes + " | bad_state:unknown_stage").strip(" |")
            lc.stage = "CANCELED"
    except Exception:
        pass
    return lc


def _normalize_state_schemas(state: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize durable state schemas in-place.

    Ensures:
      - lifecycles: dict[str, list[dict]]
      - each lifecycle dict matches current TradeLifecycle schema
      - pool_reserved remains numeric

    This prevents "bad session state" issues when upgrading builds where the
    persisted dict schema may have missing/extra keys.
    """
    try:
        if not isinstance(state, dict):
            return state

        # Normalize lifecycles container
        lifecycles = state.get("lifecycles")
        if not isinstance(lifecycles, dict):
            lifecycles = {}
            state["lifecycles"] = lifecycles

        for sym, lst in list(lifecycles.items()):
            if not isinstance(sym, str):
                # Drop non-string symbol keys
                lifecycles.pop(sym, None)
                continue
            if not isinstance(lst, list):
                lst = []
            normalized: list = []
            for raw in lst:
                try:
                    lc = lifecycle_from_raw(raw)
                    # If symbol is missing or malformed, make it explicit and cancel.
                    if not lc.symbol or lc.symbol == "UNKNOWN":
                        lc.stage = "CANCELED"
                        lc.notes = (lc.notes + " | bad_state:missing_symbol").strip(" |")
                    normalized.append(asdict(lc))
                except Exception:
                    # Worst-case: drop irrecoverable entries (prevents poisoning state)
                    continue
            lifecycles[sym] = normalized

        # Normalize pool_reserved
        try:
            state["pool_reserved"] = float(state.get("pool_reserved", 0.0) or 0.0)
        except Exception:
            state["pool_reserved"] = 0.0

    except Exception:
        return state
    return state

def _now_et() -> datetime:
    # Streamlit environment should have tzdata; fall back to naive local if needed.
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(ET_TZ))
    except Exception:
        return datetime.now()


def _exec_window_label(now: datetime) -> str:
    """Return which exec window the time falls into: OPENING / MIDDAY / POWER / OFF."""
    t = now.time()
    if time(9, 50) <= t <= time(11, 0):
        return "OPENING"
    if time(14, 0) <= t <= time(15, 30):
        return "MIDDAY" if t < time(15, 0) else "POWER"
    return "OFF"


def _in_exec_window(now: datetime, cfg: AutoExecConfig) -> bool:
    """Two windows: 09:50–11:00 and 14:00–15:30 ET.

    IMPORTANT: execution windows are controlled by AutoExecConfig.exec_allow_* and are
    intentionally decoupled from the scanner session toggles.
    """
    label = _exec_window_label(now)
    if label == "OPENING":
        return bool(getattr(cfg, "exec_allow_opening", True))
    if label == "MIDDAY":
        return bool(getattr(cfg, "exec_allow_midday", True))
    if label == "POWER":
        return bool(getattr(cfg, "exec_allow_power", True))
    return False


def _is_liquidation_time(now: datetime) -> bool:
    return now.time() >= time(15, 55)


def _get_state() -> Dict[str, Any]:
    """Return the durable autoexec state stored in st.session_state.

    Important: app.py may create st.session_state['autoexec'] during OAuth
    before this function is ever called. In that case, we *must not* wipe
    the auth tokens when we "initialize" the rest of the state.
    """

    today = _now_et().date().isoformat()

    # Start from whatever is present (OAuth flow may have created a partial dict)
    existing: Dict[str, Any] = st.session_state.get("autoexec", {}) or {}
    existing_auth = existing.get("auth", {}) if isinstance(existing, dict) else {}

    # Initialize / backfill missing keys without losing auth
    state: Dict[str, Any] = {
        "pool_reserved": float(existing.get("pool_reserved", 0.0)) if isinstance(existing, dict) else 0.0,
        "lifecycles": existing.get("lifecycles", {}) if isinstance(existing, dict) else {},
        "auth": existing_auth if isinstance(existing_auth, dict) else {},
        "day": str(existing.get("day", today)) if isinstance(existing, dict) else today,
        "skip_notices": existing.get("skip_notices", {}) if isinstance(existing, dict) else {},
        "hourly_report_last": str(existing.get("hourly_report_last", "")) if isinstance(existing, dict) else "",
        # Observability / reporting (must persist across Streamlit reruns)
        # NOTE: Without these keys, digest / hourly emails can spam on every rerun.
        "activity_log": existing.get("activity_log", []) if isinstance(existing, dict) else [],
        "activity_cutoff_ts": str(existing.get("activity_cutoff_ts", "")) if isinstance(existing, dict) else "",
        "digest_last_ts": str(existing.get("digest_last_ts", "")) if isinstance(existing, dict) else "",
        "digest_cutoff_ts": str(existing.get("digest_cutoff_ts", "")) if isinstance(existing, dict) else "",
        "last_action": str(existing.get("last_action", "")) if isinstance(existing, dict) else "",
        "realized_trades": existing.get("realized_trades", []) if isinstance(existing, dict) else [],
        "broker_ping": existing.get("broker_ping", {}) if isinstance(existing, dict) else {},
    }

    # Daily reset (preserve auth so "auth before boot" remains valid)
    if state.get("day") != today:
        state = {
            "pool_reserved": 0.0,
            "lifecycles": {},
            "auth": state.get("auth", {}),
            "day": today,
            "skip_notices": {},
            "hourly_report_last": "",
            # Reset reporting cursors daily, but keep structure so we don't spam.
            "activity_log": [],
            "activity_cutoff_ts": "",
            "digest_last_ts": "",
            "digest_cutoff_ts": "",
            "last_action": "",
            "realized_trades": [],
            "broker_ping": {},
        }

    st.session_state["autoexec"] = state

    # Schema normalize on every access so upgrades can't poison state.
    try:
        state = _normalize_state_schemas(state)
        st.session_state["autoexec"] = state
    except Exception:
        pass

    # Self-heal pool_reserved drift:
    # pool_reserved is intended to reflect ONLY the dollars reserved by ACTIVE lifecycles.
    # In rare Streamlit rerun/cancel edge-cases, pool_reserved may drift from the lifecycle
    # truth. We recompute from active lifecycles on every state access.
    try:
        lifecycles = state.get("lifecycles", {}) or {}
        s = 0.0
        for _, lst in lifecycles.items():
            for raw in (lst or []):
                stg = str((raw or {}).get("stage", ""))
                if stg in {"STAGED", "ENTRY_SENT", "IN_POSITION"}:
                    s += float((raw or {}).get("reserved_dollars", 0.0) or 0.0)
        if abs(float(state.get("pool_reserved", 0.0) or 0.0) - s) > 0.01:
            state["pool_reserved"] = float(s)
            st.session_state["autoexec"] = state
    except Exception:
        pass

    return state



def _email_settings():
    """Read SMTP settings from Streamlit secrets. Returns tuple or None."""
    try:
        cfg = st.secrets.get("email", {}) or {}
    except Exception:
        return None
    smtp_server = cfg.get("smtp_server")
    smtp_port = cfg.get("smtp_port")
    smtp_user = cfg.get("smtp_user")
    smtp_password = cfg.get("smtp_password")

    # Accept to_emails (preferred) OR to_email (string). Normalize to list[str].
    to_emails = cfg.get("to_emails")
    if to_emails is None:
        to_email = cfg.get("to_email", "")
        if isinstance(to_email, str):
            # Support comma-separated lists in legacy config.
            parts = [p.strip() for p in to_email.split(",") if p.strip()]
            to_emails = parts
        elif to_email:
            to_emails = [str(to_email).strip()]
        else:
            to_emails = []
    if isinstance(to_emails, str):
        to_emails = [e.strip() for e in to_emails.split(",") if e.strip()]
    if not (smtp_server and smtp_port and smtp_user and smtp_password and to_emails):
        return None
    try:
        smtp_port_int = int(smtp_port)
    except Exception:
        return None
    return smtp_server, smtp_port_int, str(smtp_user), str(smtp_password), [str(e).strip() for e in to_emails if str(e).strip()]


def _send_status_email(cfg: AutoExecConfig, subject: str, body: str) -> None:
    """Send auto-exec lifecycle status emails (one email per recipient)."""
    if not getattr(cfg, "status_emails", False):
        return
    settings = _email_settings()
    if settings is None:
        return
    smtp_server, smtp_port, smtp_user, smtp_password, to_emails = settings
    try:
        send_email_alert(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            to_emails=to_emails,
            subject=subject,
            body=body,
        )
    except Exception:
        # Never crash the app due to email
        return


def _should_send_hourly(now: datetime) -> Optional[str]:
    """Return an hourly key (YYYY-MM-DD:HH) if we are within the report window.

    We aim for "every hour" during the regular session. Because Streamlit reruns
    on a timer, we allow a small minute window so we don't miss the top of the hour.
    """
    # Monday=0 ... Sunday=6
    if now.weekday() > 4:
        return None

    t = now.time()
    # Regular session 09:30–16:00 ET
    if t < time(9, 30) or t > time(16, 0):
        return None

    # Send at 10:00, 11:00, ... 16:00 (inclusive). Allow minute window [0, 7].
    if now.hour < 10 or now.hour > 16:
        return None
    if not (0 <= now.minute <= 7):
        return None

    return f"{now.date().isoformat()}:{now.hour:02d}"




def _digest_activity_since_last(state: Dict[str, Any]) -> Tuple[list[dict], str]:
    """Return (events, cutoff_ts_iso) for digest emails (separate from hourly)."""
    cutoff = str(state.get("digest_cutoff_ts") or "")
    events = state.get("activity_log") or []
    if not isinstance(events, list):
        return ([], cutoff)
    if not cutoff:
        # default cutoff = start of today ET
        try:
            now = _now_et()
            cutoff = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        except Exception:
            cutoff = ""
    out_events = []
    for e in events:
        if not isinstance(e, dict):
            continue
        ts = str(e.get("ts") or "")
        if cutoff and ts and ts <= cutoff:
            continue
        out_events.append(e)
    return (out_events, cutoff)


def _maybe_send_autoexec_digest(cfg: AutoExecConfig, state: Dict[str, Any], now: datetime) -> None:
    """Send a periodic auto-exec digest email (observability only)."""
    if not getattr(cfg, "digest_emails_enabled", False):
        return

    # Respect RTH-only if enabled.
    if getattr(cfg, "digest_rth_only", True):
        try:
            ts = pd.Timestamp(now)
            rth = classify_session(ts, allow_opening=True, allow_midday=True, allow_power=True, allow_premarket=False, allow_afterhours=False)
            if rth == "OFF":
                return
        except Exception:
            pass

    interval_min = int(getattr(cfg, "digest_interval_minutes", 15) or 15)
    interval_sec = max(60, min(3600, interval_min * 60))

    last_ts = str(state.get("digest_last_ts") or "")
    if last_ts:
        try:
            prev = datetime.fromisoformat(last_ts)
            if (now - prev).total_seconds() < interval_sec:
                return
        except Exception:
            # if parse fails, send and reset
            pass

    # Summarize lifecycle state
    lifecycles = state.get("lifecycles", {}) or {}
    stage_counts: Dict[str, int] = {}
    active_rows = []
    for sym, lst in lifecycles.items():
        if not isinstance(lst, list):
            continue
        for raw in lst:
            if not isinstance(raw, dict):
                continue
            stg = str(raw.get("stage") or "")
            stage_counts[stg] = stage_counts.get(stg, 0) + 1
            if stg in {"STAGED", "PRESTAGED", "ENTRY_SENT", "IN_POSITION", "CANCEL_PENDING"}:
                active_rows.append(raw)

    # Activity since last digest
    events, _cutoff = _digest_activity_since_last(state)
    counts: Dict[str, int] = {}
    for e in events:
        k = str(e.get("kind") or "")
        counts[k] = counts.get(k, 0) + 1

    # Broker ping status
    bp = state.get("broker_ping") or {}
    bp_ok = None
    bp_err = ""
    bp_ts = ""
    if isinstance(bp, dict):
        if "ok" in bp:
            bp_ok = bool(bp.get("ok"))
        bp_err = str(bp.get("err") or "")
        bp_ts = str(bp.get("ts") or "")

    subj = f"[AUTOEXEC] Digest ({now.strftime('%H:%M')} ET)"

    lines: list[str] = []
    lines.append(f"Time (ET): {now.isoformat()}")
    lines.append(f"Env: {'SANDBOX' if getattr(cfg, 'sandbox', True) else 'LIVE'}")
    if bp_ok is not None:
        extra = f" ({bp_ts})" if bp_ts else ""
        if (not bp_ok) and bp_err:
            extra += f" — {bp_err}"
        lines.append(f"Broker ping: {'OK' if bp_ok else 'FAILED'}{extra}")

    lines.append("")
    lines.append("Lifecycle counts:")
    if stage_counts:
        for k in sorted(stage_counts.keys()):
            lines.append(f"  • {k}: {stage_counts[k]}")
    else:
        lines.append("  • —")

    lines.append("")
    lines.append("Activity since last digest:")
    if counts:
        for k in sorted(counts.keys()):
            lines.append(f"  • {k}: {counts[k]}")
    else:
        lines.append("  • —")

    if active_rows:
        lines.append("")
        lines.append("Active lifecycles (top 12):")
        def _cts(r):
            return str(r.get("created_ts") or "")
        for r in sorted(active_rows, key=_cts)[-12:]:
            sym = str(r.get("symbol") or "")
            stg = str(r.get("stage") or "")
            oid = str(r.get("entry_order_id") or "")
            evc = r.get("entry_eval_count")
            last_eval = str(r.get("last_entry_eval") or "")
            notes = str(r.get("notes") or "")
            evc_s = ""
            try:
                if evc is not None:
                    evc_s = f" | evals={int(evc)}"
            except Exception:
                evc_s = f" | evals={evc}"
            lines.append(f"  • {sym} | {stg} | entry_oid={oid or '—'}{evc_s}")
            if last_eval:
                lines.append(f"      last_entry_eval: {last_eval}")
            if notes:
                lines.append(f"      notes: {notes}")

    # Persist digest cursor
    state["digest_last_ts"] = now.isoformat()
    state["digest_cutoff_ts"] = now.isoformat()

    _send_status_email(cfg, subj, "\n".join(lines) + "\n")
def _extract_positions(portfolio_json: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Best-effort extraction of position objects from E*TRADE portfolio JSON."""

    positions: list[Dict[str, Any]] = []

    def _walk(obj):
        if isinstance(obj, dict):
            # E*TRADE commonly uses key 'Position' or 'position' for lists
            for k, v in obj.items():
                if str(k).lower() == "position":
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                positions.append(it)
                    elif isinstance(v, dict):
                        positions.append(v)
                else:
                    _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                _walk(it)

    _walk(portfolio_json)
    return positions


def _pos_symbol(pos: Dict[str, Any]) -> str:
    # Common: pos['Product']['symbol']
    try:
        sym = pos.get("Product", {}).get("symbol")
        if sym:
            return str(sym).upper().strip()
    except Exception:
        pass
    # Sometimes: pos['product']['symbol']
    try:
        sym = pos.get("product", {}).get("symbol")
        if sym:
            return str(sym).upper().strip()
    except Exception:
        pass
    # Fallback: traverse
    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() == "symbol" and isinstance(v, str):
                    return v
                r = _walk(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for it in obj:
                r = _walk(it)
                if r:
                    return r
        return ""
    sym = _walk(pos)
    return str(sym).upper().strip() if sym else ""



def _safe_num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _maybe_send_hourly_pnl(cfg: AutoExecConfig, state: Dict[str, Any], client: Optional[ETradeClient]) -> None:
    """Send an hourly P&L + analytics email during the regular session.

    IMPORTANT: This function must never crash the app. It is a best-effort
    reporting feature and must not contain execution invariants.
    """
    if not getattr(cfg, "hourly_pnl_emails", False):
        return

    now = _now_et()
    key = _should_send_hourly(now)
    if not key:
        return

    last = str(state.get("hourly_report_last", "") or "")
    if last == key:
        return

    account_id_key = (state.get("auth", {}) or {}).get("account_id_key")
    if not (client and account_id_key):
        return

    lifecycles = state.get("lifecycles", {}) or {}
    staged = entry_sent = in_pos = closed = canceled = 0
    active_symbols = 0
    managed_symbols: set[str] = set()
    for sym, lst in lifecycles.items():
        managed_symbols.add(str(sym).upper())
        sym_active = False
        for raw in (lst or []):
            stg = str((raw or {}).get("stage", ""))
            if stg == "STAGED":
                staged += 1
            elif stg == "ENTRY_SENT":
                entry_sent += 1
                sym_active = True
            elif stg == "IN_POSITION":
                in_pos += 1
                sym_active = True
            elif stg == "CLOSED":
                closed += 1
            elif stg == "CANCELED":
                canceled += 1
        if sym_active:
            active_symbols += 1

    port_lines: list[str] = []
    total_mkt = total_gl = 0.0
    try:
        pj = client.get_portfolio(str(account_id_key))
        pos_list = _extract_positions(pj)
        rows = []
        for p in pos_list:
            sym = _pos_symbol(p)
            if not sym:
                continue
            if managed_symbols and sym not in managed_symbols:
                continue
            qty = _safe_num(p.get("quantity") or p.get("Quantity"))
            mv = _safe_num(p.get("marketValue") or p.get("MarketValue"))
            gl = _safe_num(p.get("totalGainLoss") or p.get("TotalGainLoss") or p.get("unrealizedGainLoss") or p.get("UnrealizedGainLoss"))
            if mv is not None:
                total_mkt += mv
            if gl is not None:
                total_gl += gl
            rows.append((sym, qty, mv, gl))
        if rows:
            port_lines.append("Bot-managed positions (E*TRADE portfolio):")
            for sym, qty, mv, gl in sorted(rows, key=lambda x: x[0]):
                q = "—" if qty is None else f"{qty:.0f}"
                m = "—" if mv is None else f"${mv:,.2f}"
                g = "—" if gl is None else f"${gl:,.2f}"
                port_lines.append(f"  • {sym}: qty {q} | mkt {m} | P&L {g}")
            port_lines.append(f"Totals (managed): market ${total_mkt:,.2f} | P&L ${total_gl:,.2f}")
        else:
            port_lines.append("No bot-managed positions found in portfolio snapshot.")
    except Exception as e:
        port_lines.append(f"Portfolio snapshot unavailable: {e}")

    order_lines: list[str] = []
    try:
        oj = client.list_orders(str(account_id_key), status="OPEN", count=50)
        open_orders = 0
        def _walk_orders(obj):
            nonlocal open_orders
            if isinstance(obj, dict):
                if "Instrument" in obj and isinstance(obj.get("Instrument"), list):
                    sym = ""
                    try:
                        sym = str(obj.get("Instrument")[0].get("Product", {}).get("symbol", "") or "").upper().strip()
                    except Exception:
                        sym = ""
                    if not managed_symbols or (sym and sym in managed_symbols):
                        open_orders += 1
                for v in obj.values():
                    _walk_orders(v)
            elif isinstance(obj, list):
                for it in obj:
                    _walk_orders(it)
        _walk_orders(oj)
        order_lines.append(f"Open orders (managed): {open_orders}")
    except Exception as e:
        order_lines.append(f"Open orders snapshot unavailable: {e}")

    # --- Activity since last report (lightweight ledger) ---
    events, cutoff = _activity_since_last_report(state)
    activity_lines: list[str] = []
    if events:
        # Aggregate counts
        counts: Dict[str, int] = {}
        for e in events:
            k = str(e.get("kind") or "UNKNOWN").upper()
            counts[k] = counts.get(k, 0) + 1
        activity_lines.append("Activity since last report:")
        # ordered buckets
        for k in ("ENTRY_PLACED", "ENTRY_CANCELED", "ENTRY_TIMEOUT", "BRACKETS_SENT", "EXIT_EXECUTED", "CLOSE", "FLATTEN", "CLEANUP"):
            if k in counts:
                activity_lines.append(f"  • {k}: {counts[k]}")
        # show up to 12 most recent lines
        activity_lines.append("Recent events:")
        for e in events[-12:]:
            ts = str(e.get("ts") or "")
            sym = str(e.get("symbol") or "")
            kind = str(e.get("kind") or "")
            det = str(e.get("details") or "")
            activity_lines.append(f"  • {ts} | {sym} | {kind} {('- ' + det) if det else ''}".rstrip())
    else:
        activity_lines.append("Activity since last report: —")

    subj = f"[AUTOEXEC] Hourly P&L Update — {now.hour:02d}:00 ET"
    body = (
        f"Time (ET): {now.isoformat()}\n"
        f"Environment: {'SANDBOX' if cfg.sandbox else 'LIVE'}\n\n"
        f"Last bot action: {str(state.get('last_action', '') or '—')}\n\n"
        + "\n".join(activity_lines)
        + "\n\n"
        f"Auto‑exec today:\n"
        f"  • Active symbols: {active_symbols}\n"
        f"  • Lifecycles — STAGED: {staged}, ENTRY_SENT: {entry_sent}, IN_POSITION: {in_pos}, CLOSED: {closed}, CANCELED: {canceled}\n"
        f"  • Pool reserved: ${float(state.get('pool_reserved', 0.0) or 0.0):,.2f} / ${float(cfg.max_pool_dollars):,.2f}\n"
        f"  • Realized (today): {_format_realized_today(state)}\n\n"
        + "\n".join(port_lines)
        + "\n\n"
        + "\n".join(order_lines)
        + "\n\n"
        "Note: This report is informational only. Execution logic remains governed by your stop-loss + TP0 rules."
    )
    _send_status_email(cfg, subj, body)
    state["hourly_report_last"] = key
    state["activity_cutoff_ts"] = now.isoformat()

def _event_once(cfg: AutoExecConfig, lifecycle: TradeLifecycle, event_key: str, subject: str, body: str) -> None:
    """Dedup: send an event email once per lifecycle per event_key."""
    try:
        sent = (lifecycle.emailed_events or {}).get(event_key)
    except Exception:
        sent = None
    if sent:
        return
    _send_status_email(cfg, subject, body)
    try:
        lifecycle.emailed_events[event_key] = _now_et().isoformat()
    except Exception:
        pass



def _maybe_email_entry_skip(cfg: AutoExecConfig, lifecycle: TradeLifecycle, now: datetime, reason_code: str, detail: str = "") -> None:
    """Email once when a lifecycle is STAGED but an entry is not sent (observability)."""
    try:
        if not bool(getattr(cfg, "status_emails", False)):
            return
        if not bool(getattr(cfg, "email_on_entry_skip", True)):
            return
        if str(getattr(lifecycle, "stage", "") or "").upper() != "STAGED":
            return
        # Dedup by reason code
        code = ''.join(ch for ch in str(reason_code or "UNKNOWN").upper() if ch.isalnum() or ch == '_')[:40] or "UNKNOWN"
        event_key = f"ENTRY_NOT_SENT_{code}"
        # Compute timeout remaining (best-effort)
        rem = "—"
        try:
            created = datetime.fromisoformat(lifecycle.created_ts)
            age_min = (now - created).total_seconds() / 60.0
            timeout_m = int(getattr(cfg, "timeout_minutes", ENTRY_TIMEOUT_MINUTES) or ENTRY_TIMEOUT_MINUTES)
            rem = f"{max(0, timeout_m - int(age_min))}m remaining (timeout={timeout_m}m)"
        except Exception:
            pass

        subj = f"[AUTOEXEC] {lifecycle.symbol} ENTRY NOT SENT ({code})"
        body = (
            f"Time (ET): {now.isoformat()}\n"
            f"Symbol: {lifecycle.symbol}\n"
            f"Engine: {lifecycle.engine}\n\n"
            f"Stage: {lifecycle.stage}\n"
            f"Reason: {code}\n"
            + (f"Detail: {detail}\n" if detail else "")
            + f"\nWill keep evaluating this STAGED lifecycle on each refresh. {rem}.\n"
        )
        _event_once(cfg, lifecycle, event_key, subj, body)
    except Exception:
        pass


def _record_realized_trade_on_close(state: Dict[str, Any], lifecycle: TradeLifecycle, client: ETradeClient, account_id_key: str, reason: str) -> None:
    """Record best-effort realized P&L for a lifecycle close.

    This is for reporting only. Execution logic does not depend on it.
    """
    try:
        key = f"{lifecycle.symbol}:{lifecycle.created_ts}"
    except Exception:
        key = f"{getattr(lifecycle, 'symbol', 'UNK')}:{getattr(lifecycle, 'created_ts', '')}"

    ledger = state.setdefault("realized_trades", [])
    try:
        if any(isinstance(r, dict) and r.get("key") == key for r in ledger):
            return
    except Exception:
        pass

    entry_oid = _oid_int(lifecycle.entry_order_id)
    if entry_oid is None:
        # Can't compute without entry reference
        ledger.append(
            {
                "key": key,
                "symbol": lifecycle.symbol,
                "engine": lifecycle.engine,
                "closed_ts": _now_et().isoformat(),
                "reason": reason,
                "realized": None,
                "note": "missing_entry_order_id",
            }
        )
        return

    try:
        e_filled, e_avg = client.get_order_filled_and_avg_price(account_id_key, entry_oid)
    except Exception:
        e_filled, e_avg = (int(lifecycle.filled_qty or 0), None)

    # Gather exits (stop/tp0/market remainder)
    exit_components = []
    for label, oid_str in (
        ("STOP", lifecycle.stop_order_id),
        ("TP0", lifecycle.tp0_order_id),
        ("MKT", lifecycle.market_exit_order_id),
    ):
        oid = _oid_int(oid_str)
        if oid is None:
            continue
        try:
            fqty, avg = client.get_order_filled_and_avg_price(account_id_key, oid)
        except Exception:
            fqty, avg = (0, None)
        if int(fqty or 0) > 0:
            exit_components.append((label, int(fqty), avg))

    total_exit_qty = sum(q for _, q, _ in exit_components)
    realized = None
    note = ""
    if e_avg is None:
        note = "missing_entry_avg_price"
    elif total_exit_qty <= 0:
        note = "no_exit_fills_detected"
    else:
        proceeds = 0.0
        missing = False
        for _, q, avg in exit_components:
            if avg is None:
                missing = True
                continue
            proceeds += float(q) * float(avg)
        if missing:
            note = "missing_exit_avg_price"
        else:
            realized = proceeds - float(total_exit_qty) * float(e_avg)

    ledger.append(
        {
            "key": key,
            "symbol": lifecycle.symbol,
            "engine": lifecycle.engine,
            "closed_ts": _now_et().isoformat(),
            "reason": reason,
            "qty": int(total_exit_qty or 0),
            "entry_avg": e_avg,
            "realized": realized,
            "note": note,
        }
    )




def _has_active_lifecycle(state: dict, symbol: str) -> bool:
    """Return True if the symbol has any lifecycle that is not fully closed/canceled."""
    try:
        lifecycles = (state.get('lifecycles') or {}).get(symbol, [])
    except Exception:
        return False
    for lc in lifecycles:
        # lifecycles are persisted as plain dicts in st.session_state
        try:
            if isinstance(lc, dict):
                stage = str(lc.get('stage', '') or '').upper()
            else:
                stage = str(getattr(lc, 'stage', '') or '').upper()
        except Exception:
            stage = ''
        if stage and stage not in {'CLOSED', 'CANCELED'}:
            return True
    return False

def _ensure_brackets(client, account_id_key: str, symbol: str, lifecycle, log_prefix: str = '') -> None:
    """Safety invariant: if we are IN_POSITION, we must have stop+tp0 orders recorded.

    This prevents silent bookkeeping gaps from turning into uncancelable orders later.
    """
    try:
        stage = (lifecycle.stage or '').upper()
        if stage != 'IN_POSITION':
            return
        if not lifecycle.filled_qty or lifecycle.filled_qty <= 0:
            return
        # Stop
        if not lifecycle.stop_order_id and lifecycle.stop is not None:
            stop_oid, stop_pid = client.place_equity_stop_order_ex(
                account_id_key=account_id_key,
                symbol=symbol,
                action='SELL',
                qty=int(lifecycle.filled_qty),
                stop_price=float(_tick_round(lifecycle.stop) or lifecycle.stop),
                client_order_id=_mk_client_order_id(lifecycle.lifecycle_id, 'ST'),
            )
            lifecycle.stop_order_id = str(stop_oid) if stop_oid else None
            if lifecycle.stop_order_id:
                _append_note(lifecycle, f"STOP_PREVIEW_OK pid={stop_pid}")
                _append_note(lifecycle, f"STOP_PLACE_OK oid={stop_oid}")
                _log(f'{log_prefix}Repaired missing STOP orderId={lifecycle.stop_order_id} for {symbol}')
        # TP0
        if not lifecycle.tp0_order_id and lifecycle.tp0 is not None:
            tp_oid, tp_pid = client.place_equity_limit_order_ex(
                account_id_key=account_id_key,
                symbol=symbol,
                action='SELL',
                qty=int(lifecycle.filled_qty),
                limit_price=float(_tick_round(lifecycle.tp0) or lifecycle.tp0),
                client_order_id=_mk_client_order_id(lifecycle.lifecycle_id, 'TP'),
            )
            lifecycle.tp0_order_id = str(tp_oid) if tp_oid else None
            if lifecycle.tp0_order_id:
                _append_note(lifecycle, f"TP0_PREVIEW_OK pid={tp_pid}")
                _append_note(lifecycle, f"TP0_PLACE_OK oid={tp_oid}")
                _log(f'{log_prefix}Repaired missing TP0 orderId={lifecycle.tp0_order_id} for {symbol}')
    except Exception as e:
        _log(f'{log_prefix}Bracket repair failed for {symbol}: {e}')

def _active_symbols(state: Dict[str, Any]) -> int:
    n = 0
    for sym, lst in state.get("lifecycles", {}).items():
        for l in lst:
            if l.get("stage") in {"STAGED", "ENTRY_SENT", "IN_POSITION"}:
                n += 1
                break
    return n


def _symbol_lifecycle_count_today(state: Dict[str, Any], symbol: str) -> int:
    return len(state.get("lifecycles", {}).get(symbol, []))


def _reserve_pool(state: Dict[str, Any], dollars: float, max_pool: float) -> bool:
    if state["pool_reserved"] + dollars > max_pool:
        return False
    state["pool_reserved"] += dollars
    _assert_pool_invariants(state)
    return True


def _release_pool(state: Dict[str, Any], dollars: float) -> None:
    state["pool_reserved"] = max(0.0, float(state.get("pool_reserved", 0.0)) - float(dollars))
    _assert_pool_invariants(state)



def _assert_pool_invariants(state: Dict[str, Any]) -> None:
    """Warn-only: pool_reserved should equal sum(reserved_dollars) of active lifecycles."""
    try:
        lifecycles = state.get('lifecycles', {}) or {}
        s = 0.0
        for _, lst in lifecycles.items():
            for raw in (lst or []):
                stg = str((raw or {}).get('stage', ''))
                if stg in {'STAGED', 'ENTRY_SENT', 'IN_POSITION'}:
                    s += float((raw or {}).get('reserved_dollars', 0.0) or 0.0)
        pool = float(state.get('pool_reserved', 0.0) or 0.0)
        if abs(pool - s) > 0.01:
            _log(f"[AUTOEXEC][BOOKKEEP] pool_reserved mismatch: state={pool:.2f} vs sum_active={s:.2f}")
    except Exception:
        pass
def _set_last_action(state: Dict[str, Any], summary: str) -> None:
    """Store a short human-readable summary of the bot's most recent meaningful action.

    Included in hourly P&L emails so you can quickly see what the bot last did
    without digging through all status emails.
    """
    try:
        state["last_action"] = str(summary or "")
    except Exception:
        pass


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace("$", "").strip()
        return float(x)
    except Exception:
        return None


def _pget(payload: Dict[str, Any], *keys: str, default=None):
    """Case/label-tolerant payload getter.

    Alert payloads can come from different engines and UI tables and may
    use different key casing or labels (e.g., 'Score' vs 'score',
    'Bias' vs 'bias', 'Pullback Band Low' vs 'pb_low').

    This helper makes auto-exec staging robust without touching any engine logic.
    """
    if not isinstance(payload, dict):
        return default
    for k in keys:
        if k in payload:
            return payload.get(k)
    # Fallback: case-insensitive lookup
    try:
        lower_map = {str(k).lower(): k for k in payload.keys()}
        for k in keys:
            kk = str(k).lower()
            if kk in lower_map:
                return payload.get(lower_map[kk])
    except Exception:
        pass
    return default


def build_desired_entry_for_ride(pbl: float, pbh: float, stage: str) -> float:
    rng = max(0.0001, pbh - pbl)
    if str(stage).upper().startswith("PRE"):
        return pbl + 0.33 * rng
    return pbl + 0.66 * rng


def compute_qty(max_dollars: float, entry: float) -> int:
    if entry <= 0:
        return 0
    return int(math.floor(max_dollars / entry))


def should_stage_lifecycle(cfg: AutoExecConfig, payload: Dict[str, Any]) -> bool:
    # LONG-only for v1
    if str(_pget(payload, "bias", "Bias", "BIAS", default="LONG")).upper() != "LONG":
        return False
    score = _parse_float(_pget(payload, "score", "Score", "SCORE"))
    if score is None or score < cfg.min_score:
        return False
    stage = str(_pget(payload, "stage", "Stage", "STAGE", "tier", "Tier", "TIER", default="")).upper()
    if cfg.confirm_only and "CONF" not in stage:
        return False
    return True


def stage_from_payload(cfg: AutoExecConfig, engine: str, payload: Dict[str, Any], stage: str = "STAGED") -> Optional[TradeLifecycle]:
    payload = normalize_alert_payload(payload)
    symbol = str(_pget(payload, "symbol", "Symbol", "SYMBOL", default="") or "").upper().strip()
    if not symbol:
        return None

    entry = _parse_float(_pget(payload, "entry", "Entry", "ENTRY"))
    stop = _parse_float(_pget(payload, "stop", "Stop", "STOP"))
    tp0 = _parse_float(_pget(payload, "tp0", "TP0", "Tp0"))
    if entry is None or stop is None or tp0 is None:
        return None

    # Adjust TP0 by deviation (sell a bit early). For longs: tp0 - dev
    tp0_adj = max(0.0, tp0 - float(cfg.tp0_deviation or 0.0))

    # Tick-size normalization (stored + used everywhere downstream)
    entry = _tick_round(entry) or entry
    stop = _tick_round(stop) or stop
    tp0_adj = _tick_round(tp0_adj) or tp0_adj

    desired_entry = entry
    if engine == "RIDE":
        # Pullback band can be provided with different labels
        pbl = _parse_float(_pget(payload, "pb_low", "PB_Low", "Pullback Band Low", "PB Low", "PullbackLow"))
        pbh = _parse_float(_pget(payload, "pb_high", "PB_High", "Pullback Band High", "PB High", "PullbackHigh"))
        if pbl is not None and pbh is not None:
            desired_entry = build_desired_entry_for_ride(
                pbl,
                pbh,
                str(_pget(payload, "stage", "Stage", "tier", "Tier", default="")),
            )

    
    desired_entry = _tick_round(desired_entry) or desired_entry
    qty = compute_qty(cfg.max_dollars_per_trade, desired_entry)
    if qty <= 0:
        return None

    reserved = qty * desired_entry
    if str(stage).upper() == "PRESTAGED":
        # PRESTAGED lifecycles are non-executable until OAuth + account binding exists.
        # Do not reserve capital yet; reserve on promotion to STAGED.
        reserved = 0.0
    return TradeLifecycle(
        symbol=symbol,
        engine=engine,
        created_ts=_now_et().isoformat(),
        stage=str(stage).upper(),
        desired_entry=float(desired_entry),
        stop=float(stop),
        tp0=float(tp0_adj),
        qty=qty,
        reserved_dollars=float(reserved),
        notes="prestaged" if str(stage).upper() == "PRESTAGED" else "staged",
    )


def ensure_client(cfg: AutoExecConfig) -> Optional[ETradeClient]:
    state = _get_state()
    auth = state.get("auth", {})
    ck = auth.get("consumer_key")
    cs = auth.get("consumer_secret")
    at = auth.get("access_token")
    ats = auth.get("access_token_secret")
    if not (ck and cs and at and ats):
        return None
    try:
        return ETradeClient(
            consumer_key=ck,
            consumer_secret=cs,
            # Environment must match the tokens/consumer key that were used during auth.
            sandbox=bool(auth.get("sandbox", cfg.sandbox)),
            access_token=at,
            access_token_secret=ats,
        )
    except Exception:
        return None


def _broker_ping_cached(cfg: AutoExecConfig, state: dict, client: ETradeClient, account_id_key: str) -> tuple[bool, str]:
    """Return (ok, err). Uses a small cache in session_state to avoid spamming the API.

    The goal is to verify OAuth tokens are *working* (not only present).
    We ping an inexpensive endpoint (OPEN orders, count=1).
    """
    try:
        enabled = bool(getattr(cfg, 'broker_ping_enabled', True))
        interval = int(getattr(cfg, 'broker_ping_interval_sec', 60) or 60)
    except Exception:
        enabled, interval = True, 60
    if not enabled:
        return True, ''

    bp = state.get('broker_ping') if isinstance(state, dict) else None
    try:
        now_ts = _now_et().timestamp()
        last_ts = float((bp or {}).get('ts') or 0.0)
    except Exception:
        now_ts, last_ts = _now_et().timestamp(), 0.0

    # If we pinged recently, trust cached result (but still expose it in UI/state).
    if bp and (now_ts - last_ts) < interval:
        ok = bool(bp.get('ok', False))
        err = str(bp.get('err', '') or '')
        return ok, err

    ok = False
    err = ''
    try:
        # Lightweight token validity check (uses OAuth-signed request).
        client.list_orders(account_id_key, status='OPEN', count=1)
        ok = True
    except Exception as e:
        ok = False
        err = f'{type(e).__name__}: {e}'

    state['broker_ping'] = {'ok': bool(ok), 'ts': float(now_ts), 'err': err}
    return bool(ok), err


def _broker_ready(cfg: AutoExecConfig, state: Dict[str, Any]) -> Tuple[bool, Optional[ETradeClient], str, str]:
    """Return (ready, client, account_id_key, reason).

    "ready" means:
      - OAuth tokens present and a client can be constructed, AND
      - account_id_key is present (account selected/bound).

    This is the "armed" invariant for *any* broker action.
    """
    client = None
    try:
        client = ensure_client(cfg)
    except Exception:
        client = None
    account_id_key = str((state.get("auth", {}) or {}).get("account_id_key") or "")
    if client is None:
        return False, None, account_id_key, "missing_oauth"
    if not account_id_key:
        return False, None, "", "missing_account_id"
    # Optional: verify tokens are actually valid via cached broker ping
    try:
        if getattr(cfg, 'broker_ping_enabled', True):
            ok, perr = _broker_ping_cached(cfg, state, client, account_id_key)
            if not ok:
                return False, None, '', f'oauth_ping_failed:{perr}'
    except Exception:
        # Never crash readiness checks
        pass
    return True, client, account_id_key, ""


def reconcile_and_execute(
    cfg: AutoExecConfig,
    allow_pre: bool,
    allow_opening: bool,
    allow_midday: bool,
    allow_power: bool,
    allow_after: bool,
    fetch_last_price_fn,
) -> None:
    """Runs every Streamlit rerun to reconcile order state and enforce EOD."""
    if not cfg.enabled:
        return

    state = _get_state()
    now = _now_et()


    # Hourly checkpoint email (P&L + simple analytics) is independent of the
    # execution windows. It should still fire even if we're outside the
    # opening/midday/power windows (as long as OAuth is active).
    try:
        if getattr(cfg, "hourly_pnl_emails", False):
            key = _should_send_hourly(now)
            if key and str(state.get("hourly_report_last", "") or "") != key:
                client_for_report = ensure_client(cfg)
                if client_for_report is not None:
                    _maybe_send_hourly_pnl(cfg, state, client_for_report)
                    # Persist any state mutations (dedupe key)
                    st.session_state["autoexec"] = state
    except Exception:
        # Never crash the app due to reporting.
        pass

    # Auto-exec digest email (every N minutes) for visibility — independent of broker readiness.
    try:
        if getattr(cfg, "digest_emails_enabled", False):
            _maybe_send_autoexec_digest(cfg, state, now)
            st.session_state["autoexec"] = state
    except Exception:
        pass

    broker_ready, client, account_id_key, broker_reason = _broker_ready(cfg, state)
    if not broker_ready:
        # Broker not armed: no broker calls can be made (reconcile/exits/liquidation).
        # Record explicit breadcrumbs so operators can see why lifecycles are not progressing.
        lifecycles = state.get("lifecycles", {})
        for symbol, lst in list(lifecycles.items()):
            for idx, raw in enumerate(list(lst)):
                try:
                    lifecycle = lifecycle_from_raw(raw)
                except Exception:
                    continue
                if lifecycle.stage in {"ENTRY_SENT", "IN_POSITION", "CANCEL_PENDING"}:
                    lifecycle.notes = (lifecycle.notes or "") + f" | broker_not_ready:{broker_reason}"
                    # One-time alert per lifecycle (critical only; avoids spam)
                    _event_once(
                        cfg,
                        lifecycle,
                        "BROKER_NOT_READY_RECONCILE",
                        f"[AUTOEXEC] {lifecycle.symbol} BROKER NOT READY (managing orders paused)",
                        f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\nStage: {lifecycle.stage}\n\nBroker is not armed ({broker_reason}). Order management (fills/stops/targets/liquidation) is paused until OAuth/account is restored.\n",
                    )
                lst[idx] = asdict(lifecycle)
        # Persist breadcrumbs
        st.session_state["autoexec"] = state
        return

    # Liquidation enforcement
    if _is_liquidation_time(now):
        _force_liquidate_all(client, account_id_key, cfg, state)
        return

    # IMPORTANT: do NOT stop reconciliation outside the entry windows.
    # Entries are gated inside try_send_entries() via cfg.enforce_entry_windows.
    in_window_now = _in_exec_window(now, cfg)

    # Reconcile lifecycles
    lifecycles = state.get("lifecycles", {})
    for symbol, lst in list(lifecycles.items()):
        for idx, raw in enumerate(list(lst)):
            lifecycle = lifecycle_from_raw(raw)
            try:
                _reconcile_one(client, account_id_key, state, lifecycle, cfg, fetch_last_price_fn=fetch_last_price_fn)
            except Exception as e:
                lifecycle.notes = f"reconcile_error: {e}"
            lst[idx] = asdict(lifecycle)


def handle_alert_for_autoexec(
    cfg: AutoExecConfig,
    engine: str,
    payload: Dict[str, Any],
    allow_pre: bool,
    allow_opening: bool,
    allow_midday: bool,
    allow_power: bool,
    allow_after: bool,
) -> None:
    """Called only when the app has already decided to send an email alert."""
    if not cfg.enabled:
        return
    if engine not in set(cfg.engines):
        return

    now = _now_et()

    # Normalize payload keys for consistent terminology/casing across auto-exec.
    payload = normalize_alert_payload(payload)


    # NOTE: we allow *staging* outside the entry window so the bot can begin
    # tracking a high-score signal early (e.g., after premarket / early opening).
    # Actual entry placement is still gated inside try_send_entries() when
    # cfg.enforce_entry_window is enabled.
    in_window = _in_exec_window(now, cfg)

    if not should_stage_lifecycle(cfg, payload):
        return

    state = _get_state()

    symbol = str(_pget(payload, "symbol", "Symbol", "SYMBOL", default="")).upper().strip()

    # Broker "armed" invariant: any executable lifecycle requires OAuth + account binding.
    broker_ready, _c, _acct, broker_reason = _broker_ready(cfg, state)

    def _skip_once(reason: str) -> None:
        key = f"{symbol}:{reason}"
        sent = state.get("skip_notices", {}).get(key)
        if sent:
            return
        state.setdefault("skip_notices", {})[key] = _now_et().isoformat()
        subj = f"[AUTOEXEC] {symbol} SKIP — {reason}"
        body = f"Time (ET): {_now_et().isoformat()}\nSymbol: {symbol}\nEngine: {engine}\nReason: {reason}\n\nThis is a one-time notice for today."
        _send_status_email(cfg, subj, body)


    # Session gating for STAGING (entries remain RTH-only in try_send_entries).
    # If the user disables a session in the main app settings, we do not stage
    # lifecycles during that session.
    try:
        ts = pd.Timestamp(now)
        actual_session = classify_session(
            ts,
            allow_opening=True,
            allow_midday=True,
            allow_power=True,
            allow_premarket=True,
            allow_afterhours=True,
        )
        allowed_session = classify_session(
            ts,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
            allow_premarket=allow_pre,
            allow_afterhours=allow_after,
        )
        if allowed_session == "OFF":
            _skip_once(f"session_not_allowed ({actual_session})")
            return
    except Exception:
        # Never allow session gating logic to crash auto-exec staging
        pass

    # Prevent overlapping lifecycles for the same symbol.
    # You can have multiple attempts per day, but only one active at a time.
    if _has_active_lifecycle(state, symbol):
        _skip_once("active_lifecycle_exists")
        return

    # If we are not broker-armed yet, capture the signal as PRESTAGED so it's visible
    # (and can be promoted once OAuth/account is ready) but do NOT reserve capital.
    if not broker_ready:
        lifecycle = stage_from_payload(cfg, engine, payload, stage="PRESTAGED")
        if lifecycle is None:
            return
        lifecycle.notes = f"prestaged_broker_not_ready:{broker_reason}"
        state.setdefault("lifecycles", {}).setdefault(symbol, []).append(asdict(lifecycle))

        subj = f"[AUTOEXEC] {symbol} {engine} PRESTAGED (broker not armed)"
        body = (
            f"Time (ET): {now.isoformat()}\n"
            f"Symbol: {symbol}\nEngine: {engine}\n\n"
            f"PRESTAGED — OAuth/account not ready ({broker_reason}).\n"
            f"No order will be sent until authentication is complete and an account is bound.\n\n"
            f"Desired entry: {_fmt_price(lifecycle.desired_entry)}\n"
            f"Stop: {_fmt_price(lifecycle.stop)}\n"
            f"TP0 (exit limit): {_fmt_price(lifecycle.tp0)}\n"
            f"Qty (computed): {lifecycle.qty}\n"
        )
        _event_once(cfg, lifecycle, "PRESTAGED", subj, body)
        # Persist emailed_events mutation
        try:
            lifelist = state.get("lifecycles", {}).get(symbol)
            if isinstance(lifelist, list) and lifelist:
                lifelist[-1] = asdict(lifecycle)
        except Exception:
            pass
        return


    # Concurrency and lifecycle limits
    if _active_symbols(state) >= int(cfg.max_concurrent_symbols):
        _skip_once("max_concurrent_symbols")
        return

    if _symbol_lifecycle_count_today(state, symbol) >= int(cfg.lifecycles_per_symbol_per_day):
        _skip_once("lifecycle_cap")
        return

    lifecycle = stage_from_payload(cfg, engine, payload)
    if lifecycle is None:
        return

    if not in_window:
        lifecycle.notes = (lifecycle.notes or "") + " | staged outside entry window"

    if not _reserve_pool(state, lifecycle.reserved_dollars, float(cfg.max_pool_dollars)):
        _skip_once("pool_cap")
        return

    state.setdefault("lifecycles", {}).setdefault(symbol, []).append(asdict(lifecycle))

    # Status email: staged
    subj = f"[AUTOEXEC] {symbol} {engine} STAGED"
    score = payload.get("Score")
    tier = payload.get("Tier") or payload.get("Stage")
    body = (
        f"Time (ET): {now.isoformat()}\n"
        f"Symbol: {symbol}\nEngine: {engine}\nTier: {tier}\nScore: {score}\n\n"
        f"STAGED — waiting for entry conditions.\n\n"
        f"Desired entry: {_fmt_price(lifecycle.desired_entry)}\n"
        f"Stop: {_fmt_price(lifecycle.stop)}\n"
        f"TP0 (exit limit): {_fmt_price(lifecycle.tp0)}\n"
        f"Qty: {lifecycle.qty}\n"
        f"Reserved: ${lifecycle.reserved_dollars:.2f}\n"
    )

    _event_once(cfg, lifecycle, "STAGED", subj, body)

    # _event_once mutates lifecycle.emailed_events for dedupe.
    # Persist that mutation back into session_state so reruns don't resend.
    try:
        lifelist = state.get("lifecycles", {}).get(symbol)
        if isinstance(lifelist, list) and lifelist:
            lifelist[-1] = asdict(lifecycle)
    except Exception:
        # If persistence fails, it's non-fatal; worst case is a duplicate status email.
        pass


def try_send_entries(cfg: AutoExecConfig, allow_opening: bool, allow_midday: bool, allow_power: bool, fetch_last_price_fn) -> None:
    """Places entry orders for STAGED lifecycles when price is in range.

    Safety:
      - entry is ONLY sent if last is ABOVE stop AND at/below desired entry.
      - staged/entry orders time out after cfg.timeout_minutes.
    """
    if not cfg.enabled:
        return
    state = _get_state()
    broker_ready, client, account_id_key, broker_reason = _broker_ready(cfg, state)

    now = _now_et()


    # Entry-window gating controls NEW entry submissions only.
    # Exits (stops/targets/EOD) are handled in reconcile.
    in_window_now = _in_exec_window(now, cfg)
    enforce_windows = bool(getattr(cfg, "enforce_entry_windows", True))
    grace_min = int(getattr(cfg, "entry_grace_minutes", 0) or 0)
    for symbol, lst in list(state.get("lifecycles", {}).items()):
        for idx, raw in enumerate(list(lst)):
            lifecycle = lifecycle_from_raw(raw)

            # If broker is not armed, we do not attempt any entries.
            # Record an explicit reason once per lifecycle to make wiring issues obvious.
            if not broker_ready:
                lifecycle.last_entry_eval = f"SKIP: broker_not_ready ({broker_reason})"
                if lifecycle.stage in {"STAGED", "PRESTAGED"}:
                    _event_once(
                        cfg,
                        lifecycle,
                        "BROKER_NOT_READY",
                        f"[AUTOEXEC] {lifecycle.symbol} BROKER NOT READY",
                        f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nAuto-exec is enabled but OAuth/account is not armed ({broker_reason}).\nNo orders can be placed until auth is completed.\n",
                    )
                lst[idx] = asdict(lifecycle)
                continue

            # Promote PRESTAGED -> STAGED once broker becomes armed.
            if lifecycle.stage == "PRESTAGED":
                # Reserve capital now (if possible) and begin normal entry evaluation.
                if not _reserve_pool(state, float(lifecycle.qty or 0) * float(lifecycle.desired_entry or 0.0), float(cfg.max_pool_dollars)):
                    lifecycle.stage = "CANCELED"
                    lifecycle.notes = "prestaged_promotion_failed:pool_cap"
                    lifecycle.last_entry_eval = "CANCELED: pool_cap on promotion"
                    _event_once(cfg, lifecycle, "PROMOTION_FAILED", f"[AUTOEXEC] {lifecycle.symbol} PROMOTION FAILED", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nCould not reserve pool dollars when promoting PRESTAGED -> STAGED.\n")
                    lst[idx] = asdict(lifecycle)
                    continue
                lifecycle.reserved_dollars = float(lifecycle.qty or 0) * float(lifecycle.desired_entry or 0.0)
                lifecycle.stage = "STAGED"
                lifecycle.notes = (lifecycle.notes or "").strip() + " | promoted_from_prestaged"
                lifecycle.last_entry_eval = "PROMOTED: PRESTAGED->STAGED"
                _event_once(cfg, lifecycle, "PROMOTED", f"[AUTOEXEC] {lifecycle.symbol} PROMOTED", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nPromoted PRESTAGED -> STAGED now that OAuth/account is armed.\nReserved: ${float(lifecycle.reserved_dollars or 0.0):.2f}\n")
                lst[idx] = asdict(lifecycle)
                # Continue into standard STAGED checks this same rerun
                # (reload lifecycle from dict after mutation below)
                lifecycle = lifecycle_from_raw(lst[idx])

            # Timeout STAGED lifecycles that never got an entry window
            if lifecycle.stage == "STAGED":
                try:
                    created = datetime.fromisoformat(lifecycle.created_ts)
                    age_min = (now - created).total_seconds() / 60.0
                except Exception:
                    age_min = 0.0
                timeout_m = int(getattr(cfg, "timeout_minutes", ENTRY_TIMEOUT_MINUTES) or ENTRY_TIMEOUT_MINUTES)
                if age_min >= timeout_m:
                    lifecycle.stage = "CANCELED"
                    lifecycle.notes = f"staged_timeout_{timeout_m}m"
                    _event_once(cfg, lifecycle, "STAGED_TIMEOUT", f"[AUTOEXEC] {lifecycle.symbol} STAGED TIMEOUT", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nCanceled: staged timeout ({timeout_m}m)\n")
                    _record_activity(state, "ENTRY_TIMEOUT", lifecycle, f"timeout={timeout_m}m")
                    # release unused reserved dollars back to the pool
                    _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                    lifecycle.reserved_dollars = 0.0
                    lst[idx] = asdict(lifecycle)
                    continue

            if lifecycle.stage != "STAGED":
                continue

            # Observability: count how many reruns have evaluated this STAGED lifecycle.
            try:
                lifecycle.entry_eval_count = int(getattr(lifecycle, "entry_eval_count", 0) or 0) + 1
            except Exception:
                lifecycle.entry_eval_count = 1

            # Entry window gating: prevents orders outside selected execution windows.
            # If enforce_entry_windows is ON, we only send entries during the window,
            # with an optional small grace period based on lifecycle.created_ts.
            if enforce_windows and not in_window_now:
                allow_via_grace = False
                if grace_min > 0:
                    try:
                        created_dt = datetime.fromisoformat(lifecycle.created_ts)
                        if _in_exec_window(created_dt, cfg):
                            age_min = (now - created_dt).total_seconds() / 60.0
                            if age_min <= float(grace_min):
                                allow_via_grace = True
                    except Exception:
                        allow_via_grace = False
                if not allow_via_grace:
                    lifecycle.last_entry_eval = f"SKIP: outside_exec_window ({_exec_window_label(now)})"
                    _maybe_email_entry_skip(cfg, lifecycle, now, "outside_exec_window", lifecycle.last_entry_eval)
                    lst[idx] = asdict(lifecycle)
                    continue

            try:
                last = _parse_float(fetch_last_price_fn(symbol))
            except Exception:
                last = None
            if last is None:
                lifecycle.last_entry_eval = "SKIP: last_price_unavailable"
                _maybe_email_entry_skip(cfg, lifecycle, now, "last_price_unavailable", lifecycle.last_entry_eval)
                lst[idx] = asdict(lifecycle)
                continue


            # Entry gating / entry mode:
            # - touch_required: only place entry once we observe last <= desired_entry (and last > stop)
            # - early_band: place a resting limit order slightly ABOVE desired_entry to avoid missing fills due to refresh cadence,
            #               still requires last > stop and last within a configurable band (bps) above desired_entry.
            # - immediate_on_stage: as soon as we are STAGED (and within allowed entry window), place a resting limit order at desired_entry
            #                       as long as last > stop. This maximizes fill capture on pullbacks.
            entry_mode = str(getattr(cfg, "entry_mode", "") or "").strip().lower()
            if not entry_mode:
                # Backward-compat: if old toggle is on, treat as early_band, else touch_required
                entry_mode = "early_band" if bool(getattr(cfg, "early_entry_limit_orders", False)) else "touch_required"

            guard_bps = float(getattr(cfg, "entry_distance_guard_bps", 25.0) or 0.0)

            if entry_mode == "immediate_on_stage":
                if not (last > lifecycle.stop):
                    lifecycle.last_entry_eval = f"SKIP: last<=stop (mode=immediate) last={last} stop={lifecycle.stop}"
                    _maybe_email_entry_skip(cfg, lifecycle, now, "last_le_stop", lifecycle.last_entry_eval)
                    lst[idx] = asdict(lifecycle)
                    continue
            elif entry_mode == "early_band":
                band_mult = 1.0 + max(0.0, guard_bps) / 10000.0
                if not (last > lifecycle.stop and last <= float(lifecycle.desired_entry) * band_mult):
                    lifecycle.last_entry_eval = (
                        f"SKIP: early_band_not_met last={last} stop={lifecycle.stop} desired={lifecycle.desired_entry} "
                        f"guard_bps={guard_bps} band_max={float(lifecycle.desired_entry)*band_mult}"
                    )
                    _maybe_email_entry_skip(cfg, lifecycle, now, "early_band_not_met", lifecycle.last_entry_eval)
                    lst[idx] = asdict(lifecycle)
                    continue
            else:
                # touch_required
                if not (last <= lifecycle.desired_entry and last > lifecycle.stop):
                    lifecycle.last_entry_eval = (
                        f"SKIP: touch_required_not_met last={last} stop={lifecycle.stop} desired={lifecycle.desired_entry}"
                    )
                    _maybe_email_entry_skip(cfg, lifecycle, now, "touch_required_not_met", lifecycle.last_entry_eval)
                    lst[idx] = asdict(lifecycle)
                    continue
            # Place entry order (limit at desired_entry)
            try:
                # Determine entry limit price. In immediate_on_stage mode we can optionally apply
                # a small "marketable-limit" buffer to improve fill probability, while enforcing
                # strict safety bounds (tick rounding + stop invariant).
                entry_limit = float(_tick_round(lifecycle.desired_entry) or lifecycle.desired_entry)
                if entry_mode == "immediate_on_stage" and bool(getattr(cfg, "use_entry_buffer", False)):
                    raw_buf = float(getattr(cfg, "entry_buffer_max", 0.01) or 0.0)
                    buf = min(max(raw_buf, 0.0), 0.03)  # hard cap
                    entry_limit = float(_tick_round(lifecycle.desired_entry + buf) or (lifecycle.desired_entry + buf))
                    if entry_limit <= float(lifecycle.stop):
                        lifecycle.last_entry_eval = (
                            f"SKIP: entry_buffer_violates_stop entry_limit={entry_limit} stop={lifecycle.stop}"
                        )
                        lst[idx] = asdict(lifecycle)
                        continue
                coid = _mk_client_order_id(lifecycle.lifecycle_id, 'EN')
                oid, pid = client.place_equity_limit_order_ex(
                    account_id_key=account_id_key,
                    symbol=symbol,
                    qty=lifecycle.qty,
                    limit_price=float(entry_limit),
                    action="BUY",
                    market_session="REGULAR",
                    client_order_id=coid,
                )
                lifecycle.entry_order_id = oid
                lifecycle.entry_sent_ts = now.isoformat()
                lifecycle.stage = "ENTRY_SENT"
                lifecycle.notes = f"entry_sent@{lifecycle.desired_entry}"
                _append_note(lifecycle, f"ENTRY_PREVIEW_OK pid={pid}")
                _append_note(lifecycle, f"ENTRY_PLACE_OK oid={oid}")
                lifecycle.last_entry_eval = f"SENT: entry_limit oid={oid} pid={pid}"
                _log(f"[AUTOEXEC][BROKER] place_ok sym={symbol} lc={lifecycle.lifecycle_id} coid={coid} pid={pid} oid={oid}")
                _event_once(cfg, lifecycle, "ENTRY_SENT", f"[AUTOEXEC] {symbol} ENTRY SENT", f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nEngine: {lifecycle.engine}\n\nEntry limit placed.\nQty: {lifecycle.qty}\nLimit: {lifecycle.desired_entry}\nStop (planned): {lifecycle.stop}\nTP0 (planned exit): {lifecycle.tp0}\nclientOrderId: {coid}\npreviewId: {pid}\norderId: {oid}\n")
                _record_activity(state, "ENTRY_PLACED", lifecycle, f"oid={oid} pid={pid}")
            except Exception as e:
                lifecycle.notes = f"entry_send_failed: {e}"
                lifecycle.last_entry_eval = f"FAILED: entry_send_failed ({e})"
                _event_once(cfg, lifecycle, "ENTRY_SEND_FAILED", f"[AUTOEXEC] {symbol} ENTRY SEND FAILED", f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nEngine: {lifecycle.engine}\n\nEntry placement failed: {e}\n")

            lst[idx] = asdict(lifecycle)


def _oid_int(order_id: Any) -> Optional[int]:
    """Convert stored order id (often a str) into an int for E*TRADE calls."""
    if order_id is None:
        return None
    if isinstance(order_id, int):
        return order_id
    try:
        s = str(order_id).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _reconcile_one(client: ETradeClient, account_id_key: str, state: Dict[str, Any], lifecycle: TradeLifecycle, cfg: AutoExecConfig, fetch_last_price_fn=None) -> None:
    """Update lifecycle state based on order statuses.

    Contract goals:
      - All broker calls use the ETradeClient wrapper signatures (keyword args, correct types)
      - Lifecycle does NOT transition to CLOSED unless we have evidence of an exit
      - Bookkeeping reserve is released exactly once on close/cancel
    """
    now = _now_et()
    def _oid_int_safe(x) -> Optional[int]:
        try:
            return int(x) if x not in (None, "", 0) else None
        except Exception:
            return None

    FINAL_STATUSES = {"EXECUTED", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"}

    def _order_is_inactive(order_id: int) -> bool:
        """Return True only when we have *positive* evidence an order is no longer active.

        IMPORTANT: We do **not** treat UNKNOWN as inactive.
        In E*TRADE (especially sandbox), ListOrders can intermittently omit an order (pagination,
        backend delays, or status buckets). If we treat UNKNOWN as inactive we can:
          - prematurely release pool reserve
          - stop monitoring a live order
          - mark a lifecycle CANCELED while the broker still has the order

        Therefore:
          - UNKNOWN => assume ACTIVE (return False)
          - Only final broker statuses => inactive
        """
        try:
            st, _fq = client.get_order_status_and_filled_qty(account_id_key, int(order_id))
        except Exception:
            return False
        st = str(st or "UNKNOWN").upper().strip()
        if st == "UNKNOWN":
            return False
        return st in FINAL_STATUSES

    def _cancel_with_verify(order_id_str: Optional[str], label: str) -> bool:
        """Best-effort cancel with verification via list_orders search.

        Returns True if order is confirmed inactive (or missing), else False.
        """
        oid = _oid_int_safe(order_id_str)
        if oid is None:
            return True
        try:
            client.cancel_order(account_id_key, oid)
        except Exception:
            # ignore; verification below will decide
            pass
        return _order_is_inactive(oid)

    # --- CANCEL_PENDING recovery ---
    if lifecycle.stage == "CANCEL_PENDING":
        # Attempt to cancel any remaining known orders and only finalize once broker confirms inactivity.
        pending_labels = [
            ("ENTRY", lifecycle.entry_order_id),
            ("STOP", lifecycle.stop_order_id),
            ("TP0", lifecycle.tp0_order_id),
            ("MKT_EXIT", lifecycle.market_exit_order_id),
        ]
        all_inactive = True
        for lbl, oid_str in pending_labels:
            if oid_str:
                ok = _cancel_with_verify(oid_str, lbl)
                if not ok:
                    all_inactive = False

        if all_inactive:
            # If this was an entry-side cancel, reserved dollars were still held; release now.
            if float(lifecycle.reserved_dollars or 0.0) > 0.0:
                _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                lifecycle.reserved_dollars = 0.0

            lifecycle.entry_order_id = None
            lifecycle.stop_order_id = None
            lifecycle.tp0_order_id = None
            lifecycle.market_exit_order_id = None
            lifecycle.bracket_qty = int(lifecycle.filled_qty or 0)

            # Preserve prior terminal intent in notes
            if "pending_close" in (lifecycle.notes or ""):
                lifecycle.stage = "CLOSED"
                # Best-effort realized P&L capture for pending-close lifecycles
                try:
                    m = re.search(r"pending_close_reason:([^|]+)", str(lifecycle.notes or ""))
                    close_reason = m.group(1).strip() if m else "PENDING_CLOSE"
                except Exception:
                    close_reason = "PENDING_CLOSE"
                try:
                    _record_realized_trade_on_close(state, lifecycle, client, account_id_key, close_reason)
                except Exception:
                    pass
            else:
                lifecycle.stage = "CANCELED"

            _event_once(
                cfg,
                lifecycle,
                "CANCEL_PENDING_RESOLVED",
                f"[AUTOEXEC] {lifecycle.symbol} CANCEL PENDING RESOLVED",
                f"""Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nResolved cancel-pending state. Stage now: {lifecycle.stage}\n""",
            )
        # Persist and return either way
        return

    # --- Bracket invariants (safe scope: we have client/account/lifecycle here) ---
    # If we're in position and somehow lost bracket orders, recreate them best-effort.
    if lifecycle.stage == "IN_POSITION" and int(lifecycle.filled_qty or 0) > 0:
        filled_qty = int(lifecycle.filled_qty)
        try:
            if not lifecycle.stop_order_id and lifecycle.stop is not None:
                scoid = _mk_client_order_id(lifecycle.lifecycle_id, 'ST')
                soid, spid = client.place_equity_stop_order_ex(
                    account_id_key=account_id_key,
                    symbol=lifecycle.symbol,
                    qty=filled_qty,
                    stop_price=float(_tick_round(lifecycle.stop) or lifecycle.stop),
                    action="SELL",
                    market_session="REGULAR",
                    client_order_id=scoid,
                )
                lifecycle.stop_order_id = str(soid)
                _append_note(lifecycle, f"STOP_PREVIEW_OK pid={spid}")
                _append_note(lifecycle, f"STOP_PLACE_OK oid={soid}")
                _append_note(lifecycle, "stop_recreated")
                _log(f"[AUTOEXEC][BROKER] place_ok sym={lifecycle.symbol} lc={lifecycle.lifecycle_id} coid={scoid} pid={spid} oid={soid} leg=STOP")
            if not lifecycle.tp0_order_id and lifecycle.tp0 is not None:
                tcoid = _mk_client_order_id(lifecycle.lifecycle_id, 'TP')
                toid, tpid = client.place_equity_limit_order_ex(
                    account_id_key=account_id_key,
                    symbol=lifecycle.symbol,
                    qty=filled_qty,
                    limit_price=float(_tick_round(lifecycle.tp0) or lifecycle.tp0),
                    action="SELL",
                    market_session="REGULAR",
                    client_order_id=tcoid,
                )
                lifecycle.tp0_order_id = str(toid)
                _append_note(lifecycle, f"TP0_PREVIEW_OK pid={tpid}")
                _append_note(lifecycle, f"TP0_PLACE_OK oid={toid}")
                _append_note(lifecycle, "tp0_recreated")
                _log(f"[AUTOEXEC][BROKER] place_ok sym={lifecycle.symbol} lc={lifecycle.lifecycle_id} coid={tcoid} pid={tpid} oid={toid} leg=TP0")
        except Exception as e:
            lifecycle.notes = (lifecycle.notes or "") + f" | bracket_invariant_error:{e}"

    # ---- ENTRY SENT: monitor fill / timeout ----
    if lifecycle.stage == "ENTRY_SENT" and lifecycle.entry_order_id:
        # Stop-breach safety: if price breaks the stop BEFORE entry fills, cancel the resting entry.
        # This prevents late fills on invalidated setups.
        if fetch_last_price_fn is not None and int(lifecycle.filled_qty or 0) == 0:
            try:
                last_px = _parse_float(fetch_last_price_fn(lifecycle.symbol))
            except Exception:
                last_px = None
            if last_px is not None and lifecycle.stop is not None and float(last_px) <= float(lifecycle.stop):
                ok_cancel = _cancel_with_verify(lifecycle.entry_order_id, "ENTRY_STOP_BREACH")
                if ok_cancel:
                    lifecycle.entry_order_id = None
                    lifecycle.stage = "CANCELED"
                    lifecycle.notes = "entry_canceled_stop_breach"
                    _event_once(
                        cfg,
                        lifecycle,
                        "ENTRY_STOP_BREACH_CANCEL",
                        f"[AUTOEXEC] {lifecycle.symbol} ENTRY CANCELED (STOP BREACH)",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Canceled: entry invalidated because last <= stop before fill.
Last: {last_px}
Stop: {lifecycle.stop}
""",
                    )
                    _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                    lifecycle.reserved_dollars = 0.0
                    return
                else:
                    lifecycle.stage = "CANCEL_PENDING"
                    lifecycle.notes = "pending_entry_cancel | stop_breach"
                    _event_once(
                        cfg,
                        lifecycle,
                        "ENTRY_STOP_BREACH_CANCEL_PENDING",
                        f"[AUTOEXEC] {lifecycle.symbol} ENTRY CANCEL PENDING (STOP BREACH)",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Cancel requested: entry invalidated because last <= stop before fill.
Last: {last_px}
Stop: {lifecycle.stop}
""",
                    )
                    return

        # Timeout
        try:
            sent = datetime.fromisoformat(lifecycle.entry_sent_ts) if lifecycle.entry_sent_ts else datetime.fromisoformat(lifecycle.created_ts)
            age_min = (now - sent).total_seconds() / 60.0
        except Exception:
            age_min = 0.0

        timeout_m = int(getattr(cfg, "timeout_minutes", ENTRY_TIMEOUT_MINUTES) or ENTRY_TIMEOUT_MINUTES)
        if age_min >= timeout_m:
            ok_cancel = _cancel_with_verify(lifecycle.entry_order_id, "ENTRY")
            if ok_cancel:
                lifecycle.entry_order_id = None
                lifecycle.stage = "CANCELED"
                lifecycle.notes = f"entry_timeout_{timeout_m}m"
            else:
                lifecycle.stage = "CANCEL_PENDING"
                lifecycle.notes = f"pending_entry_cancel | entry_timeout_{timeout_m}m"
            _event_once(
                cfg,
                lifecycle,
                "ENTRY_TIMEOUT",
                f"[AUTOEXEC] {lifecycle.symbol} ENTRY TIMEOUT",
                f"""Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nCanceled: entry order timeout ({timeout_m}m).\n""",
            )
            if lifecycle.stage == "CANCELED":
                _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                lifecycle.reserved_dollars = 0.0
            return

        oid_int = _oid_int(lifecycle.entry_order_id)
        if oid_int is None:
            lifecycle.notes = "entry_order_id_invalid"
            return
        status, filled_qty = client.get_order_status_and_filled_qty(account_id_key, oid_int)
        lifecycle.notes = f"entry_status={status}"
        prev_filled = int(lifecycle.filled_qty or 0)

        # If we got any fills, manage position (even if partial).
        if filled_qty and int(filled_qty) > 0:
            filled_qty = int(filled_qty)
            used = float(filled_qty) * float(lifecycle.desired_entry)
            unused = max(0.0, float(lifecycle.reserved_dollars or 0.0) - used)
            if unused > 0:
                _release_pool(state, unused)
                lifecycle.reserved_dollars = used

            lifecycle.filled_qty = filled_qty
            if filled_qty > prev_filled:
                _event_once(
                    cfg,
                    lifecycle,
                    f"FILL_{filled_qty}",
                    f"[AUTOEXEC] {lifecycle.symbol} FILL UPDATE",
                    f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Filled qty: {filled_qty} (prev {prev_filled})
Entry: {lifecycle.desired_entry}
OrderId: {lifecycle.entry_order_id}
""",
                )


            # AUTOPILOT SAFETY: If the entry order partially fills, cancel the remaining
            # entry quantity immediately. Otherwise, the remaining open limit could fill
            # later while we are already IN_POSITION, and we would not resize brackets
            # because we stop monitoring the entry order after transitioning stages.
            try:
                desired_qty = int(lifecycle.qty or 0)
            except Exception:
                desired_qty = 0
            if desired_qty > 0 and filled_qty < desired_qty:
                ok_rem = _cancel_with_verify(lifecycle.entry_order_id, "ENTRY")
                if ok_rem:
                    lifecycle.entry_order_id = None
                    lifecycle.notes = (lifecycle.notes or "") + " | entry_remainder_canceled"
                else:
                    lifecycle.notes = (lifecycle.notes or "") + " | entry_remainder_cancel_pending"
                    # Keep monitoring entry order until broker confirms remainder is inactive.
                    return
                _event_once(
                    cfg,
                    lifecycle,
                    f"ENTRY_PARTIAL_CANCEL_{filled_qty}",
                    f"[AUTOEXEC] {lifecycle.symbol} ENTRY PARTIAL — REMAINDER CANCELED",
                    f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Entry partially filled; canceling remainder for safety.
Desired qty: {desired_qty}
Filled qty: {filled_qty}
""",
                )
                _record_activity(state, "ENTRY_CANCELED", lifecycle, f"partial_remainder_cancel filled={filled_qty}")

            # Place / resize brackets
            if lifecycle.bracket_qty != filled_qty or not (lifecycle.stop_order_id and lifecycle.tp0_order_id):
                # Cancel old brackets before resizing
                for oid in [lifecycle.stop_order_id, lifecycle.tp0_order_id]:
                    oid_i = _oid_int(oid)
                    if oid_i is not None:
                        try:
                            client.cancel_order(account_id_key, oid_i)
                        except Exception:
                            pass

                try:
                    soid = client.place_equity_stop_order(
                        account_id_key=account_id_key,
                        symbol=lifecycle.symbol,
                        qty=filled_qty,
                        stop_price=float(_tick_round(lifecycle.stop) or lifecycle.stop),
                        action="SELL",
                        market_session="REGULAR",
                        client_order_id=_mk_client_order_id(lifecycle.lifecycle_id, 'ST'),
                    )
                    toid = client.place_equity_limit_order(
                        account_id_key=account_id_key,
                        symbol=lifecycle.symbol,
                        qty=filled_qty,
                        limit_price=float(_tick_round(lifecycle.tp0) or lifecycle.tp0),
                        action="SELL",
                        market_session="REGULAR",
                        client_order_id=_mk_client_order_id(lifecycle.lifecycle_id, 'TP'),
                    )
                    lifecycle.stop_order_id = str(soid)
                    lifecycle.tp0_order_id = str(toid)
                    lifecycle.bracket_qty = filled_qty
                    lifecycle.notes = f"bracket_sent stop={lifecycle.stop} tp0={lifecycle.tp0} qty={filled_qty}"
                    _event_once(
                        cfg,
                        lifecycle,
                        f"BRACKETS_{filled_qty}",
                        f"[AUTOEXEC] {lifecycle.symbol} BRACKETS PLACED",
                        f"""Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nBrackets placed for filled qty {filled_qty}.\nSTOP: {lifecycle.stop} (order {lifecycle.stop_order_id})\nTP0: {lifecycle.tp0} (order {lifecycle.tp0_order_id})\n""",
                    )
                    _record_activity(state, "BRACKETS_SENT", lifecycle, f"stop_oid={lifecycle.stop_order_id} tp_oid={lifecycle.tp0_order_id} qty={filled_qty}")
                except Exception as e:
                    lifecycle.notes = (lifecycle.notes or "") + f" | bracket_send_failed:{e}"

            lifecycle.stage = "IN_POSITION"
            return

        # No fills and order ended
        if str(status).upper() in {"CANCELLED", "REJECTED", "EXPIRED"}:
            lifecycle.stage = "CANCELED"
            lifecycle.notes = f"entry_{str(status).lower()}"
            _event_once(
                cfg,
                lifecycle,
                f"ENTRY_{status}",
                f"[AUTOEXEC] {lifecycle.symbol} ENTRY {status}",
                f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Entry order ended with status: {status}.
""",
            )
            _record_activity(state, "ENTRY_CANCELED", lifecycle, f"status={status}")
            _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
            lifecycle.reserved_dollars = 0.0
            return

    # ---- IN POSITION: monitor exits ----
    if lifecycle.stage == "IN_POSITION":
        bracket_qty = int(lifecycle.bracket_qty or lifecycle.filled_qty or 0)

        # Helper to close + release
        def _close(reason: str) -> None:
            # --- OCO / bracket safety (best-effort) ---
            # If TP executes, cancel the STOP sibling. If STOP executes, cancel the TP sibling.
            def _cancel_order_best_effort(order_id_str: Optional[str], label: str) -> bool:
                oid = _oid_int(order_id_str)
                if oid is None:
                    return True
                try:
                    client.cancel_order(account_id_key, oid)
                except Exception as e:
                    # log but still verify below
                    _event_once(
                        cfg,
                        lifecycle,
                        f"OCO_CANCEL_ERR_{label}_{oid}",
                        f"[AUTOEXEC] {lifecycle.symbol} OCO cancel error ({label})",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

OCO cleanup cancel attempt had error for {label} order_id={oid}
Close reason: {reason}
Error: {repr(e)}
""",
                    )

                inactive = _order_is_inactive(int(oid))
                if inactive:
                    _event_once(
                        cfg,
                        lifecycle,
                        f"OCO_CANCEL_{label}_{oid}",
                        f"[AUTOEXEC] {lifecycle.symbol} OCO cancel {label}",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

OCO cleanup: confirmed inactive {label} order_id={oid}
Close reason: {reason}
""",
                    )
                else:
                    _event_once(
                        cfg,
                        lifecycle,
                        f"OCO_CANCEL_PENDING_{label}_{oid}",
                        f"[AUTOEXEC] {lifecycle.symbol} OCO cancel pending ({label})",
                        f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

OCO cleanup: {label} order_id={oid} still appears ACTIVE after cancel attempt.
Close reason: {reason}
""",
                    )
                return inactive

            pending_cleanup = False
            if reason == "TP0_EXECUTED":
                ok = _cancel_order_best_effort(lifecycle.stop_order_id, "STOP")
                if not ok:
                    pending_cleanup = True
            elif reason == "STOP_EXECUTED":
                ok = _cancel_order_best_effort(lifecycle.tp0_order_id, "TP0")
                if not ok:
                    pending_cleanup = True

            # --- Bookkeeping: release any remaining reserved capital exactly once ---
            if float(lifecycle.reserved_dollars or 0.0) > 0.0:
                _release_pool(state, float(lifecycle.reserved_dollars or 0.0))
                lifecycle.reserved_dollars = 0.0

            if pending_cleanup:
                lifecycle.stage = "CANCEL_PENDING"
                lifecycle.notes = (lifecycle.notes or "") + f" | pending_close_reason:{reason} | pending_close"
                return

            # Record realized P&L for reporting (best-effort)
            try:
                _record_realized_trade_on_close(state, lifecycle, client, account_id_key, reason)
            except Exception:
                pass

            # Clear broker order ids once closed (prevents orphan state)
            lifecycle.entry_order_id = None
            lifecycle.stop_order_id = None
            lifecycle.tp0_order_id = None
            lifecycle.market_exit_order_id = None

            lifecycle.stage = "CLOSED"
            lifecycle.notes = reason
            _event_once(
                cfg,
                lifecycle,
                f"CLOSE_{reason}",
                f"[AUTOEXEC] {lifecycle.symbol} CLOSED",
                f"""Time (ET): {now.isoformat()}
Symbol: {lifecycle.symbol}
Engine: {lifecycle.engine}

Closed: {reason}
""",
            )
            _record_activity(state, "EXIT_EXECUTED", lifecycle, str(reason))
        # Check STOP (protective)
        if lifecycle.stop_order_id:
            soid = _oid_int(lifecycle.stop_order_id)
            if soid is not None:
                try:
                    s_status, s_filled = client.get_order_status_and_filled_qty(account_id_key, soid)
                except Exception:
                    s_status, s_filled = ("UNKNOWN", 0)

                # Partial STOP fill: enforce full exit (mirror TP0 partial-flatten behavior)
                # If the STOP order has filled some but not all shares, we cancel remaining STOP + TP0 and
                # market-sell any remaining position to eliminate orphan-risk.
                if (
                    bracket_qty > 0
                    and s_filled
                    and 0 < int(s_filled) < bracket_qty
                    and str(s_status).upper() not in {"CANCELLED", "REJECTED", "EXPIRED"}
                ):
                    # cancel remaining STOP and TP0, market-sell remainder
                    try:
                        client.cancel_order(account_id_key, soid)
                    except Exception:
                        pass
                    lifecycle.stop_order_id = None

                    if lifecycle.tp0_order_id:
                        toid = _oid_int(lifecycle.tp0_order_id)
                        if toid is not None:
                            try:
                                client.cancel_order(account_id_key, toid)
                            except Exception:
                                pass
                        lifecycle.tp0_order_id = None

                    try:
                        positions = client.get_positions_map(account_id_key)
                        rem = int(positions.get(lifecycle.symbol, 0) or 0)
                    except Exception:
                        rem = 0

                    if rem > 0:
                        try:
                            mcoid = _mk_client_order_id(lifecycle.lifecycle_id, 'MK')
                            mid, mpid = client.place_equity_market_order_ex(
                                account_id_key=account_id_key,
                                symbol=lifecycle.symbol,
                                qty=rem,
                                action="SELL",
                                market_session="REGULAR",
                                client_order_id=mcoid,
                            )
                            lifecycle.market_exit_order_id = str(mid)
                            _append_note(lifecycle, f"MK_PREVIEW_OK pid={mpid}")
                            _append_note(lifecycle, f"MK_PLACE_OK oid={mid}")
                            _log(f"[AUTOEXEC][BROKER] place_ok sym={lifecycle.symbol} lc={lifecycle.lifecycle_id} coid={mcoid} pid={mpid} oid={mid} leg=MK")
                        except Exception:
                            pass

                    _close("STOP_PARTIAL_FLATTEN")
                    return
                if str(s_status).upper() in {"EXECUTED", "FILLED"} or (s_filled and int(s_filled) >= bracket_qty and bracket_qty > 0):
                    _close("STOP_EXECUTED")
                    return

        # Check TP0
        if lifecycle.tp0_order_id:
            toid = _oid_int(lifecycle.tp0_order_id)
            if toid is not None:
                try:
                    t_status, t_filled = client.get_order_status_and_filled_qty(account_id_key, toid)
                except Exception:
                    t_status, t_filled = ("UNKNOWN", 0)

                # Partial TP fill: enforce full exit
                if t_filled and 0 < int(t_filled) < bracket_qty and str(t_status).upper() not in {"EXECUTED", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
                    # cancel remaining TP and stop, market-sell remainder
                    try:
                        client.cancel_order(account_id_key, toid)
                    except Exception:
                        pass
                    lifecycle.tp0_order_id = None
                    if lifecycle.stop_order_id:
                        soid = _oid_int(lifecycle.stop_order_id)
                        if soid is not None:
                            try:
                                client.cancel_order(account_id_key, soid)
                            except Exception:
                                pass
                        lifecycle.stop_order_id = None
                    try:
                        positions = client.get_positions_map(account_id_key)
                        rem = int(positions.get(lifecycle.symbol, 0) or 0)
                    except Exception:
                        rem = 0
                    if rem > 0:
                        try:
                            mcoid = _mk_client_order_id(lifecycle.lifecycle_id, 'MK')
                            mid, mpid = client.place_equity_market_order_ex(
                                account_id_key=account_id_key,
                                symbol=lifecycle.symbol,
                                qty=rem,
                                action="SELL",
                                market_session="REGULAR",
                                client_order_id=mcoid,
                            )
                            lifecycle.market_exit_order_id = str(mid)
                            _append_note(lifecycle, f"MK_PREVIEW_OK pid={mpid}")
                            _append_note(lifecycle, f"MK_PLACE_OK oid={mid}")
                            _log(f"[AUTOEXEC][BROKER] place_ok sym={lifecycle.symbol} lc={lifecycle.lifecycle_id} coid={mcoid} pid={mpid} oid={mid} leg=MK")
                        except Exception:
                            pass
                    _close("TP0_PARTIAL_FLATTEN")
                    return

                if str(t_status).upper() in {"EXECUTED", "FILLED"} or (t_filled and int(t_filled) >= bracket_qty and bracket_qty > 0):
                    _close("TP0_EXECUTED")
                    return

        # Otherwise remain in position
        return


def _force_liquidate_all(client: ETradeClient, account_id_key: str, cfg: AutoExecConfig, state: Dict[str, Any]) -> None:
    """Hard liquidation by 15:55 ET.

    Actions:
      - Cancel any open entry/stop/tp0 orders for managed lifecycles
      - Market-sell any remaining positions for managed symbols
      - Mark lifecycles CLOSED and release reserved pool dollars
    """
    now = _now_et()

    managed_syms = set(state.get("lifecycles", {}).keys())

    # Cancel orders + close lifecycles
    for symbol, lst in list(state.get("lifecycles", {}).items()):
        new_lst = []
        for raw in list(lst):
            lc = lifecycle_from_raw(raw)
            if lc.stage in {"CLOSED", "CANCELED"}:
                new_lst.append(asdict(lc))
                continue

            for oid in [lc.entry_order_id, lc.stop_order_id, lc.tp0_order_id]:
                oid_i = _oid_int(oid)
                if oid_i is not None:
                    try:
                        client.cancel_order(account_id_key, oid_i)
                    except Exception:
                        pass

            # Release any reserved capital
            if float(lc.reserved_dollars or 0.0) > 0.0:
                _release_pool(state, float(lc.reserved_dollars or 0.0))
                lc.reserved_dollars = 0.0

            lc.stage = "CLOSED"
            lc.notes = "EOD_LIQUIDATION"
            _event_once(
                cfg,
                lc,
                "EOD_LIQUIDATION",
                f"[AUTOEXEC] {lc.symbol} EOD LIQUIDATION",
                f"""Time (ET): {now.isoformat()}\nSymbol: {lc.symbol}\nEngine: {lc.engine}\n\nEOD liquidation: canceled open orders; flattening remaining position if any.\n""",
            )
            new_lst.append(asdict(lc))

        state.setdefault("lifecycles", {})[symbol] = new_lst

    # Market sell remaining positions for managed symbols
    try:
        positions = client.get_positions_map(account_id_key)
    except Exception:
        positions = {}

    for sym, qty in (positions or {}).items():
        if sym not in managed_syms:
            continue
        try:
            q = int(qty or 0)
        except Exception:
            q = 0
        if q <= 0:
            continue
        try:
            mcoid = _mk_client_order_id(f'FLAT{sym}{now.strftime("%H%M")}', 'MK')
            mid, mpid = client.place_equity_market_order_ex(
                account_id_key=account_id_key,
                symbol=sym,
                qty=q,
                action="SELL",
                market_session="REGULAR",
                client_order_id=mcoid,
            )
            _log(f"[AUTOEXEC][BROKER] force_flat_ok sym={sym} coid={mcoid} pid={mpid} oid={mid}")
        except Exception:
            pass

    # Persist state
    st.session_state["autoexec"] = state
