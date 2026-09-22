"""9Router TTS backend for Hermes Agent.

Two paths:
  - Elevenlabs: direct API call with per-key voice_id mapping + round-robin.
    Keys read from 9Router DB (providerConnections table). Voice mapping from config.
  - Everything else (Gemini, OpenAI, Edge, etc.): proxy to 9Router /v1/audio/speech.

Config (``tts.9router-tts`` in config.yaml):
    base_url          — 9Router gateway URL (default: NINEROUTER_URL or http://localhost:20128)
    model             — default model id (determines active provider)
    fallback_models   — tried in order on 429/quota (same provider only)
    timeout           — per-request seconds (default 120)
    voice             — default voice alias
    db_path           — 9Router SQLite DB (default: ~/.9router/db/data.sqlite)
    elevenlabs:
      model_id        — ElevenLabs model (default: eleven_flash_v2_5)
      voices:
        <alias>:
          - connection: <connection_name_in_9router>
            voice_id: <elevenlabs_voice_id>
          - connection: <another_name>
            voice_id: <another_voice_id>

Secret: NINEROUTER_KEY (.env) for 9Router proxy path. ElevenLabs keys come from 9Router DB.
"""

from __future__ import annotations

import itertools
import json
import logging
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:20128"
DEFAULT_TIMEOUT = 120
DEFAULT_EL_MODEL = "eleven_flash_v2_5"
DEFAULT_EL_API = "https://api.elevenlabs.io"
_MODELS_CACHE_TTL = 300.0
_DB_KEY_CACHE_TTL = 600.0


def _load_settings() -> Dict[str, Any]:
    try:
        from hermes_cli.config import load_config
    except ImportError:
        return {}
    cfg = load_config() or {}
    scoped = cfg.get("tts") if isinstance(cfg, dict) else None
    scoped = scoped.get("9router-tts") if isinstance(scoped, dict) else None
    return scoped if isinstance(scoped, dict) else {}


def _base_url(settings: Dict[str, Any]) -> str:
    url = (
        str(settings.get("base_url") or "").strip()
        or os.environ.get("NINEROUTER_URL", "").strip()
        or DEFAULT_BASE_URL
    )
    return url.rstrip("/")


def _api_key() -> str:
    try:
        from agent.secret_scope import get_secret
        key = get_secret("NINEROUTER_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("NINEROUTER_KEY", "").strip()


def _db_path(settings: Dict[str, Any]) -> str:
    explicit = str(settings.get("db_path") or "").strip()
    if explicit:
        return explicit
    return str(Path.home() / ".9router" / "db" / "data.sqlite")


def _is_quota_error(status_code: int, body: Any) -> bool:
    if status_code == 429:
        return True
    if not isinstance(body, dict):
        return False
    err = body.get("error")
    if isinstance(err, dict):
        msg = str(err.get("message") or "").lower()
        code = str(err.get("code") or "").lower()
        return ("429" in msg or "quota" in msg or "rate" in msg and "limit" in msg
                or code in ("429", "rate_limit_exceeded", "insufficient_quota"))
    return False


def _is_elevenlabs_model(model: Optional[str]) -> bool:
    """Check if model string indicates ElevenLabs provider."""
    if not model:
        return False
    m = model.lower().strip()
    return (m.startswith("eleven") or m.startswith("el/")
            or "elevenlabs" in m or "/eleven" in m)


class NineRouterTTSProvider(TTSProvider):
    """TTS through 9Router (general) or ElevenLabs direct (multi-key rotation)."""

    def __init__(self) -> None:
        self._models_cache: Optional[Tuple[List[str], float]] = None
        # State per voice alias for load balancing
        self._el_states: Dict[str, Dict[str, Any]] = {}
        # Cached DB keys: {connection_name: api_key}
        self._db_keys_cache: Optional[Tuple[Dict[str, str], float]] = None

    @property
    def name(self) -> str:
        return "9router-tts"

    @property
    def display_name(self) -> str:
        return "9Router TTS"

    @property
    def voice_compatible(self) -> bool:
        return True

    def is_available(self) -> bool:
        try:
            import requests  # noqa: F401
            return True
        except ImportError:
            return False

    def default_voice(self) -> Optional[str]:
        settings = _load_settings()
        return str(settings.get("voice") or settings.get("default_voice") or "").strip() or None

    def default_model(self) -> Optional[str]:
        settings = _load_settings()
        model = str(settings.get("model") or "").strip()
        if model:
            return model
        models = self._fetch_models()
        return models[0] if models else None

    def list_voices(self) -> List[Dict[str, Any]]:
        settings = _load_settings()
        el_config = settings.get("elevenlabs") or {}
        voices_map = el_config.get("voices") or {}
        result = []
        for alias, entries in voices_map.items():
            if isinstance(entries, list) and entries:
                result.append({
                    "id": alias,
                    "display": alias,
                    "provider": "elevenlabs",
                    "entries": len(entries),
                })
        return result

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "9Router TTS",
            "badge": "",
            "tag": "TTS via 9Router + ElevenLabs direct multi-key rotation",
            "env_vars": [
                {"key": "NINEROUTER_KEY", "prompt": "9Router API key (skip for auth-free local gateway)"},
            ],
        }

    # ── Model catalog (9Router) ──────────────────────────────────────────

    def _fetch_models(self) -> List[str]:
        if self._models_cache and time.monotonic() - self._models_cache[1] < _MODELS_CACHE_TTL:
            return self._models_cache[0]
        settings = _load_settings()
        base = _base_url(settings)
        headers = {"Authorization": f"Bearer {_api_key()}"} if _api_key() else {}
        try:
            import requests
            resp = requests.get(f"{base}/v1/models/tts", headers=headers, timeout=15)
            body = resp.json()
            models = [
                str(e.get("id")) for e in (body.get("data") or [])
                if isinstance(e, dict) and e.get("id")
            ]
            self._models_cache = (models, time.monotonic())
            return models
        except Exception as exc:
            logger.debug("9Router TTS model catalog unavailable: %s", exc)
            return []

    def _model_chain(self, model_arg: Optional[str]) -> List[str]:
        settings = _load_settings()
        chain: List[str] = []
        for candidate in (
            model_arg,
            str(settings.get("model") or "").strip(),
        ):
            if candidate and candidate not in chain:
                chain.append(candidate)
        raw = settings.get("fallback_models")
        if isinstance(raw, str):
            raw = [raw]
        for candidate in raw or []:
            candidate = str(candidate).strip()
            if candidate and candidate not in chain:
                chain.append(candidate)
        if not chain:
            models = self._fetch_models()
            if models:
                chain.append(models[0])
        return chain

    # ── 9Router DB key reader ────────────────────────────────────────────

    def _read_db_keys(self, provider: str = "elevenlabs") -> Dict[str, str]:
        """Read API keys from 9Router providerConnections DB. Returns {connection_name: api_key}."""
        now = time.monotonic()
        if self._db_keys_cache and now - self._db_keys_cache[1] < _DB_KEY_CACHE_TTL:
            return self._db_keys_cache[0]
        settings = _load_settings()
        db = _db_path(settings)
        keys: Dict[str, str] = {}
        try:
            conn = sqlite3.connect(db)
            rows = conn.execute(
                "SELECT name, data FROM providerConnections WHERE provider = ? AND isActive = 1",
                (provider,),
            ).fetchall()
            conn.close()
            for name, data_str in rows:
                try:
                    data = json.loads(data_str)
                    api_key = data.get("apiKey")
                    if api_key and name:
                        keys[name] = api_key
                except (json.JSONDecodeError, TypeError):
                    continue
        except Exception as exc:
            logger.warning("Failed to read 9Router DB at %s: %s", db, exc)
        self._db_keys_cache = (keys, now)
        return keys

    # ── ElevenLabs direct path ───────────────────────────────────────────

    def _resolve_el_entries(self, voice: Optional[str]) -> List[Dict[str, str]]:
        """Resolve voice alias to list of {connection, voice_id} entries."""
        settings = _load_settings()
        el_config = settings.get("elevenlabs") or {}
        voices_map = el_config.get("voices") or {}

        # If voice matches an alias, return its entries
        if voice and voice in voices_map:
            entries = voices_map[voice]
            if isinstance(entries, list):
                return entries

        # If voice looks like a raw voice_id, wrap it for all connections
        if voice:
            db_keys = self._read_db_keys()
            return [{"connection": name, "voice_id": voice} for name in db_keys]

        # Fallback: use default_voice alias
        default = self.default_voice()
        if default and default in voices_map:
            entries = voices_map[default]
            if isinstance(entries, list):
                return entries

        return []

    def _el_pick_entry(self, voice_alias: str, entries: List[Dict[str, str]]) -> Dict[str, str]:
        """Pick next entry based on strategy (fill-first or round-robin)."""
        settings = _load_settings()
        el_config = settings.get("elevenlabs") or {}
        strategy = str(el_config.get("strategy", "fill-first")).lower().strip()

        if voice_alias not in self._el_states:
            self._el_states[voice_alias] = {"index": 0, "robin": itertools.cycle(entries)}

        state = self._el_states[voice_alias]

        if strategy == "round-robin":
            return next(state["robin"])
        else:
            # fill-first: stick to current index until it fails
            idx = state.get("index", 0) % len(entries)
            return entries[idx]

    def _el_advance(self, voice_alias: str, entries: List[Dict[str, str]]) -> None:
        """Advance fill-first to next entry (called on failure)."""
        if voice_alias not in self._el_states:
            return
        state = self._el_states[voice_alias]
        state["index"] = (state.get("index", 0) + 1) % len(entries)

    def _synthesize_elevenlabs(
        self, text: str, output_path: str, voice: Optional[str],
        model_id: Optional[str], fmt: str,
    ) -> str:
        import requests

        settings = _load_settings()
        el_config = settings.get("elevenlabs") or {}
        timeout = int(settings.get("timeout") or DEFAULT_TIMEOUT)

        # Resolve model_id: strip el/ prefix, extract model part if model/voice format
        if model_id:
            m = model_id.strip()
            if m.startswith("el/"):
                m = m[3:]
            # If format is model_id/voice_id, split
            if "/" in m:
                parts = m.split("/", 1)
                m = parts[0]
                # If no explicit voice, use voice from model string
                if not voice:
                    voice = parts[1]
            model_id = m
        if not model_id or not model_id.startswith("eleven"):
            model_id = el_config.get("model_id", DEFAULT_EL_MODEL)

        entries = self._resolve_el_entries(voice)
        if not entries:
            raise ValueError(
                f"No ElevenLabs voice entries configured for voice={voice!r}. "
                "Add entries under tts.9router-tts.elevenlabs.voices in config.yaml"
            )

        db_keys = self._read_db_keys()
        alias = voice or "default"
        errors = []

        # Try up to len(entries) attempts
        for _ in range(len(entries)):
            entry = self._el_pick_entry(alias, entries)
            conn_name = entry.get("connection", "")
            voice_id = entry.get("voice_id", "")
            api_key = db_keys.get(conn_name)

            if not api_key:
                logger.warning("No API key found for ElevenLabs connection %r in 9Router DB", conn_name)
                errors.append(f"{conn_name}: no API key in DB")
                self._el_advance(alias, entries)
                continue
            if not voice_id:
                errors.append(f"{conn_name}: no voice_id")
                self._el_advance(alias, entries)
                continue

            # Determine output format
            el_format = "mp3_44100_128"
            if output_path.endswith(".ogg") or fmt == "ogg":
                el_format = "opus_48000_64"
            elif output_path.endswith(".wav") or fmt == "wav":
                el_format = "pcm_44100"

            url = f"{DEFAULT_EL_API}/v1/text-to-speech/{voice_id}"
            headers = {
                "xi-api-key": api_key,
                "Content-Type": "application/json",
            }
            payload = {
                "text": text,
                "model_id": model_id,
                "output_format": el_format,
            }

            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
                if resp.status_code == 200:
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                    logger.info("ElevenLabs TTS OK via %s (voice_id=%s)", conn_name, voice_id)
                    return output_path

                # Parse error
                try:
                    body = resp.json()
                except ValueError:
                    body = {}

                if _is_quota_error(resp.status_code, body):
                    logger.warning("ElevenLabs %s hit quota/rate limit, rotating", conn_name)
                    errors.append(f"{conn_name}: {resp.status_code} quota/rate")
                    self._el_advance(alias, entries)
                    continue

                err_detail = body.get("detail", {})
                err_msg = err_detail.get("message", "") if isinstance(err_detail, dict) else str(err_detail)
                errors.append(f"{conn_name}: HTTP {resp.status_code} {err_msg}")
                self._el_advance(alias, entries)
                # Non-quota error on specific key: try next
                continue

            except requests.exceptions.Timeout:
                errors.append(f"{conn_name}: timeout")
                self._el_advance(alias, entries)
                continue
            except Exception as exc:
                errors.append(f"{conn_name}: {exc}")
                self._el_advance(alias, entries)
                continue

        raise RuntimeError(f"All ElevenLabs keys failed for voice={voice!r}: {'; '.join(errors)}")

    # ── 9Router proxy path ───────────────────────────────────────────────

    def _synthesize_9router(
        self, text: str, output_path: str, voice: Optional[str],
        model: Optional[str], fmt: str,
    ) -> str:
        import requests

        settings = _load_settings()
        base = _base_url(settings)
        timeout = int(settings.get("timeout") or DEFAULT_TIMEOUT)
        chain = self._model_chain(model)

        if not chain:
            raise ValueError("No TTS model available on 9Router (catalog empty and no model configured)")

        headers = {"Content-Type": "application/json"}
        api_key = _api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Build model string: if voice provided and model doesn't already contain it
        last_error = ""
        for model_id in chain:
            # Skip elevenlabs models in 9Router path
            if _is_elevenlabs_model(model_id):
                continue

            request_model = model_id
            if voice and "/" not in model_id:
                request_model = f"{model_id}/{voice}"

            payload = {"model": request_model, "input": text}
            try:
                resp = requests.post(
                    f"{base}/v1/audio/speech",
                    headers=headers, json=payload, timeout=timeout,
                )
                if resp.status_code == 200 and resp.content:
                    with open(output_path, "wb") as f:
                        f.write(resp.content)
                    logger.info("9Router TTS OK model=%s", request_model)
                    return output_path

                try:
                    body = resp.json()
                except ValueError:
                    body = {}

                if _is_quota_error(resp.status_code, body):
                    logger.warning("9Router model %s hit 429/quota; trying next", model_id)
                    last_error = f"{model_id}: {resp.status_code}"
                    continue

                err = body.get("error", {})
                last_error = str(err.get("message", "")) if isinstance(err, dict) else f"HTTP {resp.status_code}"
                continue

            except Exception as exc:
                last_error = f"{model_id}: {exc}"
                continue

        raise RuntimeError(f"9Router TTS failed: {last_error}")

    # ── Main synthesize ──────────────────────────────────────────────────

    def synthesize(
        self, text: str, output_path: str, *, voice: Optional[str] = None,
        model: Optional[str] = None, speed: Optional[float] = None,
        format: str = "mp3", **extra: Any,
    ) -> str:
        # Determine which path based on model string
        effective_model = model or self.default_model() or ""

        if _is_elevenlabs_model(effective_model):
            return self._synthesize_elevenlabs(text, output_path, voice, effective_model, format)
        else:
            return self._synthesize_9router(text, output_path, voice, model, format)

    def stream(
        self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None,
        format: str = "opus", **extra: Any,
    ) -> Iterator[bytes]:
        # ponytail: streaming ElevenLabs; add when needed
        raise NotImplementedError(
            f"TTS provider {self.name!r} does not implement streaming. "
            "Use synthesize() instead."
        )


def register(ctx) -> None:  # pragma: no cover
    ctx.register_tts_provider(NineRouterTTSProvider())
