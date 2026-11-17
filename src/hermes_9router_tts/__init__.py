"""Native Hermes TTS via 9Router, with optional local ElevenLabs key rotation.

No I/O at import or registration. Settings are read for each call from the active
Hermes profile. Gateway is the default; direct ElevenLabs is explicit or legacy-opted-in.
"""
from __future__ import annotations

import base64
from datetime import datetime
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import sqlite3
import tempfile
import threading
import time
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import quote, urlsplit

from agent.tts_provider import TTSProvider

logger = logging.getLogger(__name__)
DEFAULT_BASE_URL = "http://localhost:20128"
DEFAULT_TIMEOUT = 120
DEFAULT_EL_MODEL = "eleven_flash_v2_5"
DEFAULT_EL_API = "https://api.elevenlabs.io"
_MODELS_CACHE_TTL = 300.0
_FORMATS = {"mp3": "mp3", "wav": "wav", "ogg": "ogg", "opus": "opus", "flac": "flac"}


def _load_settings() -> Dict[str, Any]:
    from hermes_cli.config import load_config

    cfg = load_config() or {}
    tts = cfg.get("tts") if isinstance(cfg, dict) else None
    if not isinstance(tts, dict):
        return {}
    legacy, canonical = tts.get("9router"), tts.get("9router-tts")
    result = dict(legacy) if isinstance(legacy, dict) else {}
    if isinstance(canonical, dict):
        result.update(canonical)
    return result


def _scoped_env(name: str) -> str:
    try:
        from agent.secret_scope import get_secret
    except ImportError:
        return os.environ.get(name, "").strip()
    return str(get_secret(name) or "").strip()


def _api_key() -> str:
    return _scoped_env("NINEROUTER_KEY")


def _base_url(settings: Dict[str, Any]) -> str:
    value = (str(settings.get("base_url") or "").strip()
             or _scoped_env("NINEROUTER_URL") or DEFAULT_BASE_URL).rstrip("/")
    parts = urlsplit(value)
    if (parts.scheme not in ("http", "https") or not parts.hostname
            or parts.username is not None or parts.password is not None
            or parts.query or parts.fragment):
        raise ValueError("base_url must be an HTTP(S) URL without credentials, query or fragment")
    return value


def _profile() -> str:
    from hermes_constants import get_hermes_home
    return str(get_hermes_home().resolve())


def _db_path(settings: Dict[str, Any]) -> Path:
    # Direct synthesis always uses the fixed official ElevenLabs origin.
    # The gateway URL is independent; a routed profile must opt into its DB.
    explicit = str(settings.get("db_path") or "").strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    from agent.secret_scope import serves_routed_profile
    from hermes_constants import get_hermes_home, named_profile_home
    if serves_routed_profile() or named_profile_home(get_hermes_home()) is not None:
        raise ValueError("Set this profile's explicit db_path for local ElevenLabs access")
    return Path.home() / ".9router" / "db" / "data.sqlite"


def _session():
    import requests
    session = requests.Session()
    # Do not send speech or keys through ambient proxies or replace auth with netrc.
    session.trust_env = False
    return session


def _timeout(settings: Dict[str, Any]) -> float:
    value = float(settings.get("timeout", DEFAULT_TIMEOUT))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("timeout must be positive and finite")
    return value


def _is_quota_error(status: int, body: Any) -> bool:
    if status == 429:
        return True
    if not isinstance(body, dict):
        return False
    err = body.get("error") or body.get("detail")
    if not isinstance(err, dict):
        return False
    message = str(err.get("message") or "").lower()
    code = str(err.get("code") or err.get("status") or "").lower()
    # ElevenLabs reports quota_exceeded on 400/401 as well as rate-limit 429.
    if status in (400, 401):
        return code in ("quota_exceeded", "rate_limit_exceeded", "insufficient_quota")
    if status not in (402, 403, 500, 502, 503):
        return False
    if "not allowed to use model" in message or "do not have permission" in message:
        return False
    if status == 403:
        return "quota" in message or "rate limit" in message
    return ("quota" in message or "rate limit" in message
            or code in ("quota_exceeded", "rate_limit_exceeded", "insufficient_quota", "429"))


def _provider(model: str) -> str:
    prefix = model.split("/", 1)[0].lower()
    return "elevenlabs" if prefix == "el" or prefix.startswith("eleven_") else prefix


def _is_elevenlabs_model(model: Optional[str]) -> bool:
    if not model:
        return False
    return _provider(model) == "elevenlabs" or model.split("/", 1)[0].lower().startswith("eleven_")


def _legacy_direct(settings: Dict[str, Any], model: str) -> bool:
    """Recognize local-rotation intent, never a model/voice alone."""
    if not _is_elevenlabs_model(model):
        return False
    if isinstance(settings.get("db_path"), str) and settings["db_path"].strip():
        return True
    el = settings.get("elevenlabs")
    if not isinstance(el, dict):
        return False
    if any(isinstance(el.get(key), str) and el[key].strip()
           for key in ("strategy", "model_id")):
        return True
    voices = el.get("voices")
    if not isinstance(voices, dict):
        return False
    return any(
        isinstance(entry, dict)
        and all(isinstance(entry.get(key), str) and entry[key].strip()
                for key in ("connection", "voice_id"))
        for entries in voices.values() if isinstance(entries, list)
        for entry in entries
    )


def _transport(settings: Dict[str, Any], model: Optional[str] = None) -> str:
    if "transport" in settings:
        value = settings["transport"]
        if not isinstance(value, str) or value.strip() not in ("gateway", "elevenlabs-direct"):
            raise ValueError("transport must be gateway or elevenlabs-direct")
        selected = value.strip()
        if selected == "elevenlabs-direct" and not _is_elevenlabs_model(model):
            raise ValueError("elevenlabs-direct requires an explicit ElevenLabs model")
        return selected
    return "elevenlabs-direct" if _legacy_direct(settings, model or "") else "gateway"


def _encoded_voice(model: str) -> Optional[str]:
    parts = model.split("/")
    provider = _provider(model)
    if provider == "elevenlabs":
        if parts[0].lower() in ("el", "elevenlabs"):
            remainder = parts[1:]
        else:
            remainder = parts
        if len(remainder) == 2:
            return remainder[1]
        if len(remainder) == 1 and not remainder[0].startswith("eleven_"):
            return remainder[0] or None
        return None
    if provider in ("gemini", "openai", "selfhosted-tts", "xiaomi-mimo", "mimo", "mimo-desktop", "xmd") and len(parts) > 2:
        return parts[-1]
    if provider == "openrouter" and len(parts) > 3:
        return parts[-1]
    if provider in ("edge", "edge-tts", "google", "google-tts", "local-device") and len(parts) > 1:
        return parts[-1]
    return None


def _request_model(model: str, voice: Optional[str]) -> str:
    if _is_elevenlabs_model(model):
        parts = model.split("/")
        remainder = parts[1:] if parts[0].lower() in ("el", "elevenlabs") else parts
        if not remainder or len(remainder) > 2 or any(not part for part in remainder):
            raise ValueError("Use elevenlabs/voice_id or elevenlabs/model_id/voice_id")
        voice = voice or _encoded_voice(model)
        if not voice:
            raise ValueError("An ElevenLabs model requires a voice_id for gateway synthesis")
        # A one-part non-model remainder is a voice-only shorthand.
        if len(remainder) == 1 and not remainder[0].startswith("eleven_"):
            return "elevenlabs/" + voice
        return "elevenlabs/" + remainder[0] + "/" + voice
    if not voice:
        return model
    # 9Router 0.5.119 consumes the voice inside model, not body.voice.
    provider, sep, remainder = model.partition("/")
    canonical = _provider(model)
    if canonical in ("gemini", "openai", "openrouter", "selfhosted-tts", "xiaomi-mimo", "mimo", "mimo-desktop", "xmd"):
        if not sep:
            raise ValueError("Use provider/model when selecting a voice")
        if canonical == "openrouter":
            segments = remainder.split("/")
            # OpenRouter models are themselves namespaced (e.g. openai/model).
            model_part = "/".join(segments[:2])
        else:
            model_part = remainder.split("/", 1)[0]
        return provider + "/" + model_part + "/" + voice
    if canonical in ("edge", "edge-tts", "google", "google-tts", "local-device"):
        return provider + "/" + voice
    # Other adapters have their own model grammar; keep it intact.
    return model


def _audio_response(response, requested_format):
    content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
    if content_type == "application/json":
        try:
            body = response.json()
            return base64.b64decode(body["audio"], validate=True), body["format"]
        except (KeyError, ValueError, TypeError):
            raise RuntimeError("9Router returned invalid audio data") from None
    actual = {"audio/mpeg": "mp3", "audio/mp3": "mp3", "audio/wav": "wav",
              "audio/x-wav": "wav", "audio/ogg": "ogg", "audio/opus": "opus",
              "audio/flac": "flac"}.get(content_type)
    if actual is None:
        raise RuntimeError("9Router returned an unsupported audio response")
    return response.content, actual


def _future(value):
    if not value:
        return False
    try:
        if isinstance(value, (int, float)):
            return value / 1000 > time.time()
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp() > time.time()
    except (ValueError, TypeError, OverflowError):
        return True  # Unknown eligibility must not expose a key.


def _write_audio(output_path: str, audio: bytes, actual_format: str) -> str:
    fmt = _FORMATS.get(str(actual_format).lower())
    if fmt is None or not audio:
        raise RuntimeError("TTS returned empty audio or an unsupported audio format")
    path = Path(output_path)
    if path.suffix.lower() != "." + fmt:
        path = path.with_suffix("." + fmt)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".tts-", delete=False) as handle:
            temporary = handle.name
            handle.write(audio)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return str(path)


class NineRouterTTSProvider(TTSProvider):
    def __init__(self) -> None:
        self._models_cache = None
        self._models_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._el_states = {}
        self._transport_notices = set()

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
        configured = str(settings.get("model") or "").strip()
        _transport(settings, configured)
        if configured:
            return configured
        models = self._fetch_models(settings)
        return models[0] if models else None

    def list_models(self) -> List[Dict[str, Any]]:
        settings = _load_settings()
        _transport(settings, str(settings.get("model") or "").strip())
        return [{"id": model, "display": model} for model in self._fetch_models(settings)]

    def list_voices(self) -> List[Dict[str, Any]]:
        settings = _load_settings()
        if _transport(settings, str(settings.get("model") or "").strip()) != "elevenlabs-direct":
            return []
        el = settings.get("elevenlabs")
        voices = el.get("voices") if isinstance(el, dict) else None
        if not isinstance(voices, dict):
            return []
        return [{"id": alias, "display": alias, "provider": "elevenlabs"}
                for alias, entries in voices.items() if isinstance(entries, list) and entries]

    def get_setup_schema(self) -> Dict[str, Any]:
        return {"name": "9Router TTS", "badge": "",
                "tag": "9Router speech and optional local ElevenLabs rotation",
                "env_vars": [{"key": "NINEROUTER_KEY",
                              "prompt": "9Router API key (optional for an auth-free local gateway)"}]}

    def _fetch_models(self, settings=None, api_key=None) -> List[str]:
        try:
            settings = _load_settings() if settings is None else settings
            base = _base_url(settings)
            key = _api_key() if api_key is None else api_key
            identity = (_profile(), base, hashlib.sha256(key.encode()).digest())
            with self._models_lock:
                cached = self._models_cache
                if cached and cached[0] == identity and time.monotonic() - cached[2] < _MODELS_CACHE_TTL:
                    return list(cached[1])
                with _session() as session:
                    response = session.get(base + "/v1/models/tts",
                                           headers={"Authorization": "Bearer " + key} if key else {},
                                           timeout=15, allow_redirects=False)
                    try:
                        if response.status_code != 200:
                            return []
                        body = response.json()
                        entries = body.get("data") if isinstance(body, dict) else None
                        if not isinstance(entries, list):
                            return []
                        models = list(dict.fromkeys(
                            e["id"].strip() for e in entries if isinstance(e, dict)
                            and isinstance(e.get("id"), str) and e["id"].strip()))
                    finally:
                        response.close()
                self._models_cache = (identity, tuple(models), time.monotonic())
                return models
        except Exception:
            logger.debug("9Router TTS model catalog unavailable")
            return []

    def _model_chain(self, model_arg, settings=None, api_key=None) -> List[str]:
        settings = _load_settings() if settings is None else settings
        primary = str(model_arg or settings.get("model") or "").strip()
        if not primary:
            models = self._fetch_models(settings, api_key)
            primary = models[0] if models else ""
        if not primary:
            return []
        raw = settings.get("fallback_models") or []
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, (list, tuple)):
            raise ValueError("fallback_models must be a list or string")
        result = [primary]
        for candidate in raw:
            candidate = str(candidate).strip()
            if candidate and candidate not in result and _provider(candidate) == _provider(primary):
                result.append(candidate)
        return result

    def _read_db_keys(self, provider="elevenlabs", settings=None, model=None) -> Dict[str, str]:
        settings = _load_settings() if settings is None else settings
        db = _db_path(settings)
        keys = {}
        try:
            # mode=ro prevents creating a missing DB; regular read-only mode sees WAL.
            connection = sqlite3.connect(db.as_uri() + "?mode=ro", uri=True, timeout=2)
            try:
                rows = connection.execute(
                    "SELECT name, data FROM providerConnections WHERE provider = ? AND isActive = 1",
                    (provider,)).fetchall()
            finally:
                connection.close()
            duplicates = set()
            for name, raw in rows:
                try:
                    data = json.loads(raw)
                except (ValueError, TypeError):
                    continue
                if not isinstance(data, dict):
                    continue
                locks = ["rateLimitedUntil", "modelLock___all"]
                if model:
                    prefix = "modelLock_" + model
                    locks.extend(field for field in data
                                 if field == prefix or field.startswith(prefix + "/"))
                if any(_future(data.get(field)) for field in locks):
                    continue
                key = data.get("apiKey")
                if not isinstance(name, str) or not isinstance(key, str) or not key.strip():
                    continue
                if name in keys:
                    duplicates.add(name)
                keys[name] = key.strip()
            # A name must unambiguously identify a single connection.
            for name in duplicates:
                keys.pop(name, None)
        except sqlite3.Error:
            logger.debug("Local 9Router ElevenLabs keys unavailable")
        return keys

    def _el_order(self, identity, entries, strategy, attempted=()):
        with self._state_lock:
            if len(self._el_states) >= 64 and identity not in self._el_states:
                self._el_states.clear()
            state = self._el_states.setdefault(identity, {"index": 0, "cooldown": {}})
            now = time.monotonic()
            available = [i for i in range(len(entries)) if i not in attempted and state["cooldown"].get(i, 0) <= now]
            start = state["index"] % len(entries)
            ordered = sorted(available, key=lambda i: (i - start) % len(entries))
            if ordered and strategy == "round-robin":
                state["index"] = (ordered[0] + 1) % len(entries)
            return ordered

    def _el_quota(self, identity, index, count, strategy, quota=True):
        with self._state_lock:
            state = self._el_states.get(identity)
            if state is not None:
                if quota:
                    state["cooldown"][index] = time.monotonic() + 60
                if strategy == "fill-first":
                    state["index"] = (index + 1) % count

    def _synthesize_elevenlabs(self, text, output_path, voice, model, fmt, settings, speed):
        el = settings.get("elevenlabs") or {}
        if not isinstance(el, dict):
            raise ValueError("elevenlabs must be a mapping")
        parts = model.split("/")
        if parts[0].lower() in ("el", "elevenlabs"):
            parts = parts[1:]
        model_id = parts[0] if parts and parts[0].startswith("eleven_") else str(el.get("model_id") or DEFAULT_EL_MODEL)
        voice = voice or (parts[1] if len(parts) > 1 else None)
        voices = el.get("voices") or {}
        if not isinstance(voices, dict):
            raise ValueError("elevenlabs.voices must be a mapping")
        keys = self._read_db_keys(settings=settings, model=model_id)
        configured = voices.get(voice) if voice in voices else None
        if configured is None and voice:
            configured = [{"connection": name, "voice_id": voice} for name in sorted(keys)]
        entries = []
        for entry in configured if isinstance(configured, list) else []:
            if not isinstance(entry, dict):
                continue
            name, voice_id = entry.get("connection"), entry.get("voice_id")
            if isinstance(voice_id, str) and voice_id.strip() and name in keys:
                item = (name, voice_id.strip(), keys[name])
                if item not in entries:
                    entries.append(item)
        if not entries:
            raise ValueError("No eligible local ElevenLabs voice mapping and active key")
        strategy = str(el.get("strategy") or "fill-first").lower()
        if strategy not in ("fill-first", "round-robin"):
            raise ValueError("elevenlabs.strategy must be fill-first or round-robin")
        identity = (_profile(), str(_db_path(settings)), voice, model_id, strategy,
                    tuple((n, v, hashlib.sha256(k.encode()).digest()) for n, v, k in entries))
        if fmt == "wav":
            el_format, actual = "wav_44100", "wav"
        elif fmt in ("ogg", "opus"):
            el_format, actual = "opus_48000_64", "ogg"
        else:
            el_format, actual = "mp3_44100_128", "mp3"
        attempted = set()
        for _ in range(len(entries)):
            order = self._el_order(identity, entries, strategy, attempted)
            if not order:
                break
            index = order[0]
            attempted.add(index)
            _, voice_id, key = entries[index]
            payload = {"text": text, "model_id": model_id}
            if speed is not None:
                payload["voice_settings"] = {"speed": speed}
            with _session() as session:
                try:
                    response = session.post(
                        DEFAULT_EL_API + "/v1/text-to-speech/" + quote(voice_id, safe=""),
                        headers={"xi-api-key": key, "Content-Type": "application/json"},
                        params={"output_format": el_format}, json=payload,
                        timeout=_timeout(settings), allow_redirects=False)
                except Exception:
                    raise RuntimeError("ElevenLabs request failed; check connectivity and settings") from None
                try:
                    if response.status_code == 200:
                        return _write_audio(output_path, response.content, actual)
                    try:
                        body = response.json()
                    except ValueError:
                        body = {}
                    quota = _is_quota_error(response.status_code, body)
                    detail = body.get("detail") if isinstance(body, dict) else None
                    code = str(detail.get("status") or "") if isinstance(detail, dict) else ""
                    # A different account/voice mapping can resolve these failures.
                    key_specific = (response.status_code in (401, 403)
                                    or code in ("voice_not_found", "invalid_voice_id", "voice_access_denied"))
                    if not quota and not key_specific:
                        raise RuntimeError("ElevenLabs synthesis failed (HTTP %s)" % response.status_code)
                    self._el_quota(identity, index, len(entries), strategy, quota=quota)
                finally:
                    response.close()
        raise RuntimeError("No eligible ElevenLabs mapping succeeded; check key/voice access or retry after quota cooldown")

    def _synthesize_9router(self, text, output_path, voice, model, fmt, settings, speed, caller_voice=None):
        base, key = _base_url(settings), _api_key()
        chain = self._model_chain(model, settings, key)
        if not chain:
            raise ValueError("No TTS model configured or available from 9Router")
        headers = {"Authorization": "Bearer " + key} if key else {}
        for model_id in chain:
            request_voice = voice
            if _is_elevenlabs_model(model_id):
                # A fallback may encode its own voice; only a caller override wins.
                request_voice = caller_voice or _encoded_voice(model_id) or voice
            payload = {"model": _request_model(model_id, request_voice), "input": text,
                       "response_format": fmt if fmt in _FORMATS else "mp3"}
            if request_voice:
                payload["voice"] = request_voice
            if speed is not None:
                payload["speed"] = speed
            with _session() as session:
                try:
                    response = session.post(base + "/v1/audio/speech", headers=headers, json=payload,
                                            timeout=_timeout(settings), allow_redirects=False)
                except Exception:
                    raise RuntimeError("9Router TTS request failed; check connectivity and settings") from None
                try:
                    try:
                        body = response.json()
                    except ValueError:
                        body = {}
                    if response.status_code == 200 and not (isinstance(body, dict) and body.get("error")):
                        audio, actual = _audio_response(response, fmt)
                        return _write_audio(output_path, audio, actual)
                    if not _is_quota_error(response.status_code, body):
                        raise RuntimeError("9Router TTS synthesis failed (HTTP %s)" % response.status_code)
                finally:
                    response.close()
        raise RuntimeError("Configured 9Router TTS models are quota-limited; retry later")

    def _notice_implicit_transport(self, settings, model, transport):
        if "transport" in settings or not _is_elevenlabs_model(model):
            return
        identity = (_profile(), transport)
        with self._state_lock:
            if identity in self._transport_notices:
                return
            if len(self._transport_notices) >= 64:
                self._transport_notices.clear()
            self._transport_notices.add(identity)
        if transport == "elevenlabs-direct":
            logger.warning(
                "Legacy ElevenLabs direct settings detected. Set "
                "tts.9router-tts.transport: elevenlabs-direct to keep local rotation, "
                "or gateway to use 9Router."
            )
        else:
            logger.warning(
                "ElevenLabs defaults to the 9Router gateway. Set "
                "tts.9router-tts.transport: gateway explicitly; users of legacy "
                "direct model/voice-only configs must set elevenlabs-direct to retain direct routing."
            )

    def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None,
                   model: Optional[str] = None, speed: Optional[float] = None,
                   format: str = "mp3", **extra: Any) -> str:
        settings = _load_settings()
        model = str(model or settings.get("model") or "").strip()
        # Reject invalid selectors and incompatible explicit direct configs before I/O.
        transport = _transport(settings, model)
        if not model:
            models = self._fetch_models(settings)
            model = models[0] if models else ""
        if not model:
            raise ValueError("No TTS model configured or available from 9Router")
        transport = _transport(settings, model)
        caller_voice = voice
        embedded_voice = _encoded_voice(model)
        parts = model.split("/")
        if transport == "elevenlabs-direct" and parts[0].lower() in ("el", "elevenlabs") and len(parts) == 2:
            # Preserve direct-mode alias selection from pre-0.1.2 configurations.
            embedded_voice = None
        voice = (voice or embedded_voice
                 or str(settings.get("voice") or settings.get("default_voice") or "").strip() or None)
        if voice and ("/" in voice or any(ord(c) < 32 for c in voice)):
            raise ValueError("voice must be a single voice ID or alias")
        if speed is not None:
            speed = float(speed)
            if not math.isfinite(speed) or speed <= 0:
                raise ValueError("speed must be positive and finite")
        _timeout(settings)
        fmt = str(format).strip().lower()
        self._notice_implicit_transport(settings, model, transport)
        if transport == "elevenlabs-direct":
            return self._synthesize_elevenlabs(text, output_path, voice, model, fmt, settings, speed)
        return self._synthesize_9router(text, output_path, voice, model, fmt, settings, speed,
                                        caller_voice=caller_voice)

    def stream(self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None,
               format: str = "opus", **extra: Any) -> Iterator[bytes]:
        raise NotImplementedError("9Router TTS uses whole-file synthesis")


def register(ctx) -> None:
    ctx.register_tts_provider(NineRouterTTSProvider())
