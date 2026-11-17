"""Offline TTS regressions against the real pinned Hermes host, cloud runners only.

PLUGIN_SOURCE selects the plugin checkout. --baseline proves seven defects in
the original revision; it succeeds only when every selected regression fails.
All endpoints, keys, connection names, voices and audio below are synthetic.
"""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import io
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
import wave

import requests
from agent.tts_provider import TTSProvider

SOURCE = Path(os.environ.get("PLUGIN_SOURCE", Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location(
    "tts_regression_plugin", SOURCE / "__init__.py",
    submodule_search_locations=[str(SOURCE)],
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
impl = sys.modules[module.NineRouterTTSProvider.__module__]

wav_buffer = io.BytesIO()
with wave.open(wav_buffer, "wb") as wav:
    wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
    wav.writeframes(b"\0\0" * 160)
WAV = wav_buffer.getvalue()
MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00" + b"\xff\xfb\x90\x00" + b"\0" * 64


def response(status=200, audio=MP3, content_type="audio/mpeg", body=None):
    result = requests.Response()
    result.status_code = status
    result.headers["Content-Type"] = content_type
    result._content = json.dumps(body).encode() if body is not None else audio
    result._content_consumed = True
    return result


class Regressions(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="tts-regression-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.settings = {"base_url": "http://127.0.0.1:20128", "model": "gemini/fixture"}
        self.settings_patch = patch.object(impl, "_load_settings", side_effect=lambda: self.settings)
        self.settings_patch.start()
        self.addCleanup(self.settings_patch.stop)
        self.key_patch = patch.object(impl, "_api_key", return_value="synthetic-router-key")
        self.key_patch.start()
        self.addCleanup(self.key_patch.stop)
        self.provider = module.NineRouterTTSProvider()
        self.calls = []
        self.responses = [response()]
        self.transport_lock = threading.Lock()

        def transport(session, method, url, **kwargs):
            with self.transport_lock:
                self.calls.append((session, method.upper(), url, kwargs))
                if not self.responses:
                    raise AssertionError("Unexpected extra network request")
                item = self.responses.pop(0)
            if isinstance(item, BaseException):
                raise item
            return item

        self.transport = patch("requests.sessions.Session.request", new=transport)
        self.transport.start()
        self.addCleanup(self.transport.stop)

    def synthesize(self, name="speech.mp3", **kwargs):
        return self.provider.synthesize("synthetic public test text", str(self.root / name), **kwargs)

    def database(self, rows=None):
        path = self.root / "router.sqlite"
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE providerConnections (id TEXT PRIMARY KEY, name TEXT, provider TEXT, isActive INTEGER, data TEXT)")
        entries = rows if rows is not None else [
            ("one", "fixture-one", "elevenlabs", 1, {"apiKey": "synthetic-key-one"}),
            ("two", "fixture-two", "elevenlabs", 1, {"apiKey": "synthetic-key-two"}),
        ]
        conn.executemany(
            "INSERT INTO providerConnections VALUES (?, ?, ?, ?, ?)",
            [(id_, name, provider, active, json.dumps(data)) for id_, name, provider, active, data in entries],
        )
        conn.commit()
        conn.close()
        self.settings["db_path"] = str(path)
        return path

    def elevenlabs(self):
        path = self.database()
        self.settings.update({
            "model": "el/eleven_flash_v2_5", "voice": "narrator",
            "transport": "elevenlabs-direct",
            "elevenlabs": {"strategy": "round-robin", "voices": {
                "narrator": [
                    {"connection": "fixture-one", "voice_id": "Voice0001"},
                    {"connection": "fixture-two", "voice_id": "Voice0002"},
                ]
            }},
        })
        return path

    def test_passive_registration_and_availability(self):
        registered = []
        with patch("sqlite3.connect", side_effect=AssertionError("Discovery opened SQLite")):
            module.register(type("Context", (), {"register_tts_provider": lambda _, p: registered.append(p)})())
            self.assertTrue(registered[0].is_available())
        self.assertIsInstance(registered[0], TTSProvider)
        self.assertEqual(registered[0].name, "9router-tts")
        self.assertEqual(self.calls, [])

    def test_default_voice_is_applied(self):
        self.settings["voice"] = "Kore"
        self.synthesize()
        payload = self.calls[0][3]["json"]
        self.assertEqual(payload.get("voice"), "Kore")

    def test_voice_is_encoded_in_router_model(self):
        self.synthesize(voice="Kore")
        self.assertEqual(self.calls[0][3]["json"]["model"], "gemini/fixture/Kore")

    def test_explicit_voice_overrides_configured_voice(self):
        self.settings["voice"] = "Kore"
        self.synthesize(voice="Puck")
        self.assertEqual(self.calls[0][3]["json"].get("voice"), "Puck")
        self.assertEqual(self.calls[0][3]["json"]["model"], "gemini/fixture/Puck")

    def test_requested_format_is_forwarded(self):
        self.responses = [response(audio=WAV, content_type="audio/wav")]
        self.synthesize("speech.wav", format="wav")
        self.assertEqual(self.calls[0][3]["json"].get("response_format"), "wav")

    def test_actual_audio_suffix_is_returned(self):
        self.responses = [response(audio=WAV, content_type="audio/wav")]
        written = self.synthesize()
        self.assertEqual(Path(written).suffix, ".wav")
        self.assertEqual(Path(written).read_bytes(), WAV)

    def test_explicit_model_controls_request(self):
        self.synthesize(model="openai/tts-1")
        self.assertEqual(self.calls[0][3]["json"]["model"], "openai/tts-1")

    def test_empty_success_is_failure(self):
        self.responses = [response(audio=b"")]
        with self.assertRaises((RuntimeError, ValueError)):
            self.synthesize()
        self.assertFalse((self.root / "speech.mp3").exists())

    def test_json_error_with_success_status_is_not_audio(self):
        self.responses = [response(body={"error": {"message": "synthetic failure"}}, content_type="application/json")]
        with self.assertRaises((RuntimeError, ValueError)):
            self.synthesize()
        self.assertFalse((self.root / "speech.mp3").exists())

    def test_json_audio_fallback(self):
        self.responses = [response(body={"audio": base64.b64encode(WAV).decode(), "format": "wav"}, content_type="application/json")]
        written = self.synthesize()
        self.assertEqual(Path(written).suffix, ".wav")
        self.assertEqual(Path(written).read_bytes(), WAV)

    def test_hard_error_does_not_fallback(self):
        self.settings["fallback_models"] = ["gemini/second"]
        self.responses = [response(401, body={"error": {"message": "denied"}}), response()]
        with self.assertRaises((RuntimeError, ValueError)):
            self.synthesize()
        self.assertEqual(len(self.calls), 1)

    def test_acl_error_does_not_fallback(self):
        self.settings["fallback_models"] = ["gemini/second"]
        self.responses = [response(403, body={"error": {"code": "insufficient_quota", "message": "Not allowed to use model"}}), response()]
        with self.assertRaises((RuntimeError, ValueError)):
            self.synthesize()
        self.assertEqual(len(self.calls), 1)

    def test_quota_uses_only_same_provider_fallbacks(self):
        self.settings["fallback_models"] = ["openai/tts-1", "gemini/second"]
        self.responses = [response(429, body={"error": {"message": "quota"}}), response()]
        self.synthesize()
        self.assertEqual([call[3]["json"]["model"] for call in self.calls], ["gemini/fixture", "gemini/second"])

    def test_model_override_does_not_retry_configured_primary(self):
        self.responses = [response(429, body={"error": {"message": "quota"}}), response()]
        with self.assertRaises((RuntimeError, ValueError)):
            self.synthesize(model="gemini/override")
        self.assertEqual(len(self.calls), 1)

    def test_timeout_does_not_fallback(self):
        self.settings["fallback_models"] = ["gemini/second"]
        self.responses = [requests.Timeout("synthetic private timeout"), response()]
        with self.assertRaises((RuntimeError, ValueError)):
            self.synthesize()
        self.assertEqual(len(self.calls), 1)

    def test_provider_error_body_is_private(self):
        private = "PRIVATE_TEXT synthetic-private-key VoicePrivate"
        self.responses = [response(401, body={"error": {"message": private}})]
        with self.assertRaises((RuntimeError, ValueError)) as error:
            self.synthesize()
        self.assertNotIn(private, str(error.exception))

    def test_transport_error_is_private(self):
        private = "PRIVATE_TEXT synthetic-private-key"
        self.responses = [requests.ConnectionError(private)]
        with self.assertRaises((RuntimeError, ValueError)) as error:
            self.synthesize()
        self.assertNotIn(private, str(error.exception))

    def test_failed_request_preserves_existing_file(self):
        target = self.root / "speech.mp3"
        target.write_bytes(b"existing valid file")
        self.responses = [response(500, body={"error": {"message": "synthetic failure"}})]
        with self.assertRaises((RuntimeError, ValueError)):
            self.synthesize()
        self.assertEqual(target.read_bytes(), b"existing valid file")
        self.assertEqual(list(self.root.iterdir()), [target])

    def test_redirects_and_ambient_credentials_are_disabled(self):
        self.synthesize()
        session, _, _, kwargs = self.calls[0]
        self.assertIs(session.trust_env, False)
        self.assertIs(kwargs.get("allow_redirects"), False)
        self.assertEqual(kwargs["headers"].get("Authorization"), "Bearer synthetic-router-key")

    def test_scoped_key_miss_does_not_borrow_process_key(self):
        self.key_patch.stop()
        with patch.dict(os.environ, {"NINEROUTER_KEY": "private-launch-key"}), patch("agent.secret_scope.get_secret", return_value=None):
            self.assertEqual(impl._api_key(), "")

    def test_scoping_error_is_not_swallowed(self):
        from agent.secret_scope import UnscopedSecretError
        self.key_patch.stop()
        with patch.dict(os.environ, {"NINEROUTER_KEY": "private-launch-key"}), patch("agent.secret_scope.get_secret", side_effect=UnscopedSecretError("NINEROUTER_KEY")):
            with self.assertRaises(UnscopedSecretError):
                impl._api_key()

    def test_scoped_url_miss_does_not_borrow_process_url(self):
        with patch.dict(os.environ, {"NINEROUTER_URL": "https://launch-profile.invalid"}), patch("agent.secret_scope.get_secret", return_value=None):
            self.assertEqual(impl._base_url({}), impl.DEFAULT_BASE_URL)

    def test_legacy_settings_merge_with_canonical_precedence(self):
        self.settings_patch.stop()
        cfg = {"tts": {"9router": {"voice": "Kore", "timeout": 7}, "9router-tts": {"voice": "Puck"}}}
        with patch("hermes_cli.config.load_config", return_value=cfg):
            settings = impl._load_settings()
        self.assertEqual(settings["voice"], "Puck")
        self.assertEqual(settings["timeout"], 7)

    def test_db_open_is_readonly(self):
        self.database()
        connect = sqlite3.connect
        with patch("sqlite3.connect", wraps=connect) as opened:
            keys = self.provider._read_db_keys()
        self.assertEqual(keys["fixture-one"], "synthetic-key-one")
        self.assertTrue(opened.call_args.kwargs.get("uri"))
        self.assertIn("mode=ro", str(opened.call_args.args[0]))

    def test_missing_db_is_never_created(self):
        missing = self.root / "missing.sqlite"
        self.settings["db_path"] = str(missing)
        try:
            self.provider._read_db_keys()
        except (RuntimeError, ValueError):
            pass
        self.assertFalse(missing.exists())

    def test_db_key_rotation_is_immediately_visible_with_wal(self):
        path = self.database()
        writer = sqlite3.connect(path)
        self.addCleanup(writer.close)
        writer.execute("PRAGMA journal_mode=WAL")
        first = self.provider._read_db_keys()
        self.assertEqual(first["fixture-one"], "synthetic-key-one")
        writer.execute("UPDATE providerConnections SET data=? WHERE id='one'", (json.dumps({"apiKey": "synthetic-rotated-key"}),))
        writer.commit()
        second = self.provider._read_db_keys()
        self.assertEqual(second["fixture-one"], "synthetic-rotated-key")

    def test_inactive_and_other_provider_keys_are_excluded(self):
        self.database([
            ("one", "fixture-one", "elevenlabs", 1, {"apiKey": "synthetic-key-one"}),
            ("two", "fixture-two", "elevenlabs", 0, {"apiKey": "synthetic-key-two"}),
            ("three", "fixture-other", "openai", 1, {"apiKey": "synthetic-key-other"}),
        ])
        self.assertEqual(self.provider._read_db_keys(), {"fixture-one": "synthetic-key-one"})

    def test_direct_pool_never_sends_keys_to_remote_router(self):
        self.elevenlabs()
        self.settings["base_url"] = "https://remote-router.invalid"
        self.synthesize(voice="narrator")
        self.assertEqual(self.calls[0][2], "https://api.elevenlabs.io/v1/text-to-speech/Voice0001")
        self.assertEqual(self.calls[0][3]["headers"]["xi-api-key"], "synthetic-key-one")

    def test_elevenlabs_uses_format_query(self):
        self.elevenlabs()
        self.responses = [response(audio=WAV, content_type="audio/wav")]
        written = self.synthesize("speech.wav", voice="narrator", format="wav")
        self.assertEqual(Path(written).read_bytes(), WAV)
        _, _, url, kwargs = self.calls[0]
        self.assertEqual(url, "https://api.elevenlabs.io/v1/text-to-speech/Voice0001")
        self.assertEqual(kwargs.get("params", {}).get("output_format"), "wav_44100")
        self.assertNotIn("output_format", kwargs["json"])

    def test_round_robin_balances_current_entries(self):
        self.elevenlabs()
        self.responses = [response(), response(), response(), response()]
        for index in range(4):
            self.synthesize(f"speech{index}.mp3", voice="narrator")
        self.assertEqual([call[3]["headers"]["xi-api-key"] for call in self.calls],
                         ["synthetic-key-one", "synthetic-key-two", "synthetic-key-one", "synthetic-key-two"])

    def test_mapping_changes_do_not_reuse_stale_rotation(self):
        self.elevenlabs()
        self.responses = [response(), response()]
        self.synthesize("first.mp3", voice="narrator")
        self.settings["elevenlabs"]["voices"]["narrator"] = [{"connection": "fixture-one", "voice_id": "Voice0003"}]
        self.synthesize("second.mp3", voice="narrator")
        self.assertTrue(self.calls[-1][2].endswith("/Voice0003"))

    def test_revoked_key_is_not_reused(self):
        path = self.elevenlabs()
        self.responses = [response(), response()]
        self.synthesize("first.mp3", voice="narrator")
        with sqlite3.connect(path) as writer:
            writer.execute("UPDATE providerConnections SET isActive=0 WHERE id='two'")
        self.synthesize("second.mp3", voice="narrator")
        self.assertEqual(self.calls[-1][3]["headers"]["xi-api-key"], "synthetic-key-one")

    def test_quota_key_cooldown_survives_next_call(self):
        self.elevenlabs()
        self.responses = [response(429, body={"detail": {"status": "quota_exceeded", "message": "quota"}}), response(), response()]
        self.synthesize("first.mp3", voice="narrator")
        self.synthesize("second.mp3", voice="narrator")
        self.assertEqual([call[3]["headers"]["xi-api-key"] for call in self.calls],
                         ["synthetic-key-one", "synthetic-key-two", "synthetic-key-two"])

    def test_concurrent_round_robin_balances_keys(self):
        self.elevenlabs()
        self.responses = [response() for _ in range(8)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            paths = list(pool.map(lambda index: self.synthesize(f"thread{index}.mp3", voice="narrator"), range(8)))
        self.assertTrue(all(Path(path).read_bytes() == MP3 for path in paths))
        keys = [call[3]["headers"]["xi-api-key"] for call in self.calls]
        self.assertEqual(keys.count("synthetic-key-one"), 4)
        self.assertEqual(keys.count("synthetic-key-two"), 4)


    def test_pip_metadata_uses_native_plugin_entrypoint(self):
        import tomllib
        metadata = tomllib.loads((SOURCE / "pyproject.toml").read_text())
        self.assertEqual(metadata["project"]["entry-points"]["hermes_agent.plugins"]["9router-tts"],
                         "hermes_9router_tts:register")
        self.assertEqual(metadata["project"]["version"], "0.1.2")

    def test_ambiguous_connection_names_are_not_used(self):
        self.database([
            ("one", "duplicate", "elevenlabs", 1, {"apiKey": "synthetic-key-one"}),
            ("two", "duplicate", "elevenlabs", 1, {"apiKey": "synthetic-key-two"}),
        ])
        self.assertEqual(self.provider._read_db_keys(), {})

    def test_router_cooldown_and_model_lock_exclude_keys(self):
        self.database([
            ("one", "future-rate", "elevenlabs", 1, {"apiKey": "synthetic-one", "rateLimitedUntil": "2999-01-01T00:00:00Z"}),
            ("two", "future-all", "elevenlabs", 1, {"apiKey": "synthetic-two", "modelLock___all": "2999-01-01T00:00:00Z"}),
            ("three", "future-model", "elevenlabs", 1, {"apiKey": "synthetic-three", "modelLock_eleven_flash_v2_5": "2999-01-01T00:00:00Z"}),
            ("four", "eligible", "elevenlabs", 1, {"apiKey": "synthetic-four"}),
        ])
        self.assertEqual(self.provider._read_db_keys(model="eleven_flash_v2_5"), {"eligible": "synthetic-four"})

    def test_routed_profile_requires_explicit_database(self):
        self.settings.pop("db_path", None)
        with patch("agent.secret_scope.serves_routed_profile", return_value=True):
            with self.assertRaises(ValueError):
                impl._db_path(self.settings)

    def test_model_catalog_cache_is_profile_scoped(self):
        self.responses = [
            response(body={"data": [{"id": "gemini/profile-one"}]}, content_type="application/json"),
            response(body={"data": [{"id": "gemini/profile-two"}]}, content_type="application/json"),
        ]
        with patch("hermes_constants.get_hermes_home", return_value=self.root / "profile-one"):
            self.assertEqual(self.provider._fetch_models(), ["gemini/profile-one"])
        with patch("hermes_constants.get_hermes_home", return_value=self.root / "profile-two"):
            self.assertEqual(self.provider._fetch_models(), ["gemini/profile-two"])
            self.assertEqual(self.provider._fetch_models(), ["gemini/profile-two"])
        self.assertEqual(len(self.calls), 2)

    def test_atomic_replace_failure_preserves_file_and_cleans_temp(self):
        target = self.root / "speech.mp3"
        target.write_bytes(b"existing valid file")
        with patch("os.replace", side_effect=OSError("synthetic disk failure")):
            with self.assertRaises(OSError):
                self.synthesize()
        self.assertEqual(target.read_bytes(), b"existing valid file")
        self.assertEqual(list(self.root.iterdir()), [target])

    def test_elevenlabs_401_quota_rotates(self):
        self.elevenlabs()
        self.responses = [
            response(401, body={"detail": {"status": "quota_exceeded", "message": "quota"}}),
            response(),
        ]
        self.synthesize(voice="narrator")
        self.assertEqual([call[3]["headers"]["xi-api-key"] for call in self.calls],
                         ["synthetic-key-one", "synthetic-key-two"])

    def test_elevenlabs_401_invalid_key_tries_next_mapping(self):
        self.elevenlabs()
        self.responses = [
            response(401, body={"detail": {"status": "invalid_api_key", "message": "denied"}}),
            response(),
        ]
        self.synthesize(voice="narrator")
        self.assertEqual([call[3]["headers"]["xi-api-key"] for call in self.calls],
                         ["synthetic-key-one", "synthetic-key-two"])

    def test_round_robin_moves_after_successful_retry(self):
        path = self.elevenlabs()
        with sqlite3.connect(path) as writer:
            writer.execute("INSERT INTO providerConnections VALUES (?, ?, ?, ?, ?)",
                           ("three", "fixture-three", "elevenlabs", 1, json.dumps({"apiKey": "synthetic-key-three"})))
        self.settings["elevenlabs"]["voices"]["narrator"].append(
            {"connection": "fixture-three", "voice_id": "Voice0003"})
        self.responses = [
            response(429, body={"detail": {"status": "quota_exceeded", "message": "quota"}}),
            response(), response(),
        ]
        self.synthesize("first.mp3", voice="narrator")
        self.synthesize("second.mp3", voice="narrator")
        self.assertEqual([call[3]["headers"]["xi-api-key"] for call in self.calls],
                         ["synthetic-key-one", "synthetic-key-two", "synthetic-key-three"])

    def test_embedded_proxy_voice_precedes_default_voice(self):
        self.settings["voice"] = "Kore"
        self.synthesize(model="gemini/fixture/Puck")
        self.assertEqual(self.calls[0][3]["json"]["model"], "gemini/fixture/Puck")

    def test_embedded_elevenlabs_voice_precedes_default_alias(self):
        self.elevenlabs()
        self.synthesize(model="el/eleven_flash_v2_5/Voice0009")
        self.assertTrue(self.calls[0][2].endswith("/Voice0009"))


    def test_elevenlabs_invalid_model_stops_all_mappings(self):
        self.elevenlabs()
        self.responses = [response(400, body={"detail": {"status": "invalid_model_id", "message": "invalid model"}}), response()]
        with self.assertRaises(RuntimeError):
            self.synthesize()
        self.assertEqual(len(self.calls), 1)

    def test_elevenlabs_voice_access_failure_tries_next_mapping(self):
        self.elevenlabs()
        self.responses = [response(404, body={"detail": {"status": "voice_not_found", "message": "missing voice"}}), response()]
        self.synthesize()
        self.assertEqual(len(self.calls), 2)

    def test_router_voice_specific_lock_excludes_key(self):
        self.database([
            ("one", "locked", "elevenlabs", 1, {"apiKey": "synthetic-one",
                "modelLock_eleven_flash_v2_5/Voice0001": "2999-01-01T00:00:00Z"}),
            ("two", "eligible", "elevenlabs", 1, {"apiKey": "synthetic-two"}),
        ])
        self.assertEqual(self.provider._read_db_keys(model="eleven_flash_v2_5"),
                         {"eligible": "synthetic-two"})


    def test_new_elevenlabs_models_use_gateway_without_database(self):
        for model in ("el/eleven_flash_v2_5", "elevenlabs/eleven_flash_v2_5",
                      "eleven_flash_v2_5"):
            with self.subTest(model=model):
                self.settings = {"base_url": "http://127.0.0.1:20128", "model": model}
                self.calls.clear()
                self.responses = [response()]
                with patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Gateway opened keys")), \
                     patch("sqlite3.connect", side_effect=AssertionError("Gateway opened SQLite")):
                    self.synthesize(voice="Voice0009")
                self.assertEqual(len(self.calls), 1)
                _, _, url, kwargs = self.calls[0]
                self.assertEqual(url, "http://127.0.0.1:20128/v1/audio/speech")
                self.assertEqual(kwargs["json"]["model"], "elevenlabs/eleven_flash_v2_5/Voice0009")
                self.assertEqual(kwargs["headers"].get("Authorization"), "Bearer synthetic-router-key")
                self.assertNotIn("xi-api-key", kwargs["headers"])

    def test_explicit_gateway_overrides_legacy_elevenlabs_markers(self):
        self.elevenlabs()
        self.settings["transport"] = "gateway"
        with patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Gateway opened keys")), \
             patch("sqlite3.connect", side_effect=AssertionError("Gateway opened SQLite")), \
             patch.object(impl.logger, "warning") as warning:
            self.synthesize(voice="Voice0009")
        self.assertEqual(self.calls[0][2], "http://127.0.0.1:20128/v1/audio/speech")
        self.assertEqual(self.calls[0][3]["json"]["model"], "elevenlabs/eleven_flash_v2_5/Voice0009")
        warning.assert_not_called()

    def test_explicit_direct_transport_retains_rotation_without_migration_warning(self):
        self.elevenlabs()
        self.settings["transport"] = "elevenlabs-direct"
        self.responses = [response(), response()]
        with patch.object(impl.logger, "warning") as warning:
            self.synthesize("first.mp3")
            self.synthesize("second.mp3")
        self.assertEqual([call[3]["headers"]["xi-api-key"] for call in self.calls],
                         ["synthetic-key-one", "synthetic-key-two"])
        self.assertTrue(all(call[2].startswith("https://api.elevenlabs.io/v1/text-to-speech/")
                            for call in self.calls))
        warning.assert_not_called()

    def test_legacy_direct_markers_keep_routing_and_emit_safe_migration_notice(self):
        self.elevenlabs()
        self.settings.pop("transport", None)
        with self.assertLogs(impl.logger.name, level="WARNING") as logs:
            self.synthesize()
        self.assertEqual(self.calls[0][2], "https://api.elevenlabs.io/v1/text-to-speech/Voice0001")
        notice = "\n".join(logs.output)
        self.assertIn("transport", notice.lower())
        self.assertIn("elevenlabs-direct", notice.lower())
        for private in ("synthetic-key-one", "synthetic-key-two", "Voice0001",
                        "Voice0002", "fixture-one", str(self.settings["db_path"])):
            self.assertNotIn(private, notice)

    def test_invalid_transport_rejects_before_catalog_http_or_database(self):
        self.settings = {"transport": "invalid-transport"}
        with patch.object(self.provider, "_fetch_models", side_effect=AssertionError("Invalid transport fetched catalog")), \
             patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Invalid transport opened keys")), \
             patch("sqlite3.connect", side_effect=AssertionError("Invalid transport opened SQLite")):
            with self.assertRaises(ValueError):
                self.synthesize()
        self.assertEqual(self.calls, [])

    def test_direct_transport_rejects_non_elevenlabs_before_io(self):
        self.settings["transport"] = "elevenlabs-direct"
        self.settings["model"] = "gemini/fixture"
        with patch.object(self.provider, "_fetch_models", side_effect=AssertionError("Direct fetched catalog")), \
             patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Direct opened keys")), \
             patch("sqlite3.connect", side_effect=AssertionError("Direct opened SQLite")):
            with self.assertRaises(ValueError):
                self.synthesize()
        self.assertEqual(self.calls, [])

    def test_gateway_failures_never_switch_to_direct_transport(self):
        self.elevenlabs()
        self.settings["transport"] = "gateway"
        self.settings["fallback_models"] = ["elevenlabs/eleven_turbo_v2_5"]
        for status, expected_calls in ((401, 1), (429, 2)):
            with self.subTest(status=status):
                self.calls.clear()
                self.responses = [response(status, body={"error": {"message": "synthetic failure"}})
                                  for _ in range(expected_calls)]
                with patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Gateway fallback opened keys")), \
                     patch("sqlite3.connect", side_effect=AssertionError("Gateway fallback opened SQLite")):
                    with self.assertRaises(RuntimeError):
                        self.synthesize()
                self.assertEqual(len(self.calls), expected_calls)
                self.assertTrue(all(call[2] == "http://127.0.0.1:20128/v1/audio/speech"
                                    for call in self.calls))
                self.assertTrue(all("xi-api-key" not in call[3]["headers"] for call in self.calls))

    def test_elevenlabs_gateway_aliases_encode_and_replace_only_voice(self):
        cases = (
            ("el/eleven_flash_v2_5", "Voice0009", "elevenlabs/eleven_flash_v2_5/Voice0009"),
            ("elevenlabs/eleven_flash_v2_5", "Voice0009", "elevenlabs/eleven_flash_v2_5/Voice0009"),
            ("eleven_flash_v2_5", "Voice0009", "elevenlabs/eleven_flash_v2_5/Voice0009"),
            ("el/eleven_flash_v2_5/Voice0001", "Voice0009", "elevenlabs/eleven_flash_v2_5/Voice0009"),
            ("elevenlabs/eleven_flash_v2_5/Voice0001", "Voice0009", "elevenlabs/eleven_flash_v2_5/Voice0009"),
            ("eleven_flash_v2_5/Voice0001", "Voice0009", "elevenlabs/eleven_flash_v2_5/Voice0009"),
            ("el/Voice0001", "Voice0009", "elevenlabs/Voice0009"),
        )
        for model, voice, expected in cases:
            with self.subTest(model=model):
                self.assertEqual(impl._request_model(model, voice), expected)


    def test_empty_or_malformed_legacy_markers_stay_gateway(self):
        cases = (
            {"elevenlabs": {}},
            {"elevenlabs": {"voices": {}}},
            {"elevenlabs": {"voices": {"narrator": []}}},
            {"elevenlabs": {"voices": {"narrator": [{"connection": " ", "voice_id": "PrivateVoice"}]}}},
            {"elevenlabs": {"strategy": " ", "model_id": ""}, "db_path": " "},
        )
        for extra in cases:
            with self.subTest(extra=extra):
                self.settings = {"model": "el/eleven_flash_v2_5", "voice": "Voice0009", **extra}
                self.calls.clear()
                self.responses = [response()]
                with patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Empty marker opened DB")):
                    self.synthesize()
                self.assertTrue(self.calls[0][2].endswith("/v1/audio/speech"))

    def test_each_meaningful_legacy_marker_preserves_direct(self):
        cases = (
            {"elevenlabs": {"voices": {"narrator": [{"connection": "fixture-one", "voice_id": "Voice0001"}]}}},
            {"elevenlabs": {"strategy": "round-robin"}},
            {"elevenlabs": {"model_id": "eleven_flash_v2_5"}},
            {"db_path": "/synthetic/legacy.sqlite"},
        )
        for extra in cases:
            with self.subTest(extra=extra):
                self.settings = {"model": "el/eleven_flash_v2_5", "voice": "Voice0009", **extra}
                with patch.object(self.provider, "_synthesize_elevenlabs", return_value="direct.mp3") as direct, \
                     patch.object(self.provider, "_synthesize_9router", side_effect=AssertionError("Legacy changed destination")):
                    self.assertEqual(self.synthesize(), "direct.mp3")
                direct.assert_called_once()

    def test_non_elevenlabs_model_with_legacy_markers_stays_gateway(self):
        self.settings.update({"db_path": "/synthetic/legacy.sqlite",
                              "elevenlabs": {"strategy": "round-robin"}})
        with patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Other provider opened DB")):
            self.synthesize()
        self.assertTrue(self.calls[0][2].endswith("/v1/audio/speech"))

    def test_invalid_selector_values_never_infer_legacy_direct(self):
        self.elevenlabs()
        for selector in ("", "auto", None, True, [], {}):
            with self.subTest(selector=selector):
                self.settings["transport"] = selector
                with patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Invalid selector opened DB")):
                    with self.assertRaises(ValueError):
                        self.synthesize()
        self.assertEqual(self.calls, [])

    def test_direct_without_model_never_fetches_gateway_catalog(self):
        self.settings = {"transport": "elevenlabs-direct", "voice": "Voice0009"}
        with patch.object(self.provider, "_fetch_models", side_effect=AssertionError("Direct fetched catalog")):
            with self.assertRaises(ValueError):
                self.synthesize()
            with self.assertRaises(ValueError):
                self.provider.default_model()
        self.assertEqual(self.calls, [])

    def test_direct_ignores_router_url_and_router_key(self):
        self.elevenlabs()
        self.settings["base_url"] = "not-a-router-url"
        with patch.object(impl, "_api_key", side_effect=AssertionError("Direct read router key")):
            self.synthesize()
        self.assertTrue(self.calls[0][2].startswith("https://api.elevenlabs.io/"))

    def test_encoded_elevenlabs_gateway_voice_precedes_config_default(self):
        for model, expected in (
            ("el/VoiceEmbedded", "elevenlabs/VoiceEmbedded"),
            ("elevenlabs/VoiceEmbedded", "elevenlabs/VoiceEmbedded"),
            ("el/eleven_flash_v2_5/VoiceEmbedded", "elevenlabs/eleven_flash_v2_5/VoiceEmbedded"),
            ("eleven_flash_v2_5/VoiceEmbedded", "elevenlabs/eleven_flash_v2_5/VoiceEmbedded"),
        ):
            with self.subTest(model=model):
                self.settings = {"model": model, "voice": "VoiceDefault", "transport": "gateway"}
                self.calls.clear()
                self.responses = [response()]
                self.synthesize()
                self.assertEqual(self.calls[0][3]["json"]["model"], expected)

    def test_incomplete_elevenlabs_gateway_models_fail_without_request(self):
        for model in ("el/eleven_flash_v2_5", "eleven_flash_v2_5",
                      "elevenlabs/", "el/model/voice/extra", "el//voice"):
            with self.subTest(model=model):
                self.settings = {"model": model, "transport": "gateway"}
                with self.assertRaises(ValueError):
                    self.synthesize()
        self.assertEqual(self.calls, [])

    def test_catalog_elevenlabs_selection_uses_gateway_without_db(self):
        self.settings = {"voice": "Voice0009"}
        self.responses = [
            response(body={"data": [{"id": "el/eleven_flash_v2_5"}]}, content_type="application/json"),
            response(),
        ]
        with patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Catalog opened DB")):
            self.synthesize()
        self.assertEqual(self.calls[0][1], "GET")
        self.assertTrue(self.calls[1][2].endswith("/v1/audio/speech"))
        self.assertEqual(self.calls[1][3]["json"]["model"], "elevenlabs/eleven_flash_v2_5/Voice0009")

    def test_implicit_transport_notices_are_once_per_profile_and_safe(self):
        self.settings = {"model": "el/eleven_flash_v2_5", "voice": "PrivateVoice"}
        self.responses = [response(), response(), response()]
        with patch.object(impl.logger, "warning") as warning:
            with patch("hermes_constants.get_hermes_home", return_value=self.root / "profile-one"):
                self.synthesize("one.mp3")
                self.synthesize("two.mp3")
            with patch("hermes_constants.get_hermes_home", return_value=self.root / "profile-two"):
                self.synthesize("three.mp3")
        self.assertEqual(warning.call_count, 2)
        notice = " ".join(str(call) for call in warning.call_args_list)
        self.assertIn("gateway", notice)
        self.assertIn("elevenlabs-direct", notice)
        for private in ("PrivateVoice", "synthetic-router-key", str(self.root)):
            self.assertNotIn(private, notice)


    def test_gateway_fallback_preserves_encoded_voice_unless_caller_overrides(self):
        self.settings = {"transport": "gateway", "model": "el/eleven_flash_v2_5/VoicePrimary",
                         "fallback_models": ["el/eleven_turbo_v2_5/VoiceFallback"]}
        for override in (None, "VoiceCaller"):
            with self.subTest(override=override):
                self.calls.clear()
                self.responses = [response(429), response()]
                self.synthesize(voice=override)
                expected = [override or "VoicePrimary", override or "VoiceFallback"]
                self.assertEqual([call[3]["json"]["voice"] for call in self.calls], expected)
                self.assertEqual([call[3]["json"]["model"].split("/")[-1] for call in self.calls], expected)

    def test_direct_voice_only_model_preserves_existing_account_alias(self):
        self.elevenlabs()
        self.settings["model"] = "el/RawVoicePrivate"
        self.responses = [response(), response()]
        for explicit in (True, False):
            with self.subTest(explicit=explicit):
                if not explicit:
                    self.settings.pop("transport")
                self.synthesize("alias-%s.mp3" % explicit)
                key = self.calls[-1][3]["headers"]["xi-api-key"]
                expected = {"synthetic-key-one": "Voice0001", "synthetic-key-two": "Voice0002"}
                self.assertTrue(self.calls[-1][2].endswith("/" + expected[key]))

    def test_catalog_methods_validate_transport_before_network(self):
        for settings in ({"transport": "unknown"},
                         {"transport": "elevenlabs-direct", "model": "gemini/fixture"}):
            with self.subTest(settings=settings):
                self.settings = settings
                with patch.object(self.provider, "_fetch_models", side_effect=AssertionError("Invalid mode fetched catalog")):
                    with self.assertRaises(ValueError):
                        self.provider.list_models()
        self.assertEqual(self.calls, [])


    def test_malformed_direct_targets_fail_before_any_io(self):
        for model in ("el", "elevenlabs", "elevenlabs/", "el//Voice0009",
                      "el/eleven_flash_v2_5/", "el/model/voice/extra"):
            with self.subTest(model=model):
                self.settings = {"transport": "elevenlabs-direct", "model": model,
                                 "voice": "narrator", "db_path": "/synthetic/legacy.sqlite"}
                with patch.object(self.provider, "_read_db_keys", side_effect=AssertionError("Malformed target opened DB")), \
                     patch.object(self.provider, "_fetch_models", side_effect=AssertionError("Malformed target fetched catalog")), \
                     patch("sqlite3.connect", side_effect=AssertionError("Malformed target opened SQLite")):
                    with self.assertRaises(ValueError):
                        self.synthesize()
                self.assertEqual(self.calls, [])


BASELINE_CASES = [
    "test_default_voice_is_applied",
    "test_voice_is_encoded_in_router_model",
    "test_requested_format_is_forwarded",
    "test_scoped_key_miss_does_not_borrow_process_key",
    "test_db_open_is_readonly",
    "test_hard_error_does_not_fallback",
    "test_provider_error_body_is_private",
]

if __name__ == "__main__":
    if "--baseline" in sys.argv:
        selected = unittest.TestSuite(Regressions(name) for name in BASELINE_CASES)
        result = unittest.TextTestRunner(verbosity=2).run(selected)
        proven = len(result.failures) + len(result.errors)
        if result.testsRun != len(BASELINE_CASES) or proven != len(BASELINE_CASES):
            raise SystemExit("Baseline proof failed: expected every selected regression to fail")
        print(f"BASELINE PROVED: {proven} independently reproduced defects")
    else:
        unittest.main(verbosity=2)
