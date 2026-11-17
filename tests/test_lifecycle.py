"""Real Hermes lifecycle checks for a disposable GitHub-hosted runner.

TTS_CLOUD_TESTS=1 PLUGIN_REVISION=<40-char commit> python tests/test_lifecycle.py
The pinned, unmodified official Hermes runtime must be on PYTHONPATH.
All TTS HTTP is mocked; the audio fixture is generated locally on the runner.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import wave
from unittest.mock import patch

import requests


def run(*args, ok=True, cwd=None):
    result = subprocess.run(
        [str(arg) for arg in args], cwd=cwd, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=240,
    )
    print(result.stdout)
    if ok:
        assert result.returncode == 0, (args, result.returncode)
    return result


def main():
    assert os.environ.get("TTS_CLOUD_TESTS") == "1", "Use a disposable cloud runner"
    source = Path(__file__).resolve().parents[1]
    revision = os.environ.get("PLUGIN_REVISION") or run("git", "-C", source, "rev-parse", "HEAD").stdout.strip()
    assert len(revision) == 40
    with tempfile.TemporaryDirectory(prefix="tts-lifecycle-") as tmp:
        root = Path(tmp)
        home = root / "profile"
        home.mkdir()
        # Isolate the Hermes profile before importing host modules.
        # Native proxy synthesis uses no local 9Router database.
        os.environ["HERMES_HOME"] = str(home)
        for key in ("NINEROUTER_KEY", "NINEROUTER_URL", "ELEVENLABS_API_KEY"):
            os.environ.pop(key, None)

        from hermes_cli.config import load_config, save_config
        from hermes_cli.plugins import get_plugin_manager
        from agent.tts_registry import get_provider

        def cli(*args, ok=True):
            return run(sys.executable, "-m", "hermes_cli.main", "plugins", *args, ok=ok)

        def head(path):
            return run("git", "-C", path, "rev-parse", "HEAD").stdout.strip()

        def discover(present):
            with patch("requests.sessions.Session.request", side_effect=AssertionError("Discovery contacted network")):
                manager.discover_and_load(force=True)
                provider = get_provider("9router-tts")
                assert (provider is not None) == present
                if provider is not None:
                    assert provider.is_available()
            return provider

        cli("install", "dhansxd/hermes-9router-tts-plugin", "--ref", revision, "--enable", "--yes-deps")
        installed = home / "plugins" / "9router-tts"
        assert (installed / "plugin.yaml").is_file(), list((home / "plugins").iterdir())
        assert head(installed) == revision
        cli("doctor", "9router-tts", "--ci")
        manager = get_plugin_manager()
        discover(True)
        assert manager.unload("9router-tts")
        assert get_provider("9router-tts") is None
        discover(True)

        cfg = load_config()
        cfg["tts"] = {
            "provider": "9router-tts",
            "model": "gemini/fixture-model",
            "voice": "fixture-voice",
            "speed": 1.25,
            "output_format": "wav",
            "9router-tts": {
                "base_url": "http://gateway.invalid",
                "model": "gemini/scoped-default",
                "voice": "scoped-default",
            },
        }
        save_config(cfg)
        fixture = root / "fixture.wav"
        with wave.open(str(fixture), "wb") as audio:
            audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            audio.writeframes(bytes(2 * 1920))
        audio_bytes = fixture.read_bytes()
        assert audio_bytes.startswith(b"RIFF")
        output = root / "native.wav"
        calls = []

        def transport(session, method, url, **kwargs):
            assert method.upper() == "POST", (method, url)
            assert url == "http://gateway.invalid/v1/audio/speech", url
            assert session.trust_env is False
            assert kwargs["allow_redirects"] is False
            payload = kwargs["json"]
            assert payload["model"] == "gemini/fixture-model/fixture-voice", payload
            assert payload["input"] == "Native cloud speech", payload
            assert payload["voice"] == "fixture-voice", payload
            assert payload["speed"] == 1.25, payload
            assert payload["response_format"] == "wav", payload
            assert "Authorization" not in kwargs.get("headers", {})
            calls.append(url)
            response = requests.Response()
            response.status_code = 200
            response.headers["Content-Type"] = "audio/wav"
            response._content = audio_bytes
            response._content_consumed = True
            return response

        from tools.tts_tool import text_to_speech_tool, check_tts_requirements
        with patch("requests.sessions.Session.request", new=transport):
            assert check_tts_requirements()
            result = json.loads(text_to_speech_tool("Native cloud speech", str(output)))
        assert result["success"] and result["provider"] == "9router-tts", result
        assert result["voice_compatible"] is True, result
        written = Path(result["file_path"])
        assert written.is_file(), result
        if written.suffix == ".wav":
            assert written == output and written.read_bytes() == audio_bytes
        else:
            # Hermes may convert voice-compatible output when ffmpeg is present.
            assert written.suffix == ".ogg" and written.read_bytes().startswith(b"OggS"), result
        assert result["chunk_count"] == 1 and result["delivery_file_count"] == 1, result
        assert len(calls) == 1, calls

        cli("disable", "9router-tts")
        discover(False)
        cli("enable", "9router-tts")
        discover(True)
        pinned = cli("update", "9router-tts", ok=False)
        assert pinned.returncode != 0 and "pinned" in pinned.stdout.lower(), pinned.stdout
        assert head(installed) == revision
        cli("remove", "9router-tts")
        discover(False)
        assert not installed.exists()

        # Only this runner's disposable origin moves. This exercises the actual
        # official Git updater without changing the public main branch.
        origin = root / "origin.git"
        run("git", "clone", "--bare", source, origin)
        run("git", "--git-dir", origin, "update-ref", "refs/heads/ci", revision)
        run("git", "--git-dir", origin, "symbolic-ref", "HEAD", "refs/heads/ci")
        cli("install", origin.as_uri(), "--enable", "--yes-deps")
        assert head(installed) == revision
        discover(True)
        writer = root / "writer"
        run("git", "clone", origin.as_uri(), writer)
        with (writer / "README.md").open("a") as readme:
            readme.write("\n")
        run("git", "-C", writer, "add", "README.md")
        run("git", "-C", writer, "-c", "user.name=TTS CI", "-c",
            "user.email=ci@example.invalid", "commit", "-m", "Cloud lifecycle update fixture")
        updated = head(writer)
        assert updated != revision
        run("git", "-C", writer, "push", "origin", "HEAD:ci")
        cli("update", "9router-tts")
        assert head(installed) == updated
        discover(True)
        cli("remove", "9router-tts")
        discover(False)
        assert not installed.exists()
        plugins = (load_config() or {}).get("plugins") or {}
        aliases = {"9router-tts", "hermes-9router-tts-plugin"}
        for key in ("enabled", "disabled"):
            assert not aliases.intersection(plugins.get(key) or []), plugins

    print("REAL HERMES LIFECYCLE PASSED: Git install, doctor, native TTS, unload, disable, enable, pinned refusal, update, remove")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
