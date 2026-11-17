# Hermes 9Router TTS Plugin

Native TTS provider `9router-tts` for Hermes Agent. Uses a local or remote
9Router gateway, or explicitly configured local ElevenLabs connection mappings.
Version 0.1.1 targets unmodified Hermes **0.23.2** and 9Router **0.5.119**.

## Install and manage

```bash
hermes plugins install dhansxd/hermes-9router-tts-plugin --enable --yes-deps
hermes plugins doctor 9router-tts --ci
hermes plugins update 9router-tts
hermes plugins disable 9router-tts
hermes plugins enable 9router-tts
hermes plugins remove 9router-tts
```

Use the official manager; no core patch, symlink or manual cleanup is needed.
Removal leaves your TTS configuration in place, as the official manager intends.
Select another provider if you want TTS after disabling or removing this one.
Pip packages also advertise the official `hermes_agent.plugins` entry point.

## Gateway configuration

```yaml
tts:
  provider: 9router-tts
  output_format: mp3
  9router-tts:
    base_url: http://127.0.0.1:20128
    model: gemini/gemini-2.5-flash-preview-tts
    voice: Kore
    timeout: 120
    fallback_models: []
```

Set `NINEROUTER_KEY` in the active Hermes profile if your gateway requires auth.
`NINEROUTER_URL` supplies the URL when `base_url` is absent. Configuration under
legacy `tts.9router` is still accepted; canonical `tts.9router-tts` values win.
No model, voice, key or endpoint specific to the repository owner is built in.

Caller model/voice settings take priority. A voice encoded in a model is preserved
ahead of the plugin's default voice. Without a configured model, the live
`/v1/models/tts` catalog supplies one. Discovery and availability checks stay offline.
For 9Router 0.5.119, Gemini/OpenAI voices are encoded as
`provider/model/voice`, OpenRouter as `openrouter/provider/model/voice`, and
Edge/Google voices as `provider/voice`. The separate voice field is also sent
for compatible gateways; other adapters keep their own model grammar.

Only explicitly listed fallback models from the same provider are attempted, and
only after HTTP 429 or a recognized quota response. Authentication, permission,
transport and malformed-response failures stop the request. Requests disable
redirects and ambient proxy/netrc credentials; configure the gateway URL directly.

Requested output format is forwarded, but 9Router adapters may return a different
native format. The plugin checks response metadata, saves atomically with the
correct extension and returns that path. For example, Gemini WAV is saved as
`.wav` even when MP3 was requested. Hermes owns voice-message conversion to Opus.
Proxy speed is forwarded for compatible endpoints; 9Router 0.5.119 does not apply
it. No streaming synthesis or plugin-side transcoding is advertised.

## Optional direct ElevenLabs rotation

Select an ElevenLabs model to use this path:

```yaml
tts:
  provider: 9router-tts
  9router-tts:
    base_url: http://127.0.0.1:20128
    model: elevenlabs/eleven_flash_v2_5
    voice: narrator
    elevenlabs:
      strategy: round-robin  # or fill-first (default)
      voices:
        narrator:
          - connection: your_9router_connection_name_1
            voice_id: your_elevenlabs_voice_id_1
          - connection: your_9router_connection_name_2
            voice_id: your_elevenlabs_voice_id_2
```

Keys are read only when synthesizing from the local 9Router SQLite database,
default `~/.9router/db/data.sqlite`. A named or routed Hermes profile must set
its own explicit `db_path` under
`tts.9router-tts`; it never implicitly borrows the machine's key pool.
The reader uses SQLite read-only mode with WAL visibility, creates no missing DB,
does not modify 9Router, and does not cache credentials. Direct requests always go
to the official ElevenLabs API; the gateway URL does not receive these keys.

Mappings use unique active connection names and their corresponding voice IDs.
A raw voice ID can also select the active pool. Missing keys, disabled connections,
ambiguous names and active DB quota/model locks are ineligible. Rotation state
is scoped to the profile, DB, model and current mapping/key identities. Quota
failures rotate once through eligible entries and cool failed entries for 60 seconds.
Other errors stop. Credentials, voice IDs and provider error bodies are omitted
from logs and exceptions.

ElevenLabs output format is a query parameter: MP3, true WAV, or Ogg/Opus.
FLAC falls back to MP3 with its actual suffix. Caller speed is sent through
`voice_settings.speed`. Both `el/model/voice` and
`elevenlabs/model/voice` model spellings are accepted.

## Cloud verification

```bash
python test_provider.py
```

GitHub Actions runs the baseline's original test, proves original defects with
regressions, checks the fixed provider with synthetic HTTP/audio/SQLite fixtures,
and exercises the actual pinned Hermes CLI and native TTS dispatcher.
Git install, doctor, unload, disable, re-enable, pinned update refusal, unpinned
Git update and remove run with temporary homes on a hosted runner.
No live provider synthesis, real keys, model downloads or audio-quality judgment
are part of these checks. Official Hermes source is checked for modifications.
