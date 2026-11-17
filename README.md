# Hermes 9Router TTS Plugin

Native TTS provider `9router-tts` for Hermes Agent. Version **0.1.2** targets
unmodified Hermes **0.23.2** and 9Router **0.5.119**. Select it through the
official TTS provider configuration; no Hermes core changes are needed.

## Install and manage

```bash
hermes plugins install dhansxd/hermes-9router-tts-plugin --enable --yes-deps
hermes plugins doctor 9router-tts --ci
hermes plugins update 9router-tts
hermes plugins disable 9router-tts
hermes plugins enable 9router-tts
hermes plugins remove 9router-tts
```

Use the official manager; no symlink or manual cleanup is needed. Removal leaves
your TTS configuration in place. Select another provider if you want TTS after
disabling or removing this one. Pip packages advertise the official
`hermes_agent.plugins` entry point.

## Gateway configuration (default for new configurations)

```yaml
tts:
  provider: 9router-tts
  output_format: mp3
  9router-tts:
    transport: gateway
    base_url: http://127.0.0.1:20128
    model: elevenlabs/eleven_flash_v2_5
    voice: your_gateway_voice_id
    timeout: 120
    fallback_models: []
```

All models, including ElevenLabs, go through `/v1/audio/speech` in gateway mode.
Only `NINEROUTER_KEY` from the active Hermes profile is used for gateway auth.
The plugin never reads local ElevenLabs keys or SQLite on this path.
9Router owns its provider credentials and connection rotation. Its connection
pool cannot express a different voice ID for each account through this request;
use the optional direct mode below when that mapping is needed.

`NINEROUTER_URL` supplies the URL when `base_url` is absent. Legacy
`tts.9router` settings are merged first; canonical `tts.9router-tts` values win,
including `transport`. No owner-specific key, voice or model is built in.
Other provider models work as before, for example
`gemini/gemini-2.5-flash-preview-tts` with `voice: Kore`.

Voice precedence is caller voice, then voice encoded in the model, then the
plugin's configured `voice` (or legacy `default_voice`). The caller's model
overrides the configured model. With no model, gateway mode uses the live
`/v1/models/tts` catalog. Discovery and availability checks stay offline.

ElevenLabs gateway inputs accept `el/model/voice`,
`elevenlabs/model/voice`, and voice-only `el/voice_id` or
`elevenlabs/voice_id`. A bare `eleven_*` model is normalized to
`elevenlabs/model/voice`; model-only forms require a caller/default voice.
An explicit caller voice replaces only the voice part. Empty or extra model
segments fail clearly. Gemini/OpenAI use `provider/model/voice`, OpenRouter
uses `openrouter/provider/model/voice`, and Edge/Google use `provider/voice`.
The separate voice field is also sent for compatible gateways; other adapters
keep their own model grammar.

Only explicit fallback models from the same provider are tried, on HTTP 429 or a
recognized quota response. Gateway failures never switch to direct mode.
Requests disable redirects and ambient proxy/netrc credentials.

The requested format is forwarded, but adapters may return another native format.
The plugin saves atomically with the actual format's extension and returns that
path. For example, Gemini WAV is saved as `.wav` even when MP3 was requested.
The 9Router ElevenLabs adapter emits MP3 and does not apply caller speed; gateway
mode does not promise direct-mode WAV/Opus or speed parity. Proxy speed is
forwarded only for endpoints that support it. Hermes owns voice-message
conversion. Streaming and plugin-side transcoding are not advertised.

## Optional explicit direct ElevenLabs rotation

```yaml
tts:
  provider: 9router-tts
  9router-tts:
    transport: elevenlabs-direct
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

Direct mode requires an explicit ElevenLabs model. A missing or non-ElevenLabs
model is rejected before any catalog, DB or provider request. It never uses the
gateway URL or `NINEROUTER_KEY` for synthesis.

Keys are read at synthesis time from the local 9Router SQLite database, default
`~/.9router/db/data.sqlite`. A named or routed Hermes profile must set its own
explicit `db_path` under `tts.9router-tts`. The reader uses read-only SQLite
with WAL visibility, creates no missing DB, modifies no 9Router data and caches
no credentials. Requests go to the fixed official ElevenLabs API.

Mappings retain each unique active connection name's matching voice ID.
A raw voice ID can also use the active pool. Missing/revoked keys, ambiguous
names and active quota/model locks are excluded. Rotation state is scoped to
the profile, DB, model and current mapping/key identities. Quota errors rotate
through eligible entries and cool failed entries for 60 seconds. Key-specific
authentication or voice-access errors try the next mapping; general input/model
and transport errors stop. Keys, voice IDs and raw error bodies stay out of logs.

Direct mode requests MP3, true WAV, or Ogg/Opus using ElevenLabs' query
`output_format`. FLAC falls back to MP3 with the correct suffix. Caller speed
is sent through `voice_settings.speed`.

## Upgrade from 0.1.1 and older

Older versions selected direct synthesis for every ElevenLabs model. Version
0.1.2 uses this deterministic rule; it does not rewrite your configuration:

| Configuration | Selected route |
| --- | --- |
| Explicit `transport: gateway` | Gateway, even with old local mappings/DB settings |
| Explicit `transport: elevenlabs-direct` and an ElevenLabs model | Direct |
| Omitted selector, ElevenLabs model, meaningful legacy direct marker | Direct, with a migration notice |
| Omitted selector, ElevenLabs model and voice only | Gateway, with a migration notice |
| Omitted selector and non-ElevenLabs model | Gateway |
| Unknown/empty/non-string selector, or explicit direct with missing/non-ElevenLabs model | Configuration error before I/O |

A meaningful legacy marker is a nonblank string `db_path`, a nonblank
`elevenlabs.strategy` or `elevenlabs.model_id`, or at least one
`elevenlabs.voices` entry containing nonblank string `connection` and
`voice_id` values. It applies only when the effective model is ElevenLabs.
Empty objects/lists, blank strings and malformed mappings are not markers.

A historical model+raw-voice configuration using the default DB is
indistinguishable from a fresh gateway configuration. **To retain its old direct
route, add `transport: elevenlabs-direct` before upgrading.** To intentionally
use 9Router, add `transport: gateway` and provide a gateway voice ID; local
per-account aliases are not expanded in gateway mode. Existing marker-based
direct configurations should also add the explicit selector to remove ambiguity.

Migration notices contain only fixed instructions, are emitted only on synthesis,
and are deduplicated in memory per profile and implicit route. No personal values
or persistent marker files are written.

## Verification

```bash
python test_provider.py
```

The existing GitHub Actions workflow runs synthetic HTTP/audio/SQLite regressions
and the actual pinned Hermes CLI/native TTS lifecycle on a hosted runner.
Git install, doctor, unload, disable/re-enable, pinned-update refusal, Git update
and remove use a temporary `HERMES_HOME`. Official Hermes source is checked for
modifications. No live synthesis, real keys, model downloads or voice-quality
judgment is part of these checks. Live gateway/audio testing is deferred.
