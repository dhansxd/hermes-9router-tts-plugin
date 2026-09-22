# Hermes 9Router TTS Plugin

Hermes plugin for Text-to-Speech via 9Router. Supports:
- **ElevenLabs**: Direct API calls with per-key `voice_id` mapping and round-robin rotation.
- **Other Providers (Gemini, OpenAI, Edge, etc.)**: Proxies through 9Router's `/v1/audio/speech`.

## Installation

```bash
hermes plugins install https://github.com/dhansxd/hermes-9router-tts-plugin
```

## Configuration

Add to your `~/.hermes/config.yaml`:

```yaml
tts:
  provider: 9router-tts
  9router-tts:
    base_url: http://127.0.0.1:20128
    model: gemini/gemini-3.1-flash-tts-preview  # Default model
    fallback_models:
      - gemini/gemini-2.5-flash-preview-tts
    voice: rara                                 # Default voice alias
    elevenlabs:
      model_id: eleven_flash_v2_5
      voices:
        rara:
          # Automatically rotates between these connections
          - connection: <your_9router_connection_name_1>
            voice_id: <elevenlabs_voice_id_1>
          - connection: <your_9router_connection_name_2>
            voice_id: <elevenlabs_voice_id_2>
```

*(API keys for ElevenLabs are securely read directly from your local 9Router SQLite database).*
