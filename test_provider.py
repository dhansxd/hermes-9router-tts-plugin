"""Offline regression checks. Run with Hermes Python: python test_provider.py."""
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('tts_plugin_install', root / '__init__.py', submodule_search_locations=[str(root)])
assert spec is not None and spec.loader is not None
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)
registered = []
mod.register(SimpleNamespace(register_tts_provider=registered.append))
assert len(registered) == 1 and registered[0].name == '9router-tts'
provider = registered[0]
impl = sys.modules[provider.__class__.__module__]

with tempfile.TemporaryDirectory() as tmp:
    output = str(Path(tmp) / 'speech.mp3')
    response = SimpleNamespace(status_code=200, content=b'test audio')
    with patch.object(impl, '_load_settings', return_value={'model': 'gemini/gemini-3.1-flash-tts-preview', 'voice': 'Kore'}), patch('requests.post', return_value=response) as post:
        assert provider.synthesize('hello', output, voice='Kore') == output
        payload = post.call_args.kwargs['json']
        assert payload['model'] == 'gemini/gemini-3.1-flash-tts-preview', payload
        assert payload.get('voice') == 'Kore', 'Requested voice was dropped'
        assert Path(output).read_bytes() == b'test audio'
print('ALL CHECKS PASSED')
