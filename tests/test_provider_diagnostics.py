"""Exercise the real registry/orchestrator boundary without network or real keys."""
import sys
from pathlib import Path

import pytest
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from alter_ego.llm.providers import ProviderChainError, ProviderRegistry
from test_dialog import make_orch


SECRET = 'synthetic-private-marker-NEVER-PRINT'


class ScriptedProvider:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(kwargs)
        value = next(self.outcomes)
        if isinstance(value, Exception):
            raise value
        return value


def registry(provider):
    result = ProviderRegistry.__new__(ProviderRegistry)
    result._providers = {'openai': provider}
    result._dialog_chain = ['openai']
    return result


def http_error(status, message='authentication failed', param=None):
    import json
    response = requests.Response()
    response.status_code = status
    response.url = 'https://example.invalid/chat?key=' + SECRET
    response._content = json.dumps({'error': {'message': message + SECRET,
                                             'param': param}}).encode()
    return requests.HTTPError('request headers and URL ' + SECRET, response=response)


@pytest.mark.parametrize('error,label', [
    (http_error(401), 'HTTP401 authentication'),
    (http_error(403), 'HTTP403 forbidden'),
    (http_error(404), 'HTTP404 endpoint_or_model_not_found'),
    (http_error(429), 'HTTP429 rate_limit_or_quota'),
    (http_error(502), 'HTTP502 upstream_error'),
    (http_error(400, param='model'), 'HTTP400 model_parameter_error'),
    (requests.ReadTimeout(SECRET), 'request_timeout'),
    (requests.exceptions.SSLError(SECRET), 'tls_error'),
    (requests.exceptions.ProxyError(SECRET), 'proxy_error'),
    (requests.ConnectionError(SECRET), 'connection_error'),
    (KeyError(SECRET), 'response_format_error'),
    (RuntimeError(SECRET), 'provider_error'),
])
def test_failure_reaches_dialog_and_logs_without_credentials(tmp_path, caplog, error, label):
    provider = ScriptedProvider([error])
    orchestrator = make_orch(tmp_path, registry(provider))
    turn = orchestrator.reply('你好')
    assert label in turn.error
    assert 'elapsed=' in turn.error
    assert 'provider#1' in turn.error
    assert SECRET not in turn.error + caplog.text
    assert orchestrator.history_turns == 0
    assert len(provider.calls) == 1


def test_format_incompatibility_still_retries_once_with_safe_note(tmp_path, caplog):
    provider = ScriptedProvider([
        http_error(400, 'response_format unsupported '),
        '{"ja":"はい。","zh":"好的。","emotion":"neutral"}',
    ])
    turn = make_orch(tmp_path, registry(provider)).reply('你好')
    assert len(provider.calls) == 2
    assert 'response_format' in provider.calls[0]
    assert 'response_format' not in provider.calls[1]
    assert turn.subtitle_zh == '好的。'
    assert SECRET not in turn.error + caplog.text


def test_successful_fallback_keeps_sanitized_first_failure():
    result = registry(ScriptedProvider([http_error(401)]))
    result._providers['backup'] = ScriptedProvider(['valid-output'])
    raw, note = result.chat_with_fallback([], chain=['openai', 'backup'])
    assert raw == 'valid-output'
    assert 'HTTP401' in note
    assert SECRET not in note


def test_registry_exception_and_preset_are_safe():
    result = registry(ScriptedProvider([requests.ReadTimeout(SECRET)]))
    with pytest.raises(ProviderChainError) as caught:
        result.chat_with_fallback([])
    assert 'request_timeout' in str(caught.value)
    assert SECRET not in str(caught.value)
    result = registry(ScriptedProvider([requests.ReadTimeout(SECRET)]))
    raw, note = result.chat_with_fallback([], preset='preset')
    assert raw == 'preset'
    assert SECRET not in note
