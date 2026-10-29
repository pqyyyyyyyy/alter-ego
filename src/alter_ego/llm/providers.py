"""LLM 抽象层。云端 API 与本地 llama.cpp 共用同一接口。

设计要点：
  · 统一走 OpenAI 兼容协议，换后端只改配置不改代码
  · 本地 llama.cpp 额外提供 completion() —— 需要 prefill 锁定输出格式时用，
    /v1/chat/completions 无法做 prefill（实测见 docs/实验数据.md）
  · 降级链：首选失败 → 备用 → 本地 → 预设回复。板子上网络不稳是常态。
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Iterator

import requests
import yaml


class ProviderChainError(RuntimeError):
    """Only locally constructed, credential-free failure summaries may enter here."""


def safe_failure(exc: Exception, elapsed: float) -> str:
    """Keep status/timing, never URLs, headers, response text or exception text."""
    response = getattr(exc, 'response', None)
    status = getattr(response, 'status_code', None)
    if isinstance(status, int) and 100 <= status <= 599:
        category = {401: 'authentication', 403: 'forbidden',
                    404: 'endpoint_or_model_not_found', 429: 'rate_limit_or_quota'}.get(
                        status, 'upstream_error' if status >= 500 else 'request_error')
        if status in (400, 422):
            # Inspect only to choose a fixed label; never retain upstream strings.
            try:
                payload = response.json()
                error = payload.get('error', {}) if isinstance(payload, dict) else {}
                message = error.get('message', '') if isinstance(error, dict) else ''
                param = error.get('param', '') if isinstance(error, dict) else ''
                if (isinstance(message, str) and 'response_format' in message.lower()
                        and any(word in message.lower() for word in
                                ('unsupported', 'not support', 'invalid', 'unknown', 'not allowed'))):
                    category = 'response_format unsupported'
                elif param == 'model':
                    category = 'model_parameter_error'
            except Exception:
                pass
        detail = 'HTTP{} {}'.format(status, category)
    elif isinstance(exc, requests.exceptions.Timeout):
        detail = 'request_timeout'
    elif isinstance(exc, requests.exceptions.SSLError):
        detail = 'tls_error'
    elif isinstance(exc, requests.exceptions.ProxyError):
        detail = 'proxy_error'
    elif isinstance(exc, requests.exceptions.ConnectionError):
        detail = 'connection_error'
    elif isinstance(exc, (ValueError, KeyError, IndexError, TypeError)):
        detail = 'response_format_error'
    else:
        detail = 'provider_error'
    return '{} elapsed={:.3f}s'.format(detail, max(0., elapsed))


@dataclass
class ProviderConfig:
    name: str
    base_url: str
    model: str
    api_key_env: str = ""
    timeout: int = 30
    # 本地 llama.cpp 的 /completion 裸接口地址，仅本地后端有
    completion_url: str = ""

    @property
    def api_key(self) -> str:
        return os.environ.get(self.api_key_env, "") if self.api_key_env else ""


class OpenAICompatProvider:
    """OpenAI 兼容后端。通义千问、DeepSeek、智谱、本地 llama.cpp 都适用。"""

    # 本地回环地址。走代理会被拦截 —— 实测开发机配了系统代理后，
    # 对 127.0.0.1 的请求被转发给代理，代理连不上本地服务，
    # 5 秒超时后返回 502 Bad Gateway，而非预期的连接拒绝。
    _LOCAL_HOSTS = ("127.0.0.1", "localhost", "::1", "0.0.0.0")

    def __init__(self, cfg: ProviderConfig):
        self.cfg = cfg
        self._session = requests.Session()
        if self._is_local(cfg.base_url) or self._is_local(cfg.completion_url):
            # trust_env=False 同时关掉代理与 .netrc 读取
            self._session.trust_env = False
            self._session.proxies = {"http": None, "https": None}

    @classmethod
    def _is_local(cls, url: str) -> bool:
        return bool(url) and any(h in url for h in cls._LOCAL_HOSTS)

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            h["Authorization"] = f"Bearer {self.cfg.api_key}"
        return h

    def chat(self, messages: list[dict[str, str]], **kw) -> str:
        body = {"model": self.cfg.model, "messages": messages, **kw}
        r = self._session.post(
            f"{self.cfg.base_url}/chat/completions",
            headers=self._headers(), json=body, timeout=self.cfg.timeout,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    def stream(self, messages: list[dict[str, str]], **kw) -> Iterator[str]:
        body = {"model": self.cfg.model, "messages": messages, "stream": True, **kw}
        with self._session.post(
            f"{self.cfg.base_url}/chat/completions",
            headers=self._headers(), json=body,
            timeout=self.cfg.timeout, stream=True,
        ) as r:
            r.raise_for_status()
            for raw in r.iter_lines():
                if not raw or not raw.startswith(b"data: "):
                    continue
                payload = raw[6:].strip()
                if payload == b"[DONE]":
                    return
                try:
                    delta = json.loads(payload)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                piece = delta.get("content")
                if piece:
                    yield piece

    def completion(self, prompt: str, **kw) -> str:
        """llama.cpp 的 /completion 裸接口。

        用途是 prefill —— 让 prompt 以「情绪：」结尾，模型只能从此续写，
        格式被物理锁死而非靠指令约束。实测把格式遵守率从 70% 提到 100%。
        仅本地后端支持；云端 provider 调用会抛 NotImplementedError。
        """
        if not self.cfg.completion_url:
            raise NotImplementedError(
                f"provider '{self.cfg.name}' 未配置 completion_url，"
                "prefill 仅本地 llama.cpp 支持"
            )
        body = {"prompt": prompt, "cache_prompt": True, **kw}
        r = self._session.post(
            self.cfg.completion_url, json=body, timeout=self.cfg.timeout
        )
        r.raise_for_status()
        return r.json().get("content", "")


class ProviderRegistry:
    """从 providers.yaml 加载，支持按顺序降级。"""

    def __init__(self, path: str = "providers.yaml"):
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        self._default: str = raw.get("default", "")
        self._fallbacks: list[str] = raw.get("fallback_chain", [])
        # 主对话专用链。未配置时由 dialog_chain 属性自动剔除本地后端。
        self._dialog_chain: list[str] = raw.get("dialog_chain", [])
        self._providers: dict[str, OpenAICompatProvider] = {}
        for name, d in (raw.get("providers") or {}).items():
            cfg = ProviderConfig(
                name=name,
                base_url=d["base_url"],
                model=d["model"],
                api_key_env=d.get("api_key_env", ""),
                timeout=d.get("timeout", 30),
                completion_url=d.get("completion_url", ""),
            )
            self._providers[name] = OpenAICompatProvider(cfg)

    def get(self, name: str | None = None) -> OpenAICompatProvider:
        key = name or self._default
        if key not in self._providers:
            raise KeyError(f"未配置的 provider: {key}")
        return self._providers[key]

    def close(self) -> None:
        for provider in self._providers.values():
            provider._session.close()

    @property
    def dialog_chain(self) -> list[str]:
        """主对话专用降级链。

        **必须与 fallback_chain 分开，且不能包含 local。**
        local 是 llama.cpp 跑 0.5B，它做不了主对话任务（结构化 JSON +
        日文敬体 + 中文翻译），但调用**不会报错** —— 它会返回看起来成功的垃圾，
        然后 JSON 解析失败、情绪兜底 neutral、日文变成乱码。
        这比干脆失败更糟：失败能走预设回复，垃圾会直接播出去。
        """
        return self._dialog_chain or [
            n for n in (self._fallbacks or [self._default])
            if not OpenAICompatProvider._is_local(self._providers[n].cfg.base_url)
        ]

    def chat_with_fallback(
        self,
        messages: list[dict[str, str]],
        preset: str = "",
        chain: list[str] | None = None,
        **kw,
    ) -> tuple[str, str]:
        """按降级链依次尝试。返回 (内容, 错误说明)。

        错误说明非空表示走了降级 —— **容错和可观测性是两件事**，
        只返回内容的话「全链失败后用了预设回复」和「首选成功」看起来一样。
        这个教训来自 10-04 的 502 事件：静默吞异常让代理拦截伪装成「模型效果差」。

        全链失败且有 preset 时返回 preset，没有 preset 才抛异常。
        """
        names = chain or self.dialog_chain
        errors: list[str] = []
        for index, name in enumerate(names, 1):
            if name not in self._providers:
                errors.append(f"provider#{index}: 未配置")
                continue
            started = time.perf_counter()
            try:
                out = self._providers[name].chat(messages, **kw)
            except Exception as exc:
                errors.append('provider#{}: {}'.format(index, safe_failure(
                    exc, time.perf_counter() - started)))
                continue
            return out, ("; ".join(errors) if errors else "")
        detail = "全部 provider 失败: " + "; ".join(errors)
        if preset:
            return preset, detail
        raise ProviderChainError(detail)
