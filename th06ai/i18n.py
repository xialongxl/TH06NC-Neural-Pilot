"""Message catalog and display translation for the CLI and legacy console."""
from __future__ import annotations

import json
import locale
import os
import re
from functools import lru_cache
from pathlib import Path

_SLOT = re.compile(r"\{(\d+)\}")


@lru_cache(maxsize=1)
def _catalog():
    return json.loads(Path(__file__).with_name("messages.json").read_text(encoding="utf-8"))


def msg(key: str, *values: object) -> str:
    """Render the canonical text used in persisted run state and API v2."""
    return _format(_catalog()[key][0], values)


def _format(template, values):
    return _SLOT.sub(lambda match: str(values[int(match[1])]), template) if values else template


def language() -> str:
    requested = os.environ.get("TH06AI_LANG") or locale.getlocale()[0] or "en"
    normalized = requested.casefold().replace("_", "-")
    return "zh-CN" if normalized == "zh" or normalized.startswith(("zh-", "chinese")) else "en"


def ui_text(key: str, *values: object) -> str:
    return _format(_catalog()[key][int(language() == "en")], values)


@lru_cache(maxsize=1)
def _legacy_messages():
    exact, templates = {}, []
    for key, (zh, en) in _catalog().items():
        if not key.startswith("backend."):
            continue
        parts = _SLOT.split(zh)
        if len(parts) == 1:
            exact[zh] = en
        else:
            pattern = "".join(re.escape(part) if i % 2 == 0 else f"(?P<p{part}>.*?)"
                              for i, part in enumerate(parts))
            templates.append((sum(len(part) for part in parts[::2]), re.compile(pattern, re.S), en))
    templates.sort(key=lambda entry: entry[0], reverse=True)
    return exact, templates


def localized(text: str | None, depth: int = 0) -> str:
    """Translate known saved messages while keeping OS errors and paths intact."""
    if text is None:
        return ""
    if language() != "en" or depth >= 4 or text.isascii():
        return text
    exact, templates = _legacy_messages()
    if text in exact:
        return exact[text]
    for _, pattern, english in templates:
        match = pattern.fullmatch(text)
        if match:
            return _SLOT.sub(lambda slot: localized(match[f"p{slot[1]}"], depth + 1), english)
    return text
