# Localization

English · [简体中文](../localization.md)

The web console starts in the browser's preferred supported language. Its top-bar selector applies immediately and stores the choice in that browser. The CLI and legacy desktop console start in the system language. Set `TH06AI_LANG=en` or `TH06AI_LANG=zh-CN` in the launching shell to choose a language for the process and its children.

## Add or revise messages

1. Add a descriptive key to the [shared catalog](../../th06ai/messages.json). Each entry contains Simplified Chinese followed by English. Keep the same numbered placeholders in both translations. Translators can reorder placeholders to fit the sentence.
2. In React, call `t(key, ...values)` from `useI18n()`. Use one complete phrase for a value and its unit or for a sentence containing user-provided details.
3. In Python, use `msg(key, ...values)` for messages persisted or sent through API v2. This preserves canonical session text and allows known new and archived messages to be localized at display time. Use `ui_text` for terminal and legacy console labels.
4. Keep protocol keys, action IDs, mode names, model filenames, and paths language-independent. Known project messages can wrap external errors without changing their content. The raw-log copy operation keeps the original text.

Run the frontend tests and exercise `python -m th06ai --help` in both languages. CLI help can be checked offline. The [frontend guide](../../frontend/README.md) lists the build and test commands.

## Update the documentation

The project root and component entry pages default to English. The English technical guides live under `docs/en/`. The existing Chinese guides retain their paths under `docs/`. Keep links in the reader's current language and add a reciprocal link at the top of every pair. Update both editions when behavior, commands, defaults, or constraints change. Verify local links with `python tools/check_docs.py`.
