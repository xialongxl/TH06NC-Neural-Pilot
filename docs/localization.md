# 本地化 / Localization

[English README](../README.md) · [中文 README](../README.zh-CN.md)

The web console uses the browser language on the first visit. The language selector in the top bar stores the choice for this browser and applies it to the current view. The CLI and legacy desktop console use the system language. Set `TH06AI_LANG` to `en` or `zh-CN` to select a language for a terminal and its child processes.

网页首次打开时采用浏览器语言。右上角语言选项会即时更新当前页面，并记住选择。命令行与旧桌面控制台采用系统语言，也可用 `TH06AI_LANG` 指定。

## Adding a message

1. Add a descriptive key to [messages.json](../th06ai/messages.json). Each value is a pair in the order Simplified Chinese, English. Keep numbered placeholders such as `{0}` in both languages. A translation may reorder the placeholders.
2. Use `t(key, ...values)` from `useI18n()` in a React component. Keep full phrases together so the translator can choose the word order. For example, use one message for a limit and its unit.
3. In Python, use `msg(key, ...values)` for session and worker messages that are persisted or sent in API v2. Use `ui_text` for terminal and legacy console labels. The existing saved text remains readable in Chinese and the browser translates known messages at display time, including messages from older runs.

Keep action IDs, modes, state keys, filenames, and protocol fields independent of the display language. Arbitrary OS or third-party errors and copied raw logs keep their original text. The localization layer can translate the project's surrounding message and preserve a path, number, or external error passed as an argument.

Use the frontend tests and `python tools/check_docs.py --update-index` after changes. The CLI help can be checked without a game session by running `python -m th06ai --help` in each language.
