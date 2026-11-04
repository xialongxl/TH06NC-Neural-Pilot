# Contributing

English · [简体中文](CONTRIBUTING.zh-CN.md)

Reports, reproducible measurements, documentation, and code are welcome. Contributions to TH06NC-Neural-Pilot are submitted under the repository's [MIT license](LICENSE). Preserve upstream notices and submit only material you have permission to share. The project currently uses no separate contributor agreement.

Read the [development guide](docs/en/development.md) and [implementation pitfalls](docs/en/gotchas.md) before changing behavior. Use the project's [Issues](https://github.com/xialongxl/TH06NC-Neural-Pilot/issues) and [Pull Requests](https://github.com/xialongxl/TH06NC-Neural-Pilot/pulls).

## Report a bug

Give the smallest reproducible sequence from a known initial state, along with the expected and observed results. Include the details that apply.

- The project commit, Windows and Python versions, GPU if relevant, and whether the native extension was in use.
- The Steam build and executable SHA-256, difficulty, shot type, stage, and game scene.
- The operation you were running, such as training, inference, evaluation, calibration, or read-only profiling.
- The relevant status or result reason, a worker stack trace, and action or death IDs when behavior is involved.
- For timing reports, the scope and endpoints of the measurement, workload, sample count, and P50/P95/P99. Identify whether real input was sent.

Share a minimal sanitized extract. Remove personal paths, account names, GPU UUIDs, console tokens, and unrelated information. Keep game files, model archives, calibration records, CUDA wheels, dependency trees, and entire run directories out of reports. Discuss a minimal fixture in the Issue if one is needed.

## Propose a change

Describe the user problem, expected interaction, and acceptance criteria. Give a scenario or metric another developer could reproduce. Separate changes to rewards, perception, timing, and presentation so their effects can be measured independently.

For suspected token exposure, arbitrary command or path execution, or personal data leaks, use the host's private reporting feature if available. Otherwise, open a brief public request for a private channel and reserve the sensitive details for that channel.

## Open a pull request

1. Fork the repository and branch from `main`.
2. Reproduce the issue and keep the change focused. Follow the existing Python types and dataclasses, TypeScript components and state reconciliation, and Rust ownership boundaries.
3. Add regression coverage for the failure or contract your change affects. Use the [test matrix](docs/en/testing-and-performance.md) to choose the right checks.
4. Update the affected contracts and both language editions. Rebuild the generated symbol index after source changes.
5. Describe the trigger, resulting behavior, compatibility and migration impact, and actual test results. Include any unverified scenarios. Sanitize screenshots and preserve upstream attribution.

```powershell
git switch -c fix/describe-the-problem
.\.venv\Scripts\python.exe tools\check_docs.py --update-index
.\.venv\Scripts\python.exe tools\check_source.py
```

| Change area | Minimum review |
| --- | --- |
| Documentation or sample configuration | Links, portable paths, examples, and the docs checker |
| UI or SSE | Frontend tests and build, `test_console_stream.py`, `test_web_console.py` |
| Capture or phases | Empty pools, cross-tick reads, phases, enemies, and lasers |
| Planning or Rust | Python parity, switch-window fallback, full pools, shapes, and concurrency |
| Policy or rewards | Executed action and log probability, conditional bomb prior, resource loops, and death boundaries |
| Models or saves | Original files and optimizer history, hashes, unsaved runs, and partial evaluations |
| Safety or lifecycle | Fault injection, F7/F8, lease expiry, owned Job, shutdown, and late startup |

A running worker keeps the code it already loaded. Save and finish before starting a session with new code. F8 remains the emergency path. Live training, calibration, and game input require the operator's explicit choice. AI-assisted contributions are welcome when the contributor verifies the result and reports concrete evidence.
