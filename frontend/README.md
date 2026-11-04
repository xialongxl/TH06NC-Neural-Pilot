# Web console

English · [简体中文](README.zh-CN.md)

The React and TypeScript console displays session state and submits explicit operator commands. Python owns game capture, planning, PPO, and protection. The repository pins frontend dependencies in `package-lock.json`.

## Run and develop

After building the UI and launcher, double-click `LaunchAI.exe` in the project root. Python serves `frontend/dist` at [127.0.0.1:18765](http://127.0.0.1:18765). The default read-only preview uses port `18766`. The built console runs without a separate Node process.

Start the development API from the repository root, then use a second terminal in this directory.

```powershell
.\.venv\Scripts\python.exe -m th06ai web --no-browser
```

```powershell
npm ci
npm run dev
npm test
npm run build
```

Vite proxies `/api` to loopback port `18765`. Run `python deploy.py ui` from the repository root to install frontend dependencies and build. The build uses local assets. Refresh the production page after rebuilding.

## AI Battle View for OBS

In **Safety & Diagnostics**, enable **AI Battle View** and copy the Browser Source URL. Place that source beside the game capture in OBS. A 480 × 720 source is a starting size for the vertical layout. Enable **Shutdown source when not visible** if you want hidden scenes to disconnect. [OBS Browser Source](https://obsproject.com/kb/browser-source)

The view displays one sampled decision at a time, with up to ten updates per second. The outer border is the observed field and the dashed rectangle is the movement boundary. The inner arrows show focused movement and the outer arrows show unfocused movement. A white outline marks the submitted policy action. The dotted route is a prediction, and the label reports the observed input state separately. The battle view clears when the sample expires.

## API and rendering contract

| Endpoint | Purpose |
| --- | --- |
| `GET /api/bootstrap?state=false` | Fetch the console token and protocol. Omitting the query preserves the legacy initial-state response. |
| `GET /api/events?protocol=2&hidden=false` | Subscribe to the shared display stream. Protocol 1 remains available for legacy clients. |
| `GET /api/battle-view/events`, `POST /api/battle-view` | Separate versioned live snapshots and a token-protected enable switch. The regular stream carries only the switch state. |
| `GET /api/state` | Read the full diagnostic state. |
| `GET /api/logs` | Fetch a bounded log tail with ETag and 304 support. Hidden pages suspend log polling. |
| `POST /api/models/select` | Persist the selected checkpoint ID. |
| `POST /api/sessions` | Explicitly start a named session mode. |
| `POST /api/session/pause` | Pause or resume the current session using `paused`. |
| `POST /api/session/finish` | Save and finish training. |
| `POST /api/session/stop` | Ask the independent guard to stop the owned session. |
| `POST /api/game/focus`, `POST /api/session/focus` | Activate the configured game or ask the active worker to retry foreground activation. |

Mutations use a same-origin `X-Console-Token` and validated requests. The service is designed for loopback. F7/F8 belong to the protection process, so the browser sends only the explicit operations selected in the UI.

Every v2 connection starts with `snapshot`. Subsequent `delta` events include a base and target revision, changes and removals, and an append cursor for up to 180 temperature points. The client reconnects when it loses its baseline. Slow subscribers converge on the latest state. Visible subscriptions update at most 2 Hz with an idle heartbeat every three seconds. When every subscriber is hidden, the shared rate falls to 0.2 Hz. Sampling stops when the last subscriber leaves. GPU protection continues to poll independently every 250 ms.

## UI behavior

Session settings use `game_setup`. The console displays observed identity separately from `game_settings`. The practice-stage highlight is a menu selection. Unknown data remains unknown. Archived GPU samples are labeled as historical, and disconnected state loses its live marker. The current session survives refreshes and browser closure. [Game setup](../docs/en/game-settings.md)

Latency diagnostics separate observation-to-send time, policy work, input-call time, and the first game-input match observed by the reader. P95 uses at most 64 recent valid samples. A held-key action can have no new send event. Detailed timestamps remain in raw diagnostics.

Components preserve unchanged state references to avoid redundant renders. Use full phrases from the shared translation catalog. The top-bar language selector is remembered locally and updates the current view. [Localization guide](../docs/en/localization.md)
