# Difficulty, shot type, and Stage Practice

English · [简体中文](../game-settings.md)

The operator selects the game setup before starting training, resuming a checkpoint, playing a fixed policy, or evaluating it. Choices apply to the next session.

| Mode | Difficulty and starting point |
| --- | --- |
| Story | Easy, Normal, Hard, or Lunatic from Stage 1 |
| Extra | Select Extra in Story setup to use the separate in-game entry |
| Stage Practice | Easy through Lunatic, with an unlocked stage for the selected difficulty and shot type |
| Shot type | Reimu A/B or Marisa A/B |

The current game caps Easy Practice at Stage 5. Extra and its shot type must be unlocked in the game. The agent reads unlocks and navigates with ordinary keys. If the selected combination is locked, the menu is unknown, or the actual game settings differ, the worker can wait for the operator to correct it. Unknown values stay unknown.

A real stage pause at tick 0–5 is a valid attachment point. If you start at the Practice stage selection, the worker checks the earlier confirmed difficulty and shot type and navigates back to fix mismatches. Story advances with the game. If the stage number briefly disappears between stages, the worker releases input and waits for the native stage to return. It updates the console and sampling context once it has a confirmed stage. The transition produces no combat samples. A recognized menu, a real pause, or a stage that remains unconfirmed requires operator attention. Practice re-enters the selected stage after results or a return to the native Practice stage menu. First-run calibration still applies. [First-run calibration](development.md#first-run-calibration)

## Configure defaults or run from the CLI

Merge the following into your local config. The browser keeps its own selections and submits them for the session without changing the file. [Full configuration example](../../config.example.json)

```json
{
  "game_setup": {
    "mode": "practice",
    "difficulty": "hard",
    "character": "marisa_b",
    "stage": 5
  }
}
```

This command sends live game input and trains when the operator is ready.

```powershell
.\.venv\Scripts\python.exe -m th06ai train --minutes 5 --game-mode practice --difficulty hard --character marisa_b --stage 5
```

Without a setup object or CLI override, the legacy attachment path can resume a started game or use current menu selections. Web launches submit a validated `game_setup` with `mode`, `difficulty`, `character`, and `stage`. The same request is carried through the CLI, guard, and worker.

## Requested setup and observed state

`requested_game_setup` records the launch request. `game_settings` carries observed information. `source=game` describes a running stage, and `source=menu` describes a menu selection. Keep a highlighted stage separate from the active combat stage. Live motion uses validated speed reads for the active shot type when available. Unreadable speed remains unknown.

| Evidence in the supported executable | Location or encoding |
| --- | --- |
| Difficulty | RVA `0x53D410`, int32 0–4, confirmed by code at `0x4CAEE` |
| Character and shot | `0x53CAD0/0x53CAD1`, each a 0/1 byte, with confirmation at `0x4CD28/0x4D32A` |
| Practice | `0x53D404`, set from the Practice flag at `0x4F34F` |
| Challenge and Spell Practice | `0x53D414` and `0x53D405` distinguish scenes outside this selection feature |
| Stage selection | `NC_MENU+0x168B0 == 22`. The zero-based highlight is at `+0x28` |
| Unlocked stage count | `0x549D55 + (character*2+shot)*24 + difficulty`, initialized at `0x57020` |
| Move speed | Player offsets `+0x7860/+0x7864` and `+0x7710/+0x7714`, with read instructions at `0x6A861/0x6A854` |

These offsets and signatures are evidence for the supported executable. Revalidate the layout before adapting another build.

## Models and result boundaries

Observation v9 appends 19 inputs for difficulty, shot, mode, speed, and known flags. The policy uses observed values. Migration from v8 preserves the existing fusion columns, policy/value heads, and Adam state. New columns and their optimizer history start at zero. Historical v1–v7 models follow the earlier compatibility path, and source files are retained.

The native Practice result ends the episode as `practice_result`. If the sampler misses the result flag, it can still end a previously confirmed Practice round when it sees the native Practice stage menu. This is recorded as `practice_return` and does not verify a stage clear. A detected Miss is still recorded as `death`. These cases use `SamplingInterrupted.episode_end` to cut GAE and reset. Menu input produces no combat samples. Re-entry still checks the session's difficulty, shot type, stage, and unlocks. Setup changes also separate timing history.

Offline verification covers selection, unlock limits, observed/requested separation, native speed, API validation, migration, and the result boundary. The recorded 2026-11-03 live check was read-only on a Practice stage-selection screen with Normal, Reimu A, Stage 1 highlighted, and six stages available. End-to-end verification of automatic navigation, every combination, and re-entry after results remains separate.
