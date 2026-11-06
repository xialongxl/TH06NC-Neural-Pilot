# Implementation pitfalls and troubleshooting

English · [简体中文](../gotchas.md)

These checks come from actual failures and current design constraints. An implementation can change when it preserves the relevant invariant and covers the known counterexample.

## Game identity and capture

- **Port the adapter, including evidence.** The executable hash, PE layout, RVAs, instruction signatures, pool strides, collision sizes, and menu state belong to one verified build. A hash-only edit can turn unrelated bytes into believable coordinates or rewards. Establish read-only fixtures and verify real readings before enabling a new build.
- **Expect empty pools and cross-tick reads.** Startup can contain several seconds with no bullets. The game can advance between reads. Record the crossing and handle actual read, shape, or numeric failures separately. A stale cached frame must retain its age.
- **Motion validity is local.** A gap beyond the 15-tick motion-estimation range makes speed unknown. Resource and survival accounting can still use the actual elapsed ticks. Preserve known flags even if numeric fields are zero.
- **Focus has two meanings.** `Frame.focused` is the player's slow-movement flag. `GameWindow.focused` is the foreground status.
- **Contact and attack targets use different filters.** The network encodes up to 32 prioritized attack targets. Contact comes from the independent 256-slot pool and its contact flags, including relevant targets that lack HP or were omitted from the network list.
- **Straight lasers have orientation and phases.** Use verified collision width, angle, and extent. Distinguish telegraph, damaging, and fading states. Rotation and growth require more than two center samples. A telegraph can inform planning before it qualifies as damage evidence.

Relevant tests include `test_resources.py`, `test_state_capture.py`, `test_full_bullet_observation.py`, `test_enemy_contact.py`, and `test_laser_geometry.py`.

## Scene, pause, and input

- **Confirm a pause from scene evidence.** A frozen counter may also mean dialogue, script freeze, transition, or lost focus. The genuine pause menu can appear at tick 0–5 and with the old script flag unset. Inspect `phase_evidence` and menu state. At a Practice selection page, verify the earlier confirmed difficulty and shot type before accepting a highlighted stage.
- **Suspend combat movement during dialogue.** `_advance_dialogue()` releases movement, Shift, and X, then handles confirmation. After dialogue, issue a fresh Z press edge so the controller and the system agree. Dialogue yields no ordinary combat transition.
- **Verify restart and recovery separately.** R is used in the supported pause context. Verify reset and subsequent progress. Menu animation or an old zero tick is insufficient. Gameplay uses arrow keys.
- **Track SendInput and game response separately.** A successful call reports enqueue. First matching direction bits and first visible movement are later observations. A held action can legitimately generate zero events. Send key changes in one ordered batch. Release and recover on focus loss.
- **The old action keeps moving the player during inference.** Simulate the old-key prefix, then count the new action's minimum hold from the post-send sample.
- **F7 and F8 are standalone.** They must work while the agent holds Shift. Keep edge detection lightweight and keep F8 independent of F7 and UI queues.

Relevant tests include `test_game_phases.py`, `test_training_controls.py`, `test_input_latency.py`, and `test_focus_hotkeys.py`.

## Planning and probabilities

- **Expose fallback as risk.** A bounded search can fail to find a robust route. `all_routes_risky` and `fallback` describe that result. They grant no robust label.
- **Check the entire modeled switch window.** A route safe at the nominal instant can collide at an earlier or later instant. An unusually wide sampled window is incomplete. Keep the Stage 5 switch fixture and never union a nominal-only safe direction back into the fallback.
- **Keep laser telegraphs in the forecast.** When they leave the body probe with zero options, v8 fallback can prioritize moves that clear confirmed damage and body contact across the complete switch window. The full forecast still ranks these moves. `planner_active_threat_priority` identifies the reranking in action logs, and the route stays marked as fallback.
- **Retain the full hazard set.** Reachability can exclude objects under the same finite geometric model. A nearest-64 limit or omitted lasers/enemy bodies changes the threat model. Dense-scene branching limits still preserve all current moves and applicable fallback-window checks.
- **Use the same joint distribution throughout PPO.** Planner bias, action masks, bomb priors, policy sampling, execution, and `evaluate_actions()` must agree. Post-sampling overrides to move toward center or force a bomb invalidate the saved probability.
- **Distinguish native collision and search.** A Rust `collision_backend` can mean only the sweep kernel. Check `search_backend` for the whole search. Retain the Python reference and parity tests.

Relevant tests include `test_planning.py`, `test_planner_timing_recovery.py`, `test_native_search.py`, `test_policy.py`, `test_bomb_policy.py`, and `test_stage5_upgrade.py`.

## Reward and sampling

- **Reward the observable outcome.** Graze and total score are diagnostic and contextual. Rewarding total score directly could reintroduce an incentive for deliberate grazing. Point-item gains still have their own signal. Investigate persistent wall camping or missed pickups using actual actions, rewards, risk, and snapshots.
- **Account for repeated pickups.** Diminishing reward, capped inventory, Power potential, target potential, and death settlement interact. A raw pickup multiplier can reward death and recollection. Inventory snapshots give net change. Simultaneous consumption and pickup may hide an event.
- **Confirm credible damage.** Compare HP within the same vulnerable phase. A phase reset, scripted loss, teleport, or disappearing Boss is insufficient evidence for a kill. Native full-clear and final kill criteria remain separate work.
- **Separate requested bombs from consumption.** The safe initial prior is `0.00005`, with a conditional `0.01` in strict survival rescue windows. Empty inventory or NMNB masks bombs. A prior adjusts the probability. The policy still samples the action. Keep ordinary joint entropy from driving an artificial 50% bomb rate.
- **A lightweight sample may be the only Miss.** Preserve `death_probe_tick` and `death_evidence` through respawn. First Miss ends the round.
- **Separate stage histories.** Reset invalid motion and camping history at relevant boundaries, and cut GAE as appropriate. Preserve real transitions, counters, and resources without inventing a death, restart, or clear.
- **Use the right clock.** The planned training duration is at most 60 minutes and excludes user pauses and controlled waits. PPO updates while the game is paused count toward it. At the target, the active stage can have up to ten more active minutes. Finish at the next confirmed stage, results, or the end of the current attempt. Native combat time is tracked separately. The default per-round cap is disabled.
- **Evaluate fixed weights independently.** Raw reward totals differ across reward versions. Report checkpoint, setup, goal, completed rounds, actual bombs, Misses, stages, and resources. A truncated survivor or partial evaluation has a different outcome from a clear.

Relevant tests include `test_reward_balance.py`, `test_power_progress.py`, `test_graze_principle.py`, `test_stage_awareness.py`, and `test_sampling_timing.py`.

## Persistence, authorization, and concurrency

- **Preserve learned state.** Migrations retain original checkpoint files, compatible weights, Adam history, and the learned v2 bomb head. Zero-initialize new fusion columns. A hash mismatch blocks load and overwrite. An unupdated random model is not a learned checkpoint.
- **Separate in-memory pause from disk resume.** F7 keeps the same process, round, and rollout. Loading a checkpoint starts a new session without game RAM or the whole interrupted rollout.
- **Keep Windows atomic replacement compatible with readers.** `files.read_json()` uses delete sharing so an open reader cannot block `os.replace()` on files such as `current.json`. Use bounded retries. GPU authorization comes from the shared lease.
- **A busy lease read does not refresh the GPU sample.** Reuse the last complete valid record with its original timestamp. A transient concurrent write and a truly expired record have different recovery paths.
- **Own data before detaching.** Rust needs owned snapshots before dropping the GIL. Held state and key batches stay serialized. Python and Rust locks protect their respective transaction boundaries.
- **CUDA Graph must see current weights.** Share parameter storage between update and graph replay. Keep values and log probabilities aligned with their actions. Limit Auto fallback to graph capture restrictions and surface device and memory errors.

## Safety, performance, and release

- **Keep emergency stop outside the worker.** The guard must continue to process F8, telemetry expiry, and owned Job termination if the worker or driver stalls. Model saving is for orderly finish. Urgent stop can lose unsaved work.
- **Interpret GPU telemetry in context.** The Torch allocator's 15% limit covers that allocator. The watchdog monitors whole-device VRAM, including other applications and context. A stopped fan can be normal at idle. Mark unsupported sensors unavailable and distinguish a supported reading disappearing mid-session.
- **Distinguish disk size from hot-path work.** `.cache` includes downloads, tests, and builds. It does not explain per-step delay by size alone. uv hard links can also make displayed file sizes misleading.
- **Measure same-action timing.** Compute end-to-end and stage percentiles from paired timestamps. Independent P95s are not additive. A GPU stream span can overlap host wait. Read-only profiling excludes live SendInput and full training callbacks.
- **Avoid machine-specific scheduling assumptions.** CPU topology and scheduling can affect a tail without establishing its root cause. Keep logical core IDs and system power policy out of defaults.
- **Keep the console local and explicit.** Preserve token, Origin, project identity, and model ID checks. Inspect port ownership before stopping a service. Refresh, disconnect, and subscription remount only affect display.
- **Review the staged files and history separately.** Ignore rules protect new additions, not already tracked files or published history. Keep runtime and private data local. [Release checklist](open-source-readiness.md)
- **Preserve license scope.** Standard MIT governs project-owned code and docs. Non-commercial research is encouraged, and rights-holder terms apply separately. [Disclaimer](../../DISCLAIMER.md)

For a new failure, start with the full current status reason, worker stack, actual scene, and nearest action or death ID. Separate direct observation from inference before changing behavior. [Test matrix](testing-and-performance.md)
