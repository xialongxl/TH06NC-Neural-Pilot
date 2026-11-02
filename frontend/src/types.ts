export interface Model {
  id: string; name: string; path: string; steps: number; updates: number;
  saved_at: string | null; usable: boolean; compatibility: string;
  format: number; planner_version?: number;
}
export interface Gpu {
  name?: string; temperature_c?: number; power_w?: number; memory_used?: number;
  memory_total?: number; utilization?: number; fan_percent?: number[]; memory_temperature_c?: number | null;
}
export interface Plan {
  enabled?: boolean; version?: number; feasible_roots?: number; elapsed_ms?: number;
  visible_enemy_bodies?: number; considered_enemy_bodies?: number; unknown_body_geometry?: number;
  all_routes_risky?: boolean; visible_bullets?: number;
}
export interface Worker {
  goal?: 'survival' | 'nmnb'; episode_bombs_used?: number;
  inference?: { engine?: string; total_ms?: number; p50_ms?: number; p95_ms?: number };
  stage_id?: number | null; stage_label?: string; stage_seconds?: number; stage_changes?: number;
  game_phase?: string; game_phase_evidence?: Record<string, unknown>;
  player_shots?: number | null; visible_lasers?: number;
  focus_pending?: boolean;
  input_latency?: InputLatency;
  training_seconds?: number; budget_seconds?: number; paused_seconds?: number; steps?: number;
  session_steps?: number; updates?: number; episodes?: number; survival_seconds?: number;
  user_paused?: boolean; phase?: string; message?: string; model_source?: string;
  checkpoint?: string; resources?: { power?: number; bombs?: number; lives?: number };
  bomb_requests?: number; bombs_used?: number; boss_hp?: number; boss_max_hp?: number;
  boss_damage?: number; power_gained?: number; point_items_gained?: number; loiter_seconds?: number;
  planning?: Plan; bullet_observation?: { near?: number; full?: number; visible?: number };
  policy_diagnostics?: Record<string, number>; lease_diagnostics?: Record<string, unknown>; error?: string;
}
export interface InputLatency {
  observation_to_send_ms?: number | null; observation_to_send_p95_ms?: number | null;
  policy_ms?: number | null; policy_p95_ms?: number | null;
  send_ms?: number | null; send_p95_ms?: number | null; input_events?: number | null;
  game_input_state?: 'pending' | 'held' | 'matched' | 'not_observed' | 'unavailable';
  game_input_match_ms?: number | null;
}
export interface Snapshot {
  shutdown?: { state: 'idle' | 'waiting' | 'stopping' | 'error' | 'closed'; message: string };
  game_probe?: { available: boolean; phase?: string; label: string; paused?: boolean; tick?: number;
    stage_id?: number | null; stage_label?: string;
    menu_selection?: number | null; menu_ready?: boolean; observed_at: string; message: string } | null;
  acceleration?: { available: boolean; api_version?: number | null;
    collision: 'python' | 'rust' | 'auto'; input: 'python' | 'rust' | 'auto' };
  hotkeys?: { method?: string; F7?: string; F8?: string; F7_count?: number; F8_count?: number;
    F7_dispatch_ms?: number; F8_dispatch_ms?: number; listener_error?: string; F7_error?: string; F8_error?: string };
  preview: boolean; active: boolean; busy: boolean; pause_requested: boolean; finish_requested: boolean;
  session: { state?: string; mode?: string; reason?: string; time?: string; gpu?: Gpu; worker?: Worker };
  selected_model_id: string | null; models: Model[]; run_dir: string | null;
  readiness: { ready: boolean; checks: { gpu: boolean; calibration: boolean; resources: boolean };
    backend: { device?: string; cuda_runtime?: string; torch?: string; forward_check?: boolean; backward_check?: boolean } };
  safety: { temperature_limit_c: number; poll_seconds: number; telemetry_timeout_seconds: number;
    vram_fraction_limit: number; worker_heartbeat_seconds: number };
  default_minutes: number; versions: { observation: number; reward: number; planner: number };
  gpu_history: { stamp: number | null; temperature_c: number; power_w?: number; utilization?: number }[];
  gpu_live: boolean; gpu_age_seconds?: number | null;
}
export interface Logs { run_dir: string | null; worker: string; launcher: string }
export type Page = 'train' | 'models' | 'diagnostics'
export type Mode = 'resume' | 'train' | 'play' | 'evaluate' | 'check' | 'monitor' | 'calibrate'
