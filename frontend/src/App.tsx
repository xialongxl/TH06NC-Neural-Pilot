import { message, translateLog, useI18n } from './i18n'
import type { Locale } from './i18n'
import { memo, useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, Box, Check, CheckCheck,
  ChevronDown, ChevronRight, Circle, Clock3, Copy, Cpu, Crosshair, Database, Ellipsis,
  Eye, FileText, FolderOpen, Gauge, Heart, Info, LayoutDashboard,
  LoaderCircle, Menu, Moon, Pause, Play, Plus, Power, Search, ShieldCheck,
  Sparkles, Square, Sun, Target, Thermometer, TriangleAlert, X } from 'lucide-react'
import { controls, date as formatDate, duration, inputResponse, milliseconds, number as formatNumber, stateLabel } from './logic'
import type { Logs, Mode, Model, Page, Snapshot } from './types'
import { useConsole } from './useConsole'
import { GameSetupFields, useGameSetup } from './GameSetupFields'
import { gameSettingsLabel } from './gameSetup'
import { BattleViewPanel } from './BattleView'

type Command = (path: string, body?: unknown) => Promise<boolean>
const TITLES = { train: message("console.training_console"), models: message("console.model_library"), diagnostics: message("console.safety_diagnostics") }

function useGoal(key: string, fallback: 'survival' | 'nmnb') {
  const [goal, setGoal] = useState<'survival' | 'nmnb'>(() => {
    try { const saved = localStorage.getItem(key); return saved === 'survival' || saved === 'nmnb' ? saved : fallback } catch { return fallback }
  })
  useEffect(() => { try { localStorage.setItem(key, goal) } catch { /* Browser storage may be disabled. */ } }, [key, goal])
  return [goal, setGoal] as const
}

function Logo({ small = false }: { small?: boolean }) {
  return <svg className={small ? 'logo small' : 'logo'} viewBox="0 0 48 48" aria-hidden="true">
    <circle cx="24" cy="24" r="20" fill="none" stroke="currentColor" strokeWidth="1.5" strokeDasharray="105 21" transform="rotate(-60 24 24)" />
    <circle cx="24" cy="24" r="11" fill="none" stroke="currentColor" strokeWidth="1.5" opacity=".5" />
    <circle cx="41" cy="14" r="3.7" fill="currentColor" /><path d="M24 20v8m-4-4h8" stroke="currentColor" />
  </svg>
}

function Dialog({ title, children, close }: { title: string; children: ReactNode; close: () => void }) {
  const { t } = useI18n()
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => { const dialog = ref.current!; dialog.showModal(); return () => dialog.close() }, [])
  return <dialog ref={ref} onCancel={close} onClick={e => { if (e.target === ref.current) close() }}>
    <div className="dialog-head"><h2>{title}</h2><button className="icon-btn" aria-label={t("console.close_dialog")} onClick={close}><X size={19} /></button></div>
    {children}
  </dialog>
}

function TemperatureChartView({ state, live }: { state: Snapshot; live: boolean }) {
  const { t } = useI18n()
  const container = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 650, height: 178 })
  useLayoutEffect(() => {
    const element = container.current!
    const measure = (width: number, height: number) => {
      if (width > 0 && height > 0) setSize(previous =>
        previous.width === width && previous.height === height ? previous : { width, height })
    }
    const initial = element.getBoundingClientRect()
    measure(initial.width, initial.height)
    const observer = new ResizeObserver(([entry]) => measure(entry.contentRect.width, entry.contentRect.height))
    observer.observe(element)
    return () => observer.disconnect()
  }, [])
  const values = state.gpu_history.map(p => p.temperature_c)
  const limit = state.safety.temperature_limit_c
  const floor = Math.min(30, ...values)
  const ceiling = Math.max(limit + 5, ...values)
  const left = 5
  const right = Math.max(left, size.width - 48)
  const bottom = Math.max(15, size.height - 28)
  const y = (v: number) => bottom - (v - floor) / (ceiling - floor) * (bottom - 14)
  const points = values.map((v, i) => `${left + i * (right - left) / Math.max(1, values.length - 1)},${y(v)}`).join(' ')
  return <div ref={container} className="chart" role="img" aria-label={values.length ? t("console.gpu_temperature_chart_with_0_recorded_samples", values.length) : t("console.no_gpu_samples_yet")}>
    <svg viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="xMinYMin meet">
      <defs><linearGradient id="temp-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--accent)" stopOpacity=".14" /><stop offset="100%" stopColor="var(--accent)" stopOpacity="0" /></linearGradient></defs>
      {[floor, (floor + limit) / 2, limit].map((v, i) => <g key={v}>
        <line x1={left} y1={y(v)} x2={right} y2={y(v)} stroke={i === 2 ? 'var(--danger)' : 'var(--line)'} strokeDasharray={i === 2 ? '4 6' : '2 5'} opacity={i === 2 ? '.5' : '.7'} />
        <text x={right + 12} y={y(v) + 4} className={i === 2 ? 'chart-limit' : 'chart-label'}>{Math.round(v)}°</text>
      </g>)}
      {values.length > 1 && <><polygon points={`${left},${bottom} ${points} ${right},${bottom}`} fill="url(#temp-fill)" /><polyline points={points} fill="none" stroke="var(--accent)" strokeWidth="2.2" strokeLinejoin="round" vectorEffect="non-scaling-stroke" /></>}
      {values.length > 0 && <circle cx={values.length > 1 ? right : left} cy={y(values.at(-1)!)} r="3.2" fill="var(--accent)" />}
      <text x={left} y={size.height - 3} className="chart-label">{state.active ? t("console.recent_samples") : t("console.previous_session")}</text><text x={right} y={size.height - 3} textAnchor="end" className="chart-label">{live ? t("console.now") : t("console.last_sample")}</text>
    </svg>
    {!values.length && <div className="chart-empty"><Activity size={22} /><span>{t("console.samples_appear_after_monitoring_starts")}</span></div>}
  </div>
}
const TemperatureChart = memo(TemperatureChartView, (a, b) => a.live === b.live && a.state.gpu_history === b.state.gpu_history
  && a.state.active === b.state.active && a.state.gpu_live === b.state.gpu_live
  && a.state.safety.temperature_limit_c === b.state.safety.temperature_limit_c)

function MetricView({ title, value, unit, note, icon: Icon }: { title: string; value: string; unit?: string; note: string; icon: LucideIcon }) {
  return <article className="metric"><div className="metric-top">{title}<Icon size={15} /></div><div className="metric-value">{value}<span>{unit}</span></div><div className="metric-note">{note}</div></article>
}
const Metric = memo(MetricView)

function GuardView({ state, command, enabled, connected }: { state: Snapshot; command: Command; enabled: boolean; connected: boolean }) {
  const { t } = useI18n()
  const gpu = state.session.gpu || {}
  const live = connected && state.gpu_live
  const memory = gpu.memory_total ? (gpu.memory_used || 0) / gpu.memory_total * 100 : 0
  return <section className="panel guard-panel">
    <div className="panel-heading"><div className="heading-icon mint"><ShieldCheck size={18} /></div><h2>{t("console.independent_gpu_protection")}</h2><span className={`dot ${live ? 'green' : 'gray'}`} /></div>
    <div className="guard-status">{!connected ? t("console.disconnected_showing_last_sample") : live ? t("console.independent_watchdog_active") : state.active ? t("console.waiting_for_telemetry") : gpu.temperature_c != null ? t("console.previous_session_sample_monitoring_inactive") : t("console.the_watchdog_starts_with_the_session")}</div>
    <div className="temperature">{gpu.temperature_c == null ? '—' : Math.round(gpu.temperature_c)}<span>°C</span><div className="temperature-symbol"><Thermometer size={27} strokeWidth={1.3} /></div></div>
    <div className="guard-device">{gpu.name || state.readiness.backend.device || t("console.waiting_for_device_information")}</div>
    <div className="gpu-substats"><div><span>{t("console.power_draw")}</span><strong>{gpu.power_w?.toFixed(1) ?? '—'} <small>W</small></strong></div><div><span>{t("console.gpu_load")}</span><strong>{gpu.utilization ?? '—'} <small>%</small></strong></div><div><span>{t("console.fans")}</span><strong>{gpu.fan_percent?.[0] ?? '—'} <small>%</small></strong></div></div>
    <div className="memory-label"><span>{t("console.vram_usage")}</span><span>{gpu.memory_used == null ? '—' : (gpu.memory_used / 2 ** 30).toFixed(1)} <i>/ {gpu.memory_total ? (gpu.memory_total / 2 ** 30).toFixed(0) : '—'} GB</i></span></div>
    <div className="track thin"><div style={{ width: `${Math.min(memory, 100)}%` }} /></div>
    <div className="guard-rules"><div><span className="rule-dot" />{t("console.core_limit", state.safety.temperature_limit_c)}</div><div><span className="rule-dot" />{t("console.sample_interval", state.safety.poll_seconds * 1000)}</div><div><span className="rule-dot" />{t("console.stale_timeout", state.safety.telemetry_timeout_seconds)}</div></div>
    <button className="btn danger" disabled={!enabled} onClick={() => void command('session/stop')}><Square size={14} fill="currentColor" />{t("console.emergency_stop")}<kbd>F8</kbd></button>
  </section>
}
const Guard = memo(GuardView, (a, b) => a.connected === b.connected && a.command === b.command && a.enabled === b.enabled
  && a.state.session.gpu === b.state.session.gpu && a.state.active === b.state.active
  && a.state.gpu_live === b.state.gpu_live && a.state.safety === b.state.safety
  && a.state.readiness.backend.device === b.state.readiness.backend.device)

function Training({ state, connected, pending, command, navigate }: { state: Snapshot; connected: boolean; pending: boolean; command: Command; navigate: (p: Page) => void }) {
  const { t, locale, display } = useI18n()
  const number = (value?: number | null) => formatNumber(value, locale)
  const date = (value?: string | null) => formatDate(value, locale)
  const [minutes, setMinutes] = useState(state.default_minutes)
  const [goal, setGoal] = useGoal('th06-training-goal', 'survival')
  const [gameSetup, setGameSetup] = useGameSetup(state)
  const [create, setCreate] = useState(false)
  const [dismissedHistory, setDismissedHistory] = useState('')
  const c = controls(state, connected, pending)
  const selected = state.models.find(m => m.id === state.selected_model_id)
  const w = state.session.worker || {}
  const history = !!state.session.time && !state.active
  const historyKey = `${state.run_dir}|${state.session.reason}`
  const budget = state.active ? w.budget_seconds || minutes * 60 : minutes * 60
  const start = async (mode: Mode) => {
    if (await command('sessions', { mode, minutes, model_id: selected?.id, goal, game_setup: gameSetup })) setCreate(false)
  }
  return <>
    <div className="hero"><div><div className="eyebrow"><span />NEURAL PILOT / WORKSPACE</div><h1>{t("console.training_console")}<span className="title-period">.</span></h1><p>{t("console.select_a_model_manage_training_and_follow")}</p></div><div className="hero-meta"><button className="btn secondary" disabled={!connected || pending || state.preview} onClick={() => void command('game/focus')}>{t("console.focus_game")}</button><span className="chip"><Crosshair size={13} />{!state.active && w.game_settings ? t("console.previous") : ''}{gameSettingsLabel(w.game_settings, locale)}</span><span className="chip"><Cpu size={13} />PPO · CUDA</span></div></div>
    {(!connected || state.preview || (state.active && w.focus_pending) || (state.session.state === 'stopped' && state.session.reason && dismissedHistory !== historyKey)) && <div className={`banner ${!connected ? 'warning' : ''}`} role="status"><Info size={16} /><span>{!connected ? t("console.backend_disconnected_showing_cached_data_the_independent") : state.preview ? t("console.read_only_preview_browse_models_and_saved") : state.active && w.focus_pending ? display(w.message) || t("console.waiting_for_game_focus") : t("console.history_0_1", date(state.session.time), display(state.session.reason))}</span>{state.active && w.focus_pending ? <button className="btn secondary" disabled={pending || !connected || state.preview} onClick={() => void command('session/focus')}>{t("console.focus_game")}</button> : connected && !state.preview && <button className="icon-btn" aria-label={t("console.dismiss_history")} onClick={() => setDismissedHistory(historyKey)}><X size={16} /></button>}</div>}
    <GameStatePanel state={state} connected={connected} pending={pending} command={command} />
    <div className="metrics">
      <Metric title={history ? t("console.previous_active_training_time") : t("console.active_training_time")} value={duration(w.training_seconds)} icon={Clock3} note={t("console.paused_0_excluded_from_training_time", duration(w.paused_seconds))} />
      <Metric title={history ? t("console.previous_sample_steps") : t("console.total_sample_steps")} value={number(w.steps ?? selected?.steps ?? 0)} icon={Database} note={t("console.0_new_steps_this_session", number(w.session_steps ?? 0))} />
      <Metric title={t("console.parameter_updates")} value={number(w.updates ?? selected?.updates ?? 0)} icon={Sparkles} note={t("console.weights_and_optimizer_retained")} />
      <Metric title={history ? t("console.previous_round_survival") : t("console.current_round_survival")} value={w.survival_seconds?.toFixed(1) ?? '—'} unit="s" icon={Heart} note={t("console.round_0", number(w.episodes ?? 0))} />
    </div>
    <div className="workspace-grid"><div className="workspace-main">
      <section className="panel session-panel"><div className="panel-heading"><div className="heading-icon"><Activity size={18} /></div><h2>{t("console.training_session")}</h2><span className={`badge ${state.active ? 'accent' : ''}`}><span className={`dot ${state.active ? 'green' : 'gray'}`} />{stateLabel(state, locale)}</span></div>
        <div className="session-inner"><div className="field-label">{state.busy ? t("console.current_session_model") : t("console.model_for_next_session")}<button className="text-btn" onClick={() => navigate('models')}>{t("console.manage_models")}<ArrowUpRight size={13} /></button></div>
          <button className="model-select" disabled={state.busy || pending} onClick={() => navigate('models')}><span className="model-icon"><Box size={21} strokeWidth={1.5} /></span><span><strong>{state.active ? w.model_source ? t("console.resuming_a_checkpoint") : t("console.new_model") : display(selected?.name) || t("console.no_model_selected")}</strong><small>{state.active ? w.model_source || t("console.using_a_randomly_initialized_model") : selected ? t("console.model_metadata", number(selected.steps), number(selected.updates), display(selected.compatibility)) : t("console.select_a_checkpoint_in_the_model_library")}</small></span><ChevronDown size={17} /></button>
          <div className="field-label time-label">{t("console.session_duration")}<span>{t("console.minutes")}</span></div><div className="duration-options" role="group" aria-label={t("console.training_duration")}>{[1, 5, 10, 30, 60].map(v => <button key={v} disabled={state.busy || state.preview} className={v === minutes ? 'selected' : ''} onClick={() => setMinutes(v)}>{v}<span> min</span>{v === minutes && <Check size={12} />}</button>)}</div>
          <div className="field-label time-label">{t("console.training_goal")}<span>{state.busy ? w.goal === 'nmnb' ? 'NMNB' : t("console.survival") : t("console.applies_next_session")}</span></div><div className="goal-options" role="group" aria-label={t("console.training_goal")}>{(['survival','nmnb'] as const).map(value => <button key={value} disabled={state.busy || state.preview} className={(state.busy ? w.goal : goal) === value ? 'selected' : ''} aria-pressed={(state.busy ? w.goal : goal) === value} onClick={() => setGoal(value)}><strong>{value === 'survival' ? t("console.survival") : 'NMNB'}</strong><small>{value === 'survival' ? t("console.allow_the_policy_to_learn_emergency_bomb") : t("console.bombs_disabled_count_misses_separately")}</small></button>)}</div>
          <GameSetupFields value={gameSetup} disabled={state.busy || pending || state.preview} onChange={setGameSetup} />
          <div className="session-progress"><div><span>{state.active ? t("console.active_training_progress") : t("console.planned_session")}<b>{state.active ? duration(w.training_seconds) : '00:00'} <i>/ {duration(budget)}</i></b></span><span>{state.active ? `${Math.min(100, ((w.training_seconds || 0) / budget * 100)).toFixed(1)}%` : t("console.ready_to_start")}</span></div><div className="track"><div style={{ width: `${state.active ? Math.min(100, (w.training_seconds || 0) / budget * 100) : 0}%` }} /></div></div>
          {state.busy ? <div className="session-actions"><button className="btn primary" disabled={!c.pause} onClick={() => void command('session/pause', { paused: !c.paused })}>{c.paused ? <Play size={16} fill="currentColor" /> : <Pause size={16} fill="currentColor" />}{state.finish_requested ? t("console.saving_model") : !state.active ? t("console.starting_watchdog") : c.paused ? t("console.resume_0", state.session.mode === 'play' ? t("console.playback") : state.session.mode === 'evaluate' ? t("console.evaluation") : t("console.this_session")) : t("console.pause_0", state.session.mode === 'play' ? t("console.playback") : state.session.mode === 'evaluate' ? t("console.evaluation") : t("console.training"))}<kbd>F7</kbd></button><button className="btn secondary" disabled={!c.finish} onClick={() => void command('session/finish')}><ArrowDownToLine size={16} />{t("console.save_and_finish")}</button></div>
            : <div className="session-actions"><button className="btn primary" disabled={!c.start} onClick={() => void start('resume')}><Play size={15} fill="currentColor" />{t("console.resume_training")}<ArrowRight size={16} /></button><button className="btn secondary" disabled={!c.create} onClick={() => setCreate(true)}><Plus size={16} />{t("console.new_model")}</button></div>}
          <div className="session-hint"><ShieldCheck size={13} /><span>{!state.readiness.ready ? t("console.complete_the_setup_checks_on_the_safety") : state.busy ? (display(w.message) || t("console.the_gpu_watchdog_remains_active_while_progress")) : t("console.resume_training_retains_the_saved_weights_on")}</span></div>
        </div>
      </section>
      <section className="panel telemetry-panel"><div className="panel-heading"><div className="heading-icon"><Gauge size={18} /></div><h2>{t("console.gpu_telemetry")}</h2><span className="subtle-tag"><span className={`dot ${connected && state.gpu_live ? 'green' : 'gray'}`} />{connected && state.gpu_live ? t("console.live") : t("console.history")}</span></div><div className="telemetry-top"><div><span className="legend-dot" />{t("console.core_temperature")}<span className="telemetry-value">{state.session.gpu?.temperature_c ?? '—'}°C</span></div><span>{t("console.stop_threshold")} {state.safety.temperature_limit_c}°C</span></div><TemperatureChart state={state} live={connected && state.gpu_live} /></section>
    </div><aside className="workspace-aside"><Guard state={state} command={command} enabled={c.stop} connected={connected} /><section className="panel game-panel"><div className="panel-heading"><div className="heading-icon"><Target size={18} /></div><h2>{t("console.game_state")}</h2><span className="subtle-tag">{state.active ? t("console.current_round") : t("console.previous_session")}</span></div><div className="resource-grid"><div><span className="resource-letter power">P</span><strong>{number(w.resources?.power)}</strong><small>{t("console.power")}</small></div><div><span className="resource-letter bomb">B</span><strong>{number(w.resources?.bombs)}</strong><small>{t("console.bombs")}</small></div><div><Heart size={17} className="life" /><strong>{number(w.resources?.lives)}</strong><small>{t("console.lives")}</small></div></div><div className="game-footer"><span>{t("console.bomb_requests_used")}</span><strong>{w.bomb_requests ?? '—'} / {w.bombs_used ?? '—'}</strong></div></section></aside></div>
    <div className="engine-strip"><div className="engine-icon"><Crosshair size={18} /></div><div><strong>{t("console.rolling_path_planner")} <span>v{state.versions.planner}</span></strong><p>{t("console.checks_bullets_and_enemy_collision_bodies_along")}</p></div><div className="engine-stats"><span>{t("console.viable_directions")}<strong>{w.planning?.feasible_roots ?? '—'}<i>/ 18</i></strong></span><span>{t("console.planning_time")}<strong>{w.planning?.elapsed_ms?.toFixed(1) ?? '—'}<i>ms</i></strong></span><span>{t("console.enemy_bodies")}<strong>{w.planning?.considered_enemy_bodies ?? '—'}</strong></span></div></div>
    {create && <Dialog title={t("console.train_a_new_model")} close={() => setCreate(false)}><p className="dialog-copy">{t("console.new_model_description", minutes)}</p><div className="dialog-note"><Info size={17} />{t("console.select_resume_training_to_keep_the_learned")}</div><div className="dialog-actions"><button className="btn secondary" onClick={() => setCreate(false)}>{t("console.cancel")}</button><button className="btn primary" disabled={!c.create} onClick={() => void start('train')}><Plus size={15} />{t("console.create_and_start")}</button></div></Dialog>}
  </>
}

function ModelsView({ state, connected, pending, command, navigate }: { state: Snapshot; connected: boolean; pending: boolean; command: Command; navigate: (p: Page) => void }) {
  const { t, locale, display } = useI18n()
  const number = (value?: number | null) => formatNumber(value, locale)
  const date = (value?: string | null) => formatDate(value, locale)
  const [goal, setGoal] = useGoal('th06-evaluation-goal', 'nmnb')
  const [gameSetup, setGameSetup] = useGameSetup(state)
  const [search, setSearch] = useState('')
  const [usable, setUsable] = useState(false)
  const [modelPage, setModelPage] = useState(0)
  const [detail, setDetail] = useState<Model | null>(null)
  const c = controls(state, connected, pending)
  const rows = state.models.filter(m => (!usable || m.usable) && `${display(m.name)} ${m.name} ${m.path} ${m.steps} ${number(m.steps)}`.toLowerCase().includes(search.toLowerCase()))
  const pageCount = Math.max(1, Math.ceil(rows.length / 25))
  const page = Math.min(modelPage, pageCount - 1)
  const visibleRows = rows.slice(page * 25, (page + 1) * 25)
  useEffect(() => setModelPage(0), [search, usable])
  const select = async (m: Model) => { await command('models/select', { model_id: m.id }) }
  const play = async (mode: Mode) => { if (await command('sessions', { mode, minutes: mode === 'evaluate' ? 60 : 10, model_id: state.selected_model_id, goal, game_setup: gameSetup })) navigate('train') }
  return <><div className="hero"><div><div className="eyebrow"><span />MODEL REGISTRY</div><h1>{t("console.model_library")}<span className="title-period">.</span></h1><p>{t("console.resume_training_from_a_checkpoint_watch_the")}</p></div><div className="hero-meta"><span className="chip"><Database size={13} />{state.models.length} {t("console.checkpoints")}</span></div></div>
    <div className="model-toolbar"><div className="search-input"><Search size={17} /><input aria-label={t("console.search_models")} value={search} onChange={e => setSearch(e.target.value)} placeholder={t("console.search_models_step_counts_or_filenames")} />{search && <button className="icon-btn" aria-label={t("console.clear_search")} onClick={() => setSearch('')}><X size={15} /></button>}</div><div className="filter-options"><button onClick={() => setUsable(false)} className={!usable ? 'selected' : ''}>{t("console.all_models")}</button><button onClick={() => setUsable(true)} className={usable ? 'selected' : ''}>{t("console.compatible_models")}</button></div></div>
    <section className="panel model-table"><div className="table-scroll"><table><thead><tr><th>{t("console.model_name")}</th><th>{t("console.total_samples")}</th><th>{t("console.parameter_updates")}</th><th>{t("console.saved_at")}</th><th>{t("console.compatibility")}</th><th><span className="sr-only">{t("console.actions")}</span></th></tr></thead><tbody>{visibleRows.map(m => <tr key={m.id} className={state.selected_model_id === m.id ? 'selected-row' : ''}><td><button className="model-name-btn" disabled={state.busy || pending || !m.usable} onClick={() => void select(m)}><span className={`table-model-icon ${state.selected_model_id === m.id ? 'active' : ''}`}>{state.selected_model_id === m.id ? <Check size={17} /> : <Box size={17} />}</span><span><strong>{display(m.name)}{state.selected_model_id === m.id && <b className="selected-tag">{t("console.selected")}</b>}</strong><small title={m.path}>{m.path}</small></span></button></td><td className="mono">{number(m.steps)}</td><td className="mono">{number(m.updates)}</td><td>{date(m.saved_at)}</td><td><span className={`compatibility ${m.usable ? '' : 'bad'}`}><span className={`dot ${m.usable ? 'green' : 'gray'}`} />{display(m.compatibility)}</span></td><td><button className="icon-btn" aria-label={t("console.view_details_for_0", display(m.name))} onClick={() => setDetail(m)}><Ellipsis size={19} /></button></td></tr>)}</tbody></table></div>{!rows.length && <div className="empty-state"><Database size={30} /><strong>{search ? t("console.no_matching_models") : t("console.no_model_checkpoints_yet")}</strong><p>{search ? t("console.try_another_step_count_or_filename") : t("console.the_first_checkpoint_appears_here_after_training")}</p>{search && <button className="text-btn" onClick={() => setSearch('')}>{t("console.clear_search")}<ArrowRight size={14} /></button>}</div>}</section>
    {rows.length > 25 && <nav className="model-pagination" aria-label={t("console.model_pagination")}><span>{t("console.model_page_summary", rows.length, page + 1, pageCount)}</span><div><button className="btn secondary" disabled={page === 0} onClick={() => setModelPage(page - 1)}>{t("console.previous_2")}</button><button className="btn secondary" disabled={page + 1 >= pageCount} onClick={() => setModelPage(page + 1)}>{t("console.next")}</button></div></nav>}
    <GameSetupFields value={gameSetup} disabled={state.busy || pending || state.preview} onChange={setGameSetup} />
    <div className="evaluation-goal"><label htmlFor="evaluation-goal">{t("console.playback_evaluation_goal")}</label><select id="evaluation-goal" disabled={state.busy || pending || state.preview} value={goal} onChange={e => setGoal(e.target.value as 'survival' | 'nmnb')}><option value="nmnb">{t("console.nmnb_bombs_disabled")}</option><option value="survival">{t("console.survival_emergency_bombs_allowed")}</option></select><span>{t("console.inference_only_runs_that_use_bombs_are")}</span></div>
    <div className="models-bottom"><div><ShieldCheck size={16} /><span>{t("console.each_save_keeps_a_historical_checkpoint_resuming")}</span></div><div className="model-actions"><button className="btn secondary" disabled={!c.start} onClick={() => void play('play')}><Eye size={16} />{t("console.play_model")}</button><button className="btn secondary" disabled={!c.start} onClick={() => void play('evaluate')}><Target size={16} />{t("console.evaluate_10_rounds")}</button><button className="btn primary" onClick={() => navigate('train')}>{t("console.go_to_training")}<ArrowRight size={16} /></button></div></div>
    {detail && <Dialog title={t("console.checkpoint_details")} close={() => setDetail(null)}><div className="detail-model"><Box size={27} /><div><strong>{display(detail.name)}</strong><small>{date(detail.saved_at)}</small></div></div><dl className="detail-list"><div><dt>{t("console.total_samples")}</dt><dd>{number(detail.steps)} {t("console.steps")}</dd></div><div><dt>{t("console.parameter_updates")}</dt><dd>{number(detail.updates)}</dd></div><div><dt>{t("console.observation_format")}</dt><dd>v{detail.format}</dd></div><div><dt>{t("console.planner_version_at_save")}</dt><dd>{detail.planner_version ? `v${detail.planner_version}` : t("console.disabled")}</dd></div><div><dt>{t("console.compatibility")}</dt><dd>{display(detail.compatibility)}</dd></div></dl><div className="file-path"><FolderOpen size={15} /><code>{detail.path}</code></div><div className="dialog-actions"><button className="btn secondary" onClick={() => setDetail(null)}>{t("console.close")}</button><button className="btn primary" disabled={!detail.usable || state.busy || pending} onClick={async () => { if (await command('models/select', { model_id: detail.id })) setDetail(null) }}><Check size={16} />{t("console.use_this_model")}</button></div></Dialog>}
  </>
}
const Models = memo(ModelsView, (a, b) => a.command === b.command && a.navigate === b.navigate
  && a.connected === b.connected && a.pending === b.pending && a.state.models === b.state.models
  && a.state.selected_model_id === b.state.selected_model_id && a.state.busy === b.state.busy
  && a.state.preview === b.state.preview && a.state.readiness.ready === b.state.readiness.ready)

function InputLatencyPanel({ state }: { state: Snapshot }) {
  const { t, locale, display } = useI18n()
  const number = (value?: number | null) => formatNumber(value, locale)
  const timing = state.session.worker?.input_latency
  const shortcut = (key: 'F7' | 'F8') => !state.active ? t("console.no_active_session")
    : state.hotkeys?.[key] === 'key_events' ? t("console.event_listener")
    : state.hotkeys?.[key] === 'poll_5ms' ? t("console.polling_every_5_ms")
    : state.hotkeys?.[key] === 'disabled' ? t("console.unused_in_this_mode") : t("console.enabling")
  return <section className="panel input-timing-panel">
    <div className="panel-heading"><div className="heading-icon"><Activity size={18} /></div><h2>{t("console.input_response")}</h2><span className="subtle-tag">{state.active ? t("console.current_session") : t("console.previous_session")}</span></div>
    <div className="input-timing-grid">
      <dl className="safety-rules">
        <div><dt>{t("console.observation_to_key_dispatch")}</dt><dd>{milliseconds(timing?.observation_to_send_ms)}</dd></div>
        <div><dt>{t("console.p95_last_64_samples")}</dt><dd>{milliseconds(timing?.observation_to_send_p95_ms)}</dd></div>
        <div><dt>{t("console.policy_inference_and_action_return")}</dt><dd>{milliseconds(timing?.policy_ms)}</dd></div>
      </dl>
      <dl className="safety-rules">
        <div><dt>{t("console.key_dispatch_call")}</dt><dd>{timing?.input_events === 0 ? t("console.holding_existing_keys") : milliseconds(timing?.send_ms)}</dd></div>
        <div><dt>{t("console.game_input_sample")}</dt><dd>{inputResponse(timing, locale)}</dd></div>
        <div><dt>{t("console.key_events_this_step")}</dt><dd>{number(timing?.input_events)}</dd></div>
      </dl>
    </div>
    <div className="small-note"><Info size={14} />{!timing && t("console.input_measurements_begin_with_the_next_run")}{t("console.the_game_input_sample_records_when_a")}</div>
    <h3 className="shortcut-title">{t("console.global_hotkeys")}</h3>
    <div className="input-timing-grid"><dl className="safety-rules">
      <div><dt>{t("console.f7_pause_resume")}</dt><dd>{shortcut('F7')}</dd></div>
      <div><dt>{t("console.last_dispatch_count")}</dt><dd>{milliseconds(state.hotkeys?.F7_dispatch_ms)} / {number(state.hotkeys?.F7_count)}</dd></div>
    </dl><dl className="safety-rules">
      <div><dt>{t("console.f8_emergency_stop")}</dt><dd>{shortcut('F8')}</dd></div>
      <div><dt>{t("console.last_dispatch_count")}</dt><dd>{milliseconds(state.hotkeys?.F8_dispatch_ms)} / {number(state.hotkeys?.F8_count)}</dd></div>
    </dl></div>
    {state.active && (state.hotkeys?.listener_error || state.hotkeys?.F7_error || state.hotkeys?.F8_error) && <div className="small-note">{display(state.hotkeys.listener_error || state.hotkeys.F7_error || state.hotkeys.F8_error)}</div>}
  </section>
}

const GAME_PHASES: Record<string, string> = { combat: message("console.stage_in_progress"), boss: message("console.boss_fight"), pause: message("console.game_paused"), dialogue: message("console.boss_dialogue"), script_wait: message("console.cutscene"), continue: message("console.continue_menu"), result: message("console.results_screen"), dead: message("console.player_hit"), title: message("console.main_menu"), title_prompt: message("console.title_screen"), difficulty: message("console.select_difficulty"), character: message("console.select_character"), shot: message("console.select_shot_type"), run_mode: message("console.select_game_mode"), practice_stage: message("console.practice_stage_select"), start_confirm: message("console.confirm_game_start"), menu_transition: message("console.menu_transition") }

function GameStatePanel({ state, connected, pending, command }: { state: Snapshot; connected: boolean; pending: boolean; command: Command }) {
  const { t, locale, display } = useI18n()
  const date = (value?: string | null) => formatDate(value, locale)
  const w = state.session.worker || {}
  const probe = state.game_probe
  const label = state.active ? display(GAME_PHASES[w.game_phase || '']) || t("console.waiting_for_the_worker_to_identify_the") : display(probe?.label) || t("console.game_not_checked_yet")
  const settings = state.active ? w.game_settings : probe?.game_settings
  const stage = settings?.source === 'menu' && settings.stage ? t("console.stage_0_selected", settings.stage) : state.active ? display(w.stage_label) || t("console.stage_unknown") : display(probe?.stage_label)
  return <section className="panel game-state-panel" aria-label={t("console.game_state")}>
    <div className="heading-icon mint"><Eye size={17} /></div><div className="game-state-copy"><strong>{stage ? `${stage} · ${label}` : label}</strong>{settings && <small>{gameSettingsLabel(settings, locale)}</small>}<small>{state.active ? t("console.current_stage_0_reported_by_the_worker", duration(w.stage_seconds)) : probe ? t("console.read_only_probe_at_0_1", date(probe.observed_at), probe.available ? t("console.separate_from_saved_session_history") : display(probe.message)) : t("console.start_from_the_main_menu_the_worker")}</small></div>
    <button className="btn secondary" disabled={!connected || pending || state.busy || state.preview} onClick={() => void command('game/probe')}>{t("console.check_game_state")}</button>
  </section>
}

function AccelerationPanel({ state, connected, pending, command }: { state: Snapshot; connected: boolean; pending: boolean; command: Command }) {
  const { t } = useI18n()
  const cfg = state.acceleration
  const disabled = !controls(state, connected, pending).diagnostic
  const update = (field: 'collision' | 'input', value: string) => void command('settings/acceleration', { collision: cfg?.collision || 'python', input: cfg?.input || 'python', [field]: value })
  return <section className="panel acceleration-panel"><div className="panel-heading"><div className="heading-icon"><Cpu size={18} /></div><h2>{t("console.compute_backends")}</h2><span className={`badge ${cfg?.available ? 'accent' : ''}`}>{cfg?.available ? t("console.rust_available") : t("console.python_available")}</span></div>
    <div className="backend-fields">{([{ field: 'collision', name: t("console.collision_detection") }, { field: 'input', name: t("console.keyboard_input") }] as const).map(item => <label key={item.field}>{item.name}<select aria-label={t("console.0_backend", item.name)} disabled={disabled} value={cfg?.[item.field] || 'python'} onChange={e => update(item.field, e.target.value)}><option value="auto">{t("console.auto_prefer_rust")}</option><option value="python">Python</option><option value="rust" disabled={!cfg?.available}>Rust · PyO3</option></select></label>)}</div>
    <p className="small-note">{t("console.changes_apply_next_session_module_timing_and")}</p>
  </section>
}

function Diagnostics({ state, connected, pending, command }: { state: Snapshot; connected: boolean; pending: boolean; command: Command }) {
  const { t, locale } = useI18n()
  const [logs, setLogs] = useState<Logs | null>(null)
  const [logError, setLogError] = useState(false)
  const [tab, setTab] = useState<'worker' | 'launcher'>('worker')
  const rawLog = logs?.[tab]
  const visibleLog = useMemo(() => translateLog(rawLog, locale), [rawLog, locale])
  const [calibrate, setCalibrate] = useState(false)
  const [copied, setCopied] = useState(false)
  const c = controls(state, connected, pending)
  const w = state.session.worker || {}
  useEffect(() => {
    let disposed = false
    let etag: string | null = null
    let request: AbortController | null = null
    const load = async () => {
      if (disposed || document.hidden || request) return
      request = new AbortController()
      try {
        const response = await fetch('/api/logs', { cache: 'no-store', signal: request.signal,
          headers: etag ? { 'If-None-Match': etag } : {} })
        if (response.status === 304) { if (!disposed) setLogError(false); return }
        if (!response.ok) throw new Error()
        const data = await response.json() as Logs
        etag = response.headers.get('ETag')
        if (!disposed) {
          setLogs(previous => previous?.run_dir === data.run_dir && previous.worker === data.worker
            && previous.launcher === data.launcher ? previous : data)
          setLogError(false)
        }
      } catch (error) {
        if (!disposed && !(error instanceof DOMException && error.name === 'AbortError')) setLogError(true)
      } finally { request = null }
    }
    const visibility = () => { if (document.hidden) request?.abort(); else void load() }
    void load(); const timer = setInterval(load, 3000)
    document.addEventListener('visibilitychange', visibility)
    return () => { disposed = true; request?.abort(); clearInterval(timer); document.removeEventListener('visibilitychange', visibility) }
  }, [])
  const checks = [{ title: t("console.gpu_environment"), okay: state.readiness.checks.gpu, text: state.readiness.backend.device || t("console.check_pending"), icon: <Cpu size={19} />, mode: 'check' as Mode, action: t("console.gpu_check") }, { title: t("console.game_calibration"), okay: state.readiness.checks.calibration, text: t("console.arrow_keys_shift_pause_and_restart"), icon: <Crosshair size={19} />, mode: 'calibrate' as Mode, action: t("console.recalibrate") }, { title: t("console.resource_readings"), okay: state.readiness.checks.resources, text: t("console.power_bombs_lives"), icon: <Database size={19} /> }]
  return <><div className="hero"><div><div className="eyebrow"><span />SYSTEM HEALTH</div><h1>{t("console.safety_diagnostics")}<span className="title-period">.</span></h1><p>{t("console.review_gpu_safeguards_setup_checks_and_process")}</p></div><span className={`chip ${state.readiness.ready ? 'ready-chip' : ''}`}>{state.readiness.ready ? <CheckCheck size={15} /> : <Info size={15} />}{state.readiness.ready ? t("console.ready_to_train") : t("console.setup_checks_pending")}</span></div>
    <div className="diagnostic-checks">{checks.map(check => <section className="panel check-card" key={check.title}><div className="check-top"><div className="heading-icon">{check.icon}</div><span className={`badge ${check.okay ? 'accent' : ''}`}>{check.okay ? <Check size={12} /> : <Circle size={10} />}{check.okay ? t("console.passed") : t("console.pending")}</span></div><h2>{check.title}</h2><p>{check.text}</p>{check.mode && <button className="text-btn" disabled={!c.diagnostic} onClick={() => check.mode === 'calibrate' ? setCalibrate(true) : void command('sessions', { mode: check.mode, minutes: 1 })}>{check.action}<ArrowUpRight size={13} /></button>}</section>)}</div>
    <div className="diagnostic-grid"><section className="panel safety-panel"><div className="panel-heading"><div className="heading-icon mint"><ShieldCheck size={18} /></div><h2>{t("console.independent_safety_limits")}</h2><button className="text-btn" disabled={!c.diagnostic} onClick={() => void command('sessions', { mode: 'monitor', minutes: 60 })}>{t("console.monitor_only")}<ArrowUpRight size={13} /></button></div><dl className="safety-rules"><div><dt><Thermometer size={16} />{t("console.core_temperature")}</dt><dd>≥ {state.safety.temperature_limit_c}{t("console.c_stop_threshold")}</dd></div><div><dt><Database size={16} />{t("console.vram_usage")}</dt><dd>≥ {Math.round(state.safety.vram_fraction_limit * 100)}{t("console.stop_threshold_2")}</dd></div><div><dt><Activity size={16} />{t("console.sampling_interval")}</dt><dd>{state.safety.poll_seconds * 1000} ms</dd></div><div><dt><Clock3 size={16} />{t("console.telemetry_timeout_2")}</dt><dd>{state.safety.telemetry_timeout_seconds} {t("console.s_stop_timeout")}</dd></div><div><dt><Heart size={16} />{t("console.worker_heartbeat")}</dt><dd>{state.safety.worker_heartbeat_seconds} {t("console.s_2")}</dd></div></dl><div className="small-note"><Info size={14} />{t("console.calibration_readings")}</div></section>
      <section className="panel planner-panel"><div className="panel-heading"><div className="heading-icon"><Crosshair size={18} /></div><h2>{t("console.policy_and_planning")}</h2><span className="subtle-tag">{state.active ? t("console.live") : t("console.previous_session")}</span></div><dl className="safety-rules"><div><dt>{t("console.planner_version")}</dt><dd>v{state.versions.planner}</dd></div><div><dt>{t("console.observation_reward_version")}</dt><dd>v{state.versions.observation} / v{state.versions.reward}</dd></div><div><dt>{t("console.nearby_full_pool_visible_bullets")}</dt><dd>{w.bullet_observation?.near ?? '—'} / {w.bullet_observation?.full ?? '—'} / {w.bullet_observation?.visible ?? '—'}</dd></div><div><dt>{t("console.collision_bodies_unknown_geometry")}</dt><dd>{w.planning?.considered_enemy_bodies ?? '—'} / {w.planning?.unknown_body_geometry ?? '—'}</dd></div><div><dt>{t("console.predicted_route_status")}</dt><dd>{w.planning?.enabled ? w.planning.all_routes_risky ? t("console.every_searched_route_carries_risk") : t("console.viable_route_predicted") : t("console.no_records_yet")}</dd></div></dl></section></div>
    <InputLatencyPanel state={state} />
    <BattleViewPanel enabled={state.battle_view_enabled} available={connected && !pending && !state.preview}
      toggle={() => void command('battle-view', { enabled: !state.battle_view_enabled })} />
    <GameStatePanel state={state} connected={connected} pending={pending} command={command} />
    <AccelerationPanel state={state} connected={connected} pending={pending} command={command} />
    <section className="panel logs-panel"><div className="panel-heading"><div className="heading-icon"><FileText size={18} /></div><h2>{t("console.run_logs")}</h2><div className="logs-tabs"><button className={tab === 'worker' ? 'selected' : ''} onClick={() => setTab('worker')}>{t("console.training_worker")}</button><button className={tab === 'launcher' ? 'selected' : ''} onClick={() => setTab('launcher')}>{t("console.launch_log")}</button></div><button className="icon-btn" aria-label={t("console.copy_log")} disabled={!logs?.[tab]} onClick={async () => { try { await navigator.clipboard.writeText(logs?.[tab] || ''); setCopied(true); setTimeout(() => setCopied(false), 2000) } catch { setCopied(false) } }}>{copied ? <Check size={16} /> : <Copy size={16} />}</button></div><div className="log-location"><FolderOpen size={13} /><span>{logs?.run_dir || t("console.no_run_directory")}</span></div>{logError && <div className="log-error">{t("console.log_connection_interrupted_retrying_showing_the_last")}</div>}<pre className="log-content">{visibleLog || t("console.logs_will_appear_when_a_session_starts")}</pre></section>
    {calibrate && <Dialog title={t("console.recalibrate_in_game")} close={() => setCalibrate(false)}><p className="dialog-copy">{t("console.calibration_sends_movement_pause_and_restart_keys")}</p><div className="dialog-note"><Crosshair size={17} />{t("console.pause_at_the_start_of_stage_1")}</div><div className="dialog-actions"><button className="btn secondary" onClick={() => setCalibrate(false)}>{t("console.cancel")}</button><button className="btn primary" disabled={!c.diagnostic} onClick={async () => { if (await command('sessions', { mode: 'calibrate', minutes: 1 })) setCalibrate(false) }}>{t("console.start_calibration")}<ArrowRight size={15} /></button></div></Dialog>}
  </>
}

export default function App() {
  const { t, locale, display, setLocale } = useI18n()
  const { state, connected, pending, serverClosed, command, notice, dismiss } = useConsole()
  const [exitDialog, setExitDialog] = useState(false)
  const closing = !serverClosed && ['waiting', 'stopping'].includes(state?.shutdown?.state || '')
  const closeServer = async () => { if (await command('server/shutdown')) setExitDialog(false) }
  const [page, setPage] = useState<Page>(() => (['train', 'models', 'diagnostics'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'train') as Page)
  const [sidebar, setSidebar] = useState(false)
  const [light, setLight] = useState(() => localStorage.getItem('th06-theme') === 'light')
  useEffect(() => { document.documentElement.dataset.theme = light ? 'light' : 'dark'; localStorage.setItem('th06-theme', light ? 'light' : 'dark') }, [light])
  useEffect(() => { const change = () => { const value = location.hash.slice(1); if (['train', 'models', 'diagnostics'].includes(value)) setPage(value as Page) }; window.addEventListener('hashchange', change); return () => window.removeEventListener('hashchange', change) }, [])
  useEffect(() => { document.title = `${display(TITLES[page])} · TH06NC-Neural-Pilot` }, [page, display])
  useEffect(() => { window.scrollTo(0, 0) }, [page])
  useEffect(() => {
    // The independent Windows guard handles these global keys. Suppress
    // browser caret/debugger shortcuts without toggling the training twice.
    const guardKeys = (event: KeyboardEvent) => { if (event.key === 'F7' || event.key === 'F8') event.preventDefault() }
    window.addEventListener('keydown', guardKeys)
    return () => window.removeEventListener('keydown', guardKeys)
  }, [])
  const navigate = useCallback((next: Page) => { setPage(next); location.hash = next; setSidebar(false) }, [])
  return <div className="app-shell">
    {sidebar && <button className="sidebar-backdrop" aria-label={t("console.hide_navigation")} onClick={() => setSidebar(false)} />}
    <aside className={`sidebar ${sidebar ? 'open' : ''}`}><a className="brand" href="#train" onClick={() => navigate('train')}><Logo /><span>TH06NC<small>NEURAL PILOT</small></span></a><div className="sidebar-caption">{t("console.workspace")}</div><nav aria-label={t("console.main_navigation")}>{([{ id: 'train', icon: LayoutDashboard, text: t("console.training_console") }, { id: 'models', icon: Database, text: t("console.model_library") }, { id: 'diagnostics', icon: ShieldCheck, text: t("console.safety_diagnostics") }] as const).map(n => <button key={n.id} className={page === n.id ? 'active' : ''} aria-current={page === n.id ? 'page' : undefined} onClick={() => navigate(n.id)}><n.icon size={18} strokeWidth={1.7} />{n.text}{page === n.id && <span className="nav-dot" />}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-card"><span className={`dot ${connected ? 'green' : 'gray'}`} /><span>{t("console.local_workspace")}<small>{serverClosed ? t("console.backend_shut_down") : closing ? t("console.shutting_down_the_service") : connected ? t("console.python_backend_connected") : t("console.waiting_for_the_backend")}</small></span><Cpu size={17} /></div><div className="sidebar-version"><span>TH06NC AI</span><span>v1.0</span></div></div></aside>
    <div className="main-shell"><header className="topbar"><button className="icon-btn mobile-menu" aria-label={t("console.show_navigation")} onClick={() => setSidebar(!sidebar)}><Menu size={19} /></button><div className="breadcrumb"><span>{t("console.workspace")}</span><ChevronRight size={13} /><strong>{display(TITLES[page])}</strong></div><div className="topbar-right"><select className="language-select" aria-label={t("console.language")} value={locale} onChange={e => setLocale(e.target.value as Locale)}><option value="en">English</option><option value="zh-CN">{t("console.language_chinese")}</option></select>{state?.busy && !serverClosed && <button className="btn danger header-stop" aria-label={t("console.emergency_stop")} disabled={!controls(state, connected, pending).stop} onClick={() => void command('session/stop')}><Square size={11} fill="currentColor" /><span className="header-stop-label">{t("console.emergency_stop")}</span><kbd>F8</kbd></button>}<span className={`connection ${connected ? '' : 'offline'}`}><span className={`dot ${connected ? 'green' : 'gray'}`} />{serverClosed ? t("console.backend_shut_down") : closing ? t("console.shutting_down") : connected ? t("console.connected_locally") : t("console.disconnected")}</span><span className="topbar-separator" /><button className="icon-btn theme-btn" aria-label={light ? t("console.switch_to_dark_theme") : t("console.switch_to_light_theme")} onClick={() => setLight(!light)}>{light ? <Moon size={17} /> : <Sun size={17} />}</button><button className="btn secondary server-close" aria-label={t("console.shut_down")} title={t("console.finish_the_session_and_shut_down_the")} disabled={!connected || pending || closing || serverClosed} onClick={() => state?.busy ? setExitDialog(true) : void closeServer()}><Power size={15} /><span>{t("console.shut_down")}</span></button></div></header>
      <main>{serverClosed || closing ? <div className="connecting lifecycle-status" role="status">{serverClosed ? <Power size={32} /> : <LoaderCircle size={32} className="spin" />}<h1>{serverClosed ? t("console.backend_shut_down") : t("console.finishing_the_session_and_shutting_down")}</h1><p>{serverClosed ? t("console.open_with_launcher") : t("console.wait_while_training_saves_gpu_monitoring_continues")}</p><small>{serverClosed ? t("console.you_can_close_this_page_now") : t("console.press_f8_in_an_emergency")}</small></div> : state ? <>{state.shutdown?.state === 'error' && <div className="banner warning" role="alert"><Info size={16} /><span>{display(state.shutdown.message)}</span></div>}{page === 'train' ? <Training state={state} connected={connected} pending={pending} command={command} navigate={navigate} /> : page === 'models' ? <Models state={state} connected={connected} pending={pending} command={command} navigate={navigate} /> : <Diagnostics state={state} connected={connected} pending={pending} command={command} />}</> : <div className="connecting"><Logo /><h1>{t("console.connecting_to_the_training_backend")}</h1><p>{t("console.reading_models_and_session_status_training_starts")}</p><LoaderCircle size={21} className="spin" /><small>{t("console.open_with_launcher")}</small></div>}</main>
      <footer><div><span className="footer-dot" />{serverClosed ? t("console.service_stopped") : state?.active ? t("console.background_session_running_independently") : t("console.all_computation_runs_locally")}<span className="footer-divider">/</span><span>{serverClosed ? t("console.releasing_the_local_port") : t("console.use_shut_down_to_exit_the_backend")}</span></div><div><kbd>F7</kbd>{t("console.pause_resume")}<kbd>F8</kbd>{t("console.emergency_stop")}</div></footer>
    </div>
    {exitDialog && <Dialog title={t("console.finish_session_and_shut_down")} close={() => setExitDialog(false)}><p className="dialog-copy">{t("console.training_saves_before_the_backend_closes_playback")}</p><div className="dialog-actions"><button className="btn secondary" onClick={() => setExitDialog(false)}>{t("console.cancel")}</button><button className="btn primary" disabled={pending || !connected} onClick={() => void closeServer()}><Power size={15} />{t("console.finish_and_shut_down")}</button></div></Dialog>}
    {notice && <div className={`toast ${notice.error ? 'error' : ''}`} role={notice.error ? 'alert' : 'status'}>{notice.error ? <TriangleAlert size={18} /> : <Check size={18} />}<span>{display(notice.text)}</span><button className="icon-btn" aria-label={t("console.dismiss_notification")} onClick={dismiss}><X size={15} /></button></div>}
  </div>
}
