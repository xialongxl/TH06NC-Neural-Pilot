import { memo, useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'
import type { LucideIcon } from 'lucide-react'
import { Activity, ArrowDownToLine, ArrowRight, ArrowUpRight, Box, Check, CheckCheck,
  ChevronDown, ChevronRight, Circle, Clock3, Copy, Cpu, Crosshair, Database, Ellipsis,
  Eye, FileText, FolderOpen, Gauge, Heart, Info, LayoutDashboard,
  LoaderCircle, Menu, Moon, Pause, Play, Plus, Power, Search, ShieldCheck,
  Sparkles, Square, Sun, Target, Thermometer, TriangleAlert, X } from 'lucide-react'
import { controls, date, duration, inputResponse, milliseconds, number, stateLabel } from './logic'
import type { Logs, Mode, Model, Page, Snapshot } from './types'
import { useConsole } from './useConsole'

type Command = (path: string, body?: unknown) => Promise<boolean>
const TITLES = { train: '训练控制台', models: '模型库', diagnostics: '保护与诊断' }

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
  const ref = useRef<HTMLDialogElement>(null)
  useEffect(() => { const dialog = ref.current!; dialog.showModal(); return () => dialog.close() }, [])
  return <dialog ref={ref} onCancel={close} onClick={e => { if (e.target === ref.current) close() }}>
    <div className="dialog-head"><h2>{title}</h2><button className="icon-btn" aria-label="关闭对话框" onClick={close}><X size={19} /></button></div>
    {children}
  </dialog>
}

function TemperatureChartView({ state, live }: { state: Snapshot; live: boolean }) {
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
  return <div ref={container} className="chart" role="img" aria-label={values.length ? `显卡温度曲线，共 ${values.length} 次真实采样` : '暂无显卡采样'}>
    <svg viewBox={`0 0 ${size.width} ${size.height}`} preserveAspectRatio="xMinYMin meet">
      <defs><linearGradient id="temp-fill" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--accent)" stopOpacity=".14" /><stop offset="100%" stopColor="var(--accent)" stopOpacity="0" /></linearGradient></defs>
      {[floor, (floor + limit) / 2, limit].map((v, i) => <g key={v}>
        <line x1={left} y1={y(v)} x2={right} y2={y(v)} stroke={i === 2 ? 'var(--danger)' : 'var(--line)'} strokeDasharray={i === 2 ? '4 6' : '2 5'} opacity={i === 2 ? '.5' : '.7'} />
        <text x={right + 12} y={y(v) + 4} className={i === 2 ? 'chart-limit' : 'chart-label'}>{Math.round(v)}°</text>
      </g>)}
      {values.length > 1 && <><polygon points={`${left},${bottom} ${points} ${right},${bottom}`} fill="url(#temp-fill)" /><polyline points={points} fill="none" stroke="var(--accent)" strokeWidth="2.2" strokeLinejoin="round" vectorEffect="non-scaling-stroke" /></>}
      {values.length > 0 && <circle cx={values.length > 1 ? right : left} cy={y(values.at(-1)!)} r="3.2" fill="var(--accent)" />}
      <text x={left} y={size.height - 3} className="chart-label">{state.active ? '最近采样' : '上次会话'}</text><text x={right} y={size.height - 3} textAnchor="end" className="chart-label">{live ? '现在' : '最后采样'}</text>
    </svg>
    {!values.length && <div className="chart-empty"><Activity size={22} /><span>启动保护监控后显示真实采样</span></div>}
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
  const gpu = state.session.gpu || {}
  const live = connected && state.gpu_live
  const memory = gpu.memory_total ? (gpu.memory_used || 0) / gpu.memory_total * 100 : 0
  return <section className="panel guard-panel">
    <div className="panel-heading"><div className="heading-icon mint"><ShieldCheck size={18} /></div><h2>显卡独立保护</h2><span className={`dot ${live ? 'green' : 'gray'}`} /></div>
    <div className="guard-status">{!connected ? '连接中断 · 显示上次采样' : live ? '独立看门狗持续运行' : state.active ? '等待监控更新' : gpu.temperature_c != null ? '上次会话采样 · 当前未监控' : '保护将在会话启动时开启'}</div>
    <div className="temperature">{gpu.temperature_c == null ? '—' : Math.round(gpu.temperature_c)}<span>°C</span><div className="temperature-symbol"><Thermometer size={27} strokeWidth={1.3} /></div></div>
    <div className="guard-device">{gpu.name || state.readiness.backend.device || '等待设备信息'}</div>
    <div className="gpu-substats"><div><span>功耗</span><strong>{gpu.power_w?.toFixed(1) ?? '—'} <small>W</small></strong></div><div><span>GPU 负载</span><strong>{gpu.utilization ?? '—'} <small>%</small></strong></div><div><span>风扇</span><strong>{gpu.fan_percent?.[0] ?? '—'} <small>%</small></strong></div></div>
    <div className="memory-label"><span>显存占用</span><span>{gpu.memory_used == null ? '—' : (gpu.memory_used / 2 ** 30).toFixed(1)} <i>/ {gpu.memory_total ? (gpu.memory_total / 2 ** 30).toFixed(0) : '—'} GB</i></span></div>
    <div className="track thin"><div style={{ width: `${Math.min(memory, 100)}%` }} /></div>
    <div className="guard-rules"><div><span className="rule-dot" />核心 ≥ {state.safety.temperature_limit_c}°C 即停</div><div><span className="rule-dot" />每 {state.safety.poll_seconds * 1000} ms 采样</div><div><span className="rule-dot" />数据超时 {state.safety.telemetry_timeout_seconds} 秒即停</div></div>
    <button className="btn danger" disabled={!enabled} onClick={() => void command('session/stop')}><Square size={14} fill="currentColor" />紧急停止<kbd>F8</kbd></button>
  </section>
}
const Guard = memo(GuardView, (a, b) => a.connected === b.connected && a.command === b.command && a.enabled === b.enabled
  && a.state.session.gpu === b.state.session.gpu && a.state.active === b.state.active
  && a.state.gpu_live === b.state.gpu_live && a.state.safety === b.state.safety
  && a.state.readiness.backend.device === b.state.readiness.backend.device)

function Training({ state, connected, pending, command, navigate }: { state: Snapshot; connected: boolean; pending: boolean; command: Command; navigate: (p: Page) => void }) {
  const [minutes, setMinutes] = useState(state.default_minutes)
  const [goal, setGoal] = useGoal('th06-training-goal', 'survival')
  const [create, setCreate] = useState(false)
  const [dismissedHistory, setDismissedHistory] = useState('')
  const c = controls(state, connected, pending)
  const selected = state.models.find(m => m.id === state.selected_model_id)
  const w = state.session.worker || {}
  const history = !!state.session.time && !state.active
  const historyKey = `${state.run_dir}|${state.session.reason}`
  const budget = state.active ? w.budget_seconds || minutes * 60 : minutes * 60
  const start = async (mode: Mode) => {
    if (await command('sessions', { mode, minutes, model_id: selected?.id, goal })) setCreate(false)
  }
  return <>
    <div className="hero"><div><div className="eyebrow"><span />NEURAL PILOT / WORKSPACE</div><h1>训练控制台<span className="title-period">.</span></h1><p>选择模型、管理训练进度，并查看实时运行状态。</p></div><div className="hero-meta"><button className="btn secondary" disabled={!connected || pending || state.preview} onClick={() => void command('game/focus')}>切回游戏</button><span className="chip"><Crosshair size={13} />Normal · 灵梦 A</span><span className="chip"><Cpu size={13} />PPO · CUDA</span></div></div>
    {(!connected || state.preview || (state.active && w.focus_pending) || (state.session.state === 'stopped' && state.session.reason && dismissedHistory !== historyKey)) && <div className={`banner ${!connected ? 'warning' : ''}`} role="status"><Info size={16} /><span>{!connected ? '后端连接中断，当前显示缓存数据。训练由独立进程继续保护，紧急时按 F8。' : state.preview ? '只读预览 · 可浏览模型与真实历史记录，不启动训练或发送游戏按键。' : state.active && w.focus_pending ? w.message || '等待游戏前台' : `历史记录 · ${date(state.session.time)} · ${state.session.reason}`}</span>{state.active && w.focus_pending ? <button className="btn secondary" disabled={pending || !connected || state.preview} onClick={() => void command('session/focus')}>切回游戏</button> : connected && !state.preview && <button className="icon-btn" aria-label="收起历史记录" onClick={() => setDismissedHistory(historyKey)}><X size={16} /></button>}</div>}
    <GameStatePanel state={state} connected={connected} pending={pending} command={command} />
    <div className="metrics">
      <Metric title={history ? '上次有效训练' : '本次有效训练'} value={duration(w.training_seconds)} icon={Clock3} note={`暂停 ${duration(w.paused_seconds)} · 暂停不计时`} />
      <Metric title={history ? '上次累计采样' : '累计采样步数'} value={number(w.steps ?? selected?.steps ?? 0)} icon={Database} note={`本次新增 ${number(w.session_steps ?? 0)} 步`} />
      <Metric title="参数更新" value={number(w.updates ?? selected?.updates ?? 0)} icon={Sparkles} note="保留已有权重与优化器" />
      <Metric title={history ? '上次回合生存' : '当前回合生存'} value={w.survival_seconds?.toFixed(1) ?? '—'} unit="s" icon={Heart} note={`第 ${number(w.episodes ?? 0)} 回合`} />
    </div>
    <div className="workspace-grid"><div className="workspace-main">
      <section className="panel session-panel"><div className="panel-heading"><div className="heading-icon"><Activity size={18} /></div><h2>训练会话</h2><span className={`badge ${state.active ? 'accent' : ''}`}><span className={`dot ${state.active ? 'green' : 'gray'}`} />{stateLabel(state)}</span></div>
        <div className="session-inner"><div className="field-label">{state.busy ? '当前会话模型' : '准备使用的模型'}<button className="text-btn" onClick={() => navigate('models')}>管理模型<ArrowUpRight size={13} /></button></div>
          <button className="model-select" disabled={state.busy || pending} onClick={() => navigate('models')}><span className="model-icon"><Box size={21} strokeWidth={1.5} /></span><span><strong>{state.active ? w.model_source ? '继续已有模型' : '新建模型' : selected?.name || '尚未选择模型'}</strong><small>{state.active ? w.model_source || '当前使用随机初始化模型' : selected ? `${number(selected.steps)} 步 · ${number(selected.updates)} 次更新 · ${selected.compatibility}` : '前往模型库选择检查点'}</small></span><ChevronDown size={17} /></button>
          <div className="field-label time-label">本次训练时长<span>分钟</span></div><div className="duration-options" role="group" aria-label="训练时长">{[1, 5, 10, 30, 60].map(v => <button key={v} disabled={state.busy || state.preview} className={v === minutes ? 'selected' : ''} onClick={() => setMinutes(v)}>{v}<span> min</span>{v === minutes && <Check size={12} />}</button>)}</div>
          <div className="field-label time-label">训练目标<span>{state.busy ? w.goal === 'nmnb' ? 'NMNB' : '生存练习' : '下次启动生效'}</span></div><div className="goal-options" role="group" aria-label="训练目标">{(['survival','nmnb'] as const).map(value => <button key={value} disabled={state.busy || state.preview} className={(state.busy ? w.goal : goal) === value ? 'selected' : ''} aria-pressed={(state.busy ? w.goal : goal) === value} onClick={() => setGoal(value)}><strong>{value === 'survival' ? '生存练习' : 'NMNB'}</strong><small>{value === 'survival' ? '允许学习危急保命 B' : '禁止 B · Miss 单独计数'}</small></button>)}</div>
          <div className="session-progress"><div><span>{state.active ? '有效训练进度' : '本次计划'}<b>{state.active ? duration(w.training_seconds) : '00:00'} <i>/ {duration(budget)}</i></b></span><span>{state.active ? `${Math.min(100, ((w.training_seconds || 0) / budget * 100)).toFixed(1)}%` : '等待手动开始'}</span></div><div className="track"><div style={{ width: `${state.active ? Math.min(100, (w.training_seconds || 0) / budget * 100) : 0}%` }} /></div></div>
          {state.busy ? <div className="session-actions"><button className="btn primary" disabled={!c.pause} onClick={() => void command('session/pause', { paused: !c.paused })}>{c.paused ? <Play size={16} fill="currentColor" /> : <Pause size={16} fill="currentColor" />}{state.finish_requested ? '正在保存模型…' : !state.active ? '正在启动保护…' : c.paused ? `恢复${state.session.mode === 'play' ? '试玩' : state.session.mode === 'evaluate' ? '评估' : '本次训练'}` : `暂停${state.session.mode === 'play' ? '试玩' : state.session.mode === 'evaluate' ? '评估' : '训练'}`}<kbd>F7</kbd></button><button className="btn secondary" disabled={!c.finish} onClick={() => void command('session/finish')}><ArrowDownToLine size={16} />保存并结束</button></div>
            : <div className="session-actions"><button className="btn primary" disabled={!c.start} onClick={() => void start('resume')}><Play size={15} fill="currentColor" />继续训练<ArrowRight size={16} /></button><button className="btn secondary" disabled={!c.create} onClick={() => setCreate(true)}><Plus size={16} />新建模型</button></div>}
          <div className="session-hint"><ShieldCheck size={13} /><span>{!state.readiness.ready ? '环境检查尚未就绪，请查看保护与诊断。' : state.busy ? (w.message || '独立显卡保护持续运行，暂停保留当前进度。') : '继续训练保留已有权重。启动后将自动切回游戏；请保持英文输入法。'}</span></div>
        </div>
      </section>
      <section className="panel telemetry-panel"><div className="panel-heading"><div className="heading-icon"><Gauge size={18} /></div><h2>显卡遥测</h2><span className="subtle-tag"><span className={`dot ${connected && state.gpu_live ? 'green' : 'gray'}`} />{connected && state.gpu_live ? '实时' : '历史记录'}</span></div><div className="telemetry-top"><div><span className="legend-dot" />核心温度<span className="telemetry-value">{state.session.gpu?.temperature_c ?? '—'}°C</span></div><span>停机阈值 {state.safety.temperature_limit_c}°C</span></div><TemperatureChart state={state} live={connected && state.gpu_live} /></section>
    </div><aside className="workspace-aside"><Guard state={state} command={command} enabled={c.stop} connected={connected} /><section className="panel game-panel"><div className="panel-heading"><div className="heading-icon"><Target size={18} /></div><h2>游戏状态</h2><span className="subtle-tag">{state.active ? '当前回合' : '上次会话'}</span></div><div className="resource-grid"><div><span className="resource-letter power">P</span><strong>{number(w.resources?.power)}</strong><small>火力</small></div><div><span className="resource-letter bomb">B</span><strong>{number(w.resources?.bombs)}</strong><small>炸弹</small></div><div><Heart size={17} className="life" /><strong>{number(w.resources?.lives)}</strong><small>残机</small></div></div><div className="game-footer"><span>炸弹请求 / 实际消耗</span><strong>{w.bomb_requests ?? '—'} / {w.bombs_used ?? '—'}</strong></div></section></aside></div>
    <div className="engine-strip"><div className="engine-icon"><Crosshair size={18} /></div><div><strong>滚动路径规划 <span>v{state.versions.planner}</span></strong><p>连续检查弹幕与敌机本体，综合评估拾取和攻击路线。</p></div><div className="engine-stats"><span>可行方向<strong>{w.planning?.feasible_roots ?? '—'}<i>/ 18</i></strong></span><span>规划耗时<strong>{w.planning?.elapsed_ms?.toFixed(1) ?? '—'}<i>ms</i></strong></span><span>敌机本体<strong>{w.planning?.considered_enemy_bodies ?? '—'}</strong></span></div></div>
    {create && <Dialog title="新建模型训练" close={() => setCreate(false)}><p className="dialog-copy">创建随机初始化的模型，并训练 {minutes} 分钟。已有检查点会继续保留。</p><div className="dialog-note"><Info size={17} />如果希望保留已学到的策略，请使用“继续训练”。</div><div className="dialog-actions"><button className="btn secondary" onClick={() => setCreate(false)}>取消</button><button className="btn primary" disabled={!c.create} onClick={() => void start('train')}><Plus size={15} />创建并开始</button></div></Dialog>}
  </>
}

function ModelsView({ state, connected, pending, command, navigate }: { state: Snapshot; connected: boolean; pending: boolean; command: Command; navigate: (p: Page) => void }) {
  const [goal, setGoal] = useGoal('th06-evaluation-goal', 'nmnb')
  const [search, setSearch] = useState('')
  const [usable, setUsable] = useState(false)
  const [detail, setDetail] = useState<Model | null>(null)
  const c = controls(state, connected, pending)
  const rows = state.models.filter(m => (!usable || m.usable) && `${m.name} ${m.path} ${m.steps} ${number(m.steps)}`.toLowerCase().includes(search.toLowerCase()))
  const select = async (m: Model) => { await command('models/select', { model_id: m.id }) }
  const play = async (mode: Mode) => { if (await command('sessions', { mode, minutes: mode === 'evaluate' ? 60 : 10, model_id: state.selected_model_id, goal })) navigate('train') }
  return <><div className="hero"><div><div className="eyebrow"><span />MODEL REGISTRY</div><h1>模型库<span className="title-period">.</span></h1><p>选择历史检查点继续训练，或单独试玩与评估。</p></div><div className="hero-meta"><span className="chip"><Database size={13} />{state.models.length} 个检查点</span></div></div>
    <div className="model-toolbar"><div className="search-input"><Search size={17} /><input aria-label="搜索模型" value={search} onChange={e => setSearch(e.target.value)} placeholder="搜索模型、步数或文件名…" />{search && <button className="icon-btn" aria-label="清空搜索" onClick={() => setSearch('')}><X size={15} /></button>}</div><div className="filter-options"><button onClick={() => setUsable(false)} className={!usable ? 'selected' : ''}>全部模型</button><button onClick={() => setUsable(true)} className={usable ? 'selected' : ''}>可用模型</button></div></div>
    <section className="panel model-table"><div className="table-scroll"><table><thead><tr><th>模型名称</th><th>累计采样</th><th>参数更新</th><th>保存时间</th><th>兼容性</th><th><span className="sr-only">操作</span></th></tr></thead><tbody>{rows.map(m => <tr key={m.id} className={state.selected_model_id === m.id ? 'selected-row' : ''}><td><button className="model-name-btn" disabled={state.busy || pending || !m.usable} onClick={() => void select(m)}><span className={`table-model-icon ${state.selected_model_id === m.id ? 'active' : ''}`}>{state.selected_model_id === m.id ? <Check size={17} /> : <Box size={17} />}</span><span><strong>{m.name}{state.selected_model_id === m.id && <b className="selected-tag">已选择</b>}</strong><small title={m.path}>{m.path}</small></span></button></td><td className="mono">{number(m.steps)}</td><td className="mono">{number(m.updates)}</td><td>{date(m.saved_at)}</td><td><span className={`compatibility ${m.usable ? '' : 'bad'}`}><span className={`dot ${m.usable ? 'green' : 'gray'}`} />{m.compatibility}</span></td><td><button className="icon-btn" aria-label={`查看${m.name}详情`} onClick={() => setDetail(m)}><Ellipsis size={19} /></button></td></tr>)}</tbody></table></div>{!rows.length && <div className="empty-state"><Database size={30} /><strong>{search ? '没有匹配的模型' : '还没有模型检查点'}</strong><p>{search ? '试试其他步数或文件名。' : '首轮训练保存后，检查点会出现在这里。'}</p>{search && <button className="text-btn" onClick={() => setSearch('')}>清空搜索<ArrowRight size={14} /></button>}</div>}</section>
    <div className="evaluation-goal"><label htmlFor="evaluation-goal">试玩 / 评估目标</label><select id="evaluation-goal" disabled={state.busy || pending || state.preview} value={goal} onChange={e => setGoal(e.target.value as 'survival' | 'nmnb')}><option value="nmnb">NMNB · 禁止 B</option><option value="survival">生存练习 · 允许保命 B</option></select><span>只推理，不更新权重；使用 B 的存活单独统计。</span></div>
    <div className="models-bottom"><div><ShieldCheck size={16} /><span>每次保存自动保留历史检查点，继续训练不会清空已有权重。</span></div><div className="model-actions"><button className="btn secondary" disabled={!c.start} onClick={() => void play('play')}><Eye size={16} />模型试玩</button><button className="btn secondary" disabled={!c.start} onClick={() => void play('evaluate')}><Target size={16} />独立评估 10 局</button><button className="btn primary" onClick={() => navigate('train')}>前往训练<ArrowRight size={16} /></button></div></div>
    {detail && <Dialog title="检查点详情" close={() => setDetail(null)}><div className="detail-model"><Box size={27} /><div><strong>{detail.name}</strong><small>{date(detail.saved_at)}</small></div></div><dl className="detail-list"><div><dt>累计采样</dt><dd>{number(detail.steps)} 步</dd></div><div><dt>参数更新</dt><dd>{number(detail.updates)}</dd></div><div><dt>观测格式</dt><dd>v{detail.format}</dd></div><div><dt>保存时规划版本</dt><dd>{detail.planner_version ? `v${detail.planner_version}` : '未启用'}</dd></div><div><dt>兼容性</dt><dd>{detail.compatibility}</dd></div></dl><div className="file-path"><FolderOpen size={15} /><code>{detail.path}</code></div><div className="dialog-actions"><button className="btn secondary" onClick={() => setDetail(null)}>关闭</button><button className="btn primary" disabled={!detail.usable || state.busy || pending} onClick={async () => { if (await command('models/select', { model_id: detail.id })) setDetail(null) }}><Check size={16} />使用此模型</button></div></Dialog>}
  </>
}
const Models = memo(ModelsView, (a, b) => a.command === b.command && a.navigate === b.navigate
  && a.connected === b.connected && a.pending === b.pending && a.state.models === b.state.models
  && a.state.selected_model_id === b.state.selected_model_id && a.state.busy === b.state.busy
  && a.state.preview === b.state.preview && a.state.readiness.ready === b.state.readiness.ready)

function InputLatencyPanel({ state }: { state: Snapshot }) {
  const timing = state.session.worker?.input_latency
  const shortcut = (key: 'F7' | 'F8') => !state.active ? '会话未运行'
    : state.hotkeys?.[key] === 'key_events' ? '事件监听'
    : state.hotkeys?.[key] === 'poll_5ms' ? '快速轮询 · 5 ms'
    : state.hotkeys?.[key] === 'disabled' ? '本模式不使用' : '正在启用'
  return <section className="panel input-timing-panel">
    <div className="panel-heading"><div className="heading-icon"><Activity size={18} /></div><h2>输入响应</h2><span className="subtle-tag">{state.active ? '当前会话' : '上次会话'}</span></div>
    <div className="input-timing-grid">
      <dl className="safety-rules">
        <div><dt>观测到发键</dt><dd>{milliseconds(timing?.observation_to_send_ms)}</dd></div>
        <div><dt>P95 · 最近 64 次</dt><dd>{milliseconds(timing?.observation_to_send_p95_ms)}</dd></div>
        <div><dt>策略推理与动作传回</dt><dd>{milliseconds(timing?.policy_ms)}</dd></div>
      </dl>
      <dl className="safety-rules">
        <div><dt>按键发送调用</dt><dd>{timing?.input_events === 0 ? '保持原按键' : milliseconds(timing?.send_ms)}</dd></div>
        <div><dt>游戏输入采样</dt><dd>{inputResponse(timing)}</dd></div>
        <div><dt>本次按键事件</dt><dd>{number(timing?.input_events)}</dd></div>
      </dl>
    </div>
    <div className="small-note"><Info size={14} />{!timing && '响应记录从下一次运行开始采集。'}游戏输入采样表示首次读到匹配方向的时刻，实际生效可能更早；未观察到也不会单独停止训练。</div>
    <h3 className="shortcut-title">全局快捷键</h3>
    <div className="input-timing-grid"><dl className="safety-rules">
      <div><dt>F7 暂停 / 恢复</dt><dd>{shortcut('F7')}</dd></div>
      <div><dt>最近处理 / 收到次数</dt><dd>{milliseconds(state.hotkeys?.F7_dispatch_ms)} / {number(state.hotkeys?.F7_count)}</dd></div>
    </dl><dl className="safety-rules">
      <div><dt>F8 紧急停止</dt><dd>{shortcut('F8')}</dd></div>
      <div><dt>最近处理 / 收到次数</dt><dd>{milliseconds(state.hotkeys?.F8_dispatch_ms)} / {number(state.hotkeys?.F8_count)}</dd></div>
    </dl></div>
    {state.active && (state.hotkeys?.listener_error || state.hotkeys?.F7_error || state.hotkeys?.F8_error) && <div className="small-note">{state.hotkeys.listener_error || state.hotkeys.F7_error || state.hotkeys.F8_error}</div>}
  </section>
}

const GAME_PHASES: Record<string, string> = { combat: '关卡进行中', boss: 'Boss 战', pause: '游戏已暂停', dialogue: 'Boss 对话', script_wait: '剧情过场', continue: '续关菜单', result: '结算画面', dead: '自机中弹', title: '主菜单', title_prompt: '标题画面', difficulty: '选择难度', character: '选择自机', shot: '选择机体', run_mode: '选择游戏模式', start_confirm: '确认开始游戏', menu_transition: '菜单切换中' }

function GameStatePanel({ state, connected, pending, command }: { state: Snapshot; connected: boolean; pending: boolean; command: Command }) {
  const w = state.session.worker || {}
  const probe = state.game_probe
  const label = state.active ? GAME_PHASES[w.game_phase || ''] || '等待工作进程识别' : probe?.label || '尚未检测游戏'
  const stage = state.active ? w.stage_label || '关卡待识别' : probe?.stage_label
  return <section className="panel game-state-panel" aria-label="游戏状态">
    <div className="heading-icon mint"><Eye size={17} /></div><div className="game-state-copy"><strong>{stage ? `${stage} · ${label}` : label}</strong><small>{state.active ? `当前关卡 ${duration(w.stage_seconds)} · 来自当前工作进程` : probe ? `只读检测于 ${date(probe.observed_at)} · ${probe.available ? '与历史会话记录无关' : probe.message}` : '可从主菜单开始；开始后识别菜单、对话与战斗。'}</small></div>
    <button className="btn secondary" disabled={!connected || pending || state.busy || state.preview} onClick={() => void command('game/probe')}>检测游戏状态</button>
  </section>
}

function AccelerationPanel({ state, connected, pending, command }: { state: Snapshot; connected: boolean; pending: boolean; command: Command }) {
  const cfg = state.acceleration
  const disabled = !controls(state, connected, pending).diagnostic
  const update = (field: 'collision' | 'input', value: string) => void command('settings/acceleration', { collision: cfg?.collision || 'python', input: cfg?.input || 'python', [field]: value })
  return <section className="panel acceleration-panel"><div className="panel-heading"><div className="heading-icon"><Cpu size={18} /></div><h2>计算后端</h2><span className={`badge ${cfg?.available ? 'accent' : ''}`}>{cfg?.available ? 'Rust 可用' : 'Python 可用'}</span></div>
    <div className="backend-fields">{([{ field: 'collision', name: '碰撞检测' }, { field: 'input', name: '模拟按键' }] as const).map(item => <label key={item.field}>{item.name}<select aria-label={`${item.name}后端`} disabled={disabled} value={cfg?.[item.field] || 'python'} onChange={e => update(item.field, e.target.value)}><option value="auto">自动（Rust 优先）</option><option value="python">Python</option><option value="rust" disabled={!cfg?.available}>Rust · PyO3</option></select></label>)}</div>
    <p className="small-note">切换在下次会话生效。模块耗时与游戏响应延迟分别记录，60Hz 游戏每帧约 16.7ms。</p>
  </section>
}

function Diagnostics({ state, connected, pending, command }: { state: Snapshot; connected: boolean; pending: boolean; command: Command }) {
  const [logs, setLogs] = useState<Logs | null>(null)
  const [logError, setLogError] = useState(false)
  const [tab, setTab] = useState<'worker' | 'launcher'>('worker')
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
  const checks = [{ title: 'GPU 计算环境', okay: state.readiness.checks.gpu, text: state.readiness.backend.device || '尚未自检', icon: <Cpu size={19} />, mode: 'check' as Mode, action: 'GPU 自检' }, { title: '游戏实机校准', okay: state.readiness.checks.calibration, text: '方向键 · Shift · 暂停与重开', icon: <Crosshair size={19} />, mode: 'calibrate' as Mode, action: '重新校准' }, { title: '资源读取', okay: state.readiness.checks.resources, text: '火力 / 炸弹 / 残机', icon: <Database size={19} /> }]
  return <><div className="hero"><div><div className="eyebrow"><span />SYSTEM HEALTH</div><h1>保护与诊断<span className="title-period">.</span></h1><p>查看显卡保护、环境检查和进程日志。</p></div><span className={`chip ${state.readiness.ready ? 'ready-chip' : ''}`}>{state.readiness.ready ? <CheckCheck size={15} /> : <Info size={15} />}{state.readiness.ready ? '训练环境已就绪' : '环境待检查'}</span></div>
    <div className="diagnostic-checks">{checks.map(check => <section className="panel check-card" key={check.title}><div className="check-top"><div className="heading-icon">{check.icon}</div><span className={`badge ${check.okay ? 'accent' : ''}`}>{check.okay ? <Check size={12} /> : <Circle size={10} />}{check.okay ? '已通过' : '待完成'}</span></div><h2>{check.title}</h2><p>{check.text}</p>{check.mode && <button className="text-btn" disabled={!c.diagnostic} onClick={() => check.mode === 'calibrate' ? setCalibrate(true) : void command('sessions', { mode: check.mode, minutes: 1 })}>{check.action}<ArrowUpRight size={13} /></button>}</section>)}</div>
    <div className="diagnostic-grid"><section className="panel safety-panel"><div className="panel-heading"><div className="heading-icon mint"><ShieldCheck size={18} /></div><h2>独立保护规则</h2><button className="text-btn" disabled={!c.diagnostic} onClick={() => void command('sessions', { mode: 'monitor', minutes: 60 })}>仅监控<ArrowUpRight size={13} /></button></div><dl className="safety-rules"><div><dt><Thermometer size={16} />核心温度</dt><dd>≥ {state.safety.temperature_limit_c}°C 停止</dd></div><div><dt><Database size={16} />显存占用</dt><dd>≥ {Math.round(state.safety.vram_fraction_limit * 100)}% 停止</dd></div><div><dt><Activity size={16} />采样间隔</dt><dd>{state.safety.poll_seconds * 1000} ms</dd></div><div><dt><Clock3 size={16} />监控数据超时</dt><dd>{state.safety.telemetry_timeout_seconds} 秒停止</dd></div><div><dt><Heart size={16} />工作进程心跳</dt><dd>{state.safety.worker_heartbeat_seconds} 秒</dd></div></dl><div className="small-note"><Info size={14} />保护进程独立于浏览器和训练循环。显存温度等硬件未提供的读数保持缺失。</div></section>
      <section className="panel planner-panel"><div className="panel-heading"><div className="heading-icon"><Crosshair size={18} /></div><h2>策略与路径</h2><span className="subtle-tag">{state.active ? '实时' : '上次会话'}</span></div><dl className="safety-rules"><div><dt>当前规划引擎</dt><dd>v{state.versions.planner}</dd></div><div><dt>观测 / 奖励版本</dt><dd>v{state.versions.observation} / v{state.versions.reward}</dd></div><div><dt>近场 / 全场 / 已读子弹</dt><dd>{w.bullet_observation?.near ?? '—'} / {w.bullet_observation?.full ?? '—'} / {w.bullet_observation?.visible ?? '—'}</dd></div><div><dt>本体障碍 / 几何未知</dt><dd>{w.planning?.considered_enemy_bodies ?? '—'} / {w.planning?.unknown_body_geometry ?? '—'}</dd></div><div><dt>预测路线状态</dt><dd>{w.planning?.enabled ? w.planning.all_routes_risky ? '所有搜索路线均有风险' : '已找到预测可行路线' : '尚无记录'}</dd></div></dl></section></div>
    <InputLatencyPanel state={state} />
    <GameStatePanel state={state} connected={connected} pending={pending} command={command} />
    <AccelerationPanel state={state} connected={connected} pending={pending} command={command} />
    <section className="panel logs-panel"><div className="panel-heading"><div className="heading-icon"><FileText size={18} /></div><h2>运行日志</h2><div className="logs-tabs"><button className={tab === 'worker' ? 'selected' : ''} onClick={() => setTab('worker')}>训练进程</button><button className={tab === 'launcher' ? 'selected' : ''} onClick={() => setTab('launcher')}>启动记录</button></div><button className="icon-btn" aria-label="复制日志" disabled={!logs?.[tab]} onClick={async () => { try { await navigator.clipboard.writeText(logs?.[tab] || ''); setCopied(true); setTimeout(() => setCopied(false), 2000) } catch { setCopied(false) } }}>{copied ? <Check size={16} /> : <Copy size={16} />}</button></div><div className="log-location"><FolderOpen size={13} /><span>{logs?.run_dir || '暂无运行目录'}</span></div>{logError && <div className="log-error">日志连接中断，正在重试。下方为上次读取内容。</div>}<pre className="log-content">{logs?.[tab] || '暂无日志。启动会话后，这里会显示真实运行记录。'}</pre></section>
    {calibrate && <Dialog title="重新实机校准" close={() => setCalibrate(false)}><p className="dialog-copy">校准会向游戏发送移动、暂停与重开按键。已有校准通过时，无需重复操作。</p><div className="dialog-note"><Crosshair size={17} />请先将游戏置于 Normal 第一关并暂停，菜单选择“解除暂停”。启动后切回游戏。</div><div className="dialog-actions"><button className="btn secondary" onClick={() => setCalibrate(false)}>取消</button><button className="btn primary" disabled={!c.diagnostic} onClick={async () => { if (await command('sessions', { mode: 'calibrate', minutes: 1 })) setCalibrate(false) }}>开始校准<ArrowRight size={15} /></button></div></Dialog>}
  </>
}

export default function App() {
  const { state, connected, pending, serverClosed, command, notice, dismiss } = useConsole()
  const [exitDialog, setExitDialog] = useState(false)
  const closing = !serverClosed && ['waiting', 'stopping'].includes(state?.shutdown?.state || '')
  const closeServer = async () => { if (await command('server/shutdown')) setExitDialog(false) }
  const [page, setPage] = useState<Page>(() => (['train', 'models', 'diagnostics'].includes(location.hash.slice(1)) ? location.hash.slice(1) : 'train') as Page)
  const [sidebar, setSidebar] = useState(false)
  const [light, setLight] = useState(() => localStorage.getItem('th06-theme') === 'light')
  useEffect(() => { document.documentElement.dataset.theme = light ? 'light' : 'dark'; localStorage.setItem('th06-theme', light ? 'light' : 'dark') }, [light])
  useEffect(() => { const change = () => { const value = location.hash.slice(1); if (['train', 'models', 'diagnostics'].includes(value)) setPage(value as Page) }; window.addEventListener('hashchange', change); return () => window.removeEventListener('hashchange', change) }, [])
  useEffect(() => { document.title = `${TITLES[page]} · TH06NC-Neural-Pilot`; window.scrollTo(0, 0) }, [page])
  useEffect(() => {
    // The independent Windows guard handles these global keys. Suppress
    // browser caret/debugger shortcuts without toggling the training twice.
    const guardKeys = (event: KeyboardEvent) => { if (event.key === 'F7' || event.key === 'F8') event.preventDefault() }
    window.addEventListener('keydown', guardKeys)
    return () => window.removeEventListener('keydown', guardKeys)
  }, [])
  const navigate = useCallback((next: Page) => { setPage(next); location.hash = next; setSidebar(false) }, [])
  return <div className="app-shell">
    {sidebar && <button className="sidebar-backdrop" aria-label="收起导航" onClick={() => setSidebar(false)} />}
    <aside className={`sidebar ${sidebar ? 'open' : ''}`}><a className="brand" href="#train" onClick={() => navigate('train')}><Logo /><span>TH06NC<small>NEURAL PILOT</small></span></a><div className="sidebar-caption">工作空间</div><nav aria-label="主导航">{([{ id: 'train', icon: LayoutDashboard, text: '训练控制台' }, { id: 'models', icon: Database, text: '模型库' }, { id: 'diagnostics', icon: ShieldCheck, text: '保护与诊断' }] as const).map(n => <button key={n.id} className={page === n.id ? 'active' : ''} aria-current={page === n.id ? 'page' : undefined} onClick={() => navigate(n.id)}><n.icon size={18} strokeWidth={1.7} />{n.text}{page === n.id && <span className="nav-dot" />}</button>)}</nav>
      <div className="sidebar-bottom"><div className="local-card"><span className={`dot ${connected ? 'green' : 'gray'}`} /><span>本地工作空间<small>{serverClosed ? '后端已关闭' : closing ? '正在关闭服务' : connected ? 'Python 后端已连接' : '等待后端连接'}</small></span><Cpu size={17} /></div><div className="sidebar-version"><span>TH06NC AI</span><span>v1.0</span></div></div></aside>
    <div className="main-shell"><header className="topbar"><button className="icon-btn mobile-menu" aria-label="展开导航" onClick={() => setSidebar(!sidebar)}><Menu size={19} /></button><div className="breadcrumb"><span>工作空间</span><ChevronRight size={13} /><strong>{TITLES[page]}</strong></div><div className="topbar-right">{state?.busy && !serverClosed && <button className="btn danger header-stop" disabled={!controls(state, connected, pending).stop} onClick={() => void command('session/stop')}><Square size={11} fill="currentColor" />紧急停止<kbd>F8</kbd></button>}<span className={`connection ${connected ? '' : 'offline'}`}><span className={`dot ${connected ? 'green' : 'gray'}`} />{serverClosed ? '后端已关闭' : closing ? '正在关闭' : connected ? '本机已连接' : '连接中断'}</span><span className="topbar-separator" /><button className="icon-btn theme-btn" aria-label={light ? '切换深色主题' : '切换浅色主题'} onClick={() => setLight(!light)}>{light ? <Moon size={17} /> : <Sun size={17} />}</button><button className="btn secondary server-close" aria-label="关闭服务" title="结束会话并关闭后端" disabled={!connected || pending || closing || serverClosed} onClick={() => state?.busy ? setExitDialog(true) : void closeServer()}><Power size={15} /><span>关闭服务</span></button></div></header>
      <main>{serverClosed || closing ? <div className="connecting lifecycle-status" role="status">{serverClosed ? <Power size={32} /> : <LoaderCircle size={32} className="spin" />}<h1>{serverClosed ? '后端已关闭' : '正在结束会话并关闭服务'}</h1><p>{serverClosed ? '下次双击「启动AI.cmd」即可重新打开。' : '训练会先保存，显卡保护保持运行。请等待完成。'}</p><small>{serverClosed ? '现在可以关闭此页面' : '紧急时仍可按 F8 停止'}</small></div> : state ? <>{state.shutdown?.state === 'error' && <div className="banner warning" role="alert"><Info size={16} /><span>{state.shutdown.message}</span></div>}{page === 'train' ? <Training state={state} connected={connected} pending={pending} command={command} navigate={navigate} /> : page === 'models' ? <Models state={state} connected={connected} pending={pending} command={command} navigate={navigate} /> : <Diagnostics state={state} connected={connected} pending={pending} command={command} />}</> : <div className="connecting"><Logo /><h1>连接训练后端</h1><p>正在读取模型与运行状态，不会自动开始训练。</p><LoaderCircle size={21} className="spin" /><small>双击「启动AI.cmd」打开服务</small></div>}</main>
      <footer><div><span className="footer-dot" />{serverClosed ? '服务已退出' : state?.active ? '后台会话正在独立运行' : '所有计算留在本机'}<span className="footer-divider">/</span><span>{serverClosed ? '本机端口正在释放' : '退出程序请点右上角「关闭服务」；关闭网页会保留会话'}</span></div><div><kbd>F7</kbd>暂停 / 恢复<kbd>F8</kbd>紧急停止</div></footer>
    </div>
    {exitDialog && <Dialog title="结束会话并关闭服务" close={() => setExitDialog(false)}><p className="dialog-copy">训练会先保存并结束，再关闭后端。模型试玩、评估和监控也会结束。</p><div className="dialog-actions"><button className="btn secondary" onClick={() => setExitDialog(false)}>取消</button><button className="btn primary" disabled={pending || !connected} onClick={() => void closeServer()}><Power size={15} />结束并关闭</button></div></Dialog>}
    {notice && <div className={`toast ${notice.error ? 'error' : ''}`} role={notice.error ? 'alert' : 'status'}>{notice.error ? <TriangleAlert size={18} /> : <Check size={18} />}<span>{notice.text}</span><button className="icon-btn" aria-label="关闭提示" onClick={dismiss}><X size={15} /></button></div>}
  </div>
}
