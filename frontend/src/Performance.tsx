import { Activity, ArrowUpRight, Cpu, Gauge } from 'lucide-react'
import { memo, useEffect, useId, useLayoutEffect, useRef, useState } from 'react'
import { useI18n } from './i18n'
import type { Snapshot } from './types'
import './performance.css'

type CpuSample = {
  status: 'warming' | 'live' | 'unavailable'
  usage_percent: number | null
  busiest_core_percent: number | null
  logical_processors: number | null
  history: number[]
}

function useCpuSample(connected: boolean) {
  const [sample, setSample] = useState<CpuSample | null>(null)
  const [issue, setIssue] = useState<'reload' | 'connection' | null>(null)
  useEffect(() => {
    if (!connected) return
    let request: AbortController | undefined
    let unsupported = false
    async function poll() {
      if (document.hidden || request || unsupported) return
      const controller = new AbortController()
      request = controller
      const timeout = window.setTimeout(() => controller.abort('timeout'), 3000)
      try {
        const response = await fetch('/api/performance', { signal: controller.signal })
        if (controller.signal.aborted) return
        if (response.status === 404 || response.status === 405 ||
            (response.ok && !response.headers.get('content-type')?.includes('application/json'))) {
          unsupported = true
          setIssue('reload')
          return
        }
        if (!response.ok) throw new Error('CPU request failed')
        const { cpu } = await response.json() as { cpu: CpuSample }
        if (!controller.signal.aborted) {
          setSample(cpu)
          setIssue(null)
        }
      } catch {
        if (!controller.signal.aborted || controller.signal.reason === 'timeout') setIssue('connection')
      } finally {
        window.clearTimeout(timeout)
        request = undefined
      }
    }
    const visibility = () => { if (document.hidden) request?.abort(); else void poll() }
    const timer = window.setInterval(() => void poll(), 1000)
    void poll()
    document.addEventListener('visibilitychange', visibility)
    return () => {
      window.clearInterval(timer)
      request?.abort()
      document.removeEventListener('visibilitychange', visibility)
    }
  }, [connected])
  return { sample, issue }
}

type TrendProps = {
  values: number[]; min: number; max: number; limit?: number; suffix: string;
  label: string; empty: string; start: string; end: string;
}

function Trend({ values, min, max, limit, suffix, label, empty, start, end }: TrendProps) {
  const gradient = useId()
  const container = useRef<HTMLDivElement>(null)
  const [size, setSize] = useState({ width: 480, height: 178 })
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
  const left = 5
  const right = Math.max(left, size.width - 46)
  const bottom = size.height - 26
  const y = (value: number) => bottom - (value - min) / (max - min) * (bottom - 14)
  const points = values.map((value, i) => `${left + i * (right - left) / Math.max(1, values.length - 1)},${y(value)}`).join(' ')
  const top = limit ?? max
  return <div ref={container} className="chart" role="img" aria-label={values.length ? label : empty}>
    <svg viewBox={`0 0 ${size.width} ${size.height}`} aria-hidden="true">
      <defs><linearGradient id={gradient} x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stopColor="var(--trend-color)" stopOpacity=".16" /><stop offset="100%" stopColor="var(--trend-color)" stopOpacity="0" /></linearGradient></defs>
      {[min, (min + top) / 2, top].map((value, i) => {
        const threshold = i === 2 && limit != null
        return <g key={i}>
          <line x1={left} y1={y(value)} x2={right} y2={y(value)} stroke={threshold ? 'var(--danger)' : 'var(--line)'} strokeDasharray={threshold ? '4 6' : '2 5'} />
          <text x={right + 8} y={y(value) + 4} className={threshold ? 'chart-limit' : 'chart-label'}>{Math.round(value)}{suffix}</text>
        </g>
      })}
      {values.length > 1 && <><polygon points={`${left},${bottom} ${points} ${right},${bottom}`} fill={`url(#${gradient})`} /><polyline points={points} fill="none" stroke="var(--trend-color)" strokeWidth="2.2" strokeLinejoin="round" vectorEffect="non-scaling-stroke" /></>}
      {values.length > 0 && <circle cx={values.length > 1 ? right : left} cy={y(values.at(-1)!)} r="3" fill="var(--trend-color)" />}
      <text x={left} y={size.height - 3} className="chart-label">{start}</text>
      <text x={right} y={size.height - 3} textAnchor="end" className="chart-label">{end}</text>
    </svg>
    {!values.length && <div className="chart-empty"><Activity size={20} /><span>{empty}</span></div>}
  </div>
}

const number = (value: number | null | undefined, digits = 0, unit = '') =>
  value == null || !Number.isFinite(value) ? '—' : `${value.toFixed(digits)}${unit}`

function CpuPanel({ connected }: { connected: boolean }) {
  const { t } = useI18n()
  const { sample, issue } = useCpuSample(connected)
  const live = connected && !issue && sample?.status === 'live'
  const caption = !connected || issue === 'connection' ? t('console.disconnected')
    : issue === 'reload' ? t('console.performance_restart_backend')
      : sample?.status === 'unavailable' ? t('console.cpu_reading_unavailable')
        : live ? t('console.live') : t('console.collecting_cpu_samples')
  const values = sample?.history ?? []
  const badge = live ? t('console.live') : values.length ? t('console.last_sample') : t('console.waiting_for_telemetry')
  return <section className="panel performance-card performance-cpu">
    <div className="panel-heading"><div className="heading-icon"><Cpu size={18} /></div><h2>{t('console.cpu_utilization')}</h2><span className="subtle-tag"><span className={`dot ${live ? 'green' : 'gray'}`} />{badge}</span></div>
    <div className="performance-reading"><div><span>{t('console.system_cpu_usage')}</span><strong>{number(sample?.usage_percent, 1)}<small>%</small></strong></div><div><span>{t('console.busiest_logical_processor')}</span><strong>{number(sample?.busiest_core_percent, 1)}<small>%</small></strong></div></div>
    <Trend values={values} min={0} max={100} suffix="%" label={t('console.cpu_usage_trend')}
      empty={caption} start={t('console.recent_samples')} end={live ? t('console.now') : t('console.last_sample')} />
    <dl className="performance-details">
      <div><dt>{t('console.logical_processor_count')}</dt><dd>{number(sample?.logical_processors)}</dd></div>
      <div><dt>{t('console.sampling_interval')}</dt><dd>1 s</dd></div>
    </dl>
    <div className="performance-caption"><span>{values.length > 0 && !live ? caption : t('console.cpu_scope_note')}</span></div>
  </section>
}

type GpuPanelProps = { gpu: Snapshot['session']['gpu']; history: Snapshot['gpu_history']; limit: number;
  live: boolean; active: boolean; connected: boolean; diagnostics: () => void }

const GpuPanel = memo(function GpuPanel({ gpu, history, limit, live, active, connected, diagnostics }: GpuPanelProps) {
  const { t } = useI18n()
  const values = history.map(point => point.temperature_c).filter(Number.isFinite)
  const available = values.length > 0 || gpu?.temperature_c != null
  const status = !connected ? t('console.disconnected') : live ? t('console.live')
    : active ? t('console.waiting_for_telemetry') : available ? t('console.history') : t('console.no_gpu_samples_yet')
  const memory = `${number(gpu?.memory_used == null ? null : gpu.memory_used / 2 ** 30, 1)} / ${number(gpu?.memory_total == null ? null : gpu.memory_total / 2 ** 30, 1, ' GB')}`
  const fans = gpu?.fan_percent?.length ? gpu.fan_percent.map(value => number(value, 0, '%')).join(' / ') : '—'
  return <section className="panel performance-card performance-gpu">
    <div className="panel-heading"><div className="heading-icon mint"><Gauge size={18} /></div><h2>{t('console.gpu_telemetry')}</h2><span className="subtle-tag"><span className={`dot ${live ? 'green' : 'gray'}`} />{status}</span></div>
    <div className="performance-reading"><div><span>{t('console.core_temperature')}</span><strong>{number(gpu?.temperature_c)}<small>°C</small></strong></div><div><span>{t('console.gpu_load')}</span><strong>{number(gpu?.utilization)}<small>%</small></strong></div></div>
    <Trend values={values} min={Math.min(30, limit - 5, ...values)} max={Math.max(limit + 5, ...values)} limit={limit} suffix="°"
      label={t('console.gpu_temperature_chart_with_0_recorded_samples', values.length)}
      empty={t('console.samples_appear_after_monitoring_starts')} start={active ? t('console.recent_samples') : t('console.previous_session')} end={live ? t('console.now') : t('console.last_sample')} />
    <dl className="performance-details">
      <div><dt>{t('console.power_draw')}</dt><dd>{number(gpu?.power_w, 1, ' W')}</dd></div>
      <div><dt>{t('console.vram_usage')}</dt><dd>{memory}</dd></div>
      <div><dt>{t('console.fans')}</dt><dd>{fans}</dd></div>
    </dl>
    <div className="performance-caption"><span>{gpu?.name ? `${gpu.name} · ` : ''}{t('console.core_limit', limit)}</span><button className="text-btn" onClick={diagnostics}>{t('console.safety_diagnostics')}<ArrowUpRight size={13} /></button></div>
  </section>
})

export function Performance({ state, connected, diagnostics }: { state: Snapshot; connected: boolean; diagnostics: () => void }) {
  const { t } = useI18n()
  return <>
    <div className="hero"><div><div className="eyebrow"><span />NEURAL PILOT / TELEMETRY</div><h1>{t('console.performance_monitor')}<span className="title-period">.</span></h1><p>{t('console.performance_description')}</p></div><div className="hero-meta"><span className="chip"><Activity size={13} />{t('console.cpu_sample_rate')}</span></div></div>
    <div className="performance-grid">
      <CpuPanel connected={connected} />
      <GpuPanel gpu={state.session.gpu} history={state.gpu_history} limit={state.safety.temperature_limit_c}
        live={connected && state.gpu_live} active={state.active} connected={connected} diagnostics={diagnostics} />
    </div>
  </>
}
