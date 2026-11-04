import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react'
import { Copy, Crosshair, ExternalLink } from 'lucide-react'
import { actionDirection, drawBattleFrame, parseBattlePacket, RISK_COLORS } from './battleFrame'
import type { BattleFrame, BattlePacket } from './battleFrame'
import { translator, useI18n } from './i18n'
import type { Locale } from './i18n'
import './battle-view.css'

const DIRECTIONS = ['console.battle_still', 'console.battle_up', 'console.battle_down',
  'console.battle_left', 'console.battle_right', 'console.battle_up_left', 'console.battle_up_right',
  'console.battle_down_left', 'console.battle_down_right'] as const

function useBattleView(enabled: boolean) {
  const [packet, setPacket] = useState<BattlePacket>({ state: 'waiting' })
  useEffect(() => {
    if (!enabled) return
    let source: EventSource | undefined
    let expires: ReturnType<typeof setTimeout> | undefined
    const close = () => {
      source?.close()
      source = undefined
      clearTimeout(expires)
    }
    const subscribe = () => {
      close()
      setPacket({ state: 'waiting' })
      if (document.hidden) return
      const connection = new EventSource('/api/battle-view/events')
      source = connection
      connection.onerror = () => {
        if (source !== connection) return
        clearTimeout(expires)
        setPacket({ state: 'disconnected' })
      }
      connection.onmessage = event => {
        if (source !== connection) return
        clearTimeout(expires)
        try {
          const next = parseBattlePacket(event.data)
          if (next.state !== 'live') {
            setPacket(next)
            return
          }
          const remaining = Math.min(350 - next.age_ms, next.expires_at_ms - Date.now())
          if (remaining <= 0) {
            setPacket({ state: 'waiting' })
            return
          }
          setPacket(next)
          expires = setTimeout(() => setPacket({ state: 'waiting' }), remaining)
        } catch {
          setPacket({ state: 'waiting' })
        }
      }
    }
    subscribe()
    document.addEventListener('visibilitychange', subscribe)
    return () => {
      close()
      document.removeEventListener('visibilitychange', subscribe)
    }
  }, [enabled])
  return packet
}

function Battlefield({ frame }: { frame: BattleFrame | null }) {
  const surface = useRef<HTMLCanvasElement>(null)
  const latest = useRef<BattleFrame | null>(null)
  const paint = useCallback(() => {
    const canvas = surface.current
    if (!canvas) return
    const width = canvas.clientWidth, height = canvas.clientHeight
    const ratio = Math.min(window.devicePixelRatio || 1, 2)
    if (canvas.width !== Math.round(width * ratio)) canvas.width = Math.round(width * ratio)
    if (canvas.height !== Math.round(height * ratio)) canvas.height = Math.round(height * ratio)
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0)
    ctx.clearRect(0, 0, width, height)
    if (latest.current) drawBattleFrame(ctx, latest.current, width, height)
  }, [])
  useLayoutEffect(() => {
    latest.current = frame
    paint()
  }, [frame, paint])
  useEffect(() => {
    const observer = new ResizeObserver(paint)
    if (surface.current) observer.observe(surface.current)
    return () => observer.disconnect()
  }, [paint])
  return <canvas ref={surface} aria-hidden="true" />
}

function BattleView({ locale }: { locale: Locale }) {
  const packet = useBattleView(true)
  const t = translator(locale)
  const frame = packet.state === 'live' ? packet.frame : null
  const chosen = frame ? actionDirection(frame.action) : null
  const input = frame?.input === 'matched' ? 'console.battle_matched'
    : frame?.input === 'held' ? 'console.battle_held' : 'console.battle_unconfirmed'
  return <div className="battle-view">
    <header className="battle-header">
      <div><span className="battle-kicker">NEURAL PILOT</span><h2>{t('console.battle_title')}</h2></div>
      {packet.state === 'live' && <span className="battle-chip">{t('console.battle_age', packet.age_ms)}</span>}
    </header>
    <div className="battle-viewport">
      <Battlefield frame={frame} />
      {!frame && <div className="battle-empty" role="status"><Crosshair size={22} />
        <span>{t(packet.state === 'off' ? 'console.battle_off'
          : packet.state === 'disconnected' ? 'console.battle_disconnected' : 'console.battle_waiting')}</span>
      </div>}
    </div>
    <div className="battle-readout">
      {frame && chosen ? <>
        <strong>{t('console.battle_selected', t(DIRECTIONS[chosen.direction]))}</strong>
        <span>{t(chosen.focused ? 'console.battle_focus' : 'console.battle_fast')}{chosen.bomb ? ' · X' : ''} · {t(input)}</span>
        <span>{frame.stage === 7 ? 'EX' : frame.stage !== null ? t('console.battle_stage', frame.stage) : '—'} · {t('console.battle_tick', frame.tick)}
          {frame.backend ? ' · ' + frame.backend : ''}
          {frame.candidates[chosen.move][1] !== null && ' · ' + t('console.battle_score',
            frame.candidates[chosen.move][1]?.toFixed(2))}</span>
      </> : <span>{t('console.battle_sampling_note')}</span>}
    </div>
    <div className="battle-legend">
      <span><i style={{ background: RISK_COLORS.clear }} />{t('console.battle_clear')}</span>
      <span><i style={{ background: RISK_COLORS.risk }} />{t('console.battle_risk')}</span>
      <span><i style={{ background: RISK_COLORS.fallback }} />{t('console.battle_fallback')}</span>
      <span><i style={{ background: RISK_COLORS.unknown }} />{t('console.battle_unknown')}</span>
    </div>
    <p className="battle-caption">{t('console.battle_geometry_legend')}</p>
  </div>
}

export function BattleViewPage() {
  const { locale } = useI18n()
  const requested = new URLSearchParams(window.location.search).get('lang')
  const selected = requested === 'en' || requested === 'zh-CN' ? requested : locale
  useEffect(() => {
    document.title = translator(selected)('console.battle_title')
  }, [selected])
  return <div className="battle-page" lang={selected}><BattleView locale={selected} /></div>
}

export function BattleViewPanel({ enabled, available, toggle }: {
  enabled: boolean; available: boolean; toggle: () => void
}) {
  const { t, locale } = useI18n()
  const [copied, setCopied] = useState(false)
  const url = new URL('/battle-view', window.location.origin)
  url.searchParams.set('lang', locale)
  return <section className="panel battle-panel">
    <div className="panel-heading"><div className="heading-icon"><Crosshair size={18} /></div>
      <h2>{t('console.battle_title')}</h2>
      <button type="button" className="btn secondary" role="switch" aria-checked={enabled}
        aria-label={t('console.battle_title')} disabled={!available} onClick={toggle}>
        {t(enabled ? 'console.battle_turn_off' : 'console.battle_turn_on')}
      </button>
    </div>
    <div className="battle-panel-content">
      <div className="battle-instructions">
        <p>{t('console.battle_obs_help')}</p>
        <div className="battle-link">
          <input value={url.href} readOnly aria-label={t('console.battle_obs_url')} onFocus={event => event.currentTarget.select()} />
          <button type="button" className="icon-btn" aria-label={t('console.battle_copy_url')}
            onClick={async () => {
              try { await navigator.clipboard.writeText(url.href); setCopied(true) } catch { setCopied(false) }
            }}><Copy size={16} /></button>
          <a className="icon-btn" href={url.href} target="_blank" rel="noreferrer"
            aria-label={t('console.battle_open_page')}><ExternalLink size={16} /></a>
        </div>
        {copied && <small role="status">{t('console.battle_url_copied')}</small>}
      </div>
      {enabled && <BattleView locale={locale} />}
    </div>
  </section>
}
