import { translator } from './i18n'
import type { Locale } from './i18n'
import type { InputLatency, Snapshot } from './types'

export const milliseconds = (value?: number | null) => value == null || !Number.isFinite(value) ? '—' : `${value.toFixed(2)} ms`
export function inputResponse(timing?: InputLatency, locale: Locale = 'zh-CN') {
  const message = translator(locale)
  if (timing?.game_input_state === 'held') return message("console.holding_the_current_direction")
  if (timing?.game_input_state === 'matched' && timing.game_input_match_ms != null) return message("console.observed_within_0", milliseconds(timing.game_input_match_ms))
  if (timing?.game_input_state === 'not_observed') return message("console.matching_input_was_not_observed_this_step")
  if (timing?.game_input_state === 'unavailable') return message("console.input_bits_unavailable")
  if (timing?.game_input_state === 'pending') return message("console.waiting_for_an_input_sample")
  return message("console.no_latency_samples_yet")
}

export const number = (value: number | undefined | null, locale: Locale = 'zh-CN') => value == null ? '—' : value.toLocaleString(locale)
export function duration(seconds?: number | null) {
  const s = Math.max(0, Math.floor(seconds || 0))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}
export function date(value?: string | null, locale: Locale = 'zh-CN') {
  if (!value || !Number.isFinite(Date.parse(value))) return '—'
  return new Intl.DateTimeFormat(locale, { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit',
    hour: '2-digit', minute: '2-digit', hour12: false }).format(new Date(value))
}
export function controls(state: Snapshot, connected: boolean, pending: boolean) {
  const model = state.models.find(m => m.id === state.selected_model_id)
  const closing = ['waiting', 'stopping', 'closed'].includes(state.shutdown?.state || '')
  const enabled = connected && !pending && !state.preview && !closing
  const training = state.active && state.session.mode === 'train'
  const game = state.active && ['train', 'play', 'evaluate'].includes(state.session.mode || '')
  return {
    start: enabled && !state.busy && state.readiness.ready && !!model?.usable,
    create: enabled && !state.busy && state.readiness.ready,
    pause: enabled && game && !state.finish_requested,
    finish: enabled && training && !state.finish_requested,
    stop: connected && !pending && !state.preview && (state.busy || ['starting', 'running', 'paused', 'finishing', 'monitoring'].includes(state.session.state || '')),
    diagnostic: enabled && !state.busy,
    paused: state.pause_requested || state.session.worker?.user_paused === true || state.session.state === 'paused',
  }
}
export function stateLabel(s: Snapshot, locale: Locale = 'zh-CN') {
  const message = translator(locale)
  if (s.preview) return message("console.read_only_preview")
  if (s.busy && !s.active) return message("console.starting")
  if (s.active && s.session.worker?.focus_pending) return message("console.waiting_for_game_focus")
  if (s.active) return ({ paused: message("console.paused"), starting: message("console.watchdog_preflight"), finishing: message("console.saving"), monitoring: message("console.monitoring") } as Record<string, string>)[s.session.state || ''] || message("console.running")
  return ({ complete: message("console.complete"), stopped: message("console.stopped") } as Record<string, string>)[s.session.state || ''] || message("console.ready")
}
