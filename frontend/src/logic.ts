import type { InputLatency, Snapshot } from './types'

export const milliseconds = (value?: number | null) => value == null || !Number.isFinite(value) ? '—' : `${value.toFixed(2)} ms`
export function inputResponse(timing?: InputLatency) {
  if (timing?.game_input_state === 'held') return '原方向保持中'
  if (timing?.game_input_state === 'matched' && timing.game_input_match_ms != null) return `${milliseconds(timing.game_input_match_ms)} 内观察到`
  if (timing?.game_input_state === 'not_observed') return '本步未观察到匹配输入'
  if (timing?.game_input_state === 'unavailable') return '暂无输入位读数'
  if (timing?.game_input_state === 'pending') return '等待输入采样'
  return '暂无延迟采样'
}

export const number = (value: number | undefined | null) => value == null ? '—' : value.toLocaleString('zh-CN')
export function duration(seconds?: number) {
  const s = Math.max(0, Math.floor(seconds || 0))
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`
}
export function date(value?: string | null) {
  if (!value || !Number.isFinite(Date.parse(value))) return '—'
  return new Intl.DateTimeFormat('zh-CN', { timeZone: 'Asia/Shanghai', month: '2-digit', day: '2-digit',
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
export function stateLabel(s: Snapshot) {
  if (s.preview) return '只读预览'
  if (s.busy && !s.active) return '正在启动'
  if (s.active && s.session.worker?.focus_pending) return '等待游戏前台'
  if (s.active) return ({ paused: '已暂停', starting: '保护预检中', finishing: '正在保存', monitoring: '仅监控中' } as Record<string, string>)[s.session.state || ''] || '运行中'
  return ({ complete: '已完成', stopped: '已停止' } as Record<string, string>)[s.session.state || ''] || '待开始'
}
