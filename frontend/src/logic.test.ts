import { describe, expect, it } from 'vitest'
import { controls, date, duration, inputResponse, milliseconds, stateLabel } from './logic'
import type { Snapshot } from './types'

const base = {
  active: false, busy: false, preview: false, pause_requested: false, finish_requested: false,
  readiness: { ready: true }, selected_model_id: 'old', models: [{ id: 'old', usable: true }], session: {},
} as Snapshot

describe('input response readings', () => {
  it('allows the same pause action in training, play and evaluation', () => {
    for (const mode of ['train', 'play', 'evaluate']) {
      expect(controls({ ...base, active: true, session: { mode } }, true, false).pause).toBe(true)
    }
  })
  it('labels pending activation as waiting rather than claiming the game is paused', () => {
    expect(stateLabel({ ...base, active: true, session: { state: 'paused', worker: { focus_pending: true } } })).toBe('等待游戏前台')
  })
  it('distinguishes unknown readings from a measured zero', () => {
    expect(milliseconds(null)).toBe('—')
    expect(milliseconds(0)).toBe('0.00 ms')
    expect(inputResponse()).toBe('暂无延迟采样')
    expect(inputResponse({ game_input_state: 'unavailable' })).toBe('暂无输入位读数')
  })
  it('reports a sampled match without claiming that held keys were newly sent', () => {
    expect(inputResponse({ game_input_state: 'matched', game_input_match_ms: 0 })).toBe('0.00 ms 内观察到')
    expect(inputResponse({ game_input_state: 'held', game_input_match_ms: null })).toBe('原方向保持中')
    expect(inputResponse({ game_input_state: 'not_observed' })).toBe('本步未观察到匹配输入')
  })
})

describe('training control states', () => {
  it('blocks new work during service shutdown while keeping emergency stop available', () => {
    const s = { ...base, active: true, busy: true, session: { mode: 'train' }, shutdown: { state: 'waiting', message: '' } } as Snapshot
    const c = controls(s, true, false)
    for (const key of ['start', 'create', 'pause', 'finish', 'diagnostic'] as const) expect(c[key]).toBe(false)
    expect(c.stop).toBe(true)
  })
  it('continues an existing checkpoint only when ready and connected', () => {
    expect(controls(base, true, false).start).toBe(true)
    expect(controls({ ...base, selected_model_id: null }, true, false).start).toBe(false)
    expect(controls(base, false, false).start).toBe(false)
    expect(controls(base, true, true).start).toBe(false)
  })
  it('preview disables every GPU and game control', () => {
    const c = controls({ ...base, preview: true, busy: true, active: true, session: { mode: 'train' } }, true, false)
    for (const key of ['start', 'create', 'pause', 'finish', 'stop', 'diagnostic'] as const) expect(c[key]).toBe(false)
  })
  it('pause/resume preserves current session rather than issuing another start', () => {
    const running = { ...base, active: true, busy: true, session: { mode: 'train', state: 'running' } }
    const c = controls(running, true, false)
    expect(c.start).toBe(false)
    expect(c.pause).toBe(true)
    expect(c.finish).toBe(true)
    expect(c.paused).toBe(false)
    expect(controls({ ...running, pause_requested: true }, true, false).paused).toBe(true)
    expect(controls({ ...running, finish_requested: true }, true, false).pause).toBe(false)
  })
  it('startup and stale sessions retain the emergency stop action', () => {
    expect(controls({ ...base, busy: true }, true, false).stop).toBe(true)
    expect(controls({ ...base, session: { state: 'running' } }, true, false).stop).toBe(true)
    expect(controls({ ...base, session: { state: 'complete' } }, true, false).stop).toBe(false)
  })
  it('game sessions can pause while only training sessions can save updates', () => {
    expect(controls({ ...base, active: true, busy: true, session: { mode: 'play' } }, true, false).pause).toBe(true)
    expect(controls({ ...base, active: true, busy: true, session: { mode: 'play' } }, true, false).finish).toBe(false)
    expect(controls({ ...base, active: true, busy: true, session: { mode: 'monitor' } }, true, false).finish).toBe(false)
  })
})
describe('truthful labels', () => {
  it('distinguishes active, completed, startup and readonly states', () => {
    expect(stateLabel(base)).toBe('待开始')
    expect(stateLabel({ ...base, session: { state: 'complete' } })).toBe('已完成')
    expect(stateLabel({ ...base, busy: true })).toBe('正在启动')
    expect(stateLabel({ ...base, busy: true, active: true, session: { state: 'paused' } })).toBe('已暂停')
    expect(stateLabel({ ...base, preview: true })).toBe('只读预览')
  })
  it('formats real elapsed time and missing timestamps', () => {
    expect(duration(0)).toBe('00:00')
    expect(duration(3600)).toBe('60:00')
    expect(duration(-30)).toBe('00:00')
    expect(date(null)).toBe('—')
    expect(date('not-a-time')).toBe('—')
  })
})
