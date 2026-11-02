import { describe, expect, it } from 'vitest'
import { applyFrame, pageSchedule, reconcile, streamUrl } from './stream'
import type { StreamFrame, StreamState } from './stream'
import type { Snapshot } from './types'

function initial(): StreamState {
  return { revision: 1, state: {
    preview: true, active: true, busy: true, session: { state: 'running', worker: { steps: 42, error: 'old' }, gpu: { temperature_c: 49 } },
    models: [{ id: 'saved', name: '保存', usable: true }], selected_model_id: 'saved',
    readiness: { ready: true }, run_dir: 'run-a', gpu_history: [{ stamp: 1, temperature_c: 49 }],
  } as Snapshot }
}
const delta = (set: [string[], unknown][] = []): StreamFrame => ({ protocol: 2, kind: 'delta', base: 1, revision: 2, set, remove: [] })

describe('incremental display state', () => {
  it('changes only the relevant branch and preserves cold objects', () => {
    const before = initial()
    const after = applyFrame(before, delta([[['session', 'worker', 'steps'], 43]]))
    expect(after.state!.session.worker!.steps).toBe(43)
    expect(before.state!.session.worker!.steps).toBe(42)
    expect(after.state!.models).toBe(before.state!.models)
    expect(after.state!.readiness).toBe(before.state!.readiness)
    expect(after.state!.gpu_history).toBe(before.state!.gpu_history)
    expect(after.state!.session.gpu).toBe(before.state!.session.gpu)
  })
  it('does not schedule a state update for a heartbeat or equal values', () => {
    const before = initial()
    expect(applyFrame(before, { protocol: 2, kind: 'heartbeat', revision: 1 })).toBe(before)
    expect(applyFrame(before, delta([[['session', 'worker', 'steps'], 42]])).state).toBe(before.state)
  })
  it('deletes missing keys and preserves actual null separately', () => {
    const frame = delta([[['session', 'reason'], null]])
    if (frame.kind !== 'delta') throw new Error()
    frame.remove = [['session', 'worker', 'error']]
    const next = applyFrame(initial(), frame)
    expect(next.state!.session.reason).toBeNull()
    expect(next.state!.session.worker).not.toHaveProperty('error')
  })
  it('appends new samples once and enforces the 180 sample limit', () => {
    const before = initial()
    const frame = { ...delta(), history: { mode: 'append', after: 1, points: [{ stamp: 2, temperature_c: 50 }] } } as StreamFrame
    const next = applyFrame(before, frame)
    expect(next.state!.gpu_history.map(p => p.stamp)).toEqual([1, 2])
    expect(() => applyFrame(next, frame)).toThrow('状态版本不连续')
    const full = { ...delta(), history: { mode: 'reset', points: Array.from({ length: 200 }, (_, i) => ({ stamp: i, temperature_c: 49 })) } } as StreamFrame
    expect(applyFrame(before, full).state!.gpu_history).toHaveLength(180)
    expect(before.state!.gpu_history).toHaveLength(1)
  })
  it('replaces history when a new run starts', () => {
    const frame = { ...delta([[['run_dir'], 'run-b']]), history: { mode: 'reset', points: [] } } as StreamFrame
    const next = applyFrame(initial(), frame)
    expect(next.state!.run_dir).toBe('run-b')
    expect(next.state!.gpu_history).toEqual([])
  })
  it('rejects an incorrect history cursor without partial mutation', () => {
    const before = initial()
    const frame = { ...delta([[['session', 'worker', 'steps'], 999]]), history: { mode: 'append', after: 7, points: [] } } as StreamFrame
    expect(() => applyFrame(before, frame)).toThrow('曲线游标')
    expect(before.state!.session.worker!.steps).toBe(42)
  })
  it('resynchronizes after restart and retains equal catalog references', () => {
    const before = initial()
    before.revision = 500
    const incoming = JSON.parse(JSON.stringify(before.state)) as Snapshot
    const next = applyFrame(before, { protocol: 2, kind: 'snapshot', revision: 1, state: incoming })
    expect(next.revision).toBe(1)
    expect(next.state).toBe(before.state)
    expect(next.state!.models).toBe(before.state!.models)
  })
  it('rejects stale revisions and unsafe paths', () => {
    const before = initial()
    expect(() => applyFrame(before, { protocol: 2, kind: 'heartbeat', revision: 7 })).toThrow()
    expect(() => applyFrame(before, { ...delta(), base: 3 } as StreamFrame)).toThrow()
    expect(() => applyFrame(before, delta([[['__proto__', 'polluted'], true]]))).toThrow('无效状态路径')
    expect({}).not.toHaveProperty('polluted')
  })
  it('handles arrays, omitted keys and null while retaining equal data', () => {
    const old = { models: [{ name: 'a' }], error: 'old', value: null }
    const next = reconcile(old, { models: [{ name: 'a' }], value: null } as typeof old)
    expect(next.models).toBe(old.models)
    expect(next).not.toHaveProperty('error')
    expect(reconcile(null, { okay: true } as unknown as null)).toEqual({ okay: true })
  })
  it('uses the slow subscription for hidden pages and resumes on the same API', () => {
    expect(streamUrl(false)).toBe('/api/events?protocol=2&hidden=false')
    expect(streamUrl(true)).toBe('/api/events?protocol=2&hidden=true')
    expect(pageSchedule(true).reconnectMs).toBeGreaterThan(pageSchedule(false).reconnectMs)
  })
})
