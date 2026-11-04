import { describe, expect, it, vi } from 'vitest'
import { actionDirection, battleViewport, DIRECTIONS, drawBattleFrame, parseBattlePacket } from './battleFrame'
import type { BattleFrame } from './battleFrame'

const frame: BattleFrame = {
  viewport: [384, 448], bounds: [8, 16, 368, 416], player: [192, 390], tick: 42, stage: 2,
  action: 33, input: 'matched', backend: 'rust',
  candidates: Array.from({ length: 18 }, () => ['unknown', null]),
  bullets: [[8, 9], [500, 9]], route: [[192, 390], [196, 386]], bodies: [], lasers: [],
}

describe('battle view snapshots', () => {
  it('separates movement, speed, and bomb from the joint action', () => {
    const selected = actionDirection(33)
    expect(selected).toEqual({ move: 15, direction: 6, focused: true, bomb: true })
    expect(DIRECTIONS[selected.direction]).toEqual([1, -1])
    expect(actionDirection(9)).toEqual({ move: 9, direction: 0, focused: true, bomb: false })
  })

  it('accepts an empty battlefield and rejects stale or unbounded payloads', () => {
    const packet = { version: 1, state: 'live', frame: { ...frame, bullets: [] },
      age_ms: 90, expires_at_ms: 1000 }
    expect(parseBattlePacket(JSON.stringify(packet))).toMatchObject({ state: 'live', frame: { bullets: [] } })
    expect(() => parseBattlePacket(JSON.stringify({ ...packet, age_ms: 351 }))).toThrow()
    expect(() => parseBattlePacket(JSON.stringify({
      ...packet, frame: { ...frame, candidates: [['clear', 0]] },
    }))).toThrow()
    expect(parseBattlePacket('{"version":1,"state":"waiting"}')).toEqual({ state: 'waiting' })
  })

  it('keeps the playfield aspect ratio and clips out-of-range bullets', () => {
    const transform = battleViewport(1024, 600, frame.viewport)
    expect(transform.y).toBeCloseTo(8)
    expect(transform.x).toBeGreaterThan(transform.y)
    const ctx = {
      save: vi.fn(), translate: vi.fn(), scale: vi.fn(), fillRect: vi.fn(), restore: vi.fn(),
      beginPath: vi.fn(), rect: vi.fn(), clip: vi.fn(), setLineDash: vi.fn(), strokeRect: vi.fn(),
      moveTo: vi.fn(), arc: vi.fn(), fill: vi.fn(), rotate: vi.fn(), lineTo: vi.fn(), stroke: vi.fn(),
    }
    drawBattleFrame(ctx as unknown as CanvasRenderingContext2D, frame, 1024, 600)
    expect(ctx.clip).toHaveBeenCalledOnce()
    expect(ctx.arc.mock.calls.some(args => args[0] === 8 && args[1] === 9)).toBe(true)
    expect(ctx.arc.mock.calls.some(args => args[0] === 500 && args[1] === 9)).toBe(false)
  })
})
