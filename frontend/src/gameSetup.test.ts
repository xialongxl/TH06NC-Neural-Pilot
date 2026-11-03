import { describe, expect, it } from 'vitest'
import { changeSetup, DEFAULT_SETUP, gameSettingsLabel, validSetup } from './gameSetup'

describe('game setup and observed identity', () => {
  it('keeps a selectable Practice difficulty, shot and stage in the request', () => {
    const setup = changeSetup(DEFAULT_SETUP, { mode: 'practice', difficulty: 'lunatic', character: 'marisa_b', stage: 5 })
    expect(validSetup(setup)).toBe(true)
    expect(setup).toEqual({ mode: 'practice', difficulty: 'lunatic', character: 'marisa_b', stage: 5 })
  })
  it('keeps game restrictions visible when changing mode or difficulty', () => {
    expect(changeSetup({ ...DEFAULT_SETUP, difficulty: 'extra' }, { mode: 'practice' }).difficulty).toBe('normal')
    expect(changeSetup({ ...DEFAULT_SETUP, mode: 'practice', stage: 6 }, { difficulty: 'easy' }).stage).toBe(5)
    expect(changeSetup({ ...DEFAULT_SETUP, mode: 'practice', stage: 5 }, { mode: 'story' }).stage).toBe(1)
  })
  it('rejects stale or malformed persisted selections', () => {
    for (const value of [null, {}, { ...DEFAULT_SETUP, stage: true }, { ...DEFAULT_SETUP, mode: 'practice', difficulty: 'extra' }])
      expect(validSetup(value)).toBe(false)
  })
  it('never presents a requested default as an actual observation', () => {
    expect(gameSettingsLabel()).toBe('等待识别游戏设置')
    expect(gameSettingsLabel({ mode: null, character: null, difficulty: null, stage: null })).toContain('难度待识别')
    expect(gameSettingsLabel({ mode: 'practice', character: 'marisa_b', difficulty: 'lunatic', stage: 5 })).toBe('Lunatic · 魔理沙 B · Practice')
  })
})
