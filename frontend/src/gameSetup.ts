import type { GameSettings, GameSetup } from './types'

export const DIFFICULTIES = { easy: 'Easy', normal: 'Normal', hard: 'Hard', lunatic: 'Lunatic', extra: 'Extra' }
export const CHARACTERS = { reimu_a: '灵梦 A', reimu_b: '灵梦 B', marisa_a: '魔理沙 A', marisa_b: '魔理沙 B' }
export const DEFAULT_SETUP: GameSetup = { mode: 'story', difficulty: 'normal', character: 'reimu_a', stage: 1 }

export function validSetup(value: unknown): value is GameSetup {
  if (!value || typeof value !== 'object') return false
  const v = value as GameSetup
  return (v.mode === 'story' || v.mode === 'practice')
    && Object.hasOwn(DIFFICULTIES, v.difficulty) && Object.hasOwn(CHARACTERS, v.character)
    && Number.isInteger(v.stage) && v.stage >= 1 && v.stage <= 6
    && (v.mode === 'story' ? v.stage === 1 : v.difficulty !== 'extra' && !(v.difficulty === 'easy' && v.stage === 6))
}

export function changeSetup(previous: GameSetup, changes: Partial<GameSetup>): GameSetup {
  const next = { ...previous, ...changes }
  if (next.mode === 'story') next.stage = 1
  if (next.mode === 'practice' && next.difficulty === 'extra') next.difficulty = 'normal'
  if (next.mode === 'practice' && next.difficulty === 'easy' && next.stage === 6) next.stage = 5
  return next
}

export function gameSettingsLabel(value?: GameSettings): string {
  if (!value) return '等待识别游戏设置'
  const modes = { story: '完整流程', practice: 'Practice', challenge: '挑战模式', spell_practice: '符卡练习' }
  return (value.source === 'menu' ? '菜单选择 · ' : '') + [value.difficulty ? DIFFICULTIES[value.difficulty] : '难度待识别',
    value.character ? CHARACTERS[value.character] : '自机待识别',
    value.mode ? modes[value.mode] : '模式待识别'].join(' · ')
}
