import { translator } from './i18n'
import type { Locale } from './i18n'
import type { GameSettings, GameSetup } from './types'

export const DIFFICULTIES = { easy: 'Easy', normal: 'Normal', hard: 'Hard', lunatic: 'Lunatic', extra: 'Extra' }
export const CHARACTERS = { reimu_a: 'console.reimu_a', reimu_b: 'console.reimu_b', marisa_a: 'console.marisa_a', marisa_b: 'console.marisa_b' } as const
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

export function gameSettingsLabel(value?: GameSettings | null, locale: Locale = 'zh-CN'): string {
  const message = translator(locale)
  if (!value) return message("console.waiting_for_game_settings")
  const modes = { story: message("console.story"), practice: 'Practice', challenge: message("console.challenge_mode"), spell_practice: message("console.spell_practice") }
  return (value.source === 'menu' ? message("console.menu_selection") : '') + [value.difficulty ? DIFFICULTIES[value.difficulty] : message("console.difficulty_unknown"),
    value.character ? message(CHARACTERS[value.character]) : message("console.character_unknown"),
    value.mode ? modes[value.mode] : message("console.mode_unknown")].join(' · ')
}
