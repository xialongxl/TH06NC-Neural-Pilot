import { useI18n } from './i18n'
import { useEffect, useState } from 'react'
import type { GameSetup, Snapshot } from './types'
import { CHARACTERS, DIFFICULTIES, DEFAULT_SETUP, changeSetup, validSetup } from './gameSetup'

export function useGameSetup(state: Snapshot) {
  const [setup, setSetup] = useState<GameSetup>(() => {
    try {
      const saved: unknown = JSON.parse(localStorage.getItem('th06-game-setup') || 'null')
      if (validSetup(saved)) return saved
    } catch { /* Unavailable browser storage does not block setup. */ }
    return state.default_game_setup || DEFAULT_SETUP
  })
  useEffect(() => { try { localStorage.setItem('th06-game-setup', JSON.stringify(setup)) } catch { /* Optional. */ } }, [setup])
  return [setup, setSetup] as const
}

export function GameSetupFields({ value, disabled, onChange }: {
  value: GameSetup; disabled: boolean; onChange: (next: GameSetup) => void;
}) {
  const { t } = useI18n()
  const update = (changes: Partial<GameSetup>) => onChange(changeSetup(value, changes))
  return <fieldset className="game-setup" disabled={disabled}>
    <legend>{t("console.game_setup")}<span>{t("console.applies_next_session")}</span></legend>
    <div className="game-setup-fields">
      <label>{t("console.game_mode")}<select aria-label={t("console.game_mode")} value={value.mode} onChange={e => update({ mode: e.target.value as GameSetup['mode'] })}>
        <option value="story">{t("console.story")}</option><option value="practice">{t("console.stage_practice")}</option>
      </select></label>
      <label>{t("console.difficulty")}<select aria-label={t("console.difficulty")} value={value.difficulty} onChange={e => update({ difficulty: e.target.value as GameSetup['difficulty'] })}>
        {Object.entries(DIFFICULTIES).map(([key, label]) => <option key={key} value={key} disabled={value.mode === 'practice' && key === 'extra'}>{label}</option>)}
      </select></label>
      <label>{t("console.character")}<select aria-label={t("console.character")} value={value.character} onChange={e => update({ character: e.target.value as GameSetup['character'] })}>
        {Object.entries(CHARACTERS).map(([key, label]) => <option key={key} value={key}>{t(label)}</option>)}
      </select></label>
      <label>{t("console.starting_stage")}<select aria-label={t("console.starting_stage")} disabled={value.mode !== 'practice'} value={value.difficulty === 'extra' ? 7 : value.stage} onChange={e => update({ stage: Number(e.target.value) })}>
        {value.difficulty === 'extra' ? <option value={7}>EX</option> : [1,2,3,4,5,6].map(stage => <option key={stage} value={stage} disabled={value.difficulty === 'easy' && stage === 6}>Stage {stage}</option>)}
      </select></label>
    </div>
    <p>{value.mode === 'practice' ? t("console.choose_an_unlocked_stage_easy_is_limited") : value.difficulty === 'extra' ? t("console.starts_from_the_extra_menu_extra_must") : t("console.starts_at_stage_1_and_advances_through")}</p>
  </fieldset>
}
