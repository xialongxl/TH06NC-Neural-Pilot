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
  const update = (changes: Partial<GameSetup>) => onChange(changeSetup(value, changes))
  return <fieldset className="game-setup" disabled={disabled}>
    <legend>游戏设置<span>下次启动生效</span></legend>
    <div className="game-setup-fields">
      <label>游戏模式<select aria-label="游戏模式" value={value.mode} onChange={e => update({ mode: e.target.value as GameSetup['mode'] })}>
        <option value="story">完整流程</option><option value="practice">Practice · 关卡练习</option>
      </select></label>
      <label>难度<select aria-label="难度" value={value.difficulty} onChange={e => update({ difficulty: e.target.value as GameSetup['difficulty'] })}>
        {Object.entries(DIFFICULTIES).map(([key, label]) => <option key={key} value={key} disabled={value.mode === 'practice' && key === 'extra'}>{label}</option>)}
      </select></label>
      <label>自机<select aria-label="自机" value={value.character} onChange={e => update({ character: e.target.value as GameSetup['character'] })}>
        {Object.entries(CHARACTERS).map(([key, label]) => <option key={key} value={key}>{label}</option>)}
      </select></label>
      <label>起始关卡<select aria-label="起始关卡" disabled={value.mode !== 'practice'} value={value.difficulty === 'extra' ? 7 : value.stage} onChange={e => update({ stage: Number(e.target.value) })}>
        {value.difficulty === 'extra' ? <option value={7}>EX</option> : [1,2,3,4,5,6].map(stage => <option key={stage} value={stage} disabled={value.difficulty === 'easy' && stage === 6}>Stage {stage}</option>)}
      </select></label>
    </div>
    <p>{value.mode === 'practice' ? '使用游戏已解锁的关卡；Easy 最多到 Stage 5。结算后重新进入所选关卡。' : value.difficulty === 'extra' ? '从游戏 Extra 入口开始，需要该自机已解锁。' : '从 Stage 1 开始，随实际流程推进关卡。'}</p>
  </fieldset>
}
