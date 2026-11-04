import { afterEach, describe, expect, it, vi } from 'vitest'
import { renderToStaticMarkup } from 'react-dom/server'
import App from './App'
import { LanguageProvider, message, translateLog, translateSaved } from './i18n'

afterEach(() => vi.unstubAllGlobals())

describe('console localization', () => {
  it.each([
    ['en', 'Connecting to the training backend', 'Interface language'],
    ['zh-CN', '连接训练后端', '界面语言'],
  ])('renders the console in the saved %s locale', (locale, title, label) => {
    vi.stubGlobal('location', { hash: '#train' })
    vi.stubGlobal('localStorage', { getItem: (key: string) => key === 'th06-language' ? locale : null })
    const html = renderToStaticMarkup(<LanguageProvider><App /></LanguageProvider>)
    expect(html).toContain(title)
    expect(html).toContain(`aria-label="${label}"`)
    expect(html).toContain(`value="${locale}" selected=""`)
  })

  it('translates archived watchdog messages and preserves unknown log content', () => {
    const reason = message('backend.ai_error_0', message('backend.gpu_core_temperature_reached_the_stop_threshold', '71', '70'))
    expect(translateSaved(reason, 'en')).toBe('AI error: GPU core temperature reached the stop threshold: 71°C ≥ 70°C')
    expect(translateSaved(reason, 'zh-CN')).toBe(reason)
    expect(translateLog(`RuntimeError: ${reason}\r\nC:\\模型\\latest.zip`, 'en'))
      .toBe('RuntimeError: AI error: GPU core temperature reached the stop threshold: 71°C ≥ 70°C\r\nC:\\模型\\latest.zip')
  })
})
