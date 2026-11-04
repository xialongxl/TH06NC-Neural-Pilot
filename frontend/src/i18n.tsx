import { createContext, useContext, useEffect, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import catalog from '../../th06ai/messages.json'

export type Locale = 'en' | 'zh-CN'
type Key = keyof typeof catalog
const slot = /\{(\d+)\}/g
const storageKey = 'th06-language'

export function translator(locale: Locale) {
  return (key: Key, ...values: unknown[]) => catalog[key][locale === 'en' ? 1 : 0]
    .replace(slot, (placeholder, index: string) => Number(index) < values.length ? String(values[Number(index)]) : placeholder)
}

export const message = translator('zh-CN')

function savedMessages() {
  const exact = new Map<string, string>()
  const templates: { pattern: RegExp; english: string; literalLength: number }[] = []
  for (const [zh, en] of Object.values(catalog)) {
    const parts = zh.split(slot)
    if (parts.length === 1) exact.set(zh, en)
    else {
      const pattern = parts.map((part, index) => index % 2
        ? `(?<p${part}>[\\s\\S]*?)`
        : part.replace(/[.*+?^$\{\}()|[\]\\]/g, '\\$&')).join('')
      const literalLength = parts.reduce((length, part, index) => length + (index % 2 ? 0 : part.length), 0)
      templates.push({ pattern: new RegExp(`^(?:${pattern})$`), english: en, literalLength })
    }
  }
  return { exact, templates: templates.sort((a, b) => b.literalLength - a.literalLength) }
}

let saved: ReturnType<typeof savedMessages> | undefined

export function translateSaved(text: string | null | undefined, locale: Locale, depth = 0): string {
  if (!text) return ''
  if (locale !== 'en' || depth >= 4 || !/\p{Script=Han}/u.test(text)) return text
  saved ??= savedMessages()
  const exact = saved.exact.get(text)
  if (exact !== undefined) return exact
  for (const { pattern, english } of saved.templates) {
    const match = pattern.exec(text)
    if (match?.groups && match[0] === text) {
      const groups = match.groups
      return english.replace(slot, (_, index: string) => translateSaved(groups[`p${index}`], locale, depth + 1))
    }
  }
  return text
}

export function translateLog(text: string | undefined, locale: Locale): string {
  if (!text || locale !== 'en') return text || ''
  return text.split(/(\r?\n)/).map(line => {
    const translated = translateSaved(line, locale)
    if (translated !== line) return translated
    const exception = /^([\w.]+: )(.+)$/.exec(line)
    return exception ? exception[1] + translateSaved(exception[2], locale) : line
  }).join('')
}

function initialLocale(): Locale {
  try {
    const saved = localStorage.getItem(storageKey)
    if (saved === 'en' || saved === 'zh-CN') return saved
  } catch { /* Browser storage is optional. */ }
  return /^zh(?:-|$)/i.test(navigator.language) ? 'zh-CN' : 'en'
}

const LanguageContext = createContext<{ locale: Locale; setLocale: (locale: Locale) => void }>({
  locale: 'zh-CN', setLocale: () => {},
})

export function LanguageProvider({ children }: { children: ReactNode }) {
  const [locale, setLocale] = useState(initialLocale)
  useEffect(() => {
    document.documentElement.lang = locale
    try { localStorage.setItem(storageKey, locale) } catch { /* Browser storage is optional. */ }
  }, [locale])
  const value = useMemo(() => ({ locale, setLocale }), [locale])
  return <LanguageContext.Provider value={value}>{children}</LanguageContext.Provider>
}

export function useI18n() {
  const { locale, setLocale } = useContext(LanguageContext)
  return useMemo(() => ({ locale, setLocale, t: translator(locale),
    display: (value: string | null | undefined) => translateSaved(value, locale),
  }), [locale, setLocale])
}
