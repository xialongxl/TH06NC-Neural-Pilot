import { message } from './i18n'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { Snapshot } from './types'
import { applyFrame, pageSchedule, reconcile, streamUrl } from './stream'
import type { StreamFrame, StreamState } from './stream'

export function useConsole() {
  const [state, setState] = useState<Snapshot | null>(null)
  const [connected, setConnected] = useState(false)
  const [pending, setPending] = useState(false)
  const [notice, setNotice] = useState<{ text: string; error: boolean } | null>(null)
  const [serverClosed, setServerClosed] = useState(false)
  const closedRef = useRef(false)
  const token = useRef('')
  const lastMessage = useRef(0)
  const pendingRef = useRef(false)
  const wire = useRef<StreamState>({ revision: 0, state: null })

  useEffect(() => {
    let disposed = false
    let source: EventSource | undefined
    let retry: ReturnType<typeof setTimeout> | undefined
    let generation = 0
    let tokenRequest: AbortController | undefined
    const boot = async () => {
      if (closedRef.current) return
      const current = ++generation
      source?.close()
      clearTimeout(retry)
      tokenRequest?.abort()
      tokenRequest = new AbortController()
      try {
        const response = await fetch('/api/bootstrap?state=false', { cache: 'no-store', signal: tokenRequest.signal })
        if (!response.ok) throw new Error(message("console.backend_not_ready"))
        const data = await response.json()
        if (disposed || current !== generation) return
        token.current = data.token
        const modern = data.protocol === 2
        source = new EventSource(modern ? streamUrl(document.hidden) : '/api/events')
        source.addEventListener('shutdown', () => {
          if (disposed || current !== generation) return
          closedRef.current = true
          setServerClosed(true)
          setConnected(false)
          source?.close()
          clearTimeout(retry)
        })
        lastMessage.current = Date.now()
        source.onmessage = event => {
          if (disposed || current !== generation || closedRef.current) return
          try {
            const previous = wire.current
            const decoded = JSON.parse(event.data)
            const next = modern ? applyFrame(previous, decoded as StreamFrame)
              : { revision: 0, state: reconcile(previous.state, decoded as Snapshot) }
            wire.current = next
            lastMessage.current = Date.now()
            setConnected(true)
            if (!document.hidden && next.state && next.state !== previous.state) setState(next.state)
          } catch {
            setConnected(false)
            source?.close()
            retry = setTimeout(boot, pageSchedule(document.hidden).reconnectMs)
          }
        }
        // Re-establishing the stream also refreshes the small control token;
        // every connection starts with a complete, revisioned snapshot.
        source.onerror = () => {
          if (disposed || current !== generation || closedRef.current) return
          setConnected(false)
          source?.close()
          retry = setTimeout(boot, pageSchedule(document.hidden).reconnectMs)
        }
      } catch {
        if (!disposed && current === generation && !closedRef.current) { setConnected(false); retry = setTimeout(boot, pageSchedule(document.hidden).reconnectMs) }
      }
    }
    const visibility = () => {
      if (!document.hidden && wire.current.state) setState(wire.current.state)
      void boot()
    }
    void boot()
    document.addEventListener('visibilitychange', visibility)
    const timer = setInterval(() => {
      if (!document.hidden && Date.now() - lastMessage.current > pageSchedule(false).staleMs) setConnected(false)
    }, 2000)
    return () => { disposed = true; generation++; tokenRequest?.abort(); source?.close(); clearTimeout(retry); clearInterval(timer); document.removeEventListener('visibilitychange', visibility) }
  }, [])

  const command = useCallback(async (path: string, body?: unknown) => {
    if (pendingRef.current) return false
    pendingRef.current = true
    setPending(true)
    try {
      const response = await fetch(`/api/${path}`, { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'X-Console-Token': token.current },
        body: body === undefined ? undefined : JSON.stringify(body) })
      const result = await response.json()
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : message("console.invalid_request_check_the_parameters_and_try"))
      if (path === 'server/shutdown') setState(previous => previous ? { ...previous, shutdown: result } : previous)
      setNotice({ text: result.message || message("console.model_selected"), error: false })
      // The next delta acknowledges the resulting state without downloading
      // the full catalog and history again after every button click.
      return true
    } catch (error) {
      setNotice({ text: error instanceof Error ? error.message : message("console.backend_connection_failed_press_f8_in_an"), error: true })
      return false
    } finally { pendingRef.current = false; setPending(false) }
  }, [])
  useEffect(() => {
    if (!notice || notice.error) return
    const timer = setTimeout(() => setNotice(null), 4500)
    return () => clearTimeout(timer)
  }, [notice])
  return { state, connected, pending, serverClosed, command, notice, dismiss: () => setNotice(null) }
}
