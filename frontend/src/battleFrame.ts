export const DIRECTIONS = [[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0],
  [-1, -1], [1, -1], [-1, 1], [1, 1]] as const

export const RISK_COLORS = {
  clear: '#79e2ac', risk: '#ff757f', fallback: '#f3bd69', unknown: '#8a9bb1',
} as const

type Risk = keyof typeof RISK_COLORS
type Point = [number, number]
type Rect = [number, number, number, number]
export type BattleFrame = {
  viewport: Point; bounds: Rect; player: Point; tick: number; stage: number | null
  action: number; input: string; backend: string | null; candidates: [Risk, number | null][]
  bullets: Point[]; route: Point[]
  bodies: [number, number, number, number, number][]
  lasers: [number, number, number, number, number, number, number][]
}
export type BattlePacket =
  | { state: 'off' | 'waiting' | 'disconnected' }
  | { state: 'live'; frame: BattleFrame; age_ms: number; expires_at_ms: number }

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
const tuple = (value: unknown, length: number) =>
  Array.isArray(value) && value.length === length && value.every(finite)
const rows = (value: unknown, limit: number, width: number) =>
  Array.isArray(value) && value.length <= limit && value.every(row => tuple(row, width))

export function parseBattlePacket(text: string): BattlePacket {
  const packet = JSON.parse(text)
  if (packet?.version !== 1) throw new Error('Unsupported battle view packet')
  if (packet.state === 'off' || packet.state === 'waiting') return { state: packet.state }
  const f = packet.frame
  if (packet.state !== 'live' || !finite(packet.age_ms) || !finite(packet.expires_at_ms)
      || packet.age_ms < 0 || packet.age_ms > 350 || !f
      || !tuple(f.viewport, 2) || f.viewport.some((size: number) => size <= 0 || size > 3000)
      || !tuple(f.bounds, 4) || !tuple(f.player, 2) || !Number.isInteger(f.tick)
      || !(f.stage === null || Number.isInteger(f.stage))
      || !Number.isInteger(f.action) || f.action < 0 || f.action >= 36
      || typeof f.input !== 'string' || !(f.backend === null || typeof f.backend === 'string')
      || !rows(f.bullets, 640, 2) || !rows(f.route, 32, 2)
      || !rows(f.bodies, 256, 5) || !rows(f.lasers, 64, 7)
      || !Array.isArray(f.candidates) || f.candidates.length !== 18
      || !f.candidates.every((row: unknown) => Array.isArray(row) && row.length === 2
        && Object.hasOwn(RISK_COLORS, row[0]) && (row[1] === null || finite(row[1])))) {
    throw new Error('Invalid battle view packet')
  }
  return packet
}

export function actionDirection(action: number) {
  return { move: action % 18, direction: action % 9, focused: action % 18 >= 9, bomb: action >= 18 }
}

export function battleViewport(width: number, height: number, view: Point) {
  const scale = Math.max(0, Math.min((width - 16) / view[0], (height - 16) / view[1]))
  return { scale, x: (width - scale * view[0]) / 2, y: (height - scale * view[1]) / 2 }
}

export function drawBattleFrame(ctx: CanvasRenderingContext2D, f: BattleFrame, width: number, height: number) {
  const transform = battleViewport(width, height, f.viewport)
  if (!transform.scale) return
  const [w, h] = f.viewport
  const [playerX, playerY] = f.player
  const unit = 1 / transform.scale
  const arrowSize = Math.min(w, h) * .122
  ctx.save()
  ctx.translate(transform.x, transform.y)
  ctx.scale(transform.scale, transform.scale)
  ctx.fillStyle = '#111e2c'
  ctx.fillRect(0, 0, w, h)
  ctx.save()
  ctx.beginPath()
  ctx.rect(0, 0, w, h)
  ctx.clip()
  ctx.strokeStyle = '#526b83'
  ctx.lineWidth = unit
  ctx.setLineDash([5 * unit, 4 * unit])
  ctx.strokeRect(...f.bounds)
  ctx.setLineDash([])

  ctx.fillStyle = '#9acbedbd'
  ctx.beginPath()
  const radius = Math.max(1.1 * unit, Math.min(w, h) * .0034)
  for (const [x, y] of f.bullets) {
    if (x < 0 || y < 0 || x > w || y > h) continue
    ctx.moveTo(x + radius, y)
    ctx.arc(x, y, radius, 0, Math.PI * 2)
  }
  ctx.fill()
  ctx.strokeStyle = '#dca78bbf'
  for (const [x, y, halfW, halfH, known] of f.bodies) {
    if (known && halfW >= 0 && halfH >= 0) ctx.strokeRect(x - halfW, y - halfH, halfW * 2, halfH * 2)
  }
  for (const [x, y, start, end, angle, halfW, state] of f.lasers) {
    if (end < start || halfW <= 0) continue
    ctx.save()
    ctx.translate(x, y)
    ctx.rotate(angle)
    ctx.strokeStyle = state === 0 ? '#f8c8839e' : '#fa959dcf'
    ctx.fillStyle = '#fa959d24'
    if (state === 0) ctx.setLineDash([4 * unit, 3 * unit])
    else ctx.fillRect(start, -halfW, end - start, halfW * 2)
    ctx.strokeRect(start, -halfW, end - start, halfW * 2)
    ctx.restore()
  }
  if (f.route.length > 1) {
    ctx.beginPath()
    ctx.moveTo(...f.route[0])
    for (const point of f.route.slice(1)) ctx.lineTo(...point)
    ctx.setLineDash([4 * unit, 4 * unit])
    ctx.strokeStyle = '#d6ebf3a1'
    ctx.lineWidth = unit * 1.4
    ctx.stroke()
    ctx.setLineDash([])
  }

  const { move } = actionDirection(f.action)
  const arrow = (index: number) => {
    const [dx, dy] = DIRECTIONS[index % 9]
    const [risk, score] = f.candidates[index]
    const selected = index === move
    ctx.strokeStyle = RISK_COLORS[risk]
    ctx.globalAlpha = selected ? 1 : .52 + .27 * ((score ?? 0) + 1) / 2
    ctx.lineWidth = (selected ? 2.4 : 1.65) * unit
    ctx.beginPath()
    if (!dx && !dy) {
      ctx.arc(playerX, playerY, arrowSize * (index >= 9 ? .12 : .21), 0, Math.PI * 2)
    } else {
      const angle = Math.atan2(dy, dx)
      const ux = Math.cos(angle), uy = Math.sin(angle)
      const end = arrowSize * (index >= 9 ? .63 : 1)
      const start = arrowSize * (index >= 9 ? .27 : .74)
      const tip = arrowSize * .105
      const x = playerX + ux * end, y = playerY + uy * end
      ctx.moveTo(playerX + ux * start, playerY + uy * start)
      ctx.lineTo(x, y)
      ctx.moveTo(x - ux * tip - uy * tip * .58, y - uy * tip + ux * tip * .58)
      ctx.lineTo(x, y)
      ctx.lineTo(x - ux * tip + uy * tip * .58, y - uy * tip - ux * tip * .58)
    }
    if (selected) {
      const color = ctx.strokeStyle
      ctx.strokeStyle = '#f6f9ff'
      ctx.lineWidth = unit * 5
      ctx.stroke()
      ctx.strokeStyle = color
      ctx.lineWidth = unit * 2.4
    }
    ctx.stroke()
  }
  for (let index = 0; index < 18; index++) if (index !== move) arrow(index)
  arrow(move)
  ctx.globalAlpha = 1
  ctx.fillStyle = '#f7fbff'
  ctx.beginPath()
  ctx.arc(playerX, playerY, Math.max(2.3 * unit, arrowSize * .048), 0, Math.PI * 2)
  ctx.fill()
  ctx.restore()
  ctx.strokeStyle = '#a3c2d9'
  ctx.lineWidth = unit * 1.5
  ctx.strokeRect(0, 0, w, h)
  ctx.restore()
}
