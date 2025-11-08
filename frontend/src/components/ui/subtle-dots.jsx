import { useEffect, useRef } from 'react'

const DOT_COUNT = 240
const DOT_MIN_RADIUS = 2
const DOT_MAX_RADIUS = 4
const BASE_ALPHA = 0.32
const PARALLAX_FACTOR = 0.035
const BASE_VELOCITY = 0.00035

function rand(min, max) {
  return Math.random() * (max - min) + min
}

function createDot() {
  return {
    x: rand(0, 1),
    y: rand(0, 1),
    radius: rand(DOT_MIN_RADIUS, DOT_MAX_RADIUS),
    driftX: rand(-0.08, 0.08),
    driftY: rand(-0.08, 0.08),
    vx: rand(-0.015, 0.015),
    vy: rand(-0.015, 0.015),
    alpha: rand(BASE_ALPHA, BASE_ALPHA + 0.18),
    twinkleSpeed: rand(0.4, 1.1),
    twinkleOffset: rand(0, Math.PI * 2),
    depth: rand(0.4, 1.2),
  }
}

export default function SubtleDots() {
  const canvasRef = useRef(null)
  const animationRef = useRef(0)
  const pointerRef = useRef({ x: 0.5, y: 0.5, targetX: 0.5, targetY: 0.5 })

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d', { alpha: true })
    if (!ctx) return

    const dots = Array.from({ length: DOT_COUNT }, () => createDot())

    const resize = () => {
      const { innerWidth, innerHeight, devicePixelRatio = 1 } = window
      canvas.width = innerWidth * devicePixelRatio
      canvas.height = innerHeight * devicePixelRatio
      canvas.style.width = `${innerWidth}px`
      canvas.style.height = `${innerHeight}px`
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0)
    }

    const handlePointerMove = (event) => {
      pointerRef.current.targetX = event.clientX / window.innerWidth
      pointerRef.current.targetY = event.clientY / window.innerHeight
    }

    resize()
    window.addEventListener('resize', resize)
    window.addEventListener('pointermove', handlePointerMove)

    let lastTime = performance.now()

    const render = (time) => {
      const delta = Math.min((time - lastTime) / 16.67, 1.6)
      lastTime = time

      const pointer = pointerRef.current
      pointer.x += (pointer.targetX - pointer.x) * 0.08
      pointer.y += (pointer.targetY - pointer.y) * 0.08

      ctx.clearRect(0, 0, canvas.width, canvas.height)
      ctx.fillStyle = 'rgba(255, 255, 255, 0.03)'
      ctx.fillRect(0, 0, canvas.width, canvas.height)

      dots.forEach((dot) => {
        dot.x += (dot.driftX + dot.vx * dot.depth) * delta * BASE_VELOCITY
        dot.y += (dot.driftY + dot.vy * dot.depth) * delta * BASE_VELOCITY

        dot.vx += rand(-0.0005, 0.0005) * delta
        dot.vy += rand(-0.0005, 0.0005) * delta

        if (dot.x < -0.1) dot.x = 1.1
        if (dot.x > 1.1) dot.x = -0.1
        if (dot.y < -0.1) dot.y = 1.1
        if (dot.y > 1.1) dot.y = -0.1

        const parallaxX = (pointer.x - 0.5) * PARALLAX_FACTOR * dot.depth
        const parallaxY = (pointer.y - 0.5) * PARALLAX_FACTOR * dot.depth

        const cx = (dot.x + parallaxX) * canvas.width
        const cy = (dot.y + parallaxY) * canvas.height

        const twinkle = 0.7 + Math.sin(time * 0.001 * dot.twinkleSpeed + dot.twinkleOffset) * 0.3
        const alpha = Math.min(0.9, dot.alpha * twinkle + 0.1)

        const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, dot.radius * 5)
        gradient.addColorStop(0, `rgba(255, 255, 255, ${alpha})`)
        gradient.addColorStop(0.6, `rgba(255, 255, 255, ${alpha * 0.25})`)
        gradient.addColorStop(1, 'rgba(255, 255, 255, 0)')

        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(cx, cy, dot.radius * 5, 0, Math.PI * 2)
        ctx.fill()
      })

      animationRef.current = requestAnimationFrame(render)
    }

    animationRef.current = requestAnimationFrame(render)

    return () => {
      cancelAnimationFrame(animationRef.current)
      window.removeEventListener('resize', resize)
      window.removeEventListener('pointermove', handlePointerMove)
    }
  }, [])

  return <canvas ref={canvasRef} className="subtle-dots-canvas pointer-events-none absolute inset-0" />
}
