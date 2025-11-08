import { useEffect, useRef } from 'react'

export default function PointerDot() {
  const dotRef = useRef(null)
  const stateRef = useRef({ x: window.innerWidth / 2, y: window.innerHeight / 2, targetX: null, targetY: null, rafId: 0 })

  useEffect(() => {
    const dot = dotRef.current
    if (!dot) return

    const handlePointerMove = (event) => {
      stateRef.current.targetX = event.clientX
      stateRef.current.targetY = event.clientY
      dot.style.opacity = '1'
    }

    const handlePointerLeave = () => {
      stateRef.current.targetX = null
      stateRef.current.targetY = null
      dot.style.opacity = '0'
    }

    const animate = () => {
      const { targetX, targetY } = stateRef.current
      if (typeof targetX === 'number' && typeof targetY === 'number') {
        stateRef.current.x = targetX
        stateRef.current.y = targetY
        dot.style.transform = `translate3d(${targetX - 8}px, ${targetY - 8}px, 0)`
      }

      stateRef.current.rafId = requestAnimationFrame(animate)
    }

    stateRef.current.rafId = requestAnimationFrame(animate)

    window.addEventListener('pointermove', handlePointerMove)
    window.addEventListener('pointerleave', handlePointerLeave)

    return () => {
      cancelAnimationFrame(stateRef.current.rafId)
      window.removeEventListener('pointermove', handlePointerMove)
      window.removeEventListener('pointerleave', handlePointerLeave)
    }
  }, [])

  return <div ref={dotRef} className="pointer-dot pointer-dot--blue" />
}
