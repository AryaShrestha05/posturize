import { useEffect, useRef } from 'react'
import { gsap } from 'gsap'
import PostureLogo from '@/components/ui/posture-logo'

export default function LoadingScreen({ onFinish }) {
  const containerRef = useRef(null)
  const finishedRef = useRef(false)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const ctx = gsap.context(() => {
      const timeline = gsap.timeline({ defaults: { ease: 'power2.inOut' } })

      timeline
        .fromTo(
          '.loading-backdrop',
          { opacity: 0 },
          { opacity: 1, duration: 0.4 }
        )
        .fromTo(
          '.loading-logo',
          { opacity: 0, scale: 0.8 },
          { opacity: 1, scale: 1, duration: 0.6, ease: 'power3.out' }
        )
        .fromTo(
          '.loading-ring',
          { strokeDashoffset: 440, opacity: 0 },
          { strokeDashoffset: 0, opacity: 1, duration: 1.2, ease: 'power1.out' },
          '-=0.4'
        )
        .to(
          container,
          {
            opacity: 0,
            duration: 0.6,
            ease: 'power2.in',
            onComplete: () => {
              if (finishedRef.current) return
              finishedRef.current = true
              if (onFinish) onFinish()
            },
          },
          '+=0.2'
        )
    }, container)

    return () => ctx.revert()
  }, [onFinish])

  return (
    <div ref={containerRef} className="loading-screen">
      <div className="loading-backdrop" />
      <div className="loading-content">
        <div className="loading-circle-shell">
          <svg className="loading-ring" viewBox="0 0 160 160" fill="none">
            <circle cx="80" cy="80" r="70" stroke="rgba(255, 255, 255, 0.4)" strokeWidth="5" strokeLinecap="round" strokeDasharray="440" />
          </svg>
          <PostureLogo className="loading-logo text-white" />
        </div>
        <svg className="loading-loop" viewBox="0 0 160 160">
          <circle cx="80" cy="80" r="64" stroke="rgba(255, 255, 255, 0.18)" strokeWidth="2" strokeLinecap="round" strokeDasharray="80" />
        </svg>
      </div>
    </div>
  )
}

