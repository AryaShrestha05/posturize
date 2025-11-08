import { useEffect, useRef, useState } from 'react'
import { gsap } from 'gsap'
import PostureLogo from '@/components/ui/posture-logo'
import LoadingScreen from '@/components/ui/loading-screen'

export default function HomePage({ onBegin, onViewSummary, hasSummary }) {
  const titleRef = useRef(null)
  const heroRef = useRef(null)
  const [showLoader, setShowLoader] = useState(true)
  const [isReady, setIsReady] = useState(false)

  useEffect(() => {
    if (!isReady) return
    if (!onBegin) return

    const handleKeydown = (event) => {
      if (event.code === 'Space') {
        event.preventDefault()
        onBegin()
      }
    }

    window.addEventListener('keydown', handleKeydown)
    return () => window.removeEventListener('keydown', handleKeydown)
  }, [isReady, onBegin])

  useEffect(() => {
    if (!isReady) return

    const scope = titleRef.current
    if (!scope) return

    const letters = scope.querySelectorAll('.posturize-char')
    const caret = scope.querySelector('.posturize-caret')

    const ctx = gsap.context(() => {
      gsap.set(letters, { opacity: 0 })
      const tl = gsap.timeline({ delay: 0.15 })
      tl.to(letters, {
        opacity: 1,
        duration: 0.08,
        stagger: 0.045,
        ease: 'none',
      }).to(
        caret,
        {
          opacity: 0,
          repeat: -1,
          yoyo: true,
          ease: 'none',
          duration: 0.4,
        },
        '-=0.15'
      )
    }, scope)

    return () => ctx.revert()
  }, [isReady])

  useEffect(() => {
    if (!isReady) return
    const scope = heroRef.current
    if (!scope) return

    const ctx = gsap.context(() => {
      gsap.from('.hero-stagger', {
        opacity: 0,
        y: 40,
        duration: 0.7,
        ease: 'power3.out',
        stagger: 0.1,
      })
      gsap.from('.hero-button', {
        opacity: 0,
        scale: 0.92,
        duration: 0.55,
        ease: 'back.out(1.8)',
        delay: 0.4,
      })
    }, scope)

    return () => ctx.revert()
  }, [isReady])

  const titleCharacters = Array.from('Posturize')

  const handleLoaderFinish = () => {
    setIsReady(true)
    setShowLoader(false)
  }

  return (
    <div className="relative min-h-screen overflow-hidden">
      {showLoader && <LoadingScreen onFinish={handleLoaderFinish} />}

      <div
        ref={heroRef}
        className={`relative z-10 flex min-h-screen flex-col items-stretch justify-center px-6 py-16 md:flex-row md:px-16 lg:px-24 transition-opacity duration-500 ${
          isReady ? 'opacity-100' : 'opacity-0'
        }`}
      >
        <aside className="hero-stagger flex w-full flex-none flex-col items-center justify-center gap-10 py-12 md:w-2/5 md:items-start">
          <div className="inline-flex items-center gap-6 px-4 py-4">
            <PostureLogo className="h-20 w-20 text-white/85" />
            <div className="space-y-1">
              <p className="text-sm font-semibold uppercase tracking-[0.4em] text-white/60">posturize</p>
              <p className="text-lg font-medium text-white">movement intelligence</p>
            </div>
          </div>
        </aside>

        <section className="flex w-full flex-1 flex-col items-center justify-center gap-10 py-12 text-center md:w-3/5 md:pl-16 lg:w-3/5">
          <div className="w-full max-w-4xl space-y-6 px-4 sm:px-8 md:px-10">
            <div className="space-y-4">
              <h1
                ref={titleRef}
                className="hero-title text-balance text-6xl tracking-tight text-white sm:text-7zl lg:text-8xl"
              >
                {titleCharacters.map((char, index) => (
                  <span key={`${char}-${index}`} className="posturize-char inline-block lowercase will-change-transform">
                    {char.toLowerCase()}
                  </span>
                ))}
                <span className="posturize-caret ml-1 inline-block w-[0.12em] bg-white/80 align-middle" />
              </h1>
              <p className="hero-stagger hero-subtext muted text-xl leading-relaxed">
                blending computer vision so you
                can monitor, learn, and improve your posture.
              </p>
            </div>

            <div className="flex justify-center pt-2">
              <button
                type="button"
                className="hero-button primary-button px-16 text-lg"
                onClick={onBegin}
              >
                <span>press space to start</span>
              </button>
            </div>

            {hasSummary && (
              <div className="flex justify-center pt-4">
                <button type="button" className="primary-button primary-button--compact px-12" onClick={onViewSummary}>
                  <span>view last session summary</span>
                </button>
              </div>
            )}
          </div>
        </section>
      </div>
    </div>
  )
}