import { useCallback, useMemo, useState } from 'react'
import HomePage from './components/HomePage'
import SessionSummary from './pages/SessionSummary'
import WebcamViewer from './pages/WebcamViewer'
import SubtleDots from './components/ui/subtle-dots'
import PointerDot from './components/ui/pointer-dot'
import SmokeyCursor from './components/ui/smokey-cursor'

export default function App() {
  const [activeView, setActiveView] = useState('home')
  const [lastSummary, setLastSummary] = useState(null)
  const [lastSessionId, setLastSessionId] = useState(null)

  const handleNavigate = useCallback((view) => {
    setActiveView(view)
  }, [])

  const handleSessionComplete = useCallback((summary) => {
    setLastSummary(summary)
    if (summary?.session_id) {
      setLastSessionId(summary.session_id)
    }
    setActiveView('summary')
  }, [])

  const views = useMemo(
    () => ({
      home: (
        <HomePage
          onBegin={() => handleNavigate('live')}
          onViewSummary={() => handleNavigate('summary')}
          hasSummary={Boolean(lastSummary)}
        />
      ),
      live: (
        <WebcamViewer
          onSessionComplete={handleSessionComplete}
          onNavigate={handleNavigate}
          sessionIdOverride={lastSessionId}
        />
      ),
      summary: (
        <SessionSummary
          summary={lastSummary}
          sessionId={lastSessionId}
          onNavigate={handleNavigate}
        />
      ),
    }),
    [handleNavigate, handleSessionComplete, lastSummary, lastSessionId]
  )

  const content = views[activeView] ?? views.home

  return (
    <div className="relative min-h-screen overflow-hidden bg-aurora">
      <div className="pointer-events-none absolute inset-0">
        <div className="grid-overlay absolute inset-0" aria-hidden="true" />
      </div>
      <SubtleDots />
      <PointerDot />
      <SmokeyCursor
        simulationResolution={96}
        dyeResolution={960}
        densityDissipation={4.5}
        velocityDissipation={2.5}
        curl={2}
        splatRadius={0.12}
        splatForce={3200}
        colorUpdateSpeed={6}
      />
      <div className="relative z-10">{content}</div>
    </div>
  )
}
