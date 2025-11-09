import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { gsap } from 'gsap'
import SubtleDots from '@/components/ui/subtle-dots'
import SmokeyCursor from '@/components/ui/smokey-cursor'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000'

export default function LiveWebcamShell() {
  const rootRef = useRef(null)
  const [streamError, setStreamError] = useState('')
  const [summary, setSummary] = useState(null)
  const [sessionState, setSessionState] = useState('idle') // idle | starting | running | stopping
  const chartRef = useRef(null)
  const pollRef = useRef(null)
  const videoRef = useRef(null)

  useEffect(() => {
    const el = rootRef.current
    if (!el) return
    gsap.fromTo(
      el,
      { opacity: 0 },
      { opacity: 1, duration: 0.45, ease: 'power1.out' }
    )
  }, [])

  const startSession = async () => {
    setSessionState('starting')
    setStreamError('')
    try {
      const response = await fetch(`${API_BASE_URL}/api/session/start`, { method: 'POST' })
      if (!response.ok) throw new Error(await response.text())
      await response.json()
      setSessionState('running')
      if (videoRef.current) {
        videoRef.current.src = `${API_BASE_URL}/api/video_feed?ts=${Date.now()}`
      }
    } catch (error) {
      console.error('start session failed', error)
      setStreamError('unable to start session. check the backend.')
      setSessionState('idle')
    }
  }

  const stopSession = async () => {
    setSessionState('stopping')
    try {
      const response = await fetch(`${API_BASE_URL}/api/session/stop`, { method: 'POST' })
      if (!response.ok) throw new Error(await response.text())
      const payload = await response.json()
      if (payload?.snapshot) {
        setSummary((prev) => ({ ...(prev ?? {}), ...payload.snapshot }))
      }
    } catch (error) {
      console.error('stop session failed', error)
      setStreamError('unable to stop session. check the backend.')
    } finally {
      setSessionState('idle')
      if (videoRef.current) {
        videoRef.current.src = ''
      }
    }
  }

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/session/posture`)
        if (!response.ok) {
          throw new Error(await response.text())
        }
        const payload = await response.json()
        setSummary(payload)
      } catch (error) {
        console.error('posture summary error', error)
      }
    }

    fetchSummary()
    pollRef.current = setInterval(fetchSummary, 3000)
    return () => {
      if (pollRef.current) clearInterval(pollRef.current)
    }
  }, [])

  useEffect(() => {
    const intervals = summary?.intervals ?? []
    const svgElement = chartRef.current
    if (!svgElement) return

    const width = svgElement.clientWidth || 640
    const height = svgElement.clientHeight || 260

    const svg = d3.select(svgElement)
    svg.selectAll('*').remove()
    svg.attr('viewBox', `0 0 ${width} ${height}`)

    if (!intervals.length) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#94a3b8')
        .text('collecting posture samples…')
      return
    }

    const margin = { top: 24, right: 24, bottom: 40, left: 56 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const xScale = d3
      .scaleLinear()
      .domain([0, d3.max(intervals, (d) => d.end) ?? 30])
      .range([0, innerWidth])

    const yExtent = d3.extent(intervals, (d) => d.average_delta)
    const padding = 2
    const yScale = d3
      .scaleLinear()
      .domain([Math.min(-10, (yExtent[0] ?? 0) - padding), Math.max(10, (yExtent[1] ?? 0) + padding)])
      .range([innerHeight, 0])
      .nice()

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const xAxis = d3.axisBottom(xScale).ticks(Math.max(3, intervals.length)).tickFormat((d) => `${Math.round(d)}s`)
    const yAxis = d3.axisLeft(yScale).ticks(6).tickFormat((d) => `${d}°`)

    g.append('g').attr('class', 'chart__axis').attr('transform', `translate(0,${innerHeight})`).call(xAxis)
    g.append('g').attr('class', 'chart__axis').call(yAxis)

    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', '#4b5563')
      .attr('stroke-dasharray', '4,4')
      .attr('stroke-width', 1)
      .attr('opacity', 0.6)

    const line = d3
      .line()
      .x((d) => xScale(d.end))
      .y((d) => yScale(d.average_delta))
      .curve(d3.curveMonotoneX)

    g.append('path')
      .datum(intervals)
      .attr('fill', 'none')
      .attr('stroke', '#f8fafc')
      .attr('stroke-width', 2)
      .attr('d', line)

    g.selectAll('.posture-dot')
      .data(intervals)
      .enter()
      .append('circle')
      .attr('class', 'posture-dot')
      .attr('cx', (d) => xScale(d.end))
      .attr('cy', (d) => yScale(d.average_delta))
      .attr('r', 5)
      .attr('fill', (d) => {
        if (d.classification === 'bad') return '#ef4444'
        if (d.classification === 'moderate') return '#f59e0b'
        return '#22c55e'
      })
      .attr('stroke', '#0f172a')
      .attr('stroke-width', 1.5)

    g.append('text')
      .attr('x', innerWidth)
      .attr('y', innerHeight + 36)
      .attr('text-anchor', 'end')
      .attr('fill', '#f8fafc')
      .text('time (s)')

    g.append('text')
      .attr('x', -margin.left + 12)
      .attr('y', -12)
      .attr('text-anchor', 'start')
      .attr('fill', '#f8fafc')
      .text('average nose delta (°)')
  }, [summary?.intervals])

  return (
    <div ref={rootRef} className="relative min-h-screen overflow-hidden bg-aurora">
      <SubtleDots />
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

      <div className="pointer-events-none absolute inset-0">
        <div className="grid-overlay absolute inset-0" aria-hidden="true" />
      </div>
      <div className="relative z-10 flex min-h-screen flex-col items-stretch justify-center gap-8 px-6 py-10 md:px-12 lg:px-16">
        <div className="live-layout-top">
          <div className="live-layout-top__video">
            <section className="webcam-shell">
              <div className="webcam-shell__placeholder">
                {streamError ? (
                  <div className="webcam-shell__error">
                    <h1>stream unavailable</h1>
                    <p>{streamError}</p>
                  </div>
                ) : (
                  <div className="webcam-shell__stream">
                    <img
                      ref={videoRef}
                      className="webcam-shell__video"
                      onError={() => {
                        setStreamError('unable to load the webcam stream. is the flask backend running?')
                        setSessionState('idle')
                      }}
                    />
                  </div>
                )}
              </div>
            </section>
          </div>
          <div className="live-layout-top__aside">
            <div className="live-status-panel">
              {summary && !streamError ? (
                <div className={`webcam-shell__status webcam-shell__status--${summary.classification}`}>
                  <p>{summary.status_message}</p>
                  {typeof summary.baseline_angle === 'number' && (
                    <p>
                      baseline: <span>{summary.baseline_angle.toFixed(1)}°</span>
                    </p>
                  )}
                  {typeof summary.current_delta === 'number' && (
                    <p>
                      delta: <span>{summary.current_delta.toFixed(1)}°</span>
                    </p>
                  )}
                </div>
              ) : (
                <div className="webcam-shell__status webcam-shell__status--calibrating">
                  <p>Session idle. Start to begin calibration.</p>
                </div>
              )}
            </div>
            <button
              type="button"
              className="primary-button primary-button--compact live-layout__button"
              onClick={sessionState === 'running' ? stopSession : startSession}
              disabled={sessionState === 'starting' || sessionState === 'stopping'}
            >
              <span>
                {sessionState === 'starting'
                  ? 'starting…'
                  : sessionState === 'stopping'
                  ? 'stopping…'
                  : sessionState === 'running'
                  ? 'end session'
                  : 'start session'}
              </span>
            </button>
          </div>
        </div>
        <div className="webcam-analytics">
          <div className="webcam-analytics__header">
            <h2>Your Posture Trends, Live:</h2>
          </div>
          <svg ref={chartRef} className="webcam-chart" role="img" aria-label="posture delta line chart" />
        </div>
      </div>
    </div>
  )
}

