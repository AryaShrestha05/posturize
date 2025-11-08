import { useCallback, useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'
import { io } from 'socket.io-client'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000'
const CALIBRATION_SECONDS = 3.5
const POSE_CONNECTIONS = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [0, 5],
  [0, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [15, 17],
  [16, 18],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [24, 26],
  [25, 27],
  [26, 28],
  [27, 29],
  [28, 30],
  [27, 31],
  [28, 32],
  [29, 31],
  [30, 32],
]

export default function WebcamViewer({ onSessionComplete, onNavigate, sessionIdOverride }) {
  const videoRef = useRef(null)
  const streamRef = useRef(null)
  const socketRef = useRef(null)
  const sessionIdRef = useRef(null)
  const baseAngleRef = useRef(null)
  const stageRef = useRef(null)
  const canvasRef = useRef(null)
  const landmarksRef = useRef(null)
  const calibrationTotalRef = useRef(CALIBRATION_SECONDS)
  const svgRef = useRef(null)

  const [isSessionActive, setIsSessionActive] = useState(false)
  const [isBusy, setIsBusy] = useState(false)
  const [error, setError] = useState('')
  const [sessionId, setSessionId] = useState('')
  const [postureData, setPostureData] = useState([])
  const [summary, setSummary] = useState(null)
  const [baseAngle, setBaseAngle] = useState(null)
  const [currentDelta, setCurrentDelta] = useState(0)
  const [trendSlope, setTrendSlope] = useState(0)
  const [calibrationRemaining, setCalibrationRemaining] = useState(null)
  const [calibrationTotal, setCalibrationTotal] = useState(CALIBRATION_SECONDS)

  const drawPose = useCallback((landmarks) => {
    landmarksRef.current = landmarks
    const stage = stageRef.current
    const canvas = canvasRef.current
    if (!stage || !canvas) return

    const width = stage.clientWidth
    const height = stage.clientHeight
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!Array.isArray(landmarks) || landmarks.length === 0) {
      return
    }

    const project = (lm) => [lm.x * width, lm.y * height]

    ctx.strokeStyle = 'rgba(120, 190, 255, 0.7)'
    ctx.lineWidth = 3
    ctx.lineCap = 'round'
    ctx.lineJoin = 'round'

    POSE_CONNECTIONS.forEach(([startIndex, endIndex]) => {
      const start = landmarks[startIndex]
      const end = landmarks[endIndex]
      if (!start || !end) return
      if ((start.visibility ?? 0) < 0.2 || (end.visibility ?? 0) < 0.2) return
      const [sx, sy] = project(start)
      const [ex, ey] = project(end)
      ctx.beginPath()
      ctx.moveTo(sx, sy)
      ctx.lineTo(ex, ey)
      ctx.stroke()
    })

    ctx.fillStyle = 'rgba(248, 253, 255, 0.92)'
    landmarks.forEach((lm) => {
      if (!lm) return
      if ((lm.visibility ?? 0) < 0.2) return
      const [x, y] = project(lm)
      ctx.beginPath()
      ctx.arc(x, y, 5, 0, Math.PI * 2)
      ctx.fill()
    })
  }, [])

  const resizeCanvas = useCallback(() => {
    const stage = stageRef.current
    const canvas = canvasRef.current
    if (!stage || !canvas) return

    const width = stage.clientWidth
    const height = stage.clientHeight
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width
      canvas.height = height
    }

    if (landmarksRef.current) {
      drawPose(landmarksRef.current)
    }
  }, [drawPose])

  useEffect(() => {
    sessionIdRef.current = sessionId
  }, [sessionId])

  useEffect(() => {
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)
    return () => {
      window.removeEventListener('resize', resizeCanvas)
    }
  }, [resizeCanvas])

  useEffect(() => {
    if (sessionIdOverride && !sessionId) {
      setSessionId(sessionIdOverride)
    }
  }, [sessionIdOverride, sessionId])

  useEffect(() => {
    const socket = io(API_BASE_URL, { transports: ['websocket'] })
    socketRef.current = socket

    socket.on('posture_frame', (payload) => {
      if (!sessionIdRef.current || payload.session_id !== sessionIdRef.current) {
        return
      }

      if (payload.type === 'calibrating') {
        const totalSeconds = Number(
          payload.total_seconds ?? calibrationTotalRef.current ?? CALIBRATION_SECONDS
        )
        calibrationTotalRef.current = totalSeconds
        setCalibrationTotal(totalSeconds)
        setCalibrationRemaining(Math.max(0, Number(payload.remaining_seconds ?? 0)))
        if (Array.isArray(payload.landmarks)) {
          drawPose(payload.landmarks)
        }
        return
      }

      if (payload.type === 'calibrated') {
        const baseline = Number(payload.base_nose_angle ?? 0)
        baseAngleRef.current = baseline
        setBaseAngle(baseline)
        if (payload.calibration_seconds !== undefined) {
          const totalSeconds = Number(payload.calibration_seconds) || CALIBRATION_SECONDS
          calibrationTotalRef.current = totalSeconds
          setCalibrationTotal(totalSeconds)
        }
        setCalibrationRemaining(null)
        return
      }

      if (payload.type !== 'frame') {
        return
      }

      const noseAngle = Number(payload.nose_angle ?? 0)
      const relativeTime = Number(payload.relative_time ?? payload.timestamp ?? 0)
      const baseline = Number(
        payload.base_nose_angle ?? baseAngleRef.current ?? noseAngle
      )
      const noseDelta =
        payload.nose_delta !== undefined
          ? Number(payload.nose_delta)
          : noseAngle - baseline
      const slope = Number(payload.trend_slope ?? 0)

      baseAngleRef.current = baseline
      setBaseAngle(baseline)
      setTrendSlope(slope)
      setCalibrationRemaining(null)
      setCurrentDelta(noseDelta)

      setPostureData((prev) => {
        const next = [...prev, { time: relativeTime, noseAngle, noseDelta }]
        if (next.length > 600) {
          next.shift()
        }
        return next
      })

      if (Array.isArray(payload.landmarks)) {
        drawPose(payload.landmarks)
      } else {
        drawPose(null)
      }
    })

    return () => {
      drawPose(null)
      socket.disconnect()
    }
  }, [drawPose])

  useEffect(() => {
    const svgElement = svgRef.current
    if (!svgElement) return

    const svg = d3.select(svgElement)
    const width = svgElement.clientWidth || 640
    const height = svgElement.clientHeight || 260

    svg.attr('viewBox', `0 0 ${width} ${height}`)
    svg.selectAll('*').remove()

    if (postureData.length === 0) {
      svg
        .append('text')
        .attr('x', width / 2)
        .attr('y', height / 2)
        .attr('text-anchor', 'middle')
        .attr('fill', '#64748b')
        .attr('font-size', 14)
        .text('start a session to capture your posture change')
      return
    }

    const margin = { top: 24, right: 24, bottom: 40, left: 56 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    const xExtent = d3.extent(postureData, (d) => d.time)
    const yExtent = d3.extent(postureData, (d) => d.noseDelta)

    const xScale = d3.scaleLinear().domain([Math.max(0, xExtent[0] ?? 0), xExtent[1] ?? 10]).range([0, innerWidth])
    const yMin = Math.min(-2, Math.floor((yExtent[0] ?? 0) - 2))
    const yMax = Math.max(2, Math.ceil((yExtent[1] ?? 0) + 2))
    const yScale = d3.scaleLinear().domain([yMin, yMax]).range([innerHeight, 0]).nice()

    const g = svg.append('g').attr('transform', `translate(${margin.left},${margin.top})`)

    const xAxis = d3.axisBottom(xScale).ticks(6).tickFormat((d) => `${d}s`)
    const yAxis = d3.axisLeft(yScale).ticks(6).tickFormat((d) => `${d}°`)

    g.append('g').attr('class', 'chart__axis').attr('transform', `translate(0,${innerHeight})`).call(xAxis)
    g.append('g').attr('class', 'chart__axis').call(yAxis)

    g.append('line')
      .attr('x1', 0)
      .attr('x2', innerWidth)
      .attr('y1', yScale(0))
      .attr('y2', yScale(0))
      .attr('stroke', '#94a3b8')
      .attr('stroke-dasharray', '4,4')
      .attr('stroke-width', 1)
      .attr('opacity', 0.6)

    g.append('path')
      .datum(postureData)
      .attr('fill', 'none')
      .attr('stroke', '#4f46e5')
      .attr('stroke-width', 2)
      .attr(
        'd',
        d3
          .line()
          .x((d) => xScale(d.time))
          .y((d) => yScale(d.noseDelta))
          .curve(d3.curveMonotoneX)
      )

    g.append('text').attr('x', innerWidth).attr('y', innerHeight + 32).attr('text-anchor', 'end').attr('fill', '#0f172a').text('seconds')

    g.append('text')
      .attr('x', -margin.left + 12)
      .attr('y', -12)
      .attr('text-anchor', 'start')
      .attr('fill', '#0f172a')
      .text('nose angle delta (degrees)')
  }, [postureData])

  useEffect(() => {
    return () => {
      stopStream()
      socketRef.current?.disconnect()
    }
  }, [])

  const stopStream = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop())
    streamRef.current = null
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
  }

  const fetchJson = async (path, options) => {
    const response = await fetch(`${API_BASE_URL}${path}`, options)
    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || 'request failed')
    }
    return response.json()
  }

  const startSession = async () => {
    setIsBusy(true)
    setError('')
    try {
      const payload = await fetchJson('/api/session/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ calibration_seconds: CALIBRATION_SECONDS }),
      })
      if (payload.status !== 'started') {
        throw new Error(payload.message || 'unable to start session')
      }
      setSessionId(payload.session_id)
      setSummary(null)
      setPostureData([])
      setBaseAngle(null)
      baseAngleRef.current = null
      setTrendSlope(0)
      setCurrentDelta(0)
      const sessionCalibrationSeconds =
        Number(payload.calibration_seconds ?? CALIBRATION_SECONDS) || CALIBRATION_SECONDS
      calibrationTotalRef.current = sessionCalibrationSeconds
      setCalibrationTotal(sessionCalibrationSeconds)
      setCalibrationRemaining(sessionCalibrationSeconds)
      drawPose(null)
      resizeCanvas()

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play()
          streamRef.current = stream
        }
      } catch (mediaError) {
        console.warn('webcam preview unavailable', mediaError)
        setError('session started, but webcam preview is unavailable. posture metrics will still stream.')
      }

      setIsSessionActive(true)
    } catch (err) {
      console.error(err)
      setError(err.message || 'unable to start session')
    } finally {
      setIsBusy(false)
    }
  }

  const stopSession = async () => {
    setIsBusy(true)
    setError('')
    try {
      const payload = await fetchJson('/api/session/stop', { method: 'POST' })
      if (payload.status === 'idle') {
        setError('no active session to stop.')
      } else {
        setSummary(payload)
        onSessionComplete?.(payload)
      }
    } catch (err) {
      console.error(err)
      setError(err.message || 'unable to stop session')
    } finally {
      stopStream()
      setIsSessionActive(false)
      setSessionId('')
      baseAngleRef.current = null
      setBaseAngle(null)
      setTrendSlope(0)
      setCurrentDelta(0)
      calibrationTotalRef.current = CALIBRATION_SECONDS
      setCalibrationTotal(CALIBRATION_SECONDS)
      setCalibrationRemaining(null)
      drawPose(null)
      setIsBusy(false)
    }
  }

  const toggleSession = () => {
    if (isBusy) return
    if (isSessionActive) {
      stopSession()
    } else {
      startSession()
    }
  }

  const handleViewSummary = () => {
    if (!summary && !sessionIdOverride) return
    onNavigate?.('summary')
  }

  return (
    <main className="page-shell">
      <div className="page-actions">
        <button
          type="button"
          className="primary-button primary-button--compact"
          onClick={() => onNavigate?.('home')}
        >
          <span>back to home</span>
        </button>
        <button
          type="button"
          className="primary-button primary-button--compact"
          onClick={handleViewSummary}
          disabled={!summary && !sessionIdOverride}
        >
          <span>view summary</span>
        </button>
      </div>

      <div className="page-header">
        <h1 className="page-title">live posture session</h1>
        <p className="page-subtitle">monitor your alignment in real-time with gentle visual guidance</p>
      </div>

      <section className="page-card page-card--wide">
        <div ref={stageRef} className="webcam-stage">
          <video ref={videoRef} className="webcam-video" playsInline muted />
          <canvas ref={canvasRef} className="webcam-overlay-canvas" />
          {!isSessionActive && <div className="webcam-overlay">camera idle</div>}
          {isSessionActive && calibrationRemaining !== null && (
            <div className="webcam-overlay">
              calibrating · hold steady ({calibrationRemaining.toFixed(1)}s / {calibrationTotal.toFixed(1)}s)
            </div>
          )}
        </div>
        <div className="webcam-controls">
          <button
            type="button"
            className={`primary-button primary-button--compact ${isSessionActive ? 'is-active' : ''}`}
            onClick={toggleSession}
            disabled={isBusy}
          >
            <span>{isBusy ? 'please wait...' : isSessionActive ? 'end session' : 'begin session'}</span>
          </button>
          {sessionId && <p className="page-note">session id: {sessionId}</p>}
          {error && <p className="page-note">{error}</p>}
          {baseAngle !== null && Number.isFinite(baseAngle) && (
            <p className="page-note">baseline nose angle: {baseAngle.toFixed(1)}°</p>
          )}
          {isSessionActive && Number.isFinite(currentDelta) && (
            <p className="page-note">
              current delta: {currentDelta >= 0 ? '+' : ''}
              {currentDelta.toFixed(2)}°
            </p>
          )}
          {isSessionActive && Number.isFinite(trendSlope) && (
            <p className="page-note">
              trend slope: {trendSlope >= 0 ? '+' : ''}
              {trendSlope.toFixed(3)}°/s
            </p>
          )}
        </div>
      </section>

      <section className="page-card page-card--wide">
        <h2 className="page-card-title">nose delta trend</h2>
        <svg ref={svgRef} className="posture-chart" role="img" aria-label="line chart showing nose angle delta over time" />
      </section>

      {summary && (
        <section className="page-card">
          <h2 className="page-card-title">session snapshot</h2>
          <ul className="page-list">
            <li>
              <span>duration</span>
              <span>{summary.duration_seconds ? summary.duration_seconds.toFixed(1) : 0} s</span>
            </li>
            <li>
              <span>frames processed</span>
              <span>{summary.frames ?? 0}</span>
            </li>
            <li>
              <span>baseline nose angle</span>
              <span>{summary.base_nose_angle ? summary.base_nose_angle.toFixed(1) : 0}°</span>
            </li>
            <li>
              <span>average nose angle</span>
              <span>{summary.average_nose_angle ? summary.average_nose_angle.toFixed(1) : 0}°</span>
            </li>
            <li>
              <span>trend slope</span>
              <span>
                {summary.trend_slope ? summary.trend_slope.toFixed(3) : 0}°/s
              </span>
            </li>
          </ul>
        </section>
      )}
    </main>
  )
}
