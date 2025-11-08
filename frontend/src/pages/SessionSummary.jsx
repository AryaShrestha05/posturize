import { useCallback, useEffect, useMemo, useState } from 'react'

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:5000'

export default function SessionSummary({ summary, sessionId, onNavigate }) {
  const [data, setData] = useState(summary ?? null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    setData(summary ?? null)
  }, [summary])

  const fetchJson = useCallback(async (path, options) => {
    const response = await fetch(`${API_BASE_URL}${path}`, options)
    if (!response.ok) {
      const message = await response.text()
      throw new Error(message || 'request failed')
    }
    return response.json()
  }, [])

  const loadSummary = useCallback(async () => {
    if (!sessionId) return
    setIsLoading(true)
    setError('')
    try {
      const result = await fetchJson(`/api/session/${sessionId}/summary`)
      setData(result)
    } catch (err) {
      console.error(err)
      setError(err.message || 'unable to load session summary')
    } finally {
      setIsLoading(false)
    }
  }, [fetchJson, sessionId])

  useEffect(() => {
    if (summary || !sessionId) return
    let active = true
    setIsLoading(true)
    setError('')

    fetchJson(`/api/session/${sessionId}/summary`)
      .then((result) => {
        if (active) {
          setData(result)
        }
      })
      .catch((err) => {
        if (!active) return
        console.error(err)
        setError(err.message || 'unable to load session summary')
      })
      .finally(() => {
        if (active) {
          setIsLoading(false)
        }
      })

    return () => {
      active = false
    }
  }, [summary, sessionId, fetchJson])

  const metrics = useMemo(() => (data?.report?.metrics ? { ...data.report.metrics } : null), [data])
  const rawSeries = useMemo(() => data?.report?.raw ?? null, [data])
  const chartImage = data?.report?.chart?.image_base64 ?? null

  const hasData = Boolean(data)
  const formatNumber = (value, digits = 1) =>
    Number.isFinite(value) ? Number(value).toFixed(digits) : '0'
  const baselineAngle = data?.base_nose_angle ?? metrics?.base_nose_angle ?? null
  const averageAngle = data?.average_nose_angle ?? metrics?.average_nose_angle ?? null
  const maxAngle = metrics?.max_nose_angle ?? null
  const stabilityIndex = metrics?.stability_index ?? null
  const trendSlopeValue = data?.trend_slope ?? metrics?.trend_slope ?? null
  const framesProcessed = data?.frames ?? metrics?.frames ?? 0

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
          onClick={() => onNavigate?.('live')}
        >
          <span>return to live</span>
        </button>
        {sessionId && (
          <button
            type="button"
            className="primary-button primary-button--compact"
            onClick={loadSummary}
            disabled={isLoading}
          >
            <span>{isLoading ? 'loading…' : 'refresh summary'}</span>
          </button>
        )}
      </div>

      <div className="page-header">
        <h1 className="page-title">session summary</h1>
        <p className="page-subtitle">
          baseline posture, trend slope, and key stats from your latest calibration—captured directly from the posture analyzer.
        </p>
      </div>

      {error && (
        <section className="page-card">
          <h2 className="page-card-title">unable to load summary</h2>
          <p className="page-note">{error}</p>
          {sessionId && (
            <button
              type="button"
              className="primary-button primary-button--compact"
              onClick={loadSummary}
              disabled={isLoading}
            >
              <span>try again</span>
            </button>
          )}
        </section>
      )}

      {!hasData && !isLoading && !error && (
        <section className="page-card">
          <h2 className="page-card-title">no session yet</h2>
          <p className="page-note">
            start a live session to generate a posture report. once you stop the capture, your baseline and trend insights will appear here.
          </p>
        </section>
      )}

      {isLoading && (
        <section className="page-card">
          <h2 className="page-card-title">loading summary</h2>
          <p className="page-note">fetching your latest posture metrics…</p>
        </section>
      )}

      {hasData && (
        <>
          <section className="page-grid">
            <article className="page-card">
              <h2 className="page-card-title">core metrics</h2>
              <ul className="page-list">
                <li>
                  <span>session id</span>
                  <span>{data.session_id}</span>
                </li>
                <li>
                  <span>duration</span>
                  <span>{data.duration_seconds ? data.duration_seconds.toFixed(1) : 0} s</span>
                </li>
                <li>
                  <span>frames processed</span>
                  <span>{framesProcessed}</span>
                </li>
                <li>
                  <span>baseline nose angle</span>
                  <span>{`${formatNumber(baselineAngle, 1)}°`}</span>
                </li>
              </ul>
            </article>

            <article className="page-card">
              <h2 className="page-card-title">trend insights</h2>
              <ul className="page-list">
                <li>
                  <span>average nose angle</span>
                  <span>{`${formatNumber(averageAngle, 1)}°`}</span>
                </li>
                <li>
                  <span>max nose angle</span>
                  <span>{`${formatNumber(maxAngle, 1)}°`}</span>
                </li>
                <li>
                  <span>stability index</span>
                  <span>{formatNumber(stabilityIndex, 2)}</span>
                </li>
                <li>
                  <span>trend slope</span>
                  <span>{`${formatNumber(trendSlopeValue, 3)}°/s`}</span>
                </li>
              </ul>
            </article>

            {chartImage && (
              <article className="page-card page-card--wide">
                <h2 className="page-card-title">nose angle chart</h2>
                <img
                  className="summary-chart"
                  src={`data:image/png;base64,${chartImage}`}
                  alt="line chart of nose angle over time from the session"
                />
              </article>
            )}

            {rawSeries && rawSeries.timestamps?.length > 0 && (
              <article className="page-card page-card--wide">
                <h2 className="page-card-title">raw arrays</h2>
                <p className="page-note">
                  stored entirely in memory using numpy—ideal for experimentation or exporting to notebooks. below are the first few values.
                </p>
                <div className="summary-raw-grid">
                  <div>
                    <h3 className="page-note">timestamps (s)</h3>
                    <p className="page-note">
                      {rawSeries.timestamps.slice(0, 8).join(', ')}
                      {rawSeries.timestamps.length > 8 ? ', …' : ''}
                    </p>
                  </div>
                  <div>
                    <h3 className="page-note">nose angles (°)</h3>
                    <p className="page-note">
                      {rawSeries.nose_angles.slice(0, 8).join(', ')}
                      {rawSeries.nose_angles.length > 8 ? ', …' : ''}
                    </p>
                  </div>
                  {rawSeries.nose_deltas && (
                    <div>
                      <h3 className="page-note">nose deltas (°)</h3>
                      <p className="page-note">
                        {rawSeries.nose_deltas.slice(0, 8).join(', ')}
                        {rawSeries.nose_deltas.length > 8 ? ', …' : ''}
                      </p>
                    </div>
                  )}
                </div>
              </article>
            )}
          </section>
        </>
      )}
    </main>
  )
}
