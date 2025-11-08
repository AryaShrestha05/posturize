# HackPrinceton Posture Analyzer

## Prerequisites
- Python 3.10+
- Node.js 20+
- A webcam accessible from the machine running the backend

## Backend Setup
```bash
cd /Users/aryashrestha/Projects/hackprinceton
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend.app
```
The backend exposes REST endpoints on `http://localhost:5000` and pushes live posture updates over Socket.IO.

## Frontend Setup
```bash
cd /Users/aryashrestha/Projects/hackprinceton/frontend
npm install
npm run dev -- --host
```
Open the Vite dev server URL (typically `http://localhost:5173`) in your browser. The frontend automatically targets `http://localhost:5000` for API/WebSocket traffic; override by setting `VITE_API_BASE_URL`.

## Workflow
1. From the landing page, choose **press space to start** to open the live view.
2. Click **begin session**. The backend starts capturing frames, the webcam preview appears, and a 3.5 s calibration countdown overlays the video.
3. Hold still while the countdown runs. The analyzer samples the nose angle during this 3–5 second window, stores it as the baseline in NumPy arrays, and then begins streaming posture deltas and trend slopes.
4. Click **end session** to finalize. The data remains in memory (no database). Review metrics via **view summary**.

## Notes
- Calibration duration can be tuned per session (3–5 seconds); the frontend currently requests 3.5 seconds.
- All posture metrics (timestamps, nose angles, deltas) are retained in NumPy arrays for the life of the backend process.
- Charts are rendered client-side with D3 during a session and Matplotlib for saved summaries.
