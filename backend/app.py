from __future__ import annotations

import threading
import time
from collections import deque
from typing import Generator, List

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, jsonify
from flask_cors import CORS
from mediapipe.framework.formats import landmark_pb2

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# We store at most one hour of samples (1 per second) in memory – no database needed.
MAX_SAMPLES = 3600
# Duration of the stillness window we use to learn the user's neutral head angle.
CALIBRATION_SECONDS = 3.0

state_lock = threading.Lock()

# session_state keeps all of the posture session metadata in-memory. We guard every
# read/write with `state_lock`, so the camera thread and HTTP handlers do not race.
session_state = {
    "is_running": False,
    "start_time": None,
    "calibration_start": None,
    "calibration_samples": [],
    "baseline_angle": None,
    "status_message": "Session idle. Start to begin calibration.",
    "classification": "idle",
    "current_delta": 0.0,
    "data": deque(maxlen=MAX_SAMPLES),
    "last_logged_time": None,
}


def create_app() -> Flask:
    """Configure the Flask app, register the endpoints, and wire up CORS."""
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})
    app.config['PROPAGATE_EXCEPTIONS'] = True

    @app.errorhandler(Exception)
    def handle_error(error):
        import traceback
        traceback.print_exc()
        return {"error": str(error)}, 500

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/session/start")
    def start_session():
        # Reset all posture state so the backend returns to the calibration phase.
        reset_session_state()
        return {"status": "started"}

    @app.post("/api/session/stop")
    def stop_session():
        with state_lock:
            snapshot = {
                "baseline_angle": session_state["baseline_angle"],
                "data": list(session_state["data"]),
                "intervals": compute_intervals(list(session_state["data"])),
            }
            # Flip the running flag so the stream loop knows to ignore stale frames.
            session_state["is_running"] = False
            session_state["status_message"] = "Session stopped. Start to capture a new baseline."
            session_state["classification"] = "idle"
        return {"status": "stopped", "snapshot": snapshot}

    @app.get("/api/video_feed")
    def video_feed() -> Response:
        try:
            return Response(
                stream_with_visualization(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 500

    @app.get("/api/session/posture")
    def posture_summary() -> Response:
        with state_lock:
            payload = {
                "baseline_angle": session_state["baseline_angle"],
                "status_message": session_state["status_message"],
                "classification": session_state["classification"],
                "current_delta": session_state["current_delta"],
                "start_time": session_state["start_time"],
                "samples": list(session_state["data"]),
                "intervals": compute_intervals(list(session_state["data"])),
                "is_running": session_state["is_running"],
            }
        return jsonify(payload)

    return app


def _initialize_session_locked(now: float) -> None:
    session_state["is_running"] = True
    session_state["start_time"] = now
    session_state["calibration_start"] = now
    session_state["calibration_samples"] = []
    session_state["baseline_angle"] = None
    session_state["status_message"] = "Hold steady to capture your baseline posture…"
    session_state["classification"] = "calibrating"
    session_state["current_delta"] = 0.0
    session_state["data"] = deque(maxlen=MAX_SAMPLES)
    session_state["last_logged_time"] = None


def reset_session_state() -> None:
    now = time.time()
    with state_lock:
        _initialize_session_locked(now)


def stream_with_visualization() -> Generator[bytes, None, None]:
    """Capture webcam frames, draw pose overlays, and stream them as MJPEG."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Return a placeholder black frame if webcam is not available
        print("WARNING: Webcam not available, streaming placeholder frames")
        while True:
            # Create a black frame
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "Webcam not available", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, buffer = cv2.imencode('.jpg', frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
        return

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            mirrored = cv2.flip(frame, 1)
            annotated = mirrored.copy()

            if results.pose_landmarks:
                update_session_state(results, frame)

                mirrored_landmarks = landmark_pb2.NormalizedLandmarkList()
                for landmark in results.pose_landmarks.landmark:
                    mirrored_landmark = mirrored_landmarks.landmark.add()
                    mirrored_landmark.x = 1.0 - landmark.x
                    mirrored_landmark.y = landmark.y
                    mirrored_landmark.z = landmark.z
                    mirrored_landmark.visibility = landmark.visibility

                mp_drawing.draw_landmarks(
                    annotated,
                    mirrored_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245, 255, 255), thickness=2, circle_radius=2),
                )

                h, w, _ = annotated.shape

                def to_pixel(normalized_landmark):
                    return np.array([normalized_landmark.x * w, normalized_landmark.y * h])

                nose_point = to_pixel(mirrored_landmarks.landmark[0])
                viewer_left_point = to_pixel(mirrored_landmarks.landmark[12])
                viewer_right_point = to_pixel(mirrored_landmarks.landmark[11])

                for point in (nose_point, viewer_left_point, viewer_right_point):
                    cv2.circle(annotated, tuple(point.astype(int)), 4, (245, 255, 255), -1)

                left_angle = calculate_angle(nose_point, viewer_left_point, viewer_right_point)
                right_angle = calculate_angle(nose_point, viewer_right_point, viewer_left_point)
                nose_angle = calculate_angle(viewer_left_point, nose_point, viewer_right_point)

                if not np.isnan(left_angle):
                    cv2.putText(
                        annotated,
                        f"{int(round(left_angle))}°",
                        tuple(viewer_left_point.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if not np.isnan(right_angle):
                    cv2.putText(
                        annotated,
                        f"{int(round(right_angle))}°",
                        tuple(viewer_right_point.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                if not np.isnan(nose_angle):
                    cv2.putText(
                        annotated,
                        f"{int(round(nose_angle))}°",
                        tuple(nose_point.astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )

            ret, buffer = cv2.imencode(".jpg", annotated)
            if not ret:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
    finally:
        cap.release()
        pose.close()


def update_session_state(results, frame) -> None:
    """Update calibration and posture metrics for the latest MediaPipe frame."""
    now = time.time()
    with state_lock:
        if not session_state["is_running"]:
            return
        if session_state["start_time"] is None:
            _initialize_session_locked(now)
        start_time = session_state["start_time"]
        calibration_start = session_state["calibration_start"]
        baseline = session_state["baseline_angle"]
        samples = session_state["calibration_samples"]

    landmarks = results.pose_landmarks.landmark
    nose = np.array([landmarks[0].x * frame.shape[1], landmarks[0].y * frame.shape[0]])
    left_shoulder = np.array([landmarks[11].x * frame.shape[1], landmarks[11].y * frame.shape[0]])
    right_shoulder = np.array([landmarks[12].x * frame.shape[1], landmarks[12].y * frame.shape[0]])
    nose_angle = calculate_angle(left_shoulder, nose, right_shoulder)

    if np.isnan(nose_angle):
        # Skip frames with unreliable geometry (landmarks collapsed)
        return

    if baseline is None:
        elapsed = now - calibration_start
        samples.append(nose_angle)
        remaining = max(0.0, CALIBRATION_SECONDS - elapsed)
        status = f"Hold steady to capture baseline… {remaining:.1f}s"

        if elapsed >= CALIBRATION_SECONDS and samples:
            baseline = float(np.mean(samples))
            status = "Baseline captured. Tracking posture."

            with state_lock:
                session_state["baseline_angle"] = baseline
                session_state["calibration_samples"] = []

        with state_lock:
            session_state["status_message"] = status
            session_state["classification"] = "calibrating" if baseline is None else "good"
        return

    delta = float(nose_angle - baseline)  # Positive == looking up, negative == looking down.
    classification = classify_delta(delta)

    with state_lock:
        session_state["status_message"] = classification_message(classification)
        session_state["classification"] = classification
        session_state["current_delta"] = delta

        rel_time = now - start_time
        last_logged = session_state["last_logged_time"]
        if last_logged is None or (now - last_logged) >= 1.0:
            session_state["data"].append({"time": rel_time, "delta": delta})
            session_state["last_logged_time"] = now


def classify_delta(delta: float) -> str:
    """Bucket the head tilt delta into one of our posture zones."""
    if delta >= 10.0:
        return "bad"
    if delta >= 5.0:
        return "moderate"
    return "good"


def classification_message(classification: str) -> str:
    return {
        "bad": "Adjust now: posture is poor.",
        "moderate": "Heads up: posture drifting.",
        "good": "Posture on point.",
        "calibrating": "Calibrating posture…",
        "idle": "Session idle. Start to begin calibration.",
    }[classification]


def compute_intervals(data_points: List[dict]) -> List[dict]:
    """Roll up 1 Hz samples into 10 second windows so the D3 chart stays tidy."""
    if not data_points:
        return []

    last_time = data_points[-1]["time"]
    interval_count = int(last_time // 10) + 1
    intervals = []

    for idx in range(interval_count):
        start = idx * 10
        end = start + 10
        window = [p for p in data_points if start <= p["time"] < end]
        if not window:
            continue
        avg_delta = float(np.mean([p["delta"] for p in window]))
        intervals.append(
            {
                "start": start,
                "end": end,
                "average_delta": avg_delta,
                "classification": classify_delta(avg_delta),
            }
        )

    return intervals


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the angle ABC in degrees, guarding against collapsed vectors."""
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0.0:
        return float("nan")
    cosine_angle = np.dot(ba, bc) / denom
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return float(angle)


app = create_app()


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, threaded=True)
