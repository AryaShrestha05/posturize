from __future__ import annotations

from typing import Generator, List
import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from flask import Flask, Response, jsonify
from flask_cors import CORS
from mediapipe.framework.formats import landmark_pb2
import threading
import time

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

state_lock = threading.Lock()
MAX_SAMPLES = 3600  # keep up to an hour of 1Hz samples

session_state = {
    "start_time": None,
    "calibration_start": None,
    "calibration_samples": [],
    "baseline_angle": None,
    "status_message": "calibrating…",
    "classification": "calibrating",
    "current_delta": 0.0,
    "data": deque(maxlen=MAX_SAMPLES),
    "last_logged_time": None,
}


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "*"}})

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/api/session/start")
    def start_session():
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
        return {"status": "stopped", "snapshot": snapshot}

    @app.get("/api/video_feed")
    def video_feed() -> Response:
        return Response(
            stream_with_visualization(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/session/posture")
    def posture_summary() -> Response:
        with state_lock:
            baseline = session_state["baseline_angle"]
            status_message = session_state["status_message"]
            classification = session_state["classification"]
            current_delta = session_state["current_delta"]
            start_time = session_state["start_time"]
            data_points = list(session_state["data"])

        intervals = compute_intervals(data_points)

        payload = {
            "baseline_angle": baseline,
            "status_message": status_message,
            "classification": classification,
            "current_delta": current_delta,
            "start_time": start_time,
            "samples": data_points,
            "intervals": intervals,
        }
        return jsonify(payload)

    return app


def _initialize_session_locked(now: float) -> None:
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam")

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            mirrored = cv2.flip(frame, 1)
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image_rgb)

            annotated = mirrored.copy()
            if results.pose_landmarks:
                update_session_state(results, frame, mirrored)
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

                cv2.putText(annotated, f"{int(left_angle)}°", tuple(viewer_left_point.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"{int(right_angle)}°", tuple(viewer_right_point.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(annotated, f"{int(nose_angle)}°", tuple(nose_point.astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode(".jpg", annotated)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
    finally:
        cap.release()
        pose.close()


def update_session_state(results, frame, mirrored) -> None:
    now = time.time()
    with state_lock:
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

    if baseline is None:
        elapsed = now - calibration_start
        samples.append(nose_angle)
        remaining = max(0.0, 3.0 - elapsed)
        status = f"Hold steady to capture baseline… {remaining:.1f}s"

        if elapsed >= 3.0 and samples:
            baseline = float(np.mean(samples))
            status = "Baseline captured. Tracking posture."

            with state_lock:
                session_state["baseline_angle"] = baseline
                session_state["calibration_samples"] = []

        with state_lock:
            session_state["status_message"] = status
            session_state["classification"] = "calibrating" if baseline is None else "good"
        return

    delta = float(nose_angle - baseline)

    # ✅ FIX: invert sign to match mirrored camera behavior
    delta *= -1

    classification = classify_delta(delta)

    with state_lock:
        session_state["status_message"] = classification_message(classification)
        session_state["classification"] = classification
        session_state["current_delta"] = delta

        last_logged = session_state["last_logged_time"]
        rel_time = now - start_time
        if last_logged is None or (now - last_logged) >= 1.0:
            session_state["data"].append({"time": rel_time, "delta": delta})
            session_state["last_logged_time"] = now


def overlay_status(frame: np.ndarray) -> None:
    with state_lock:
        status = session_state["status_message"]
        classification = session_state["classification"]

    color = {
        "good": (0, 255, 0),
        "moderate": (0, 215, 255),
        "bad": (0, 0, 255),
        "calibrating": (255, 255, 255),
    }.get(classification, (255, 255, 255))

    overlay = frame.copy()
    cv2.rectangle(overlay, (20, 20), (460, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
    cv2.putText(frame, status, (36, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def classify_delta(delta: float) -> str:
    if delta <= -10.0:
        return "bad"
    if delta <= -5.0:
        return "moderate"
    return "good"





def compute_intervals(data_points: List[dict]) -> List[dict]:
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
        classification = classify_delta(avg_delta)
        intervals.append({
            "start": start,
            "end": end,
            "average_delta": avg_delta,
            "classification": classification,
        })

    return intervals


def calculate_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))
    return float(angle)


app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
