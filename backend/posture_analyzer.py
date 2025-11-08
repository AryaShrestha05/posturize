from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np

from .report_generator import generate_posture_report

mp_pose = mp.solutions.pose


@dataclass
class SessionData:
    session_id: str
    started_at: float
    timestamps: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    nose_angles: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    left_shoulder_angles: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    right_shoulder_angles: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    base_nose_angle: Optional[float] = None
    calibration_started_at: Optional[float] = None
    calibration_duration: float = 4.0
    calibration_angles: List[float] = field(default_factory=list)
    calibration_complete: bool = False

    def record_calibration_sample(self, timestamp: float, nose_angle: float) -> float:
        if self.calibration_started_at is None:
            self.calibration_started_at = timestamp
        self.calibration_angles.append(float(nose_angle))
        return self.calibration_remaining(timestamp)

    def calibration_remaining(self, timestamp: float) -> float:
        if self.calibration_started_at is None:
            return self.calibration_duration
        elapsed = timestamp - self.calibration_started_at
        return max(0.0, self.calibration_duration - elapsed)

    def finalize_calibration(self) -> None:
        if self.calibration_complete:
            return
        if self.calibration_angles:
            self.base_nose_angle = float(
                np.mean(np.array(self.calibration_angles, dtype=np.float32))
            )
        else:
            self.base_nose_angle = 0.0
        self.calibration_angles.clear()
        self.calibration_complete = True

    def append_sample(
        self, timestamp: float, nose_angle: float, left_angle: float, right_angle: float
    ) -> Dict[str, float]:
        if self.base_nose_angle is None:
            self.base_nose_angle = float(nose_angle)

        self.timestamps = np.append(self.timestamps, timestamp)
        self.nose_angles = np.append(self.nose_angles, nose_angle)
        self.left_shoulder_angles = np.append(self.left_shoulder_angles, left_angle)
        self.right_shoulder_angles = np.append(self.right_shoulder_angles, right_angle)

        relative_time = self.relative_timestamps()[-1]
        nose_delta = float(nose_angle - self.base_nose_angle)

        return {
            "relative_time": float(relative_time),
            "nose_delta": nose_delta,
            "trend_slope": self.trend_slope(),
        }

    def relative_timestamps(self) -> np.ndarray:
        if self.timestamps.size == 0:
            return np.array([], dtype=np.float64)
        return self.timestamps - self.timestamps[0]

    def trend_slope(self, window: int = 120) -> float:
        if self.timestamps.size < 2 or self.base_nose_angle is None:
            return 0.0

        if self.timestamps.size > window:
            timestamps = self.relative_timestamps()[-window:]
            deltas = (self.nose_angles[-window:] - self.base_nose_angle).astype(
                np.float64
            )
        else:
            timestamps = self.relative_timestamps()
            deltas = (self.nose_angles - self.base_nose_angle).astype(np.float64)

        if timestamps.size < 2:
            return 0.0

        slope, _ = np.polyfit(timestamps, deltas, 1)
        return float(slope)

    def to_summary(self) -> Dict[str, float | int | None]:
        frames = int(self.timestamps.size)
        if frames == 0:
            return {
                "session_id": self.session_id,
                "started_at": self.started_at,
                "duration_seconds": 0,
                "frames": 0,
                "average_nose_angle": 0.0,
                "average_left_shoulder_angle": 0.0,
                "average_right_shoulder_angle": 0.0,
                "base_nose_angle": self.base_nose_angle,
                "trend_slope": 0.0,
            }

        duration_seconds = float(self.timestamps[-1] - self.timestamps[0])
        nose_avg = float(np.mean(self.nose_angles))
        left_avg = float(np.mean(self.left_shoulder_angles))
        right_avg = float(np.mean(self.right_shoulder_angles))

        return {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "duration_seconds": float(duration_seconds),
            "frames": frames,
            "average_nose_angle": nose_avg,
            "average_left_shoulder_angle": left_avg,
            "average_right_shoulder_angle": right_avg,
            "base_nose_angle": self.base_nose_angle,
            "trend_slope": self.trend_slope(),
        }


class PostureAnalyzer:
    """Wraps MediaPipe pose detection and streams posture metrics."""

    def __init__(self, frame_callback: Callable[[str, Dict[str, Any]], None]):
        self._frame_callback = frame_callback
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._session: Optional[SessionData] = None
        self._summary_store: Dict[str, Dict[str, Any]] = {}
        self._cap: Optional[cv2.VideoCapture] = None

    def start_session(self, *, calibration_seconds: Optional[float] = None) -> Dict[str, Any]:
        with self._lock:
            if self._thread and self._thread.is_alive():
                assert self._session is not None
                return {
                    "status": "already_running",
                    "session_id": self._session.session_id,
                    "calibration_seconds": float(self._session.calibration_duration),
                }

            if calibration_seconds is not None:
                try:
                    calibrated_seconds = float(calibration_seconds)
                except (TypeError, ValueError):
                    calibrated_seconds = 3.5
                else:
                    if not np.isfinite(calibrated_seconds):
                        calibrated_seconds = 3.5
            else:
                calibrated_seconds = 3.5
            calibrated_seconds = float(
                min(max(calibrated_seconds, 3.0), 5.0)
            )
            session_id = uuid.uuid4().hex
            self._session = SessionData(
                session_id=session_id,
                started_at=time.time(),
                calibration_duration=calibrated_seconds,
            )
            self._stop_event.clear()

            self._thread = threading.Thread(target=self._run_capture_loop, daemon=True)
            self._thread.start()

        return {
            "status": "started",
            "session_id": session_id,
            "calibration_seconds": float(self._session.calibration_duration),
        }

    def stop_session(self) -> Dict[str, Any]:
        with self._lock:
            if not self._thread or not self._thread.is_alive():
                return {"status": "idle"}

            self._stop_event.set()
            self._thread.join(timeout=4)
            self._thread = None

            if self._cap is not None:
                self._cap.release()
                self._cap = None

            summary = self._session.to_summary() if self._session else {"status": "no_session"}
            if self._session:
                report = generate_posture_report(
                    self._session.relative_timestamps(),
                    self._session.nose_angles,
                    base_angle=self._session.base_nose_angle,
                )
                result: Dict[str, Any] = {**summary, "status": "completed", "report": report}
                self._summary_store[self._session.session_id] = result
            else:
                result = summary
            self._session = None

        return result

    def get_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._session and self._session.session_id == session_id:
                summary = self._session.to_summary()
                report = generate_posture_report(
                    self._session.relative_timestamps(),
                    self._session.nose_angles,
                    base_angle=self._session.base_nose_angle,
                )
                return {**summary, "status": "running", "report": report}

            return self._summary_store.get(session_id)

    def generate_report(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            if self._session and self._session.session_id == session_id:
                summary = self._session.to_summary()
                report = generate_posture_report(
                    self._session.relative_timestamps(),
                    self._session.nose_angles,
                    base_angle=self._session.base_nose_angle,
                )
                return {**summary, "status": "running", "report": report}

            stored = self._summary_store.get(session_id)

        return stored

    def _run_capture_loop(self):
        self._cap = cv2.VideoCapture(0)
        session_id = None
        with self._lock:
            if self._session:
                session_id = self._session.session_id

        if not self._cap.isOpened():
            if session_id:
                self._frame_callback(
                    session_id,
                    {"type": "error", "message": "unable to open webcam"},
                )
            return

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while not self._stop_event.is_set():
                success, frame = self._cap.read()
                if not success:
                    break

                metrics = self._process_frame(frame, pose)
                if metrics is None:
                    continue

                with self._lock:
                    session = self._session

                if session is None:
                    continue

                if not session.calibration_complete:
                    remaining = session.record_calibration_sample(
                        metrics["timestamp"], metrics["nose_angle"]
                    )
                    self._frame_callback(
                        session.session_id,
                        {
                            "type": "calibrating",
                            "remaining_seconds": remaining,
                            "total_seconds": session.calibration_duration,
                            "landmarks": metrics.get("landmarks"),
                        },
                    )
                    if remaining > 0:
                        continue
                    session.finalize_calibration()
                    self._frame_callback(
                        session.session_id,
                        {
                            "type": "calibrated",
                            "base_nose_angle": session.base_nose_angle,
                            "calibration_seconds": session.calibration_duration,
                        },
                    )
                    continue

                with self._lock:
                    if (
                        self._session is None
                        or self._session.session_id != session.session_id
                    ):
                        continue
                    append_payload = self._session.append_sample(
                        metrics["timestamp"],
                        metrics["nose_angle"],
                        metrics["left_shoulder_angle"],
                        metrics["right_shoulder_angle"],
                    )
                    callback_session_id = self._session.session_id
                    base_angle = self._session.base_nose_angle

                self._frame_callback(
                    callback_session_id,
                    {
                        **metrics,
                        **append_payload,
                        "type": "frame",
                        "base_nose_angle": base_angle if base_angle is not None else 0.0,
                    },
                )

        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _process_frame(self, frame, pose) -> Optional[Dict[str, float]]:
        timestamp = time.time()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        height, width, _ = frame.shape

        nose = np.array([lm[0].x * width, lm[0].y * height])
        left_shoulder = np.array([lm[11].x * width, lm[11].y * height])
        right_shoulder = np.array([lm[12].x * width, lm[12].y * height])

        left_angle = self._calculate_angle(nose, left_shoulder, right_shoulder)
        right_angle = self._calculate_angle(nose, right_shoulder, left_shoulder)
        nose_angle = self._calculate_angle(left_shoulder, nose, right_shoulder)

        landmarks_payload = [
            {
                "x": float(point.x),
                "y": float(point.y),
                "z": float(point.z),
                "visibility": float(point.visibility),
            }
            for point in lm
        ]

        return {
            "timestamp": float(timestamp),
            "nose_angle": float(nose_angle),
            "left_shoulder_angle": float(left_angle),
            "right_shoulder_angle": float(right_angle),
            "landmarks": landmarks_payload,
        }

    @staticmethod
    def _calculate_angle(a, b, c):
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return 0.0
        cosine_angle = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        angle = np.degrees(np.arccos(cosine_angle))
        return float(angle)

