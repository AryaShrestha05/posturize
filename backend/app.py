from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS
from flask_socketio import SocketIO

from .posture_analyzer import PostureAnalyzer


def create_app() -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/*": {"origins": "http://localhost:5173"}})

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

    analyzer = PostureAnalyzer(
        frame_callback=lambda session_id, payload: socketio.emit(
            "posture_frame", {"session_id": session_id, **payload}
        )
    )

    @app.post("/api/session/start")
    def start_session():
        payload = request.get_json(silent=True) or {}
        calibration_seconds = payload.get("calibration_seconds") or payload.get("calibrationSeconds")
        result = analyzer.start_session(calibration_seconds=calibration_seconds)
        status_code = 200 if result["status"] == "started" else 409
        return jsonify(result), status_code

    @app.post("/api/session/stop")
    def stop_session():
        summary = analyzer.stop_session()
        return jsonify(summary)

    @app.get("/api/session/<session_id>/summary")
    def session_summary(session_id: str):
        summary = analyzer.get_summary(session_id)
        if summary is None:
            return {"message": "session not found"}, 404
        return jsonify(summary)

    @app.get("/api/session/<session_id>/report")
    def session_report(session_id: str):
        report = analyzer.generate_report(session_id)
        if report is None:
            return {"message": "session not found"}, 404
        return jsonify(report)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @socketio.on("connect")
    def handle_connect():
        socketio.emit("status", {"message": "connected"})

    app.socketio = socketio  # type: ignore[attr-defined]
    return app


def run_app():
    app = create_app()
    socketio: SocketIO = app.socketio  # type: ignore[attr-defined]
    socketio.run(app, host="0.0.0.0", port=5000)


if __name__ == "__main__":
    run_app()