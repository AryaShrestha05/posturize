from __future__ import annotations

import base64
import io
from typing import Dict, Optional

import matplotlib

# Use a non-interactive backend so matplotlib works without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def _encode_plot_to_png(fig: plt.Figure) -> str:
    """Render a matplotlib figure to a base64 encoded PNG string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    plt.close(fig)
    return encoded


def generate_posture_report(
    timestamps: np.ndarray,
    nose_angles: np.ndarray,
    *,
    base_angle: Optional[float] = None,
) -> Dict[str, object]:
    """Create a posture report from the collected samples.

    The report bundles numeric summary statistics, the raw arrays that the
    frontend can chart, and a ready-to-display PNG chart encoded as base64.
    """
    if timestamps.size == 0 or nose_angles.size == 0:
        return {
            "metrics": {
                "frames": 0,
                "average_nose_angle": 0.0,
                "min_nose_angle": 0.0,
                "max_nose_angle": 0.0,
                "stability_index": 0.0,
                "base_nose_angle": base_angle,
                "trend_slope": 0.0,
            },
            "chart": None,
            "raw": {
                "timestamps": [],
                "nose_angles": [],
                "nose_deltas": [],
            },
        }

    relative_times = timestamps - (timestamps[0] if timestamps.size > 0 else 0.0)
    base_angle_value = base_angle if base_angle is not None else float(np.mean(nose_angles))
    nose_deltas = nose_angles - base_angle_value

    metrics = {
        "frames": int(nose_angles.size),
        "average_nose_angle": float(np.mean(nose_angles)),
        "min_nose_angle": float(np.min(nose_angles)),
        "max_nose_angle": float(np.max(nose_angles)),
        # Standard deviation provides a rough sense of how steady the posture is.
        "stability_index": float(np.std(nose_angles)),
        "base_nose_angle": float(base_angle_value),
        "trend_slope": 0.0,
    }

    if relative_times.size >= 2:
        slope, _ = np.polyfit(relative_times, nose_deltas, 1)
        metrics["trend_slope"] = float(slope)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(relative_times, nose_angles, color="#4F46E5", linewidth=2, label="nose angle")
    ax.set_title("Nose Angle Over Time")
    ax.set_xlabel("Seconds since session start")
    ax.set_ylabel("Angle (degrees)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper right")

    chart_payload = {
        "image_base64": _encode_plot_to_png(fig),
        "content_type": "image/png",
    }

    raw_payload = {
        "timestamps": relative_times.round(3).tolist(),
        "nose_angles": nose_angles.round(3).tolist(),
        "nose_deltas": nose_deltas.round(3).tolist(),
    }

    return {"metrics": metrics, "chart": chart_payload, "raw": raw_payload}


