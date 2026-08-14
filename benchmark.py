"""Measure YOLO inference only, using the same video, model, and frame sample."""

from __future__ import annotations

import argparse
import csv
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import cv2
import torch
from ultralytics import YOLO

import config


def load_frames(limit: int) -> list:
    capture = cv2.VideoCapture(str(config.INPUT_VIDEO))
    frames = []
    while len(frames) < limit:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    return frames


def measure(device: str, frames: list, trials: int) -> list[dict[str, str]]:
    model = YOLO(config.MODEL_NAME)
    model.predict(frames[0], device=device, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
    rows = []
    for trial in range(1, trials + 1):
        started = time.perf_counter()
        for frame in frames:
            model.predict(frame, device=device, conf=config.CONFIDENCE_THRESHOLD, verbose=False)
        elapsed = time.perf_counter() - started
        rows.append({
            "device": device,
            "trial": str(trial),
            "frames": str(len(frames)),
            "total_seconds": f"{elapsed:.6f}",
            "milliseconds_per_frame": f"{elapsed * 1000 / len(frames):.3f}",
            "fps": f"{len(frames) / elapsed:.3f}",
            "status": "measured",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    config.ensure_directories()
    if not config.INPUT_VIDEO.exists():
        raise FileNotFoundError(config.INPUT_VIDEO)
    frames = load_frames(args.frames)
    if not frames:
        raise RuntimeError("No video frames were decoded")
    rows = measure("cpu", frames, args.trials)
    if torch.backends.mps.is_available():
        try:
            rows.extend(measure("mps", frames, args.trials))
        except Exception as exc:
            rows.append({"device": "mps", "trial": "", "frames": str(len(frames)), "total_seconds": "", "milliseconds_per_frame": "", "fps": "", "status": f"failed: {exc}"})
    else:
        rows.append({"device": "mps", "trial": "", "frames": str(len(frames)), "total_seconds": "", "milliseconds_per_frame": "", "fps": "", "status": "unavailable: torch.backends.mps.is_available() is False"})
    fields = ["device", "trial", "frames", "total_seconds", "milliseconds_per_frame", "fps", "status"]
    with config.BENCHMARK_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)
    print(f"Saved {config.BENCHMARK_CSV}")


if __name__ == "__main__":
    main()
