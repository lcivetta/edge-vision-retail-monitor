"""Video detection, tracking, temporal rules, and human-review event generation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / ".ultralytics"))
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".matplotlib"))

import cv2
import torch
from ultralytics import YOLO

import config
from risk_engine import RiskEngine
from shelf_monitor import ShelfMonitor

EVENT_FIELDS = ["event_id", "video_timestamp", "person_id", "risk_score", "alert_level", "active_signals"]
FEEDBACK_FIELDS = EVENT_FIELDS + ["operator_label", "operator_notes", "timestamp"]


def choose_device(requested: str) -> str:
    mps_available = bool(torch.backends.mps.is_available())
    if requested == "mps" and not mps_available:
        raise RuntimeError("Apple MPS is not available in this PyTorch environment. Re-run with --device cpu.")
    if requested == "auto":
        return "mps" if mps_available else "cpu"
    return requested


def point_in_roi(box: tuple[int, int, int, int], roi: tuple[int, int, int, int]) -> bool:
    """Use box center, an explainable proxy for overlap with the display area."""
    x1, y1, x2, y2 = box
    point = ((x1 + x2) // 2, (y1 + y2) // 2)
    rx1, ry1, rx2, ry2 = roi
    return rx1 <= point[0] <= rx2 and ry1 <= point[1] <= ry2


def bag_is_near(person: tuple[int, int, int, int], bags: list[tuple[int, int, int, int]]) -> bool:
    px1, py1, px2, py2 = person
    pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
    scale = max(1.0, math.hypot(px2 - px1, py2 - py1))
    for bx1, by1, bx2, by2 in bags:
        bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
        overlap = max(0, min(px2, bx2) - max(px1, bx1)) * max(0, min(py2, by2) - max(py1, by1))
        if overlap > 0 or math.hypot(pcx - bcx, pcy - bcy) / scale <= config.BAG_PROXIMITY_DISTANCE:
            return True
    return False


def bag_intersects_product_change(
    bags: list[tuple[int, int, int, int]], regions: list[tuple[int, int, int, int]]
) -> bool:
    """Require spatial destination evidence, not mere bag presence."""
    for bx1, by1, bx2, by2 in bags:
        padding_x = int((bx2 - bx1) * 0.15)
        padding_y = int((by2 - by1) * 0.15)
        for rx1, ry1, rx2, ry2 in regions:
            overlap = (
                min(bx2 + padding_x, rx2) > max(bx1 - padding_x, rx1)
                and min(by2 + padding_y, ry2) > max(by1 - padding_y, ry1)
            )
            if overlap:
                return True
    return False


def point_in_box(point: tuple[int, int], box: tuple[int, int, int, int], padding: int = 0) -> bool:
    x, y = point
    x1, y1, x2, y2 = box
    return x1 - padding <= x <= x2 + padding and y1 - padding <= y <= y2 + padding


def detected_wrists(frame, pose_model: YOLO, device: str) -> list[tuple[int, int, float]]:
    """Return confident wrist keypoints from the lightweight pose model."""
    result = pose_model.predict(frame, device=device, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
    if result.keypoints is None or result.keypoints.xy is None or result.keypoints.conf is None:
        return []
    wrists = []
    for points, confidences in zip(result.keypoints.xy.cpu().tolist(), result.keypoints.conf.cpu().tolist()):
        for index in (9, 10):  # COCO left and right wrist keypoints
            confidence = float(confidences[index])
            if confidence >= config.WRIST_CONFIDENCE_THRESHOLD:
                wrists.append((int(points[index][0]), int(points[index][1]), confidence))
    return wrists


def init_csv(path: Path, fields: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        with path.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()


def append_review_event(event_id: str, timestamp: float, state, frame, events_csv: Path, feedback_csv: Path, alerts_dir: Path) -> None:
    row = {
        "event_id": event_id,
        "video_timestamp": f"{timestamp:.3f}",
        "person_id": state.person_id,
        "risk_score": f"{state.risk_score:.2f}",
        "alert_level": state.alert_level,
        "active_signals": "|".join(sorted(state.active_signals)),
    }
    with events_csv.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=EVENT_FIELDS).writerow(row)
    feedback_row = {**row, "operator_label": "", "operator_notes": "", "timestamp": ""}
    with feedback_csv.open("a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=FEEDBACK_FIELDS).writerow(feedback_row)
    image_path = alerts_dir / f"{event_id}_person_{state.person_id}.jpg"
    cv2.imwrite(str(image_path), frame)
    print(f"[ALERT] {event_id} saved metadata and {image_path}")


def run(args: argparse.Namespace) -> Path:
    config.ensure_directories()
    input_video = Path(args.input).expanduser().resolve()
    if not input_video.exists():
        raise FileNotFoundError(f"Input video missing: {input_video}")
    device = choose_device(args.device)
    if args.run_name:
        run_dir = config.RUNS_DIR / args.run_name
        alerts_dir = run_dir / "alerts"
        run_dir.mkdir(parents=True, exist_ok=True)
        alerts_dir.mkdir(parents=True, exist_ok=True)
        output_path = run_dir / f"{args.mode}_output.mp4"
        events_csv = run_dir / "events.csv"
        feedback_csv = run_dir / "feedback.csv"
    else:
        output_name = {"detection": "detection_output.mp4", "tracking": "tracked_output.mp4", "risk": "risk_output.mp4"}[args.mode]
        output_path = config.PROCESSED_DIR / output_name
        alerts_dir, events_csv, feedback_csv = config.ALERTS_DIR, config.EVENTS_CSV, config.FEEDBACK_CSV

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {input_video}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create {output_path}")

    model = YOLO(config.MODEL_NAME)
    bag_model = model if args.mode == "risk" else None  # reuse weights for per-frame bag context
    pose_model = YOLO(config.POSE_MODEL_NAME) if args.mode == "risk" else None
    print(f"Model: {config.MODEL_NAME}\nDevice: {device}\nVideo: {width}x{height} at {fps:.2f} FPS ({frame_count} frames)")
    roi_norm = config.MERCHANDISE_ROI
    roi = (int(roi_norm[0] * width), int(roi_norm[1] * height), int(roi_norm[2] * width), int(roi_norm[3] * height))
    shelf_norm = config.PRODUCT_SHELF_ROI
    shelf_roi = (
        int(shelf_norm[0] * width), int(shelf_norm[1] * height),
        int(shelf_norm[2] * width), int(shelf_norm[3] * height),
    )
    engine = RiskEngine()
    shelf_monitor = ShelfMonitor(shelf_roi, fps) if args.mode == "risk" else None
    last_roi_person_id: int | None = None
    last_table_wrist_contact: dict[int, float] = {}
    class_counts: Counter[str] = Counter()
    event_index = sum(1 for _ in events_csv.open()) if events_csv.exists() else 1
    if args.mode == "risk":
        init_csv(events_csv, EVENT_FIELDS)
        init_csv(feedback_csv, FEEDBACK_FIELDS)

    processed = 0
    max_risk = 0.0
    last_annotated = None
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or (args.max_frames and processed >= args.max_frames):
                break
            timestamp = processed / fps
            try:
                if args.mode == "detection":
                    result = model.predict(frame, device=device, conf=config.CONFIDENCE_THRESHOLD, verbose=False)[0]
                else:
                    # Only people need persistent pretrained tracks. Products are
                    # monitored class-agnostically inside the table ROI, while
                    # bags are detected separately as optional context.
                    result = model.track(
                        frame, persist=True, device=device, classes=[0, 24, 26],
                        conf=config.CONFIDENCE_THRESHOLD, verbose=False
                    )[0]
            except Exception as exc:
                if device == "mps":
                    raise RuntimeError(f"MPS inference failed ({exc}). Re-run with --device cpu.") from exc
                raise

            boxes = result.boxes
            detections = []
            if boxes is not None:
                ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(boxes)
                for xyxy, confidence, class_id, track_id in zip(
                    boxes.xyxy.int().cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.int().cpu().tolist(), ids
                ):
                    name = model.names[class_id]
                    class_counts[name] += 1
                    detections.append((tuple(xyxy), confidence, name, track_id))

            annotated = frame.copy()
            bags = [box for box, _, name, _ in detections if name in config.BAG_CLASS_NAMES]
            # The multi-object tracker may omit intermittent bag boxes before it
            # confirms a stable track. Bags are context only, so per-frame bag
            # detection is more honest than requiring a persistent bag ID.
            if bag_model is not None:
                bag_result = bag_model.predict(
                    frame, device=device, classes=[24, 26], conf=config.BAG_DETECTION_CONFIDENCE, verbose=False
                )[0]
                if bag_result.boxes is not None:
                    for xyxy, confidence, class_id in zip(
                        bag_result.boxes.xyxy.int().cpu().tolist(),
                        bag_result.boxes.conf.cpu().tolist(),
                        bag_result.boxes.cls.int().cpu().tolist(),
                    ):
                        name = bag_model.names[class_id]
                        class_counts[name] += 1
                        bags.append(tuple(xyxy))
                        detections.append((tuple(xyxy), confidence, name, None))
            wrists = detected_wrists(frame, pose_model, device) if pose_model is not None else []
            seen_people: set[int] = set()
            inside_people: set[int] = set()
            table_to_bag_people: set[int] = set()
            person_boxes = [box for box, _, name, track_id in detections if name == "person" and track_id is not None]
            if args.mode == "risk":
                cv2.rectangle(annotated, roi[:2], roi[2:], (0, 200, 255), 2)
                cv2.putText(annotated, "MERCHANDISE ZONE", (roi[0], max(25, roi[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                cv2.rectangle(annotated, shelf_roi[:2], shelf_roi[2:], (255, 180, 0), 2)
                cv2.putText(annotated, "PRODUCT SHELF", (shelf_roi[0], max(25, shelf_roi[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 180, 0), 2)

            for box, confidence, name, track_id in detections:
                x1, y1, x2, y2 = box
                if name == "person" and track_id is not None and args.mode != "detection":
                    seen_people.add(track_id)
                    label = f"Person #{track_id}"
                    if args.mode == "risk":
                        inside = point_in_roi(box, roi)
                        if inside:
                            inside_people.add(track_id)
                            last_roi_person_id = track_id
                        person_wrists = [(wx, wy) for wx, wy, _ in wrists if point_in_box((wx, wy), box)]
                        for wrist in person_wrists:
                            containing_bag = next((bag for bag in bags if point_in_box(wrist, bag, padding=5)), None)
                            if point_in_box(wrist, shelf_roi) and containing_bag is None:
                                last_table_wrist_contact[track_id] = timestamp
                                cv2.circle(annotated, wrist, 5, (255, 255, 0), -1)
                            elif containing_bag is not None and timestamp - last_table_wrist_contact.get(track_id, -1e9) <= config.TABLE_TO_BAG_MAX_SECONDS:
                                table_to_bag_people.add(track_id)
                                cv2.circle(annotated, wrist, 7, (0, 0, 255), -1)
                                cv2.putText(annotated, "TABLE -> BAG HAND PATH", (x1, min(height - 10, y2 + 36)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2)
                        changes = engine.observe(track_id, timestamp, inside, bag_is_near(box, bags))
                        state = engine.person_states[track_id]
                        max_risk = max(max_risk, state.risk_score)
                        for change in changes:
                            print(change.message)
                            if change.old_level != change.new_level:
                                print(f"[STATE] Person {track_id} -> {change.new_level} (risk {state.risk_score:.2f})")
                            if change.crossed_review_threshold:
                                event_id = f"EVT_{event_index:04d}"
                                event_index += 1
                                append_review_event(event_id, timestamp, state, annotated, events_csv, feedback_csv, alerts_dir)
                        label += f" Risk {state.risk_score:.2f} {state.alert_level.split(' - ')[0]}"
                        signals = ", ".join(sorted(state.active_signals))
                        if signals:
                            cv2.putText(annotated, signals, (x1, min(height - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
                else:
                    label = f"{name} {confidence:.2f}" + (f" #{track_id}" if track_id is not None else "")
                color = (60, 220, 60) if name == "person" else (255, 160, 40)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)

            if args.mode == "risk":
                # Mask people inside the ROI, but keep observing the rest of the
                # table. Pausing here previously hid a product removal whenever
                # the shopper stayed beside the table for the whole clip.
                shelf = shelf_monitor.update(frame, person_boxes, paused=False)
                for index, product_box in enumerate(shelf.region_boxes, start=1):
                    cv2.rectangle(annotated, product_box[:2], product_box[2:], (0, 0, 255), 2)
                    cv2.putText(
                        annotated, f"PRODUCT CHANGE {index}",
                        (product_box[0], max(18, product_box[1] - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (0, 0, 255), 2,
                    )
                if last_roi_person_id is not None:
                    pose_bag_destination = last_roi_person_id in table_to_bag_people
                    bag_destination = pose_bag_destination or (
                        shelf.persistent and bag_intersects_product_change(bags, shelf.region_boxes)
                    )
                    removal_evidence = shelf.product_removed or bag_destination
                    shelf_changes = engine.apply_shelf_state(
                        last_roi_person_id,
                        timestamp,
                        removal_evidence,
                        removal_evidence,
                        len(shelf.removed_region_boxes) or (1 if bag_destination else 0),
                        person_exited=not inside_people,
                    )
                    if bag_destination:
                        shelf_changes += engine.mark_concealed(
                            last_roi_person_id, 1, timestamp,
                            "backpack opening after table-to-bag hand path" if pose_bag_destination else "backpack opening"
                        )
                    state = engine.person_states.get(last_roi_person_id)
                    if state is not None:
                        max_risk = max(max_risk, state.risk_score)
                        for change in shelf_changes:
                            print(change.message)
                            if change.old_level != change.new_level:
                                print(
                                    f"[STATE] Person {last_roi_person_id} -> {change.new_level} "
                                    f"(risk {state.risk_score:.2f})"
                                )
                            if change.crossed_review_threshold:
                                event_id = f"EVT_{event_index:04d}"
                                event_index += 1
                                append_review_event(
                                    event_id, timestamp, state, annotated, events_csv, feedback_csv, alerts_dir
                                )
                shelf_text = (
                    f"TABLE {'PRODUCT REMOVED' if shelf.product_removed else ('REARRANGED' if shelf.changed else 'BASELINE')} "
                    f"diff {shelf.change_ratio:.3f} regions {shelf.estimated_regions}"
                )
                cv2.putText(annotated, shelf_text, (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1)
                for expired in engine.expire_missing(timestamp):
                    print(f"[STATE] Person {expired.person_id} finalized after tracking grace period (risk {expired.risk_score:.2f})")
                cv2.rectangle(annotated, (8, 8), (250, 82), (20, 20, 20), -1)
                for i, line in enumerate(("NORMAL < .15", "TRACKING .15-.29", "AMBIGUOUS .30-.59", "REVIEW >= .60")):
                    cv2.putText(annotated, line, (16, 25 + 17 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (240, 240, 240), 1)

            writer.write(annotated)
            last_annotated = annotated.copy()
            processed += 1
            if args.show:
                cv2.imshow("Edge Vision Retail Monitor - press q to stop", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped live visualization after q was pressed; saved frames processed so far.")
                    break
            if processed % 100 == 0 or processed == frame_count:
                print(f"Progress: {processed}/{frame_count} frames")
    finally:
        capture.release()
        writer.release()
        if args.show:
            cv2.destroyAllWindows()

    final_timestamp = processed / fps
    for change in engine.finalize_video(final_timestamp):
        state = engine.person_states.get(change.person_id)
        if state is None:
            continue
        max_risk = max(max_risk, state.risk_score)
        print(change.message)
        if change.crossed_review_threshold and last_annotated is not None:
            event_id = f"EVT_{event_index:04d}"
            event_index += 1
            append_review_event(event_id, final_timestamp, state, last_annotated, events_csv, feedback_csv, alerts_dir)
    final_risk = engine.current_risk()

    elapsed = time.perf_counter() - started
    print("Detected classes (frame-level detection counts):")
    for name, count in class_counts.most_common():
        print(f"  {name}: {count}")
    print(f"Processed {processed} frames in {elapsed:.2f}s ({processed / elapsed:.2f} processing FPS)")
    verify = cv2.VideoCapture(str(output_path))
    valid = verify.isOpened() and int(verify.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    verify.release()
    if not valid:
        raise RuntimeError(f"Output validation failed for {output_path}")
    print(f"Verified output video: {output_path}")
    if args.run_name:
        outcome = engine.outcome_for(final_risk)
        expected = args.expected_label.upper()
        evaluation = "NOT_SCORED"
        if expected in {"SHOPLIFTING", "NORMAL"}:
            evaluation = "PASS" if (
                (expected == "SHOPLIFTING" and outcome == "REVIEW")
                or (expected == "NORMAL" and outcome == "NO_REVIEW")
            ) else "FAIL"
        manifest = {
            "run_name": args.run_name,
            "input_video": str(input_video),
            "expected_label": expected,
            "system_outcome": outcome,
            "evaluation": evaluation,
            "max_risk": round(max_risk, 3),
            "frames": processed,
            "detected_classes": json.dumps(dict(class_counts), sort_keys=True),
            "model": config.MODEL_NAME,
            "device": device,
            "output_video": str(output_path),
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        new_file = not config.RUN_RESULTS_CSV.exists()
        with config.RUN_RESULTS_CSV.open("a", newline="") as handle:
            result_writer = csv.DictWriter(handle, fieldnames=list(manifest))
            if new_file:
                result_writer.writeheader()
            result_writer.writerow(manifest)
        print(f"Run outcome: {outcome}; evaluation against {expected}: {evaluation}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--mode", choices=("detection", "tracking", "risk"), default="risk")
    parser.add_argument("--max-frames", type=int, default=0, help="Optional test limit; 0 processes the full video")
    parser.add_argument("--input", default=str(config.INPUT_VIDEO), help="Video to process")
    parser.add_argument("--run-name", help="Store outputs and a manifest under output/runs/<name>")
    parser.add_argument("--expected-label", choices=("unknown", "normal", "shoplifting"), default="unknown")
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open a live OpenCV visualization while continuing to save output; press q to stop",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
