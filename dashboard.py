"""Interactive Streamlit dashboard for inspecting retail-video inference."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import cv2
import streamlit as st
from ultralytics import YOLO

import config
from main import bag_intersects_product_change, bag_is_near, choose_device, detected_wrists, point_in_box, point_in_roi
from risk_engine import RiskEngine
from shelf_monitor import ShelfMonitor

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad" / "Dataset"
CURATED = {
    "Curated normal": DATASET / "Normal Good data",
    "Curated shoplifting": DATASET / "Shoplifitng Good Data",
}
DASHBOARD_RUNS = config.OUTPUT_DIR / "dashboard_runs"
UPLOADS = config.OUTPUT_DIR / "uploads"
FEEDBACK_FIELDS = ["timestamp", "run_name", "video", "operator_label", "notes"]
REVIEW_STATUS_PATH = config.OUTPUT_DIR / "review_status.csv"
REVIEW_STATUS_FIELDS = ["review_key", "status", "manager_notes", "reviewed_at"]


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "video"


def available_videos(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.mp4"), key=lambda path: path.name.lower()) if folder.exists() else []


def save_upload(uploaded) -> Path:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    path = UPLOADS / safe_name(uploaded.name)
    path.write_bytes(uploaded.getbuffer())
    return path


def load_review_statuses() -> dict[str, dict]:
    if not REVIEW_STATUS_PATH.exists():
        return {}
    with REVIEW_STATUS_PATH.open(newline="") as handle:
        return {row["review_key"]: row for row in csv.DictReader(handle)}


def save_review_status(review_key: str, status: str, notes: str) -> None:
    statuses = load_review_statuses()
    statuses[review_key] = {
        "review_key": review_key,
        "status": status,
        "manager_notes": notes,
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
    }
    REVIEW_STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = REVIEW_STATUS_PATH.with_suffix(".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_STATUS_FIELDS)
        writer.writeheader()
        writer.writerows(statuses.values())
    temporary.replace(REVIEW_STATUS_PATH)


def risk_explanation(signals: str) -> str:
    names = {name.strip() for name in signals.split("|") if name.strip()}
    explanations = []
    mapping = {
        "ROI": "The person interacted with the merchandise area.",
        "BAG NEARBY": "A backpack or handbag was near the person; this is context, not proof of concealment.",
        "SHELF CHANGED": "A localized product/table region changed from its baseline.",
        "SHELF CHANGE PERSISTED": "The table change remained visible instead of immediately recovering.",
        "SECOND PRODUCT CHANGED": "More than one product-sized region changed.",
        "EXITED WITH SHELF CHANGED": "The person left while the product state remained unresolved.",
        "PICKED UP": "A product-sized table change began and is being tracked provisionally.",
        "WITH PERSON": "The product remains unresolved during the interaction.",
        "TRACK LOST": "Tracking was lost without positive evidence of where the product went.",
        "VIDEO ENDED UNRESOLVED": "The clip ended before the product was returned or its destination was confirmed.",
        "CONCEALED": "The product was positively observed entering a pocket, bag, clothing, or body-associated region.",
        "UNRESOLVED EXIT": "The person left without the tracked product returning to the table.",
        "BAG NEARBY (CONTEXT ONLY)": "A bag was nearby, but proximity alone adds no risk.",
    }
    for name in sorted(names):
        explanations.append(mapping.get(name, f"Recorded evidence: {name.lower()}."))
    return " ".join(explanations) or "The saved event crossed the configured review threshold."


def threat_label(score: float) -> str:
    if score >= config.CONFIRMED_THRESHOLD:
        return "🔴 HIGHEST PRIORITY"
    if score >= config.HIGH_THRESHOLD:
        return "🔴 HIGH PRIORITY"
    if score >= config.CONCEALMENT_THRESHOLD:
        return "🔴 IMMEDIATE REVIEW"
    if score >= config.MODERATE_THRESHOLD:
        return "🟠 MANAGER REVIEW"
    if score >= config.AMBIGUOUS_THRESHOLD:
        return "🟡 AMBIGUOUS"
    if score >= config.TRACKING_THRESHOLD:
        return "🔵 TRACKING"
    return "🟢 NORMAL"


def collect_review_events() -> list[dict]:
    events = []
    for base in (config.RUNS_DIR, DASHBOARD_RUNS):
        if not base.exists():
            continue
        for manifest_path in base.glob("*/manifest.json"):
            try:
                manifest = json.loads(manifest_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            events_path = manifest_path.parent / "events.csv"
            if not events_path.exists():
                continue
            with events_path.open(newline="") as handle:
                for row in csv.DictReader(handle):
                    event_id = row.get("event_id", "EVENT")
                    image_path = row.get("image") or str(manifest_path.parent / "alerts" / f'{event_id}_person_{row.get("person_id", "unknown")}.jpg')
                    events.append({
                        **row,
                        "review_key": f'{manifest.get("run_name", manifest_path.parent.name)}::{event_id}',
                        "run_name": manifest.get("run_name", manifest_path.parent.name),
                        "input_video": manifest.get("input_video", ""),
                        "output_video": manifest.get("output_video", ""),
                        "expected_label": manifest.get("expected_label", "UNKNOWN"),
                        "system_outcome": manifest.get("system_outcome", "REVIEW"),
                        "image_path": image_path,
                        "manifest_mtime": manifest_path.stat().st_mtime,
                    })
    return sorted(events, key=lambda event: event["manifest_mtime"], reverse=True)


def render_review_queue() -> None:
    statuses = load_review_statuses()
    events = collect_review_events()
    st.header("Manager review queue")
    st.caption("Every alert remains a recommendation for human review. Manager decisions are saved locally and do not silently retrain the model.")
    if not events:
        st.info("No threshold-crossing events have been saved yet.")
        return
    pending = sum(statuses.get(event["review_key"], {}).get("status", "Pending") == "Pending" for event in events)
    queue_a, queue_b, queue_c = st.columns(3)
    queue_a.metric("Total events", len(events))
    queue_b.metric("Pending", pending)
    queue_c.metric("Reviewed", len(events) - pending)
    filter_left, filter_right = st.columns([1, 1])
    with filter_left:
        filter_status = st.selectbox("Show", ["All", "Pending", "Reviewed ✓", "Dismissed ✕", "Needs follow-up"])
    with filter_right:
        open_cards = st.toggle(
            "Show videos",
            value=True,
            help="Open review cards so their annotated videos are immediately visible.",
        )
    st.caption("Each review video is stored inside its event card. Turn off Show videos to collapse reviewed footage and make the page lighter.")
    for event in events:
        existing = statuses.get(event["review_key"], {})
        current_status = existing.get("status", "Pending")
        if filter_status != "All" and current_status != filter_status:
            continue
        score = float(event.get("risk_score") or 0)
        threat = threat_label(score)
        title = f'{threat} · Risk {score:.2f} · {current_status} · {event["run_name"]} · {event.get("event_id", "EVENT")}'
        with st.expander(title, expanded=open_cards or current_status == "Pending"):
            st.subheader(f"Threat level: {threat}")
            st.markdown(f'**Why it was flagged:** {risk_explanation(event.get("active_signals", ""))}')
            st.caption(
                f'Video time: {event.get("video_timestamp", "unknown")}s · '
                f'Person #{event.get("person_id", "unknown")} · '
                f'Expected label: {event.get("expected_label", "UNKNOWN")}'
            )
            media_left, media_right = st.columns([1.4, 1])
            output_video = Path(event["output_video"]) if event["output_video"] else None
            image_path = Path(event["image_path"]) if event["image_path"] else None
            with media_left:
                if output_video and output_video.exists():
                    st.video(str(web_video(output_video)))
                else:
                    st.warning("The saved annotated video is unavailable.")
            with media_right:
                if image_path and image_path.exists():
                    st.image(str(image_path), caption="Evidence frame")
            with st.form(f'review_form_{safe_name(event["review_key"])}'):
                choices = ["Pending", "Reviewed ✓", "Dismissed ✕", "Needs follow-up"]
                selected_status = st.selectbox("Manager decision", choices, index=choices.index(current_status) if current_status in choices else 0)
                manager_notes = st.text_area("Manager notes", value=existing.get("manager_notes", ""))
                if st.form_submit_button("Save review decision", type="primary"):
                    save_review_status(event["review_key"], selected_status, manager_notes)
                    st.success("Review decision saved.")


def web_video(path: Path) -> Path:
    """Return a browser-compatible H.264 copy when ffmpeg is available."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return path
    converted = path.with_name(path.stem + "_web.mp4")
    if converted.exists() and converted.stat().st_mtime >= path.stat().st_mtime:
        return converted
    result = subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(converted)],
        capture_output=True,
        text=True,
    )
    return converted if result.returncode == 0 and converted.exists() else path


def draw_key(frame) -> None:
    cv2.rectangle(frame, (8, 8), (258, 84), (20, 20, 20), -1)
    lines = ("NORMAL < .15", "TRACKING .15-.29", "AMBIGUOUS .30-.59", "REVIEW >= .60")
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (16, 25 + 17 * index), cv2.FONT_HERSHEY_SIMPLEX, .43, (240, 240, 240), 1)


def analyze(video: Path, expected: str, device_request: str, display_stride: int) -> dict:
    device = choose_device(device_request)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{safe_name(video.stem)}_{stamp}"
    run_dir = DASHBOARD_RUNS / run_name
    alerts_dir = run_dir / "alerts"
    alerts_dir.mkdir(parents=True, exist_ok=True)
    raw_output = run_dir / "annotated.mp4"
    events_path = run_dir / "events.csv"

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video.name}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create the annotated output video")

    model = YOLO(config.MODEL_NAME)
    bag_model = model  # reuse weights for per-frame bag context without another model instance
    pose_model = YOLO(config.POSE_MODEL_NAME)
    engine = RiskEngine()
    roi = tuple(int(value * (width if index % 2 == 0 else height)) for index, value in enumerate(config.MERCHANDISE_ROI))
    shelf_roi = tuple(int(value * (width if index % 2 == 0 else height)) for index, value in enumerate(config.PRODUCT_SHELF_ROI))
    shelf_monitor = ShelfMonitor(shelf_roi, fps)
    counts: Counter[str] = Counter()
    event_rows: list[dict] = []
    event_messages: list[str] = []
    last_roi_person: int | None = None
    last_table_wrist_contact: dict[int, float] = {}
    max_risk = 0.0
    processed = 0
    last_annotated = None
    started = time.perf_counter()

    progress = st.progress(0, text="Loading model and opening video…")
    live_frame = st.empty()
    metric_columns = st.columns(4)
    metric_frame, metric_people, metric_risk, metric_state = [column.empty() for column in metric_columns]
    live_events = st.empty()

    def record(changes, timestamp: float, frame) -> None:
        nonlocal max_risk
        for change in changes:
            state = engine.person_states.get(change.person_id)
            if state is None:
                continue
            max_risk = max(max_risk, state.risk_score)
            event_messages.append(f"{timestamp:6.2f}s  {change.message}")
            if change.crossed_review_threshold:
                event_id = f"EVT_{len(event_rows) + 1:04d}"
                image = alerts_dir / f"{event_id}_person_{state.person_id}.jpg"
                cv2.imwrite(str(image), frame)
                event_rows.append({
                    "event_id": event_id, "video_timestamp": round(timestamp, 3),
                    "person_id": state.person_id, "risk_score": round(state.risk_score, 2),
                    "alert_level": state.alert_level,
                    "active_signals": "|".join(sorted(state.active_signals)),
                    "image": str(image),
                })

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = processed / fps
            result = model.track(
                frame, persist=True, device=device, classes=[0, 24, 26],
                conf=config.CONFIDENCE_THRESHOLD, verbose=False
            )[0]
            detections = []
            if result.boxes is not None:
                ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [None] * len(result.boxes)
                for box, confidence, class_id, track_id in zip(
                    result.boxes.xyxy.int().cpu().tolist(), result.boxes.conf.cpu().tolist(),
                    result.boxes.cls.int().cpu().tolist(), ids,
                ):
                    name = model.names[class_id]
                    counts[name] += 1
                    detections.append((tuple(box), confidence, name, track_id))

            bags = []
            bags = [box for box, _, name, _ in detections if name in config.BAG_CLASS_NAMES]
            bag_result = bag_model.predict(
                frame, device=device, classes=[24, 26], conf=config.BAG_DETECTION_CONFIDENCE, verbose=False
            )[0]
            if bag_result.boxes is not None:
                for box, confidence, class_id in zip(
                    bag_result.boxes.xyxy.int().cpu().tolist(),
                    bag_result.boxes.conf.cpu().tolist(),
                    bag_result.boxes.cls.int().cpu().tolist(),
                ):
                    name = bag_model.names[class_id]
                    counts[name] += 1
                    bags.append(tuple(box))
                    detections.append((tuple(box), confidence, name, None))
            wrists = detected_wrists(frame, pose_model, device)

            annotated = frame.copy()
            cv2.rectangle(annotated, roi[:2], roi[2:], (0, 200, 255), 2)
            cv2.putText(annotated, "MERCHANDISE ZONE", (roi[0], max(24, roi[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 200, 255), 2)
            cv2.rectangle(annotated, shelf_roi[:2], shelf_roi[2:], (255, 180, 0), 2)
            cv2.putText(annotated, "PRODUCT / TABLE REGION", (shelf_roi[0], max(24, shelf_roi[1] - 7)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 180, 0), 2)
            inside_people: set[int] = set()
            table_to_bag_people: set[int] = set()
            person_boxes = [box for box, _, name, track_id in detections if name == "person" and track_id is not None]

            for box, confidence, name, track_id in detections:
                x1, y1, x2, y2 = box
                label = f"{name} {confidence:.2f}"
                color = (255, 160, 40)
                if name == "person" and track_id is not None:
                    color = (60, 220, 60)
                    inside = point_in_roi(box, roi)
                    if inside:
                        inside_people.add(track_id)
                        last_roi_person = track_id
                    person_wrists = [(wx, wy) for wx, wy, _ in wrists if point_in_box((wx, wy), box)]
                    for wrist in person_wrists:
                        containing_bag = next((bag for bag in bags if point_in_box(wrist, bag, padding=5)), None)
                        if point_in_box(wrist, shelf_roi) and containing_bag is None:
                            last_table_wrist_contact[track_id] = timestamp
                            cv2.circle(annotated, wrist, 5, (255, 255, 0), -1)
                        elif containing_bag is not None and timestamp - last_table_wrist_contact.get(track_id, -1e9) <= config.TABLE_TO_BAG_MAX_SECONDS:
                            table_to_bag_people.add(track_id)
                            cv2.circle(annotated, wrist, 7, (0, 0, 255), -1)
                            cv2.putText(annotated, "TABLE -> BAG HAND PATH", (x1, min(height - 10, y2 + 36)), cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 0, 255), 2)
                    changes = engine.observe(track_id, timestamp, inside, bag_is_near(box, bags))
                    record(changes, timestamp, annotated)
                    state = engine.person_states[track_id]
                    label = f"Person #{track_id} Risk {state.risk_score:.2f} {state.alert_level.split(' - ')[0]}"
                    signals = ", ".join(sorted(state.active_signals))
                    if signals:
                        cv2.putText(annotated, signals, (x1, min(height - 8, y2 + 18)), cv2.FONT_HERSHEY_SIMPLEX, .42, (255, 255, 255), 1)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .46, color, 2)

            shelf = shelf_monitor.update(frame, person_boxes, paused=False)
            for index, product_box in enumerate(shelf.region_boxes, start=1):
                cv2.rectangle(annotated, product_box[:2], product_box[2:], (0, 0, 255), 2)
                cv2.putText(annotated, f"PRODUCT CHANGE {index}", (product_box[0], max(18, product_box[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 0, 255), 2)
            if last_roi_person is not None:
                pose_bag_destination = last_roi_person in table_to_bag_people
                bag_destination = pose_bag_destination or (
                    shelf.persistent and bag_intersects_product_change(bags, shelf.region_boxes)
                )
                removal_evidence = shelf.product_removed or bag_destination
                changes = engine.apply_shelf_state(
                    last_roi_person, timestamp, removal_evidence, removal_evidence,
                    len(shelf.removed_region_boxes) or (1 if bag_destination else 0), not inside_people
                )
                if bag_destination:
                    changes += engine.mark_concealed(
                        last_roi_person, 1, timestamp,
                        "backpack opening after table-to-bag hand path" if pose_bag_destination else "backpack opening"
                    )
                record(changes, timestamp, annotated)
            max_risk = max([max_risk] + [state.risk_score for state in engine.person_states.values()])
            for state in engine.expire_missing(timestamp):
                event_messages.append(f"{timestamp:6.2f}s  Person {state.person_id} tracking finalized at risk {state.risk_score:.2f}")
            table_state = "PRODUCT REMOVED" if shelf.product_removed else ("REARRANGED" if shelf.changed else "BASELINE")
            cv2.putText(annotated, f"TABLE {table_state} diff {shelf.change_ratio:.3f} regions {shelf.estimated_regions}", (8, height - 12), cv2.FONT_HERSHEY_SIMPLEX, .43, (0, 220, 255), 1)
            draw_key(annotated)
            writer.write(annotated)
            last_annotated = annotated.copy()
            processed += 1

            if processed == 1 or processed % display_stride == 0 or processed == total:
                live_frame.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", caption="Live annotated inference frame")
                progress.progress(min(1.0, processed / max(1, total)), text=f"Analyzing frame {processed:,} of {total:,}")
                metric_frame.metric("Frame", f"{processed:,} / {total:,}")
                metric_people.metric("Active people", len(engine.person_states))
                metric_risk.metric("Maximum risk", f"{max_risk:.2f} / 1.00")
                metric_state.metric("Current outcome", engine.outcome_for(max_risk).replace("_", " "))
                live_events.code("\n".join(event_messages[-12:]) or "No risk events yet", language=None)
    finally:
        capture.release()
        writer.release()

    final_timestamp = processed / fps
    if last_annotated is not None:
        record(engine.finalize_video(final_timestamp), final_timestamp, last_annotated)
    final_risk = engine.current_risk()

    output = web_video(raw_output)
    outcome = engine.outcome_for(final_risk)
    expected_upper = expected.upper()
    evaluation = "NOT_SCORED" if expected_upper == "UNKNOWN" else (
        "PASS" if (
            (expected_upper == "SHOPLIFTING" and outcome == "REVIEW")
            or (expected_upper == "NORMAL" and outcome == "NO_REVIEW")
        ) else "FAIL"
    )
    elapsed = time.perf_counter() - started
    with events_path.open("w", newline="") as handle:
        writer_csv = csv.DictWriter(handle, fieldnames=["event_id", "video_timestamp", "person_id", "risk_score", "alert_level", "active_signals", "image"])
        writer_csv.writeheader()
        writer_csv.writerows(event_rows)
    manifest = {
        "run_name": run_name, "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_video": str(video), "expected_label": expected_upper, "system_outcome": outcome,
        "evaluation": evaluation, "max_risk": round(max_risk, 3), "frames": processed,
        "processing_seconds": round(elapsed, 2), "processing_fps": round(processed / max(elapsed, .001), 2),
        "detected_classes": dict(counts), "model": config.MODEL_NAME, "device": device,
        "output_video": str(output), "events_csv": str(events_path),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    progress.progress(1.0, text=f"Complete: {processed:,} frames analyzed")
    return manifest


st.set_page_config(page_title="Edge Vision Retail Monitor", page_icon="👁️", layout="wide")
st.title("Edge Vision AI Retail Monitor")
st.caption("Inspect live tracking evidence. This prototype recommends human review; it does not prove theft.")

workspace = st.sidebar.radio("Workspace", ["Review queue", "Analyze video"])
if workspace == "Review queue":
    render_review_queue()
    st.stop()

with st.sidebar:
    st.header("Video source")
    source = st.radio("Choose a source", ["Choose a file from this computer", *CURATED])
    selected_video: Path | None = None
    expected = "unknown"
    if source in CURATED:
        choices = available_videos(CURATED[source])
        if choices:
            selected_video = st.selectbox("Video", choices, format_func=lambda path: path.name)
            expected = "normal" if source == "Curated normal" else "shoplifting"
        else:
            st.error("No curated videos were found in this folder.")
    else:
        st.caption("Browse Documents, Downloads, Desktop, or any accessible folder on this computer.")
        upload = st.file_uploader(
            "Select a video file",
            type=["mp4", "mov", "avi"],
            help="Click Browse files to open the normal system file picker.",
        )
        expected = st.selectbox("Known label (optional)", ["unknown", "normal", "shoplifting"])
        if upload is not None:
            selected_video = save_upload(upload)
    st.header("Run settings")
    device_request = st.selectbox("Processing device", ["auto", "cpu", "mps"])
    display_stride = st.slider("Live preview every N frames", 1, 30, 10)

if selected_video is None:
    st.info("Choose a curated video or upload one to begin.")
    st.stop()

left, right = st.columns([1.35, 1])
with left:
    st.subheader("Selected video")
    preview_dir = config.OUTPUT_DIR / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_source = preview_dir / safe_name(selected_video.name)
    if not preview_source.exists() or preview_source.stat().st_mtime < selected_video.stat().st_mtime:
        shutil.copy2(selected_video, preview_source)
    st.video(str(web_video(preview_source)))
with right:
    st.subheader("What the run will show")
    st.markdown(
        "- Green boxes: tracked people with persistent IDs\n"
        "- Orange boxes: backpacks/handbags used only as optional context\n"
        "- Yellow region: person interaction zone\n"
        "- Blue region: class-agnostic table/shelf surveillance\n"
        "- Red boxes: changed product regions; background chair classes are ignored\n"
        "- Risk: progressive evidence from 0.00 to 1.00; dwell time is not used"
    )
    analyze_clicked = st.button("Analyze selected video", type="primary", use_container_width=True)

if analyze_clicked:
    try:
        st.session_state["last_manifest"] = analyze(selected_video, expected, device_request, display_stride)
    except Exception as exc:
        st.exception(exc)

manifest = st.session_state.get("last_manifest")
if manifest:
    st.divider()
    st.header("Latest result")
    result_a, result_b, result_c, result_d = st.columns(4)
    result_a.metric("System outcome", manifest["system_outcome"])
    result_b.metric("Maximum risk", f'{manifest["max_risk"]:.2f} / 1.00')
    result_c.metric("Expected-label check", manifest["evaluation"])
    result_d.metric("Processing speed", f'{manifest["processing_fps"]:.1f} FPS')
    st.video(manifest["output_video"])
    with st.expander("Detected classes and saved artifacts", expanded=True):
        st.json({"detected_classes": manifest["detected_classes"], "run_folder": str(Path(manifest["output_video"]).parent), "events_csv": manifest["events_csv"]})
    st.warning("A PASS only means the current review threshold matched the folder label on this clip. It is not yet an accuracy claim; evaluation requires the full curated set and reliable ground truth.")

    with st.form("operator_feedback", clear_on_submit=True):
        st.subheader("Operator feedback")
        operator_label = st.radio("Was the system result useful?", ["Correct", "Incorrect", "Uncertain"], horizontal=True)
        notes = st.text_area("Notes", placeholder="What did the tracker or shelf logic get right or wrong?")
        if st.form_submit_button("Save feedback"):
            feedback_path = DASHBOARD_RUNS / "operator_feedback.csv"
            new_file = not feedback_path.exists()
            with feedback_path.open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=FEEDBACK_FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow({"timestamp": datetime.now(timezone.utc).isoformat(), "run_name": manifest["run_name"], "video": manifest["input_video"], "operator_label": operator_label, "notes": notes})
            st.success(f"Feedback saved to {feedback_path}")
