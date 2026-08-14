"""All user-tunable settings for the retail monitor MVP."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
INPUT_VIDEO = ROOT / "data" / "input.mp4"
MODEL_NAME = "yolo26n.pt"
POSE_MODEL_NAME = "yolo11n-pose.pt"

# Normalized (x1, y1, x2, y2). Tuned for the table in the selected MNNIT clip.
MERCHANDISE_ROI = (0.28, 0.28, 0.66, 0.96)
# Wider table/display region used only for class-agnostic shelf comparison.
PRODUCT_SHELF_ROI = (0.40, 0.25, 0.97, 0.95)
PERSON_CLASS_ID = 0
BAG_CLASS_NAMES = {"backpack", "handbag"}
CONFIDENCE_THRESHOLD = 0.25
BAG_DETECTION_CONFIDENCE = 0.30

PERSON_GRACE_SECONDS = 2.0
MAX_ACTIVE_PERSON_STATES = 50
RUN_RETENTION_DAYS = 3
BAG_PROXIMITY_DISTANCE = 0.75  # max center distance / person-box diagonal
BAG_SIGNAL_COOLDOWN_SECONDS = 5.0

# Fixed-camera, class-agnostic shelf-state evidence. These values are intentionally
# configurable because camera placement and lighting affect pixel differences.
SHELF_CHANGE_RATIO = 0.035
SHELF_RECOVERY_RATIO = 0.015
SHELF_PERSISTENCE_SECONDS = 1.5
# Small packages in the MNNIT table clips occupy only tens of stable changed
# pixels after thresholding. Persistence, not a large global area, rejects
# momentary hand/lighting noise.
SHELF_MIN_COMPONENT_AREA = 60
# Negative local texture change is evidence that a textured product was
# removed and exposed a simpler table surface. Smaller changes are treated as
# relocation/occlusion rather than product removal.
PRODUCT_REMOVAL_TEXTURE_DELTA = -8.0
PRODUCT_ADDITION_TEXTURE_DELTA = 8.0
# Approved product-flow vocabulary. Area dwell and bag presence add no risk.
TRACKING_THRESHOLD = 0.15
AMBIGUOUS_THRESHOLD = 0.30
MODERATE_THRESHOLD = 0.60
CONCEALMENT_THRESHOLD = 0.75
HIGH_THRESHOLD = 0.90
CONFIRMED_THRESHOLD = 1.00
HOLDING_MILESTONES_SECONDS = (10, 20, 30)
HOLDING_RISK_STEP = 0.05
HOLDING_RISK_CAP = 0.30
MAX_SIMULTANEOUS_PRODUCT_REGIONS = 3
WRIST_CONFIDENCE_THRESHOLD = 0.50
TABLE_TO_BAG_MAX_SECONDS = 4.0

OUTPUT_DIR = ROOT / "output"
PROCESSED_DIR = OUTPUT_DIR / "processed"
ALERTS_DIR = OUTPUT_DIR / "alerts"
EVENTS_CSV = OUTPUT_DIR / "events.csv"
FEEDBACK_CSV = ROOT / "feedback" / "feedback.csv"
BENCHMARK_CSV = OUTPUT_DIR / "benchmark_results.csv"
RUNS_DIR = OUTPUT_DIR / "runs"
RUN_RESULTS_CSV = OUTPUT_DIR / "run_results.csv"


def ensure_directories() -> None:
    for directory in (INPUT_VIDEO.parent, PROCESSED_DIR, ALERTS_DIR, FEEDBACK_CSV.parent, RUNS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
