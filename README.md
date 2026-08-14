# Edge Vision AI Retail Monitor

An explainable, human-in-the-loop Vision AI prototype for surfacing retail video events that deserve manager review. Built with Python, Ultralytics YOLO, OpenCV, and Streamlit.

## Interactive dashboard

Run `source .venv/bin/activate && streamlit run dashboard.py`, then open `http://localhost:8501`. The dashboard lets you select curated clips or upload a video, watch annotated inference update during processing, inspect the risk-event stream, replay the saved result, and record operator feedback. See `DEPLOYMENT.md` for remote-hosting requirements.

The dashboard opens on the persistent **Review queue** by default. Each threshold-crossing event shows a prominent color-coded threat level in its title, the annotated video, evidence frame, risk score, and plain-language reasons. A manager can mark it `Reviewed ✓`, `Dismissed ✕`, or `Needs follow-up`; decisions are stored locally in `output/review_status.csv`. These decisions are project feedback and do not automatically retrain or transmit data.

Switch the sidebar workspace to **Analyze video** to process another clip. **Choose a file from this computer** opens the normal system file picker for Documents, Downloads, Desktop, or another folder; curated datasets remain available as secondary sources.

The current product-flow policy has regression tests in `tests/test_product_flow.py`. Run them with `python -m unittest discover -s tests`.

The dashboard is optional. For the most direct visualization of the real pipeline, run:

```bash
python main.py --input "/path/to/video.mp4" --run-name visual_check --show
```

This opens an OpenCV window containing the same annotated frames written to the output video. Press `q` to stop; the processed portion is still finalized and saved. Without `--show`, processing remains headless.

## Dataset roles

Curated videos are assigned by whole video in `data/splits/`: development for iteration, validation for occasional checks, a locked holdout for final evaluation, excluded/quarantine for unusable clips, and future intake for genuinely new data. Original media is not moved or duplicated. See `data/splits/README.md` before changing the algorithm or evaluating accuracy.

## Problem and goal

Camera operators cannot continuously inspect every person and event in every feed. This understandable local prototype uses observable video events to surface moments that may deserve human review. It never determines that a person is a thief, shoplifter, criminal, or dangerous.

```text
Video -> Pretrained YOLO detector -> Multi-object tracker
      -> Persistent person state -> Temporal event rules
      -> Rule-based review score -> Review event -> Human feedback
```

## Dataset and model

This project uses **Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad**, created by Mohd. Aquib Ansari and Dushyant Kumar Singh and published through Mendeley Data under the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/):

> Ansari, M. A., & Singh, D. K. (2023). *Shoplifting Dataset (2022) - CV Laboratory MNNIT Allahabad* (Version 1). Mendeley Data. https://doi.org/10.17632/r3yjf35hzr.1

Download the footage from the [official dataset page](https://data.mendeley.com/datasets/r3yjf35hzr/1). Dataset videos are intentionally not committed to this repository. The CSV manifests in `data/` contain project-specific curation notes and split assignments that reference the original filenames; they do not contain the media itself. The footage is controlled research data, not video from an operating store and not evidence of production performance.

The perception model is pretrained YOLO26 nano (`yolo26n.pt`). No model was trained or fine-tuned for this MVP. Consequently, only COCO categories the general model can actually recognize are available; small merchandise may not be recognized.

## What was built

- OpenCV video decoding/encoding with approximate source resolution and FPS preserved
- Ultralytics detection and persistent multi-object tracking
- Normalized, configurable merchandise ROI using each person's bounding-box center
- Per-person temporal state with a two-second missed-detection grace period
- Debounced ROI-entry, contextual bag-proximity, and reversible shelf-state rules
- Configurable rule weights and `0.0`-to-`1.0` review score
- Status overlays, event CSV rows, alert screenshots, and operator feedback rows
- Repeatable CPU/MPS inference benchmark

All settings are in `config.py`. A bag is only context and is not suspicious by itself.

The default cap is 50 simultaneously active person states because these clips contain only a few people. `MAX_ACTIVE_PERSON_STATES` is editable for measured store loads, including 1,000. Inactive people are removed after the grace period; tracker IDs are not reset at 1,000 because ID reuse could corrupt event history. Named run artifacts are intended for a configurable three-day retention period, but automatic deletion is not enabled until a real storage policy is chosen.

## Setup and use

```bash
/opt/homebrew/bin/python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py --device auto --mode risk
python main.py --device cpu --mode risk
python main.py --device mps --mode risk
python benchmark.py --frames 300 --trials 3
python feedback/review_event.py EVT_0001
```

For incremental inspection, use `--mode detection`, then `--mode tracking`, then `--mode risk`. Download the cited dataset or use your own authorized footage, then either place a selected clip at `data/input.mp4` or pass its path with `--input`.

For a stored, auditable run:

```bash
python main.py --device cpu --mode risk \
  --input "path/to/video.mp4" \
  --run-name shoplifting_22 --expected-label shoplifting
python report.py
```

Named results live in `output/runs/<run-name>/`. Each contains the processed video, event and feedback CSVs, alert screenshots, and a JSON manifest naming the exact source video, model, detections, score, outcome, and evaluation. `output/run_results.csv` indexes all runs, `output/report.html` is the local interactive results page, and `CHANGELOG.md` records code and validation history.

## Current risk engine

The engine is rule-based and heuristic, not a learned risk model. ROI entry adds 0.10 and a nearby detected backpack/handbag adds 0.10 once per episode. A fixed-camera shelf monitor adds reversible evidence for a changed shelf, persistence, multiple changed product regions, and a person exiting while the shelf remains changed. Time spent in the area does not increase risk. Scores map to NORMAL, MONITOR, MODERATE CAUTION - REVIEW EVENT, and HIGH CAUTION - REVIEW EVENT.

## Human in the loop

CURRENT: pretrained visual perception plus configurable rules creates review events. Operators label those events in `feedback/feedback.csv`; feedback does not automatically train or change the system.

FUTURE: aggregate operator labels could support false-positive analysis, manual rule tuning, or training and evaluating a learned temporal event classifier before redeployment.

## Edge AI and future deployment

Local inference can reduce dependence on continuous cloud connectivity, video transmission, bandwidth, and response latency, subject to the actual deployment. This version runs on Mac CPU or Apple MPS when PyTorch reports compatibility. It does not use CUDA, an NVIDIA GPU, TensorRT, DeepStream, or Jetson.

A future NVIDIA path could preserve the product architecture while moving inference to an NVIDIA GPU or Jetson, then evaluating TensorRT optimization and a DeepStream pipeline. A future VLM should inspect only selected alert clips after the inexpensive detector/tracker/rule stage, not every frame.

## Later: device benchmark

Run `benchmark.py` to write actual measurements to `output/benchmark_results.csv`. It uses the same model, frames, confidence, and trials on each available device, excludes output-video writing, and warms up first. An unavailable or failed MPS path is recorded explicitly rather than replaced with invented numbers.

On this Apple M3 environment, three 300-frame CPU trials measured 34.971, 35.775, and 36.313 FPS (35.686 FPS mean). PyTorch reported `torch.backends.mps.is_available() == False`, so no MPS result or speedup is claimed.

## Limitations and future work

- Staged dataset and one manually configured ROI; not production ready
- General pretrained categories miss many merchandise objects; tracking can fail under occlusion
- Rule weights are heuristics, not learned probabilities or calibrated measures of intent
- Object-to-bag interaction, disappearance, and literal holding are deliberately not claimed or implemented without reliable object detections
- No trained behavior classifier, production camera ingestion, multi-camera identity, or automated feedback learning

Logical future work includes a merchandise-specific detector, stronger tracking/pose or temporal modeling, feedback-driven evaluation, multiple-camera support, selected-clip VLM reasoning, and measured NVIDIA/Jetson deployment experiments.

## License

Project code is available under the MIT License. The cited MNNIT dataset is a separate work distributed under CC BY 4.0; see its official source and attribution above.
