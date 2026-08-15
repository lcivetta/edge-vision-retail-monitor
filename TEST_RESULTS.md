# Test results

## Recruiter-facing holdout evaluation: v1

- **Run date:** August 14, 2026
- **Frozen code:** Git commit `1a0d128`
- **Model:** pretrained YOLO26 nano plus YOLO11 nano pose; no model fine-tuning
- **Device:** Apple M3 CPU
**Input:** four previously locked videos from the cited MNNIT dataset

The decision unit is one complete video. `REVIEW` is treated as the positive class and `NO_REVIEW` as the negative class. A result passes when the system's review/no-review decision matches the dataset label.

| Video | Expected behavior | System result | Maximum risk | Decisive evidence | Result |
|---|---|---|---:|---|---|
| Normal (20) | Product returned | No review | 0.00 | No review-level evidence | Pass |
| Normal (38) | Product returned | No review | 0.00 | No review-level evidence | Pass |
| Shoplifting (8) | Product not returned / placed in bag | Immediate review | 0.75 | Table-to-bag wrist trajectory with detected bag | Pass |
| Shoplifting (9) | Product not returned / placed in bag | Immediate review | 0.75 | Table-to-bag wrist trajectory with detected bag | Pass |

## Results

- Video-level accuracy: **4/4 (100%)**
- Shoplifting recall: **2/2 (100%)**
- Normal-video specificity: **2/2 (100%)**
- False positives: **0**
- False negatives: **0**
- Total evaluated frames: **1,335** (44.5 seconds of source video at 30 FPS)
- End-to-end processing: **10.16 FPS aggregate throughput** on Apple M3 CPU, including detection, pose, tracking, risk logic, overlays, and video writing
- Software regression suite: **10/10 tests passing**

Raw, non-media results are available in [`evaluation/holdout_v1_results.csv`](evaluation/holdout_v1_results.csv). Videos, model weights, manager records, and generated surveillance artifacts remain excluded from GitHub.

## What occurred

The two normal-return clips never crossed a review threshold and ended at risk `0.00`. In both shoplifting clips, the pose layer observed a wrist path from the merchandise table into a detected bag region. That positive destination evidence triggered `CONCEALED`, risk `0.75`, and an immediate manager-review event. Bag presence alone contributed no risk.

## Interpretation and limitations

This is a successful **small holdout smoke evaluation**, not a claim of production-ready 100% accuracy. Four staged clips cannot establish generalization, and the single camera angle, manually configured merchandise region, curated labels, and controlled dataset differ from a live store.

The correct video-level decisions also do not prove that every intermediate product-state transition was correct. In the shoplifting runs, later shelf changes were interpreted as product returns even though the earlier concealment event remained preserved. Temporal alert localization and product identity therefore remain important failure-analysis targets. A credible next benchmark needs more unseen videos, multiple camera angles and stores, item-level annotations, confidence intervals, and comparisons against learned temporal baselines.

## Reproduction

Download the dataset from the source linked in the README, preserve the relative paths in `data/splits/holdout_locked.csv`, install the pinned requirements, and run each clip with:

```bash
python main.py --device cpu --mode risk \
  --input "path/to/video.mp4" \
  --run-name your_run_name \
  --expected-label normal-or-shoplifting
```

The run writes a manifest, event CSV, evidence image, and annotated video under `output/runs/<run-name>/`. Those local artifacts support audit and manager review but are intentionally not committed.
