# Video experiment splits

The original videos remain in their existing dataset folders. These manifests assign each curated video one experimental role without duplicating or deleting media.

- `development.csv`: may be watched repeatedly and used to change thresholds, ROIs, rules, or train a future model.
- `validation.csv`: may be run occasionally to select among approaches, but should not drive frame-by-frame tuning.
- `holdout_locked.csv`: final evaluation only. Do not inspect results or tune against these clips until a version is frozen.
- `excluded.csv`: quarantined unusable, ambiguous, corrupt, truncated, or incorrectly labeled videos. Never train or score on these.
- `future_intake.csv`: genuinely new videos added after the current algorithm version. Review and label them before assigning them to another split.

Splits are made by whole video. Frames from one video must never appear in different splits because adjacent frames are nearly duplicates and would leak information into evaluation.

The current split is balanced by top-level label: 14 development videos, 4 validation videos, and 4 locked holdout videos. `Normal (10).mp4` remains in development as the `normal_restock` stress test because its behavior has already influenced the logic discussion.

When a custom learned model is introduced, development data may be subdivided into model-training and model-validation sets. The locked holdout remains untouched and is evaluated once after the code, model, and thresholds are frozen.
