# Project change history

## 2026-08-14

- Created the Python 3.12 environment and installed pinned vision dependencies.
- Downloaded pretrained `yolo26n.pt`; no training or fine-tuning was performed.
- Added detection, tracking, ROI, temporal state, rule scoring, alerts, feedback, and benchmarking.
- Validated detection, tracking, and risk outputs on staged MNNIT clip 15.
- Changed ROI membership from bottom-center to box-center after visual validation.
- Compared staged shoplifting clips 1 and 22 for backpack and merchandise detections.
- Added configurable active-person capacity and named per-video result storage.
- Separated per-frame backpack/handbag detection from persistent person tracking; clip 22 now produces a real BAG NEARBY event while clip 1 does not.
- Added a local HTML results page with playable processed videos and alert screenshots.
- Corrected bag scoring so continuous proximity contributes once per episode rather than once per cooldown interval.
- Curated 11 GOOD_NORMAL and 11 GOOD_SHOPLIFTING clips using visible return versus non-return standards without deleting originals.
- Added `data/curation_manifest.csv` and documented current versus proposed product-state risk logic.
- Removed dwell-time risk contributions entirely.
- Added a fixed-camera class-agnostic shelf monitor with occlusion masking, persistence debounce, multiple-region evidence, exit-with-change escalation, and reversible recovery contributions.
- Raised bag-context confidence and paused shelf comparisons while a person occupies the merchandise zone after a curated Normal clip exposed false positives.
- Split the person interaction ROI from a wider product-shelf ROI after validation showed the original rectangle omitted much of the table inventory.
- Reverted aggressive shelf thresholds after they produced a false review on curated Normal 35; conservative shelf comparison remains experimental and fails closed on uncertain small-object changes.
- Made system behavior the priority; device comparison remains a later step.
- Restricted pretrained tracking to people and bag context so chairs/background classes are ignored.
- Replaced whole-table pausing with per-person occlusion masking, added visible localized `PRODUCT CHANGE` boxes, and lowered the candidate-region size for small packages.
- Re-tested Shoplifting 1: the previously missed red-package removal now reaches REVIEW (0.75), while Normal 10 also exposes a false positive because that clip begins with the product already in the shopper's hand rather than on the table.
- Relabeled Normal 10 as the `NORMAL_RESTOCK` stress-test scenario instead of removing it.
- Added optional `main.py --show` live OpenCV visualization with `q` to stop while retaining normal annotated-video saving.
- Added balanced whole-video development, validation, and locked-holdout manifests plus empty excluded and future-intake queues; originals remain untouched.
- Added a persistent manager review queue to the interactive dashboard with video/evidence playback, plain-language risk explanations, and reviewed/dismissed/follow-up decisions.
- Replaced additive person-level shelf scoring with per-product flow states and the approved 0.15/0.30/0.60/0.75/0.90/1.00 vocabulary; area dwell and bag proximity no longer add risk, returns reset only the relevant product, and unresolved video endings become ambiguous reviews.
- Added prominent color-coded threat labels to manager review titles and recorded the manager's first four review observations locally.
- Made the manager review queue the default dashboard workspace and promoted the local computer file picker to the first Analyze Video source.
- Used dismissed Normal 35 feedback to replace generic table-change risk with unmatched-removal evidence; balanced source/destination changes inside the table are treated as relocation. Current Normal 35 now passes at risk 0.00 while Shoplifting 1 retains unresolved-removal evidence.
- Connected persistent product-region changes to detected bag openings as positive destination evidence; bag proximity alone remains zero risk and confirmed concealment persists at 0.75.
- Added pose-based table-to-bag wrist trajectories for small unknown products: table contact followed by wrist entry into a detected bag within four seconds triggers immediate concealment review.
- Preserved terminal concealment risk after person-track expiration and recorded Shoplifting 22 as manager-confirmed theft rather than follow-up.
