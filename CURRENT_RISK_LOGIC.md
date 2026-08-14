# Current risk-analysis logic

This score prioritizes observable events for human review. It is not a probability of shoplifting and does not classify a person as a thief.

## Perception and state

1. YOLO26 nano detects people and available COCO objects.
2. Ultralytics tracking assigns persistent person IDs.
3. Each person has independent temporal state: first/last seen, ROI status, active signals, reversible shelf contributions, score, and review level.
4. Missing tracks remain active for a two-second grace period before finalization.
5. At most 50 simultaneous person states are retained by default; this is configurable.

## Current signals

The pretrained tracker is deliberately restricted to the `person` class. Chairs and other background labels are ignored. Merchandise is handled separately with class-agnostic change detection inside the configured product/table region, so it need not belong to a COCO object class.

## Product-flow policy

Risk is maintained independently for each provisional product and the person's displayed risk is the highest unresolved product risk. Time spent merely standing in the area and bag proximity add no risk.

- pickup/tracking: `0.15`
- unresolved or lost tracking: `0.30` (`AMBIGUOUS REVIEW`)
- manager review: `0.60`
- positively observed concealment: `0.75`
- exit with an unresolved product: `0.90`
- concealment plus unresolved exit: `1.00`
- returned or pure restocking: product risk `0.00`

Holding duration adds `0.05` at 10, 20, and 30 seconds, capped at `0.30`. A short clip therefore receives no holding-duration increase. Video end with an unresolved product becomes ambiguous rather than confirmed theft. Track loss is not concealment: concealment requires positive destination evidence.

## Manager-review requirements discovered

- Review explanations must identify which provisional product triggered the event and describe its flow, not only say that the table changed.
- Moving a product to another location on the same merchandise table is a relocation and should not be treated as theft.
- A product trajectory clearly directed into a pocket/body region is positive concealment evidence even if part of the product remains momentarily visible.
- Manager decisions are stored in `output/review_status.csv` for evaluation; they do not automatically modify or retrain the model.

Localized table changes are direction-filtered before entering product flow. A texture disappearance paired with a texture appearance elsewhere inside the same table ROI is treated as relocation. Only unmatched persistent disappearance regions are provisional removals.

For bag concealment, bag presence or proximity alone remains zero-risk context. A persistent product-region change must spatially converge with the detected backpack/handbag opening before the product can transition to `CONCEALED` at `0.75`. Confirmed concealment is not cleared merely because table pixels later stabilize.

The pipeline also uses a lightweight pose model for destination trajectories. A confident wrist must contact the merchandise table outside the bag and subsequently enter a detected bag opening within four seconds. This table-to-bag hand path can create positive concealment evidence when the product is too small or unknown for the general object detector. Wrist or bag presence by itself remains zero risk.

Review-event metadata and a screenshot are created only when a person first crosses into a review level.

## What the current logic does not yet measure

- Guaranteed identity for a specific generic product leaving its original position
- Semantic knowledge of what each generic product is
- Whether two products were removed and only one returned
- Reliable product-to-pocket or product-to-backpack transfer
- Shelf inventory state after the person leaves

Consequently, current scores reflect interaction context, not the curated dataset's decisive returned-versus-not-returned distinction.

## Implemented shelf-state logic and recommended next logic

An experimental class-agnostic fixed-camera shelf-state layer detects persistent visual changes and reverses its contributions if the shelf recovers. It is configured conservatively because validation showed that aggressive whole-ROI differencing creates Normal false positives, while conservative settings can miss small removed products. It is supporting evidence, not yet the primary classifier. The next refinement is explicit product identity:

1. Establish initial product regions and assign generic product IDs.
2. Detect a product leaving its original region.
3. Associate its movement with the nearest persistent person track.
4. Maintain an occlusion grace period while the person blocks the table.
5. Clear the signal if the product reappears near its original region.
6. Escalate only if the person leaves and the product remains absent.

Suggested product-state contributions:

| Observable product event | Proposed change |
|---|---:|
| Product removed from original region | +0.20 |
| Product remains absent beyond grace period | +0.25 |
| Person exits while product remains absent | +0.35 |
| A second product remains absent | +0.20 |
| Product returns to original region | Remove that product's active contributions |

ROI entry and bag proximity remain lower-weight context. Time spent is deliberately not a risk factor. Product non-return should drive the review decision.
