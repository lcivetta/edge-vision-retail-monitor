# Product vision and future markets

## The customer problem

Small-business owners and store employees often have to serve customers, manage inventory, operate checkout, and monitor the store at the same time. Continuous attention to multiple camera feeds is unrealistic and creates stress without guaranteeing that important events will be noticed.

The product vision is an edge Vision AI assistant that converts long video streams into a short, explainable review queue. It should reduce monitoring burden and help staff understand what occurred while leaving intervention decisions to a person. It is not intended to label people as criminals or replace store policy, employee judgment, or due process.

## Market progression

### 1. Independent and tourist retail

The first product tier could serve convenience stores, gift shops, tourist shops, pharmacies, and other small retailers. A low-cost edge device would connect to one or a few existing cameras, identify merchandise interactions, and present only review-worthy clips. Simple installation, low ongoing cost, clear explanations, and minimal video retention matter more here than a complex command center.

### 2. Growing and multi-zone stores

A second tier could support several departments or cameras, shared review across a shift, configurable merchandise regions, basic inventory or point-of-sale integration, and analytics about system performance. The objective remains operational clarity rather than automatic accusation.

### 3. Large retail fleets

Large apparel or general-retail chains could use the architecture across many stores and concurrent video streams. Enterprise requirements would include centralized configuration, role-based access, audit trails, health monitoring, APIs, multi-camera event association, regional data controls, and hardware-aware scheduling across edge GPUs. A retailer such as ZARA illustrates the scale of this potential market, not a current customer or deployment claim.

## Multimodal sensor roadmap

Video-only product tracking is fragile when merchandise is small, visually similar, blocked by a shopper, or moved outside the camera view. Additional sensors can provide independent evidence:

| Signal | Potential contribution | Important limitation |
|---|---|---|
| Shelf weight or pressure | Confirms that physical inventory was removed or returned | Requires shelf installation and calibration |
| RFID, NFC, or tagged inventory | Identifies the specific product and its movement | Tag and reader cost; incomplete coverage |
| Infrared or low-light imaging | Improves visibility in poor lighting | Does not identify intent and may require new cameras |
| Depth sensing | Separates hands, products, shelves, and bodies during occlusion | Higher hardware and compute cost |
| Point-of-sale events | Distinguishes unresolved possession from completed purchase | Timing and item-matching integration are required |
| Door or exit sensors | Adds evidence that an unresolved interaction reached an exit | Exit alone must never imply theft |

The strongest future design is evidence fusion: video proposes an event, a shelf sensor confirms inventory change, item identity comes from a tag or product model, and checkout data resolves whether the item was purchased. Disagreement should lower confidence or send the event for human review instead of forcing a conclusion.

## Faster and more accurate response

The current prototype processes video locally but is not yet optimized for real-time deployment. A production path could use NVIDIA Jetson or an NVIDIA GPU, TensorRT optimization, and DeepStream for multi-stream decoding, batching, tracking, and message transport. A lightweight detector should handle continuous processing, while more expensive pose, temporal, or vision-language reasoning runs only on selected event windows.

Accuracy improvements should focus on merchandise-specific detection, persistent product identity, occlusion recovery, multi-camera association, checkout-aware state transitions, and evaluation on new stores and camera angles. Faster response is useful only when alerts remain explainable and false alarms are controlled.

## Integration model

A scalable implementation could expose event APIs instead of forcing retailers to replace existing systems. Possible integrations include camera-management software, point-of-sale platforms, inventory tools, mobile manager notifications, case-management systems, and enterprise analytics. Store configuration—camera geometry, merchandise regions, retention, review thresholds, and active sensors—should be managed separately from the core inference engine.

## Success measures

A production pilot should measure more than headline accuracy:

- event-level recall and false alerts per camera-hour
- time from observable event to manager notification
- percentage of alerts a manager finds useful
- reduction in hours spent watching uneventful footage
- performance across lighting, layouts, products, occlusion, and customer demographics
- edge throughput, power consumption, storage use, and uptime
- privacy, retention, access-control, and audit compliance

The desired outcome is less worry and less wasted attention for store staff, paired with a review process that is faster, more consistent, and still accountable to a human.
