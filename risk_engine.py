"""Per-product flow states and explainable review-risk progression."""

from __future__ import annotations

from dataclasses import dataclass, field

import config


@dataclass
class ProductState:
    product_id: int
    state: str
    first_observed: float
    last_observed: float
    risk_score: float = 0.15
    holding_milestones: set[int] = field(default_factory=set)
    concealed: bool = False


@dataclass
class PersonState:
    person_id: int
    first_seen: float
    last_seen: float
    risk_score: float = 0.0
    inside_roi: bool = False
    roi_entry_time: float | None = None
    active_signals: set[str] = field(default_factory=set)
    products: dict[int, ProductState] = field(default_factory=dict)
    bag_nearby: bool = False
    alert_level: str = "NORMAL"

    @property
    def time_in_roi(self) -> float:
        if not self.inside_roi or self.roi_entry_time is None:
            return 0.0
        return max(0.0, self.last_seen - self.roi_entry_time)


@dataclass
class RiskChange:
    person_id: int
    message: str
    old_level: str
    new_level: str
    crossed_review_threshold: bool


class RiskEngine:
    """Maintain independent product histories and derive one person-level risk."""

    def __init__(self) -> None:
        self.person_states: dict[int, PersonState] = {}
        self.finalized_risk = 0.0

    @staticmethod
    def level_for(score: float) -> str:
        if score >= config.CONFIRMED_THRESHOLD:
            return "HIGHEST-PRIORITY REVIEW"
        if score >= config.HIGH_THRESHOLD:
            return "HIGH-PRIORITY REVIEW"
        if score >= config.CONCEALMENT_THRESHOLD:
            return "IMMEDIATE REVIEW"
        if score >= config.MODERATE_THRESHOLD:
            return "MANAGER REVIEW"
        if score >= config.AMBIGUOUS_THRESHOLD:
            return "AMBIGUOUS REVIEW"
        if score >= config.TRACKING_THRESHOLD:
            return "TRACKING PRODUCT"
        return "NORMAL"

    @staticmethod
    def outcome_for(score: float) -> str:
        if score >= config.MODERATE_THRESHOLD:
            return "REVIEW"
        if score >= config.AMBIGUOUS_THRESHOLD:
            return "AMBIGUOUS_REVIEW"
        return "NO_REVIEW"

    def observe(self, person_id: int, timestamp: float, inside_roi: bool, bag_nearby: bool) -> list[RiskChange]:
        state = self.person_states.get(person_id)
        if state is None:
            if len(self.person_states) >= config.MAX_ACTIVE_PERSON_STATES:
                oldest_id = min(self.person_states, key=lambda pid: self.person_states[pid].last_seen)
                del self.person_states[oldest_id]
            state = PersonState(person_id, timestamp, timestamp)
            self.person_states[person_id] = state
        state.last_seen = timestamp
        state.bag_nearby = bag_nearby
        if inside_roi and not state.inside_roi:
            state.inside_roi = True
            state.roi_entry_time = timestamp
        elif not inside_roi and state.inside_roi:
            state.inside_roi = False
            state.roi_entry_time = None
        # Time spent in the area and merely having a bag never add risk.
        return []

    def apply_shelf_state(
        self,
        person_id: int,
        timestamp: float,
        changed: bool,
        persistent: bool,
        estimated_regions: int,
        person_exited: bool,
    ) -> list[RiskChange]:
        state = self.person_states.get(person_id)
        if state is None:
            return []
        changes: list[RiskChange] = []
        if not changed:
            for product in state.products.values():
                if product.state in {"PICKED_UP", "WITH_PERSON", "TRACK_LOST", "VIDEO_ENDED_UNRESOLVED"}:
                    changes += self.mark_returned(person_id, product.product_id, timestamp)
            return changes

        # Changed regions are provisional products. Region tracking is
        # intentionally capped until stable region IDs replace component order.
        product_count = max(1, min(config.MAX_SIMULTANEOUS_PRODUCT_REGIONS, estimated_regions))
        for product_id in range(1, product_count + 1):
            product = state.products.get(product_id)
            if product is None:
                product = ProductState(product_id, "PICKED_UP", timestamp, timestamp)
                state.products[product_id] = product
                changes.append(self._recalculate(state, f"Product {product_id} picked up; tracking started"))
            else:
                product.last_observed = timestamp
            if persistent and product.state not in {"RETURNED", "RESTOCKED"}:
                product.state = "WITH_PERSON"
                held_for = timestamp - product.first_observed
                for seconds in config.HOLDING_MILESTONES_SECONDS:
                    if held_for >= seconds and seconds not in product.holding_milestones:
                        product.holding_milestones.add(seconds)
                        product.risk_score = min(config.HOLDING_RISK_CAP, product.risk_score + config.HOLDING_RISK_STEP)
                        changes.append(self._recalculate(state, f"Product {product_id} held for {seconds} seconds"))
            if person_exited and product.state not in {"RETURNED", "RESTOCKED"}:
                product.state = "UNRESOLVED_EXIT"
                product.risk_score = 1.0 if product.concealed else config.HIGH_THRESHOLD
                changes.append(self._recalculate(state, f"Product {product_id} was not returned before person exit"))
        return changes

    def mark_returned(self, person_id: int, product_id: int, timestamp: float) -> list[RiskChange]:
        state = self.person_states.get(person_id)
        product = state.products.get(product_id) if state else None
        if state is None or product is None or product.state == "RETURNED":
            return []
        product.state = "RETURNED"
        product.last_observed = timestamp
        product.risk_score = 0.0
        return [self._recalculate(state, f"Product {product_id} returned; product risk reset")]

    def mark_restocked(self, person_id: int, product_id: int, timestamp: float) -> list[RiskChange]:
        state = self.person_states.get(person_id)
        if state is None:
            return []
        product = state.products.get(product_id) or ProductState(product_id, "RESTOCKED", timestamp, timestamp, 0.0)
        state.products[product_id] = product
        product.state = "RESTOCKED"
        product.last_observed = timestamp
        product.risk_score = 0.0
        return [self._recalculate(state, f"Product {product_id} added to table; restocking recorded with no risk change")]

    def mark_concealed(self, person_id: int, product_id: int, timestamp: float, destination: str) -> list[RiskChange]:
        state = self.person_states.get(person_id)
        product = state.products.get(product_id) if state else None
        if state is None or product is None:
            return []
        if product.concealed:
            return []
        product.state = "CONCEALED"
        product.concealed = True
        product.last_observed = timestamp
        product.risk_score = config.CONCEALMENT_THRESHOLD
        self.finalized_risk = max(self.finalized_risk, config.CONCEALMENT_THRESHOLD)
        return [self._recalculate(state, f"Product {product_id} positively observed entering {destination}")]

    def mark_track_lost(self, person_id: int, product_id: int, timestamp: float) -> list[RiskChange]:
        state = self.person_states.get(person_id)
        product = state.products.get(product_id) if state else None
        if state is None or product is None or product.state in {"RETURNED", "RESTOCKED", "CONCEALED"}:
            return []
        product.state = "TRACK_LOST"
        product.last_observed = timestamp
        product.risk_score = max(product.risk_score, config.AMBIGUOUS_THRESHOLD)
        return [self._recalculate(state, f"Product {product_id} track lost; destination is unknown")]

    def finalize_video(self, timestamp: float) -> list[RiskChange]:
        changes: list[RiskChange] = []
        for state in self.person_states.values():
            for product in state.products.values():
                if product.state in {"PICKED_UP", "WITH_PERSON", "TRACK_LOST"}:
                    product.state = "VIDEO_ENDED_UNRESOLVED"
                    product.last_observed = timestamp
                    product.risk_score = max(product.risk_score, config.AMBIGUOUS_THRESHOLD)
                    changes.append(self._recalculate(state, f"Video ended while product {product.product_id} remained unresolved"))
        return changes

    def current_risk(self) -> float:
        return max(
            [self.finalized_risk]
            + [state.risk_score for state in self.person_states.values()]
        )

    def _recalculate(self, state: PersonState, message: str) -> RiskChange:
        old_level = state.alert_level
        old_score = state.risk_score
        unresolved = [product for product in state.products.values() if product.state not in {"RETURNED", "RESTOCKED"}]
        state.risk_score = max((product.risk_score for product in unresolved), default=0.0)
        state.active_signals = {product.state.replace("_", " ") for product in unresolved}
        if state.bag_nearby:
            state.active_signals.add("BAG NEARBY (CONTEXT ONLY)")
        state.alert_level = self.level_for(state.risk_score)
        if any(product.concealed for product in unresolved):
            self.finalized_risk = max(self.finalized_risk, state.risk_score)
        crossed = old_score < config.AMBIGUOUS_THRESHOLD <= state.risk_score
        direction = "+" if state.risk_score >= old_score else ""
        return RiskChange(
            state.person_id,
            f"[PRODUCT FLOW] Person {state.person_id}: {message}; risk {direction}{state.risk_score:.2f}",
            old_level,
            state.alert_level,
            crossed,
        )

    def expire_missing(self, timestamp: float) -> list[PersonState]:
        expired = [state for state in self.person_states.values() if timestamp - state.last_seen > config.PERSON_GRACE_SECONDS]
        for state in expired:
            self.finalized_risk = max(self.finalized_risk, state.risk_score)
            del self.person_states[state.person_id]
        return expired
