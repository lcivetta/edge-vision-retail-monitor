"""Class-agnostic fixed-camera shelf change detection inside one configured ROI."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

import config


@dataclass
class ShelfObservation:
    changed: bool
    persistent: bool
    estimated_regions: int
    change_ratio: float
    region_boxes: list[tuple[int, int, int, int]]
    product_removed: bool
    removed_region_boxes: list[tuple[int, int, int, int]]


class ShelfMonitor:
    """Compare unobstructed ROI pixels with the initial shelf appearance."""

    def __init__(self, roi: tuple[int, int, int, int], fps: float) -> None:
        self.roi = roi
        self.fps = fps
        self.baseline: np.ndarray | None = None
        self.baseline_valid: np.ndarray | None = None
        self.changed_frames = 0
        self.recovered_frames = 0
        self.active = False
        self.removal_changed_frames = 0
        self.removal_recovered_frames = 0
        self.removal_active = False
        self.last_removed_boxes: list[tuple[int, int, int, int]] = []

    def initialize(self, frame: np.ndarray, person_boxes: list[tuple[int, int, int, int]]) -> None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = self.roi
        self.baseline = cv2.GaussianBlur(gray[y1:y2, x1:x2], (5, 5), 0)
        self.baseline_valid = np.full(self.baseline.shape, 255, dtype=np.uint8)
        for box in person_boxes:
            bx1, by1, bx2, by2 = box
            cv2.rectangle(
                self.baseline_valid,
                (max(0, bx1 - x1 - 10), max(0, by1 - y1 - 10)),
                (min(x2 - x1, bx2 - x1 + 10), min(y2 - y1, by2 - y1 + 10)),
                0,
                -1,
            )

    def update(
        self, frame: np.ndarray, person_boxes: list[tuple[int, int, int, int]], paused: bool = False
    ) -> ShelfObservation:
        if self.baseline is None or self.baseline_valid is None:
            self.initialize(frame, person_boxes)
            return ShelfObservation(False, False, 0, 0.0, [], False, [])
        if paused:
            # Do not learn "missing products" while a shopper occludes the shelf.
            self.changed_frames = 0
            self.recovered_frames = 0
            return ShelfObservation(self.active, self.active, 0, 0.0, [], self.removal_active, self.last_removed_boxes)
        x1, y1, x2, y2 = self.roi
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current = cv2.GaussianBlur(gray[y1:y2, x1:x2], (5, 5), 0)
        valid = self.baseline_valid.copy()
        for bx1, by1, bx2, by2 in person_boxes:
            cv2.rectangle(
                valid,
                (max(0, bx1 - x1 - 12), max(0, by1 - y1 - 12)),
                (min(x2 - x1, bx2 - x1 + 12), min(y2 - y1, by2 - y1 + 12)),
                0,
                -1,
            )
        diff = cv2.absdiff(self.baseline, current)
        _, binary = cv2.threshold(diff, 32, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_and(binary, valid)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        valid_pixels = max(1, cv2.countNonZero(valid))
        ratio = cv2.countNonZero(binary) / valid_pixels
        components, _, stats, _ = cv2.connectedComponentsWithStats(binary)
        region_boxes = []
        for index in range(1, components):
            if stats[index, cv2.CC_STAT_AREA] < config.SHELF_MIN_COMPONENT_AREA:
                continue
            left, top, width, height = stats[index, :4]
            region_boxes.append((x1 + int(left), y1 + int(top), x1 + int(left + width), y1 + int(top + height)))
        regions = len(region_boxes)
        removed_candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        added_boxes = []
        for box in region_boxes:
            bx1, by1, bx2, by2 = box
            base_patch = self.baseline[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1]
            current_patch = current[by1 - y1:by2 - y1, bx1 - x1:bx2 - x1]
            if not base_patch.size or not current_patch.size:
                continue
            baseline_texture = float(np.abs(cv2.Laplacian(base_patch, cv2.CV_32F)).mean())
            current_texture = float(np.abs(cv2.Laplacian(current_patch, cv2.CV_32F)).mean())
            texture_delta = current_texture - baseline_texture
            if texture_delta <= config.PRODUCT_REMOVAL_TEXTURE_DELTA:
                removed_candidates.append((texture_delta, box))
            elif texture_delta >= config.PRODUCT_ADDITION_TEXTURE_DELTA:
                added_boxes.append(box)
        # A disappearance paired with an appearance elsewhere in the same
        # table ROI is a relocation. Only unmatched disappearance regions are
        # treated as provisional product removals.
        unmatched_count = max(0, len(removed_candidates) - len(added_boxes))
        removed_candidates.sort(key=lambda item: item[0])
        removed_boxes = [box for _, box in removed_candidates[:unmatched_count]]

        # A single small product may occupy far less than the global ROI ratio.
        # Treat any product-sized connected component as candidate evidence and
        # require it to persist for the configured debounce interval.
        candidate_change = ratio >= config.SHELF_CHANGE_RATIO or regions > 0
        if candidate_change:
            self.changed_frames += 1
            self.recovered_frames = 0
        elif ratio <= config.SHELF_RECOVERY_RATIO and regions == 0:
            self.recovered_frames += 1
            self.changed_frames = 0
        required = max(1, int(config.SHELF_PERSISTENCE_SECONDS * self.fps))
        if self.changed_frames >= required:
            self.active = True
        elif self.recovered_frames >= required:
            self.active = False

        if removed_boxes:
            self.removal_changed_frames += 1
            self.removal_recovered_frames = 0
            self.last_removed_boxes = removed_boxes
        else:
            self.removal_recovered_frames += 1
            self.removal_changed_frames = 0
        if self.removal_changed_frames >= required:
            self.removal_active = True
        elif self.removal_recovered_frames >= required:
            self.removal_active = False
            self.last_removed_boxes = []
        return ShelfObservation(
            self.active,
            self.active and self.changed_frames >= required,
            regions,
            ratio,
            region_boxes,
            self.removal_active,
            self.last_removed_boxes,
        )
