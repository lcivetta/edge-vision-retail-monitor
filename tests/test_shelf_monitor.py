import unittest

import cv2
import numpy as np

from shelf_monitor import ShelfMonitor


def table_frame(product_x: int | None) -> np.ndarray:
    frame = np.full((120, 160, 3), 150, dtype=np.uint8)
    if product_x is not None:
        for row in range(30, 70, 4):
            color = 20 if (row // 4) % 2 else 240
            cv2.rectangle(frame, (product_x, row), (product_x + 27, row + 3), (color, color, color), -1)
    return frame


class ShelfDirectionTests(unittest.TestCase):
    def monitor(self) -> ShelfMonitor:
        return ShelfMonitor((0, 0, 160, 120), fps=1.0)

    def test_unmatched_disappearance_is_removal(self):
        monitor = self.monitor()
        monitor.update(table_frame(20), [], False)
        observation = monitor.update(table_frame(None), [], False)
        self.assertTrue(observation.product_removed)
        self.assertGreaterEqual(len(observation.removed_region_boxes), 1)

    def test_balanced_source_and_destination_is_relocation(self):
        monitor = self.monitor()
        monitor.update(table_frame(20), [], False)
        observation = monitor.update(table_frame(100), [], False)
        self.assertTrue(observation.changed)
        self.assertFalse(observation.product_removed)
        self.assertEqual(observation.removed_region_boxes, [])


if __name__ == "__main__":
    unittest.main()
