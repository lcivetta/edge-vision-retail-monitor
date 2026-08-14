import unittest

from risk_engine import RiskEngine


class ProductFlowTests(unittest.TestCase):
    def person(self, engine: RiskEngine):
        engine.observe(1, 0.0, True, False)
        return engine.person_states[1]

    def test_pickup_long_hold_and_return(self):
        engine = RiskEngine()
        state = self.person(engine)
        engine.apply_shelf_state(1, 0.0, True, True, 1, False)
        self.assertEqual(state.risk_score, 0.15)
        engine.apply_shelf_state(1, 11.0, True, True, 1, False)
        self.assertAlmostEqual(state.risk_score, 0.20)
        engine.apply_shelf_state(1, 12.0, False, False, 0, False)
        self.assertEqual(state.risk_score, 0.0)
        self.assertEqual(state.products[1].state, "RETURNED")

    def test_pure_restocking_has_no_risk(self):
        engine = RiskEngine()
        state = self.person(engine)
        engine.mark_restocked(1, 1, 1.0)
        self.assertEqual(state.risk_score, 0.0)

    def test_track_loss_is_ambiguous_not_concealment(self):
        engine = RiskEngine()
        state = self.person(engine)
        engine.apply_shelf_state(1, 0.0, True, True, 1, False)
        engine.mark_track_lost(1, 1, 2.0)
        self.assertEqual(state.risk_score, 0.30)
        self.assertEqual(state.products[1].state, "TRACK_LOST")

    def test_concealment_then_exit_reaches_one(self):
        engine = RiskEngine()
        state = self.person(engine)
        engine.apply_shelf_state(1, 0.0, True, True, 1, False)
        engine.mark_concealed(1, 1, 1.0, "pocket")
        self.assertEqual(state.risk_score, 0.75)
        engine.apply_shelf_state(1, 2.0, True, True, 1, True)
        self.assertEqual(state.risk_score, 1.0)

    def test_concealment_is_not_cleared_by_settled_table_pixels(self):
        engine = RiskEngine()
        state = self.person(engine)
        engine.apply_shelf_state(1, 0.0, True, True, 1, False)
        engine.mark_concealed(1, 1, 1.0, "backpack opening")
        engine.apply_shelf_state(1, 2.0, False, False, 0, False)
        self.assertEqual(state.risk_score, 0.75)
        self.assertEqual(state.products[1].state, "CONCEALED")

    def test_concealment_survives_person_track_expiration(self):
        engine = RiskEngine()
        state = self.person(engine)
        engine.apply_shelf_state(1, 0.0, True, True, 1, False)
        engine.mark_concealed(1, 1, 1.0, "backpack opening")
        engine.expire_missing(3.1)
        self.assertNotIn(1, engine.person_states)
        self.assertEqual(engine.current_risk(), 0.75)

    def test_short_unresolved_video_is_ambiguous(self):
        engine = RiskEngine()
        state = self.person(engine)
        engine.apply_shelf_state(1, 0.0, True, True, 1, False)
        engine.finalize_video(5.0)
        self.assertEqual(state.risk_score, 0.30)
        self.assertEqual(state.products[1].state, "VIDEO_ENDED_UNRESOLVED")

    def test_products_are_independent(self):
        engine = RiskEngine()
        state = self.person(engine)
        engine.apply_shelf_state(1, 0.0, True, True, 2, False)
        engine.mark_returned(1, 1, 2.0)
        self.assertEqual(state.products[1].risk_score, 0.0)
        self.assertEqual(state.products[2].risk_score, 0.15)
        self.assertEqual(state.risk_score, 0.15)


if __name__ == "__main__":
    unittest.main()
