import unittest

from backend.features.phase4_analysis import (
    _scenario_confidence,
    _scenario_confidences_from_outputs,
    evaluate_required_modality_status,
)


class TestScenarioConfidence(unittest.TestCase):
    def test_single_scenario_confidence_bounded(self):
        conf = _scenario_confidence(
            overall=0.42,
            scenario_likelihood=0.50,
            scenario_intensity=0.70,
            edge_score=0.20,
            agreement_score=0.60,
            horizon_alignment_score=0.55,
        )
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)

    def test_confidences_non_zero_when_overall_non_trivial(self):
        scenarios = {
            "best_case": {"likelihood": 0.44, "intensity": 0.62},
            "base_case": {"likelihood": 0.33, "intensity": 0.48},
            "worst_case": {"likelihood": 0.23, "intensity": 0.71},
        }
        out = _scenario_confidences_from_outputs(
            overall=0.25,
            scenarios=scenarios,
            agreement_score=0.58,
            horizon_alignment_score=0.52,
        )
        vals = [out["best_case"], out["base_case"], out["worst_case"]]
        for v in vals:
            self.assertGreaterEqual(v, 0.05)
            self.assertLessEqual(v, 0.95)
        self.assertFalse(all(v == 0.0 for v in vals))
        self.assertGreater(len(set(round(v, 6) for v in vals)), 1)

    def test_required_modality_status_flags_missing_and_stale(self):
        status = evaluate_required_modality_status(
            {
                "tech_avg_age_hours": 80.0,
                "fundamentals_avg_age_hours": None,
            }
        )
        self.assertFalse(status["ok"])
        codes = {f.get("code") for f in status.get("failures", [])}
        self.assertIn("STALE", codes)
        self.assertIn("MISSING", codes)


if __name__ == "__main__":
    unittest.main()
