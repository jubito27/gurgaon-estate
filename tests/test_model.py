"""Regression-model smoke tests for the DVC artifacts used by the web app."""

import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = (
    ROOT
    / "data"
    / "processed"
    / "feature_selection"
    / "gurgaon_properties_post_feature_selection_v2.csv"
)
PIPELINE_PATH = ROOT / "model" / "pipeline.pkl"


class TestRegressionPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        for path in (FEATURES_PATH, PIPELINE_PATH):
            if not path.is_file():
                raise FileNotFoundError(
                    f"Required DVC deployment artifact is missing: {path}"
                )
        cls.pipeline = joblib.load(PIPELINE_PATH)
        cls.frame = pd.read_csv(FEATURES_PATH)
        if cls.frame.empty or "price" not in cls.frame:
            raise ValueError("Feature-selection data is empty or has no price target")
        cls.features = cls.frame.drop(columns=["price"]).head(25)

    def test_pipeline_loads(self):
        self.assertIsNotNone(self.pipeline)
        self.assertTrue(callable(getattr(self.pipeline, "predict", None)))

    def test_pipeline_predicts_finite_prices(self):
        predictions = np.asarray(self.pipeline.predict(self.features)).reshape(-1)
        self.assertEqual(len(predictions), len(self.features))
        self.assertTrue(np.isfinite(predictions).all())


if __name__ == "__main__":
    unittest.main()
