"""Tests for the Flask real-estate dashboard and its model API contracts."""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FLASK_APP_DIR = ROOT / "flask-app"
sys.path.insert(0, str(FLASK_APP_DIR))

import app as flask_app  # noqa: E402


class FakePipeline:
    def __init__(self, prediction):
        self.prediction = prediction
        self.input_frame = None

    def predict(self, input_frame):
        self.input_frame = input_frame
        return np.array([self.prediction])


class FlaskAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = flask_app.app.test_client()
        cls.options = cls.client.get("/api/options").get_json()

    def make_prediction_payload(self):
        return {
            "property_type": flask_app.training_df["property_type"]
            .dropna()
            .iloc[0],
            "sector": self.options["sectors"][0],
            "bedrooms": self.options["bedrooms"][0],
            "bathroom": self.options["bathrooms"][0],
            "balcony": self.options["balconies"][0],
            "property_age": self.options["property_age"][0],
            "built_up_area": 1000,
            "servant_room": 0,
            "store_room": 0,
            "furnishing_type": self.options["furnishing_type"][0],
            "luxury_category": self.options["luxury_category"][0],
            "floor_category": self.options["floor_category"][0],
        }

    def test_home_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<title>Gurgaon Real Estate Analytics</title>", response.data)

    def test_options_use_project_data(self):
        response = self.client.get("/api/options")
        self.assertEqual(response.status_code, 200)
        options = response.get_json()
        self.assertTrue(options["sectors"])
        self.assertTrue(options["locations"])
        self.assertTrue(options["apartments"])
        self.assertEqual(options["analytics_sectors"][0], "overall")

    def test_prediction_returns_direct_price_range(self):
        pipeline = FakePipeline(prediction=1.5)
        with patch.object(
            flask_app,
            "load_prediction_pipeline",
            return_value=pipeline,
        ):
            response = self.client.post(
                "/api/predict",
                json=self.make_prediction_payload(),
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.get_json(),
            {"status": "success", "low": 1.28, "high": 1.72},
        )
        self.assertEqual(
            pipeline.input_frame.loc[0, "built_up_area"],
            1000,
        )

    def test_prediction_rejects_unknown_categories(self):
        payload = self.make_prediction_payload()
        payload["sector"] = "not-a-training-sector"
        response = self.client.post("/api/predict", json=payload)

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["status"], "error")

    def test_apartment_recommendations_exclude_selected_property(self):
        model = {
            "property_names": ["Alpha", "Beta", "Gamma"],
            "combined_similarity": np.array(
                [
                    [1.0, 0.8, 0.2],
                    [0.8, 1.0, 0.3],
                    [0.2, 0.3, 1.0],
                ]
            ),
        }
        with patch.object(
            flask_app,
            "load_recommendation_model",
            return_value=model,
        ):
            response = self.client.post(
                "/api/recommend/apartments",
                json={"apartment": "Beta"},
            )

        self.assertEqual(response.status_code, 200)
        results = response.get_json()["results"]
        self.assertEqual(results[0], {"property": "Alpha", "score": 0.8})
        self.assertNotIn("Beta", [result["property"] for result in results])


if __name__ == "__main__":
    unittest.main()
