"""
End-to-End Test Suite for Star Predictive Analysis.
Tests dataset integrity, preprocessing transforms, training metrics,
inference stability, error handling, and Flask web endpoints.
"""

import os
import sys
import unittest
import json
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocessing import (
    load_raw_dataset,
    get_dataset_summary,
    clean_star_color,
    clean_spectral_class,
    STAR_TYPE_MAP
)
from ml.train import train_and_evaluate, load_model_artifact, MODEL_PATH
from ml.predict import predict_star, get_demo_examples, validate_and_sanitize_inputs
from app import app


class TestStarPredictiveAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = app.test_client()

    def test_01_dataset_integrity(self):
        """Verify dataset contains 240 records with exact 6 balanced classes."""
        df = load_raw_dataset('data/stars.csv')
        self.assertEqual(len(df), 240)
        self.assertEqual(df['Star type'].nunique(), 6)
        
        # Check balance
        counts = df['Star type'].value_counts()
        for star_id in range(6):
            self.assertEqual(counts[star_id], 40, f"Class {star_id} should have 40 samples")

        # Check no missing values
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_02_preprocessing_normalization(self):
        """Verify color normalization and spectral class sanitization."""
        self.assertEqual(clean_star_color('Blue-white'), 'Blue-White')
        self.assertEqual(clean_star_color('Blue White '), 'Blue-White')
        self.assertEqual(clean_star_color('yellow-white'), 'Yellow-White')
        self.assertEqual(clean_star_color('RED'), 'Red')
        self.assertEqual(clean_star_color('whitish'), 'White')
        
        self.assertEqual(clean_spectral_class('m'), 'M')
        self.assertEqual(clean_spectral_class(' B '), 'B')
        self.assertEqual(clean_spectral_class('INVALID'), 'M')

    def test_03_model_training_and_artifact(self):
        """Verify model trains, evaluates, and saves joblib artifact."""
        artifact = train_and_evaluate()
        self.assertIsNotNone(artifact)
        self.assertIn('pipeline', artifact)
        self.assertIn('metrics', artifact)
        self.assertIn('confusion_matrix', artifact)
        
        metrics = artifact['metrics']
        self.assertGreaterEqual(metrics['accuracy'], 0.95)
        self.assertGreaterEqual(metrics['f1_macro'], 0.95)
        
        # Check confusion matrix dimensions (6x6)
        cm = artifact['confusion_matrix']
        self.assertEqual(len(cm), 6)
        for row in cm:
            self.assertEqual(len(row), 6)

        # Check joblib persistence
        self.assertTrue(os.path.exists(MODEL_PATH))
        loaded = load_model_artifact(MODEL_PATH)
        self.assertIsNotNone(loaded)

    def test_04_inference_on_all_demo_stars(self):
        """Verify that all 6 demo presets predict their expected star classes."""
        examples = get_demo_examples()
        self.assertEqual(len(examples), 6)
        
        for ex in examples:
            result = predict_star(ex)
            self.assertEqual(
                result['predicted_type_name'],
                ex['target_class'],
                f"Expected {ex['target_class']} but got {result['predicted_type_name']}"
            )
            self.assertGreater(result['confidence_percentage'], 70.0)
            
            # Check probabilities structure
            probs = result['probabilities']
            self.assertEqual(len(probs), 6)
            prob_sum = sum(p['percentage'] for p in probs)
            self.assertAlmostEqual(prob_sum, 100.0, delta=1.0)

    def test_05_input_validation_and_error_handling(self):
        """Verify invalid inputs trigger clean ValueError."""
        # Temperature out of bounds
        with self.assertRaises(ValueError):
            predict_star({'temperature': -100, 'luminosity': 1.0, 'radius': 1.0, 'absolute_magnitude': 5.0})

        # Non-numeric values
        with self.assertRaises(ValueError):
            predict_star({'temperature': 'hot', 'luminosity': 1.0, 'radius': 1.0, 'absolute_magnitude': 5.0})

    def test_06_flask_page_routes(self):
        """Verify all HTML template pages render with 200 OK."""
        pages = ['/', '/dashboard', '/analysis', '/dataset', '/training', '/prediction', '/analytics', '/about']
        for page in pages:
            res = self.client.get(page)
            self.assertEqual(res.status_code, 200, f"Page {page} failed to render")
            self.assertIn(b'STAR PREDICTIVE ANALYSIS', res.data)

    def test_07_flask_rest_apis(self):
        """Verify all JSON REST endpoints return successful responses."""
        # Summary API
        res = self.client.get('/api/dataset/summary')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['total_records'], 240)

        # Records API
        res = self.client.get('/api/dataset/records')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['count'], 240)

        # Model Status API
        res = self.client.get('/api/model/status')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'ready')

        # Demo Examples API
        res = self.client.get('/api/demo/examples')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(len(data['examples']), 6)

        # Analytics Charts API
        res = self.client.get('/api/analytics/charts')
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data['status'], 'success')
        self.assertIn('distribution', data)
        self.assertIn('temp_vs_mag', data)

        # Prediction API (Valid)
        payload = {
            'temperature': 5778,
            'luminosity': 1.0,
            'radius': 1.0,
            'absolute_magnitude': 4.83,
            'spectral_class': 'G',
            'star_color': 'Yellowish'
        }
        res = self.client.post('/api/model/predict', data=json.dumps(payload), content_type='application/json')
        self.assertEqual(res.status_code, 200)
        pred_data = res.get_json()
        self.assertEqual(pred_data['status'], 'success')
        self.assertIn('predicted_type_name', pred_data['data'])

        # Prediction API (Invalid payload error handling)
        bad_payload = {'temperature': -9999}
        res = self.client.post('/api/model/predict', data=json.dumps(bad_payload), content_type='application/json')
        self.assertEqual(res.status_code, 422)
        err_data = res.get_json()
        self.assertEqual(err_data['status'], 'error')


if __name__ == '__main__':
    unittest.main()
