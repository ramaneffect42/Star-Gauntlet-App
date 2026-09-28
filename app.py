"""
Star Predictive Analysis - Web Application
Flask backend serving interactive pages and REST APIs for stellar classification,
exploratory data analysis, model training, and prediction.
"""

import os
import sys
import json
import logging
from flask import Flask, render_template, request, jsonify, send_file

# Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocessing import (
    load_raw_dataset,
    get_dataset_summary,
    CANONICAL_SPECTRAL_CLASSES,
    CANONICAL_STAR_COLORS,
    STAR_TYPE_MAP,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from ml.train import train_and_evaluate, load_model_artifact, MODEL_PATH
from ml.predict import predict_star, get_demo_examples
from ml import STAR_CLASSES, CLASS_DESCRIPTIONS

# Initialize Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'stellar-ml-secret-key-2026'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure model is trained on startup if not already existing
if not os.path.exists(MODEL_PATH):
    logger.info("Model not found on startup. Training initial model...")
    try:
        train_and_evaluate()
        logger.info("Initial model trained and persisted.")
    except Exception as e:
        logger.error(f"Error training initial model: {e}")


# ==========================================
# PAGE ROUTES
# ==========================================

@app.route('/')
@app.route('/dashboard')
def index():
    """Dashboard landing page."""
    artifact = load_model_artifact()
    metrics = artifact['metrics'] if artifact else {
        'accuracy_percentage': 0,
        'f1_percentage': 0,
        'precision_percentage': 0,
        'recall_percentage': 0
    }
    return render_template(
        'index.html',
        active_page='dashboard',
        metrics=metrics,
        star_classes=STAR_CLASSES,
        total_records=240,
        total_features=6
    )


@app.route('/dataset')
def dataset_page():
    """Dataset Explorer page."""
    return render_template(
        'dataset.html',
        active_page='dataset',
        star_classes=STAR_CLASSES
    )


@app.route('/analysis')
def analysis_page():
    """Analysis Workspace with pipeline stages, live execution, evaluation, and prediction."""
    artifact = load_model_artifact()
    return render_template(
        'analysis.html',
        active_page='analysis',
        artifact=artifact,
        spectral_classes=CANONICAL_SPECTRAL_CLASSES,
        star_colors=CANONICAL_STAR_COLORS,
        star_classes=STAR_CLASSES
    )


@app.route('/training')
def training_page():
    """Model Training and evaluation page."""
    return analysis_page()


@app.route('/prediction')
def prediction_page():
    """Star Prediction interactive page."""
    return analysis_page()


@app.route('/analytics')
def analytics_page():
    """Analytics & Hertzsprung-Russell charts page."""
    return render_template(
        'analytics.html',
        active_page='analytics',
        star_classes=STAR_CLASSES
    )


@app.route('/about')
def about_page():
    """About the Model and Astrophysical Principles page."""
    return render_template(
        'about.html',
        active_page='about',
        star_classes=STAR_CLASSES,
        class_descriptions=CLASS_DESCRIPTIONS
    )


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route('/api/dataset/summary', methods=['GET'])
def api_dataset_summary():
    """Return dataset statistics, missing values, and column schemas."""
    try:
        df = load_raw_dataset()
        summary = get_dataset_summary(df)
        return jsonify({'status': 'success', 'data': summary})
    except Exception as e:
        logger.error(f"Error reading dataset: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/dataset/records', methods=['GET'])
def api_dataset_records():
    """Return all records in the dataset for interactive data tables."""
    try:
        df = load_raw_dataset()
        records = df.to_dict(orient='records')
        return jsonify({
            'status': 'success',
            'count': len(records),
            'records': records
        })
    except Exception as e:
        logger.error(f"Error fetching dataset records: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/model/status', methods=['GET'])
def api_model_status():
    """Return current model metrics, confusion matrix, and feature importances."""
    artifact = load_model_artifact()
    if artifact is None:
        return jsonify({
            'status': 'not_trained',
            'message': 'Model has not been trained yet.'
        }), 404
        
    return jsonify({
        'status': 'ready',
        'metrics': artifact['metrics'],
        'train_samples': artifact.get('train_samples', 192),
        'test_samples': artifact.get('test_samples', 48),
        'confusion_matrix': artifact['confusion_matrix'],
        'class_labels': artifact['class_labels'],
        'per_class_report': artifact['per_class_report'],
        'feature_importance_high_level': artifact['feature_importance_high_level'],
        'feature_importance_detailed': artifact['feature_importance_detailed'],
        'comparison': artifact.get('comparison', {}),
        'leakage_audit': artifact.get('leakage_audit', {}),
        'trained_at': artifact.get('trained_at', 'Recently')
    })


@app.route('/api/model/train', methods=['POST'])
def api_model_train():
    """Trigger training workflow and return fresh evaluation metrics."""
    try:
        logger.info("Model retraining triggered by user.")
        artifact = train_and_evaluate()
        return jsonify({
            'status': 'success',
            'message': 'Model retrained successfully!',
            'metrics': artifact['metrics'],
            'train_samples': artifact.get('train_samples', 192),
            'test_samples': artifact.get('test_samples', 48),
            'confusion_matrix': artifact['confusion_matrix'],
            'class_labels': artifact['class_labels'],
            'feature_importance_high_level': artifact['feature_importance_high_level'],
            'comparison': artifact.get('comparison', {}),
            'leakage_audit': artifact.get('leakage_audit', {}),
            'trained_at': artifact.get('trained_at')
        })
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return jsonify({'status': 'error', 'message': f"Training failed: {str(e)}"}), 500


@app.route('/api/model/predict', methods=['POST'])
def api_model_predict():
    """Execute classification inference on input stellar parameters."""
    try:
        data = request.get_json(force=True, silent=True)
        if not data:
            return jsonify({'status': 'error', 'message': 'Missing JSON payload.'}), 400
            
        result = predict_star(data)
        return jsonify({'status': 'success', 'data': result})
    except ValueError as ve:
        return jsonify({'status': 'error', 'message': str(ve)}), 422
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'status': 'error', 'message': f"Prediction failed: {str(e)}"}), 500


@app.route('/api/demo/examples', methods=['GET'])
def api_demo_examples():
    """Return presentation demo examples from real dataset rows."""
    examples = get_demo_examples()
    return jsonify({'status': 'success', 'examples': examples})


@app.route('/api/analytics/charts', methods=['GET'])
def api_analytics_charts():
    """Return grouped dataset data for 5 analytical and astrophysical charts."""
    try:
        df = load_raw_dataset()
        
        # 1. Class Distribution
        class_dist = df['Star category'].value_counts().to_dict()
        ordered_dist = [
            {'class': cls, 'count': int(class_dist.get(cls, 0))}
            for cls in STAR_CLASSES
        ]
        
        # 2. Scatter Points grouped by star type
        scatter_temp_lum = {cls: [] for cls in STAR_CLASSES}
        scatter_temp_rad = {cls: [] for cls in STAR_CLASSES}
        scatter_temp_mag = {cls: [] for cls in STAR_CLASSES}
        
        for _, row in df.iterrows():
            cls = row['Star category']
            t = float(row['Temperature (K)'])
            l = float(row['Luminosity (L/Lo)'])
            r = float(row['Radius (R/Ro)'])
            m = float(row['Absolute magnitude (Mv)'])
            
            scatter_temp_lum[cls].append({'x': t, 'y': l, 'r': r, 'name': cls})
            scatter_temp_rad[cls].append({'x': t, 'y': r, 'l': l, 'name': cls})
            scatter_temp_mag[cls].append({'x': t, 'y': m, 'l': l, 'name': cls})

        # 3. Feature Importance
        artifact = load_model_artifact()
        feature_importance = artifact['feature_importance_high_level'] if artifact else []
        
        return jsonify({
            'status': 'success',
            'classes': STAR_CLASSES,
            'distribution': ordered_dist,
            'temp_vs_lum': scatter_temp_lum,
            'temp_vs_rad': scatter_temp_rad,
            'temp_vs_mag': scatter_temp_mag,
            'feature_importance': feature_importance
        })
    except Exception as e:
        logger.error(f"Error generating analytics data: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"\n=======================================================")
    print(f" STAR PREDICTIVE ANALYSIS WEB SERVER")
    print(f" Running at: http://127.0.0.1:{port}")
    print(f"=======================================================\n")
    app.run(host='0.0.0.0', port=port, debug=True)
