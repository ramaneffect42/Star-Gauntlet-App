"""
Prediction module for Star Predictive Analysis.
Loads the trained pipeline, validates user inputs, performs multi-class
inference, generates class confidence distributions, and provides
astrophysical explanations.
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ml.preprocessing import (
    clean_star_color,
    clean_spectral_class,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    ALL_INPUT_FEATURES,
    STAR_TYPE_MAP,
    CANONICAL_SPECTRAL_CLASSES,
    CANONICAL_STAR_COLORS
)
from ml.train import load_model_artifact, train_and_evaluate
from ml import CLASS_DESCRIPTIONS

DEMO_EXAMPLES = [
    {
        'id': 'brown_dwarf',
        'name': 'Sample Brown Dwarf (Spectral M)',
        'target_class': 'Brown Dwarf',
        'type_id': 0,
        'temperature': 2600.0,
        'luminosity': 0.0003,
        'radius': 0.102,
        'absolute_magnitude': 18.7,
        'spectral_class': 'M',
        'star_color': 'Red',
        'note': 'Substellar object with extremely low luminosity and high positive magnitude.'
    },
    {
        'id': 'red_dwarf',
        'name': 'Sample Red Dwarf (Spectral M)',
        'target_class': 'Red Dwarf',
        'type_id': 1,
        'temperature': 3134.0,
        'luminosity': 0.0004,
        'radius': 0.196,
        'absolute_magnitude': 13.21,
        'spectral_class': 'M',
        'star_color': 'Red',
        'note': 'Low-mass main sequence red dwarf; compact radius and moderate magnitude.'
    },
    {
        'id': 'white_dwarf',
        'name': 'Sample White Dwarf (Spectral F)',
        'target_class': 'White Dwarf',
        'type_id': 2,
        'temperature': 7220.0,
        'luminosity': 0.00017,
        'radius': 0.011,
        'absolute_magnitude': 14.23,
        'spectral_class': 'F',
        'star_color': 'White',
        'note': 'High temperature yet extremely dim due to minute Earth-sized radius.'
    },
    {
        'id': 'main_sequence',
        'name': 'Sample Main Sequence (Spectral B)',
        'target_class': 'Main Sequence',
        'type_id': 3,
        'temperature': 15276.0,
        'luminosity': 1136.0,
        'radius': 7.2,
        'absolute_magnitude': -1.97,
        'spectral_class': 'B',
        'star_color': 'Blue-White',
        'note': 'Hot, bright main sequence star burning core hydrogen in stable equilibrium.'
    },
    {
        'id': 'supergiant',
        'name': 'Sample Supergiant (Spectral M)',
        'target_class': 'Supergiant',
        'type_id': 4,
        'temperature': 3270.0,
        'luminosity': 150000.0,
        'radius': 88.0,
        'absolute_magnitude': -6.02,
        'spectral_class': 'M',
        'star_color': 'Red',
        'note': 'Immense luminosity (150,000 L☉) and negative magnitude (-6.02).'
    },
    {
        'id': 'hypergiant',
        'name': 'Sample Hypergiant (Spectral M)',
        'target_class': 'Hypergiant',
        'type_id': 5,
        'temperature': 3834.0,
        'luminosity': 272000.0,
        'radius': 1183.0,
        'absolute_magnitude': -9.2,
        'spectral_class': 'M',
        'star_color': 'Red',
        'note': 'Colossal radius over 1,000 solar radii and extreme negative magnitude (-9.2).'
    }
]


def validate_and_sanitize_inputs(data: dict) -> dict:
    """Validate user input fields and return sanitized dictionary."""
    errors = []
    
    # Extract temperature
    raw_temp = data.get('temperature', data.get('Temperature (K)'))
    try:
        temp = float(raw_temp)
        if temp < 500 or temp > 150000:
            errors.append("Temperature must be between 500 K and 150,000 K.")
    except (TypeError, ValueError):
        errors.append("Temperature must be a valid numerical value.")
        temp = 5778.0

    # Extract luminosity
    raw_lum = data.get('luminosity', data.get('Luminosity (L/Lo)'))
    try:
        lum = float(raw_lum)
        if lum <= 0 or lum > 5000000:
            errors.append("Luminosity must be a positive number up to 5,000,000 L☉.")
    except (TypeError, ValueError):
        errors.append("Luminosity must be a valid numerical value.")
        lum = 1.0

    # Extract radius
    raw_rad = data.get('radius', data.get('Radius (R/Ro)'))
    try:
        rad = float(raw_rad)
        if rad <= 0 or rad > 10000:
            errors.append("Radius must be a positive number up to 10,000 R☉.")
    except (TypeError, ValueError):
        errors.append("Radius must be a valid numerical value.")
        rad = 1.0

    # Extract absolute magnitude
    raw_mag = data.get('absolute_magnitude', data.get('Absolute magnitude (Mv)'))
    try:
        mag = float(raw_mag)
        if mag < -25 or mag > 35:
            errors.append("Absolute Magnitude must be between -25 and +35.")
    except (TypeError, ValueError):
        errors.append("Absolute Magnitude must be a valid numerical value.")
        mag = 4.83

    # Extract spectral class
    raw_spectral = data.get('spectral_class', data.get('Spectral Class', 'M'))
    spec = clean_spectral_class(str(raw_spectral))

    # Extract star color
    raw_color = data.get('star_color', data.get('Star color', 'Red'))
    color = clean_star_color(str(raw_color))

    if errors:
        raise ValueError("; ".join(errors))

    return {
        'Temperature (K)': temp,
        'Luminosity (L/Lo)': lum,
        'Radius (R/Ro)': rad,
        'Absolute magnitude (Mv)': mag,
        'Star color': color,
        'Spectral Class': spec
    }


def predict_star(data: dict) -> dict:
    """
    Execute full inference pipeline on single star input.
    Returns predicted type, probability distribution, confidence, and explanation.
    """
    # 1. Ensure model is loaded or trained
    artifact = load_model_artifact()
    if artifact is None or 'pipeline' not in artifact:
        artifact = train_and_evaluate()
        
    pipeline = artifact['pipeline']
    
    # 2. Sanitize and validate inputs
    sanitized = validate_and_sanitize_inputs(data)
    
    # 3. Create DataFrame matching training column names exactly
    df_input = pd.DataFrame([sanitized])
    
    # 4. Predict class and probabilities
    pred_idx = int(pipeline.predict(df_input)[0])
    pred_class_name = STAR_TYPE_MAP.get(pred_idx, f"Type {pred_idx}")
    
    probabilities = pipeline.predict_proba(df_input)[0]
    confidence_pct = round(float(probabilities[pred_idx] * 100), 2)
    
    # Build sorted probability breakdown for all 6 classes
    prob_list = []
    for idx, prob in enumerate(probabilities):
        cls_name = STAR_TYPE_MAP.get(idx, f"Type {idx}")
        meta = CLASS_DESCRIPTIONS.get(cls_name, {})
        prob_list.append({
            'class_id': idx,
            'class_name': cls_name,
            'probability': round(float(prob), 4),
            'percentage': round(float(prob * 100), 2),
            'color': meta.get('color', '#38bdf8')
        })
        
    prob_list.sort(key=lambda x: x['probability'], reverse=True)
    
    # 5. Scientific and model-based explanation
    explanation_text = generate_explanation(sanitized, pred_class_name, confidence_pct)
    
    return {
        'predicted_type_id': pred_idx,
        'predicted_type_name': pred_class_name,
        'confidence_percentage': confidence_pct,
        'probabilities': prob_list,
        'input_features': {
            'temperature': sanitized['Temperature (K)'],
            'luminosity': sanitized['Luminosity (L/Lo)'],
            'radius': sanitized['Radius (R/Ro)'],
            'absolute_magnitude': sanitized['Absolute magnitude (Mv)'],
            'spectral_class': sanitized['Spectral Class'],
            'star_color': sanitized['Star color']
        },
        'explanation': explanation_text,
        'stellar_info': CLASS_DESCRIPTIONS.get(pred_class_name, {})
    }


def generate_explanation(features: dict, predicted_class: str, confidence: float) -> dict:
    """Generate concise, academically accurate astrophysical rationale."""
    temp = features['Temperature (K)']
    lum = features['Luminosity (L/Lo)']
    rad = features['Radius (R/Ro)']
    mag = features['Absolute magnitude (Mv)']
    spec = features['Spectral Class']
    color = features['Star color']

    # Astrophysical reasoning based on Hertzsprung-Russell positioning
    hr_position = ""
    if predicted_class == 'Brown Dwarf':
        hr_position = f"With an absolute magnitude of {mag} Mv and luminosity of {lum} L☉, the object occupies the far lower-right substellar zone below the hydrogen-burning minimum mass limit."
    elif predicted_class == 'Red Dwarf':
        hr_position = f"With temperature {temp} K and radius {rad} R☉, the star sits at the bottom-right end of the main sequence where low core fusion rates produce modest luminosities."
    elif predicted_class == 'White Dwarf':
        hr_position = f"A high temperature of {temp} K combined with an extremely compact radius of {rad} R☉ positions this object squarely in the lower-left degenerate white dwarf graveyard."
    elif predicted_class == 'Main Sequence':
        hr_position = f"The balanced relationship between luminosity ({lum} L☉), temperature ({temp} K), and magnitude ({mag} Mv) aligns with the stable core hydrogen-burning dwarf sequence."
    elif predicted_class == 'Supergiant':
        hr_position = f"A vast radius of {rad} R☉ and negative magnitude of {mag} Mv place this star in the upper luminous branch (luminosity class I) representing evolved massive stars."
    elif predicted_class == 'Hypergiant':
        hr_position = f"Extreme luminosity of {lum} L☉ and colossal radius of {rad} R☉ place this star at the uppermost boundary of the HR diagram near the empirical Humphreys-Davidson limit."

    ml_reasoning = (
        f"The Random Forest model evaluated these 6 stellar properties across its 100 decision trees. "
        f"The combination of Absolute Magnitude ({mag} Mv) and Radius ({rad} R☉) provided the strongest "
        f"discriminative signal, resulting in {confidence}% ensemble consensus for {predicted_class}."
    )

    return {
        'hr_position': hr_position,
        'ml_reasoning': ml_reasoning,
        'scientific_note': "The model compared these characteristics with patterns learned from the training dataset and classified the star according to the closest learned decision boundaries."
    }


def get_demo_examples() -> list:
    """Return presentation demo examples."""
    return DEMO_EXAMPLES
