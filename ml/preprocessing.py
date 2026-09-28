"""
Preprocessing module for Star Predictive Analysis.
Handles data loading, cleaning, validation, and feature transformation pipelines.
"""

import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

NUMERICAL_FEATURES = [
    'Temperature (K)',
    'Luminosity (L/Lo)',
    'Radius (R/Ro)',
    'Absolute magnitude (Mv)'
]

CATEGORICAL_FEATURES = [
    'Star color',
    'Spectral Class'
]

ALL_INPUT_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = 'Star type'
TARGET_NAME_COLUMN = 'Star category'

CANONICAL_SPECTRAL_CLASSES = ['O', 'B', 'A', 'F', 'G', 'K', 'M']

CANONICAL_STAR_COLORS = [
    'Red',
    'Blue',
    'Blue-White',
    'White',
    'Yellow-White',
    'Yellowish',
    'Orange',
    'Pale Yellow Orange',
    'Orange-Red'
]

STAR_TYPE_MAP = {
    0: 'Brown Dwarf',
    1: 'Red Dwarf',
    2: 'White Dwarf',
    3: 'Main Sequence',
    4: 'Supergiant',
    5: 'Hypergiant'
}

REVERSE_STAR_TYPE_MAP = {v: k for k, v in STAR_TYPE_MAP.items()}


def clean_star_color(val: str) -> str:
    """Normalize inconsistent star color strings from the dataset."""
    if not isinstance(val, str) or not val.strip():
        return 'Red'
    
    clean = val.strip().lower().replace('-', ' ')
    
    if 'blue' in clean and 'white' in clean:
        return 'Blue-White'
    elif 'yellow' in clean and 'white' in clean:
        return 'Yellow-White'
    elif 'orange' in clean and 'red' in clean:
        return 'Orange-Red'
    elif 'pale' in clean and 'yellow' in clean:
        return 'Pale Yellow Orange'
    elif 'blue' in clean:
        return 'Blue'
    elif 'white' in clean or 'whitish' in clean:
        return 'White'
    elif 'yellow' in clean:
        return 'Yellowish'
    elif 'orange' in clean:
        return 'Orange'
    elif 'red' in clean:
        return 'Red'
    return val.strip().title()


def clean_spectral_class(val: str) -> str:
    """Validate and normalize spectral class string."""
    if not isinstance(val, str) or not val.strip():
        return 'M'
    val = val.strip().upper()
    if val in CANONICAL_SPECTRAL_CLASSES:
        return val
    return 'M'


def load_raw_dataset(csv_path: str = 'data/stars.csv') -> pd.DataFrame:
    """Load star dataset and ensure target mappings exist."""
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Star dataset not found at {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Handle missing values if any
    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].median())
                
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    # Standardize color column
    if 'Star color' in df.columns:
        df['Star color'] = df['Star color'].apply(clean_star_color)

    # Standardize spectral class
    if 'Spectral Class' in df.columns:
        df['Spectral Class'] = df['Spectral Class'].apply(clean_spectral_class)

    # Ensure Star category is populated
    if 'Star type' in df.columns:
        df['Star type'] = df['Star type'].astype(int)
        df['Star category'] = df['Star type'].map(STAR_TYPE_MAP)

    return df


def get_preprocessor() -> ColumnTransformer:
    """
    Build scikit-learn ColumnTransformer for numerical scaling
    and categorical one-hot encoding.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    return preprocessor


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Calculate descriptive statistics and metadata for dataset explorer."""
    total_records = len(df)
    total_features = len(NUMERICAL_FEATURES) + len(CATEGORICAL_FEATURES)
    
    # Missing values
    missing_dict = {col: int(df[col].isnull().sum()) for col in df.columns}
    
    # Data types
    dtypes_dict = {col: str(df[col].dtype) for col in df.columns}
    
    # Numerical statistics
    num_stats = {}
    for col in NUMERICAL_FEATURES:
        series = df[col]
        num_stats[col] = {
            'mean': round(float(series.mean()), 4),
            'std': round(float(series.std()), 4),
            'min': round(float(series.min()), 4),
            '25%': round(float(series.quantile(0.25)), 4),
            '50%': round(float(series.median()), 4),
            '75%': round(float(series.quantile(0.75)), 4),
            'max': round(float(series.max()), 4),
        }
        
    # Class distribution
    class_counts = df['Star category'].value_counts().to_dict()
    # Ensure all 6 classes are represented
    ordered_distribution = {
        cls: int(class_counts.get(cls, 0)) for cls in STAR_TYPE_MAP.values()
    }
    
    # Sample rows (first 10 records)
    sample_rows = df.head(10).to_dict(orient='records')
    
    return {
        'total_records': total_records,
        'total_features': total_features,
        'feature_names': ALL_INPUT_FEATURES,
        'numerical_features': NUMERICAL_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'missing_values': missing_dict,
        'data_types': dtypes_dict,
        'numerical_stats': num_stats,
        'class_distribution': ordered_distribution,
        'sample_rows': sample_rows,
        'spectral_classes': CANONICAL_SPECTRAL_CLASSES,
        'star_colors': CANONICAL_STAR_COLORS
    }
