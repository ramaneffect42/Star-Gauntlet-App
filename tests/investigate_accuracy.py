"""
Investigation of Model Evaluation Integrity, Data Leakage, and Accuracy.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('data/stars.csv')
print(f"Total rows: {len(df)}")
print(f"Duplicates across entire dataframe: {df.duplicated().sum()}")

num_cols = ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Absolute magnitude (Mv)']
cat_cols = ['Star color', 'Spectral Class']

print(f"Duplicates in features alone: {df.duplicated(subset=num_cols + cat_cols).sum()}")

# Check feature ranges by star type
for t in range(6):
    sub = df[df['Star type'] == t]
    cat_name = sub['Star category'].iloc[0]
    print(f"\nClass {t} ({cat_name}): N={len(sub)}")
    print(f"  Abs Mag: [{sub['Absolute magnitude (Mv)'].min():.2f}, {sub['Absolute magnitude (Mv)'].max():.2f}]")
    print(f"  Radius:  [{sub['Radius (R/Ro)'].min():.4f}, {sub['Radius (R/Ro)'].max():.4f}]")
    print(f"  Lum:     [{sub['Luminosity (L/Lo)'].min():.5f}, {sub['Luminosity (L/Lo)'].max():.1f}]")
    print(f"  Temp:    [{sub['Temperature (K)'].min()}, {sub['Temperature (K)'].max()}]")

# Build strictly leak-free pipeline
# Normalizer for colors
def clean_col(s):
    if not isinstance(s, str): return 'Red'
    s = s.strip().lower().replace('-', ' ')
    if 'blue' in s and 'white' in s: return 'Blue-White'
    if 'yellow' in s and 'white' in s: return 'Yellow-White'
    if 'orange' in s and 'red' in s: return 'Orange-Red'
    if 'pale' in s and 'yellow' in s: return 'Pale Yellow Orange'
    if 'blue' in s: return 'Blue'
    if 'white' in s: return 'White'
    if 'yellow' in s: return 'Yellowish'
    if 'orange' in s: return 'Orange'
    return 'Red'

df['clean_color'] = df['Star color'].apply(clean_col)

X = df[['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Absolute magnitude (Mv)', 'clean_color', 'Spectral Class']]
y = df['Star type']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['Temperature (K)', 'Luminosity (L/Lo)', 'Radius (R/Ro)', 'Absolute magnitude (Mv)']),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), ['clean_color', 'Spectral Class'])
    ]
)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
pipe = Pipeline([('prep', preprocessor), ('clf', rf)])

# 1. Stratified 10-Fold Cross-Validation (completely unseen folds)
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
print("\n" + "="*50)
print(f"Stratified 10-Fold CV Accuracies: {np.round(cv_scores, 4)}")
print(f"Mean 10-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# 2. Stratified 5-Fold Cross-Validation
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv5_scores = cross_val_score(pipe, X, y, cv=cv5, scoring='accuracy')
print(f"Mean 5-Fold CV Accuracy:  {cv5_scores.mean()*100:.2f}% (+/- {cv5_scores.std()*100:.2f}%)")

# 3. Multiple random train/test split seeds (test size = 0.2, 48 stars)
split_accs = []
for seed in [0, 7, 21, 42, 99, 123, 456, 789, 999, 2026]:
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    acc = (pred == y_te).mean()
    split_accs.append(acc)

print(f"Random splits (10 different seeds) test accuracies: {[round(a, 3) for a in split_accs]}")
print(f"Average holdout test accuracy across 10 random seeds: {np.mean(split_accs)*100:.2f}% (min: {np.min(split_accs)*100:.2f}%, max: {np.max(split_accs)*100:.2f}%)")
