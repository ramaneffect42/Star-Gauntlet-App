# Star Predictive Analysis: Machine Learning Based Stellar Classification

A complete, interactive machine learning web prototype designed to classify astronomical stellar observations into **six distinct star types** using observational, photometric, and physical parameters.

---

## 1. Project Overview

In stellar astrophysics, determining a star's evolutionary state and taxonomic classification typically requires complex multi-band photometric observations and high-resolution spectroscopy. The **Star Predictive Analysis** application demonstrates how supervised ensemble machine learning—specifically a **Random Forest Classifier** wrapped inside a reproducible **scikit-learn Pipeline**—can classify stars into their physical classifications with **100% test accuracy** while outputting real class probability distributions and astrophysically grounded explanations.

### The Six Target Star Classes
The model classifies stars into the following 6 astronomical categories:
1. **Brown Dwarf** (Substellar object, incapable of sustained core hydrogen fusion)
2. **Red Dwarf** (Low-mass main sequence dwarf star, M spectral type)
3. **White Dwarf** (Electron-degenerate stellar remnant core)
4. **Main Sequence** (Stable core hydrogen-burning dwarf star, luminosity class V)
5. **Supergiant** (Evolved massive star, luminosity class I)
6. **Hypergiant** (Extremely luminous and massive star near the Eddington limit, luminosity class 0)

---

## 2. Dataset

* **Source:** [Kaggle: Star Dataset to Predict Star Type](https://www.kaggle.com/datasets/deepu1109/star-dataset) (Authored by *Deepraj Baidya*)
* **Total Observations:** 240 stellar records
* **Class Balance:** Perfectly balanced (exactly 40 observations per class)
* **Missing Values:** 0 (Clean and verified across all features)

### Input Features (6 Total)
| Feature Name | Variable Type | Physical Unit | Description |
| :--- | :--- | :--- | :--- |
| **Temperature** | Numerical | Kelvin ($K$) | Photospheric effective surface temperature |
| **Luminosity** | Numerical | Solar Luminosity ($L/L_\odot$) | Total radiated electromagnetic power relative to the Sun |
| **Radius** | Numerical | Solar Radii ($R/R_\odot$) | Photospheric physical radius relative to the Sun |
| **Absolute Magnitude** | Numerical | Magnitude ($M_v$) | Intrinsic stellar brightness observed at standard 10 parsecs |
| **Spectral Class** | Categorical | Harvard $O, B, A, F, G, K, M$ | Absorption line sequence indicating atmospheric temperature |
| **Star Colour** | Categorical | Color Name | Visual radiation wavelength (Red, Blue, White, etc.) |

*Note: Star Type is the target variable (0–5) and is strictly excluded from input features.*

---

## 3. Machine Learning Architecture

```
                       DATASET (240 Stars)
                                │
                                ▼
                       DATA PREPROCESSING
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
     Numerical Features                   Categorical Features
  (Temp, Lum, Radius, Mag)                (Spectral, Star Colour)
             │                                     │
             ▼                                     ▼
       StandardScaler                      OneHotEncoder
             │                                     │
             └──────────────────┬──────────────────┘
                                ▼
                         SCIKIT-LEARN
                       COLUMNTRANSFORMER
                                │
                                ▼
                     STRATIFIED TRAIN/TEST
                      (80% Train / 20% Test)
                                │
                                ▼
                     RANDOM FOREST CLASSIFIER
                        (100 Decision Trees)
                                │
             ┌──────────────────┴──────────────────┐
             ▼                                     ▼
      MODEL EVALUATION                     REAL-TIME INFERENCE
  • Accuracy: 100.0%                    • Predicted Star Type
  • Precision: 100.0%                   • Class Probabilities (predict_proba)
  • Recall: 100.0%                      • HR Diagram Positioning
  • 6x6 Confusion Matrix                • Astrophysical Rationale
  • Feature Importances
```

### Why Random Forest?
1. **Non-linear Discontinuities:** Stellar evolutionary tracks across the Hertzsprung-Russell diagram feature non-linear boundaries between degenerate dwarfs, main-sequence stars, and expanded giants. Decision trees naturally carve these multi-dimensional hyperplanes.
2. **Tabular Robustness:** Resilient against outliers in extreme physical quantities (e.g. hypergiant radii > 1,000 $R_\odot$).
3. **True Probabilities:** Generates true confidence percentages using ensemble voting averages via `predict_proba()`.
4. **Feature Importance:** Computes explicit Gini impurity reduction showing feature contributions.

### Evaluation Integrity & Investigation of Accuracy
An audit was conducted on the dataset and model evaluation methodology to investigate whether the 100% test accuracy was legitimate or a consequence of data leakage:
1. **Data Leakage Audit:** Target variable `Star type` and label `Star category` are strictly quarantined from input feature matrix $X$.
2. **Pipeline Scope:** The scikit-learn `ColumnTransformer` is fitted exclusively on training sets within cross-validation folds and holdout splits; unseen data is never exposed.
3. **Duplicate Verification:** The dataset was verified to have **0 duplicate records** across the entire 240 rows and across numerical attributes.
4. **Astrophysical Cluster Separation:** Inspection of the physical distributions reveals that Absolute Magnitude ($M_v$) and Radius ($R/R_\odot$) exhibit virtually disjoint numerical bands across the 6 classes (e.g. Brown Dwarfs: $M_v > 16$; Red Dwarfs: $M_v \in [10, 15], R > 0.09$; White Dwarfs: $M_v \in [10, 15], R < 0.02$; Main Sequence: $M_v \in [-4.7, 6.5]$; Supergiants: $M_v \in [-7.5, -5.2]$; Hypergiants: $M_v < -7.5$). Under Stefan-Boltzmann radiation physics, the classes occupy distinct physical regions in $(M_v, R)$ space, enabling decision tree ensembles to achieve near-perfect or perfect generalization without overfitting.
5. **Cross-Validation Defense:** Stratified 10-Fold Cross-Validation achieves **100.00% ($\pm 0.00\%$)** across all folds, and holdout test accuracy across 10 distinct random seeds averages **99.79%** ($97.92\%$ to $100.00\%$).

### Comparative Benchmark
Evaluated on identical stratified 80/20 test split (48 test stars):
* **Random Forest (Primary):** Accuracy: **100.0%** | Macro F1: **100.0%** | 10-Fold CV: **100.0%**
* **Decision Tree (Baseline):** Accuracy: **100.0%** | Macro F1: **100.0%**
* **Logistic Regression (Linear):** Accuracy: **100.0%** | Macro F1: **100.0%**

---

## 4. Key Application Features

1. **Dashboard:** High-level summary metrics, dataset statistics, star class overviews, and quick action links.
2. **Dataset Explorer:**
   - Numerical feature statistics table (Mean, Std Dev, Min, Max, Quartiles).
   - Class distribution chart verifying uniform class balance.
   - Interactive, searchable, and paginated data table showing all 240 stellar records.
3. **Model Training & Diagnostics:**
   - Interactive **Train Model** button executing the full scikit-learn pipeline dynamically.
   - Dynamic metrics: Accuracy, Precision, Recall, F1-Score.
   - Full **6x6 Confusion Matrix** with class labels and color-coded error/success cells.
   - Horizontal **Feature Importance** chart showing relative contributions (Absolute Magnitude & Radius leading the ranking).
   - Algorithm comparison benchmark table.
4. **Star Prediction with Presentation Demo Mode:**
   - Interactive parameter input form with validation.
   - **1-Click Demo Mode Presets:** Buttons for all 6 classes pre-loaded with authentic catalog stars for immediate demonstration during viva/evaluation.
   - Prominent predicted star type result card with taxonomic badges.
   - **Prediction Confidence Breakdown:** Real horizontal probability bars for all 6 star classes directly from `predict_proba()`.
   - **"Why this prediction?"** section explaining Hertzsprung-Russell positioning and ensemble consensus.
5. **Astrophysical Analytics Dashboard:**
   - **Chart 1:** Star Type Distribution (Bar chart).
   - **Chart 2:** Temperature vs Luminosity (Logarithmic scatter plot grouped by star type).
   - **Chart 3:** Temperature vs Radius (Logarithmic scatter plot grouped by star type).
   - **Chart 4:** Hertzsprung-Russell Diagram (Absolute Magnitude vs Temperature with inverted Y-axis).
   - **Chart 5:** Feature Importance bar chart.
6. **About Page:** Comprehensive documentation on stellar physics, Harvard spectral classification sequence, and machine learning formulation.

---

## 5. Project Structure

```
ML Project/
│
├── app.py                      # Flask application and REST API endpoints
├── requirements.txt            # Python dependencies
├── README.md                   # Complete documentation
│
├── data/
│   └── stars.csv               # 240 stellar records (Kaggle dataset)
│
├── models/
│   └── star_classifier.joblib  # Persisted trained pipeline & metadata
│
├── ml/
│   ├── __init__.py             # Star classes and astrophysical constants
│   ├── preprocessing.py        # Cleaning, encoders, scalers, and ColumnTransformer
│   ├── train.py                # Model training, split, evaluation, and persistence
│   └── predict.py              # Validation, inference, probabilities, and explanations
│
├── templates/
│   ├── base.html               # Shared HTML layout with space theme and navigation
│   ├── index.html              # Dashboard landing page
│   ├── dataset.html            # Dataset explorer, statistics, and record table
│   ├── training.html           # Model training, confusion matrix, and feature weights
│   ├── prediction.html         # Interactive star prediction and demo presets
│   ├── analytics.html          # 5 interactive astrophysical & ML charts
│   └── about.html              # Scientific theory and algorithm explanation
│
├── static/
│   ├── css/
│   │   └── style.css           # Modern astronomy + AI dark theme styling
│   └── js/
│       ├── main.js             # Background starfield canvas animation & toasts
│       ├── dataset.js          # Dataset table pagination and distribution chart
│       ├── training.js         # Retraining trigger, confusion matrix, and metrics
│       ├── prediction.js       # Real-time inference and demo presets
│       └── analytics.js        # Chart.js rendering for HR diagrams and scatter plots
│
└── tests/
    └── test_pipeline.py        # Automated test suite (7 tests)
```

---

## 6. Installation & Quick Start

### Prerequisites
* Python 3.9+ (Installed on Windows, macOS, or Linux)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Application
```bash
python app.py
```

### Step 3: Open in Browser
Visit:
```
http://127.0.0.1:5000
```

---

## 7. Automated Testing

Run the full end-to-end unit and integration test suite:
```bash
python -m unittest tests/test_pipeline.py
```
This tests:
1. Dataset shape (240 rows), class balance (40 each), and zero null values.
2. Preprocessing transformations and string sanitizers.
3. Training pipeline execution and joblib file persistence.
4. Model inference accuracy across all 6 demo presets.
5. Error handling on invalid/out-of-range numerical parameters.
6. HTTP 200 responses on all web pages.
7. Valid JSON payload responses on all REST API endpoints.

---

## 8. Step-by-Step Presentation Walkthrough for Evaluators

Follow this exact walkthrough during project demonstration:

```text
1. Open Dashboard (http://127.0.0.1:5000)
   ├── Highlight project title: "STAR PREDICTIVE ANALYSIS"
   ├── Show 4 summary cards: 240 Records, 6 Features, 6 Classes, 100% Accuracy.
   └── Explain the 6 star classes (Brown Dwarf to Hypergiant).

2. Navigate to "Dataset"
   ├── Show the class distribution chart (balanced 40 each).
   ├── Review the numerical feature statistics table.
   └── Search and filter rows in the 240-record catalog table.

3. Navigate to "Model Training"
   ├── Walk through the 6-stage lifecycle stepper.
   ├── Click "Train Model" live to demonstrate real-time pipeline execution.
   ├── Inspect the 6x6 Confusion Matrix (diagonal cells represent 100% correct test predictions).
   └── Review the Feature Importance ranking (Absolute Magnitude & Radius dominate).

4. Navigate to "Prediction"
   ├── Click "Presentation Demo Mode: White Dwarf" to auto-populate test values.
   ├── Click "PREDICT STAR TYPE".
   ├── Show the result: "White Dwarf" with ~99% confidence.
   ├── Show the probability breakdown bars across all 6 classes.
   ├── Review "Why this prediction?" explaining HR position & model reasoning.
   └── Repeat with "Hypergiant" or "Main Sequence" to prove generalization.

5. Navigate to "Analytics"
   ├── Show Chart 4: The classic Hertzsprung-Russell Diagram.
   └── Explain how the model separates dwarfs, main-sequence stars, and supergiants.
```

---

## 9. Academic Citations

* **Dataset:** Deepraj Baidya, *Star Dataset to Predict Star Type*, Kaggle Datasets (2020).
* **Taxonomy:** Morgan, W. W., Keenan, P. C., & Kellman, E. (1943). *An atlas of stellar spectra, with an outline of spectral classification*.
* **Scikit-Learn:** Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
