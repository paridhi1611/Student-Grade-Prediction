# Student Grade Prediction using Machine Learning

A beginner-friendly predictive analytics project that estimates a student's final mathematics grade using the UCI Student Performance dataset and Linear Regression.

## Dataset
- **Source:** UCI Machine Learning Repository, Student Performance (ID 320)
- **Official page:** https://archive.ics.uci.edu/dataset/320/student%2Bperformance
- **Records:** 649 student records across two subject datasets; the code uses the UCI feature/target table returned by `ucimlrepo`.
- **License:** CC BY 4.0. Credit the original dataset creators in your report.
- The data was collected from school reports and questionnaires at two Portuguese schools. It is real collected data, not a dataset generated for this project.

## Prediction target
`G3` — final mathematics grade on a 0–20 scale.

## Important design choice
The model intentionally does **not** use `G1` or `G2` (first- and second-period grades). Those grades strongly correlate with G3 and using them can make prediction look easier than it is. This version instead studies non-grade features such as study time, absences, past failures and family/school context.

## Model and evaluation
- Train/test split: 80% / 20%, random state 42
- Model: Linear Regression
- Metrics: MAE, RMSE and R²
- Preprocessing: median imputation and scaling for numeric features; most-frequent imputation and one-hot encoding for categorical features.

## Run on macOS / Windows
1. Install Python 3.10 or newer.
2. Open Terminal in this folder.
3. Create and activate a virtual environment (recommended).
4. Install dependencies:
   ```bash
   python3 -m pip install -r requirements.txt
   ```
5. Run:
   ```bash
   python3 student_grade_prediction.py
   ```
The script downloads the dataset through the UCI repository package, so an internet connection is required on first run.

## Generated outputs
The script creates:
- `outputs/metrics.json` — actual metrics from your run
- `outputs/predictions.csv` — actual and predicted grades for the test set
- `outputs/actual_vs_predicted.png` — prediction plot
- `outputs/residual_distribution.png` — error distribution plot

## Limitations and responsible use
The dataset is small and comes from two schools in Portugal. It should not be assumed to represent students in India or other education systems. A prediction is not a judgement of a student's ability and should never be used alone to penalize or label a student.

## Dataset citation
Cortez, P. (2008). *Student Performance* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T
