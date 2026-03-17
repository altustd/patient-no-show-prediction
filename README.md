# Medical Appointment No-Show Prediction

**Predict whether a patient will miss their appointment** using the famous Kaggle Brazil dataset (110k+ records).

## Goal
Reduce no-shows — a major source of lost revenue and inefficiency in healthcare clinics.

## Key Features
- Strong patient history features (previous no-show rate, days since last appointment)
- Native categorical support in XGBoost (`enable_categorical=True`)
- Clean, reproducible pipeline
- Rich EDA visualizations

## Results (example)
- ROC AUC: ~0.81
- PR AUC: ~0.45
- Top predictors: `prev_no_show_rate`, `wait_days`, `neighbourhood`

## Project Structure
