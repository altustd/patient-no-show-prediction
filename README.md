# Medical Appointment No-Show Prediction

**Goal**: Predict whether a patient will miss their scheduled medical appointment (`target = 1` = No-show).

This project uses the classic [Kaggle V2 Medical Appointment No Shows dataset](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (~110k appointments) and focuses on **realistic evaluation** using time-based splitting to avoid leakage.

### Version History

| Version | Key Change                              | ROC AUC | PR AUC  | Notes |
|---------|-----------------------------------------|---------|---------|-------|
| v3.3    | Random stratified split                 | ~0.76   | ~0.43   | Leaky (future data in training) |
| v4.0    | Native XGBoost categoricals + patient history features | ~0.76   | ~0.436  | Still random split |
| **v4.1** | **Chronological (time-based) split** on `scheduled_day` | **0.7827** | **0.3161** | **Most realistic evaluation** — no future leakage |

### Current Performance (v4.1 — Time-based split)

- **ROC AUC**: 0.7827
- **PR AUC**: 0.3161
- **Accuracy**: 0.79
- **No-show class (1)**: Precision = 0.29, Recall = 0.51, F1 = 0.37

The drop in PR AUC compared to previous versions is **expected and desired** — it reflects real-world conditions where we can only use past data to predict future appointments.

### Features

- **Core**: Age, wait days, scheduled hour, SMS received, clinical conditions
- **Engineered**:
  - Patient history (previous no-show rate, days since last appointment, cumulative no-shows)
  - Temporal: same-day flag, appointment day-of-week, weekend flag
  - Neighborhood grouping (rare categories → "Other")

### Tech Stack

- Python 3
- XGBoost (native categorical support)
- pandas, scikit-learn, matplotlib/seaborn

### How to Run

1. Clone the repo
2. Open `v4.1.ipynb` in Jupyter / VS Code / JupyterLab
3. Run all cells (data is loaded from `KaggleV2-May-2016.csv`)

**Note**: The notebook now uses a proper **chronological split** (80% earliest appointments for training, 20% most recent for testing).

### Next Steps (planned)

- Stronger wait-time binning + interactions
- Hyperparameter tuning with Optuna
- CatBoost / LightGBM comparison
- Probability calibration + optimal threshold

---

**Repository**: [patient-no-show-prediction](https://github.com/altustd/patient-no-show-prediction)
