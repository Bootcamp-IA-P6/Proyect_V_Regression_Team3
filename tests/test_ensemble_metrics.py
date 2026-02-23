import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


MODEL_PATH = "models/best_model_XGBoost.pkl"
DATA_PATH = "data/pet_adoption_model.csv"
TARGET_COL = "TimeInShelterDays_log"


def test_xgboost_model_metrics():
    """
    Performance validation test for the selected XGBoost ensemble model.

    ------------------------------------------------------------
    📊 Observed evaluation metrics (final notebook results):

        Validation R²       ≈ 0.2537
        Test R²             ≈ 0.2377
        Test RMSE (days)    ≈ 61.2 days

    ------------------------------------------------------------
    📌 Dataset characteristics:

    - High variance in adoption duration.
    - Target variable is log-transformed.
    - Limited predictive signal in available structured features.
    - Real-world adoption time is inherently noisy.

    ------------------------------------------------------------
    🎯 Threshold justification:

    Since the model achieved Test R² ≈ 0.2377,
    we define a conservative lower bound:

        R² >= 0.20

    Since observed RMSE ≈ 61 days,
    we allow small fluctuation margin:

        RMSE <= 70 days

    Observed train-test R² gap ≈ 0.0614 (6.14%).

    Given:
    - Natural dataset variance
    - XGBoost model complexity
    - Minor expected fluctuation between splits

    We define acceptable overfitting threshold:

        Overfitting gap < 0.07  (7%)

    This ensures the model does not exhibit severe overfitting,
    while allowing realistic variance across splits.

    ------------------------------------------------------------
    """

    # -----------------------------
    # Load dataset
    # -----------------------------
    df = pd.read_csv(DATA_PATH)

    assert TARGET_COL in df.columns, \
        f"{TARGET_COL} not found in dataset."

    X = df.drop(columns=[TARGET_COL])
    y_log = df[TARGET_COL]  # Already log-transformed target

    # -----------------------------
    # Recreate exact split (must match training notebook)
    # -----------------------------
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y_log, test_size=0.15, random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.1765, random_state=42
    )

    # -----------------------------
    # Load trained model
    # -----------------------------
    model = joblib.load(MODEL_PATH)

    # -----------------------------
    # Generate predictions
    # -----------------------------
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # -----------------------------
    # Compute metrics (log scale)
    # -----------------------------
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    # -----------------------------
    # Convert back to original scale (days)
    # -----------------------------
    y_test_days = np.expm1(y_test)
    y_test_pred_days = np.expm1(y_test_pred)

    rmse_test_days = np.sqrt(
        mean_squared_error(y_test_days, y_test_pred_days)
    )

    # -----------------------------
    # Overfitting calculation
    # -----------------------------
    overfitting_gap = abs(r2_train - r2_test)

    # -----------------------------
    # Assertions
    # -----------------------------
    assert r2_test >= 0.20, \
        f"R² too low: {r2_test:.4f}"

    assert rmse_test_days <= 70, \
        f"RMSE too high: {rmse_test_days:.2f}"

    assert overfitting_gap < 0.07, \
        f"Overfitting gap too high: {overfitting_gap:.4f}"