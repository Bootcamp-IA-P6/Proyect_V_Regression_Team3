import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/best_model_XGBoost.pkl"
DATA_PATH = "data/pet_adoption_model.csv"
TARGET_COL = "TimeInShelterDays_log"


def test_prediction_output_shape():
    """Ensure prediction returns correct number of outputs."""
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL]).head(10)

    model = joblib.load(MODEL_PATH)

    preds = model.predict(X)

    assert len(preds) == 10, \
        "Prediction output size mismatch."


def test_no_nan_predictions():
    """Ensure model does not produce NaN predictions."""
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL]).head(20)

    model = joblib.load(MODEL_PATH)

    preds = model.predict(X)

    assert not np.isnan(preds).any(), \
        "Model produced NaN predictions."


def test_no_negative_days_after_inverse_transform():
    """Ensure predicted adoption days are not negative."""
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=[TARGET_COL]).head(20)

    model = joblib.load(MODEL_PATH)

    preds_log = model.predict(X)
    preds_days = np.expm1(preds_log)

    assert (preds_days >= 0).all(), \
        "Model predicted negative adoption days."