import os
import joblib


def test_model_file_exists():
    """Ensure the trained model file exists."""
    assert os.path.exists("models/best_model_XGBoost.pkl"), \
        "Model file not found."


def test_model_loads_successfully():
    """Ensure the model can be loaded without errors."""
    model = joblib.load("models/best_model_XGBoost.pkl")
    assert model is not None, "Model failed to load."


if __name__ == "__main__":
    test_model_file_exists()
    test_model_loads_successfully()

