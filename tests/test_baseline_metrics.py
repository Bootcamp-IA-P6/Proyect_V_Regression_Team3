# tests/test_baseline_metrics.py
"""
Tests de validación para el modelo baseline de regresión (LinearRegression) 
del proyecto de predicción de días en refugio.

Se verifica:
- Métricas de desempeño mínimas en test set
- Diferencia de overfitting entre train y validation
- Integridad de predicciones (shape, sin NaN, valores positivos)
- Serialización y carga del pipeline

Thresholds definidos según análisis del baseline:
- R² mínimo en test: 0.1 (baseline simple, esperamos que capture algo de señal)
- Overfitting gap máximo: 0.10 (diferencia R² train - val)
- RMSE máximo (log scale): 1.5
"""

import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

# ==============================
# Configuración
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../models/best_baseline_pipeline.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "../data/pet_adoption_model.csv")

# ==============================
# Fixture: cargar datos y modelo
# ==============================
@pytest.fixture(scope="module")
def load_data_model():
    # Cargar dataframe
    df = pd.read_csv(DATA_PATH)

    FEATURES = [
        'AnimalType',
        'Sex',
        'IntakeType',
        'IntakeCondition',
        'AgeInDays',
        'AgeGroup',
        'breed_type',
        'Breed_grouped',
        'Color_grouped'
    ]
    TARGET = 'TimeInShelterDays_log'

    X = df[FEATURES]
    y = df[TARGET]

    # Cargar pipeline serializado
    pipeline = joblib.load(MODEL_PATH)

    return X, y, pipeline

# ==============================
# 1. Test: métricas en test
# ==============================
def test_baseline_model_metrics(load_data_model):
    X, y, pipeline = load_data_model

    # Split simple: 70% train, 30% test para evaluación
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Entrenar pipeline
    pipeline.fit(X_train, y_train)
    y_train_pred = pipeline.predict(X_train)
    y_test_pred  = pipeline.predict(X_test)

    # Métricas
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2  = r2_score(y_test, y_test_pred)
    overfit_gap = train_r2 - test_r2
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    # Debug print (opcional)
    print(f"Train R²: {train_r2:.4f}")
    print(f"Test  R²: {test_r2:.4f}")
    print(f"Overfitting gap: {overfit_gap:.4f}")
    print(f"Test RMSE (log scale): {test_rmse:.4f}")

    # Asserts
    assert test_r2 >= 0.1, f"R² en test demasiado bajo: {test_r2:.4f}"
    assert overfit_gap <= 0.10, f"Overfitting gap demasiado alto: {overfit_gap:.4f}"
    assert test_rmse <= 1.5, f"RMSE en test demasiado alto: {test_rmse:.4f}"

# ==============================
# 2. Test: predicciones consistentes
# ==============================
def test_prediction_consistency(load_data_model):
    X, y, pipeline = load_data_model

    # Tomar una muestra
    X_sample = X.head(10)
    y_pred = pipeline.predict(X_sample)

    # Shape consistente
    assert y_pred.shape[0] == X_sample.shape[0], "Shape de predicciones no coincide con X"
    # Valores numéricos
    assert np.issubdtype(y_pred.dtype, np.floating), "Predicciones no son floats"
    # No NaN ni inf
    assert not np.any(np.isnan(y_pred)), "Predicciones contienen NaN"
    assert not np.any(np.isinf(y_pred)), "Predicciones contienen inf"
    # Valores positivos tras expm1 (escala días reales)
    y_real_days = np.expm1(y_pred)
    assert np.all(y_real_days > 0), "Predicciones negativas tras expm1()"

# ==============================
# 3. Test: pipeline serialización
# ==============================
def test_pipeline_serialization():
    # Intentar cargar modelo
    pipeline = joblib.load(MODEL_PATH)
    assert pipeline is not None, "Pipeline no pudo cargarse"
    # Comprobar que tenga steps
    assert hasattr(pipeline, "named_steps"), "Pipeline no tiene named_steps"
    assert "model" in pipeline.named_steps, "Pipeline no contiene step 'model'"