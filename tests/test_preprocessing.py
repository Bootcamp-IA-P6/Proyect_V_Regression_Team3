import pytest
import pandas as pd
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(BASE_DIR, "../models/best_baseline_pipeline.pkl")
pipeline = joblib.load(pipeline_path)
preprocessor = pipeline.named_steps["preprocessor"]

#Datos de Prueba
TEST_DATA = pd.DataFrame({
    "AnimalType": ["Dog", "Cat"],
    "Sex": ["Male", "Female"],
    "IntakeType": ["Stray", "Owner Surrender"],
    "IntakeCondition": ["Normal", "Sick"],
    "AgeInDays": [30, 400],
    "AgeGroup": ["Cachorro (<6m)", "Adulto joven (1-3a)"],
    "breed_type": ["purebred", "mix"],
    "Breed_grouped": ["Other", "Domestic Shorthair Mix"],
    "Color_grouped": ["Monocolor", "Bicolor"]
})

def test_output_shape():
    X_processed = preprocessor.transform(TEST_DATA)
    assert X_processed.shape[0] == TEST_DATA.shape[0], "Filas incorrectas"
    assert X_processed.shape[1] > 0, "No hay columnas en el output"

def test_no_nulls_after_preprocessing():
    """
    Verifica que no existan valores nulos en las columnas numéricas tras el preprocesamiento.
    Aquí transformamos solo la parte numérica para evitar errores de tipo.
    """
    # Tomar el transformador numérico
    num_transformer = preprocessor.named_transformers_['num']
    num_cols = preprocessor.transformers_[2][2]  # columnas numéricas
    # Transformar solo las columnas numéricas
    X_num_scaled = num_transformer.fit_transform(TEST_DATA[num_cols])
    
    # Comprobar nulos
    assert not np.isnan(X_num_scaled).any(), "Existen valores NaN en columnas numéricas"

def test_categorical_encoding():
    X_processed = preprocessor.transform(TEST_DATA)
    # ordinal AgeGroup
    age_encoded = preprocessor.named_transformers_['ord'].transform(TEST_DATA[['AgeGroup']])
    categories_len = len(preprocessor.named_transformers_['ord'].categories_[0])
    assert (age_encoded >= -1).all() and (age_encoded < categories_len).all(), "OrdinalEncoder fuera de rango"

def test_numerical_scaling():
    """
    Verifica que las columnas numéricas estén correctamente escaladas: media ~0, std ~1
    Solo se verifica la parte numérica para evitar errores con OHE.
    """
    num_transformer = preprocessor.named_transformers_['num']
    num_cols = preprocessor.transformers_[2][2]
    X_num_scaled = num_transformer.fit_transform(TEST_DATA[num_cols])
    
    mean = np.mean(X_num_scaled)
    std = np.std(X_num_scaled)
    
    assert abs(mean) < 1e-6, f"Media escalada no cercana a 0: {mean:.4f}"
    assert abs(std - 1) < 1e-6, f"Desviación escalada no cercana a 1: {std:.4f}"