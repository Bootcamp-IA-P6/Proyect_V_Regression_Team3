import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

# ============================================
# 1. CARGAR MODELO SERIALIZADO
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, "../../models/best_baseline_pipeline.pkl")
modelo = joblib.load(modelo_path)

# ============================================
# 2. CONFIGURACIÓN DE LA PÁGINA
# ============================================

st.set_page_config(
    page_title="Predicción de Días en Refugio 🏠",
    page_icon="🐾",
    layout="centered"
)

st.title("Predicción de Días en Refugio 🏠")
st.markdown(
    "Completa los datos del animal y obtén una estimación de cuántos días podría permanecer en el refugio."
)

# ============================================
# 3. FORMULARIO DE ENTRADA
# ============================================

with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        animal_type = st.selectbox(
            "Tipo de animal",
            ["Perro", "Gato", "Otro"]
        )
        sex = st.selectbox(
            "Sexo",
            ["Macho", "Hembra"]
        )
        age_months = st.number_input(
            "Edad en meses",
            min_value=0, max_value=240, value=12, step=1
        )
        intake_type = st.selectbox(
            "Tipo de ingreso",
            ["Stray", "Owner Surrender","Public Assist", "Abandoned"]
        )
        intake_condition = st.selectbox(
            "Condición al ingreso",
            ["Normal", "Injured", "Sick", "Other"]
        )

    with col2:
        breed_type = st.selectbox(
            "Tipo de raza",
            ["Pura", "Mixta"]
        )
        breed_grouped = st.text_input(
            "Raza",
            value="",
            placeholder="Escribe la raza aquí"
        )
        color_grouped = st.multiselect(
            "Color(es)",
            ["Negro", "Blanco", "Marrón", "Gris", "Otro"],
            default=["Otro"]
        )
        weight_kg = st.number_input(
            "Peso (kg)",
            min_value=0.0, max_value=100.0, value=5.0, step=0.1
        )

    submit_button = st.form_submit_button(label="Predecir")

# ============================================
# 4. CALCULAR SIZE AUTOMÁTICAMENTE
# ============================================

def calcular_size(especie, peso):
    if especie.lower() == "perro":
        if peso < 10:
            return "Pequeño"
        elif peso < 25:
            return "Mediano"
        else:
            return "Grande"
    elif especie.lower() == "gato":
        return "Mediano"
    else:
        return "Mediano"

# ============================================
# 5. PROCESAR PREDICCIÓN
# ============================================

if submit_button:
    if breed_grouped.strip() == "":
        st.warning("Por favor, ingresa la raza o deja 'Sin especificar'.")
    else:
        # Crear dataframe para el pipeline
        input_df = pd.DataFrame({
            "AnimalType": [animal_type],
            "Sex": [sex],
            "IntakeType": [intake_type],
            "IntakeCondition": [intake_condition],
            "AgeInDays": [age_months * 30],  # convertir meses a días
            "AgeGroup": [(
                "Cachorro (<6m)" if age_months < 6 else
                "Joven (6m-1a)" if age_months < 12 else
                "Adulto joven (1-3a)" if age_months < 36 else
                "Adulto (3-7a)" if age_months < 84 else
                "Senior (>7a)"
            )],
            "breed_type": [breed_type],
            "Breed_grouped": [breed_grouped],
            "Color_grouped": [", ".join(color_grouped)],
            "Size": [calcular_size(animal_type, weight_kg)]
        })

        # Predicción
        pred = modelo.predict(input_df)
        dias_pred = pred[0]

        # ============================================
        # 6. COLOREAR PREDICCIÓN SEGÚN DÍAS
        # ============================================
        if dias_pred <= 10:
            delta_color = "normal"  # verde
        elif dias_pred <= 30:
            delta_color = "warning"  # amarillo
        else:
            delta_color = "inverse"  # rojo

        # Mostrar resultado
        st.subheader("Predicción de días en refugio")
        st.metric(
            label="Tiempo estimado",
            value=f"{dias_pred:.1f} días",
            delta=f"±5 días aprox.",
            delta_color=delta_color
        )

        st.info(
            "Esta predicción se basa en un modelo de regresión entrenado con datos históricos del refugio.\n"
            "La estimación puede variar según el comportamiento del animal y condiciones del refugio."
        )

        st.caption(f"Rango aproximado: {max(0,dias_pred-5):.1f} - {dias_pred+5:.1f} días")