import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================
# Cargar modelo y preprocesador (solo una vez)
# ============================================
import os

# Obtener ruta absoluta desde este script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, "../../models/baseline_best_model.joblib")

modelo = joblib.load(modelo_path)


# ============================================
# Configuración de la página
# ============================================
st.set_page_config(
    page_title="Predicción de Días en Refugio 🏠",
    page_icon="🐾",
    layout="centered"
)

st.title("Predicción de Días en Refugio 🏠")
st.markdown(
    """
    Llena los datos del animal y obtén una estimación de cuántos días podría permanecer en el refugio.
    """
)

# ============================================
# Formulario de entrada
# ============================================
with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input(
            "Edad (meses)", min_value=0, max_value=240, value=12, step=1
        )
        animal_type = st.selectbox(
            "Tipo de animal",
            ["Perro", "Gato", "Conejo", "Ave", "Otro"]
        )
    
    with col2:
        weight = st.number_input(
            "Peso (kg)", min_value=0.0, max_value=100.0, value=5.0, step=0.1
        )
        breed = st.text_input(
            "Raza", value="", placeholder="Escribe la raza aquí"
        )

    submit_button = st.form_submit_button(label="Predecir")

# ============================================
# Procesar predicción
# ============================================
if submit_button:
    # Validación simple
    if breed.strip() == "":
        st.warning("Por favor, ingresa una raza o selecciona 'Otro'.")
    else:
        # Crear dataframe para el modelo
        input_df = pd.DataFrame({
            'AgeMonths': [age],
            'WeightKg': [weight],
            'PetType': [animal_type],
            'Breed': [breed],
            'Color': ['Otro'],           # Default, se puede extender
            'Size': ['Mediano'],         # Default, se puede extender
            'AdoptionFee': [50.0],       # Default
            'Vaccinated': [1],            # Default
            'HealthCondition': [1],       # Default
            'PreviousOwner': [0]          # Default
        })

        # Predicción
        pred = modelo.predict(input_df)
        dias_pred = pred[0]

        # Mostrar resultado resaltado
        st.subheader("Predicción de días en refugio")
        st.metric(label="Tiempo estimado", value=f"{dias_pred:.1f} días")

        # Información adicional
        st.info(
            "Esta predicción se basa en un modelo de regresión entrenado con datos históricos del refugio.\n"
            "La estimación puede variar según el comportamiento del animal y condiciones del refugio."
        )

        # Opcional: rango de confianza ±5 días
        st.caption(f"Rango aproximado: {max(0,dias_pred-5):.1f} - {dias_pred+5:.1f} días")

# ============================================