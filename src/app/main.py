import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# ============================================
# 1. CARGAR MODELO SERIALIZADO
# ============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
modelo_path = os.path.join(BASE_DIR, "../../models/best_baseline_pipeline.pkl")
modelo = joblib.load(modelo_path)

# ============================================
# 2. ARCHIVO CSV PARA FEEDBACK (definido siempre)
# ============================================
feedback_file = os.path.join(BASE_DIR, "feedback_streamlit.csv")

# ============================================
# 3. CONFIGURACIÓN DE LA PÁGINA
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
# 4. FORMULARIO DE ENTRADA
# ============================================
with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        animal_type = st.selectbox("Tipo de animal", ["Perro", "Gato", "Otro"])
        sex = st.selectbox("Sexo", ["Macho", "Hembra"])
        age_months = st.number_input("Edad en meses", min_value=0, max_value=240, value=12, step=1)
        intake_type = st.selectbox("Tipo de ingreso", ["Stray", "Owner Surrender","Public Assist", "Abandoned"])
        intake_condition = st.selectbox("Condición al ingreso", ["Normal", "Injured", "Sick", "Other"])

    with col2:
        breed_type = st.selectbox("Tipo de raza", ["Pura", "Mixta"])
        breed_grouped = st.text_input("Raza", value="", placeholder="Escribe la raza aquí")
        color_grouped = st.selectbox("Color del pelaje", ["Monocolor", "Bicolor", "Tricolor"])
        weight_kg = st.number_input("Peso (kg)", min_value=0.0, max_value=100.0, value=5.0, step=0.1)

    submit_button = st.form_submit_button(label="Predecir")

# ============================================
# 5. CALCULAR SIZE AUTOMÁTICAMENTE
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
# 6. PROCESAR PREDICCIÓN
# ============================================
if submit_button:
    if breed_grouped.strip() == "":
        st.warning("Por favor, ingresa la raza o deja 'Sin especificar'.")
    else:
        input_df = pd.DataFrame({
            "AnimalType": [animal_type],
            "Sex": [sex],
            "IntakeType": [intake_type],
            "IntakeCondition": [intake_condition],
            "AgeInDays": [age_months * 30],
            "AgeGroup": [(
                "Cachorro (<6m)" if age_months < 6 else
                "Joven (6m-1a)" if age_months < 12 else
                "Adulto joven (1-3a)" if age_months < 36 else
                "Adulto (3-7a)" if age_months < 84 else
                "Senior (>7a)"
            )],
            "breed_type": [breed_type],
            "Breed_grouped": [breed_grouped],
            "Color_grouped": [color_grouped],
            "Size": [calcular_size(animal_type, weight_kg)]
        })

        # Predicción
        pred = modelo.predict(input_df)
        dias_pred = pred[0]

        # Colorear según días
        if dias_pred <= 10:
            delta_color = "normal"
        elif dias_pred <= 30:
            delta_color = "warning"
        else:
            delta_color = "inverse"

        # Mostrar resultado
        st.subheader("Predicción de días en refugio")
        st.metric(label="Tiempo estimado", value=f"{dias_pred:.1f} días", delta=f"±5 días aprox.", delta_color=delta_color)
        st.info("Esta predicción se basa en un modelo de regresión entrenado con datos históricos del refugio.\nLa estimación puede variar según el comportamiento del animal y condiciones del refugio.")
        st.caption(f"Rango aproximado: {max(0,dias_pred-5):.1f} - {dias_pred+5:.1f} días")

        # ============================================
        # 7. FORMULARIO DE FEEDBACK
        # ============================================
        st.markdown("---")
        st.subheader("¿La predicción fue correcta?")
        with st.form(key="feedback_form"):
            feedback_option = st.radio("Selecciona una opción", ["Sí", "Aproximada", "No"])
            real_value = st.number_input("Valor real observado (opcional)", min_value=0.0, max_value=365.0, step=1.0, value=0.0)
            feedback_submit = st.form_submit_button(label="Enviar Feedback")

            # 🔹 mover guardado dentro del mismo form para que funcione
            if feedback_submit:
                feedback_dict = {
                    "timestamp": datetime.now().isoformat(),
                    "AnimalType": animal_type,
                    "Sex": sex,
                    "AgeMonths": age_months,
                    "IntakeType": intake_type,
                    "IntakeCondition": intake_condition,
                    "breed_type": breed_type,
                    "Breed_grouped": breed_grouped,
                    "Color_grouped": color_grouped,
                    "Size": calcular_size(animal_type, weight_kg),
                    "PredictedDays": dias_pred,
                    "Feedback": feedback_option,
                    "RealValue": real_value if real_value > 0 else np.nan
                }

                df_feedback = pd.DataFrame([feedback_dict])
                if os.path.exists(feedback_file):
                    df_feedback.to_csv(feedback_file, mode="a", header=False, index=False)
                else:
                    df_feedback.to_csv(feedback_file, index=False)

                st.success("¡Gracias por tu feedback! ✅")

        # ============================================
        # 8. ESTADÍSTICAS Y GRÁFICOS DE FEEDBACK
        # ============================================
        if os.path.exists(feedback_file):
            df_fb = pd.read_csv(feedback_file)
            st.markdown("---")
            st.subheader("📊 Estadísticas de Feedback")

            total = len(df_fb)
            counts = df_fb["Feedback"].value_counts(normalize=True) * 100

            st.write(f"Total de respuestas: {total}")
            st.write(f"✅ Sí: {counts.get('Sí',0):.1f}%")
            st.write(f"⚠️ Aproximada: {counts.get('Aproximada',0):.1f}%")
            st.write(f"❌ No: {counts.get('No',0):.1f}%")

            if df_fb["RealValue"].notna().sum() > 0:
                mean_real = df_fb["RealValue"].mean()
                st.write(f"Promedio de días reales reportados: {mean_real:.1f} días")

            # Pie chart del feedback
            fig1, ax1 = plt.subplots()
            feedback_counts = df_fb["Feedback"].value_counts()
            ax1.pie(feedback_counts, labels=feedback_counts.index, autopct="%1.1f%%", startangle=90)
            ax1.axis("equal")
            st.pyplot(fig1)

            # Histograma de valores reales
            if df_fb["RealValue"].notna().sum() > 0:
                fig2, ax2 = plt.subplots()
                ax2.hist(df_fb["RealValue"].dropna(), bins=10, color="skyblue", edgecolor="black")
                ax2.set_xlabel("Días reales reportados")
                ax2.set_ylabel("Cantidad")
                ax2.set_title("Distribución de días reales reportados")
                st.pyplot(fig2)
                