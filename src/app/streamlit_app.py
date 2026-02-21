import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================
# 1. CARGAR MODELO SERIALIZADO
# ============================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# FIX 1: Cargamos el modelo XGBoost ganador, no el baseline
modelo_path = os.path.join(BASE_DIR, "../../models/best_model_XGBoost.pkl")
modelo = joblib.load(modelo_path)

# ============================================
# 2. CONFIGURACIÓN DE LA PÁGINA
# ============================================

st.set_page_config(
    page_title="Predicción de Días en Refugio 🐾",
    page_icon="🐾",
    layout="centered"
)

st.title("Predicción de Días en Refugio 🐾")
st.markdown(
    "Completa los datos del animal y obtén una estimación de "
    "cuántos días podría permanecer en el refugio antes de ser adoptado."
)

# ============================================
# 3. OPCIONES VÁLIDAS
# FIX 2: Todos los valores coinciden exactamente con los del modelo
# ============================================

# AnimalType: el modelo solo conoce Dog y Cat (eliminamos Other/Bird/Livestock)
ANIMAL_TYPE_OPCIONES = {
    "Perro": "Dog",
    "Gato":  "Cat"
}

# Sex: el modelo conoce 4 valores exactos
SEX_OPCIONES = {
    "Macho entero (no esterilizado)":    "Intact Male",
    "Hembra entera (no esterilizada)":   "Intact Female",
    "Macho castrado":                    "Neutered Male",
    "Hembra esterilizada":               "Spayed Female"
}

# IntakeType: 4 categorías válidas tras la limpieza
INTAKE_TYPE_OPCIONES = {
    "Callejero (Stray)":          "Stray",
    "Entregado por dueño":        "Owner Surrender",
    "Asistencia pública":         "Public Assist",
    "Abandonado":                 "Abandoned"
}

# IntakeCondition: FIX 4 - eliminamos "Other" y "Feral" que fueron borrados
INTAKE_CONDITION_OPCIONES = {
    "Normal":       "Normal",
    "Lesionado":    "Injured",
    "Enfermo":      "Sick",
    "Embarazada":   "Pregnant",
    "Médico":       "Medical",
    "Comportamiento": "Behavior",
    "Lactante":     "Nursing",
    "Anciano":      "Aged"
}

# breed_type: solo 2 categorías tras la limpieza
BREED_TYPE_OPCIONES = {
    "Raza pura":  "purebred",
    "Mestizo":    "mix"
}

# Top 25 razas del modelo + Other
BREED_GROUPED_OPCIONES = [
    "Other",
    "Domestic Shorthair Mix",
    "Pit Bull Mix",
    "Labrador Retriever Mix",
    "Domestic Shorthair",
    "Chihuahua Shorthair Mix",
    "German Shepherd Mix",
    "Domestic Medium Hair Mix",
    "Australian Cattle Dog Mix",
    "Domestic Longhair Mix",
    "Siamese Mix",
    "Pit Bull",
    "Border Collie Mix",
    "Dachshund Mix",
    "Boxer Mix",
    "Labrador Retriever",
    "German Shepherd",
    "Chihuahua Shorthair",
    "Staffordshire Mix",
    "Catahoula Mix",
    "Domestic Medium Hair",
    "Siberian Husky Mix",
    "Pointer Mix",
    "Australian Shepherd Mix",
    "Beagle Mix",
    "Miniature Poodle Mix"
]

# Color: 3 categorías exactas del modelo
COLOR_GROUPED_OPCIONES = {
    "Un solo color (Monocolor)":  "Monocolor",
    "Dos colores (Bicolor)":      "Bicolor",
    "Tres colores (Tricolor)":    "Tricolor"
}

# ============================================
# 4. FORMULARIO DE ENTRADA
# FIX 3: Eliminamos Size y weight_kg que no están en el modelo
# ============================================

with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        animal_type_label = st.selectbox(
            "🐾 Tipo de animal",
            list(ANIMAL_TYPE_OPCIONES.keys())
        )
        sex_label = st.selectbox(
            "⚧ Sexo / Esterilización",
            list(SEX_OPCIONES.keys())
        )
        age_months = st.number_input(
            "📅 Edad en meses",
            min_value=0, max_value=240, value=12, step=1,
            help="Introduce la edad aproximada en meses"
        )
        intake_type_label = st.selectbox(
            "🚪 Tipo de ingreso",
            list(INTAKE_TYPE_OPCIONES.keys())
        )
        intake_condition_label = st.selectbox(
            "🏥 Condición al ingreso",
            list(INTAKE_CONDITION_OPCIONES.keys())
        )

    with col2:
        breed_type_label = st.selectbox(
            "🧬 Tipo de raza",
            list(BREED_TYPE_OPCIONES.keys())
        )
        breed_grouped = st.selectbox(
            "🐕 Raza",
            BREED_GROUPED_OPCIONES,
            help="Selecciona la raza más cercana. Si no está en la lista, selecciona 'Other'."
        )
        color_label = st.selectbox(
            "🎨 Coloración del pelaje",
            list(COLOR_GROUPED_OPCIONES.keys())
        )

    st.markdown("---")
    submit_button = st.form_submit_button(
        label="🔍 Predecir días en refugio",
        use_container_width=True
    )

# ============================================
# 5. FUNCIÓN: CALCULAR AgeGroup DESDE MESES
# ============================================

def calcular_age_group(meses):
    if meses < 6:
        return "Cachorro (<6m)"
    elif meses < 12:
        return "Joven (6m-1a)"
    elif meses < 36:
        return "Adulto joven (1-3a)"
    elif meses < 84:
        return "Adulto (3-7a)"
    else:
        return "Senior (>7a)"

# ============================================
# 6. PREDICCIÓN
# ============================================

if submit_button:

    # Mapear etiquetas en español a valores que el modelo conoce
    animal_type_val      = ANIMAL_TYPE_OPCIONES[animal_type_label]
    sex_val              = SEX_OPCIONES[sex_label]
    intake_type_val      = INTAKE_TYPE_OPCIONES[intake_type_label]
    intake_condition_val = INTAKE_CONDITION_OPCIONES[intake_condition_label]
    breed_type_val       = BREED_TYPE_OPCIONES[breed_type_label]
    color_val            = COLOR_GROUPED_OPCIONES[color_label]

    age_days  = int(age_months * 30.44)   # meses → días
    age_group = calcular_age_group(age_months)

    # DataFrame con exactamente las mismas columnas y valores que el modelo espera
    input_df = pd.DataFrame({
        "AnimalType":       [animal_type_val],
        "Sex":              [sex_val],
        "IntakeType":       [intake_type_val],
        "IntakeCondition":  [intake_condition_val],
        "AgeInDays":        [age_days],
        "AgeGroup":         [age_group],
        "breed_type":       [breed_type_val],
        "Breed_grouped":    [breed_grouped],
        "Color_grouped":    [color_val]
    })

    # FIX 3: Aplicar np.expm1() para convertir de escala log a días reales
    pred_log  = modelo.predict(input_df)[0]
    dias_pred = float(np.expm1(pred_log))
    dias_pred = max(1.0, round(dias_pred, 1))   # mínimo 1 día

    # ============================================
    # 7. MOSTRAR RESULTADO
    # ============================================

    st.markdown("---")
    st.subheader("📊 Resultado de la predicción")

    # Color semáforo según días
    if dias_pred <= 14:
        color_msg = "🟢 Alta probabilidad de adopción rápida"
        st.success(color_msg)
    elif dias_pred <= 45:
        color_msg = "🟡 Tiempo de adopción moderado"
        st.warning(color_msg)
    else:
        color_msg = "🔴 Este animal puede necesitar más apoyo para encontrar hogar"
        st.error(color_msg)

    col_res1, col_res2, col_res3 = st.columns(3)
    with col_res1:
        st.metric("Días estimados", f"{dias_pred:.0f} días")
    with col_res2:
        st.metric("Rango mínimo", f"{max(1, dias_pred * 0.6):.0f} días")
    with col_res3:
        st.metric("Rango máximo", f"{dias_pred * 1.4:.0f} días")

    # Detalles del animal introducido
    # FIX Arrow: .astype(str) evita el error de tipos mezclados al transponer
    with st.expander("📋 Ver datos introducidos"):
        st.dataframe(
            input_df.T
                    .rename(columns={0: "Valor"})
                    .astype(str)
        )

    st.info(
        "ℹ️ Esta predicción se basa en un modelo XGBoost entrenado con datos históricos "
        "del Austin Animal Center. El error medio del modelo es de ~30 días. "
        "La estimación es orientativa y puede variar según factores no recogidos en los datos."
    )