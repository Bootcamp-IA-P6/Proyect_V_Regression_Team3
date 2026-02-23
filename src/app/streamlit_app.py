import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys

# ============================================
# 0. PATH PARA IMPORTS RELATIVOS
# ============================================

# Permite importar desde src/app/ sin instalar el paquete
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from supabase_client import get_supabase_client  # noqa: E402
from database import save_prediction, save_feedback  # noqa: E402

# ============================================
# 1. CARGAR MODELO SERIALIZADO
# ============================================

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
# ============================================

ANIMAL_TYPE_OPCIONES = {
    "Perro": "Dog",
    "Gato":  "Cat"
}

SEX_OPCIONES = {
    "Macho entero (no esterilizado)":  "Intact Male",
    "Hembra entera (no esterilizada)": "Intact Female",
    "Macho castrado":                  "Neutered Male",
    "Hembra esterilizada":             "Spayed Female"
}

INTAKE_TYPE_OPCIONES = {
    "Callejero (Stray)":   "Stray",
    "Entregado por dueño": "Owner Surrender",
    "Asistencia pública":  "Public Assist",
    "Abandonado":          "Abandoned"
}

INTAKE_CONDITION_OPCIONES = {
    "Normal":          "Normal",
    "Lesionado":       "Injured",
    "Enfermo":         "Sick",
    "Embarazada":      "Pregnant",
    "Médico":          "Medical",
    "Comportamiento":  "Behavior",
    "Lactante":        "Nursing",
    "Anciano":         "Aged"
}

BREED_TYPE_OPCIONES = {
    "Raza pura": "purebred",
    "Mestizo":   "mix"
}

BREED_GROUPED_OPCIONES = [
    "Other", "Domestic Shorthair Mix", "Pit Bull Mix",
    "Labrador Retriever Mix", "Domestic Shorthair",
    "Chihuahua Shorthair Mix", "German Shepherd Mix",
    "Domestic Medium Hair Mix", "Australian Cattle Dog Mix",
    "Domestic Longhair Mix", "Siamese Mix", "Pit Bull",
    "Border Collie Mix", "Dachshund Mix", "Boxer Mix",
    "Labrador Retriever", "German Shepherd", "Chihuahua Shorthair",
    "Staffordshire Mix", "Catahoula Mix", "Domestic Medium Hair",
    "Siberian Husky Mix", "Pointer Mix", "Australian Shepherd Mix",
    "Beagle Mix", "Miniature Poodle Mix"
]

COLOR_GROUPED_OPCIONES = {
    "Un solo color (Monocolor)": "Monocolor",
    "Dos colores (Bicolor)":     "Bicolor",
    "Tres colores (Tricolor)":   "Tricolor"
}

# ============================================
# 4. FORMULARIO DE ENTRADA
# ============================================

with st.form(key="input_form"):
    col1, col2 = st.columns(2)

    with col1:
        animal_type_label = st.selectbox(
            "🐾 Tipo de animal", list(ANIMAL_TYPE_OPCIONES.keys())
        )
        sex_label = st.selectbox(
            "⚧ Sexo / Esterilización", list(SEX_OPCIONES.keys())
        )
        age_months = st.number_input(
            "📅 Edad en meses",
            min_value=0, max_value=240, value=12, step=1,
            help="Introduce la edad aproximada en meses"
        )
        intake_type_label = st.selectbox(
            "🚪 Tipo de ingreso", list(INTAKE_TYPE_OPCIONES.keys())
        )
        intake_condition_label = st.selectbox(
            "🏥 Condición al ingreso", list(INTAKE_CONDITION_OPCIONES.keys())
        )

    with col2:
        breed_type_label = st.selectbox(
            "🧬 Tipo de raza", list(BREED_TYPE_OPCIONES.keys())
        )
        breed_grouped = st.selectbox(
            "🐕 Raza", BREED_GROUPED_OPCIONES,
            help="Selecciona la raza más cercana. Si no está en la lista, elige 'Other'."
        )
        color_label = st.selectbox(
            "🎨 Coloración del pelaje", list(COLOR_GROUPED_OPCIONES.keys())
        )

    st.markdown("---")
    submit_button = st.form_submit_button(
        label="🔍 Predecir días en refugio",
        use_container_width=True
    )

# ============================================
# 5. HELPERS
# ============================================

def calcular_age_group(meses: int) -> str:
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


def mostrar_resultado(dias_pred: float) -> None:
    """Renderiza el bloque de resultado con semáforo de colores."""
    st.markdown("---")
    st.subheader("📊 Resultado de la predicción")

    if dias_pred <= 14:
        st.success("🟢 Alta probabilidad de adopción rápida")
    elif dias_pred <= 45:
        st.warning("🟡 Tiempo de adopción moderado")
    else:
        st.error("🔴 Este animal puede necesitar más apoyo para encontrar hogar")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        st.metric("Días estimados",  f"{dias_pred:.0f} días")
    with col_r2:
        st.metric("Rango mínimo",    f"{max(1, dias_pred * 0.6):.0f} días")
    with col_r3:
        st.metric("Rango máximo",    f"{dias_pred * 1.4:.0f} días")

    st.info(
        "ℹ️ Esta predicción se basa en un modelo XGBoost entrenado con datos históricos "
        "del Austin Animal Center. El error medio del modelo es de ~30 días. "
        "La estimación es orientativa y puede variar según factores no recogidos en los datos."
    )

# ============================================
# 6. PREDICCIÓN + GUARDADO EN SUPABASE
# ============================================

if submit_button:

    # --- 6a. Mapear etiquetas → valores del modelo ---
    animal_type_val      = ANIMAL_TYPE_OPCIONES[animal_type_label]
    sex_val              = SEX_OPCIONES[sex_label]
    intake_type_val      = INTAKE_TYPE_OPCIONES[intake_type_label]
    intake_condition_val = INTAKE_CONDITION_OPCIONES[intake_condition_label]
    breed_type_val       = BREED_TYPE_OPCIONES[breed_type_label]
    color_val            = COLOR_GROUPED_OPCIONES[color_label]
    age_days             = int(age_months * 30.44)
    age_group            = calcular_age_group(age_months)

    input_df = pd.DataFrame({
        "AnimalType":      [animal_type_val],
        "Sex":             [sex_val],
        "IntakeType":      [intake_type_val],
        "IntakeCondition": [intake_condition_val],
        "AgeInDays":       [age_days],
        "AgeGroup":        [age_group],
        "breed_type":      [breed_type_val],
        "Breed_grouped":   [breed_grouped],
        "Color_grouped":   [color_val]
    })

    # --- 6b. Predecir ---
    try:
        pred_log  = modelo.predict(input_df)[0]
        dias_pred = float(np.expm1(pred_log))
        dias_pred = max(1.0, round(dias_pred, 1))
    except Exception as e:
        st.error(f"❌ Error al realizar la predicción: {str(e)}")
        st.stop()

    # --- 6c. Mostrar resultado ---
    mostrar_resultado(dias_pred)

    with st.expander("📋 Ver datos introducidos"):
        st.dataframe(input_df.T.rename(columns={0: "Valor"}).astype(str))

    # --- 6d. Guardar predicción en Supabase ---
    prediction_data = {
        "animal_type":      animal_type_val,
        "sex":              sex_val,
        "intake_type":      intake_type_val,
        "intake_condition": intake_condition_val,
        "age_in_days":      age_days,
        "age_group":        age_group,
        "breed_type":       breed_type_val,
        "breed_grouped":    breed_grouped,
        "color_grouped":    color_val,
        "predicted_days":   dias_pred,
        "predicted_log":    float(pred_log),
        "user_ip":          None  # Streamlit Cloud no expone la IP directamente
    }

    prediction_id = save_prediction(prediction_data)

    # Guardamos el prediction_id y los días en session_state para el bloque de feedback
    st.session_state["prediction_id"] = prediction_id
    st.session_state["dias_pred"]      = dias_pred
    st.session_state["show_feedback"]  = True

# ============================================
# 7. BLOQUE DE FEEDBACK (fuera del form)
# ============================================

if st.session_state.get("show_feedback") and st.session_state.get("prediction_id"):

    prediction_id = st.session_state["prediction_id"]

    st.markdown("---")
    st.subheader("💬 ¿Nos das tu opinión?")
    st.markdown(
        "Tu feedback nos ayuda a mejorar el modelo. "
        "Puedes rellenar solo los campos que quieras."
    )

    with st.form(key="feedback_form"):

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            accuracy_rating = st.slider(
                "⭐ ¿Cómo de precisa fue la predicción? (1 = nada, 5 = muy precisa)",
                min_value=1, max_value=5, value=3
            )
            was_accurate = st.radio(
                "¿La predicción fue correcta en general?",
                options=["No lo sé todavía", "Sí", "No"],
                horizontal=True
            )

        with col_f2:
            actual_days = st.number_input(
                "📅 Días reales hasta la adopción (opcional)",
                min_value=0, max_value=365, value=0, step=1,
                help="Déjalo en 0 si no lo sabes todavía"
            )
            comments = st.text_area(
                "💬 Comentarios (opcional)",
                placeholder="¿Algo que quieras contarnos?",
                max_chars=500
            )

        send_feedback = st.form_submit_button(
            "📨 Enviar feedback", use_container_width=True
        )

    if send_feedback:
        # Mapear radio → bool o None
        was_accurate_bool = None
        if was_accurate == "Sí":
            was_accurate_bool = True
        elif was_accurate == "No":
            was_accurate_bool = False

        save_feedback(
            prediction_id  = prediction_id,
            actual_days    = int(actual_days) if actual_days > 0 else None,
            was_accurate   = was_accurate_bool,
            accuracy_rating= accuracy_rating,
            comments       = comments.strip() if comments.strip() else None
        )
        # Ocultar el formulario de feedback tras enviarlo
        st.session_state["show_feedback"] = False