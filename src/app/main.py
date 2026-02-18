import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

def main():
    print("Hello from proyect-v-regression-team3!")

def app():
    st.title("Formulario de Predicción de días en Refugio 🏠")
    
    with st.form(key='input_form'):
        age = st.number_input("Edad (meses)", 0, 240, 12)
        weight = st.number_input("Peso (kg)", 0.0, 100.0, 5.0)
        animal_type = st.selectbox("Tipo de animal", ["Perro", "Gato", "Conejo", "Ave", "Otro"])
        breed = st.text_input("Raza", value="", placeholder="Escribe la raza aquí")
        submit_button = st.form_submit_button(label='Predecir')
    
    if submit_button:
        input_df = pd.DataFrame({
            'PetType': [animal_type],
            'Breed': [breed],
            'Color': ['Otro'],
            'Size': ['Mediano'],
            'AgeMonths': [age],
            'WeightKg': [weight],
            'AdoptionFee': [0],
            'Vaccinated': [0],
            'HealthCondition': [1],
            'PreviousOwner': [0]
        })
        
        BASE_DIR = Path(__file__).resolve().parent.parent.parent
        modelo_path = BASE_DIR / "models" / "baseline_best_model.joblib"
        modelo = joblib.load(modelo_path)
        
        pred = modelo.predict(input_df)
        st.subheader("Predicción de días en refugio")
        st.write(f"{pred[0]:.1f} días")

if __name__ == "__main__":
    main()
    app()
