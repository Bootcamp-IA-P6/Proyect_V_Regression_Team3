import streamlit as st
import pandas as pd
import joblib

def main():
    print("Hello from proyect-v-regression-team3!")  # Para terminal

def app():
    st.title("Formulario de Predicción de días en refugio 🏠")
    
    # --- Formulario ---
    with st.form(key='input_form'):
        age = st.number_input("Edad (meses)", 0, 240, 12)
        weight = st.number_input("Peso (kg)", 0.0, 100.0, 5.0)
        animal_type = st.selectbox("Tipo de animal", ["Perro", "Gato", "Conejo", "Ave", "Otro"])
        breed = st.text_input("Raza", "Desconocida")
        submit_button = st.form_submit_button(label='Predecir')
    
    if submit_button:
        input_df = pd.DataFrame({
            'age': [age],
            'weight': [weight],
            'animal_type': [animal_type],
            'breed': [breed]
        })
        # Cargar modelo
        modelo = joblib.load("models/modelo_regresion.pkl")
        pred = modelo.predict(input_df)
        st.subheader("Predicción de días en refugio")
        st.write(f"{pred[0]:.1f} días")

if __name__ == "__main__":
    main()  # Para terminal
    app()   # Para Streamlit
