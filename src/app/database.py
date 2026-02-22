"""
Funciones para interactuar con la base de datos de Supabase.
Incluye manejo de errores y mensajes amigables al usuario.
"""

import streamlit as st
from typing import Dict, Any, Optional
from supabase_client import get_supabase_client


def save_prediction(prediction_data: Dict[str, Any]) -> Optional[str]:
    """
    Guarda una predicción en la tabla 'predictions' de Supabase.
    
    Args:
        prediction_data: Diccionario con los datos de la predicción
        
    Returns:
        ID de la predicción guardada (UUID) o None si hay error
    """
    client = get_supabase_client()
    
    if client is None:
        st.warning("⚠️ No se pudo conectar a la base de datos. La predicción no se guardó.")
        return None
    
    try:
        # Preparar datos para inserción
        data_to_insert = {
            "animal_type": prediction_data["animal_type"],
            "sex": prediction_data["sex"],
            "intake_type": prediction_data["intake_type"],
            "intake_condition": prediction_data["intake_condition"],
            "age_in_days": prediction_data["age_in_days"],
            "age_group": prediction_data["age_group"],
            "breed_type": prediction_data["breed_type"],
            "breed_grouped": prediction_data["breed_grouped"],
            "color_grouped": prediction_data["color_grouped"],
            "predicted_days": prediction_data["predicted_days"],
            "predicted_log": prediction_data["predicted_log"],
            "user_ip": prediction_data.get("user_ip", None)
        }
        
        # Insertar en Supabase
        response = client.table("predictions").insert(data_to_insert).execute()
        
        # Verificar respuesta
        if response.data and len(response.data) > 0:
            prediction_id = response.data[0]["id"]
            return prediction_id
        else:
            st.warning("⚠️ La predicción se realizó pero no se pudo guardar en la base de datos.")
            return None
            
    except Exception as e:
        st.error(f"❌ Error guardando predicción: {str(e)}")
        st.info("💡 La predicción se muestra correctamente, pero no se guardó en el historial.")
        return None


def save_feedback(
    prediction_id: str,
    actual_days: Optional[int] = None,
    was_accurate: Optional[bool] = None,
    accuracy_rating: Optional[int] = None,
    comments: Optional[str] = None
) -> bool:
    """
    Guarda feedback del usuario en la tabla 'feedback' de Supabase.
    
    Args:
        prediction_id: UUID de la predicción asociada
        actual_days: Días reales que tardó la adopción (opcional)
        was_accurate: Si la predicción fue precisa (opcional)
        accuracy_rating: Calificación 1-5 (opcional)
        comments: Comentarios del usuario (opcional)
        
    Returns:
        True si se guardó exitosamente, False en caso contrario
    """
    client = get_supabase_client()
    
    if client is None:
        st.warning("⚠️ No se pudo conectar a la base de datos. El feedback no se guardó.")
        return False
    
    try:
        # Preparar datos para inserción
        data_to_insert = {
            "prediction_id": prediction_id,
            "actual_days": actual_days,
            "was_accurate": was_accurate,
            "accuracy_rating": accuracy_rating,
            "comments": comments
        }
        
        # Eliminar campos None
        data_to_insert = {k: v for k, v in data_to_insert.items() if v is not None}
        
        # Insertar en Supabase
        response = client.table("feedback").insert(data_to_insert).execute()
        
        if response.data and len(response.data) > 0:
            st.success("✅ ¡Gracias por tu feedback! Nos ayuda a mejorar el modelo.")
            return True
        else:
            st.warning("⚠️ No se pudo guardar el feedback. Inténtalo de nuevo.")
            return False
            
    except Exception as e:
        st.error(f"❌ Error guardando feedback: {str(e)}")
        return False