"""
Cliente de Supabase para la aplicación de predicción.
Maneja la conexión y configuración de Supabase.
"""

import streamlit as st
from supabase import create_client, Client
from typing import Optional

@st.cache_resource
def get_supabase_client() -> Optional[Client]:
    """
    Crea y retorna cliente de Supabase.
    Usa st.cache_resource para mantener una única instancia de conexión.
    
    Returns:
        Client de Supabase o None si hay error
    """
    try:
        # Intentar obtener credenciales de secrets.toml
        url = st.secrets["supabase"]["url"]
        key = st.secrets["supabase"]["key"]
        
        # Crear cliente
        client: Client = create_client(url, key)
        
        # Verificar conexión haciendo una query simple
        try:
            client.table("predictions").select("id").limit(1).execute()
            return client
        except Exception as e:
            st.error(f"❌ Error verificando conexión a Supabase: {str(e)}")
            return None
            
    except KeyError:
        st.error(
            "❌ **Error de configuración:** No se encontraron las credenciales de Supabase.\n\n"
            "Por favor, crea el archivo `.streamlit/secrets.toml` con:\n"
            "```toml\n"
            "[supabase]\n"
            'url = "https://texaavadxillwepwxegd.supabase.co"\n'
            'key = "tu_anon_public_key"\n'
            "```"
        )
        return None
    except Exception as e:
        st.error(f"❌ Error inesperado conectando a Supabase: {str(e)}")
        return None


def test_connection() -> bool:
    """
    Prueba la conexión a Supabase.
    
    Returns:
        True si la conexión es exitosa, False en caso contrario
    """
    client = get_supabase_client()
    return client is not None