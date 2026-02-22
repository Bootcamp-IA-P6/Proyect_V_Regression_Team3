# docker/Dockerfile
FROM python:3.11-slim

# Variables de entorno para Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Instalar uv
RUN pip install uv

# Copiar ficheros de dependencias primero (aprovecha caché de Docker)
COPY pyproject.toml uv.lock ./

# Instalar dependencias con uv
RUN uv sync --frozen --no-dev

# Copiar el código de la app y el modelo
COPY src/app/streamlit_app.py ./src/app/streamlit_app.py
COPY models/best_model_XGBoost.pkl ./models/best_model_XGBoost.pkl

# Puerto de Streamlit
EXPOSE 8501

# Healthcheck para docker-compose
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Comando de arranque
CMD ["uv", "run", "streamlit", "run", "src/app/streamlit_app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
