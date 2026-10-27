# Imagen del dashboard del Segundo Entregable (Dash + Plotly) para Render
# Se construye desde la carpeta entregable2 (contexto de Docker = entregable2)
FROM python:3.9-slim

# Sin archivos .pyc y con registros sin búfer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Carpeta de trabajo dentro del contenedor
WORKDIR /app

# Usuario sin privilegios (buena práctica de seguridad)
RUN useradd --create-home appuser

# Dependencias primero, para aprovechar la caché de capas
COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

# Solo la aplicación: incluye tablero/artefactos.joblib, así que el contenedor no necesita el CSV ni reentrenar (≈ 180 MB de RAM)
COPY --chown=appuser:appuser tablero/ ./tablero/
USER appuser

# Puerto por defecto (Render asigna el suyo en la variable PORT)
EXPOSE 8050

# Gunicorn sirve la app Flask interna de Dash: 1 proceso con 2 hilos
CMD ["sh", "-c", "exec gunicorn tablero.app:server --bind 0.0.0.0:${PORT:-8050} --workers 1 --threads 2 --timeout 120"]
