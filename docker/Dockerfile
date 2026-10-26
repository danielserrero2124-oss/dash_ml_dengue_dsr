# Imagen del dashboard de brotes de dengue (Dash + Plotly) para Render
FROM python:3.9-slim

# Sin archivos .pyc y con registros sin búfer
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LOKY_MAX_CPU_COUNT=1

# Carpeta de trabajo (raíz del proyecto dentro del contenedor)
WORKDIR /app

# Usuario sin privilegios (buena práctica de seguridad)
RUN useradd --create-home appuser

# Dependencias primero, para aprovechar la caché de capas
COPY docker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código y artefactos entrenados (incluye eda.joblib: los agregados del EDA, así que el contenedor no necesita el CSV)
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser dashboard/ ./dashboard/
USER appuser

# Puerto por defecto (Render asigna el suyo en la variable PORT)
EXPOSE 8050

# Gunicorn sirve la app Flask interna de Dash: 1 proceso con 2 hilos (≈ 440 MB con el simulador en uso: cabe en los 512 MB del plan gratuito)
CMD ["sh", "-c", "exec gunicorn dashboard.app:server --bind 0.0.0.0:${PORT:-8050} --workers 1 --threads 2 --timeout 180"]
