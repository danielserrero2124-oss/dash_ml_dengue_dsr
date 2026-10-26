# Predicción de brotes de dengue en Colombia según las condiciones climáticas

Sistema de **alerta temprana a 4 semanas**: con la información de la semana *t* (clima, El Niño, historia de casos y
señal espacial del departamento) se estima la probabilidad de que un municipio entre en brote en la semana *t + 4*.
El brote se define con el **canal endémico** (tercer cuartil de las mismas semanas de los cinco años anteriores).

**Entregables:** el dashboard interactivo (Dash + Plotly) y el Jupyter Book compilado con la ejecución de los modelos.

## Modelos
- Recomendados: `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier`, `GradientBoostingClassifier`,
  con `Pipeline` + `GridSearchCV` y validación cruzada temporal hacia adelante por años.
- **Original: HetDS en línea** (`src/hetds.py`), del listado del curso: selección dinámica de un ensamble heterogéneo de
  cinco expertos con autocorrelación espacial, ponderados cada semana con el algoritmo Hedge usando solo etiquetas ya
  confirmadas (retraso de 4 semanas).

## Estructura
```text
data/final_dataset.csv     # panel municipio × semana (2010–2024)
src/config.py              # rutas, periodos, horizonte y variables
src/datos.py               # canal endémico, objetivo y variables
src/modelado.py            # validación temporal, Pipeline, GridSearchCV, métricas, DeLong y bootstrap por municipio
src/hetds.py               # modelo original HetDS en línea
src/entrenamiento.py       # flujo completo de entrenamiento y evaluación
src/pruebas.py             # métricas de la literatura, pruebas de significancia, ablación y tiempos
dashboard/                 # aplicación Dash (Contexto · EDA · ML Models) y sus imágenes
notebooks/                 # cuadernos del Jupyter Book (se ejecutan con nbconvert)
informe/                   # capítulos en Markdown del Jupyter Book
_config.yml, _toc.yml      # configuración del Jupyter Book
```

## Cómo ejecutar (Windows, Python 3.9)
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/2_modelos.ipynb --ExecutePreprocessor.timeout=-1
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/3_pruebas.ipynb --ExecutePreprocessor.timeout=-1
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/1_eda.ipynb --ExecutePreprocessor.timeout=-1
.venv/Scripts/jupyter-book build .        # informe en _build/html/index.html
.venv/Scripts/python -m dashboard.app     # dashboard en http://localhost:8050
```
El cuaderno `2_modelos` entrena los cinco modelos (≈ 1 hora) y guarda los artefactos que usa el dashboard; también se
puede entrenar con `python -m src.entrenamiento` y `python -m src.pruebas`.

## Despliegue en Render
`render.yaml` define un *Blueprint* que construye `docker/Dockerfile` (Python 3.9, plan gratuito). El contenedor no
necesita el CSV: usa `dashboard/eda.joblib` (agregados del EDA, 0,4 MB), `dashboard/artefactos.joblib`,
`dashboard/pruebas.joblib` y `dashboard/hetds.joblib` (el modelo original, que se carga solo al usar el simulador).
Memoria medida con límite de 512 MB: ≈ 116 MB al iniciar y ≈ 440 MB con el simulador en uso.

1. Subir el proyecto a GitHub.
2. En Render: **New → Blueprint** → repositorio `dash_ml_dengue_dsr` → **Apply**.
3. Abrir `https://dengue-dashboard.onrender.com` (en el plan gratuito, la primera carga tras 15 minutos sin visitas tarda ≈ 1 minuto).
