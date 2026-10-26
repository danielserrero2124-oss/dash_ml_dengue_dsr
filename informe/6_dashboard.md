# El dashboard

## Cómo ejecutarlo

Desde la carpeta raíz del proyecto, con el entorno de Python 3.9:

```bash
.venv\Scripts\python -m src.entrenamiento   # entrena los cinco modelos (o ejecutar notebooks/2_modelos.ipynb)
.venv\Scripts\python -m src.pruebas         # pruebas de significancia, ablación y tiempos
.venv\Scripts\python -m dashboard.app       # abre http://localhost:8050
```

## Estructura según la rúbrica de sustentación

| Pestaña | Parte de la rúbrica | Contenido |
|---|---|---|
| Contexto | Problema y formulación matemática | Importancia, objetivos, tipo de tarea, variables, canal endémico y ecuaciones |
| EDA | EDA, datos y preprocesamiento | Fuente, estadística descriptiva, series, estacionalidad, canal endémico interactivo, clima vs. brote, atípicos, faltantes, correlaciones, transformaciones y fuga |
| ML Models › Fundamentos teóricos | Fundamentos teóricos del modelo | Formulación de HetDS, Hedge, FTRL, garantía de arrepentimiento, supuestos, pesos semanales |
| ML Models › Diseño experimental | Implementación y diseño experimental | Partición temporal, validación hacia adelante, código del Pipeline y GridSearchCV, hiperparámetros, tiempos |
| ML Models › Resultados | Resultados y evaluación crítica | Métricas, ROC/PR, ranking, pruebas de significancia, ablación, aporte del clima, simulador climático |
| ML Models › Conclusiones y preguntas | Conclusiones, limitaciones y preguntas | Conclusiones, limitaciones, generalización, trabajo futuro y preguntas esperadas |

## Arquitectura

| Archivo | Responsabilidad |
|---|---|
| `src/config.py` | Rutas, periodos, horizonte y listas de variables |
| `src/datos.py` | Canal endémico, objetivo a 4 semanas, variables (rezagos, anomalías, rezago espacial) |
| `src/modelado.py` | Validación temporal, Pipeline, catálogo, GridSearchCV, métricas, DeLong y bootstrap por conglomerados |
| `src/hetds.py` | Modelo original HetDS en línea (desde cero) |
| `src/entrenamiento.py` | Flujo completo de entrenamiento y evaluación |
| `src/pruebas.py` | Métricas de la literatura, pruebas de significancia, ablación y tiempos |
| `dashboard/datos.py` | Datos y agregados del EDA, cargados una sola vez |
| `dashboard/figuras.py` | Figuras de Plotly con un tema común (también las usa este libro) |
| `dashboard/sec_*.py` | Una sección por pestaña, con sus *callbacks* |
| `dashboard/assets/` | Estilos y las imágenes de fondo (mosquito, mapa, virus, vigilancia) |

## Diseño visual

El tema oscuro usa un acento rojo «dengue» para la marca y una paleta categórica validada para los datos:
azul para «sin brote», rojo para «brote» y un color fijo por modelo (magenta para HetDS). Las imágenes se
usan como fondo de los encabezados de cada sección, con un degradado oscuro que conserva la legibilidad.
