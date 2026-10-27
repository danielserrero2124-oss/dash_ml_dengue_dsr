# El dashboard

## Estructura

El dashboard (Dash 2.18 + Plotly 5.24) tiene tres pestañas. Cada resultado va acompañado de un recuadro de **interpretación**.

| Pestaña | Sub-pestañas | Contenido |
|---|---|---|
| **1. Contexto del problema** | — | Importancia, pregunta y objetivos, tipo de tarea, formulación matemática del objetivo y de los dos modelos base, fuente y licencia, diccionario de variables, partición con purga, casos por año y ética |
| **2. EDA** | Datos y calidad · Variable objetivo · Univariado · Bivariado · Multivariado · Fuga de datos · Temporal · Espacial · Espacio-temporal · Preprocesamiento | Las secciones 1 y 2.1–2.9 de la guía del primer entregable, con controles interactivos |
| **3. Modelos base** | Entrenamiento · Resultados · Diagnóstico · Interpretación · Simulador · Nota crítica | Regresión logística y SVM lineal frente a cuatro líneas base triviales |

## Componentes interactivos

| Dónde | Control | Qué hace |
|---|---|---|
| EDA › Variable objetivo | Municipio | Dibuja el canal endémico de cualquiera de los 376 municipios |
| EDA › Univariado | Variable | Histograma y caja de cualquier variable original o predictora |
| EDA › Bivariado | Par de variables, métrica, variable | Densidad conjunta, asociación con el objetivo (información mutua, biserial de rangos, AUC) y tasa de brote por decil |
| EDA › Multivariado | Color | t-SNE coloreado por objetivo o por región |
| EDA › Temporal | Agregación | Serie nacional por semana, mes o año |
| EDA › Espacial | Variable, indicador | Mapa de incidencia, brote, temperatura, lluvia o flujo; LISA o Getis-Ord |
| EDA › Espacio-temporal | Año | Mapa de la incidencia de cada año 2015–2021 |
| Modelos base › Resultados | Modelo y umbral | Matriz de confusión, precisión, recall, F1, MCC y alertas por semana con cualquier umbral |
| Modelos base › Resultados | Modelos | Curvas ROC y PR de los modelos elegidos |
| Modelos base › Diagnóstico | Modelo, métrica | Calibración, sobreajuste, residuos y curva de aprendizaje |
| Modelos base › Simulador | Municipio, temperatura, lluvia, ONI | Recalcula la probabilidad de brote con el modelo logístico entrenado |

## Arquitectura

```text
entregable2/
├── precalculo.py          # Calcula todo una vez y guarda tablero/artefactos.joblib (≈ 10 min)
├── tablero/               # La aplicación Dash
│   ├── app.py             # Encabezado, pestañas y pie de página
│   ├── datos.py           # Carga los artefactos; matriz de confusión, curvas y simulador en vivo
│   ├── figuras.py         # Todas las figuras (las usa también este libro)
│   ├── componentes.py     # Tarjetas, indicadores, tablas, fórmulas e interpretaciones
│   ├── sec_contexto.py    # Pestaña 1
│   ├── sec_eda.py         # Pestaña 2 y sus callbacks
│   ├── sec_modelos.py     # Pestaña 3 y sus callbacks
│   └── assets/            # Estilos (tema oscuro) e imágenes de fondo
├── construir/             # Generadores de los cuadernos de este libro
└── cuadernos/             # Capítulos ejecutados
```

**Decisión de diseño.** Todo lo costoso (EDA, GridSearchCV, bootstrap de 1 000 réplicas, pruebas espaciales) se calcula una vez en
`precalculo.py`. El dashboard solo lee el archivo de artefactos: abre en segundos, no reentrena nada y consume poca memoria, lo que lo
hace apto para un plan gratuito de despliegue. Los únicos cálculos en vivo son baratos: la matriz de confusión con el umbral elegido, las
curvas ROC/PR y el simulador (un producto punto con los coeficientes del Pipeline).

**Una sola fuente de verdad.** Este libro importa `tablero.figuras` y `tablero.datos`: las figuras y cifras del informe son las mismas
del dashboard.

## Ejecución

Desde la carpeta `entregable2` (con el entorno del proyecto):

```bash
python precalculo.py
```

```bash
python -m tablero.app
```

y abrir `http://localhost:8050`.
