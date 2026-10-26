# Problema y formulación matemática

## Importancia del problema

El dengue es la arbovirosis de mayor impacto en las Américas. Lo transmite el mosquito *Aedes aegypti*,
cuya reproducción, supervivencia y capacidad de transmitir el virus dependen de la temperatura, la lluvia
y la disponibilidad de criaderos de agua. En Colombia la enfermedad es endémica en casi todo el territorio
por debajo de 2 200 m de altitud y presenta ciclos epidémicos asociados a El Niño; 2024 fue el año con más
casos de la serie (261 382 en los 376 municipios del panel).

La vigilancia en salud pública compara cada semana los casos de un municipio con su **canal endémico**: si
superan lo habitual para esa época del año, se declara una situación de brote. Anticipar ese momento con
cuatro semanas de margen permite intervenir criaderos, fumigar, alistar servicios de salud y alertar a la
población antes del pico.

## Objetivos

**General.** Predecir con cuatro semanas de anticipación si un municipio entrará en brote de dengue, a partir
de las condiciones climáticas, la historia de casos y la señal espacial del departamento, con un flujo libre
de fuga de datos.

**Específicos.**

1. Definir el brote con el método de cuartiles del canal endémico y construir variables que solo usen información
   disponible en la semana de la predicción.
2. Explorar la relación entre el clima, El Niño, la estacionalidad y los brotes.
3. Entrenar y comparar `LogisticRegression`, `RandomForestClassifier`, `KNeighborsClassifier` y
   `GradientBoostingClassifier` con `Pipeline`, `GridSearchCV` y validación temporal.
4. Implementar desde cero un modelo original del listado del curso (HetDS en línea) y contrastar su superioridad
   con pruebas de significancia.
5. Entregar un dashboard interactivo y este Jupyter Book con la ejecución de los modelos.

## Datos

| Elemento | Descripción |
|---|---|
| Archivo | `data/final_dataset.csv` |
| Unidad | Municipio × semana epidemiológica (panel balanceado) |
| Tamaño | 294 032 filas = 376 municipios × 782 semanas (2010–2024), 21 columnas |
| Casos | Totales, por grupo de edad (0–4, 5–14, 15–64, 65+) y por sexo |
| Clima y ambiente | Temperatura media, precipitación, NDVI y bandera de temperatura imputada |
| El Niño | Anomalía y valor total del índice ONI |
| Contexto | Población y flujo de personas que ingresan al municipio |

## Tipo de tarea

* **Aprendizaje supervisado, clasificación binaria:** brote (1) o no brote (0) cuatro semanas después.
* **Probabilística:** el modelo estima $P(\text{brote})$ y un umbral elegido en validación la convierte en alerta.
* **Panel espacio-temporal:** las filas están correlacionadas en el tiempo (semanas de un municipio) y en el espacio
  (municipios de un mismo departamento), lo que obliga a validar respetando el tiempo y a usar pruebas por conglomerados.
* **Desbalance cambiante:** 10,2 % de brotes en el entrenamiento (2015–2021) y 30,3 % en la prueba (2023–2024).

## Formulación matemática

**Canal endémico.** Para el municipio $m$ y la semana $t$, con $c_{m,t}$ los casos notificados:

$$
u_{m,t} = Q_{0.75}\big\{c_{m,\,t - 52k + d} : k = 1,\dots,5;\; d = -2,\dots,2\big\},
\qquad
b_{m,t} = \mathbb{1}\left[c_{m,t} > u_{m,t}\;\wedge\; c_{m,t} \ge 3\right].
$$

**Objetivo de alerta temprana** ($h = 4$ semanas): $y_{m,t} = b_{m,\,t+4}$. Se busca

$$
f(\mathbf{x}_{m,t}) \approx P\big(y_{m,t} = 1 \mid \mathbf{x}_{m,t}\big), \qquad f: \mathcal{X}\rightarrow[0,1],
$$

donde $\mathbf{x}_{m,t}$ solo contiene información de semanas $\le t$ (el umbral $u_{m,t+4}$ usa años anteriores,
así que ya se conoce en $t$).

**Entrenamiento** (riesgo empírico regularizado con pérdida logarítmica):

$$
\hat\theta = \arg\min_\theta -\frac{1}{n}\sum_{(m,t)}\Big[y_{m,t}\ln f_\theta(\mathbf{x}_{m,t}) + (1-y_{m,t})\ln\big(1-f_\theta(\mathbf{x}_{m,t})\big)\Big] + \lambda\,\Omega(\theta).
$$

**Decisión:** alerta si $f(\mathbf{x}_{m,t}) \ge \tau$, con $\tau^{*} = \arg\max_\tau F_1(\tau)$ en la validación de 2022.

**Evaluación** en 2023–2024 (nunca vistos al entrenar):

$$
\text{AUC-ROC} = P\big(f(\mathbf{x}^{+}) > f(\mathbf{x}^{-})\big), \qquad \text{AUC-PR} = \sum_k (R_k - R_{k-1})\,P_k .
$$

## División temporal

| Periodo | Años | Uso |
|---|---|---|
| Historia | 2010–2014 | Solo para calcular el canal endémico |
| Entrenamiento | 2015–2021 | Ajuste; `GridSearchCV` con validación hacia adelante (2018–2021) |
| Validación | 2022 | Umbral de decisión e hiperparámetros del modelo original |
| Prueba | 2023–2024 | Evaluación final, una sola vez |
