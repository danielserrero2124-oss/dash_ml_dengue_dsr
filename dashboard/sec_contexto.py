"""Sección «Contexto» — Problema y formulación matemática: importancia, objetivos, variables, tipo de tarea y ecuaciones."""

import dash  # get_asset_url para las imágenes
from dash import html  # Componentes HTML

from dashboard import figuras as fg  # Formato de porcentajes
from dashboard.componentes import formula, kpi, miles, numero, pasos, rubrica, tabla, tarjeta  # Piezas de la interfaz
from dashboard.datos import ANUAL, RES, RESUMEN  # Datos compartidos (agregados precalculados)
from src import config as cf  # Configuración

FORMULACION = r"""
**Datos.** Panel $\{(\mathbf{x}_{m,t},\, c_{m,t})\}$ de $M = 376$ municipios y $T = 782$ semanas epidemiológicas (2010–2024), donde
$c_{m,t}$ son los casos notificados y $\mathbf{x}_{m,t}$ el clima, el contexto y la historia de casos disponibles en la semana $t$.

**Canal endémico (definición de brote).** Para cada municipio y semana se toma el tercer cuartil de los casos de las mismas semanas
($\pm 2$) en los cinco años anteriores:

$$u_{m,t} = Q_{0.75}\big\{c_{m,\,t - 52k + d} : k = 1,\dots,5;\; d = -2,\dots,2\big\}, \qquad
b_{m,t} = \mathbb{1}\left[c_{m,t} > u_{m,t}\;\wedge\; c_{m,t} \ge 3\right].$$

**Objetivo (alerta temprana a $h = 4$ semanas).** $y_{m,t} = b_{m,\,t+4}$. Se estima

$$f(\mathbf{x}_{m,t}) \approx P\big(y_{m,t} = 1 \mid \mathbf{x}_{m,t}\big), \qquad f: \mathcal{X} \rightarrow [0, 1].$$

**Entrenamiento** (riesgo empírico regularizado con pérdida logarítmica):

$$\hat{\theta} = \arg\min_{\theta}\; -\frac{1}{n}\sum_{(m,t)}\Big[y_{m,t}\ln f_\theta(\mathbf{x}_{m,t}) + (1 - y_{m,t})\ln\big(1 - f_\theta(\mathbf{x}_{m,t})\big)\Big] + \lambda\,\Omega(\theta).$$

**Decisión.** Alerta si $f(\mathbf{x}_{m,t}) \ge \tau$, con $\tau^{*} = \arg\max_\tau F_1(\tau)$ en la validación de 2022.

**Evaluación.** $\text{AUC-ROC} = P\big(f(\mathbf{x}^{+}) > f(\mathbf{x}^{-})\big)$ y $\text{AUC-PR} = \sum_k (R_k - R_{k-1})P_k$ en 2023–2024, nunca vistos al entrenar.

**Restricción de información.** Toda variable de la fila $(m, t)$ usa datos de semanas $\le t$; el umbral $u_{m,t+4}$ solo usa años anteriores, así que ya se conoce en $t$.
"""


def seccion_contexto():  # Construye la sección
    familias = [  # Diccionario de variables por familia
        ("Identificación", "COD_MUN_N, ANO, SEMANA, MES, week_start, time_idx", "Municipio (código DANE) y semana epidemiológica"),
        ("Casos", "casos_totales, casos_0_4, casos_5_14, casos_15_64, casos_65_plus, casos_m, casos_f", "Casos notificados de dengue, por edad y sexo"),
        ("Clima", "temp_mean, prec_total, temp_mean_missing", "Temperatura media, precipitación acumulada y bandera de imputación"),
        ("Ambiente", "ndvi_mean", "Índice de vegetación NDVI (criaderos y humedad)"),
        ("El Niño", "oni_anom, oni_total", "Índice Oceánico El Niño (anomalía y valor total de la temperatura del Pacífico)"),
        ("Contexto", "poblacion, Flujo_in", "Población y flujo de personas que entran al municipio"),
    ]
    return html.Div([
        rubrica(1),  # Título de la parte de la rúbrica
        html.Div([  # Portada: texto + collage
            tarjeta(
                html.H2(["Anticipar los ", html.Span("brotes de dengue", className="resaltado"), " en Colombia a partir del clima"]),  # Titular
                html.P("El dengue es la arbovirosis de mayor impacto en las Américas. Lo transmite el mosquito Aedes aegypti, cuya reproducción "
                       "y capacidad de contagio dependen de la temperatura, la lluvia y los criaderos de agua. En Colombia es endémico en casi "
                       "todo el territorio por debajo de 2 200 m de altitud, y en 2024 el país vivió la mayor epidemia de su historia reciente."),  # Importancia
                html.P("Las secretarías de salud vigilan cada semana el «canal endémico» de cada municipio. Anticipar con cuatro semanas un brote "
                       "da tiempo para fumigar, eliminar criaderos, alistar camas y alertar a la población. Este proyecto construye un sistema de "
                       "alerta temprana que combina el clima, la historia de casos y la señal espacial del departamento."),  # Motivación
                html.Div([html.Span(t, className="chip") for t in ["Aprendizaje supervisado", "Clasificación binaria", "Datos de panel espacio-temporales",
                                                                    "Alerta a 4 semanas", "Validación temporal", "Modelo original: HetDS en línea"]]),  # Etiquetas
            ),
            tarjeta(html.Img(src=dash.get_asset_url("brote_colombia.webp"), className="imagen-tarjeta", alt="Vigilancia del dengue en Colombia"), clase="sin-relleno"),  # Collage
        ], className="rejilla cols-heroe"),
        html.Div([  # Indicadores
            kpi("Casos 2010–2024", miles(RESUMEN["casos_total"]), f"{RESUMEN['n_mun']} municipios · {RESUMEN['n_semanas']} semanas"),
            kpi("Año récord", "2024", f"{miles(ANUAL.loc[2024, 'casos'])} casos"),
            kpi("Semanas de brote", miles(RESUMEN["n_brotes"]), f"{fg.pct(RESUMEN['tasa_brote'])} de 2015–2024"),
            kpi("AUC-ROC de HetDS", numero(RES["HetDS"]["test"]["auc_roc"]) if RES else "—", "prueba 2023–2024"),
        ], className="rejilla cols-4"),
        html.Div(html.Div([html.H2(["El territorio: ", html.Span("376 municipios", className="resaltado-ambar"), " con vigilancia semanal"]),  # Banner del mapa
                           html.P("El panel cubre los municipios con transmisión de dengue de 32 departamentos. Cada semana se observan los casos, "
                                  "la temperatura, la lluvia, la vegetación y el estado de El Niño en el Pacífico.")]), className="banner banner-mapa"),
        html.Div([  # Objetivos + tipo de tarea
            tarjeta(
                html.H3("Objetivos"),
                html.P([html.B("General: "), "predecir con cuatro semanas de anticipación si un municipio entrará en brote de dengue, usando las "
                        "condiciones climáticas, la historia de casos y la señal espacial, con un flujo libre de fuga de datos."]),
                pasos(["Definir el brote con el canal endémico (método de cuartiles) y construir variables sin información del futuro.",
                       "Explorar cómo el clima, El Niño y la estacionalidad se relacionan con los brotes.",
                       "Entrenar y comparar LogisticRegression, RandomForest, KNN y GradientBoosting con Pipeline, GridSearchCV y validación temporal.",
                       "Implementar desde cero el modelo original HetDS en línea y probar su superioridad con pruebas de significancia.",
                       "Entregar un dashboard interactivo y un Jupyter Book con la ejecución de los modelos."]),
            ),
            tarjeta(
                html.H3("Tipo de tarea de Machine Learning"),
                html.Ul([
                    html.Li([html.B("Supervisada, clasificación binaria: "), "brote (1) o no brote (0) en la semana t + 4."]),
                    html.Li([html.B("Probabilística: "), "el modelo entrega P(brote) y un umbral la convierte en alerta."]),
                    html.Li([html.B("Panel espacio-temporal: "), "filas correlacionadas en el tiempo (semanas) y en el espacio (municipios vecinos)."]),
                    html.Li([html.B("Desbalance moderado y cambiante: "), f"{fg.pct(RESUMEN['tasa_tr'])} de brotes en entrenamiento y "
                             f"{fg.pct(RESUMEN['tasa_te'])} en la prueba (epidemia de 2024)."]),
                ]),
                html.H3("Variables", style={"marginTop": "14px"}),
                html.P([html.B("Objetivo: "), html.Code("y = brote en t + 4"), ". ", html.B("Predictoras: "),
                        f"{len(cf.FEATURES)} variables para los modelos recomendados ({len(cf.CASOS)} de historia de casos, {len(cf.CLIMA)} climáticas y "
                        f"{len(cf.CONTEXTO)} de contexto) y {len(cf.ESPACIALES)} espaciales adicionales para el modelo original."]),
            ),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Formulación matemática del problema"), formula(FORMULACION)),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Diccionario de datos (21 columnas originales)"),
                    tabla(["Familia", "Columnas", "Qué describe"], [(html.B(f), html.Code(c), d) for f, c, d in familias])),
            tarjeta(html.Img(src=dash.get_asset_url("mosquitos_brazo.webp"), className="imagen-tarjeta", alt="Mosquitos Aedes picando un brazo"), clase="sin-relleno"),
        ], className="rejilla cols-heroe"),
    ])
