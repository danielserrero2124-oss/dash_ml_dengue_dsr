"""Pestaña 1 — Contexto del problema: importancia, pregunta, objetivos, datos, formulación matemática y partición."""

import dash  # get_asset_url para las imágenes
from dash import html  # Componentes HTML

from tablero import figuras as fg  # Figuras
from tablero.componentes import chips, formula, grafico, kpi, lectura, miles, numero, pasos, pct, seccion, tabla, tarjeta  # Interfaz
from tablero.datos import A, MOD  # Artefactos

FORMULACION = r"""
**Canal endémico.** Para el municipio $m$ y la semana $t$, con $c_{m,t}$ los casos notificados, el umbral es el tercer cuartil de
las mismas semanas ($\pm 2$) de los cinco años anteriores, y hay brote si se supera con al menos 3 casos:

$$u_{m,t} = Q_{0.75}\big\{c_{m,\,t-52k+j} : k = 1,\dots,5;\ j = -2,\dots,2\big\}, \qquad b_{m,t} = \mathbb{1}\big[c_{m,t} > u_{m,t} \,\wedge\, c_{m,t} \ge 3\big].$$

**Objetivo de alerta temprana** ($h = 4$ semanas): $y_{m,t} = b_{m,t+4}$. Se busca $f(\mathbf{x}_{m,t}) \approx P(y_{m,t} = 1 \mid \mathbf{x}_{m,t})$,
donde $\mathbf{x}_{m,t}$ solo usa semanas $\le t$.

**Modelo base logístico** (pérdida logarítmica con penalización L2):
$$\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta}\ \frac{1}{2C}\lVert\boldsymbol\beta\rVert^2 - \sum_{i}\Big[y_i \ln \sigma(\boldsymbol\beta^\top \mathbf{x}_i) + (1-y_i)\ln\big(1-\sigma(\boldsymbol\beta^\top \mathbf{x}_i)\big)\Big], \qquad \sigma(z) = \frac{1}{1+e^{-z}}.$$

**Modelo base SVM lineal** (pérdida bisagra al cuadrado con penalización L2; el margen se convierte en probabilidad con el escalado de Platt):
$$\min_{\mathbf w, b}\ \tfrac12\lVert\mathbf w\rVert^2 + C\sum_i \max\big(0,\ 1 - \tilde y_i(\mathbf w^\top \mathbf x_i + b)\big)^2, \quad \tilde y_i \in \{-1, 1\}; \qquad \hat p = \sigma\big(a\,(\mathbf w^\top\mathbf x + b) + c\big).$$

**Decisión y evaluación.** Alerta si $\hat p \ge \tau$, con $\tau$ que maximiza $F_1$ en la validación 2022; se evalúa en 2023–2024 con
AUC-ROC $= P(\hat p^{+} > \hat p^{-})$, AUC-PR, $F_1$, MCC y Brier, con intervalos de confianza por bootstrap de municipios.
"""


def seccion_contexto():  # Construye la pestaña
    k = A["contexto"]["kpi"]  # Indicadores
    lr = MOD["ic"]["Regresión logística"]["valor"]  # Métricas del modelo logístico en la prueba
    return html.Div([
        html.Div([  # Portada: texto + imagen
            tarjeta(
                html.H2(["Anticipar los ", html.Span("brotes de dengue", className="resaltado"), " cuatro semanas antes"]),  # Titular
                html.P("El dengue es la enfermedad transmitida por vectores de mayor carga en Colombia. Lo transmite el mosquito Aedes "
                       "aegypti, cuya reproducción y capacidad de contagio dependen de la temperatura, la lluvia y los criaderos de agua. "
                       "En 2024 el país vivió la mayor epidemia de la serie."),  # Importancia
                html.P("Cada semana, las secretarías de salud comparan los casos de cada municipio con su «canal endémico». Anticipar el "
                       "brote con cuatro semanas de margen da tiempo para fumigar, eliminar criaderos, alistar camas y alertar a la "
                       "población. Este entregable presenta el análisis exploratorio y los modelos base de ese sistema de alerta."),  # Motivación
                chips(["Aprendizaje supervisado", "Clasificación binaria", "Panel espacio-temporal", "Alerta a 4 semanas",
                       "Regresión logística", "SVM lineal"]),  # Etiquetas
            ),
            tarjeta(html.Img(src=dash.get_asset_url("brote_colombia.webp"), className="imagen-tarjeta", alt="Dengue en Colombia"), clase="sin-relleno"),  # Imagen
        ], className="rejilla cols-heroe"),
        html.Div([  # Indicadores
            kpi("Casos 2010–2024", miles(k["casos"]), f"{k['municipios']} municipios · {k['semanas']} semanas"),
            kpi("Año récord", str(k["anio_record"]), f"{miles(k['casos_record'])} casos"),
            kpi("Brotes en entrenamiento", pct(k["prev_tr"]), f"{miles(k['brotes_tr'])} de {miles(k['n_tr'])} filas"),
            kpi("AUC-ROC del modelo base", numero(lr["AUC-ROC"]), "regresión logística · prueba 2023–2024"),
        ], className="rejilla cols-4"),
        html.Div(html.Div([html.H2(["El territorio: ", html.Span(f"{k['municipios']} municipios", className="resaltado-ambar"), " con vigilancia semanal"]),  # Banner
                           html.P(f"Panel balanceado de {k['municipios']} municipios de {k['departamentos']} departamentos y {k['semanas']} semanas "
                                  "epidemiológicas (2010–2024): casos, temperatura, lluvia, vegetación, El Niño, población y movilidad.")]),
                 className="banner banner-mapa"),
        html.Div([  # Pregunta, objetivos y tipo de tarea
            tarjeta(html.H3("Pregunta de investigación"),
                    html.P("¿Es posible anticipar, con cuatro semanas de margen, si un municipio colombiano entrará en brote de dengue usando "
                           "solo la información disponible en la semana actual (historia de casos, clima, vegetación, El Niño, población y movilidad)?"),
                    html.H3("Objetivos", style={"marginTop": "14px"}),
                    pasos(["Describir la base de datos, su calidad y su estructura temporal, espacial y espacio-temporal.",
                           "Definir el brote con el canal endémico y construir variables sin información del futuro.",
                           "Explorar cómo la historia de casos, el clima y El Niño se relacionan con los brotes.",
                           "Entrenar dos modelos base lineales (regresión logística y SVM lineal) y compararlos con líneas base triviales."])),
            tarjeta(html.H3("Tipo de tarea"),
                    tabla(["Aspecto", "Decisión"], [
                        ["Aprendizaje", "Supervisado"],
                        ["Tarea", "Clasificación binaria probabilística (brote / no brote)"],
                        ["Unidad de observación", "Municipio × semana epidemiológica"],
                        ["Horizonte", "4 semanas (objetivo en t + 4)"],
                        ["Estructura", "Panel espacio-temporal: filas dependientes en tiempo y espacio"],
                        ["Desbalance", f"{pct(k['prev_tr'])} de brotes en entrenamiento, variable por año"],
                        ["Predictoras", f"{k['p']} variables (historia de casos, clima, El Niño y contexto)"],
                    ])),
        ], className="rejilla cols-2"),
        seccion("Formulación matemática"),
        tarjeta(formula(FORMULACION)),
        html.Div(style={"height": "18px"}),
        html.Div([  # Datos
            tarjeta(html.H3("Fuente de los datos y licencia"),
                    tabla(["Componente", "Origen", "Licencia"], [
                        ["Panel integrado", "data/final_dataset.csv (casos, clima, NDVI, ONI, población y flujo)", "Archivo del proyecto"],
                        ["Casos de dengue", "SIVIGILA, Instituto Nacional de Salud", "Datos abiertos (Ley 1712 de 2014)"],
                        ["Índice ONI", "NOAA Climate Prediction Center", "Dominio público"],
                        ["Centroides municipales", "DANE, DIVIPOLA (datos.gov.co, gdxc-w37w)", "CC BY-SA 4.0"],
                    ])),
            tarjeta(html.H3("Diccionario de variables (21 columnas originales)"),
                    tabla(["Familia", "Columnas", "Significado"], [
                        ["Identificación", "COD_MUN_N, ANO, SEMANA, MES, week_start, time_idx", "Municipio (código DANE) y semana"],
                        ["Casos", "casos_totales; por edad (0–4, 5–14, 15–64, 65+) y sexo", "Casos notificados por semana"],
                        ["Clima", "temp_mean (°C), prec_total (mm), temp_mean_missing", "Temperatura, lluvia y bandera de imputación"],
                        ["Ambiente", "ndvi_mean", "Índice de vegetación [−1, 1]"],
                        ["El Niño", "oni_anom, oni_total (°C)", "Anomalía y temperatura del Pacífico (Niño 3.4)"],
                        ["Contexto", "poblacion, Flujo_in", "Habitantes y personas que ingresan"],
                    ])),
        ], className="rejilla cols-2"),
        seccion("Partición: primero se reserva la prueba",
                "Cronológica, con purga de 4 semanas en cada frontera para que ninguna etiqueta del periodo siguiente se use al entrenar."),
        tarjeta(grafico(figura=fg.fig_particion(A["contexto"]["linea_particion"])),
                tabla(["Conjunto", "Años", "Filas", "Uso"], [
                    ["Historia", "2010–2014", "—", "Solo construye el canal endémico (necesita 5 años previos)"],
                    ["Entrenamiento", "2015–2021", miles(k["n_tr"]), "Ajuste y validación cruzada hacia adelante por años (2018–2021)"],
                    ["Validación", "2022", miles(k["n_va"]), "Umbral de decisión y escalado de Platt de la SVM"],
                    ["Prueba", "2023–2024", miles(k["n_te"]), "Se usa una sola vez, al evaluar los modelos base"],
                ], numericas=(2,)),
                lectura("La partición es temporal porque las semanas de un municipio se parecen entre sí: mezclarlas al azar filtraría "
                        "información del futuro. Como el objetivo de la semana t está en t + 4, las cuatro semanas de cada frontera se "
                        "descartan (franjas grises): así, por ejemplo, ninguna fila de diciembre de 2021 usa una etiqueta de enero de 2022.")),
        html.Div(style={"height": "18px"}),
        html.Div([
            tarjeta(html.H3("Casos por año"), grafico(figura=fg.fig_anual(A["contexto"]["anual"])),
                    lectura("Las epidemias de 2010, 2013, 2019 y 2023–2024 dominan la serie. 2024 es el año récord y queda en la prueba: "
                            "los modelos se evalúan en condiciones más extremas que las del entrenamiento.")),
            tarjeta(html.H3("Consideraciones éticas"),
                    html.P("El panel solo contiene conteos agregados por municipio y semana: no hay datos personales. Las coordenadas son "
                           "centroides oficiales, no ubicaciones de viviendas, y el riesgo de reidentificación es bajo. El riesgo principal "
                           "es de equidad: si la notificación es peor en municipios pobres, un modelo podría subestimar su riesgo. Una alerta "
                           "es apoyo a la vigilancia, no reemplaza el criterio epidemiológico."),
                    html.Div(className="banner banner-brazo", style={"minHeight": "120px", "marginTop": "10px"})),
        ], className="rejilla cols-2"),
    ])
