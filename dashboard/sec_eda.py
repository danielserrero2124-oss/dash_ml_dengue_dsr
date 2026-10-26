"""Sección «EDA» — EDA, datos y preprocesamiento: fuente y estructura, estadística descriptiva, series, estacionalidad,
canal endémico, clima frente a brotes, atípicos, faltantes, correlaciones, transformaciones y prevención de fuga."""

import pandas as pd  # Tablas
from dash import Input, Output, callback, html  # Componentes y callbacks

from dashboard import figuras as fg  # Figuras
from dashboard.componentes import desplegable, grafico, kpi, miles, numero, rubrica, tabla, tarjeta  # Interfaz
from dashboard.datos import (ANUAL, ART, ATIPICOS, CANAL, CCF, CORR, CORR_Y, CORTES, DEP, DESCRIPTIVA, EDAD, EST, FALTANTES, NAC, ONI_ANUAL,  # Datos
                             RESUMEN, SEXO, TOP_MUN, VARS_DECILES, deciles, nombre_municipio)

VARS_CLIMA = VARS_DECILES  # Variables del gráfico de deciles (precalculados)


def tabla_atipicos():  # Atípicos por la regla de Tukey (precalculados en eda.joblib)
    return ATIPICOS  # Lista de diccionarios


def tabla_descriptiva():  # Estadística descriptiva de las variables originales (precalculada)
    filas = [(html.Code(v), numero(r["mean"], 2), numero(r["std"], 2), numero(r["min"], 2), numero(r["25%"], 2), numero(r["50%"], 2),  # Fila por variable
              numero(r["75%"], 2), numero(r["max"], 2), numero(r["asimetria"], 2)) for v, r in DESCRIPTIVA.iterrows()]
    return tabla(["Variable", "Media", "Desv.", "Mín.", "Q1", "Mediana", "Q3", "Máx.", "Asimetría"], filas, numericas=range(1, 9))  # Tabla


def seccion_eda():  # Construye la sección
    estructura = [("Identificación y tiempo", 6, "Municipio, año, semana, mes, fecha, índice de tiempo"), ("Casos", 7, "Totales, 4 grupos de edad y sexo"),
                  ("Clima y ambiente", 4, "Temperatura, precipitación, NDVI y bandera de imputación"), ("El Niño", 2, "Anomalía y valor total ONI"),
                  ("Contexto", 2, "Población y flujo de personas")]
    transformaciones = [
        ("Canal endémico (Q3 de 5 años, ± 2 semanas)", "casos_totales", "Convierte conteos en «brote / no brote» relativo a lo habitual de cada municipio (método de Bortman, usado por el INS)."),
        ("log(1 + casos)", "Casos y flujo", "Los conteos tienen asimetría > 10; el logaritmo estabiliza la varianza y reduce el peso de las epidemias extremas."),
        ("Rezagos y medias móviles (1–12 semanas)", "Casos y clima", "El clima actúa con retraso sobre el mosquito y el virus (ciclo de 2–8 semanas); solo se usan semanas ≤ t."),
        ("Anomalías climáticas", "Temperatura, lluvia, NDVI", "Desviación respecto a la normal del municipio en ese mes (calculada solo con 2010–2021): separa el clima del lugar."),
        ("Codificación seno/coseno", "Semana epidemiológica", "Representa la estacionalidad como un ciclo (la semana 52 está junto a la 1)."),
        ("Rezago espacial del departamento", "Casos y brotes vecinos", "Autocorrelación espacial: el dengue se propaga entre municipios cercanos (solo el modelo original)."),
        ("Imputación con la mediana + estandarización", "Todas (dentro del Pipeline)", "La mediana es robusta a atípicos; la escala común es necesaria para KNN y la regresión logística."),
        ("Sin remuestreo; class_weight comparado", "Desbalance", "Se compara class_weight=None vs. balanced en la búsqueda; el umbral se ajusta en 2022."),
    ]
    fuga = ART["fuga"] if ART else None  # Demostración de fuga
    imputada = RESUMEN["temp_imputada"]  # Proporción de temperatura imputada
    return html.Div([
        rubrica(2),  # Título de la parte de la rúbrica
        html.Div(html.Div([html.H2(["El vector: ", html.Span("Aedes aegypti", className="resaltado"), " y las condiciones que lo favorecen"]),  # Banner
                           html.P("El mosquito se reproduce en agua estancada y su ciclo se acelera con el calor: más temperatura acorta la incubación "
                                  "del virus en el mosquito y la lluvia crea criaderos. Por eso el clima de hoy anticipa los casos de las próximas semanas.")]),
                 className="banner banner-vector"),
        html.Div([
            kpi("Municipio-semanas", miles(RESUMEN["n_raw"]), f"{RESUMEN['n_mun']} municipios × {RESUMEN['n_semanas']} semanas"),
            kpi("Filas para modelar", miles(RESUMEN["n_mod"]), "2015–2024 con canal endémico"),
            kpi("Brote a 4 semanas", fg.pct(RESUMEN["tasa_mod"]), "proporción de positivos"),
            kpi("Temperatura imputada", fg.pct(imputada), "bandera temp_mean_missing"),
            kpi("Casos por semana", numero(RESUMEN["casos_media"], 1), f"mediana {numero(RESUMEN['casos_mediana'], 0)}: muy asimétrico"),
        ], className="rejilla cols-5"),
        html.Div([
            tarjeta(html.H3("Fuente y estructura de los datos"),
                    html.P(["Archivo ", html.Code("final_dataset.csv"), ": panel balanceado de 294 032 filas (376 municipios × 782 semanas, 2010–2024) "
                            "que integra la notificación semanal de dengue (SIVIGILA) con clima, vegetación satelital (NDVI), el índice ONI de la NOAA, "
                            "población y flujo de personas. No tiene nulos: la temperatura faltante ya viene imputada y marcada."]),
                    tabla(["Familia", "Columnas", "Contenido"], estructura, numericas=[1])),
            tarjeta(html.H3("Casos por año y periodo del modelo"),
                    html.P("Gris: solo historia del canal endémico · Azul: entrenamiento · Verde: validación · Ámbar: prueba (incluye el récord de 2024).", className="nota"),
                    grafico("eda-anual", fg.fig_anual(ANUAL))),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Casos semanales en Colombia y El Niño"),
                html.P("Los picos epidémicos (2010, 2013, 2019, 2023–2024) coinciden con episodios de El Niño (ONI > 0,5, rojo). Sombreado: validación y prueba.", className="nota"),
                grafico("eda-serie", fg.fig_serie_nacional(NAC, CORTES))),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Estacionalidad: casos y lluvia por semana epidemiológica"),
                    html.P("Los casos alcanzan su máximo hacia las semanas 15–20, durante la primera temporada de lluvias, y vuelven a subir con las lluvias de fin de año.", className="nota"),
                    grafico("eda-estacion", fg.fig_estacionalidad(EST))),
            tarjeta(html.H3("¿Quiénes enferman?"),
                    html.P(f"Los adultos de 15 a 64 años son el grupo más numeroso, pero los menores de 15 años suman el {fg.pct((EDAD['0–4'] + EDAD['5–14']) / sum(EDAD.values()), 0)} de los casos.", className="nota"),
                    grafico("eda-grupos", fg.fig_grupos(EDAD, SEXO))),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Canal endémico de un municipio: así se define un brote"),
                html.P("Un brote es una semana cuyos casos superan el tercer cuartil de las mismas semanas de los 5 años anteriores (y al menos 3 casos).", className="nota"),
                html.Label("Municipio", className="control-etiqueta"),
                desplegable("eda-mun", [{"label": nombre_municipio(m), "value": m} for m in TOP_MUN], TOP_MUN[0]),
                grafico("eda-canal")),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Clima de hoy y brote en 4 semanas"),
                    html.P("Proporción de brotes cuatro semanas después según el decil de la variable climática.", className="nota"),
                    html.Label("Variable", className="control-etiqueta"),
                    desplegable("eda-clima", [{"label": fg.ETIQUETAS[v], "value": v} for v in VARS_CLIMA], "temp_mean_m8"),
                    grafico("eda-deciles")),
            tarjeta(html.H3("Correlación cruzada: el clima antecede a los casos"),
                    html.P("Correlación media por municipio entre la anomalía climática en t − k y el log de casos en t. La señal climática "
                           "es real pero débil (|r| ≈ 0,05): el clima modula el riesgo, no lo determina por sí solo.", className="nota"),
                    grafico("eda-ccf", fg.fig_ccf(CCF))),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("El Niño y los años epidémicos"),
                    html.P("El ONI del mismo año explica poco (2015 tuvo un El Niño fuerte y pocos brotes); el efecto llega con rezago: El Niño de 2023 antecede a la epidemia récord de 2024.", className="nota"),
                    grafico("eda-oni", fg.fig_oni_anual(ONI_ANUAL))),
            tarjeta(html.H3("Departamentos con mayor incidencia"),
                    html.P("Incidencia media anual 2010–2024. Los valles interandinos y los Llanos concentran la carga.", className="nota"),
                    grafico("eda-dep", fg.fig_departamentos(DEP))),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Estadística descriptiva de las variables originales"),
                    html.P("Casos, flujo y población tienen asimetrías muy altas: motivan el logaritmo y la mediana como imputación.", className="nota"),
                    html.Div(tabla_descriptiva(), className="caja-scroll")),
            tarjeta(html.H3("Datos faltantes: temperatura imputada"),
                    html.P(f"No hay nulos, pero el {fg.pct(imputada)} de las semanas tiene temperatura imputada (concentrada cerca de 22 °C). "
                           "Se conserva la bandera temp_missing como variable para que el modelo sepa cuándo el dato es estimado.", className="nota"),
                    grafico("eda-faltantes", fg.fig_faltantes(FALTANTES))),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Valores atípicos (regla de Tukey: 1,5 × IQR)"),
                    html.P(["Los atípicos ", html.B("no se eliminan"), ": en casos e incidencia son justamente las epidemias, y en ellos la tasa de brote es mucho mayor. "
                            "Se manejan con el logaritmo y con modelos robustos (árboles)."], className="nota"),
                    grafico("eda-atipicos", fg.fig_atipicos(tabla_atipicos()))),
            tarjeta(html.H3("Correlación de cada variable con el brote a 4 semanas"),
                    html.P("La historia de casos domina (r ≈ 0,5); las variables climáticas tienen correlaciones pequeñas pero útiles en combinación.", className="nota"),
                    grafico("eda-corr-y", fg.fig_correlacion_objetivo(CORR_Y))),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Relaciones entre variables"),
                html.P("Los rezagos de casos y las medias móviles del clima están muy correlacionados entre sí; la regularización y los árboles lo manejan.", className="nota"),
                grafico("eda-corr", fg.fig_matriz_correlacion(CORR))),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta(html.H3("Justificación de las transformaciones"), tabla(["Transformación", "Se aplica a", "Justificación"], transformaciones)),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Prevención de la fuga de información (data leakage)"),
                    tabla(["Riesgo", "Decisión"], [
                        ("Usar casos de la semana objetivo (t + 4) o posteriores", "Prohibido: todas las variables usan semanas ≤ t"),
                        ("Umbral del canal endémico calculado con el año actual", "Solo con los 5 años anteriores"),
                        ("Normal climática calculada con años de prueba", "Solo con 2010–2021"),
                        ("Partición aleatoria de semanas (autocorrelación)", "División temporal y validación cruzada hacia adelante por años"),
                        ("Preprocesamiento ajustado con todos los datos", "Dentro del Pipeline, por pliegue"),
                        ("Etiquetas aún no confirmadas en el modelo en línea", "HetDS solo usa etiquetas de semanas ≤ t − 4"),
                    ])),
            tarjeta(html.H3("Demostración: dos fugas típicas en series de tiempo"),
                    html.P("Mezclar semanas al azar o incluir los casos de la semana objetivo infla las métricas: el modelo parece mejor de lo que será en producción.", className="nota"),
                    grafico("eda-fuga", fg.fig_fuga(fuga)) if fuga else html.P("Ejecute el entrenamiento para ver esta figura.", className="nota")),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Hallazgos principales del EDA"), html.Ul([
            html.Li([html.B("Epidemias cíclicas: "), "2010, 2013, 2019 y 2023–2024; 2024 fue récord (261 382 casos), con 37 % de municipio-semanas en brote."]),
            html.Li([html.B("El Niño: "), "los picos epidémicos siguen a episodios de El Niño con meses de rezago (2023 → 2024); el ONI del mismo año, por sí solo, explica poco."]),
            html.Li([html.B("Estacionalidad: "), "el pico de casos (semanas 15–20) coincide con la primera temporada de lluvias."]),
            html.Li([html.B("Clima débil pero consistente: "), "temperaturas medias de 25–28 °C concentran la mayor tasa de brote; las anomalías climáticas anteceden a los casos por 8–12 semanas."]),
            html.Li([html.B("La historia manda: "), "los casos recientes frente al umbral del canal endémico son el mejor predictor; el clima anticipa el riesgo por sí solo, pero su efecto ya se refleja en los casos recientes."]),
            html.Li([html.B("Calidad de datos: "), f"sin nulos, pero {fg.pct(imputada)} de temperatura imputada; los atípicos son epidemias reales y se conservan."]),
        ])),
    ])


@callback(Output("eda-canal", "figure"), Input("eda-mun", "value"))  # Canal endémico del municipio elegido
def actualizar_canal(mun):
    s = CANAL[CANAL["COD_MUN_N"] == mun]  # Serie del municipio (precalculada, 2015–2024)
    serie = {"fecha": s["week_start"], "casos": s["casos_totales"], "umbral": s["umbral"], "brote": s["brote"]}  # Columnas

    return fg.fig_canal(pd.DataFrame(serie), nombre_municipio(mun))  # Figura


@callback(Output("eda-deciles", "figure"), Input("eda-clima", "value"))  # Tasa de brote por decil climático
def actualizar_deciles(var):
    return fg.fig_tasa_por_decil(deciles(var), var)  # Figura
