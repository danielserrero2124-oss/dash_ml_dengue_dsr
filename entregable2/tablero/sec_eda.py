"""Pestaña 2 — Análisis exploratorio de datos, organizado en sub-pestañas que siguen la guía del entregable."""

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from dash import Input, Output, callback, dcc, html  # Componentes y callbacks

from tablero import figuras as fg  # Figuras
from tablero.componentes import (control, desplegable, grafico, kpi, lectura, miles, numero, pct, pestanas_internas,  # Interfaz
                                 seccion, tabla, tabla_df, tarjeta)
from tablero.datos import A, FEATURES, M, NOMBRES, rezago_espacial  # Artefactos

SUB = [("calidad", "Datos y calidad"), ("objetivo", "Variable objetivo"), ("uni", "Univariado"), ("bi", "Bivariado"),
       ("multi", "Multivariado"), ("fuga", "Fuga de datos"), ("temporal", "Temporal"), ("espacial", "Espacial"),
       ("st", "Espacio-temporal"), ("prep", "Preprocesamiento")]  # Sub-pestañas


def seccion_eda():  # Contenedor de la pestaña
    return html.Div([
        html.Div(html.Div([html.H2(["Lo que dicen los ", html.Span("datos", className="resaltado"), " antes de modelar"]),  # Banner
                           html.P("Todas las decisiones se toman con las semanas 2010–2021: la prueba 2023–2024 queda sellada. "
                                  "Cada resultado trae su interpretación.")]), className="banner banner-vector"),
        pestanas_internas("eda-sub", SUB, "calidad"),  # Sub-pestañas
        html.Div(id="eda-contenido"),  # Contenido
    ])


@callback(Output("eda-contenido", "children"), Input("eda-sub", "value"))  # Cambia de sub-pestaña
def mostrar(sub):
    return {"calidad": calidad, "objetivo": objetivo, "uni": univariado, "bi": bivariado, "multi": multivariado, "fuga": fuga,
            "temporal": temporal, "espacial": espacial, "st": espacio_temporal, "prep": preprocesamiento}[sub]()  # Construye


# ---------------------------------------------------------------------------
def calidad():  # Datos y calidad
    c, k = A["calidad"], A["contexto"]["kpi"]  # Artefactos
    smd = c["smd"]  # Diferencias estandarizadas
    rep = c["representatividad"]  # Representatividad
    return html.Div([
        html.Div([kpi("Filas × columnas", f"{miles(k['filas'])} × {k['columnas']}", "sin celdas nulas"),
                  kpi("Temperatura imputada", pct(c["frac_imputada"]), "faltante oculto"),
                  kpi("Duplicados", "0", "exactos y por clave"),
                  kpi("Tamaño efectivo", "≈ 4 800", "observaciones independientes (ICC 0,075)")], className="rejilla cols-4"),
        html.Div([
            tarjeta(html.H3("Patrón de faltantes: temperatura imputada por municipio y año"), grafico(figura=fg.fig_matriz_faltantes(c["matriz_faltantes"])),
                    lectura(f"No hay celdas nulas, pero la temperatura está imputada en el {pct(c['frac_imputada'])} de las semanas. El patrón es por "
                            f"municipio, no por semana: {c['patron']['siempre imputada']} municipios nunca tienen temperatura medida, "
                            f"{c['patron']['nunca imputada']} la tienen siempre y {c['patron']['parcialmente']} parcialmente, de forma estable en el tiempo.")),
            tarjeta(html.H3("Temperatura observada frente a imputada"), grafico(figura=fg.fig_temp_imputada(c["hist_temp"])),
                    lectura("La imputación original es casi una constante (≈ 22,4 °C): borra la diferencia climática entre municipios y crea un "
                            "pico artificial. Por eso se conserva la bandera temp_missing como predictora.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Mecanismo de los faltantes"),
                    tabla(["Evidencia", "Resultado"], [
                        ["Prueba de Little (H0: MCAR), n = 20 000", f"d² = {miles(c['little']['d2'])}, gl = {c['little']['gl']}, p < 0,001"],
                        ["Diferencia estandarizada en log población", numero(smd["log_poblacion"], 2)],
                        ["Diferencia estandarizada en log casos", numero(smd["log_casos"], 2)],
                        ["Diferencia estandarizada en NDVI", numero(smd["ndvi_mean"], 2)],
                        ["AUC al predecir el faltante con variables observadas", "≈ 0,67"],
                    ]),
                    lectura("No es MCAR: los municipios sin temperatura son mucho más pequeños y con menos casos. La falta se explica en parte con "
                            "variables observadas y no cambia con el tiempo: es MAR estructural (municipios sin estación meteorológica). Un componente "
                            "MNAR no es verificable. Decisión: indicador de faltante, sin reimputar con vecinos (la altitud cambia mucho entre ellos).")),
            tarjeta(html.H3("Valores imposibles, inconsistentes y atípicos"),
                    tabla(["Chequeo", "Filas con problema"], [[n, miles(v)] for n, v in c["chequeos"].items()], numericas=(1,)),
                    tabla(["Variable", "Atípicos (Tukey)"], [[fg.etiqueta(v), pct(x / 100)] for v, x in c["atipicos"].items()], numericas=(1,)),
                    lectura("No hay valores imposibles: edades y sexo suman el total, los rangos físicos son válidos y la frecuencia es estrictamente "
                            "semanal. Los atípicos de casos son epidemias reales (las 8 semanas con más casos son de Cali en 2024): no se eliminan, "
                            "se comprimen con log(1 + x). El 18 % de atípicos de temperatura es un artefacto de la imputación.")),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Sesgos de muestreo y representatividad"),
                tabla(["Región", "Municipios del país", "En el panel", "Cobertura", "Participación en el panel", "Participación en el país"],
                      [[r, miles(f["país"]), miles(f["panel"]), pct(f["cobertura"]), pct(f["participación panel"]), pct(f["participación país"])] for r, f in rep.iterrows()],
                      numericas=(1, 2, 3, 4, 5)),
                lectura(f"El panel tiene 376 de 1 122 municipios (33,5 %) pero cubre {numero(c['pob_cubierta'] / 1e6, 1)} millones de habitantes (≈ 60 % del país): "
                        "contiene las capitales y los municipios endémicos. El Caribe y la Orinoquía están sobrerrepresentados y la región Andina "
                        "y la Pacífica, subrepresentadas. Las conclusiones aplican a municipios con transmisión habitual; además, los casos son los "
                        "notificados (subregistro variable) y la temperatura falta justo en los municipios pequeños.")),
    ])


# ---------------------------------------------------------------------------
def objetivo():  # Variable objetivo
    o, k = A["objetivo"], A["contexto"]["kpi"]  # Artefactos
    mapa = M.merge(o["mapa"].reset_index(), on="COD_MUN_N")  # Prevalencia por municipio
    return html.Div([
        html.Div([kpi("Prevalencia en entrenamiento", pct(k["prev_tr"]), f"{miles(k['brotes_tr'])} brotes"),
                  kpi("Razón no brote : brote", numero((1 - k["prev_tr"]) / k["prev_tr"], 1), "desbalance moderado"),
                  kpi("Prevalencia en validación 2022", pct(k["prev_va"]), f"{miles(k['n_va'])} filas"),
                  kpi("Rango anual", f"{pct(o['anual']['mean'].min())} – {pct(o['anual']['mean'].max())}", "2017 frente a 2019")], className="rejilla cols-4"),
        tarjeta(html.H3("Canal endémico: así se define el brote"),
                html.Div([control("Municipio", desplegable("eda-canal", [{"label": NOMBRES[m], "value": int(m)} for m in M["COD_MUN_N"]], 76001))],
                         style={"maxWidth": "420px"}),
                grafico("eda-canal-fig"),
                lectura("La línea punteada es el umbral: el tercer cuartil de las mismas semanas (± 2) de los cinco años anteriores. Los puntos "
                        "rojos son semanas de brote (casos sobre el umbral y al menos 3). El objetivo del modelo es predecir esos puntos cuatro "
                        "semanas antes de que ocurran.")),
        html.Div(style={"height": "18px"}),
        html.Div([
            tarjeta(html.H3("Prevalencia del brote por año (texto: número de brotes)"), grafico(figura=fg.fig_prevalencia_anual(o["anual"])),
                    lectura("El desbalance cambia mucho entre años: del 1,7 % en 2017 al 25 % en 2019. Por eso la exactitud es engañosa (no alertar "
                            "nunca acierta el 90 %), se priorizan AUC-PR, MCC y Brier, y la validación usa años completos.")),
            tarjeta(html.H3("Prevalencia por mes y por región"), grafico(figura=fg.fig_mes_region(o["mes"], o["region"])),
                    lectura("La estacionalidad del objetivo es débil (9–12 % según el mes): el canal endémico ya descuenta la estacionalidad "
                            "habitual de cada municipio. Las regiones difieren poco; la Amazonía tiene la prevalencia más alta.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Municipios en brote y casos nacionales (2015–2021)"), grafico(figura=fg.fig_objetivo_tiempo(o["nacional"])),
                    lectura("Hay ondas epidémicas nacionales (2015–2016 y 2019) separadas por años casi sin brotes (2017–2018): la prevalencia de "
                            "la prueba puede ser muy distinta a la del entrenamiento.")),
            tarjeta(html.H3("Proporción de semanas con brote por municipio"), grafico(figura=fg.fig_mapa_valor(mapa, "prevalencia", "brote", ".0%", (0, 0.3))),
                    lectura("La proporción de semanas en brote varía entre municipios (mediana ≈ 8 %), pero como cada municipio se compara consigo "
                            "mismo, tener muchos casos no implica tener más brotes.")),
        ], className="rejilla cols-2"),
    ])


@callback(Output("eda-canal-fig", "figure"), Input("eda-canal", "value"))  # Canal del municipio elegido
def canal(mun):
    return fg.fig_canal(A["objetivo"]["canal"][mun], NOMBRES[mun])  # Figura


# ---------------------------------------------------------------------------
def univariado():  # Análisis unidimensional
    u = A["univariado"]  # Artefactos
    opciones = [{"label": fg.etiqueta(v) + " (original)", "value": v} for v in u["tabla"].index] + [{"label": fg.etiqueta(v), "value": v} for v in FEATURES]  # Variables
    t = u["tabla"].copy()  # Tabla
    t.index = [fg.etiqueta(v) for v in t.index]  # Nombres legibles
    cat = u["categoricas"]  # Categóricas
    raros = cat["departamento"][cat["departamento"] < 3]  # Departamentos raros
    return html.Div([
        html.Div([
            tarjeta(html.H3("Distribución de una variable"),
                    control("Variable", desplegable("eda-uni-var", opciones, "casos_totales")), grafico("eda-uni-fig"),
                    lectura("Arriba el histograma y abajo la caja de Tukey (bigotes a 1,5 RIC). Los casos, la población y el flujo se muestran en "
                            "escala log(1 + x) por sus colas extremas.")),
            tarjeta(html.H3("Estadísticos de las variables originales (2010–2021)"), tabla_df(t, dec=2),
                    lectura("Los casos son extremadamente asimétricos (asimetría > 18, curtosis en cientos, mediana 0): muchas semanas tranquilas y "
                            "pocas explosivas. La temperatura es bimodal por la imputación. Ninguna variable es normal (todas las pruebas rechazan "
                            "con 5 000 filas): no es un problema para modelos lineales en las predictoras, pero justifica el logaritmo y las "
                            "pruebas no paramétricas.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Variables categóricas"),
                    tabla(["Variable", "Categorías", "Observación"], [
                        ["Departamento", "32", f"{len(raros)} con menos de 3 municipios ({', '.join(raros.index)})"],
                        ["Región", "5", ", ".join(f"{r}: {n}" for r, n in cat["region"].items())],
                        ["Fase ENSO (semanas)", "3", ", ".join(f"{f}: {n}" for f, n in cat["enso"].items())],
                        ["Mes / semana", "12 / 53", "La semana 53 solo existe en 2015 y 2020"],
                        ["Temperatura imputada", "2", "64 % de las filas"],
                    ]),
                    lectura("Las categorías raras (departamentos con 1–2 municipios, semana 53) no se codifican como indicadoras: el departamento "
                            "haría memorizar lugares, y la semana se codifica con seno y coseno, que pone la semana 53 junto a la 1.")),
            tarjeta(html.H3("Predictoras derivadas (entrenamiento)"),
                    html.Div(tabla_df(u["derivadas"][["mean", "std", "5%", "50%", "95%", "asimetría"]].rename(index=fg.etiqueta), dec=2), className="caja-scroll"),
                    lectura("Tras el logaritmo, la historia de casos sigue sesgada (la mayoría de semanas tienen 0 casos), pero las colas pasan de "
                            "cientos a pocas unidades de curtosis. La incidencia conserva una cola larga.")),
        ], className="rejilla cols-2"),
    ])


@callback(Output("eda-uni-fig", "figure"), Input("eda-uni-var", "value"))  # Histograma de la variable elegida
def hist(v):
    return fg.fig_histograma(A["univariado"]["hist"][v], v)  # Figura


# ---------------------------------------------------------------------------
def bivariado():  # Análisis bidimensional
    b = A["bivariado"]  # Artefactos
    pares = [{"label": f"{fg.etiqueta(p.split('|')[1])} vs {fg.etiqueta(p.split('|')[0])}", "value": p} for p in b["pares"]]  # Pares
    asoc = b["asociacion"].sort_values("mi", ascending=False)  # Asociación
    vif = b["vif"].sort_values(ascending=False)  # VIF
    cr = b["cramer"].copy()  # Cramér
    return html.Div([
        html.Div([
            tarjeta(html.H3("Numéricas contra numéricas (densidad conjunta)"),
                    control("Par de variables", desplegable("eda-bi-par", pares, pares[0]["value"])), grafico("eda-bi-par-fig"), html.Div(id="eda-bi-par-txt"),
                    lectura("Con más de 100 000 filas, un diagrama de dispersión se satura; el mapa de densidad (equivalente a hexbin) muestra dónde se "
                            "concentran los datos. La temperatura y la lluvia están poco correlacionadas; la historia de casos es muy persistente "
                            "(ρ ≈ 0,78 entre la semana actual y la media de 8 semanas); el ONI rezagado se asocia positivamente con los casos.")),
            tarjeta(html.H3("Tasa de brote por decil de una predictora"),
                    control("Variable", desplegable("eda-bi-dec", [{"label": fg.etiqueta(v), "value": v} for v in FEATURES], "temp_mean_m8")), grafico("eda-bi-dec-fig"),
                    lectura("Revela relaciones no lineales: con la temperatura medida, el riesgo sube hacia los 25–28 °C y no sigue subiendo; con la "
                            "historia de casos, el riesgo crece de forma monótona y fuerte. Una regresión logística lineal no puede representar "
                            "la forma de U o de meseta sin transformaciones.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Asociación de cada predictora con el brote"),
                    dcc.RadioItems(id="eda-bi-met", options=["Información mutua", "|r biserial de rangos|", "AUC univariado"], value="Información mutua",
                                   className="opciones", inline=True), grafico("eda-bi-met-fig"),
                    lectura(f"La historia de casos domina (rojo): AUC univariados de 0,77–0,88 y la mayor información mutua. El flujo de personas, "
                            f"el ONI y la población tienen asociación moderada. El clima local tiene efectos pequeños (|r_rb| < 0,15) aunque "
                            f"{int((asoc['q_fdr'] < 0.05).sum())} de 34 predictoras son significativas tras FDR: con n grande, el tamaño de efecto "
                            f"dice más que el valor p.")),
            tarjeta(html.H3("Matriz de correlación de Spearman"), grafico(figura=fg.fig_correlacion(b["corr"])),
                    lectura("Hay bloques muy correlacionados: los rezagos y medias móviles de una misma variable (|ρ| > 0,9 en 13 pares).")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Multicolinealidad (VIF)"),
                    tabla(["Variable", "VIF"], [[fg.etiqueta(v), numero(x, 1)] for v, x in vif.head(12).items()], numericas=(1,)),
                    lectura("VIF ≫ 10 en las medias móviles de temperatura, NDVI y casos. No daña la predicción, pero vuelve inestables los "
                            "coeficientes: por eso los modelos base usan penalización L2 y los coeficientes se leen por grupos.")),
            tarjeta(html.H3("Categóricas contra numéricas: por región"),
                    control("Variable", desplegable("eda-bi-reg", [{"label": fg.etiqueta(v), "value": v} for v in b["cajas_region"]], "temp_mean_m8")),
                    grafico("eda-bi-reg-fig"),
                    tabla(["Variable", "H de Kruskal-Wallis", "ε²", "p (Holm)"], [[fg.etiqueta(v), miles(f["H"]), numero(f["ε²"]), "< 0,001"] for v, f in b["kruskal"].iterrows()],
                          numericas=(1, 2)),
                    lectura("Todo es significativo, pero el tamaño de efecto separa lo importante: la temperatura difiere muchísimo entre regiones "
                            "(ε² ≈ 0,41), la lluvia moderadamente y la historia de casos casi nada. Por eso se usan anomalías respecto a la normal "
                            "de cada municipio.")),
            tarjeta(html.H3("Categóricas contra categóricas: V de Cramér con el objetivo"),
                    tabla(["Variable", "Categorías", "χ²", "V de Cramér"], [[v, str(f["categorías"]), miles(f["χ²"]), numero(f["V de Cramér"])] for v, f in cr.iterrows()],
                          numericas=(1, 2, 3)),
                    tabla(["Fase ENSO", "Brote en 4 semanas"], [[str(f), pct(r[1])] for f, r in b["enso"].iterrows()], numericas=(1,)),
                    lectura("La fase de El Niño es la categórica más asociada (V ≈ 0,17): durante El Niño el brote en 4 semanas es casi cuatro veces "
                            "más frecuente que en La Niña. El mes y la región explican poco, y el departamento no se usa porque memorizaría lugares.")),
        ], className="rejilla cols-3"),
    ])


@callback(Output("eda-bi-par-fig", "figure"), Output("eda-bi-par-txt", "children"), Input("eda-bi-par", "value"))  # Densidad del par elegido
def par(p):
    d = A["bivariado"]["pares"][p]  # Datos del par
    x, y = p.split("|")  # Variables
    return fg.fig_densidad(d, x, y), html.P(f"Pearson {numero(d['pearson'])} · Spearman {numero(d['spearman'])} · n = {miles(d['n'])}", className="nota")  # Figura y correlaciones


@callback(Output("eda-bi-dec-fig", "figure"), Input("eda-bi-dec", "value"))  # Deciles
def deciles(v):
    return fg.fig_deciles(A["bivariado"]["deciles"][v], v)  # Figura


@callback(Output("eda-bi-met-fig", "figure"), Input("eda-bi-met", "value"))  # Asociación
def asociacion(met):
    return fg.fig_asociacion(A["bivariado"]["asociacion"], met)  # Figura


@callback(Output("eda-bi-reg-fig", "figure"), Input("eda-bi-reg", "value"))  # Cajas por región
def cajas(v):
    return fg.fig_cajas_region(A["bivariado"]["cajas_region"], v)  # Figura


# ---------------------------------------------------------------------------
def multivariado():  # Análisis multivariado
    mu = A["multivariado"]  # Artefactos
    acum = np.cumsum(mu["varianza"])  # Varianza acumulada
    perfil = mu["perfil"].groupby("grupo").agg(municipios=("prevalencia", "size"), prevalencia=("prevalencia", "mean"),
                                               temperatura=("temp", "mean"), lpob=("lpob", "mean"), flujo=("flujo", "mean"))  # Perfil de los grupos
    top = {c: ", ".join(fg.etiqueta(v) for v in mu["cargas"][c].abs().sort_values(ascending=False).head(4).index) for c in mu["cargas"]}  # Cargas dominantes
    return html.Div([
        tarjeta(html.H3("Componentes principales (PCA)"), grafico(figura=fg.fig_pca(mu["varianza"], mu["pca_xy"], mu["pca_y"])),
                tabla(["Componente", "Varianza", "Variables dominantes"], [[c, pct(mu["varianza"][i]), top[c]] for i, c in enumerate(top)], numericas=(1,)),
                lectura(f"{int(np.searchsorted(acum, 0.8) + 1)} componentes explican el 80 % de la varianza y {int((mu['autovalores'] > 1).sum())} tienen autovalor > 1: "
                        "la dimensionalidad efectiva es mucho menor que 34 porque las predictoras forman bloques redundantes. CP1 es el eje de "
                        "historia de casos; CP2–CP3, temperatura y vegetación; CP4, lluvia. Los brotes se concentran hacia un extremo de CP1 pero "
                        "se superponen con los no brotes.")),
        html.Div(style={"height": "18px"}),
        html.Div([
            tarjeta(html.H3("t-SNE (solo visual)"),
                    dcc.RadioItems(id="eda-tsne", options=[{"label": "Colorear por objetivo", "value": "objetivo"}, {"label": "Colorear por región", "value": "region"}],
                                   value="objetivo", className="opciones", inline=True), grafico("eda-tsne-fig"),
                    lectura("Los grupos locales coinciden más con la región (45 % de vecinos de la misma región frente a 31 % al azar) que con el "
                            "objetivo (88 % frente a 81 %): el brote no es una subpoblación, es un estado transitorio que depende de la historia reciente.")),
            tarjeta(html.H3("Atípicos multivariados"), tabla_df(mu["atipicos"], dec=3),
                    lectura("Las filas marcadas por Mahalanobis e Isolation Forest tienen una tasa de brote cuatro veces mayor: son semanas epidémicas, "
                            "grandes ciudades (" + ", ".join(mu["atipicos_municipios"].index[:3]) + ") o climas extremos. Se conservan.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3(f"Agrupamiento de municipios (k-medias, k = {mu['k']})"), grafico(figura=fg.fig_grupos(mu["perfil"]))),
            tarjeta(html.H3("Perfil de los grupos"), tabla_df(perfil, dec=2),
                    html.P("Silueta por k: " + ", ".join(f"{kk}: {numero(v, 2)}" for kk, v in mu["silueta"].items()), className="nota"),
                    lectura("Estructura débil a moderada (silueta 0,33). Los dos perfiles se distinguen por tamaño y movilidad, no por clima: los "
                            "municipios grandes y conectados tienen más del doble de semanas en brote. Respalda usar población y flujo como "
                            "predictoras y validar por bloques espaciales.")),
        ], className="rejilla cols-2"),
    ])


@callback(Output("eda-tsne-fig", "figure"), Input("eda-tsne", "value"))  # t-SNE
def tsne(color):
    mu = A["multivariado"]  # Artefactos
    return fg.fig_tsne(mu["tsne"], mu["tsne_y"], mu["tsne_region"], color)  # Figura


# ---------------------------------------------------------------------------
def fuga():  # Auditoría de fuga
    f = A["fuga"]  # Artefactos
    return html.Div([
        html.Div([
            tarjeta(html.H3("AUC univariado: alerta si se acerca a 1"), grafico(figura=fg.fig_auc_univariado(f["auc_univariado"])),
                    lectura("Ninguna de las 34 predictoras supera 0,9 (la más fuerte, la media de 4 semanas del log de casos, llega a ≈ 0,88). Los casos "
                            "de la semana objetivo, en rojo, alcanzan ≈ 0,96: es la firma de una fuga y por eso esa columna se descarta. Los "
                            "identificadores (gris) no se usan porque memorizarían cuándo y dónde hubo brotes.")),
            html.Div([
                tarjeta(html.H3("Disponibilidad en la semana de la predicción"),
                        tabla(["Familia", "¿Disponible en t?", "Justificación"], [
                            ["Historia de casos", "Sí", "Semanas t, t−1, …"],
                            ["Umbral objetivo u(t+4)", "Sí", "Usa semanas t−258 a t−46 (años anteriores)"],
                            ["Clima y anomalías", "Sí", "Medias móviles hacia atrás; normal estimada con 2010–2021"],
                            ["El Niño", "Sí*", "El ONI mensual tiene días de rezago: oni_l8 no tiene riesgo"],
                            ["Casos de t+4", "No", "Es información del objetivo: descartada"],
                            ["Identificadores", "Excluidos", "Municipio, año, índice de semana"],
                        ])),
                tarjeta(html.H3("Demostración: cuánto infla la fuga el desempeño"),
                        tabla(["Esquema", "AUC-ROC (validación cruzada)"], [[n, numero(v)] for n, v in f["demo"].items()], numericas=(1,)),
                        lectura("Con la validación correcta el AUC ronda 0,90; mezclar semanas al azar lo infla un poco y usar los casos de la semana "
                                "objetivo lo lleva casi a 1: un modelo «perfecto» e inútil. Además, las particiones no comparten filas y la purga "
                                "de 4 semanas impide reutilizar etiquetas entre periodos.")),
            ], className="columna"),
        ], className="rejilla cols-2"),
    ])


# ---------------------------------------------------------------------------
def temporal():  # Componente temporal
    t = A["temporal"]  # Artefactos
    est = t["estacionariedad"]  # ADF y KPSS
    der = t["deriva"]  # PSI
    return html.Div([
        html.Div([kpi("Frecuencia", "semanal", "7 días exactos, sin huecos ni duplicados"), kpi("Fuerza de la tendencia", numero(t["fuerza"]["tendencia"], 2), "STL"),
                  kpi("Fuerza estacional", numero(t["fuerza"]["estacional"], 2), "STL"), kpi("ACF en el rezago 4", numero(t["acf"][4], 2), "serie nacional")],
                 className="rejilla cols-4"),
        html.Div([
            tarjeta(html.H3("Serie nacional y agregaciones"),
                    dcc.RadioItems(id="eda-serie", options=[{"label": "Semana", "value": "semana"}, {"label": "Mes", "value": "mes"}, {"label": "Año", "value": "anio"}],
                                   value="semana", className="opciones", inline=True), grafico("eda-serie-fig"),
                    lectura("Ondas epidémicas de varios años (2010, 2013, 2019) con valles en 2011–2012 y 2017–2018; la caída de 2020 coincide con la "
                            "pandemia (de ≈ 2 950 a ≈ 1 170 casos por semana entre el primer trimestre y el segundo).")),
            tarjeta(html.H3("Estacionalidad por ciclo"), grafico(figura=fg.fig_estacionalidad(t["calendario"])),
                    lectura("Algo más de casos en el primer semestre (tras la primera temporada de lluvias), pero el ciclo anual es débil frente a "
                            "las ondas de varios años; el efecto de fin de año (festividades) es de solo ≈ 3 %.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Descomposición STL"), grafico(figura=fg.fig_stl(t["stl"])),
                    lectura(f"La tendencia domina (fuerza {numero(t['fuerza']['tendencia'], 2)}) y la estacionalidad es casi nula ({numero(t['fuerza']['estacional'], 2)}): "
                            "la información útil está en el nivel reciente de la serie (rezagos y medias móviles), no en el calendario.")),
            html.Div([
                tarjeta(html.H3("Estacionariedad"), tabla_df(est, dec=3),
                        html.P(f"Por municipio: ADF rechaza raíz unitaria en el {pct(t['mun_adf'], 0)}; KPSS rechaza estacionariedad en el {pct(t['mun_kpss'], 0)}.", className="nota"),
                        lectura("En la serie nacional ADF no rechaza la raíz unitaria y KPSS tampoco rechaza la estacionariedad: serie muy persistente "
                                "con ondas largas. La diferencia es estacionaria. Consecuencia: variables relativas al municipio y validación en varios años.")),
                tarjeta(html.H3("Regímenes y El Niño"), grafico(figura=fg.fig_regimenes(t["stl"], t["cortes"], t["nino"])),
                        lectura("Puntos de cambio (ámbar): " + ", ".join(str(c.date()) for c in t["cortes"]) + ". Los episodios de El Niño (franjas) anteceden "
                                "a los regímenes altos.")),
            ], className="columna"),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("ACF y PACF de la serie nacional"), grafico(figura=fg.fig_acf(t["acf"], t["pacf"], t["n"])),
                    lectura("La ACF decae lentamente (proceso persistente) y es casi 0 en el rezago 52 (no hay ciclo anual fuerte); la PACF se concentra "
                            "en los primeros rezagos: justifica rezagos de 1–3 semanas y medias de 4, 8 y 12.")),
            tarjeta(html.H3("Correlación cruzada: clima en t − k frente a casos en t"), grafico(figura=fg.fig_ccf(t["ccf"])),
                    lectura("El clima local tiene correlaciones pequeñas (|r| < 0,06); la lluvia y el NDVI llegan con 12–15 semanas de rezago. El ONI es "
                            "la señal más fuerte (r ≈ 0,17, unas 5 semanas de rezago): se usan medias móviles largas y el ONI rezagado.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Deriva de las covariables (PSI)"), tabla_df(der.rename(index=fg.etiqueta), dec=3),
                    html.P("Prevalencia del objetivo: " + " · ".join(f"{k}: {pct(v)}" for k, v in t["prevalencias"].items()), className="nota"),
                    lectura("Deriva baja para la historia de casos y el clima local (PSI < 0,1), moderada hacia 2022 para la temperatura anómala y la "
                            "lluvia, y muy fuerte para el ONI (cada periodo tiene una fase distinta de El Niño). La prevalencia también cambia (prior shift).")),
            tarjeta(html.H3("Concept drift: AUC univariado año a año"), grafico(figura=fg.fig_deriva(t["concepto"])),
                    lectura("El concepto es estable para la historia de casos (AUC 0,83–0,89 todos los años) e inestable para el ONI y la anomalía de "
                            "temperatura, que llegan a quedar por debajo de 0,5. La relación casos → brote es la base confiable.")),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Panel: brote en 4 semanas por municipio (filas, ordenadas por región) y semana"), grafico(figura=fg.fig_calor(t["calor"], t["calor_regiones"])),
                lectura("Los brotes llegan en ondas sincronizadas (franjas verticales, como en 2019) y también como focos locales aislados: hay "
                        "heterogeneidad entre municipios en nivel y en tiempo, lo que justifica normalizar por municipio y evaluar por conglomerados.")),
    ])


@callback(Output("eda-serie-fig", "figure"), Input("eda-serie", "value"))  # Serie con la agregación elegida
def serie(nivel):
    t = A["temporal"]  # Artefactos
    return fg.fig_serie(t["nacional"], t["mensual"], A["contexto"]["anual"].loc[:2021], nivel)  # Figura


# ---------------------------------------------------------------------------
def espacial():  # Componente espacial
    s = A["espacial"]  # Artefactos
    mapa = s["mapa"]  # Municipios con indicadores
    ce, mo = s["clark_evans"], s["moran"]  # Patrón de puntos y Moran
    sens = s["sensibilidad"]  # Sensibilidad
    opciones = [{"label": "Log incidencia anual", "value": "log_incidencia"}, {"label": "Proporción de semanas con brote", "value": "prevalencia"},
                {"label": "Temperatura medida (°C)", "value": "temp_mean_m4"}, {"label": "Lluvia (mm/semana)", "value": "prec_total_m4"},
                {"label": "Flujo de personas (log)", "value": "flujo_m4"}]  # Variables del mapa
    return html.Div([
        html.Div([kpi("Sistema de referencia", "EPSG:4326", "centroides DIVIPOLA válidos"), kpi("Clark-Evans R", numero(ce["R"]), "< 1: agrupados"),
                  kpi("I de Moran (incidencia)", numero(mo["log_incidencia"]["I"]), f"p = {numero(mo['log_incidencia']['p'])}"),
                  kpi("Focos LISA Alto-Alto", str(int((mapa["LISA"] == "Alto-Alto").sum())), f"{s['fdr_lisa']} significativos tras FDR")], className="rejilla cols-4"),
        html.Div([
            tarjeta(html.H3("Mapa de una variable"), control("Variable", desplegable("eda-esp-var", opciones, "log_incidencia")), grafico("eda-esp-fig"),
                    lectura("Las coordenadas son centroides: no hay polígonos municipales, así que se usan mapas de símbolos. La incidencia se agrupa "
                            "en el centro y el oriente (Tolima, Huila, Meta); la temperatura solo aparece donde se mide.")),
            tarjeta(html.H3("Cobertura espacial frente al resto del país"), grafico(figura=fg.fig_cobertura(mapa, s["fuera"])),
                    lectura("El panel cubre la región Andina y el Caribe de forma densa y deja vacíos en la Orinoquía y la Amazonía: sesgo de muestreo "
                            "espacial que hace que las métricas globales estén dominadas por las zonas densas.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Función L de Ripley y DBSCAN"), grafico(figura=fg.fig_ripley(s["ripley"])),
                    lectura(f"Los municipios están agrupados: Clark-Evans R = {numero(ce['R'])} (vecino más cercano a {numero(ce['r_obs'], 1)} km frente a "
                            f"{numero(ce['r_esp'], 1)} km esperados) y la función L está sobre la envolvente de aleatoriedad en todas las escalas. "
                            f"DBSCAN (haversine, 30 km) encuentra {mapa['dbscan'].nunique() - 1} núcleos y "
                            f"{int((mapa['dbscan'] == 'ruido').sum())} municipios aislados.")),
            tarjeta(html.H3("Agrupamientos por densidad (DBSCAN)"), grafico(figura=fg.fig_dbscan(mapa))),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Diagrama de dispersión de Moran (k = 6 vecinos)"), grafico(figura=fg.fig_moran(mapa, rezago_espacial(mapa["log_incidencia"]))),
                    tabla(["Matriz de pesos", "Sin vecinos", "I de Moran", "z"], [[n, str(f["sin vecinos"]), numero(f["I"]), numero(f["z"], 1)] for n, f in sens.iterrows()],
                          numericas=(1, 2, 3)),
                    lectura("Se usan k = 6 vecinos más cercanos (haversine) porque el panel no tiene todos los municipios y la contigüidad no aplica. El "
                            "I de Moran es positivo y significativo con cualquier matriz: municipios vecinos tienen cargas parecidas. La proporción de "
                            f"semanas en brote, en cambio, tiene I ≈ {numero(mo['prevalencia']['I'], 2)} (p ≈ {numero(mo['prevalencia']['p'], 2)}): el canal endémico "
                            "normaliza el nivel de cada zona.")),
            tarjeta(html.H3("Indicadores locales"),
                    dcc.RadioItems(id="eda-local", options=[{"label": "LISA", "value": "lisa"}, {"label": "Getis-Ord Gi*", "value": "gi"}], value="lisa",
                                   className="opciones", inline=True), grafico("eda-local-fig"),
                    lectura("Focos calientes (Alto-Alto, Gi* > 1,96) en el Meta, el Tolima, el Huila y el Putumayo; focos fríos en municipios de montaña. "
                            "Tras corregir las 376 pruebas por FDR siguen siendo significativos " + str(s["fdr_lisa"]) + " municipios.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Semivariograma empírico"), grafico(figura=fg.fig_vario(s["vario"], float(mapa["log_incidencia"].var()))),
                    lectura("La semivarianza alcanza la meseta hacia los 100 km (el alcance de la dependencia) con un efecto pepita alto (≈ 70 %): "
                            "la dependencia espacial es clara pero parcial.")),
            tarjeta(html.H3("Heterogeneidad espacial y escala (MAUP)"),
                    tabla_df(s["heterogeneidad"][["municipios", "prevalencia", "AUC lc_m4", "AUC oni_l8", "AUC temp_anom_m8", "AUC prec_total_m8"]], dec=3),
                    html.P(f"Correlación temperatura–incidencia: municipios {numero(s['maup']['municipios'], 2)} · departamentos {numero(s['maup']['departamentos'], 2)}. "
                           f"Error de la distancia euclidiana sobre grados: mediana {pct(s['error_grados'][0])}, máximo {pct(s['error_grados'][1])}.", className="nota"),
                    lectura("La historia de casos predice bien en todas las regiones, pero la relación del ONI y del clima cambia de intensidad y la de "
                            "la lluvia incluso de dirección. Al agregar a departamentos la correlación con la temperatura se duplica (MAUP): se trabaja "
                            "a escala municipal, con distancias haversine.")),
        ], className="rejilla cols-2"),
    ])


@callback(Output("eda-esp-fig", "figure"), Input("eda-esp-var", "value"))  # Mapa de la variable elegida
def mapa_var(v):
    mapa = A["espacial"]["mapa"]  # Municipios
    s = mapa[mapa[v].notna()]  # Con dato
    formato = ".0%" if v == "prevalencia" else None  # Formato
    return fg.fig_mapa_valor(s, v, {"log_incidencia": "log incidencia", "prevalencia": "brote", "temp_mean_m4": "°C", "prec_total_m4": "mm",
                                    "flujo_m4": "log flujo"}[v], formato, escala=fg.CALOR)  # Mapa


@callback(Output("eda-local-fig", "figure"), Input("eda-local", "value"))  # LISA o Gi*
def local(tipo):
    mapa = A["espacial"]["mapa"]  # Municipios
    return fg.fig_lisa(mapa) if tipo == "lisa" else fg.fig_gi(mapa)  # Mapa


# ---------------------------------------------------------------------------
def espacio_temporal():  # Componente espacio-temporal
    st = A["espacio_temporal"]  # Artefactos
    anios = sorted(st["anual"]["ANO"].unique())  # Años
    focos = st["focos"]["categoría"].value_counts()  # Conteo de focos
    kx = st["knox"]  # Prueba de Knox
    return html.Div([
        html.Div([
            tarjeta(html.H3("Incidencia anual por municipio"),
                    dcc.Slider(id="eda-st-anio", min=anios[0], max=anios[-1], step=1, value=2019, marks={int(a): str(a) for a in anios}), grafico("eda-st-fig"),
                    lectura("El patrón espacial cambia año a año: 2017 casi no tiene brotes, 2019 los extiende por gran parte del país y los demás años "
                            "muestran focos localizados que se mueven.")),
            tarjeta(html.H3("Focos calientes: Gi* anual + Mann-Kendall"), grafico(figura=fg.fig_focos(st["focos"])),
                    lectura(f"{focos.get('Nunca foco', 0)} municipios nunca fueron foco, {focos.get('Esporádico', 0)} lo fueron de forma esporádica y solo "
                            f"{focos.get('Persistente', 0)} son persistentes ({focos.get('Emergente / intensificándose', 0)} emergentes y {focos.get('Decreciente', 0)} "
                            "decrecientes): los focos se desplazan, por eso la ubicación fija no basta para predecir.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Proporción de municipios en brote por región"), grafico(figura=fg.fig_regiones(st["regiones"])),
                    lectura("Las regiones comparten las grandes ondas pero con desfase: Andina, Amazonía, Orinoquía y Caribe están bastante sincronizadas "
                            "(correlaciones 0,59–0,72) y el Pacífico sigue una dinámica casi propia.")),
            tarjeta(html.H3(f"Prueba de Knox ({miles(st['n_eventos'])} inicios de brote, 999 permutaciones)"),
                    tabla(["Distancia (km)", "Tiempo (semanas)", "Pares observados", "Esperados", "Razón", "p"],
                          [[str(r["distancia (km)"]), str(r["tiempo (semanas)"]), miles(r["observado"]), miles(r["esperado"]), numero(r["razon"], 2), numero(r["p"])]
                           for _, r in kx.iterrows()], numericas=(2, 3, 4, 5)),
                    lectura("En todas las escalas hay más inicios de brote cercanos a la vez en espacio y tiempo de lo esperado (razón 1,1–1,3, p = 0,001): "
                            "los brotes se contagian o comparten causas entre vecinos. Es la evidencia para incluir variables de vecindad rezagadas.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("I de Moran semana a semana"), grafico(figura=fg.fig_moran_semanal(st["moran_semanal"])),
                    lectura(f"La autocorrelación espacial de los casos (centrados por municipio) es positiva casi todas las semanas (mediana "
                            f"{numero(st['moran_semanal'].median(), 2)}) y sube durante las epidemias.")),
            tarjeta(html.H3("¿Anticipa el vecindario al municipio?"), grafico(figura=fg.fig_cruzada(st["cruzada"])),
                    lectura(f"El promedio de los vecinos en t se correlaciona con el municipio en t + 4 (r ≈ {numero(st['cruzada'].loc[4, 'vecinos'], 2)}), menos que la "
                            f"propia historia (r ≈ {numero(st['cruzada'].loc[4, 'propio'], 2)}): el vecindario añade información pero no reemplaza la historia propia.")),
        ], className="rejilla cols-2"),
    ])


@callback(Output("eda-st-fig", "figure"), Input("eda-st-anio", "value"))  # Mapa del año elegido
def anio(a):
    return fg.fig_anual_mapa(A["espacio_temporal"]["anual"], a)  # Mapa


# ---------------------------------------------------------------------------
def preprocesamiento():  # Preprocesamiento
    filas = [
        ["Partición", "Cronológica con purga de 4 semanas", "Autocorrelación; objetivo en t + 4"],
        ["Validación", "Ventana creciente por años (2018–2021) con purga", "Prevalencia y régimen cambian por año"],
        ["Imputación", "Indicador temp_missing + SimpleImputer(mediana) como protección", "Faltantes MAR estructurales"],
        ["Transformación", "log(1 + x) de casos, flujo y población", "Asimetría > 14, curtosis en cientos"],
        ["Escalamiento", "StandardScaler dentro del Pipeline", "Escalas distintas; penalización L2"],
        ["Codificación", "Semana → seno y coseno; departamento no se codifica", "Estacionalidad cíclica; categorías raras"],
        ["Atípicos", "Se conservan; el logaritmo los comprime", "Atípicos = epidemias reales"],
        ["Rezagos y ventanas", "log casos en t…t−3; medias de 4, 8 y 12 semanas", "PACF y ACF persistentes"],
        ["Anomalías climáticas", "Desviación de la normal mensual del municipio (2010–2021)", "Clima muy distinto entre regiones"],
        ["Canal endémico", "Casos frente al umbral objetivo", "El brote es relativo a la historia propia"],
        ["El Niño", "ONI actual y rezagado 8 semanas", "Efecto rezagado y deriva fuerte"],
        ["Regularización", "L2 con C elegido por validación cruzada", "VIF ≫ 10"],
        ["Desbalance", "class_weight None vs balanced en la malla; sin SMOTE", "≈ 10 % de brotes variable por año"],
        ["Coordenadas", "No se usan como predictoras", "Focos que se desplazan; zonas nuevas"],
    ]
    return html.Div([
        html.Div([
            tarjeta(html.H3("Cada decisión y el hallazgo del EDA que la motiva"), tabla(["Paso", "Decisión", "Hallazgo"], filas)),
            html.Div([
                tarjeta(html.H3("Pipeline (ajustado solo con el entrenamiento)"),
                        html.Div("SimpleImputer(median) → StandardScaler → LogisticRegression(L2)\n"
                                 "SimpleImputer(median) → StandardScaler → LinearSVC(L2) → escalado de Platt (2022)", className="codigo"),
                        lectura("Dentro de GridSearchCV el Pipeline se reajusta en cada pliegue con la parte de entrenamiento del pliegue: la mediana "
                                "y la estandarización nunca ven la validación ni la prueba.")),
                tarjeta(html.H3(f"Las {len(FEATURES)} predictoras"),
                        tabla(["Familia", "Variables"], [
                            ["Historia de casos (12)", "log casos t…t−3, medias 4/8/12, brote en t, semanas en brote (8), casos/umbral, umbral, incidencia"],
                            ["Clima y ambiente (18)", "temperatura, lluvia y NDVI (semana y medias 4/8/12), anomalías (8 sem.), ONI y ONI t−8, bandera de imputación"],
                            ["Contexto (4)", "seno y coseno de la semana, log población, flujo de personas (4 sem.)"],
                        ])),
            ], className="columna"),
        ], className="rejilla cols-2"),
    ])
