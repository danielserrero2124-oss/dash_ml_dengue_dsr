"""Pestaña 3 — Modelos base: regresión logística y SVM lineal frente a líneas base triviales."""

import numpy as np  # Cálculo numérico
from dash import Input, Output, callback, dcc, html  # Componentes y callbacks

from tablero import figuras as fg  # Figuras
from tablero.componentes import (control, desplegable, formula, grafico, kpi, lectura, miles, numero, pct, pestanas_internas,  # Interfaz
                                 seccion, tabla, tabla_df, tarjeta)
from tablero.datos import M, MOD, curvas, metricas_umbral, simular  # Artefactos y cálculos ligeros

BASES = ["Regresión logística", "SVM lineal"]  # Modelos base
TODOS = BASES + ["Persistencia (8 sem.)", "Persistencia", "Ingenuo estacional", "Dummy (prior)"]  # Con líneas base
SUB = [("entrenamiento", "Entrenamiento"), ("resultados", "Resultados"), ("diagnostico", "Diagnóstico"),
       ("interpretacion", "Interpretación"), ("simulador", "Simulador"), ("critica", "Nota crítica")]  # Sub-pestañas


def seccion_modelos():  # Contenedor de la pestaña
    lr, svm = MOD["ic"]["Regresión logística"]["valor"], MOD["ic"]["SVM lineal"]["valor"]  # Métricas en prueba
    per = MOD["ic"]["Persistencia (8 sem.)"]["valor"]  # Mejor línea base
    return html.Div([
        html.Div(html.Div([html.H2(["Dos ", html.Span("modelos base", className="resaltado"), " lineales frente a reglas triviales"]),  # Banner
                           html.P("Regresión logística y SVM lineal en un Pipeline, con GridSearchCV y validación cruzada hacia adelante por años "
                                  "con purga. Se evalúan una sola vez en la prueba 2023–2024.")]), className="banner banner-virus"),
        html.Div([kpi("AUC-ROC regresión logística", numero(lr["AUC-ROC"]), f"AUC-PR {numero(lr['AUC-PR'])}"),
                  kpi("AUC-ROC SVM lineal", numero(svm["AUC-ROC"]), f"AUC-PR {numero(svm['AUC-PR'])}"),
                  kpi("Mejor línea base", numero(per["AUC-ROC"]), f"persistencia de 8 semanas · AUC-PR {numero(per['AUC-PR'])}"),
                  kpi("Prevalencia en la prueba", pct(MOD["y_te"].mean()), "triplica la del entrenamiento")], className="rejilla cols-4"),
        pestanas_internas("mod-sub", SUB, "entrenamiento"),  # Sub-pestañas
        html.Div(id="mod-contenido"),  # Contenido
    ])


@callback(Output("mod-contenido", "children"), Input("mod-sub", "value"))  # Cambia de sub-pestaña
def mostrar(sub):
    return {"entrenamiento": entrenamiento, "resultados": resultados, "diagnostico": diagnostico, "interpretacion": interpretacion,
            "simulador": simulador, "critica": critica}[sub]()  # Construye


# ---------------------------------------------------------------------------
def entrenamiento():  # Diseño y validación cruzada
    filas = []  # Tabla de validación cruzada
    for n in BASES:  # Modelos
        t, _ = MOD["cv"][n]  # Resultados
        for _, f in t.iterrows():  # Combinaciones
            filas.append([n, numero(f["C"], 4), f["class_weight"], numero(f["AUC-PR"]), numero(f["AUC-PR desv."]), numero(f["AUC-ROC"]), numero(f["AUC-ROC entrenamiento"])])
    pl = {n: MOD["cv"][n][1] for n in BASES}  # Por pliegue
    return html.Div([
        html.Div([
            tarjeta(html.H3("Diseño experimental"),
                    formula(r"""
**Líneas base triviales** (la referencia mínima): `DummyClassifier` (siempre la prevalencia de entrenamiento), **persistencia**
($\hat y_{t} = b_t$: si hay brote ahora, habrá brote en 4 semanas), **persistencia de 8 semanas** (proporción reciente de semanas en
brote) e **ingenuo estacional** ($\hat y_t = b_{t+4-52}$, el brote de hace un año).

**Validación cruzada hacia adelante con purga.** En el pliegue del año $a \in \{2018, \dots, 2021\}$ se entrena con las filas cuyo
objetivo es anterior a $a$ y se valida con las filas cuya semana $t$ y objetivo $t+4$ caen en $a$. Es `TimeSeriesSplit` por años con
un `gap` igual al horizonte.

**Selección.** Malla de $C$ (inverso de la regularización) y `class_weight` (None o balanced); criterio AUC-PR (clase minoritaria).
El umbral que maximiza $F_1$ y el escalado de Platt de la SVM se ajustan con la validación 2022.
"""),
                    tabla(["Modelo", "Mejores hiperparámetros", "Tiempo de GridSearchCV"],
                          [[n, ", ".join(f"{k.split('__')[1]} = {v}" for k, v in MOD["mejores"][n].items()), f"{numero(MOD['tiempos'][n], 0)} s"] for n in BASES])),
            tarjeta(html.H3("Desempeño de la mejor combinación por pliegue"),
                    tabla(["Año validado"] + [f"{n} · {m}" for n in BASES for m in ("AUC-PR", "AUC-ROC")],
                          [[str(a)] + [numero(pl[n].loc[a, m]) for n in BASES for m in ("AUC-PR", "AUC-ROC")] for a in pl[BASES[0]].index], numericas=(1, 2, 3, 4)),
                    lectura("El desempeño varía con el año (2019, epidémico, es el más fácil en AUC-PR; 2021 el más difícil): una sola partición no basta "
                            "para medir la incertidumbre. Los dos modelos se comportan casi igual en cada pliegue.")),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("GridSearchCV: todas las combinaciones (media de 4 pliegues)"),
                html.Div(tabla(["Modelo", "C", "class_weight", "AUC-PR", "desv.", "AUC-ROC", "AUC-ROC entrenamiento"], filas, numericas=(1, 3, 4, 5, 6)), className="caja-scroll"),
                lectura("Con 130 000 filas la fuerza de la regularización cambia poco el desempeño; C = 0,01 estabiliza los coeficientes ante la "
                        "multicolinealidad. class_weight = balanced sube un poco el AUC-ROC pero baja el AUC-PR (y en la logística empeora la "
                        "calibración), así que no se pondera. El AUC-ROC de entrenamiento y de validación son casi iguales: no hay sobreajuste.")),
    ])


# ---------------------------------------------------------------------------
def resultados():  # Métricas en la prueba
    ic = MOD["ic"]  # Métricas con IC
    mets = ["AUC-ROC", "AUC-PR", "F1", "MCC", "Precision", "Recall", "Accuracy", "Brier"]  # Métricas
    filas = [[n] + [f"{numero(ic[n].loc[mt, 'valor'])} [{numero(ic[n].loc[mt, 'IC 2,5 %'])}; {numero(ic[n].loc[mt, 'IC 97,5 %'])}]" for mt in mets] for n in TODOS]  # Tabla
    dif = MOD["diferencias"]  # Diferencias pareadas
    return html.Div([
        tarjeta(html.H3("Métricas en la prueba 2023–2024 con IC 95 % (bootstrap de 1 000 réplicas por municipio)"),
                tabla(["Modelo"] + mets, filas),
                lectura("Los dos modelos base superan con claridad a todas las líneas base en AUC-ROC, AUC-PR, F1 y MCC. La exactitud engaña: la "
                        "persistencia simple acierta casi lo mismo (≈ 0,82) y el Dummy, que nunca alerta, ya llega a 0,70. El ingenuo estacional es "
                        "casi inútil (AUC ≈ 0,54): los brotes no se repiten en la misma semana del año.")),
        html.Div(style={"height": "18px"}),
        html.Div([
            tarjeta(html.H3("Matriz de confusión con el umbral elegido"),
                    html.Div([control("Modelo", desplegable("mod-cm-modelo", [{"label": n, "value": n} for n in BASES], BASES[0])),
                              control("Umbral de decisión", dcc.Slider(id="mod-umbral", min=0.05, max=0.8, step=0.001, value=round(MOD["umbrales"][BASES[0]], 3),
                                                                       marks={v: numero(v, 1) for v in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)}))],
                             className="rejilla cols-2"),
                    grafico("mod-cm-fig"), html.Div(id="mod-cm-txt"),
                    lectura("Al bajar el umbral sube el recall (menos brotes sin anticipar) y baja la precisión (más falsas alarmas). El umbral inicial "
                            "maximiza F1 en 2022; en la práctica se ajusta a la capacidad de respuesta de la secretaría de salud.")),
            tarjeta(html.H3("Curvas ROC y precisión-recall"),
                    dcc.Checklist(id="mod-curvas", options=TODOS, value=BASES + ["Persistencia (8 sem.)", "Persistencia"], className="opciones", inline=True),
                    grafico("mod-curvas-fig"),
                    lectura("En la curva PR la referencia del azar es la prevalencia (≈ 0,30). Los dos modelos base dominan a las líneas base en casi "
                            "todo el rango; las curvas de la logística y la SVM prácticamente se superponen.")),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Diferencias pareadas (bootstrap por municipio)"),
                tabla(["Comparación", "Métrica", "Diferencia", "IC 2,5 %", "IC 97,5 %", "p"],
                      [[r["comparación"], r["métrica"], numero(r["diferencia"], 4), numero(r["IC 2,5 %"], 4), numero(r["IC 97,5 %"], 4), numero(r["p"])] for _, r in dif.iterrows()],
                      numericas=(2, 3, 4, 5)),
                lectura("Frente a la persistencia, la ganancia de los modelos base es significativa (los intervalos excluyen el 0): en AUC-PR es de "
                        "unas 8 centésimas frente a la persistencia de 8 semanas y de 22 frente a la persistencia simple. Entre la SVM y la logística "
                        "la diferencia es mínima: +0,001 de AUC-ROC a favor de la SVM (significativa por el tamaño de la muestra, irrelevante en "
                        "la práctica) y ninguna diferencia en AUC-PR. Con las mismas variables lineales, ambas encuentran casi el mismo hiperplano.")),
    ])


@callback(Output("mod-umbral", "value"), Input("mod-cm-modelo", "value"))  # Al cambiar de modelo, vuelve a su umbral óptimo
def umbral_inicial(n):
    return round(MOD["umbrales"][n], 3)  # Umbral de máximo F1 en 2022


@callback(Output("mod-cm-fig", "figure"), Output("mod-cm-txt", "children"), Input("mod-cm-modelo", "value"), Input("mod-umbral", "value"))  # Matriz
def matriz(n, u):
    cm, met = metricas_umbral(n, u)  # Métricas con el umbral
    texto = html.P(f"Anticipa {miles(cm[1][1])} de {miles(cm[1].sum())} brotes (recall {pct(met['recall'])}) · precisión {pct(met['precisión'])} · "
                   f"F1 {numero(met['F1'])} · MCC {numero(met['MCC'])} · {numero(met['alertas por semana'], 0)} alertas por semana", className="nota")  # Lectura operativa
    return fg.fig_matriz(cm.tolist(), u), texto  # Figura y texto


@callback(Output("mod-curvas-fig", "figure"), Input("mod-curvas", "value"))  # Curvas
def curvas_fig(nombres):
    return fg.fig_curvas(curvas([n for n in TODOS if n in nombres]), MOD["y_te"].mean())  # Figura


# ---------------------------------------------------------------------------
def diagnostico():  # Calibración, residuos, curva de aprendizaje y bloques espaciales
    bl = MOD["bloques"]  # Validación espacial
    return html.Div([
        html.Div([control("Modelo", desplegable("mod-diag", [{"label": n, "value": n} for n in BASES], BASES[0]))], style={"maxWidth": "360px", "marginBottom": "14px"}),
        html.Div(id="mod-diag-contenido"),  # Contenido que depende del modelo
        html.Div([
            tarjeta(html.H3("Curva de aprendizaje (regresión logística)"),
                    dcc.RadioItems(id="mod-curva-met", options=["AUC-ROC", "AUC-PR"], value="AUC-ROC", className="opciones", inline=True), grafico("mod-curva-fig"),
                    lectura("La curva es plana: con el 10 % de los municipios el modelo ya alcanza casi el mismo desempeño de validación. La muestra "
                            "es suficiente y no hay sobreajuste; el límite es de sesgo (un modelo lineal no captura la no linealidad del clima ni la "
                            "dinámica de vecindad).")),
            tarjeta(html.H3("Generalización a zonas nuevas (bloques de departamentos, 50 km de separación)"),
                    tabla_df(bl.drop(columns=["departamentos"]), dec=3),
                    tabla(["Esquema", "Regresión logística", "SVM lineal"], [
                        ["1. Futuro, mismas zonas (prueba)", numero(MOD["ic"]["Regresión logística"].loc["AUC-ROC", "valor"]), numero(MOD["ic"]["SVM lineal"].loc["AUC-ROC", "valor"])],
                        ["2. Zonas nuevas, mismo periodo", numero(bl["LR zonas nuevas"].mean()), numero(bl["SVM zonas nuevas"].mean())],
                        ["3. Zonas nuevas y futuro", numero(bl["LR zonas nuevas + futuro"].mean()), numero(bl["SVM zonas nuevas + futuro"].mean())]], numericas=(1, 2)),
                    lectura("Los modelos generalizan a departamentos que no vieron, incluso quitando del entrenamiento a los municipios a menos de 50 km: "
                            "las predictoras son relativas a cada municipio (casos frente a su canal endémico, anomalías) y no identifican lugares. "
                            "El esquema 3 es el más exigente y el más variable entre bloques.")),
        ], className="rejilla cols-2"),
    ])


@callback(Output("mod-diag-contenido", "children"), Input("mod-diag", "value"))  # Diagnóstico del modelo elegido
def diag(n):
    dg = MOD["diagnosticos"][n]  # Diagnósticos
    mo = dg["moran"]  # Moran de residuos
    return html.Div([
        html.Div([
            tarjeta(html.H3("Calibración"), grafico(figura=fg.fig_calibracion(dg["calibracion"])),
                    lectura("En la validación de 2022 el error de calibración es pequeño; en la prueba el modelo subestima las probabilidades porque "
                            "la prevalencia (≈ 30 %) triplica la del entrenamiento (prior shift). El ordenamiento (AUC) se mantiene; la escala se "
                            "corrige recalibrando la intersección o el umbral con datos recientes.")),
            tarjeta(html.H3("Sobreajuste: mismas métricas en los tres conjuntos"), tabla_df(dg["brecha"], dec=3),
                    lectura("La brecha de AUC-ROC entre entrenamiento y prueba es de unas dos centésimas. Las diferencias de AUC-PR, precisión y "
                            "Brier se deben sobre todo al cambio de prevalencia, no al sobreajuste.")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Autocorrelación temporal de los residuos (ACF media por municipio)"), grafico(figura=fg.fig_acf_residuos(dg["acf"])),
                    lectura(f"Los residuos tienen autocorrelación en los primeros rezagos (rezago 1 = {numero(dg['acf'][0], 2)}): el modelo se equivoca en "
                            "rachas, porque los episodios de brote duran varias semanas. Hay información temporal sin capturar.")),
            tarjeta(html.H3(f"Residuo medio por municipio (I de Moran = {numero(mo['I'])}, p = {numero(mo['p'])})"), grafico(figura=fg.fig_residuo_mapa(M, dg["residuo_mun"])),
                    lectura("Los residuos tienen autocorrelación espacial significativa: zonas enteras con brotes no anticipados (rojo). Es la "
                            "dinámica de vecindad que mostraron Knox y Moran en el EDA: motiva variables de rezago espacial en los siguientes modelos.")),
        ], className="rejilla cols-2"),
    ])


@callback(Output("mod-curva-fig", "figure"), Input("mod-curva-met", "value"))  # Curva de aprendizaje
def curva(met):
    return fg.fig_curva_aprendizaje(MOD["curva"], met)  # Figura


# ---------------------------------------------------------------------------
def interpretacion():  # Coeficientes
    coef = MOD["coef_lr"]  # Coeficientes con IC
    top = coef.reindex(coef["coef"].abs().sort_values(ascending=False).index).head(8)  # Más grandes
    a, c = MOD["platt"]  # Escalado de Platt
    return html.Div([
        html.Div([
            tarjeta(html.H3("Regresión logística: coeficientes estandarizados con IC 95 %"), grafico(figura=fg.fig_coeficientes(coef)),
                    lectura("Cada barra es el cambio en las log-odds de brote al subir una desviación estándar la variable (manteniendo las demás). "
                            "Los intervalos se obtienen reajustando el Pipeline en 200 remuestreos de municipios.")),
            tarjeta(html.H3("SVM lineal: pesos del hiperplano"), grafico(figura=fg.fig_pesos_svm(MOD["coef_svm"])),
                    lectura(f"Los pesos de la SVM se parecen a los coeficientes logísticos (correlación ≈ 0,87; la proporción reciente de semanas en "
                            f"brote es la variable más pesada en ambos), aunque la SVM reparte el efecto de la temperatura entre las ventanas de 8 y "
                            f"12 semanas con signos opuestos: un síntoma de la colinealidad entre medias móviles, no un efecto real. Su escala no "
                            f"es comparable con la logística porque la pérdida bisagra no se mide en log-odds. Su margen se convierte en probabilidad con el escalado de Platt ajustado en 2022: "
                            f"p = σ({numero(a, 2)} · margen {'+' if c >= 0 else '−'} {numero(abs(c), 2)}).")),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Las variables con más peso (regresión logística)"),
                tabla(["Variable", "Coeficiente", "IC 95 %", "Odds ratio por 1 DE"],
                      [[fg.etiqueta(r["variable"]), numero(r["coef"]), f"[{numero(r['IC 2,5 %'])}; {numero(r['IC 97,5 %'])}]", numero(np.exp(r["coef"]), 2)] for _, r in top.iterrows()],
                      numericas=(1, 3)),
                lectura("La historia de casos domina: la proporción reciente de semanas en brote, los casos de las últimas semanas y los casos frente "
                        "al umbral objetivo elevan el riesgo. Las medias largas (8 y 12 semanas) tienen signo negativo condicional a las cortas: lo que "
                        "predice un brote es la aceleración reciente, no un nivel alto sostenido (un municipio que ya venía en epidemia tiende a salir). "
                        "El ONI actual multiplica las odds por ≈ 1,26 por desviación estándar. Los efectos de la temperatura no son distinguibles de 0 "
                        "(el 64 % está imputada) y la lluvia tiene signos mezclados entre ventanas: relaciones no lineales que un modelo lineal no captura.")),
    ])


# ---------------------------------------------------------------------------
def simulador():  # Simulador del modelo logístico
    filas = MOD["simulador"]["filas"]  # Semana de máximo de casos de cada municipio en la prueba
    opciones = [{"label": f"{r['municipio']} ({r['departamento']}) · semana del {r['week_start'].date()}", "value": i} for i, r in filas.iterrows()]  # Opciones
    probs = np.array([simular(f) for _, f in filas.iterrows()])  # Probabilidad observada de cada escenario
    cali = int(np.argmin(np.abs(probs - MOD["simulador"]["umbral"])))  # Por defecto, el municipio más cercano al umbral (donde el clima puede cambiar la alerta)
    return html.Div([
        html.Div([
            tarjeta(html.H3("Escenario"),
                    control("Municipio y semana (la de más casos en 2023–2024)", desplegable("sim-fila", opciones, cali)),
                    control("Cambio de temperatura (°C)", dcc.Slider(id="sim-temp", min=-3, max=3, step=0.5, value=0, marks={v: str(v) for v in range(-3, 4)})),
                    control("Lluvia (× la observada)", dcc.Slider(id="sim-lluvia", min=0, max=2, step=0.1, value=1, marks={0: "0", 0.5: "0,5", 1: "1", 1.5: "1,5", 2: "2"})),
                    control("ONI (anomalía de El Niño)", dcc.Slider(id="sim-oni", min=-2, max=2.5, step=0.01, value=0, marks={-2: "−2", -1: "−1", 0: "0", 1: "1", 2: "2"})),
                    html.Div(id="sim-info", className="nota", style={"marginTop": "12px"})),
            tarjeta(html.H3("Probabilidad de brote en 4 semanas (regresión logística)"), grafico("sim-fig"), html.Div(id="sim-txt"),
                    lectura("El simulador cambia el clima de la semana elegida y recalcula la probabilidad con el modelo entrenado (estandarización y "
                            "coeficientes del Pipeline). La marca ámbar es el umbral de alerta. Como el modelo es lineal y la historia de casos domina, "
                            "el clima mueve la probabilidad poco: es una limitación del modelo base, no una conclusión causal.")),
        ], className="rejilla cols-control"),
    ])


@callback(Output("sim-oni", "value"), Input("sim-fila", "value"))  # Al elegir la semana, el ONI parte del valor observado
def oni_inicial(i):
    return round(float(MOD["simulador"]["filas"].loc[i, "oni_anom"]), 2)  # ONI observado (exacto: el escenario parte del clima real)


@callback(Output("sim-fig", "figure"), Output("sim-txt", "children"), Output("sim-info", "children"),
          Input("sim-fila", "value"), Input("sim-temp", "value"), Input("sim-lluvia", "value"), Input("sim-oni", "value"))  # Recalcula
def simular_cb(i, dt, fl, oni):
    fila = MOD["simulador"]["filas"].loc[i]  # Semana elegida
    u = MOD["simulador"]["umbral"]  # Umbral de alerta
    p0 = simular(fila)  # Probabilidad observada
    p = simular(fila, dt, fl, oni)  # Probabilidad del escenario
    txt = html.P([html.Span(pct(p), className=f"cifra-grande {'rojo' if p >= u else 'verde'}"),
                  f"  frente a {pct(p0)} con el clima observado · {'ALERTA' if p >= u else 'sin alerta'} (umbral {pct(u)}) · "
                  f"brote real en 4 semanas: {'sí' if fila['y'] == 1 else 'no'}"])  # Resultado
    info = f"Semana con {miles(fila['casos_totales'])} casos · temperatura media 4 sem. {numero(fila['temp_mean_m4'], 1)} °C · lluvia 4 sem. {numero(fila['prec_total_m4'], 1)} mm · ONI {numero(fila['oni_anom'], 2)}"  # Contexto
    return fg.fig_probabilidad(p, u), txt, info  # Salidas


# ---------------------------------------------------------------------------
def critica():  # Nota crítica
    ic = MOD["ic"]  # Métricas
    return html.Div([
        html.Div([
            tarjeta(html.H3("¿Un desempeño demasiado alto?"),
                    html.P(f"La exactitud en la prueba es {pct(ic['Regresión logística'].loc['Accuracy', 'valor'])}, dentro del rango 80–90 % que pide revisar. La revisión:"),
                    tabla(["Revisión", "Resultado"], [
                        ["1. Fuga de datos", "Ninguna predictora supera AUC univariado 0,9; la columna del futuro se descartó; partición temporal con purga"],
                        ["2. Línea base trivial", f"Dummy {pct(ic['Dummy (prior)'].loc['Accuracy', 'valor'])} y persistencia {pct(ic['Persistencia'].loc['Accuracy', 'valor'])} de exactitud: el 83 % no es mérito del modelo"],
                        ["3. Estructura temporal y espacial", "Validación por años con purga y por bloques de departamentos con 50 km de separación"],
                        ["4. Complejidad", f"Mejor regla simple AUC-PR {numero(ic['Persistencia (8 sem.)'].loc['AUC-PR', 'valor'], 2)} frente a {numero(ic['Regresión logística'].loc['AUC-PR', 'valor'], 2)}; residuos con estructura sin explicar"],
                    ]),
                    lectura("El desempeño alto en exactitud es consecuencia del desbalance y de la persistencia natural de las epidemias, no de una "
                            "fuga. Por eso los modelos se juzgan con AUC-ROC, AUC-PR, F1 y MCC, donde sí superan a las líneas base. El problema no es "
                            "trivial y el dataset es adecuado para el curso.")),
            tarjeta(html.H3("Conclusiones y siguientes pasos"),
                    html.Ul([html.Li("Los dos modelos base lineales son equivalentes (AUC-ROC ≈ 0,886) y superan significativamente a la persistencia."),
                             html.Li("La historia de casos domina; El Niño aporta; la temperatura local casi no aporta en un modelo lineal."),
                             html.Li("No hay sobreajuste: el límite es de sesgo (no linealidad del clima y dinámica de vecindad sin capturar)."),
                             html.Li("La probabilidad queda subestimada en 2023–2024 por el cambio de prevalencia: hay que recalibrar."),
                             html.Li("Siguientes entregables: variables de rezago espacial, modelos no lineales y recalibración periódica.")]),
                    html.Div(className="banner banner-mapa", style={"minHeight": "110px", "marginTop": "12px"})),
        ], className="rejilla cols-2"),
    ])
