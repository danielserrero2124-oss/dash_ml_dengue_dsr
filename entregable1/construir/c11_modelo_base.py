"""Cuaderno 11: modelo base (sección 3 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Modelo base: regresión logística\n\n"
           "* **Objetivo:** $y_{m,t} = 1$ si el municipio $m$ está en brote en la semana $t+4$ (canal endémico).\n"
           "* **Modelo base:** regresión logística con penalización L2, dentro de un `Pipeline` (imputación → estandarización → "
           "modelo), ajustada con `GridSearchCV` y **validación cruzada hacia adelante por años con purga** (el mismo tipo de "
           "partición que entrenamiento/prueba).\n"
           "* **Líneas base triviales:** `DummyClassifier` (proporción de entrenamiento), **persistencia** (¿hay brote ahora?), "
           "**persistencia de 8 semanas** (proporción reciente de semanas en brote) e **ingenuo estacional** (¿hubo brote en "
           "la misma semana hace un año?).\n"
           "* **Selección:** AUC-PR en la validación cruzada (clase minoritaria); umbral de decisión que maximiza F1 en la "
           "validación de 2022. La prueba 2023–2024 se usa **una sola vez**, en este capítulo."),
        code(PREAMBULO + '''
from sklearn.metrics import average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve  # Métricas y curvas
df = d.panel()  # Panel completo
mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Particiones purgadas
print(f"Entrenamiento {len(tr):,} · validación {len(va):,} · prueba {len(te):,} filas")  # Tamaños'''),

        md("## Entrenamiento: Pipeline + GridSearchCV con validación purgada"),
        code('''grid = mb.ajustar_base(tr)  # Prueba 10 combinaciones × 4 pliegues
res = pd.DataFrame(grid.cv_results_)  # Resultados de la validación cruzada
tabla = pd.DataFrame({"C": res["param_clf__C"], "class_weight": res["param_clf__class_weight"].astype(str),  # Hiperparámetros
                      "AUC-PR (media)": res["mean_test_auc_pr"], "AUC-PR (desv.)": res["std_test_auc_pr"],  # Criterio
                      "AUC-ROC (media)": res["mean_test_auc_roc"], "AUC-ROC entrenamiento": res["mean_train_auc_roc"],  # Diagnóstico de sobreajuste
                      "Brier (media)": -res["mean_test_neg_brier"]}).sort_values("AUC-PR (media)", ascending=False)  # Calibración
print("Mejores hiperparámetros:", grid.best_params_)  # Elección
tabla.round(4)'''),
        code('''mejor = grid.best_index_  # Fila de la mejor combinación
pliegues = pd.DataFrame({"año validado": [2018, 2019, 2020, 2021],  # Desempeño por pliegue
                         "AUC-PR": [res.loc[mejor, f"split{i}_test_auc_pr"] for i in range(4)],
                         "AUC-ROC": [res.loc[mejor, f"split{i}_test_auc_roc"] for i in range(4)]}).set_index("año validado")
pliegues.round(4)'''),
        md("**Interpretación.** El AUC-PR cambia muy poco entre valores de $C$: con 130 000 filas la regularización importa "
           "poco para el desempeño, pero la más fuerte razonable ($C = 0{,}01$) estabiliza los coeficientes ante la "
           "multicolinealidad. `class_weight=\"balanced\"` sube levemente el AUC-ROC pero baja el AUC-PR y empeora la "
           "calibración (infla las probabilidades), así que no se pondera. El desempeño por pliegue varía con el año "
           "(2019, año epidémico, es el más fácil en AUC-PR), lo que confirma que una sola partición no basta para estimar la "
           "incertidumbre."),

        md("## Umbral de decisión (validación 2022) y evaluación en la prueba 2023–2024"),
        code('''modelo = grid.best_estimator_  # Pipeline reentrenado con todo el entrenamiento
p_va = modelo.predict_proba(va[cf.FEATURES])[:, 1]  # Probabilidades en validación
p_te = modelo.predict_proba(te[cf.FEATURES])[:, 1]  # Probabilidades en prueba (única vez)
umbral = mb.umbral_f1(va[cf.TARGET], p_va)  # Umbral que maximiza F1 en 2022
lb_va, lb_te = mb.lineas_base(tr, va), mb.lineas_base(tr, te)  # Líneas base
umbrales = {"Regresión logística": umbral, "Dummy (prior)": 0.5, "Persistencia": 1.0,  # Reglas binarias: alerta si el indicador es 1
            "Persistencia (8 sem.)": mb.umbral_f1(va[cf.TARGET], lb_va["Persistencia (8 sem.)"]), "Ingenuo estacional": 1.0}  # Persistencia continua: F1 en 2022
puntajes = {"Regresión logística": p_te, **lb_te}  # Puntajes en prueba
print(f"Umbral de la regresión logística (máximo F1 en 2022): {umbral:.3f}")  # Umbral
print(f"Prevalencia en la prueba: {te[cf.TARGET].mean():.1%} (entrenamiento {tr[cf.TARGET].mean():.1%})")  # Cambio de prevalencia'''),
        code('''resultados = {}  # Métricas con IC 95 % (bootstrap de 1 000 réplicas por municipio)
for nombre, p in puntajes.items():  # Cada modelo
    resultados[nombre] = mb.bootstrap_metricas(te[cf.TARGET], p, umbrales[nombre], te["COD_MUN_N"], B=1000)  # Métricas e IC
tab = pd.concat({n: r.apply(lambda f: f"{f['valor']:.3f} [{f['IC 2,5 %']:.3f}; {f['IC 97,5 %']:.3f}]", axis=1) for n, r in resultados.items()}, axis=1).T  # Formato valor [IC]
tab'''),
        code('''comparaciones = []  # Diferencias pareadas con la mejor línea base
for base in ["Persistencia (8 sem.)", "Persistencia"]:  # Líneas base más fuertes
    for nombre_m, met in [("AUC-ROC", roc_auc_score), ("AUC-PR", average_precision_score)]:  # Métricas
        dif, lo, hi, p = mb.bootstrap_diferencia(te[cf.TARGET], p_te, puntajes[base], te["COD_MUN_N"], metrica=met, B=1000)  # Bootstrap pareado
        comparaciones.append({"comparación": f"Regresión logística − {base}", "métrica": nombre_m, "diferencia": dif,  # Resultado
                              "IC 2,5 %": lo, "IC 97,5 %": hi, "p (bootstrap)": p})
pd.DataFrame(comparaciones).round(4)'''),
        md("**Interpretación.** La regresión logística supera con claridad a todas las líneas base en las métricas que "
           "importan: AUC-ROC ≈ 0,885 y AUC-PR ≈ 0,81, frente a 0,5 / 0,30 del Dummy y 0,86 / 0,73 de la mejor persistencia; "
           "las diferencias pareadas tienen intervalos que no contienen el 0. Su F1 y su MCC también son los más altos. Los "
           "intervalos (por municipio) son estrechos, de ± 0,01 en AUC-ROC, porque hay 376 municipios y dos años de prueba.\n\n"
           "Ojo con la **exactitud**: la persistencia simple acierta casi lo mismo que el modelo (≈ 0,82) y el Dummy, que nunca "
           "alerta, ya llega a 0,70. La exactitud no distingue un buen sistema de alerta de uno trivial; por eso no se usa "
           "para decidir. El ingenuo estacional es casi inútil (AUC ≈ 0,54): los brotes no se repiten en la misma semana del "
           "año, lo que confirma la estacionalidad débil encontrada en el EDA."),

        md("## Matriz de confusión, curvas ROC y precisión-recall"),
        code('''pred = (p_te >= umbral).astype(int)  # Alertas
cm = confusion_matrix(te[cf.TARGET], pred)  # [[VN, FP], [FN, VP]]
etiquetas = [[f"VN<br>{cm[0, 0]:,}", f"FP<br>{cm[0, 1]:,}"], [f"FN<br>{cm[1, 0]:,}", f"VP<br>{cm[1, 1]:,}"]]  # Textos
fig = go.Figure(go.Heatmap(z=cm, x=["Predicho: sin brote", "Predicho: brote"], y=["Real: sin brote", "Real: brote"], text=etiquetas,  # Mapa de calor
                           texttemplate="%{text}", colorscale="Blues", showscale=False, textfont=dict(size=16)))
fig.update_layout(title=f"Matriz de confusión en la prueba (umbral {umbral:.3f})", yaxis_autorange="reversed")  # Formato
mostrar(fig, 380)  # Muestra
print(f"Anticipa {cm[1, 1]:,} de {cm[1].sum():,} brotes (recall {cm[1, 1] / cm[1].sum():.1%}); {cm[0, 1]:,} falsas alarmas; "
      f"{cm[:, 1].sum() / te['time_idx'].nunique():.0f} alertas por semana en promedio")  # Lectura operativa'''),
        code('''fig = make_subplots(1, 2, subplot_titles=("Curva ROC", "Curva precisión-recall"))  # Dos paneles
colores = {"Regresión logística": AZUL, "Persistencia (8 sem.)": VERDE, "Persistencia": AMBAR, "Ingenuo estacional": MORADO, "Dummy (prior)": GRIS}  # Colores
for nombre, col in colores.items():  # Cada modelo
    p = puntajes[nombre]  # Puntajes
    fpr, tpr, _ = roc_curve(te[cf.TARGET], p)  # ROC
    pr, rc, _ = precision_recall_curve(te[cf.TARGET], p)  # PR
    fig.add_scatter(x=fpr, y=tpr, name=f"{nombre} ({roc_auc_score(te[cf.TARGET], p):.3f})", line=dict(color=col), row=1, col=1)  # ROC
    fig.add_scatter(x=rc, y=pr, showlegend=False, line=dict(color=col), row=1, col=2)  # PR
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS, dash="dot"), showlegend=False, row=1, col=1)  # Azar ROC
fig.add_hline(y=te[cf.TARGET].mean(), line_dash="dot", line_color=GRIS, row=1, col=2)  # Azar PR = prevalencia
fig.update_xaxes(title_text="tasa de falsos positivos", row=1, col=1)  # Ejes
fig.update_yaxes(title_text="recall", row=1, col=1)  # Ejes
fig.update_xaxes(title_text="recall", row=1, col=2)  # Ejes
fig.update_yaxes(title_text="precisión", row=1, col=2)  # Ejes
fig.update_layout(title="Curvas en la prueba 2023–2024 (leyenda: AUC-ROC)")  # Título
mostrar(fig, 440)  # Muestra'''),

        md("## Calibración"),
        code('''frac_te, media_te, ece_te = mb.calibracion(te[cf.TARGET].to_numpy(), p_te)  # Calibración en prueba
frac_va, media_va, ece_va = mb.calibracion(va[cf.TARGET].to_numpy(), p_va)  # Calibración en validación
fig = go.Figure()  # Curva de confiabilidad
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=GRIS, dash="dot"), name="Calibración perfecta")  # Diagonal
fig.add_scatter(x=media_va, y=frac_va, mode="lines+markers", line=dict(color=VERDE), name=f"Validación 2022 (ECE {ece_va:.3f})")  # Validación
fig.add_scatter(x=media_te, y=frac_te, mode="lines+markers", line=dict(color=AZUL), name=f"Prueba 2023–24 (ECE {ece_te:.3f})")  # Prueba
fig.update_layout(title="Curva de calibración por deciles de probabilidad", xaxis_title="probabilidad predicha", yaxis_title="proporción observada de brotes")  # Formato
mostrar(fig, 420)  # Muestra
print(f"Brier: validación {np.mean((p_va - va[cf.TARGET]) ** 2):.4f} · prueba {np.mean((p_te - te[cf.TARGET]) ** 2):.4f} · "
      f"probabilidad media en prueba {p_te.mean():.3f} frente a prevalencia {te[cf.TARGET].mean():.3f}")  # Resumen'''),
        md("**Interpretación.** En la validación de 2022 (prevalencia parecida a la del entrenamiento) el error de calibración "
           "es pequeño (ECE ≈ 0,03), aunque el modelo ya subestima un poco (probabilidad media 9,7 % frente a 12,4 % "
           "observado). En la prueba la **subestimación** crece (ECE ≈ 0,05; 25,7 % frente a 30,3 %), sobre todo en las "
           "probabilidades intermedias: la prevalencia de 2023–2024 (≈ 30 %) triplica la del entrenamiento (≈ 10 %), y la regresión logística aprendió una intersección acorde con la prevalencia "
           "antigua. Es la consecuencia directa del *prior shift* documentado en el EDA. El ordenamiento (AUC) se mantiene; lo "
           "que se desajusta es la escala, y se corrige recalibrando la intersección o el umbral con datos recientes."),

        md("## Diagnóstico de residuos: autocorrelación temporal y espacial\n\n"
           "Residuo de respuesta: $r = y - \\hat p$. Si el modelo capturara toda la estructura, los residuos no tendrían "
           "dependencia temporal (ACF ≈ 0) ni espacial (I de Moran ≈ 0)."),
        code('''resid = (te[cf.TARGET] - p_te).to_numpy()  # Residuos en la prueba
acf_r, n_mun = mb.acf_residuos(te, resid, rezagos=12)  # ACF media dentro de cada municipio
banda = 1.96 / np.sqrt(te.groupby("COD_MUN_N").size().mean())  # Banda aproximada por serie
fig = go.Figure(go.Bar(x=np.arange(1, 13), y=acf_r, marker_color=AZUL))  # ACF
fig.add_hline(y=banda, line_dash="dot", line_color=GRIS)  # Banda
fig.add_hline(y=-banda, line_dash="dot", line_color=GRIS)  # Banda
fig.update_layout(title=f"ACF media de los residuos dentro de los municipios ({n_mun} series)", xaxis_title="rezago (semanas)", yaxis_title="autocorrelación")  # Formato
mostrar(fig, 340)  # Muestra
m = d.municipios()  # Centroides
D = sp.haversine(m["latitud"].to_numpy(), m["longitud"].to_numpy())  # Distancias
r_mun = te.assign(r=resid).groupby("COD_MUN_N")["r"].mean().loc[m["COD_MUN_N"]]  # Residuo medio por municipio
mor = sp.moran(r_mun.to_numpy(), sp.pesos_knn(D, 6), permutaciones=999)  # Moran de los residuos
print(f"ACF de los residuos: rezago 1 = {acf_r[0]:.3f}, rezago 4 = {acf_r[3]:.3f} · I de Moran del residuo medio por municipio: {mor['I']:.3f} (p = {mor['p']:.3f})")  # Resumen
fig = px.scatter_geo(m.assign(residuo=r_mun.to_numpy()), lat="latitud", lon="longitud", color="residuo", hover_name="municipio",  # Mapa de residuos
                     color_continuous_scale="RdBu_r", color_continuous_midpoint=0)
fig.update_traces(marker=dict(size=7, line=dict(width=0.3)))  # Tamaño
fig.update_layout(title="Residuo medio por municipio en la prueba (rojo: brotes no anticipados)")  # Título
mostrar(mapa_base(fig), 540)  # Muestra'''),
        md("**Interpretación.** Los residuos **sí tienen dependencia**: autocorrelación temporal clara en los primeros 3 rezagos "
           "(los episodios de brote duran varias semanas y el modelo se equivoca en rachas) y autocorrelación espacial "
           "significativa (municipios vecinos comparten el signo del error, con zonas enteras subestimadas en 2023–2024). "
           "Según la guía, esto indica **información estructural sin capturar**: la dinámica de vecindad (Knox y Moran del "
           "EDA) y la no linealidad del clima. Es la motivación para los modelos de los siguientes entregables: variables "
           "de rezago espacial y modelos no lineales."),

        md("## Curva de aprendizaje y sobreajuste"),
        code('''curva = mb.curva_aprendizaje(tr, va, C=grid.best_params_["clf__C"], cw=grid.best_params_["clf__class_weight"])  # Fracciones de municipios
fig = make_subplots(1, 2, subplot_titles=("AUC-ROC", "AUC-PR"))  # Dos paneles
for i, met in enumerate(["AUC-ROC", "AUC-PR"]):  # Métricas
    fig.add_scatter(x=curva["filas"], y=curva[f"{met} entrenamiento"], name="Entrenamiento", line=dict(color=AZUL), showlegend=i == 0, row=1, col=i + 1)  # Entrenamiento
    fig.add_scatter(x=curva["filas"], y=curva[f"{met} validación"], name="Validación 2022", line=dict(color=VERDE), showlegend=i == 0, row=1, col=i + 1)  # Validación
fig.update_xaxes(title_text="filas de entrenamiento")  # Eje
fig.update_layout(title="Curva de aprendizaje (fracciones crecientes de municipios)")  # Título
mostrar(fig, 360)  # Muestra
curva.round(4)'''),
        code('''p_tr = modelo.predict_proba(tr[cf.FEATURES])[:, 1]  # Probabilidades en entrenamiento
brecha = pd.DataFrame({conj: mb.metricas(yy, pp, umbral) for conj, yy, pp in [("Entrenamiento 2015–21", tr[cf.TARGET], p_tr),  # Mismo umbral
                                                                            ("Validación 2022", va[cf.TARGET], p_va), ("Prueba 2023–24", te[cf.TARGET], p_te)]}).T
brecha.round(4)'''),
        md("**Interpretación.** La curva de aprendizaje es plana: con el 10 % de los municipios el modelo ya alcanza casi el "
           "mismo AUC de validación que con todos, y las curvas de entrenamiento y validación están cerca. El tamaño de "
           "muestra es más que suficiente para un modelo lineal y **no hay sobreajuste** (la brecha de AUC-ROC entre "
           "entrenamiento y prueba es de unas dos centésimas). El límite es de **sesgo**: el modelo es demasiado simple para "
           "la estructura no lineal y espacial; más datos no lo mejorarán, pero un modelo más flexible sí podría. Las "
           "diferencias de AUC-PR, precisión y Brier entre conjuntos se deben sobre todo al cambio de prevalencia, no al "
           "sobreajuste."),

        md("## Interpretación de los coeficientes\n\n"
           "Los coeficientes están en unidades de desviación estándar (las variables se estandarizan): $e^{\\beta}$ es el "
           "cambio en las *odds* de brote cuando la variable sube una desviación estándar, manteniendo las demás fijas. Los "
           "intervalos se obtienen reajustando el Pipeline en 200 remuestreos de municipios."),
        code('''coef = mb.coeficientes_bootstrap(tr, grid.best_params_["clf__C"], grid.best_params_["clf__class_weight"], B=200)  # Coeficientes e IC
coef["odds ratio por 1 DE"] = np.exp(coef["coef"])  # Razón de odds
coef["IC excluye 0"] = (coef["IC 2,5 %"] > 0) | (coef["IC 97,5 %"] < 0)  # Significativo
orden = coef.sort_values("coef")  # Orden
fig = go.Figure(go.Bar(x=orden["coef"], y=orden["variable"], orientation="h", marker_color=np.where(orden["coef"] > 0, ROJO, AZUL),  # Coeficientes
                       error_x=dict(type="data", symmetric=False, array=orden["IC 97,5 %"] - orden["coef"], arrayminus=orden["coef"] - orden["IC 2,5 %"])))
fig.update_layout(title="Coeficientes estandarizados con IC 95 % (bootstrap de municipios)", xaxis_title="coeficiente (log-odds por 1 DE)")  # Formato
mostrar(fig, 760)  # Muestra
coef.set_index("variable").sort_values("coef", key=np.abs, ascending=False).round(3)'''),
        md("**Interpretación.**\n\n"
           "* **Historia de casos.** Los coeficientes más grandes y estables: la proporción reciente de semanas en brote "
           "(`brote_m8`), los casos de las últimas semanas (`lc_l0`, `lc_l1`, `lc_m4`) y los casos frente al umbral objetivo "
           "(`ratio_umbral`) aumentan el riesgo. Las medias largas (`lc_m8`, `lc_m12`) tienen signo **negativo** condicional a "
           "las cortas: con los casos recientes fijos, una historia larga alta significa que el municipio ya venía en epidemia "
           "(y el brote tiende a agotarse); lo que predice un brote es la **aceleración** reciente. Es un efecto de "
           "colinealidad que debe leerse en conjunto, no variable por variable.\n"
           "* **El Niño.** El ONI actual aumenta el riesgo (una desviación estándar multiplica las *odds* por ≈ 1,26).\n"
           "* **Clima local.** Efectos pequeños, de signos mezclados entre ventanas (lluvia de 4–8 semanas positiva, de 12 "
           "negativa), coherentes con las relaciones no lineales vistas en el EDA. Los de temperatura no son distinguibles de 0: con el "
           "64 % de la temperatura imputada, su señal está muy atenuada.\n"
           "* **Contexto.** Más flujo de personas, más riesgo. La estacionalidad y la bandera de temperatura imputada no son "
           "distinguibles de 0."),

        md("## Generalización a zonas nuevas: esquemas espaciales y espacio-temporales\n\n"
           "Bloques por departamento (`GroupKFold`, 5 bloques) con 50 km de separación: se quitan del entrenamiento los "
           "municipios a menos de 50 km de cualquier municipio del bloque evaluado."),
        code('''from sklearn.model_selection import GroupKFold  # Bloques espaciales
mun_xy = m.set_index("COD_MUN_N")[["latitud", "longitud"]]  # Coordenadas por código
filas = []  # Resultados
for k, (i_tr, i_ev) in enumerate(GroupKFold(5).split(tr, groups=tr["depto"])):  # Bloques de departamentos
    deptos = set(tr.iloc[i_ev]["depto"])  # Departamentos evaluados
    m_ev = tr.iloc[i_ev]["COD_MUN_N"].unique()  # Municipios evaluados
    m_tr = tr.iloc[i_tr]["COD_MUN_N"].unique()  # Municipios candidatos al entrenamiento
    Dx = sp.haversine(np.r_[mun_xy.loc[m_tr, "latitud"], mun_xy.loc[m_ev, "latitud"]], np.r_[mun_xy.loc[m_tr, "longitud"], mun_xy.loc[m_ev, "longitud"]])[: len(m_tr), len(m_tr):]  # Distancias cruzadas
    lejos = m_tr[Dx.min(axis=1) > 50]  # Municipios a más de 50 km del bloque evaluado
    ent = tr[tr["COD_MUN_N"].isin(lejos)]  # Entrenamiento con zona de separación
    mdl = mb.pipeline_base().set_params(clf__C=grid.best_params_["clf__C"], clf__class_weight=grid.best_params_["clf__class_weight"]).fit(ent[cf.FEATURES], ent[cf.TARGET])  # Ajuste
    ev_mismo = tr.iloc[i_ev]  # Zonas nuevas, mismo periodo
    ev_futuro = te[te["depto"].isin(deptos)]  # Zonas nuevas y futuro
    filas.append({"bloque": k + 1, "departamentos": len(deptos), "municipios evaluados": len(m_ev), "municipios excluidos por la zona de 50 km": len(m_tr) - len(lejos),
                  "AUC-ROC zonas nuevas (2015–21)": roc_auc_score(ev_mismo[cf.TARGET], mdl.predict_proba(ev_mismo[cf.FEATURES])[:, 1]),
                  "AUC-ROC zonas nuevas + futuro (2023–24)": roc_auc_score(ev_futuro[cf.TARGET], mdl.predict_proba(ev_futuro[cf.FEATURES])[:, 1]),
                  "AUC-PR zonas nuevas + futuro (2023–24)": average_precision_score(ev_futuro[cf.TARGET], mdl.predict_proba(ev_futuro[cf.FEATURES])[:, 1])})
bloques = pd.DataFrame(filas).set_index("bloque")  # Tabla
bloques.round(4)'''),
        code('''esquemas = pd.DataFrame({  # Resumen de los tres esquemas
    "AUC-ROC": [roc_auc_score(te[cf.TARGET], p_te), bloques["AUC-ROC zonas nuevas (2015–21)"].mean(), bloques["AUC-ROC zonas nuevas + futuro (2023–24)"].mean()],
    "dispersión entre bloques": [np.nan, bloques["AUC-ROC zonas nuevas (2015–21)"].std(), bloques["AUC-ROC zonas nuevas + futuro (2023–24)"].std()],
}, index=["1. Futuro, mismas zonas (prueba 2023–24)", "2. Zonas nuevas, mismo periodo", "3. Zonas nuevas y futuro"])
esquemas.round(4)'''),
        md("**Interpretación.** El modelo **generaliza a departamentos que no vio**, incluso con la zona de separación de 50 km: "
           "el AUC por bloques es comparable al del esquema principal, con variación entre bloques (algunas regiones son más "
           "difíciles). Esto es esperable: las predictoras son relativas a cada municipio (casos frente a su canal endémico, "
           "anomalías climáticas) y no identifican lugares. El esquema 3 (zonas nuevas y futuro) es el más exigente y el más "
           "variable."),

        md("## Nota crítica: ¿un desempeño demasiado alto?\n\n"
           "La exactitud en la prueba es ≈ 0,83, dentro del rango 80–90 % que la guía pide revisar. La revisión:\n\n"
           "1. **Fuga de datos.** Auditada en el capítulo 6: ninguna predictora supera un AUC univariado de 0,9, la columna con "
           "información del futuro se descartó, la partición es temporal con purga y las particiones no comparten filas.\n"
           "2. **Comparación con la línea base trivial.** El 83 % no es mérito del modelo: el Dummy que nunca alerta acierta el "
           "70 % y la persistencia simple, el 82 %. Por eso el modelo se juzga por AUC-ROC, AUC-PR, F1 y MCC, donde sí "
           "supera a las líneas base con intervalos que excluyen el 0.\n"
           "3. **Estructura temporal y espacial.** La validación respeta el tiempo (años hacia adelante con purga) y el espacio "
           "(bloques de departamentos con 50 km de separación); el desempeño se mantiene en ambos esquemas.\n"
           "4. **Complejidad del dataset.** El problema no es trivial: la mejor regla simple llega a AUC-PR 0,73 y el modelo a 0,81, "
           "los residuos conservan estructura espacial y temporal sin explicar y el clima tiene efectos no lineales. Hay "
           "margen claro para los modelos de los siguientes entregables.\n\n"
           "**Conclusión:** el desempeño alto en exactitud es consecuencia del desbalance y de la persistencia natural de las "
           "epidemias, no de una fuga; el dataset es adecuado para el curso."),
    ]
    guardar("11_modelo_base", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
