"""Cuaderno 5: resultados de clasificación — bucle externo, balanceo, prueba, matrices de confusión, ROC y calibración."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Resultados de clasificación\n\n"
           "Primero la métrica del **bucle externo** (media ± desviación de 3 pliegues) para las 112 combinaciones; después la "
           "evaluación única en la **prueba 2023–2024** de los reajustes finales, con el umbral elegido en 2022."),
        code(PREAMBULO + '''
from e3 import evaluacion as ev  # Métricas y calibración
from sklearn.metrics import roc_curve  # Curva ROC
tabla = pd.read_csv(c3.RESULTADOS / "tabla_maestra.csv")  # Tabla maestra
clf = tabla[tabla["tarea"] == "clf"]  # Clasificación
externo = clf.groupby(["modelo", "balanceo", "optimizador"]).agg(**{"AUC-PR media": ("AUC-PR", "mean"), "AUC-PR desv.": ("AUC-PR", "std"),
                                                                  "AUC-ROC media": ("AUC-ROC", "mean"), "AUC-ROC desv.": ("AUC-ROC", "std"),
                                                                  "Brier media": ("Brier", "mean"), "segundos": ("segundos_optimizacion", "sum")}).sort_values("AUC-PR media", ascending=False)
print(f"Combinaciones de clasificación: {len(externo)}")  # 112
externo.head(15)'''),
        code('''mejores = externo.reset_index().groupby("modelo").first().sort_values("AUC-PR media", ascending=False)  # Mejor combinación de cada modelo (según el bucle externo)
mejores[["balanceo", "optimizador", "AUC-PR media", "AUC-PR desv.", "AUC-ROC media", "AUC-ROC desv."]]'''),
        md("**Interpretación.** En el bucle externo el mejor es **XGBoost sin balanceo** optimizado con el algoritmo genético (AUC-PR "
           "0,679 ± 0,104), seguido muy de cerca por Random Forest (0,678) y la SVM con núcleo aproximado (0,671). Los modelos lineales y "
           "de vecindad quedan unas 2–3 centésimas atrás (regresión logística 0,653, KNN 0,651), el árbol único en 0,634 y Naive Bayes "
           "bastante más lejos (0,546): su supuesto de independencia condicional choca con los bloques de variables colineales del EDA. La "
           "**desviación entre pliegues externos (≈ 0,10) es mucho mayor que las diferencias entre los mejores modelos (≈ 0,001–0,01)**: "
           "cambia más el año evaluado (2019 epidémico frente a 2021 de baja transmisión) que el modelo. Por eso las diferencias se juzgan "
           "con pruebas estadísticas formales (capítulo 8) y no a simple vista."),

        md("## Efecto del balanceo"),
        code('''bal = clf.pivot_table(index=["modelo", "optimizador", "anio_externo"], columns="balanceo", values="AUC-PR")  # Pareado: mismo modelo, optimizador y pliegue
dif = bal.sub(bal["Sin balanceo"], axis=0).drop(columns="Sin balanceo")  # Diferencia frente a no balancear
fig = go.Figure()  # Diferencias por modelo
for b, col in zip(dif.columns, [AMBAR, ROJO, AZUL]):  # Estrategias
    g = dif[b].groupby(level="modelo").mean()  # Media por modelo
    fig.add_bar(x=g.index, y=g.values, name=b, marker_color=col)  # Barras
fig.add_hline(y=0, line_color=GRIS)  # Sin efecto
fig.update_layout(barmode="group", title="AUC-PR externo de cada estrategia de balanceo menos «Sin balanceo» (media por modelo)", yaxis_title="Δ AUC-PR")  # Formato
mostrar(fig, 400)  # Muestra
pd.DataFrame({"Δ AUC-PR medio": dif.mean(), "veces que mejora": (dif > 0).mean(), "Δ AUC-ROC medio":
              clf.pivot_table(index=["modelo", "optimizador", "anio_externo"], columns="balanceo", values="AUC-ROC").pipe(lambda t: t.sub(t["Sin balanceo"], axis=0)).drop(columns="Sin balanceo").mean()})'''),
        md("**Interpretación.** El balanceo **no mejora** la discriminación: en promedio, ADASYN reduce el AUC-PR externo en 0,024, "
           "SMOTE en 0,010 y `class_weight` en 0,005, y solo mejora en el 13–20 % de los casos pareados. El AUC-ROC casi no cambia. Hay "
           "tres razones: (1) el AUC-ROC y el AUC-PR no dependen del umbral, y pesar o replicar la clase minoritaria cambia la escala de "
           "las probabilidades más que su orden; (2) en un panel temporal, los brotes sintéticos interpolan semanas de municipios y años "
           "distintos y agregan ruido, sobre todo ADASYN, que los concentra en la frontera más confusa; y (3) con ≈ 10 % de brotes y más de "
           "13 000 casos de la clase minoritaria, el desbalance no es extremo. El desbalance se maneja mejor **eligiendo el umbral** en 2022: "
           "así se gana recall sin deformar las probabilidades."),

        md("## Evaluación en la prueba 2023–2024\n\n"
           "Cada combinación se reajustó con 2015–2021; su umbral maximiza F1 en 2022. La SVM entrega márgenes: para sus métricas de "
           "probabilidad (Brier) se usa el escalado de Platt ajustado en 2022."),
        code('''tr, va, te = dd.cargar()  # Periodos
y_va, y_te = va["y"].to_numpy(), te["y"].to_numpy()  # Objetivos
finales = {}  # Predicciones de las 112 combinaciones
for f in sorted(ex.FINALES.glob("clf__*.pkl")):  # Cada reajuste final
    r = pickle.load(open(f, "rb"))  # Predicciones
    finales[(r["modelo"], r["balanceo"], r["optimizador"])] = r  # Indexado
filas = []  # Métricas en la prueba
for k, r in finales.items():  # Cada combinación
    p_va, p_te = r["p_va"].astype(float), r["p_te"].astype(float)  # Puntajes
    if k[0] == "SVM":  # Márgenes → probabilidad con Platt (ajustado en 2022)
        p_te, _ = ev.recalibrar(p_va, y_va, p_te)  # Platt
        p_va, _ = ev.recalibrar(p_va, y_va, p_va)  # Validación en la misma escala
    u = ev.umbral_f1(y_va, p_va)  # Umbral de 2022
    filas.append({"modelo": k[0], "balanceo": k[1], "optimizador": k[2], "umbral": u, **ev.metricas_clf(y_te, p_te, u), "ECE": ev.ece(y_te, p_te)})  # Métricas
prueba = pd.DataFrame(filas).set_index(["modelo", "balanceo", "optimizador"])  # Tabla
sel = [tuple(x) for x in mejores.reset_index()[["modelo", "balanceo", "optimizador"]].to_numpy()]  # Las 7 elegidas por el bucle externo
prueba.loc[sel, ["Accuracy", "Precision", "Recall", "F1", "AUC-ROC", "AUC-PR", "Brier", "ECE", "umbral"]].sort_values("AUC-PR", ascending=False)'''),
        md("**Interpretación.** En la prueba 2023–2024, **XGBoost sin balanceo** es el mejor en AUC-PR (0,823) y AUC-ROC (0,898), con "
           "F1 0,73: anticipa el 76 % de los brotes con 70 % de precisión. Supera al modelo base logístico de los entregables anteriores "
           "(AUC-PR 0,809). La SVM alcanza casi el mismo AUC-PR (0,820) pero con un umbral más bajo: más recall (0,83) a costa de "
           "precisión (0,64). El orden de los modelos en la prueba coincide en lo esencial con el del bucle externo, lo que indica que la "
           "validación anidada estimó bien la generalización. Naive Bayes vuelve a ser el peor (AUC-PR 0,676)."),
        code('''prueba.sort_values("AUC-PR", ascending=False)  # Las 112 combinaciones en la prueba'''),

        md("## Matrices de confusión y curvas ROC de los 7 modelos"),
        code('''fig = make_subplots(2, 4, subplot_titles=[f"{m}<br><sup>{b} · {o}</sup>" for m, b, o in sel] + ["Curvas ROC"], specs=[[{}] * 4, [{}] * 3 + [{}]])  # Rejilla
for j, k in enumerate(sel):  # Cada modelo
    r = finales[k]  # Predicciones
    p_va, p_te = r["p_va"].astype(float), r["p_te"].astype(float)  # Puntajes
    u = ev.umbral_f1(y_va, p_va)  # Umbral
    cm = ev.matriz(y_te, p_te, u)  # Matriz
    fila, col = (j // 4) + 1, (j % 4) + 1  # Posición
    if j < 7:  # Matrices
        fig.add_heatmap(z=cm, x=["pred. no", "pred. sí"], y=["real no", "real sí"], text=[[f"VN {cm[0, 0]:,}", f"FP {cm[0, 1]:,}"], [f"FN {cm[1, 0]:,}", f"VP {cm[1, 1]:,}"]],
                        texttemplate="%{text}", colorscale="Blues", showscale=False, row=fila, col=col)
        fig.update_yaxes(autorange="reversed", row=fila, col=col)  # Real arriba
roc_fila, roc_col = 2, 4  # Panel de las curvas ROC
for j, k in enumerate(sel):  # Curvas
    fpr, tpr, _ = roc_curve(y_te, finales[k]["p_te"].astype(float))  # ROC
    fig.add_scatter(x=fpr, y=tpr, name=k[0], line=dict(color=px.colors.qualitative.Bold[j]), row=roc_fila, col=roc_col)  # Curva
fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GRIS, dash="dot"), showlegend=False, row=roc_fila, col=roc_col)  # Azar
fig.update_layout(title="Prueba 2023–2024: matriz de confusión (umbral de 2022) y curva ROC del mejor de cada modelo")  # Título
mostrar(fig, 680)  # Muestra'''),
        md("**Interpretación.** Las matrices muestran el compromiso de cada umbral elegido en 2022: los modelos con umbral bajo (SVM, KNN) "
           "tienen más verdaderos positivos y más falsas alarmas; los de umbral alto, al revés. Las curvas ROC de XGBoost, Random Forest y "
           "SVM prácticamente se superponen; la de Naive Bayes queda claramente por debajo."),

        md("## Calibración de probabilidades\n\n"
           "Diagrama de confiabilidad, Brier y ECE; después, recalibración con **Platt** (logística sobre el logit) y **regresión isotónica**, "
           "ambas ajustadas en 2022 y evaluadas en la prueba."),
        code('''fig = make_subplots(1, 2, subplot_titles=("Sin recalibrar", "Recalibrado con Platt (2022)"))  # Dos paneles
filas = []  # Efecto de la recalibración
for j, k in enumerate(sel):  # Cada modelo
    p_va, p_te = finales[k]["p_va"].astype(float), finales[k]["p_te"].astype(float)  # Puntajes
    platt, iso = ev.recalibrar(p_va, y_va, p_te)  # Recalibradas
    crudo = p_te if k[0] != "SVM" else platt  # La SVM no tiene probabilidad cruda
    col = px.colors.qualitative.Bold[j]  # Color
    for c, p in [(1, crudo), (2, platt)]:  # Paneles
        cf_ = ev.confiabilidad(y_te, p)  # Diagrama
        fig.add_scatter(x=cf_["predicha"], y=cf_["observada"], mode="lines+markers", name=k[0], line=dict(color=col), showlegend=c == 1, row=1, col=c)  # Curva
    filas.append({"modelo": k[0], "balanceo": k[1], "Brier crudo": ev.metricas_clf(y_te, crudo, 0.5).get("Brier", np.nan), "ECE crudo": ev.ece(y_te, crudo),
                  "Brier Platt": ev.metricas_clf(y_te, platt, 0.5)["Brier"], "ECE Platt": ev.ece(y_te, platt),
                  "Brier isotónica": ev.metricas_clf(y_te, iso, 0.5)["Brier"], "ECE isotónica": ev.ece(y_te, iso)})
for c in (1, 2):  # Diagonal
    fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(color=GRIS, dash="dot"), showlegend=False, row=1, col=c)
fig.update_xaxes(title_text="probabilidad predicha")  # Ejes
fig.update_yaxes(title_text="frecuencia observada")  # Ejes
mostrar(fig, 440)  # Muestra
pd.DataFrame(filas).set_index("modelo")'''),
        code('''cal_bal = prueba.groupby("balanceo")[["Brier", "ECE", "AUC-ROC", "Recall", "Precision"]].mean()  # Efecto del balanceo en la calibración (promedio de modelos y optimizadores)
cal_bal'''),
        md("**Interpretación.** Los modelos **sin balanceo** están razonablemente calibrados (ECE ≈ 0,04–0,07); los que usaron balanceo "
           "(Random Forest y árbol con `class_weight`, Naive Bayes con ADASYN) tienen ECE ≈ 0,19–0,21: aprendieron una prevalencia ficticia "
           "del 50 % y **sobreestiman** el riesgo (sus umbrales óptimos suben a 0,75–1,0). Recalibrar con **Platt** en 2022 baja el ECE a "
           "0,01–0,06 y el Brier de Random Forest de 0,174 a 0,114; la **regresión isotónica** logra un efecto parecido y es mejor para la SVM "
           "(0,047 → 0,012). En promedio sobre las 112 combinaciones, no balancear deja el mejor Brier (0,131) y ECE (0,067); el balanceo sube "
           "el recall solo ≈ 1 punto. Conclusión: el balanceo **cambia la calibración, no la discriminación**."),
    ]
    guardar("5_clasificacion", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
