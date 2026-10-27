"""Cuaderno 3: comparación de los métodos de optimización de hiperparámetros."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Optimización de hiperparámetros: comparación de métodos\n\n"
           "Se comparan Grid, Random, Bayesiana (TPE) y Genética con tres criterios exigidos: (1) la métrica del **bucle externo**, (2) el "
           "**tiempo** total y por evaluación y (3) el **presupuesto** consumido; además se analiza la convergencia con curvas de desempeño "
           "en cualquier momento (*anytime performance*)."),
        code(PREAMBULO + '''
tabla = pd.read_csv(c3.RESULTADOS / "tabla_maestra.csv")  # Tabla maestra (420 registros)
historiales = {}  # Historial de cada tarea
for f in sorted(ex.TAREAS.glob("*.pkl")):  # Cada tarea
    r = pickle.load(open(f, "rb"))  # Registro
    historiales[(r["tarea"], r["modelo"], r["balanceo"], r["optimizador"], r["anio_externo"])] = r  # Indexado
print(f"Tareas: {len(historiales)}")  # Conteo'''),

        md("## 1. Métrica externa, tiempo y presupuesto por optimizador"),
        code('''clf, reg = tabla[tabla["tarea"] == "clf"], tabla[tabla["tarea"] == "reg"]  # Por tarea
resumen = pd.concat({  # Promedios sobre todas las combinaciones y pliegues
    "clasificación": clf.groupby("optimizador").agg(**{"AUC-PR externo": ("AUC-PR", "mean"), "AUC-ROC externo": ("AUC-ROC", "mean"),
                     "evaluaciones": ("evaluaciones", "mean"), "segundos por tarea": ("segundos_optimizacion", "mean"), "segundos por evaluación": ("segundos_por_evaluacion", "mean"),
                     "segundos totales (todas las tareas)": ("segundos_optimizacion", "sum")}),
    "regresión": reg.groupby("optimizador").agg(**{"RMSE externo": ("RMSE", "mean"), "MAE externo": ("MAE", "mean"), "evaluaciones": ("evaluaciones", "mean"),
                 "segundos por tarea": ("segundos_optimizacion", "mean"), "segundos por evaluación": ("segundos_por_evaluacion", "mean"),
                 "segundos totales (todas las tareas)": ("segundos_optimizacion", "sum")})}, axis=0)
resumen'''),
        code('''pares = clf.pivot_table(index=["modelo", "balanceo", "anio_externo"], columns="optimizador", values="AUC-PR")  # Mismo problema, distinto optimizador
mejor = pares.idxmax(axis=1).value_counts().rename("veces que fue el mejor (de 84)")  # Victorias por optimizador
rangos = pares.rank(axis=1, ascending=False).mean().rename("rango medio (1 = mejor)")  # Rango medio
display(pd.concat([mejor, rangos], axis=1))  # Tabla
fig = go.Figure()  # Diferencia de cada optimizador frente al Random Search en el mismo problema
for o in [m for m in op.METODOS if m != "Random Search"]:  # Optimizadores
    fig.add_box(y=pares[o] - pares["Random Search"], name=o, marker_color=COLORES[o], boxmean=True)  # Diferencias pareadas
fig.add_hline(y=0, line_dash="dot", line_color=GRIS)  # Sin diferencia
fig.update_layout(title="AUC-PR externo de cada optimizador menos el de Random Search (mismo modelo, balanceo y pliegue)", yaxis_title="Δ AUC-PR")  # Formato
mostrar(fig, 400)  # Muestra'''),
        md("**Interpretación.** Con el mismo presupuesto (20 evaluaciones), los cuatro optimizadores llegan a desempeños **externos "
           "prácticamente iguales**: AUC-PR externo medio de 0,629 a 0,632 en clasificación y RMSE de 0,490 a 0,492 en regresión. En las "
           "comparaciones pareadas (mismo modelo, balanceo y pliegue) la búsqueda bayesiana gana más veces (28 de 84) y tiene el mejor "
           "rango medio (2,33), seguida muy de cerca por la aleatoria (25; 2,36); la genética (16) y la malla (15) quedan detrás, y la malla "
           "además usa en promedio solo 18,6 evaluaciones porque su malla regular no siempre llena el presupuesto. Las diferencias en la caja "
           "están centradas en 0: elegir el optimizador mueve el AUC-PR externo en milésimas, mucho menos que elegir el modelo. En tiempo, "
           "el genético es el más barato por evaluación (5,8 s frente a 7,7–8,8 s): converge hacia configuraciones parecidas, más baratas "
           "de entrenar, mientras que la búsqueda aleatoria visita con frecuencia regiones costosas (muchos árboles, poca regularización)."),

        md("## 2. Curvas de desempeño en cualquier momento\n\n"
           "Para cada tarea, la curva es el mejor valor de la función objetivo (AUC-PR interno) encontrado hasta la evaluación $t$. Como "
           "cada problema tiene su propia escala, se normaliza a $[0, 1]$ con el peor y el mejor valor observados por **cualquier** método "
           "en ese mismo problema, y se promedia sobre los 84 problemas de clasificación (modelo × balanceo × pliegue externo)."),
        code('''curvas = {o: [] for o in op.METODOS}  # Curva normalizada por optimizador
for (t, m, b, o, a), r in historiales.items():  # Cada tarea
    if t != "clf":  # Solo clasificación
        continue  # Siguiente
    todos = [h["valor"] for oo in op.METODOS for h in historiales[(t, m, b, oo, a)]["historial"]]  # Todos los valores del problema
    lo, hi = min(todos), max(todos)  # Escala del problema
    v = np.maximum.accumulate([h["valor"] for h in r["historial"]])  # Mejor hasta t
    v = np.concatenate([v, np.full(c3.PRESUPUESTO - len(v), v[-1])])[: c3.PRESUPUESTO]  # Grid puede usar menos evaluaciones: se extiende plano
    curvas[o].append((v - lo) / (hi - lo + 1e-12))  # Normalizada
fig = go.Figure()  # Curvas medias
for o, cs in curvas.items():  # Cada optimizador
    cs = np.array(cs)  # Matriz tareas × evaluaciones
    media, ee = cs.mean(axis=0), cs.std(axis=0) / np.sqrt(len(cs))  # Media y error estándar
    x = np.arange(1, c3.PRESUPUESTO + 1)  # Evaluaciones
    fig.add_scatter(x=np.r_[x, x[::-1]], y=np.r_[media + 1.96 * ee, (media - 1.96 * ee)[::-1]], fill="toself", fillcolor=COLORES[o], opacity=0.15, line=dict(width=0), showlegend=False)  # Banda
    fig.add_scatter(x=x, y=media, name=o, line=dict(color=COLORES[o], width=3))  # Media
fig.update_layout(title="Desempeño en cualquier momento (promedio de 84 problemas, banda: IC 95 %)", xaxis_title="evaluación t", yaxis_title="mejor valor hasta t (normalizado)")  # Formato
mostrar(fig, 420)  # Muestra
auc_curva = pd.Series({o: np.mean(np.array(cs).mean(axis=1)) for o, cs in curvas.items()}, name="área bajo la curva anytime (media)")  # Calidad a lo largo del presupuesto
llega = pd.Series({o: np.mean([np.argmax(cv >= 0.99 * cv[-1]) + 1 for cv in cs]) for o, cs in curvas.items()}, name="evaluaciones para llegar al 99 % de su mejor valor")  # Velocidad
pd.concat([auc_curva, llega], axis=1)'''),
        md("**Interpretación.** La búsqueda **bayesiana (TPE)** tiene la mejor curva: el área bajo la curva normalizada es la más alta "
           "(0,913), seguida de la aleatoria (0,886), la genética (0,858) y la malla (0,766). Es decir, en cualquier punto del presupuesto "
           "TPE tiene en promedio la mejor configuración encontrada hasta ese momento. La malla es la más lenta: necesita ≈ 12 evaluaciones "
           "para acercarse al 99 % de su propio mejor valor, porque recorre el espacio en un orden fijo sin aprender. El genético alcanza su "
           "propio óptimo antes (≈ 8,7 evaluaciones), pero a un nivel más bajo: síntoma de convergencia prematura (siguiente sección)."),

        md("## 3. Diversidad genética del algoritmo genético"),
        code('''div = [r["diversidad"] for k, r in historiales.items() if k[3] == "Genética (DEAP)" and r["diversidad"]]  # Diversidad por generación
L = max(map(len, div))  # Generaciones máximas
D = np.array([np.r_[d, [np.nan] * (L - len(d))] for d in div])  # Matriz tareas × generaciones
fig = go.Figure()  # Evolución de la diversidad
fig.add_scatter(x=np.arange(L), y=np.nanmean(D, axis=0), line=dict(color=VERDE, width=3), name="media")  # Media
fig.add_scatter(x=np.arange(L), y=np.nanpercentile(D, 10, axis=0), line=dict(color=VERDE, dash="dot"), name="percentil 10")  # Percentil bajo
fig.update_layout(title="Diversidad de la población por generación (distancia media entre individuos / √d)", xaxis_title="generación", yaxis_title="diversidad")  # Formato
mostrar(fig, 340)  # Muestra
print(f"Diversidad inicial media: {np.nanmean(D[:, 0]):.3f} · final media: {np.nanmean([d[-1] for d in div]):.3f} · tareas con diversidad final < 0,05: {np.mean([d[-1] < 0.05 for d in div]):.0%}")'''),
        md("**Interpretación.** La diversidad cae de 0,37 a 0,05 en pocas generaciones y el 63 % de las corridas termina con diversidad "
           "< 0,05: la población converge rápido. Es el efecto esperado de una población pequeña (6 individuos, impuesta por el presupuesto "
           "de 20 evaluaciones) con elitismo (μ + λ) y selección por torneo: explota bien la región encontrada, pero se arriesga a "
           "**convergencia prematura**. La caché (no se paga dos veces la misma configuración) y el inmigrante aleatorio cuando la generación "
           "no aporta nada nuevo mitigan el problema; con más presupuesto convendría una población mayor o una tasa de mutación adaptativa."),

        md("## 4. Estudio de eficiencia con más presupuesto y varias semillas\n\n"
           "Para demostrar cuantitativamente la eficiencia se repite la búsqueda de **XGBoost sin balanceo** (el espacio más grande: 5 "
           "hiperparámetros) en el pliegue externo 2021, con **40 evaluaciones** y **3 semillas**, e incluye Successive Halving."),
        code('''est = __import__("joblib").load(c3.RESULTADOS / "estudio_optimizadores.joblib")  # 15 corridas
filas, curvas_xgb = [], {}  # Resumen y curvas
for r in est:  # Cada corrida
    h = pd.DataFrame(r["historial"])  # Historial
    completas = h[h["fidelidad"] >= 1.0]  # Evaluaciones completas
    costo = h["fidelidad"].sum()  # Presupuesto en evaluaciones completas equivalentes
    filas.append({"optimizador": r["optimizador"], "semilla": r["semilla_optimizador"], "mejor AUC-PR interno": completas["valor"].max(), "AUC-PR externo": r["AUC-PR"],
                  "evaluaciones": len(h), "costo (evaluaciones completas)": costo, "segundos": r["segundos_optimizacion"]})
    curvas_xgb.setdefault(r["optimizador"], []).append((h["fidelidad"].cumsum().to_numpy(), np.maximum.accumulate(np.where(h["fidelidad"] >= 1, h["valor"], -np.inf))))
res = pd.DataFrame(filas)  # Tabla
agg = res.groupby("optimizador").agg(**{"mejor interno (media)": ("mejor AUC-PR interno", "mean"), "mejor interno (desv. entre semillas)": ("mejor AUC-PR interno", "std"),
                                        "AUC-PR externo (media)": ("AUC-PR externo", "mean"), "AUC-PR externo (desv.)": ("AUC-PR externo", "std"),
                                        "costo (eval. completas)": ("costo (evaluaciones completas)", "mean"), "segundos (media)": ("segundos", "mean")})
mejor_global = res["mejor AUC-PR interno"].max()  # Mejor valor conocido
agg["brecha al mejor conocido"] = mejor_global - agg["mejor interno (media)"]  # Calidad
agg["calidad por minuto (AUC-PR interno / min)"] = agg["mejor interno (media)"] / (agg["segundos (media)"] / 60)  # Calidad por unidad de cómputo
agg'''),
        code('''fig = go.Figure()  # Curvas anytime con el costo en el eje x
for o, cs in curvas_xgb.items():  # Cada optimizador
    for k, (x, y) in enumerate(cs):  # Cada semilla
        y = np.where(np.isfinite(y), y, np.nan)  # Antes de la primera evaluación completa
        fig.add_scatter(x=x, y=y, line=dict(color=COLORES[o], width=2), name=o, legendgroup=o, showlegend=k == 0, opacity=0.8)  # Curva
fig.update_layout(title="XGBoost: mejor AUC-PR interno frente al presupuesto consumido (3 semillas por método)",
                  xaxis_title="presupuesto acumulado (evaluaciones completas equivalentes)", yaxis_title="mejor AUC-PR interno hasta t")  # Formato
mostrar(fig, 440)  # Muestra
umbral = mejor_global - 0.002  # «Casi óptimo»: a 0,002 del mejor conocido
costo_llegar = {}  # Costo para llegar al casi óptimo
for o, cs in curvas_xgb.items():  # Cada optimizador
    costos = [x[np.argmax(y >= umbral)] if np.any(y >= umbral) else np.nan for x, y in cs]  # Primera vez que lo alcanza
    costo_llegar[o] = {"semillas que llegan": int(np.sum(np.isfinite(costos))), "costo medio para llegar": np.nanmean(costos) if np.any(np.isfinite(costos)) else np.nan}
pd.DataFrame(costo_llegar).T'''),
        md("**Interpretación (evidencia cuantitativa de eficiencia).** Con los mismos datos y solo la semilla del optimizador variando:\n\n"
           "* **Calidad de la solución:** TPE encuentra el mejor valor interno medio (0,7171) y la menor brecha al mejor conocido (0,0010); "
           "Random queda a 0,0028 y Grid a 0,0053, aun con 32–40 evaluaciones.\n"
           "* **Calidad por unidad de presupuesto:** las **3 semillas** de TPE llegan a menos de 0,002 del mejor valor conocido con un costo "
           "medio de ≈ 14 evaluaciones; el genético lo logra en 1 de 3 semillas (≈ 14 evaluaciones) y Random y Grid **no lo logran en 40**. "
           "Esa es la ventaja de un modelo sustituto: concentra las evaluaciones donde la mejora esperada es mayor.\n"
           "* **Estabilidad ante semillas:** Random es el más estable (desviación 0,0003), TPE 0,0008 y el genético el más variable (0,0014); "
           "Grid es determinista.\n"
           "* **Multi-fidelidad:** Successive Halving gasta el equivalente a **9 evaluaciones completas** (27 candidatos con 1/9 de los datos, "
           "9 con 1/3 y 3 con todos; factor de reducción 3, fidelidad = fracción de filas de entrenamiento de cada pliegue interno) y tarda "
           "29 s frente a 81–118 s de los demás: la mejor **calidad por minuto** (1,47 frente a 0,36–0,53), a cambio de una brecha de 0,005.\n\n"
           "**Advertencia importante:** las mejoras de la función objetivo interna (milésimas) **no se trasladan** al bucle externo (AUC-PR "
           "0,594–0,598, el mejor fue Grid): por debajo de ≈ 0,005 la optimización ya ajusta el ruido de los pliegues internos. Por eso, en "
           "este problema, más importante que el optimizador es el modelo, y los métodos eficientes (TPE, Successive Halving) valen porque "
           "logran lo mismo con menos cómputo."),

        md("## 5. Visualización del proceso bayesiano\n\n"
           "Historial de una corrida de TPE (semilla 42) y la nube de configuraciones evaluadas en dos de sus dimensiones: muestra dónde "
           "concentró TPE sus evaluaciones (la región que su modelo sustituto considera prometedora)."),
        code('''r = next(r for r in est if r["optimizador"] == "Bayesiana (Optuna TPE)" and r["semilla_optimizador"] == 42)  # Corrida TPE
h = pd.DataFrame(r["historial"])  # Historial
P = pd.DataFrame(list(h["params"]))  # Hiperparámetros evaluados
fig = make_subplots(1, 2, subplot_titles=("Historial: valor de cada evaluación y mejor hasta t", "Configuraciones evaluadas (color: AUC-PR interno)"))  # Dos paneles
fig.add_scatter(x=h["eval"], y=h["valor"], mode="markers", marker=dict(color=AZUL, size=7), name="evaluación", row=1, col=1)  # Puntos
fig.add_scatter(x=h["eval"], y=np.maximum.accumulate(h["valor"]), line=dict(color=ROJO), name="mejor hasta t", row=1, col=1)  # Mejor
fig.add_vline(x=5.5, line_dash="dot", line_color=GRIS, row=1, col=1)  # Fin del arranque aleatorio
fig.add_scatter(x=P["learning_rate"], y=P["max_depth"], mode="markers", marker=dict(color=h["valor"], colorscale="Viridis", size=10, showscale=True,
                colorbar=dict(x=1.02)), text=h["eval"], showlegend=False, row=1, col=2)  # Nube de configuraciones
fig.update_xaxes(type="log", title_text="learning_rate (log)", row=1, col=2)  # Escala log
fig.update_yaxes(title_text="max_depth", row=1, col=2)  # Eje
fig.update_xaxes(title_text="evaluación", row=1, col=1)  # Eje
mostrar(fig, 420)  # Muestra'''),
        md("**Interpretación.** Las 5 primeras evaluaciones son aleatorias (a la izquierda de la línea punteada) y exploran tasas de "
           "aprendizaje de 0,03 a 0,19. Desde la sexta, TPE **explota**: sus propuestas se concentran en tasas de aprendizaje bajas (0,01–0,02), "
           "profundidades de 5 a 7, submuestreo ≈ 0,6 y 100–240 árboles, la región de las mejores configuraciones (AUC-PR interno ≈ 0,717). "
           "Las evaluaciones dispersas que siguen apareciendo son la exploración residual de la razón $l(x)/g(x)$."),
    ]
    guardar("3_optimizacion", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
