"""Cuaderno 4: optimización computacional — complejidad, variantes eficientes, paralelización y perfilamiento."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Optimización computacional\n\n"
           "Cada mejora se sustenta primero en la **complejidad asintótica** (en función de $n$ filas, $p$ variables, $k$ vecinos, $T$ "
           "árboles de profundidad $d$ y $m$ componentes) y después se contrasta con el **tiempo empírico**. Las mediciones se hicieron "
           "con el experimento principal terminado (sin competencia por los núcleos), con la mediana de 3 repeticiones."),
        code(PREAMBULO + '''
import joblib  # Lectura
C = joblib.load(c3.RESULTADOS / "complejidad.joblib")  # Resultados del análisis computacional
C["teoria"]  # Complejidad teórica'''),

        md("## 1. Complejidad teórica frente a tiempo empírico\n\n"
           "Se mide el tiempo de ajuste y de predicción variando $n$ de 2 500 a 80 000 filas. Si el tiempo crece como $n^{b}$, la pendiente "
           "de $\\log t$ frente a $\\log n$ estima el **exponente empírico** $b$, que se compara con el teórico."),
        code('''tiempos, exponentes = C["escalamiento"]  # Tiempos y exponentes
fig = make_subplots(1, 2, subplot_titles=("Ajuste", "Predicción de 1 000 filas"))  # Dos paneles
for j, (nombre, g) in enumerate(tiempos.groupby("modelo")):  # Cada modelo
    col = px.colors.qualitative.Bold[j % 10]  # Color
    fig.add_scatter(x=g["n"], y=g["ajuste (s)"], mode="lines+markers", name=nombre, line=dict(color=col), row=1, col=1)  # Ajuste
    fig.add_scatter(x=g["n"], y=g["predicción 1000 filas (s)"], mode="lines+markers", showlegend=False, line=dict(color=col), row=1, col=2)  # Predicción
fig.update_xaxes(type="log", title_text="n (log)")  # Escala log
fig.update_yaxes(type="log", title_text="segundos (log)")  # Escala log
fig.update_layout(title="Escalamiento empírico (ejes logarítmicos: la pendiente es el exponente)")  # Título
mostrar(fig, 460)  # Muestra
teorico = {"KNN": "0 (ajuste) / 1 (predicción)", "Naive Bayes": "1 / 0", "Regresión logística": "≈ 1 / 0", "Árbol de decisión": "≈ 1 (n log n) / 0",
           "Random Forest": "≈ 1 (n log n) / 0", "XGBoost": "≈ 1 / 0", "SVM (Nyström + lineal)": "1 / 0", "SVM RBF exacto": "2 a 3 / ≈ 1 (n_sv crece con n)"}
exponentes.assign(**{"exponente teórico (ajuste / predicción)": pd.Series(teorico)})'''),
        md("**Interpretación.** Los exponentes empíricos coinciden con la teoría en los casos claros: la regresión logística escala "
           "linealmente (1,00); el árbol y Random Forest, como $n \\log n$ (1,11–1,14); la SVM con Nyström, linealmente (1,13); y la **SVM con "
           "núcleo exacto, cuadráticamente (1,99)**, con una predicción que también crece con $n$ (0,87) porque crece el número de vectores de "
           "soporte. Las desviaciones tienen explicación: KNN no «solo almacena» (scikit-learn construye un árbol en el ajuste, de ahí el "
           "exponente ≈ 1, y la búsqueda en árbol hace la predicción sublineal, 0,68); Naive Bayes y XGBoost (0,62–0,75) quedan por debajo de "
           "1 porque en este rango de $n$ pesan los costos fijos (en XGBoost, construir los histogramas de contenedores). En los modelos de "
           "costo constante por fila, el tiempo de predicción no depende de $n$ (exponente ≈ 0)."),

        md("## 2. KNN: búsqueda exhaustiva, KD-Tree, Ball Tree y FAISS\n\n"
           "La búsqueda exhaustiva cuesta $O(n \\cdot p)$ por consulta. KD-Tree y Ball Tree podan regiones del espacio ($\\approx O(\\log n)$ en baja "
           "dimensión) pero con $p = 34$ la poda pierde eficacia (maldición de la dimensionalidad). FAISS HNSW construye un grafo navegable "
           "de varias capas y encuentra vecinos **aproximados** en tiempo casi logarítmico; el costo es una fracción de vecinos incorrectos "
           "(recall@k < 1)."),
        code('''C["knn"]'''),
        md("**Interpretación.** Con $p = 34$, KD-Tree y Ball Tree son **60–80 veces más lentos** que la fuerza bruta (2,5–3,5 s frente a "
           "0,04 s por 1 000 consultas): la poda deja de funcionar en dimensión media (maldición de la dimensionalidad), mientras que la fuerza "
           "bruta se reduce a un producto de matrices optimizado (BLAS). FAISS **HNSW** responde 13 veces más rápido que la fuerza bruta "
           "(0,003 s) recuperando el 93 % de los vecinos exactos, con un AUC-ROC casi idéntico (0,888 frente a 0,890): el compromiso "
           "exactitud–velocidad es muy favorable porque el voto de 51 vecinos tolera que algunos sean aproximados."),

        md("## 3. Modelos lineales: solvers (SAGA)\n\n"
           "SAGA es un gradiente estocástico con reducción de varianza: cada iteración cuesta $O(p)$ por fila ($O(n \\cdot p)$ por época), "
           "admite L1 y escala a datos grandes y dispersos. Se compara con lbfgs/liblinear (logística) y con Cholesky / descenso por "
           "coordenadas (Ridge / Lasso), con las 132 728 filas del entrenamiento."),
        code('''C["lineales"]'''),
        md("**Interpretación.** Con datos densos, estandarizados y solo 34 variables, **SAGA no es la opción más rápida**: lbfgs ajusta la "
           "logística en 0,4 s frente a 2,5 s de SAGA, y Ridge con su solución cerrada (Cholesky, $O(np^2 + p^3)$) tarda 0,05 s frente a 9,7 s, "
           "con métricas idénticas. SAGA brilla cuando $n$ y $p$ son enormes o dispersos, o con L1 (es el único solver de la logística que "
           "admite todas las penalizaciones). Para Lasso, el SGD con L1 es 35 veces más rápido que el descenso por coordenadas con un RMSE "
           "casi igual (0,463 frente a 0,461). La lección: el solver eficiente depende de la forma de los datos, y aquí se verifica en lugar "
           "de suponerse."),

        md("## 4. Naive Bayes incremental (`partial_fit`)\n\n"
           "Naive Bayes gaussiano solo necesita, por clase, conteos, medias y varianzas: se actualizan por lotes con las fórmulas de "
           "Chan et al., así que el modelo final es el mismo y los datos nunca necesitan estar completos en memoria."),
        code('''C["naive_bayes"]'''),
        md("**Interpretación.** `partial_fit` en 10 lotes produce **exactamente el mismo modelo** (diferencia de medias 0, mismo AUC) con "
           "**10 veces menos memoria** pico (6,7 MB frente a 64,9 MB): los estadísticos suficientes de Naive Bayes se actualizan exactamente. "
           "Permite entrenar con datos que no caben en memoria o que llegan semana a semana."),

        md("## 5. XGBoost: `hist` frente a `exact` y parada temprana"),
        code('''C["xgboost"]'''),
        md("**Interpretación.** El método `hist` es **16 veces más rápido** que `exact` (1,5 s frente a 23,3 s) con el mismo AUC-PR: "
           "discretiza cada variable en contenedores y evita ordenar las 132 728 filas en cada división. La **parada temprana** con 2022 como "
           "conjunto independiente se detiene en 125 árboles (de un máximo de 2 000) y mejora el AUC-PR (0,712 frente a 0,708 con 300 árboles): "
           "menos cómputo y menos sobreajuste a la vez."),

        md("## 6. SVM: núcleo exacto frente a aproximaciones"),
        code('''svm = C["svm"]  # Tiempos por variante y tamaño
fig = make_subplots(1, 2, subplot_titles=("Ajuste (s)", "AUC-PR en 2022"))  # Dos paneles
for j, (v, g) in enumerate(svm.groupby("variante")):  # Variantes
    col = [ROJO, AZUL, VERDE, AMBAR][j % 4]  # Color
    fig.add_scatter(x=g["n"], y=g["ajuste (s)"], mode="lines+markers", name=v, line=dict(color=col), row=1, col=1)  # Tiempo
    fig.add_scatter(x=g["n"], y=g["AUC-PR 2022"], mode="lines+markers", showlegend=False, line=dict(color=col), row=1, col=2)  # Calidad
fig.update_xaxes(type="log", title_text="n (log)")  # Escala
fig.update_yaxes(type="log", row=1, col=1)  # Escala
mostrar(fig, 400)  # Muestra
svm.pivot_table(index="variante", columns="n", values=["ajuste (s)", "AUC-PR 2022"]).round(3)'''),
        md("**Interpretación.** El tiempo del núcleo exacto crece de forma **cuadrática**: de 0,05 s a 3,2 s al pasar de 2 500 a 20 000 "
           "filas (×64 para ×8 filas), y su AUC-PR (0,61–0,64 con hiperparámetros por defecto) no mejora con más datos. Nyström + LinearSVC "
           "escala linealmente (0,57 s con 20 000 filas) y logra AUC-PR 0,673; LinearSVC sin núcleo es aún más rápido (0,08 s) con un AUC-PR "
           "casi igual, señal de que el problema es casi lineal en estas variables. SGD necesita más datos para converger (0,49 con 2 500 filas, "
           "0,66 desde 10 000). Por eso la SVM del experimento usa Nyström: conserva la flexibilidad del núcleo RBF sin el costo cuadrático."),

        md("## 7. Paralelización\n\n"
           "Dos niveles: **entre combinaciones** (el experimento corrió 10 procesos con `joblib`/`loky`, cada uno con `n_jobs = 1` para no "
           "sobresuscribir los núcleos) y **dentro de un modelo** (`n_jobs` de Random Forest)."),
        code('''C["paralelo"]'''),
        md("**Interpretación.** Random Forest acelera 1,9× con 2 núcleos, 3,3× con 4 y 4,9× con 8 (eficiencia 0,61): la paralelización "
           "de árboles es casi perfecta hasta 4 núcleos, y después pesan las partes secuenciales (ley de Amdahl) y que el equipo tiene 6 núcleos "
           "físicos (12 lógicos). Para tareas cortas, paralelizar con procesos (**loky**) es **más lento** que no paralelizar (1,22 s frente "
           "a 0,31 s): crear procesos y serializar los datos cuesta más que el cálculo; con **hilos** (0,23 s) sí se gana, porque los solvers "
           "liberan el GIL. Por eso el experimento paraleliza al nivel más grueso, **entre combinaciones** (10 procesos con `n_jobs = 1` "
           "dentro de cada modelo): 420 tareas en 105 minutos. **Dask o Ray** solo se justificarían para repartir las tareas entre varias "
           "máquinas; en un solo equipo de 12 núcleos, joblib basta. **GPU:** el equipo no tiene GPU NVIDIA, así que `device='cuda'` de "
           "XGBoost no pudo medirse; con `hist` en CPU el ajuste ya tarda ≈ 1,5 s."),

        md("## 8. Perfilamiento y memoria"),
        code('''C["perfil"].head(15)'''),
        code('''C["memoria"]'''),
        md("**Interpretación.** El perfil de una tarea anidada muestra que el **≈ 70 % del tiempo** está en el entrenamiento de XGBoost "
           "(`update` de los árboles) y ≈ 23 % en cargar los datos, un costo que se paga una sola vez por proceso gracias a la caché "
           "(`lru_cache`). Optuna, el remuestreo y el Pipeline cuestan casi nada: optimizarlos sería prematuro. En memoria, la SVM con Nyström "
           "es la más exigente (≈ 280 MB: la matriz de 300 componentes por fila), mientras los árboles necesitan pocos MB."),

        md("## 9. Tabla comparativa: modelo estándar frente a optimizado"),
        code('''knn, lin, xg, nb = C["knn"], C["lineales"], C["xgboost"], C["naive_bayes"]  # Resultados
sv = C["svm"][C["svm"]["n"] == C["svm"]["n"].max()].set_index("variante")  # SVM con el n más grande
mem = C["memoria"]["memoria adicional pico (MB)"]  # Memoria
comparacion = pd.DataFrame([
    {"modelo": "KNN", "estándar": "fuerza bruta", "optimizado": "FAISS HNSW", "complejidad estándar → optimizada (consulta)": "O(n·p) → ≈ O(log n)",
     "consulta 1000 estándar (s)": knn.loc["scikit-learn brute", "consulta 1000 (s)"], "consulta 1000 optimizada (s)": knn.loc["FAISS aproximado (HNSW, M = 32)", "consulta 1000 (s)"],
     "Δ desempeño": knn.loc["FAISS aproximado (HNSW, M = 32)", "AUC-ROC (voto de vecinos)"] - knn.loc["scikit-learn brute", "AUC-ROC (voto de vecinos)"]},
    {"modelo": "XGBoost", "estándar": "exact, 300 árboles", "optimizado": "hist + parada temprana", "complejidad estándar → optimizada (consulta)": "O(T·p·n log n) → O(T'·n·p), T' < T",
     "ajuste estándar (s)": xg.iloc[0]["ajuste (s)"], "ajuste optimizado (s)": xg.iloc[2]["ajuste (s)"], "Δ desempeño": xg.iloc[2]["AUC-PR 2022"] - xg.iloc[0]["AUC-PR 2022"]},
    {"modelo": "SVM", "estándar": "núcleo RBF exacto", "optimizado": "Nyström + LinearSVC", "complejidad estándar → optimizada (consulta)": "O(n²·p)–O(n³) → O(n·m·p)",
     "ajuste estándar (s)": sv.loc["SVC RBF exacto", "ajuste (s)"], "ajuste optimizado (s)": sv.loc["Nyström (300) + LinearSVC", "ajuste (s)"],
     "Δ desempeño": sv.loc["Nyström (300) + LinearSVC", "AUC-PR 2022"] - sv.loc["SVC RBF exacto", "AUC-PR 2022"]},
    {"modelo": "Regresión logística", "estándar": "lbfgs", "optimizado": "saga", "complejidad estándar → optimizada (consulta)": "O(n·p) por iteración (ambos)",
     "ajuste estándar (s)": lin.loc["Regresión logística (lbfgs)", "ajuste (s)"], "ajuste optimizado (s)": lin.loc["Regresión logística (saga)", "ajuste (s)"],
     "Δ desempeño": lin.loc["Regresión logística (saga)", "métrica en 2022"] - lin.loc["Regresión logística (lbfgs)", "métrica en 2022"]},
    {"modelo": "Naive Bayes", "estándar": "fit completo", "optimizado": "partial_fit por lotes", "complejidad estándar → optimizada (consulta)": "O(n·p) (igual); memoria O(n·p) → O(lote·p)",
     "memoria estándar (MB)": nb.iloc[0]["memoria pico (MB)"], "memoria optimizada (MB)": nb.iloc[1]["memoria pico (MB)"], "Δ desempeño": nb.iloc[1]["AUC-ROC 2022"] - nb.iloc[0]["AUC-ROC 2022"]},
]).set_index("modelo")
comparacion'''),
        md("**Interpretación.** Las variantes optimizadas logran: KNN **13× más rápido** en las consultas (FAISS HNSW, −0,001 de AUC); "
           "XGBoost **15× más rápido** en el ajuste y con mejor AUC-PR (hist + parada temprana); SVM **5,6× más rápida** y con +0,047 de "
           "AUC-PR (Nyström frente al núcleo exacto con valores por defecto); Naive Bayes con **10× menos memoria** y el mismo modelo. La "
           "excepción documentada es la logística con SAGA, más lenta que lbfgs en este conjunto denso y pequeño en $p$."),
    ]
    guardar("4_computacional", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
