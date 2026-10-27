"""Optimización computacional: complejidad teórica frente a tiempos empíricos, variantes eficientes de cada modelo,
paralelización y perfilamiento.

Se ejecuta cuando el experimento principal ya terminó, para que las mediciones no compitan por los núcleos.
Uso (desde la carpeta entregable3):  python correr_complejidad.py
"""

import cProfile  # Perfilamiento de CPU
import io  # Captura del reporte
import pstats  # Lectura del perfil
import time  # Cronómetro
import tracemalloc  # Memoria máxima reservada en Python

import joblib  # Guardado
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from joblib import Parallel, delayed, parallel_backend  # Paralelización
from memory_profiler import memory_usage  # Memoria residente del proceso
from sklearn.ensemble import RandomForestClassifier  # Bosque
from sklearn.kernel_approximation import Nystroem  # Núcleo aproximado
from sklearn.linear_model import Lasso, LogisticRegression, Ridge, SGDClassifier  # Lineales
from sklearn.metrics import average_precision_score, roc_auc_score  # Métricas
from sklearn.model_selection import cross_val_score  # Validación cruzada
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors  # Vecinos
from sklearn.pipeline import make_pipeline  # Pipeline corto
from sklearn.preprocessing import StandardScaler  # Estandarización
from sklearn.svm import SVC, LinearSVC  # SVM
from sklearn.tree import DecisionTreeClassifier  # Árbol
from xgboost import XGBClassifier  # Boosting

from e3 import config as c3, datos as dd  # Configuración y datos

COMPLEJIDAD = {  # Complejidad asintótica teórica (n filas, p variables, k vecinos, T árboles, d profundidad, m componentes)
    "KNN (fuerza bruta)": ("O(1) (almacena)", "O(n·p) por consulta"),
    "KNN (KD-Tree / Ball Tree)": ("O(p·n log n)", "≈ O(log n) por consulta en baja dimensión; se degrada a O(n) con p alto"),
    "KNN (FAISS HNSW, aproximado)": ("O(n log n)", "≈ O(log n) por consulta, con pérdida controlada de exactitud"),
    "Naive Bayes": ("O(n·p)", "O(p) por fila"),
    "Regresión logística": ("O(n·p) por iteración", "O(p) por fila"),
    "Árbol de decisión": ("O(p·n log n)", "O(d) por fila"),
    "Random Forest": ("O(T·p'·n log n), p' = √p", "O(T·d) por fila"),
    "XGBoost (hist)": ("O(T·n·p) (histogramas de B contenedores)", "O(T·d) por fila"),
    "XGBoost (exact)": ("O(T·p·n log n) (ordena cada variable)", "O(T·d) por fila"),
    "SVM núcleo RBF exacto": ("O(n²·p) a O(n³)", "O(n_sv·p) por fila"),
    "SVM Nyström + lineal": ("O(n·m·p + n·m²)", "O(m·p) por fila"),
    "SVM lineal / SGD": ("O(n·p) por época", "O(p) por fila"),
    "Ridge / Lasso (SAGA)": ("O(n·p) por iteración", "O(p) por fila"),
}


def _datos(n=None, semilla=c3.SEMILLA):  # Datos estandarizados (entrenamiento y validación 2022)
    tr, va, _ = dd.cargar()  # Periodos
    if n:  # Submuestra
        tr = dd.submuestra(tr, n, semilla)  # Estratificada
    esc = StandardScaler().fit(tr[dd.X_COLS])  # Estandarización con el entrenamiento
    return esc.transform(tr[dd.X_COLS]), tr[dd.Y_CLF].to_numpy(), esc.transform(va[dd.X_COLS]), va[dd.Y_CLF].to_numpy(), tr, va  # Matrices


def cronometrar(f, repeticiones=3):
    """Mediana del tiempo de pared de ``f`` en varias repeticiones (reduce la varianza de la medición)."""
    t = []  # Tiempos
    for _ in range(repeticiones):  # Repeticiones
        t0 = time.perf_counter()  # Inicio
        salida = f()  # Ejecuta
        t.append(time.perf_counter() - t0)  # Duración
    return float(np.median(t)), salida  # Mediana y última salida


def escalamiento(tamanos=(2500, 5000, 10000, 20000, 40000, 80000)):
    """Tiempo de ajuste y de predicción de cada modelo al variar n; el exponente empírico es la pendiente log-log."""
    X, y, Xv, yv, _, _ = _datos()  # Todo el entrenamiento
    rng = np.random.default_rng(c3.SEMILLA)  # Muestras
    modelos = {"KNN": lambda: KNeighborsClassifier(51, n_jobs=1), "Naive Bayes": GaussianNB, "Regresión logística": lambda: LogisticRegression(C=0.01, max_iter=2000),
               "Árbol de decisión": lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=c3.SEMILLA),
               "Random Forest": lambda: RandomForestClassifier(100, max_depth=12, min_samples_leaf=20, n_jobs=1, random_state=c3.SEMILLA),
               "XGBoost": lambda: XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, tree_method="hist", n_jobs=1, random_state=c3.SEMILLA),
               "SVM (Nyström + lineal)": lambda: make_pipeline(Nystroem(gamma=0.01, n_components=300, random_state=c3.SEMILLA), LinearSVC(C=0.01, dual=False)),
               "SVM RBF exacto": lambda: SVC(C=1, gamma="scale")}
    Xq = Xv[:2000]  # Consultas para el tiempo de predicción
    filas = []  # Resultados
    for nombre, fab in modelos.items():  # Cada modelo
        for n in tamanos:  # Cada tamaño
            if nombre == "SVM RBF exacto" and n > 20000:  # El núcleo exacto se vuelve impracticable
                continue  # Se omite (justamente el punto del análisis)
            i = rng.choice(len(X), n, replace=False)  # Muestra
            t_fit, m = cronometrar(lambda: fab().fit(X[i], y[i]), 3 if n <= 20000 else 1)  # Ajuste
            f = (m.predict_proba if hasattr(m, "predict_proba") else m.decision_function)  # Predicción
            t_pred, _ = cronometrar(lambda: f(Xq), 3)  # Predicción de 2 000 filas
            filas.append({"modelo": nombre, "n": n, "ajuste (s)": t_fit, "predicción 1000 filas (s)": t_pred / 2})  # Registro
    t = pd.DataFrame(filas)  # Tabla
    pend = {}  # Exponentes empíricos
    for nombre, g in t.groupby("modelo"):  # Pendiente log-log
        g = g[g["n"] >= 5000]  # Sin el tamaño más pequeño (dominan costos fijos)
        pend[nombre] = {"exponente ajuste": np.polyfit(np.log(g["n"]), np.log(g["ajuste (s)"]), 1)[0],
                        "exponente predicción": np.polyfit(np.log(g["n"]), np.log(g["predicción 1000 filas (s)"]), 1)[0]}
    return t, pd.DataFrame(pend).T  # Tiempos y exponentes


def knn_variantes(n=60000, k=51):
    """KNN: búsqueda exhaustiva frente a KD-Tree, Ball Tree y FAISS (exacto y aproximado HNSW)."""
    import faiss  # Búsqueda de vecinos de Facebook AI
    X, y, Xv, yv, _, _ = _datos(n)  # Datos
    X32, Xv32 = X.astype(np.float32), Xv.astype(np.float32)  # FAISS trabaja en float32
    ref = NearestNeighbors(n_neighbors=k, algorithm="brute").fit(X).kneighbors(Xv, return_distance=False)  # Vecinos exactos (referencia)
    filas = []  # Resultados
    for alg in ("brute", "kd_tree", "ball_tree"):  # Estructuras de scikit-learn
        t_fit, nn = cronometrar(lambda: NearestNeighbors(n_neighbors=k, algorithm=alg).fit(X), 1)  # Construcción
        t_q, vec = cronometrar(lambda: nn.kneighbors(Xv, return_distance=False), 1)  # Consultas
        filas.append({"método": f"scikit-learn {alg}", "construcción (s)": t_fit, "consulta 1000 (s)": 1000 * t_q / len(Xv), "vecinos": vec})
    for nombre, indice in [("FAISS exacto (IndexFlatL2)", lambda: faiss.IndexFlatL2(X.shape[1])),
                           ("FAISS aproximado (HNSW, M = 32)", lambda: faiss.IndexHNSWFlat(X.shape[1], 32))]:  # FAISS
        def construir():  # Índice con los datos
            ix = indice()  # Índice vacío
            ix.add(X32)  # Agrega los vectores
            return ix  # Índice
        t_fit, ix = cronometrar(construir, 1)  # Construcción
        t_q, (_, vec) = cronometrar(lambda: ix.search(Xv32, k), 1)  # Consultas
        filas.append({"método": nombre, "construcción (s)": t_fit, "consulta 1000 (s)": 1000 * t_q / len(Xv), "vecinos": vec})
    for f in filas:  # Exactitud de la búsqueda y desempeño del clasificador
        vec = f.pop("vecinos")  # Vecinos encontrados
        f["recall@k frente a exacto"] = float(np.mean([len(set(a) & set(b)) / k for a, b in zip(vec, ref)]))  # Fracción de vecinos correctos
        f["AUC-ROC (voto de vecinos)"] = roc_auc_score(yv, y[vec].mean(axis=1))  # KNN con esos vecinos
    return pd.DataFrame(filas).set_index("método")  # Tabla


def lineales_solvers():
    """Regresión logística (lbfgs, liblinear, saga) y Ridge/Lasso (solver por defecto frente a SAGA) con todos los datos."""
    X, y, Xv, yv, tr, va = _datos()  # Todo el entrenamiento
    yr, yrv = np.log1p(tr["casos_obj"]).to_numpy(), np.log1p(va["casos_obj"]).to_numpy()  # Objetivo de regresión
    filas = []  # Resultados
    for solver in ("lbfgs", "liblinear", "saga"):  # Solvers de la logística
        t, m = cronometrar(lambda: LogisticRegression(C=0.01, solver=solver, max_iter=3000, tol=1e-4).fit(X, y), 1)  # Ajuste
        filas.append({"modelo": f"Regresión logística ({solver})", "ajuste (s)": t, "métrica en 2022": roc_auc_score(yv, m.predict_proba(Xv)[:, 1]), "métrica": "AUC-ROC"})
    for nombre, fab in [("Ridge (auto)", lambda: Ridge(alpha=10)), ("Ridge (saga)", lambda: Ridge(alpha=10, solver="saga", tol=1e-4, random_state=c3.SEMILLA)),
                        ("Lasso (descenso por coordenadas)", lambda: Lasso(alpha=0.001, max_iter=3000)),
                        ("Lasso (SGD, penalización L1)", lambda: __import__("sklearn.linear_model", fromlist=["SGDRegressor"]).SGDRegressor(penalty="l1", alpha=0.001, random_state=c3.SEMILLA))]:
        t, m = cronometrar(lambda: fab().fit(X, yr), 1)  # Ajuste
        filas.append({"modelo": nombre, "ajuste (s)": t, "métrica en 2022": float(np.sqrt(np.mean((m.predict(Xv) - yrv) ** 2))), "métrica": "RMSE"})
    return pd.DataFrame(filas).set_index("modelo")  # Tabla


def naive_bayes_incremental(lotes=10):
    """Naive Bayes con ``partial_fit`` por lotes (aprendizaje en línea) frente al ajuste completo: mismo modelo, menos memoria."""
    X, y, Xv, yv, _, _ = _datos()  # Datos
    tracemalloc.start()  # Memoria
    m1 = GaussianNB().fit(X, y)  # Ajuste completo
    _, pico1 = tracemalloc.get_traced_memory()  # Pico
    tracemalloc.stop()  # Fin
    tracemalloc.start()  # Memoria
    m2 = GaussianNB()  # Modelo incremental
    for parte_X, parte_y in zip(np.array_split(X, lotes), np.array_split(y, lotes)):  # Lote por lote (como si llegaran de disco)
        m2.partial_fit(parte_X, parte_y, classes=[0, 1])  # Actualiza medias y varianzas
    _, pico2 = tracemalloc.get_traced_memory()  # Pico
    tracemalloc.stop()  # Fin
    return pd.DataFrame({"AUC-ROC 2022": [roc_auc_score(yv, m1.predict_proba(Xv)[:, 1]), roc_auc_score(yv, m2.predict_proba(Xv)[:, 1])],
                         "memoria pico (MB)": [pico1 / 1e6, pico2 / 1e6],
                         "máx. diferencia de medias": [0.0, float(np.abs(m1.theta_ - m2.theta_).max())]}, index=["fit completo", f"partial_fit en {lotes} lotes"])


def xgboost_variantes():
    """XGBoost: método exacto frente a histogramas, y parada temprana con la validación 2022 como conjunto independiente."""
    X, y, Xv, yv, _, _ = _datos()  # Todo el entrenamiento
    filas = []  # Resultados
    for metodo in ("exact", "hist"):  # Métodos de construcción de árboles
        t, m = cronometrar(lambda: XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.05, tree_method=metodo, n_jobs=4, random_state=c3.SEMILLA).fit(X, y), 1)
        filas.append({"variante": f"tree_method = {metodo} (300 árboles)", "ajuste (s)": t, "árboles": 300, "AUC-PR 2022": average_precision_score(yv, m.predict_proba(Xv)[:, 1])})
    t, m = cronometrar(lambda: XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.05, tree_method="hist", n_jobs=4, early_stopping_rounds=50,
                                             eval_metric="aucpr", random_state=c3.SEMILLA).fit(X, y, eval_set=[(Xv, yv)], verbose=False), 1)  # Parada temprana
    filas.append({"variante": "hist + parada temprana (máx. 2 000, paciencia 50)", "ajuste (s)": t, "árboles": int(m.best_iteration + 1),
                  "AUC-PR 2022": average_precision_score(yv, m.predict_proba(Xv, iteration_range=(0, m.best_iteration + 1))[:, 1])})
    return pd.DataFrame(filas).set_index("variante")  # Tabla


def svm_variantes(tamanos=(2500, 5000, 10000, 20000)):
    """SVM: núcleo RBF exacto frente a Nyström + lineal, LinearSVC y SGD (bisagra) al crecer n."""
    X, y, Xv, yv, _, _ = _datos()  # Datos
    rng = np.random.default_rng(c3.SEMILLA)  # Muestras
    fab = {"SVC RBF exacto": lambda: SVC(C=1, gamma="scale"), "Nyström (300) + LinearSVC": lambda: make_pipeline(Nystroem(gamma=0.01, n_components=300, random_state=c3.SEMILLA), LinearSVC(C=0.01, dual=False)),
           "LinearSVC": lambda: LinearSVC(C=0.01, dual=False), "SGD (bisagra)": lambda: SGDClassifier(loss="hinge", alpha=1e-4, random_state=c3.SEMILLA)}
    filas = []  # Resultados
    for n in tamanos:  # Tamaños
        i = rng.choice(len(X), n, replace=False)  # Muestra
        for nombre, f in fab.items():  # Variantes
            t, m = cronometrar(lambda: f().fit(X[i], y[i]), 1)  # Ajuste
            tp, s = cronometrar(lambda: m.decision_function(Xv), 1)  # Predicción
            filas.append({"variante": nombre, "n": n, "ajuste (s)": t, "predicción 1000 (s)": 1000 * tp / len(Xv), "AUC-PR 2022": average_precision_score(yv, s)})
    return pd.DataFrame(filas)  # Tabla


def paralelizacion():
    """Aceleración de Random Forest con n_jobs y comparación de backends de joblib (loky frente a threading)."""
    X, y, Xv, yv, _, _ = _datos(60000)  # Datos
    base_t = None  # Tiempo secuencial
    filas = []  # Resultados
    for nj in (1, 2, 4, 8):  # Núcleos
        t, _ = cronometrar(lambda: RandomForestClassifier(100, max_depth=12, min_samples_leaf=20, n_jobs=nj, random_state=c3.SEMILLA).fit(X, y), 1)  # Ajuste
        base_t = base_t or t  # Referencia
        filas.append({"configuración": f"Random Forest, n_jobs = {nj}", "segundos": t, "aceleración": base_t / t, "eficiencia": base_t / t / nj})
    for backend in ("loky", "threading"):  # Backends de joblib para paralelizar pliegues
        with parallel_backend(backend, n_jobs=4):  # Contexto
            t, _ = cronometrar(lambda: cross_val_score(LogisticRegression(C=0.01, max_iter=2000), X, y, cv=4, n_jobs=4), 1)  # 4 pliegues en paralelo
        filas.append({"configuración": f"4 pliegues de regresión logística, backend {backend}", "segundos": t, "aceleración": np.nan, "eficiencia": np.nan})
    t, _ = cronometrar(lambda: cross_val_score(LogisticRegression(C=0.01, max_iter=2000), X, y, cv=4, n_jobs=1), 1)  # Secuencial
    filas.append({"configuración": "4 pliegues de regresión logística, secuencial", "segundos": t, "aceleración": np.nan, "eficiencia": np.nan})
    return pd.DataFrame(filas).set_index("configuración")  # Tabla


def memoria():
    """Memoria máxima del proceso durante el ajuste de cada modelo (memory_profiler)."""
    X, y, _, _, _, _ = _datos(40000)  # Datos
    fab = {"KNN": lambda: KNeighborsClassifier(51), "Naive Bayes": GaussianNB, "Regresión logística": lambda: LogisticRegression(C=0.01, max_iter=2000),
           "Árbol de decisión": lambda: DecisionTreeClassifier(max_depth=10), "Random Forest": lambda: RandomForestClassifier(100, max_depth=12, n_jobs=1),
           "XGBoost (hist)": lambda: XGBClassifier(n_estimators=200, max_depth=5, tree_method="hist", n_jobs=1),
           "SVM Nyström + lineal": lambda: make_pipeline(Nystroem(n_components=300, random_state=c3.SEMILLA), LinearSVC(dual=False))}
    filas = {}  # Resultados
    for nombre, f in fab.items():  # Cada modelo
        base = memory_usage(-1, interval=0.05, timeout=0.2, max_usage=True)  # Memoria antes
        pico = memory_usage((lambda: f().fit(X, y), (), {}), interval=0.02, max_usage=True)  # Pico durante el ajuste
        filas[nombre] = {"memoria adicional pico (MB)": max(0.0, float(pico) - float(base))}  # Diferencia
    return pd.DataFrame(filas).T  # Tabla


def perfil(tarea=("clf", "XGBoost", "SMOTE", "Bayesiana (Optuna TPE)", 2021), presupuesto=4):
    """Perfil de CPU (cProfile) de una tarea de validación anidada: ¿dónde se va realmente el tiempo?"""
    from e3 import experimento as ex  # Experimento
    pr = cProfile.Profile()  # Perfilador
    pr.enable()  # Inicia
    ex.ejecutar_tarea(*tarea, presupuesto=presupuesto, guardar=False)  # Tarea
    pr.disable()  # Detiene
    s = io.StringIO()  # Salida
    st = pstats.Stats(pr, stream=s).sort_stats("cumulative")  # Ordena por tiempo acumulado
    filas = []  # Funciones
    for (archivo, linea, funcion), (cc, nc, tt, ct, _) in st.stats.items():  # Cada función
        filas.append({"función": f"{funcion} ({archivo.split('site-packages')[-1].replace(chr(92), '/')[-60:]})", "llamadas": nc, "tiempo propio (s)": tt, "tiempo acumulado (s)": ct})
    return pd.DataFrame(filas).sort_values("tiempo acumulado (s)", ascending=False).head(25).reset_index(drop=True)  # Las 25 más costosas


def main():  # Corre todo y guarda
    salida = {"teoria": pd.DataFrame(COMPLEJIDAD, index=["entrenamiento", "inferencia"]).T}  # Tabla teórica
    for nombre, f in [("escalamiento", escalamiento), ("knn", knn_variantes), ("lineales", lineales_solvers), ("naive_bayes", naive_bayes_incremental),
                      ("xgboost", xgboost_variantes), ("svm", svm_variantes), ("paralelo", paralelizacion), ("memoria", memoria), ("perfil", perfil)]:
        t0 = time.perf_counter()  # Cronómetro
        salida[nombre] = f()  # Calcula
        print(f"  {nombre}: {time.perf_counter() - t0:.0f} s", flush=True)  # Progreso
    joblib.dump(salida, c3.RESULTADOS / "complejidad.joblib")  # Guarda
