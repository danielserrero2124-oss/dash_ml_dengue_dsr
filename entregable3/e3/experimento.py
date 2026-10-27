"""Validación anidada de las 140 combinaciones y reajuste final.

* Bucle externo: 3 pliegues por año (2019, 2020, 2021). Cada uno entrena con los años anteriores (purga de 4 semanas) y
  mide el desempeño de generalización en el año evaluado completo. Es la métrica que se reporta.
* Bucle interno: dentro del entrenamiento de cada pliegue externo, 2 pliegues hacia adelante (los dos últimos años). El
  optimizador (Grid, Random, Bayesiano o Genético) solo ve el promedio de estos pliegues internos.
* Reajuste final: con los hiperparámetros del pliegue externo más reciente (2021), se ajusta sobre 2015–2021 y se predice
  la validación 2022 (umbral y calibración) y la prueba 2023–2024 (evaluación única).

Cada tarea se guarda en su propio archivo (resultados/tareas/), de modo que una ejecución interrumpida se reanuda.

Uso (desde la carpeta entregable3):  python -m e3.experimento
"""

import functools  # Caché de los datos por proceso
import os  # Variables de entorno
import pickle  # Guardado de cada tarea
import sys  # Rutas de importación
import time  # Cronómetro
import traceback  # Registro de errores
import warnings  # Control de avisos

AQUI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Carpeta entregable3
for ruta in (AQUI, os.path.join(AQUI, ".."), os.path.join(AQUI, "..", "entregable1")):  # e3, src y e1
    if ruta not in sys.path:  # Evita duplicados
        sys.path.insert(0, ruta)  # Agrega
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # Evita un aviso de joblib en Windows
warnings.filterwarnings("ignore")  # Oculta avisos de convergencia y versiones

import numpy as np  # noqa: E402  Cálculo numérico
import pandas as pd  # noqa: E402  Tablas
from joblib import Parallel, delayed  # noqa: E402  Paralelización entre combinaciones
from sklearn.metrics import (average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error,  # noqa: E402  Métricas
                             r2_score, roc_auc_score)

from e3 import config as c3, datos as dd, modelos as mo, optimizadores as op  # noqa: E402  Módulos del entregable

TAREAS = c3.RESULTADOS / "tareas"  # Un archivo por combinación × pliegue externo
FINALES = c3.RESULTADOS / "finales"  # Un archivo por combinación (predicciones de 2022 y 2023–2024)
PESADOS = {"Random Forest": 0, "XGBoost": 1, "SVM": 2, "SVR": 2, "KNN": 3}  # Orden de ejecución (los costosos primero)


@functools.lru_cache(maxsize=1)  # Una sola carga por proceso
def datos():
    """Panel de modelado (entrenamiento, validación y prueba) cargado una vez por proceso."""
    return dd.cargar()  # Tres periodos purgados


def _nombre(tarea, modelo, balanceo, metodo, anio=None):  # Nombre de archivo de una tarea
    base = f"{tarea}__{modelo}__{balanceo}__{metodo}" + (f"__{anio}" if anio else "")  # Identificador
    return "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in base) + ".pkl"  # Solo caracteres seguros


def _metricas(tarea, y, s, probabilistico):  # Métricas del bucle externo
    if tarea == "clf":  # Clasificación: independientes del umbral
        m = {"AUC-ROC": roc_auc_score(y, s), "AUC-PR": average_precision_score(y, s)}  # Discriminación
        if probabilistico:  # Calibración (solo si el puntaje es una probabilidad)
            m["Brier"] = brier_score_loss(y, np.clip(s, 0, 1))  # Error cuadrático de la probabilidad
        return m  # Métricas
    return {"RMSE": float(np.sqrt(mean_squared_error(y, s))), "MAE": mean_absolute_error(y, s), "R2": r2_score(y, s)}  # Regresión


def ejecutar_tarea(tarea, modelo, balanceo, metodo, anio, semilla=c3.SEMILLA, presupuesto=c3.PRESUPUESTO, guardar=True, semilla_optimizador=None):
    """Ejecuta un pliegue externo de la validación anidada para una combinación.

    Parameters
    ----------
    tarea : {"clf", "reg"}
        Clasificación o regresión.
    modelo, balanceo, metodo : str
        Combinación (balanceo = "—" en regresión).
    anio : int
        Año del pliegue externo.
    semilla : int
        Semilla del optimizador, del muestreo y de los modelos.
    presupuesto : int
        Evaluaciones de hiperparámetros.
    guardar : bool
        Si True, guarda el resultado en resultados/tareas.
    semilla_optimizador : int, optional
        Semilla solo del optimizador (los datos y los modelos conservan ``semilla``); sirve para aislar la variabilidad del
        método de búsqueda en el estudio de estabilidad.

    Returns
    -------
    dict
        Registro con los mejores hiperparámetros, las métricas del bucle externo, los tiempos y el historial.
    """
    c3.fijar_semilla(semilla)  # Semilla global en este proceso
    tr, _, _ = datos()  # Entrenamiento 2015–2021
    y_col = dd.Y_CLF if tarea == "clf" else dd.Y_REG  # Objetivo
    espacio = (mo.ESPACIOS_CLF if tarea == "clf" else mo.ESPACIOS_REG)[modelo]  # Espacio de búsqueda
    ent_full, ev = dd.pliegue_externo(tr, anio)  # Pliegue externo
    ent = dd.submuestra(ent_full, c3.N_EXTERNO, semilla)  # Submuestra estratificada del entrenamiento externo
    X, y = ent[dd.X_COLS].to_numpy(), ent[y_col].to_numpy()  # Matrices
    internos = dd.pliegues_internos(ent)  # Pliegues internos hacia adelante
    rng = np.random.default_rng(semilla)  # Para las fidelidades parciales
    errores = []  # Configuraciones que fallaron

    def objetivo(p, fidelidad=1.0):  # Promedio de los pliegues internos (lo único que ve el optimizador)
        puntajes = []  # Por pliegue interno
        for i_tr, i_va in internos:  # Cada pliegue interno
            if fidelidad < 1.0:  # Multi-fidelidad: fracción de las filas de entrenamiento
                i_tr = np.sort(rng.choice(i_tr, max(200, int(fidelidad * len(i_tr))), replace=False))  # Submuestra
            try:  # Algunas combinaciones pueden fallar (por ejemplo, ADASYN sin vecinos minoritarios)
                m = mo.construir(tarea, modelo, p, balanceo, semilla).fit(X[i_tr], y[i_tr])  # Pipeline completo ajustado en el pliegue
                s = mo.puntaje(m, X[i_va]) if tarea == "clf" else m.predict(X[i_va])  # Predicción en la validación interna
                puntajes.append(average_precision_score(y[i_va], s) if tarea == "clf" else -np.sqrt(mean_squared_error(y[i_va], s)))  # AUC-PR o −RMSE
            except Exception as e:  # Falla registrada
                errores.append(repr(e))  # Guarda el motivo
                puntajes.append(0.0 if tarea == "clf" else -10.0)  # Peor puntaje posible
        return float(np.mean(puntajes))  # Promedio

    s_opt = semilla if semilla_optimizador is None else semilla_optimizador  # Semilla del optimizador
    t0 = time.perf_counter()  # Cronómetro total
    diversidad = None  # Solo para el genético
    if metodo == "Genética (DEAP)":  # Devuelve también la diversidad
        historial, diversidad = op.genetica(espacio, objetivo, presupuesto, s_opt)
    elif metodo == "Successive Halving":  # Multi-fidelidad (análisis complementario de eficiencia)
        historial = op.halving(espacio, objetivo, semilla=s_opt)
    else:  # Los demás
        historial = op.METODOS[metodo](espacio, objetivo, presupuesto, s_opt)
    t_opt = time.perf_counter() - t0  # Tiempo de optimización
    completas = [h for h in historial if h["fidelidad"] >= 1.0] or historial  # Solo evaluaciones con fidelidad completa
    mejor = max(completas, key=lambda h: h["valor"])  # Mejor configuración según el bucle interno
    t1 = time.perf_counter()  # Cronómetro del reajuste
    m = mo.construir(tarea, modelo, mejor["params"], balanceo, semilla).fit(X, y)  # Reajuste con todo el entrenamiento externo
    t_fit = time.perf_counter() - t1  # Tiempo de ajuste
    t2 = time.perf_counter()  # Cronómetro de predicción
    Xe, ye = ev[dd.X_COLS].to_numpy(), ev[y_col].to_numpy()  # Año evaluado completo
    s = mo.puntaje(m, Xe) if tarea == "clf" else m.predict(Xe)  # Predicción
    t_pred = time.perf_counter() - t2  # Tiempo de inferencia
    probabilistico = tarea == "clf" and modelo != "SVM"  # La SVM da márgenes, no probabilidades
    registro = {"tarea": tarea, "modelo": modelo, "balanceo": balanceo, "optimizador": metodo, "anio_externo": anio, "semilla": semilla, "semilla_optimizador": s_opt,
                "mejores_params": mejor["params"], "valor_interno": mejor["valor"], **_metricas(tarea, ye, s, probabilistico),
                "evaluaciones": len(historial), "segundos_optimizacion": t_opt, "segundos_por_evaluacion": t_opt / len(historial),
                "segundos_ajuste": t_fit, "segundos_prediccion_1000": 1000 * t_pred / len(ye), "n_entrenamiento": len(ent),
                "n_evaluacion": len(ye), "errores": len(errores), "historial": historial, "diversidad": diversidad}
    if guardar:  # Guarda en disco
        TAREAS.mkdir(parents=True, exist_ok=True)  # Carpeta
        with open(TAREAS / _nombre(tarea, modelo, balanceo, metodo, anio), "wb") as f:  # Archivo de la tarea
            pickle.dump(registro, f)  # Guarda
    return registro  # Registro


def reajuste_final(tarea, modelo, balanceo, metodo, semilla=c3.SEMILLA, guardar=True):
    """Ajusta la combinación sobre 2015–2021 con los hiperparámetros del pliegue externo 2021 y predice 2022 y 2023–2024.

    Returns
    -------
    dict
        Predicciones de validación y prueba, hiperparámetros y tiempos.
    """
    c3.fijar_semilla(semilla)  # Semilla
    with open(TAREAS / _nombre(tarea, modelo, balanceo, metodo, c3.ANIOS_EXTERNOS[-1]), "rb") as f:  # Pliegue externo más reciente
        params = pickle.load(f)["mejores_params"]  # Sus hiperparámetros
    tr, va, te = datos()  # Periodos
    y_col = dd.Y_CLF if tarea == "clf" else dd.Y_REG  # Objetivo
    ent = dd.submuestra(tr, c3.N_FINAL, semilla)  # Submuestra de 2015–2021
    t0 = time.perf_counter()  # Cronómetro
    m = mo.construir(tarea, modelo, params, balanceo, semilla).fit(ent[dd.X_COLS].to_numpy(), ent[y_col].to_numpy())  # Ajuste
    t_fit = time.perf_counter() - t0  # Tiempo
    f = (lambda X: mo.puntaje(m, X)) if tarea == "clf" else m.predict  # Predicción
    t0 = time.perf_counter()  # Cronómetro de inferencia
    p_te = f(te[dd.X_COLS].to_numpy()).astype(np.float32)  # Prueba
    t_pred = time.perf_counter() - t0  # Tiempo
    salida = {"tarea": tarea, "modelo": modelo, "balanceo": balanceo, "optimizador": metodo, "params": params,
              "p_va": f(va[dd.X_COLS].to_numpy()).astype(np.float32), "p_te": p_te, "p_tr": f(ent[dd.X_COLS].to_numpy()).astype(np.float32),
              "idx_tr": ent.index.to_numpy(), "segundos_ajuste": t_fit, "segundos_prediccion_1000": 1000 * t_pred / len(te)}
    if guardar:  # Guarda
        FINALES.mkdir(parents=True, exist_ok=True)  # Carpeta
        with open(FINALES / _nombre(tarea, modelo, balanceo, metodo), "wb") as fh:  # Archivo
            pickle.dump(salida, fh)  # Guarda
    return salida  # Predicciones


def _seguro(funcion, *args):  # Ejecuta una tarea sin detener el resto si falla
    try:  # Intenta
        funcion(*args)  # Ejecuta
        return None  # Sin error
    except Exception:  # Error inesperado
        return (args, traceback.format_exc())  # Devuelve el error para registrarlo


def combinaciones():
    """Lista de las 140 combinaciones (tarea, modelo, balanceo, optimizador)."""
    clf = [("clf", m, b, o) for m in mo.MODELOS_CLF for b in mo.BALANCEOS for o in op.METODOS]  # 7 × 4 × 4 = 112
    reg = [("reg", m, "—", o) for m in mo.MODELOS_REG for o in op.METODOS]  # 7 × 4 = 28
    return clf + reg  # 140


def tabla_maestra():
    """Une todos los registros del bucle externo en la tabla maestra (una fila por combinación × pliegue externo)."""
    filas = []  # Registros
    for archivo in sorted(TAREAS.glob("*.pkl")):  # Cada tarea
        with open(archivo, "rb") as f:  # Lee
            r = pickle.load(f)  # Registro
        filas.append({k: v for k, v in r.items() if k not in ("historial", "diversidad")})  # Sin el historial
    t = pd.DataFrame(filas)  # Tabla
    t["mejores_params"] = t["mejores_params"].astype(str)  # Texto legible en CSV
    t.to_csv(c3.RESULTADOS / "tabla_maestra.csv", index=False, encoding="utf-8")  # Guarda
    return t  # Tabla


def main():  # Ejecuta todo lo pendiente
    c3.RESULTADOS.mkdir(parents=True, exist_ok=True)  # Carpeta de resultados
    TAREAS.mkdir(parents=True, exist_ok=True)  # Carpeta de tareas
    pendientes = [(t, m, b, o, a) for (t, m, b, o) in combinaciones() for a in c3.ANIOS_EXTERNOS
                  if not (TAREAS / _nombre(t, m, b, o, a)).exists()]  # Tareas sin resultado
    pendientes.sort(key=lambda x: PESADOS.get(x[1], 9))  # Las más costosas primero (mejor reparto entre procesos)
    print(f"Validación anidada: {len(pendientes)} tareas pendientes de {len(combinaciones()) * len(c3.ANIOS_EXTERNOS)}", flush=True)  # Progreso
    t0 = time.perf_counter()  # Cronómetro
    fallos = Parallel(n_jobs=c3.TRABAJOS, verbose=5)(delayed(_seguro)(ejecutar_tarea, *p) for p in pendientes)  # En paralelo
    print(f"Bucle externo terminado en {(time.perf_counter() - t0) / 60:.1f} min", flush=True)  # Tiempo
    finales = [c for c in combinaciones() if not (FINALES / _nombre(*c)).exists()]  # Reajustes pendientes
    fallos += Parallel(n_jobs=c3.TRABAJOS, verbose=5)(delayed(_seguro)(reajuste_final, *c) for c in finales)  # En paralelo
    for f in [f for f in fallos if f]:  # Errores
        print("FALLO:", f[0], f[1][-600:], flush=True)  # Registro
    tabla_maestra()  # Tabla maestra
    print(f"Total: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)  # Tiempo total


if __name__ == "__main__":  # Ejecución directa
    main()  # Corre
