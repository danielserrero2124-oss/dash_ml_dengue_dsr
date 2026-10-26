"""Entrena y evalúa los cuatro modelos recomendados y el modelo original (HetDS en línea); guarda los artefactos.

Flujo sin fuga de datos:
1. División temporal: entrenamiento 2015–2021, validación 2022, prueba 2023–2024 (se usa una sola vez).
2. Cada modelo recomendado es un Pipeline ajustado con GridSearchCV y validación temporal anual (2018–2021).
3. El umbral de decisión y los hiperparámetros del modelo original se eligen en 2022.
4. HetDS predice la prueba semana a semana usando solo etiquetas ya confirmadas (retraso de 4 semanas).

Ejecución (desde la raíz del proyecto):  python -m src.entrenamiento
"""

import os  # Variables de entorno
import time  # Tiempos de cómputo

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # Evita un aviso de joblib en Windows

import joblib  # Guardado de artefactos
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from sklearn.ensemble import HistGradientBoostingClassifier  # Modelo auxiliar para los análisis de aporte del clima y de fuga
from sklearn.inspection import permutation_importance  # Importancia por permutación
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # Métricas y curvas
from sklearn.model_selection import train_test_split  # Muestras estratificadas

from src import config as cf  # Configuración
from src import datos as dt  # Panel de datos
from src import modelado as mod  # Modelos y pruebas
from src.hetds import HetDSEnLinea  # Modelo original

TAM_MUESTRA_KNN = 30_000  # Filas de entrenamiento para KNN
TAM_MUESTRA_IMPORTANCIA = 4_000  # Filas de prueba para la importancia por permutación
MALLA_HETDS = [(eta, v) for eta in (0.01, 0.05, 0.2) for v in (4, 8, 26)]  # Hiperparámetros del modelo original (se eligen en 2022)


def reducir_curva(x, y, max_puntos=400):  # Reduce los puntos de una curva para el dashboard
    if len(x) <= max_puntos:  # Ya es pequeña
        return np.asarray(x).tolist(), np.asarray(y).tolist()  # Sin cambios
    idx = np.unique(np.linspace(0, len(x) - 1, max_puntos).astype(int))  # Índices equiespaciados
    return np.asarray(x)[idx].tolist(), np.asarray(y)[idx].tolist()  # Curva reducida


def resumen_cv(grid):  # Tabla con todas las combinaciones probadas en GridSearchCV
    r = grid.cv_results_  # Resultados
    filas = [{"params": {k.replace("clf__", ""): str(v) for k, v in pr.items()}, "cv_auc_pr": float(r["mean_test_auc_pr"][i]),  # Combinación y métricas
              "cv_auc_roc": float(r["mean_test_auc_roc"][i]), "train_auc_roc": float(r["mean_train_auc_roc"][i]), "rank": int(r["rank_test_auc_pr"][i])}
             for i, pr in enumerate(r["params"])]
    return sorted(filas, key=lambda f: f["rank"])  # De mejor a peor


def evaluar(nombre, p_tr, y_tr, p_va, y_va, p_te, te, t_seg, extra=None):  # Métricas y curvas comunes a todos los modelos
    y_te = te[cf.TARGET].to_numpy()  # Objetivo de prueba
    umbral = mod.umbral_max_f1(y_va, p_va)  # Umbral elegido en validación
    fpr, tpr, _ = roc_curve(y_te, p_te)  # Curva ROC
    prec, rec, _ = precision_recall_curve(y_te, p_te)  # Curva PR
    anual = {int(a): float(roc_auc_score(y_te[(te["ANO"] == a).to_numpy()], p_te[(te["ANO"] == a).to_numpy()])) for a in sorted(te["ANO"].unique())}  # AUC por año
    r = {"train_auc_roc": float(roc_auc_score(y_tr, p_tr)) if p_tr is not None else None,  # AUC en entrenamiento
         "val_auc_roc": float(roc_auc_score(y_va, p_va)), "val_auc_pr": float(average_precision_score(y_va, p_va)),  # Validación
         "umbral": umbral, "test": mod.metricas(y_te, p_te, umbral), "ic_auc": mod.ic_auc_delong(y_te, p_te),  # Prueba
         "auc_anual": anual, "p_test": p_te.astype(np.float32), "p_val": p_va.astype(np.float32),  # Predicciones
         "roc": reducir_curva(fpr, tpr), "pr": reducir_curva(rec[::-1], prec[::-1]), "segundos": round(t_seg, 1)}  # Curvas y tiempo
    r.update(extra or {})  # Datos propios del modelo
    print(f"{nombre:<20} val AUC-PR={r['val_auc_pr']:.4f}  test AUC-ROC={r['test']['auc_roc']:.4f}  AUC-PR={r['test']['auc_pr']:.4f}  ({r['segundos']} s)", flush=True)
    return r  # Resultados


def importancia(modelo, X, y, columnas):  # Importancia por permutación en una muestra estratificada de prueba
    Xm, _, ym, _ = train_test_split(X, y, train_size=TAM_MUESTRA_IMPORTANCIA, stratify=y, random_state=cf.SEMILLA)  # Muestra
    imp = permutation_importance(modelo, Xm, ym, scoring="roc_auc", n_repeats=3, random_state=cf.SEMILLA, n_jobs=1)  # Caída del AUC
    return dict(sorted(zip(columnas, imp.importances_mean.tolist()), key=lambda kv: kv[1], reverse=True))  # Ordenadas


def aporte_clima(tr, te):  # ¿Cuánto aporta cada bloque de variables? (mismo algoritmo, distintos conjuntos)
    bloques = {"Solo clima y contexto": cf.CLIMA + cf.CONTEXTO, "Solo historia de casos": cf.CASOS + cf.CONTEXTO,  # Conjuntos de variables
               "Casos + clima (modelos recomendados)": cf.FEATURES, "Casos + clima + espacial (HetDS)": cf.FEATURES_ORIGINAL}
    salida = {}  # AUC por bloque
    for nombre, cols in bloques.items():  # Recorre los bloques
        m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, random_state=cf.SEMILLA).fit(tr[cols], tr[cf.TARGET])  # Mismo algoritmo
        p = m.predict_proba(te[cols])[:, 1]  # Prueba
        salida[nombre] = {"auc_roc": float(roc_auc_score(te[cf.TARGET], p)), "auc_pr": float(average_precision_score(te[cf.TARGET], p)), "n_vars": len(cols)}  # Métricas
    return salida  # Diccionario


def demo_fuga(mod_df):  # Dos fugas típicas en series de tiempo: una variable del futuro y una partición aleatoria
    tr, _, te = dt.division_temporal(mod_df)  # Partición temporal correcta
    hgb = lambda: HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=cf.SEMILLA)  # Modelo auxiliar
    p_ok = hgb().fit(tr[cf.FEATURES], tr[cf.TARGET]).predict_proba(te[cf.FEATURES])[:, 1]  # Flujo correcto
    con = cf.FEATURES + ["casos_obj"]  # ERROR deliberado: incluye los casos de la semana objetivo (futuro)
    p_fut = hgb().fit(tr[con], tr[cf.TARGET]).predict_proba(te[con])[:, 1]  # Modelo con fuga temporal
    a, b = train_test_split(mod_df[mod_df["ANO"] <= 2024], test_size=0.2, random_state=cf.SEMILLA)  # ERROR deliberado: semanas mezcladas al azar
    p_al = hgb().fit(a[cf.FEATURES], a[cf.TARGET]).predict_proba(b[cf.FEATURES])[:, 1]  # Partición aleatoria
    return {"correcto": {"auc_roc": float(roc_auc_score(te[cf.TARGET], p_ok)), "auc_pr": float(average_precision_score(te[cf.TARGET], p_ok))},  # Métricas
            "futuro": {"auc_roc": float(roc_auc_score(te[cf.TARGET], p_fut)), "auc_pr": float(average_precision_score(te[cf.TARGET], p_fut))},
            "aleatoria": {"auc_roc": float(roc_auc_score(b[cf.TARGET], p_al)), "auc_pr": float(average_precision_score(b[cf.TARGET], p_al))},
            "azar": float(te[cf.TARGET].mean())}


def entrenar_todo():  # Flujo completo
    panel = dt.panel_modelado()  # Panel procesado
    tr, va, te = dt.division_temporal(panel)  # Periodos
    resultados, modelos = {}, {}  # Resultados y modelos ajustados
    for nombre, (modelo, malla, usa_muestra) in mod.catalogo_modelos().items():  # Cuatro modelos recomendados
        t0 = time.time()  # Inicio
        datos = tr  # Datos de entrenamiento
        if usa_muestra:  # KNN: muestra estratificada por año y clase
            datos, _ = train_test_split(tr, train_size=TAM_MUESTRA_KNN, stratify=tr["ANO"].astype(str) + tr[cf.TARGET].astype(str), random_state=cf.SEMILLA)  # Muestra
        grid = mod.train_pipeline(datos[cf.FEATURES], datos[cf.TARGET], datos["ANO"], modelo, malla)  # GridSearchCV temporal
        mejor = grid.best_estimator_  # Pipeline ganador
        p_tr = mejor.predict_proba(datos[cf.FEATURES])[:, 1]  # Entrenamiento (para medir sobreajuste)
        p_va = mejor.predict_proba(va[cf.FEATURES])[:, 1]  # Validación
        p_te = mejor.predict_proba(te[cf.FEATURES])[:, 1]  # Prueba
        extra = {"params": {k.replace("clf__", ""): str(v) for k, v in grid.best_params_.items()},  # Mejores hiperparámetros
                 "cv_auc_pr": float(grid.cv_results_["mean_test_auc_pr"][grid.best_index_]), "cv_auc_roc": float(grid.cv_results_["mean_test_auc_roc"][grid.best_index_]),
                 "cv_tabla": resumen_cv(grid), "n_train": int(len(datos))}
        resultados[nombre] = evaluar(nombre, p_tr, datos[cf.TARGET], p_va, va[cf.TARGET], p_te, te, time.time() - t0, extra)  # Métricas
        resultados[nombre]["importancia"] = importancia(mejor, te[cf.FEATURES], te[cf.TARGET], cf.FEATURES)  # Importancia
        modelos[nombre] = mejor  # Guarda el modelo

    # --- Modelo original: HetDS en línea ---
    t0 = time.time()  # Inicio
    h = HetDSEnLinea().fit(tr, tr[cf.TARGET])  # Entrena los expertos con 2015–2021
    P_va = h.probas_expertos(va)  # Expertos en validación
    seleccion = []  # AUC-PR de validación de cada configuración
    for eta, v in MALLA_HETDS:  # Malla de hiperparámetros
        h.set_params(eta=eta, ventana=v)  # Configura
        p, _ = h.predecir_en_linea(va, va[cf.TARGET], va["time_idx"], P_va)  # Recorre 2022 en línea
        seleccion.append({"eta": eta, "ventana": v, "val_auc_pr": float(average_precision_score(va[cf.TARGET], p)), "val_auc_roc": float(roc_auc_score(va[cf.TARGET], p))})  # Resultado
    mejor_cfg = max(seleccion, key=lambda s: s["val_auc_pr"])  # Mejor configuración en validación
    h.set_params(eta=mejor_cfg["eta"], ventana=mejor_cfg["ventana"])  # La aplica
    flujo = pd.concat([va, te])  # Flujo 2022–2024: 2022 calienta los pesos antes de la prueba
    P_fl = h.probas_expertos(flujo)  # Expertos en el flujo
    p_fl, pesos = h.predecir_en_linea(flujo, flujo[cf.TARGET], flujo["time_idx"], P_fl)  # Predicción en línea
    es_va = (flujo["ANO"] == cf.ANIO_VALIDACION).to_numpy()  # Filas de validación
    muestra_tr = tr.sample(min(30_000, len(tr)), random_state=cf.SEMILLA)  # Muestra de entrenamiento (para medir el sobreajuste)
    p_tr_h = h.predict_proba(muestra_tr)[:, 1]  # Probabilidades estáticas en entrenamiento
    y_tr_h = muestra_tr[cf.TARGET]  # Etiquetas de esa muestra
    P_te = P_fl[~es_va]  # Expertos en prueba
    extra = {"params": {"eta": str(mejor_cfg["eta"]), "ventana": str(mejor_cfg["ventana"]), "retraso": str(cf.HORIZONTE)},  # Hiperparámetros
             "seleccion": seleccion, "cv_auc_pr": mejor_cfg["val_auc_pr"], "cv_auc_roc": mejor_cfg["val_auc_roc"], "n_train": int(len(tr)),
             "expertos_auc": {k: float(roc_auc_score(te[cf.TARGET], P_te[:, j])) for j, k in enumerate(h.expertos_)},  # AUC de cada experto
             "estatico_auc": float(roc_auc_score(te[cf.TARGET], 1 / (1 + np.exp(-np.log(np.clip(P_te, 1e-5, 1 - 1e-5) / (1 - np.clip(P_te, 1e-5, 1 - 1e-5))).mean(axis=1))))),  # Sin pesos dinámicos
             "pesos": {"semana": [str(d.date()) for d in flujo.drop_duplicates("time_idx").set_index("time_idx").loc[pesos.index, "week_start"]],  # Historial de pesos
                       **{k: pesos[k].round(4).tolist() for k in pesos.columns}}}
    resultados["HetDS"] = evaluar("HetDS", p_tr_h, y_tr_h, p_fl[es_va], va[cf.TARGET], p_fl[~es_va], te, time.time() - t0, extra)  # Métricas
    resultados["HetDS"]["importancia"] = importancia(h, te[h.columnas_], te[cf.TARGET], h.columnas_)  # Importancia (versión estática: pesos iguales)

    delong = {}  # Comparación del modelo original contra cada recomendado
    for n in [k for k in resultados if k != "HetDS"]:  # Recomendados
        d, ic, p = mod.prueba_delong(te[cf.TARGET].to_numpy(), resultados["HetDS"]["p_test"], resultados[n]["p_test"])  # DeLong pareada
        delong[n] = {"dif": d, "ic": ic, "p": p}  # Guarda
    return {"resultados": resultados, "delong": delong, "aporte_clima": aporte_clima(tr, te), "fuga": demo_fuga(panel),  # Artefactos
            "y_test": te[cf.TARGET].to_numpy().astype(np.int8), "mun_test": te["COD_MUN_N"].to_numpy(), "anio_test": te["ANO"].to_numpy(),
            "n": {"train": len(tr), "val": len(va), "test": len(te), "knn": TAM_MUESTRA_KNN}}, h, modelos


if __name__ == "__main__":  # Solo al ejecutar el archivo directamente
    art, hetds, modelos = entrenar_todo()  # Entrena todo
    joblib.dump(art, cf.RUTA_ARTEFACTOS, compress=3)  # Métricas y predicciones para el dashboard
    joblib.dump(hetds, cf.RAIZ / "dashboard" / "hetds.joblib", compress=3)  # Modelo original (simulador de escenarios climáticos)
    for k, v in art["delong"].items():  # Resumen
        print(f"HetDS vs {k:<20} dif AUC={v['dif']:+.4f}  IC95%=({v['ic'][0]:+.4f}, {v['ic'][1]:+.4f})  p={v['p']:.3g}")  # DeLong
    print("Artefactos guardados en", cf.RUTA_ARTEFACTOS)  # Confirmación
