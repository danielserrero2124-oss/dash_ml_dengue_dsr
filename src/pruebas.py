"""Métricas de la literatura, pruebas de significancia, ablación del modelo original y tiempos de cómputo.

Pruebas (HetDS frente a cada modelo recomendado, sobre la prueba 2023–2024):
  * DeLong para AUC-ROC (DeLong et al., 1988).
  * Bootstrap por conglomerados (municipios) para AUC-ROC y AUC-PR: remuestrea municipios completos y respeta la
    autocorrelación de sus semanas (Field y Welsh, 2007), más estricto que DeLong, que supone filas independientes.
  * McNemar para los errores con el umbral elegido (Dietterich, 1998).
  * Friedman + Nemenyi con las 104 semanas de prueba como bloques (Demšar, 2006).
  * Corrección de Holm por comparaciones múltiples.
Ablación: qué parte de la ganancia aporta el componente espacial y qué parte la ponderación dinámica.

Ejecución (desde la raíz del proyecto, después de python -m src.entrenamiento):  python -m src.pruebas
"""

import ast  # Convierte hiperparámetros guardados como texto
import os  # Variables de entorno
import time  # Tiempos

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # Evita un aviso de joblib en Windows

import joblib  # Carga y guardado
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from scipy import stats  # Distribuciones y Friedman
from sklearn.base import clone  # Copias de estimadores
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss, log_loss,  # Métricas
                             matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split  # Muestras
from sklearn.pipeline import Pipeline  # Pipelines

from src import config as cf  # Configuración
from src import datos as dt  # Datos
from src import modelado as mod  # Modelos y pruebas
from src.hetds import HetDSEnLinea  # Modelo original

TOP_K = 0.05  # Fracción de municipio-semanas de mayor riesgo para precision@k (capacidad de vigilancia)
Q_NEMENYI = {5: 2.728}  # Valor crítico q_α de Nemenyi (α = 0,05, k = 5 modelos; Demšar, 2006)


def valor(texto):  # "0.01" → 0.01, "None" → None, "balanced" → "balanced"
    try:  # Literales de Python
        return ast.literal_eval(texto)  # Evaluación segura
    except (ValueError, SyntaxError):  # Cadenas
        return texto  # Sin cambios


def construir(nombre, params, columnas=None):  # Pipeline de un modelo recomendado con sus mejores hiperparámetros
    estimador, _, usa_muestra = mod.catalogo_modelos()[nombre]  # Estimador base
    pipe = Pipeline([("prep", mod.make_preprocessor()), ("clf", clone(estimador))])  # Mismo Pipeline del entrenamiento
    pipe.set_params(**{f"clf__{k}": valor(v) for k, v in params.items()})  # Mejores hiperparámetros
    return pipe, usa_muestra  # Pipeline y bandera de muestreo


def metricas_literatura(y, p, umbral):  # Métricas recomendadas para comparar clasificadores con clases desbalanceadas
    pred = (p >= umbral).astype(int)  # Predicción con el umbral de validación
    k = int(np.ceil(TOP_K * len(y)))  # Alertas que caben en la capacidad de vigilancia
    top = np.argsort(-p)[:k]  # Municipio-semanas de mayor riesgo
    return {"auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p)), "mcc": float(matthews_corrcoef(y, pred)),
            "bal_acc": float(balanced_accuracy_score(y, pred)), "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, np.clip(p, 1e-7, 1 - 1e-7))), "prec_k": float(y[top].mean()), "rec_k": float(y[top].sum() / y.sum())}


def ic_conglomerados(y, p, grupos, metrica, B=300):  # IC del 95 % de una métrica remuestreando municipios completos
    rng = np.random.default_rng(cf.SEMILLA)  # Generador reproducible
    unicos = np.unique(grupos)  # Municipios
    filas = {g: np.flatnonzero(grupos == g) for g in unicos}  # Filas por municipio
    vals = [metrica(y[i], p[i]) for i in (np.concatenate([filas[g] for g in rng.choice(unicos, len(unicos))]) for _ in range(B))]  # Réplicas
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))  # Percentiles 2,5 y 97,5


def mcnemar(y, a, b):  # Prueba de McNemar con corrección de continuidad
    ok_a, ok_b = a == y, b == y  # Aciertos
    n10, n01 = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))  # Discordancias
    chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01) if n10 + n01 > 0 else 0.0  # Estadístico
    return {"b": n10, "c": n01, "chi2": float(chi2), "p": float(stats.chi2.sf(chi2, 1))}  # Resultado


def holm(p):  # Corrección de Holm-Bonferroni
    orden, m, acum, ajust = np.argsort(p), len(p), 0.0, np.empty(len(p))  # Preparación
    for r, i in enumerate(orden):  # De menor a mayor p
        acum = max(acum, min(1.0, (m - r) * p[i]))  # Ajuste monótono
        ajust[i] = acum  # Guarda
    return ajust.tolist()  # p ajustados


def friedman_semanal(y, preds, semanas):  # Friedman + Nemenyi con cada semana de prueba como bloque
    nombres = list(preds)  # Modelos
    filas = []  # AUC semanal de cada modelo
    for s in np.unique(semanas):  # Semanas
        m = semanas == s  # Filas de la semana
        if y[m].sum() >= 10 and (1 - y[m]).sum() >= 10:  # Suficientes brotes y no brotes para un AUC estable
            filas.append([roc_auc_score(y[m], preds[n][m]) for n in nombres])  # AUC de cada modelo en la semana
    M = np.array(filas)  # Semanas × modelos
    rangos = np.array([stats.rankdata(-f) for f in M])  # 1 = mejor en la semana
    chi2, p = stats.friedmanchisquare(*M.T)  # Prueba de Friedman
    k, N = len(nombres), M.shape[0]  # Modelos y bloques
    return {"rangos": dict(zip(nombres, rangos.mean(axis=0).tolist())), "chi2": float(chi2), "p": float(p),  # Resultado
            "cd": float(Q_NEMENYI[k] * np.sqrt(k * (k + 1) / (6 * N))), "n_semanas": int(N),
            "gana_hetds": float((M[:, nombres.index("HetDS")] >= M.max(axis=1) - 1e-12).mean())}  # Fracción de semanas en que HetDS es el mejor


def main():  # Calcula todo y guarda
    art = joblib.load(cf.RUTA_ARTEFACTOS)  # Artefactos del entrenamiento
    res = art["resultados"]  # Resultados por modelo
    nombres = list(res)  # Modelos (los 4 recomendados + HetDS)
    panel = dt.panel_modelado()  # Datos
    tr, va, te = dt.division_temporal(panel)  # Periodos
    y = te[cf.TARGET].to_numpy()  # Objetivo de prueba
    mun = te["COD_MUN_N"].to_numpy()  # Municipio de cada fila (conglomerado)
    P = {n: res[n]["p_test"].astype(float) for n in nombres}  # Probabilidades de prueba

    metricas = {n: metricas_literatura(y, P[n], res[n]["umbral"]) for n in nombres}  # Métricas de la literatura
    for n in nombres:  # Intervalos de confianza
        metricas[n]["ic_auc_roc"] = res[n]["ic_auc"]  # DeLong
        metricas[n]["ic_auc_pr"] = ic_conglomerados(y, P[n], mun, average_precision_score)  # IC del AUC-PR por bootstrap de municipios
    print("Métricas listas", flush=True)  # Progreso

    pruebas = {}  # HetDS frente a cada recomendado
    pred_h = (P["HetDS"] >= res["HetDS"]["umbral"]).astype(int)  # Predicción de HetDS
    for n in [m for m in nombres if m != "HetDS"]:  # Recomendados
        d, ic, p = mod.prueba_delong(y, P["HetDS"], P[n])  # DeLong
        se = (ic[1] - ic[0]) / (2 * 1.96)  # Error estándar
        bc_roc = mod.bootstrap_conglomerados(y, P["HetDS"], P[n], mun, roc_auc_score, B=1000)  # Bootstrap por municipio (AUC-ROC)
        bc_pr = mod.bootstrap_conglomerados(y, P["HetDS"], P[n], mun, average_precision_score, B=1000)  # Bootstrap por municipio (AUC-PR)
        anual = {}  # DeLong por año
        for a in (2023, 2024):  # Años de prueba
            m = te["ANO"].to_numpy() == a  # Filas del año
            da, _, pa = mod.prueba_delong(y[m], P["HetDS"][m], P[n][m])  # DeLong del año
            anual[a] = {"dif": da, "p": pa}  # Guarda
        pruebas[n] = {"delong": {"dif": d, "ic": ic, "z": d / se, "p": p},  # Resultados
                      "cluster_roc": {"dif": bc_roc[0], "ic": bc_roc[1], "p": bc_roc[2]},
                      "cluster_pr": {"dif": bc_pr[0], "ic": bc_pr[1], "p": bc_pr[2]},
                      "mcnemar": mcnemar(y, pred_h, (P[n] >= res[n]["umbral"]).astype(int)), "anual": anual}
        print(f"HetDS vs {n}: cluster AUC p={bc_roc[2]:.3g}", flush=True)  # Progreso
    for clave in ["delong", "cluster_roc", "cluster_pr", "mcnemar"]:  # Holm por familia
        for n, pa in zip(pruebas, holm([pruebas[n][clave]["p"] for n in pruebas])):  # Ajuste
            pruebas[n][clave]["p_holm"] = pa  # p ajustado

    friedman = {"auc_roc": friedman_semanal(y, P, te["time_idx"].to_numpy())}  # Friedman + Nemenyi semanal

    # --- Ablación: ¿de dónde viene la ganancia? ---
    gbc_pipe, _ = construir("GradientBoosting", res["GradientBoosting"]["params"])  # GradientBoosting con sus mejores hiperparámetros
    t0 = time.time()  # Mide el ajuste
    p_gbc_esp = gbc_pipe.fit(tr[cf.FEATURES_ORIGINAL], tr[cf.TARGET]).predict_proba(te[cf.FEATURES_ORIGINAL])[:, 1]  # GB con las variables espaciales
    t_gbc = time.time() - t0  # Tiempo de un ajuste de GB
    h_sin = HetDSEnLinea(eta=float(res["HetDS"]["params"]["eta"]), ventana=int(res["HetDS"]["params"]["ventana"]), columnas=cf.FEATURES).fit(tr, tr[cf.TARGET])  # HetDS sin espaciales
    flujo = pd.concat([va, te])  # Flujo 2022–2024
    p_sin, _ = h_sin.predecir_en_linea(flujo, flujo[cf.TARGET], flujo["time_idx"])  # En línea
    p_sin = p_sin[(flujo["ANO"] >= cf.ANIO_INICIO_PRUEBA).to_numpy()]  # Solo prueba
    abl = {"GradientBoosting (variables base)": P["GradientBoosting"], "GradientBoosting + espaciales": p_gbc_esp,  # Variantes
           "HetDS sin espaciales (en línea)": p_sin, "HetDS completo (en línea)": P["HetDS"]}
    ablacion = {}  # Resultados de la ablación
    for k, p in abl.items():  # Cada variante frente a GradientBoosting base
        dif, ic, pv = mod.bootstrap_conglomerados(y, p, P["GradientBoosting"], mun, roc_auc_score, B=500) if k != "GradientBoosting (variables base)" else (0.0, (0.0, 0.0), 1.0)  # Bootstrap por municipio
        ablacion[k] = {"auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p)), "dif_gb": dif, "ic": ic, "p": pv}  # Guarda
    ablacion["HetDS estático (pesos iguales)"] = {"auc_roc": res["HetDS"]["estatico_auc"]}  # Sin ponderación dinámica (del entrenamiento)
    print("Ablación lista", flush=True)  # Progreso

    # --- Tiempos de cómputo: un ajuste con los mejores hiperparámetros y predicción por 1 000 filas ---
    tiempos = {}  # Por modelo
    for n in [m for m in nombres if m != "HetDS"]:  # Recomendados
        pipe, usa_muestra = construir(n, res[n]["params"])  # Pipeline
        datos = train_test_split(tr, train_size=min(30_000, len(tr) // 2), stratify=tr["ANO"].astype(str) + tr[cf.TARGET].astype(str), random_state=cf.SEMILLA)[0] if usa_muestra else tr  # KNN usa muestra
        if n == "GradientBoosting":  # GB ya se midió en la ablación (mismos hiperparámetros, 3 variables más)
            t_fit = t_gbc  # Reutiliza
            pipe.fit(datos[cf.FEATURES], datos[cf.TARGET])  # Ajuste para medir la predicción
        else:  # El resto se mide aquí
            t0 = time.time()  # Inicio
            pipe.fit(datos[cf.FEATURES], datos[cf.TARGET])  # Ajuste
            t_fit = time.time() - t0  # Tiempo
        t0 = time.time()  # Inicio de la predicción
        pipe.predict_proba(te[cf.FEATURES])  # Predice la prueba completa
        tiempos[n] = {"grid_s": float(res[n]["segundos"]), "fit_s": float(t_fit), "n_fit": int(len(datos)), "pred_ms_1000": float(1e6 * (time.time() - t0) / len(te))}  # Guarda
    h = joblib.load(cf.RAIZ / "dashboard" / "hetds.joblib")  # Modelo original entrenado
    t0 = time.time()  # Inicio
    HetDSEnLinea(eta=h.eta, ventana=h.ventana).fit(tr, tr[cf.TARGET])  # Un ajuste completo de los cinco expertos
    t_fit_h = time.time() - t0  # Tiempo
    t0 = time.time()  # Inicio
    h.predecir_en_linea(te, te[cf.TARGET], te["time_idx"])  # Predicción en línea de toda la prueba
    tiempos["HetDS"] = {"grid_s": float(res["HetDS"]["segundos"]), "fit_s": float(t_fit_h), "n_fit": int(len(tr)), "pred_ms_1000": float(1e6 * (time.time() - t0) / len(te))}  # Guarda

    joblib.dump({"metricas": metricas, "pruebas": pruebas, "friedman": friedman, "ablacion": ablacion, "tiempos": tiempos, "top_k": TOP_K}, cf.RUTA_PRUEBAS)  # Guarda
    for n, d in pruebas.items():  # Resumen
        print(f"HetDS vs {n:<20} dif AUC={d['delong']['dif']:+.4f}  DeLong p={d['delong']['p_holm']:.2g}  cluster p={d['cluster_roc']['p_holm']:.3g}  dif AP={d['cluster_pr']['dif']:+.4f} p={d['cluster_pr']['p_holm']:.3g}")
    print("Friedman:", friedman["auc_roc"]["p"], friedman["auc_roc"]["rangos"])  # Friedman
    for k, v in ablacion.items():  # Ablación
        print(k, {a: (round(b, 4) if isinstance(b, float) else b) for a, b in v.items()})  # Resultado
    print("Guardado en", cf.RUTA_PRUEBAS)  # Confirmación


if __name__ == "__main__":  # Solo al ejecutar el archivo directamente
    main()  # Corre todo
