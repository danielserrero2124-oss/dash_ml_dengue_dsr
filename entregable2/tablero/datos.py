"""Carga de los artefactos precalculados (tablero/artefactos.joblib) y cálculos ligeros que dependen de ellos.

El dashboard no reentrena modelos ni recorre el panel completo: todo lo pesado se hizo en precalculo.py.
"""

from pathlib import Path  # Rutas

import joblib  # Lectura de artefactos
import numpy as np  # Cálculo numérico
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, average_precision_score  # Métricas

RUTA = Path(__file__).resolve().parent / "artefactos.joblib"  # Archivo generado por precalculo.py
A = joblib.load(RUTA)  # Diccionario con todas las secciones
M = A["municipios"]  # 376 municipios con coordenadas, departamento y región
MOD = A["modelos"]  # Resultados de los modelos base
FEATURES = list(MOD["simulador"]["filas"].columns[6:])  # 34 predictoras (después de las 6 columnas de identificación)
NOMBRES = dict(zip(M["COD_MUN_N"], M["municipio"] + " (" + M["departamento"] + ")"))  # Nombre visible de cada municipio


def rezago_espacial(valores, k=6):  # Promedio de los k vecinos más cercanos (haversine) de cada municipio
    la, lo = np.radians(M["latitud"].to_numpy())[:, None], np.radians(M["longitud"].to_numpy())[:, None]  # Radianes
    a = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2  # Haversine
    D = 2 * 6371.0088 * np.arcsin(np.sqrt(np.clip(a, 0, 1))) + np.eye(len(M)) * 1e12  # Distancias (sin el propio municipio)
    vecinos = np.argsort(D, axis=1)[:, :k]  # k vecinos
    z = np.asarray(valores, float) - np.mean(valores)  # Desviaciones
    return z[vecinos].mean(axis=1)  # Promedio de los vecinos


def metricas_umbral(nombre, umbral):  # Matriz de confusión y métricas de un modelo con un umbral elegido
    y, p = MOD["y_te"], np.asarray(MOD["P_te"][nombre], float)  # Objetivo y puntajes en la prueba
    pred = (p >= umbral).astype(int)  # Alertas
    cm = confusion_matrix(y, pred, labels=[0, 1])  # [[VN, FP], [FN, VP]]
    vn, fp, fn, vp = cm.ravel()  # Celdas
    prec = vp / (vp + fp) if vp + fp else 0.0  # Precisión
    rec = vp / (vp + fn)  # Recall
    den = np.sqrt(float((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn)))  # Denominador del MCC
    return cm, {"precisión": prec, "recall": rec, "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,  # Métricas
                "exactitud": (vp + vn) / len(y), "MCC": (vp * vn - fp * fn) / den if den else 0.0,
                "alertas por semana": (vp + fp) / MOD["te_semanas"]}


def curvas(nombres):  # Curvas ROC y PR de varios modelos (en la prueba)
    salida = {}  # Por modelo
    for n in nombres:  # Modelos
        p = np.asarray(MOD["P_te"][n], float)  # Puntajes
        fpr, tpr, _ = roc_curve(MOD["y_te"], p)  # ROC
        pr, rc, _ = precision_recall_curve(MOD["y_te"], p)  # PR
        salida[n] = {"fpr": fpr, "tpr": tpr, "prec": pr, "rec": rc, "auc": roc_auc_score(MOD["y_te"], p), "ap": average_precision_score(MOD["y_te"], p)}  # Curvas
    return salida  # Diccionario


def simular(fila, d_temp=0.0, f_lluvia=1.0, oni=None):  # Probabilidad del modelo logístico con el clima modificado
    s = MOD["simulador"]  # Parámetros del modelo
    x = fila[FEATURES].astype(float).copy()  # Variables de la semana elegida
    for v in ("temp_mean_m1", "temp_mean_m4", "temp_mean_m8", "temp_mean_m12", "temp_anom_m8"):  # Temperatura y su anomalía
        x[v] += d_temp  # Suma el cambio de temperatura
    for v in ("prec_total_m1", "prec_total_m4", "prec_total_m8", "prec_total_m12"):  # Lluvia
        x[v] *= f_lluvia  # Multiplica la lluvia
    x["prec_anom_m8"] += fila["prec_total_m8"] * (f_lluvia - 1)  # La anomalía de lluvia cambia en la misma cantidad
    if oni is not None:  # El Niño
        x["oni_anom"] = oni  # ONI actual
    z = (x.to_numpy() - s["media"]) / s["escala"]  # Estandariza con los parámetros del entrenamiento
    return float(1 / (1 + np.exp(-(z @ s["coef"] + s["intercepto"]))))  # Sigmoide del predictor lineal
