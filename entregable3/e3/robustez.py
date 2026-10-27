"""Sensibilidad a la semilla: el mejor ajuste final de cada modelo de clasificación se repite con 5 semillas distintas
(submuestra, remuestreo, inicialización de los árboles y del núcleo aproximado) y se mide la variación en la prueba."""

import pickle  # Lectura de los reajustes finales

import joblib  # Guardado
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from joblib import Parallel, delayed  # Paralelización
from sklearn.metrics import average_precision_score, roc_auc_score  # Métricas

from e3 import config as c3, datos as dd, experimento as ex, modelos as mo  # Módulos

SEMILLAS = (1, 2, 3, 4, 5)  # Semillas de la prueba de sensibilidad


def una(tarea, modelo, balanceo, metodo, semilla):  # Reajuste final con otra semilla
    with open(ex.FINALES / ex._nombre(tarea, modelo, balanceo, metodo), "rb") as f:  # Hiperparámetros elegidos
        params = pickle.load(f)["params"]  # Mismos hiperparámetros
    tr, _, te = ex.datos()  # Periodos
    ent = dd.submuestra(tr, c3.N_FINAL, semilla)  # Submuestra con la nueva semilla
    m = mo.construir(tarea, modelo, params, balanceo, semilla).fit(ent[dd.X_COLS].to_numpy(), ent[dd.Y_CLF].to_numpy())  # Ajuste
    s = mo.puntaje(m, te[dd.X_COLS].to_numpy())  # Puntaje en la prueba
    return {"modelo": modelo, "balanceo": balanceo, "optimizador": metodo, "semilla": semilla,
            "AUC-ROC": roc_auc_score(te[dd.Y_CLF], s), "AUC-PR": average_precision_score(te[dd.Y_CLF], s)}  # Métricas


def main(mejores):  # mejores: lista de (modelo, balanceo, optimizador), uno por modelo
    filas = Parallel(n_jobs=c3.TRABAJOS, verbose=5)(delayed(una)("clf", m, b, o, s) for m, b, o in mejores for s in SEMILLAS)  # En paralelo
    t = pd.DataFrame(filas)  # Tabla
    joblib.dump(t, c3.RESULTADOS / "robustez_semillas.joblib")  # Guarda
    return t  # Tabla
