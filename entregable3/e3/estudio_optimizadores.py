"""Estudio de eficiencia de los optimizadores: curvas de desempeño en cualquier momento, estabilidad ante semillas y
multi-fidelidad (Successive Halving), con un presupuesto mayor que el de la tabla maestra.

Caso de estudio: XGBoost sin balanceo (el espacio de búsqueda más grande: 5 hiperparámetros) en el pliegue externo 2021.
"""

import joblib  # Guardado
from joblib import Parallel, delayed  # Paralelización

from e3 import config as c3, experimento as ex  # Configuración y tarea anidada

SEMILLAS = (42, 7, 2024)  # Tres semillas para medir la estabilidad
PRESUPUESTO = 40  # Evaluaciones por corrida (el doble de la tabla maestra, para ver la convergencia)
METODOS = ("Grid Search", "Random Search", "Bayesiana (Optuna TPE)", "Genética (DEAP)", "Successive Halving")  # Métodos comparados


def correr(metodo, semilla, modelo="XGBoost"):  # Una corrida completa (bucle interno + evaluación externa)
    return ex.ejecutar_tarea("clf", modelo, "Sin balanceo", metodo, 2021, presupuesto=PRESUPUESTO, guardar=False,
                             semilla_optimizador=semilla)  # Mismos datos y modelos (semilla 42); solo cambia la semilla del optimizador


def main():  # Corre las 15 corridas en paralelo y guarda
    tareas = [(m, s) for m in METODOS for s in SEMILLAS]  # Combinaciones
    res = Parallel(n_jobs=c3.TRABAJOS, verbose=5)(delayed(correr)(m, s) for m, s in tareas)  # En paralelo
    joblib.dump(res, c3.RESULTADOS / "estudio_optimizadores.joblib")  # Guarda
