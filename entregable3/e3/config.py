"""Configuración del Tercer Entregable: una única semilla global y todos los tamaños y presupuestos en un solo lugar."""

import os  # Variables de entorno
import random  # Semilla de Python
from pathlib import Path  # Rutas

import numpy as np  # Semilla de NumPy

SEMILLA = 42  # Semilla global: se propaga a NumPy, scikit-learn, XGBoost, Optuna, DEAP e imbalanced-learn
AQUI = Path(__file__).resolve().parent.parent  # Carpeta entregable3
RESULTADOS = AQUI / "resultados"  # Tabla maestra, historiales, predicciones y análisis

# Validación anidada (bucle externo por años, con purga de 4 semanas)
ANIOS_EXTERNOS = (2019, 2020, 2021)  # Cada año es un pliegue externo: se entrena con los años anteriores y se evalúa en él
PLIEGUES_INTERNOS = 2  # Dentro de cada pliegue externo, los 2 últimos años de entrenamiento son los pliegues internos

# Tamaños (submuestras estratificadas para que 140 combinaciones × anidación quepan en el presupuesto de cómputo)
N_EXTERNO = 20_000  # Filas del entrenamiento externo (se reparten entre los pliegues internos)
N_FINAL = 40_000  # Filas para el reajuste final con todo 2015–2021 (evaluación única en la prueba)
PRESUPUESTO = 20  # Evaluaciones de hiperparámetros por optimizador (igual para los cuatro: comparación justa)
TRABAJOS = 10  # Procesos en paralelo (de 12 núcleos lógicos; se dejan 2 libres)


def fijar_semilla(semilla=SEMILLA):  # Propaga la semilla a todas las fuentes de aleatoriedad
    """Fija la semilla de Python, NumPy y del hash de Python.

    Parameters
    ----------
    semilla : int
        Valor de la semilla global.
    """
    random.seed(semilla)  # Python (usado por DEAP)
    np.random.seed(semilla)  # NumPy (usado por scikit-learn cuando random_state=None)
    os.environ["PYTHONHASHSEED"] = str(semilla)  # Hash reproducible
