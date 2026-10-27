"""Datos del Tercer Entregable: panel, objetivos de clasificación y regresión, y particiones de la validación anidada.

Reutiliza el panel y la partición purgada del Primer Entregable (paquete e1). Una fila pertenece a un periodo solo si su
semana t y su semana objetivo t + 4 caen dentro de él, así que ninguna etiqueta cruza de un pliegue a otro.
"""

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas

from e1 import datos as d1  # Panel y partición del Primer Entregable
from e3 import config as c3  # Configuración
from src import config as cf  # Variables y objetivo

X_COLS = cf.FEATURES  # Las 34 predictoras (todas usan semanas ≤ t)
Y_CLF = cf.TARGET  # Clasificación: brote en t + 4
Y_REG = "y_reg"  # Regresión: log(1 + casos) en t + 4


def cargar():
    """Carga el panel de modelado con los dos objetivos.

    Returns
    -------
    tr, va, te : pandas.DataFrame
        Entrenamiento 2015–2021, validación 2022 y prueba 2023–2024 (purgados), con las columnas ``y`` y ``y_reg``.
    """
    mod = d1.modelado()  # Filas con objetivo y variables completas
    mod[Y_REG] = np.log1p(mod["casos_obj"])  # Objetivo continuo: casos de la semana objetivo en escala log
    return d1.particion(mod)  # Tres periodos purgados


def submuestra(df, n, semilla=c3.SEMILLA, estratos=("ANO", Y_CLF)):
    """Submuestra estratificada por año y clase (conserva la estructura temporal y la prevalencia de cada año).

    Parameters
    ----------
    df : pandas.DataFrame
        Datos a submuestrear.
    n : int
        Tamaño deseado (si df es más pequeño, se devuelve completo).
    semilla : int
        Semilla del muestreo.
    estratos : tuple of str
        Columnas que definen los estratos.

    Returns
    -------
    pandas.DataFrame
        Submuestra ordenada por el índice original.
    """
    if len(df) <= n:  # Ya es pequeño
        return df  # Sin cambios
    frac = n / len(df)  # Fracción a conservar
    return df.groupby(list(estratos), group_keys=False).apply(lambda g: g.sample(frac=frac, random_state=semilla)).sort_index()  # Muestra por estrato


def pliegue_externo(tr, anio):
    """Particiona el entrenamiento 2015–2021 para un pliegue externo de la validación anidada.

    Parameters
    ----------
    tr : pandas.DataFrame
        Entrenamiento completo (2015–2021).
    anio : int
        Año evaluado por el pliegue externo.

    Returns
    -------
    entrena, evalua : pandas.DataFrame
        Filas cuyo objetivo es anterior a ``anio`` (purga) y filas con semana t y objetivo en ``anio``.
    """
    entrena = tr[(tr["ANO"] < anio) & (tr["anio_obj"] < anio)]  # Etiquetas anteriores al año evaluado
    evalua = tr[(tr["ANO"] == anio) & (tr["anio_obj"] == anio)]  # El año evaluado completo
    return entrena, evalua  # Pliegue externo


def pliegues_internos(entrena, k=c3.PLIEGUES_INTERNOS):
    """Pliegues internos hacia adelante dentro del entrenamiento de un pliegue externo.

    Parameters
    ----------
    entrena : pandas.DataFrame
        Entrenamiento del pliegue externo (ya submuestreado).
    k : int
        Número de pliegues internos: los ``k`` últimos años.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        Posiciones de entrenamiento y validación de cada pliegue interno (con la misma purga por año).
    """
    anios = sorted(entrena["ANO"].unique())  # Años disponibles
    salida = []  # Pliegues
    for a in anios[-k:]:  # Los k últimos años como validación interna
        i_tr = np.flatnonzero(((entrena["ANO"] < a) & (entrena["anio_obj"] < a)).to_numpy())  # Antes de a
        i_va = np.flatnonzero(((entrena["ANO"] == a) & (entrena["anio_obj"] == a)).to_numpy())  # En a
        salida.append((i_tr, i_va))  # Pliegue
    return salida  # Lista de pliegues
