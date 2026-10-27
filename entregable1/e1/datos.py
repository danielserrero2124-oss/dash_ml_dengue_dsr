"""Carga de datos del Primer Entregable: panel municipio × semana, coordenadas DANE, regiones y partición.

La partición se define ANTES de cualquier decisión basada en los datos. Como el objetivo de la fila t es el brote de
la semana t + 4, una fila pertenece a un periodo solo si su semana t Y su semana objetivo t + 4 caen en ese periodo.
Las filas que cruzan una frontera se descartan: es la «purga» con un intervalo de separación (gap) de 4 semanas,
que impide que una etiqueta del periodo siguiente se use al entrenar o al elegir el umbral.
"""

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas

from src import config as cf  # Configuración del proyecto (horizonte, periodos y variables)
from src.datos import construir_panel  # Canal endémico, objetivo y variables (mismo código del proyecto final)

RUTA_DIVIPOLA = cf.RAIZ / "data" / "divipola_municipios.csv"  # Centroides DANE (DIVIPOLA, datos.gov.co, CC BY-SA 4.0)

REGIONES = {  # Región natural de cada departamento (código DANE de dos dígitos)
    8: "Caribe", 13: "Caribe", 20: "Caribe", 23: "Caribe", 44: "Caribe", 47: "Caribe", 70: "Caribe", 88: "Caribe",  # Costa norte e islas
    5: "Andina", 11: "Andina", 15: "Andina", 17: "Andina", 25: "Andina", 41: "Andina", 54: "Andina",  # Cordilleras (1/2)
    63: "Andina", 66: "Andina", 68: "Andina", 73: "Andina",  # Cordilleras (2/2)
    19: "Pacífica", 27: "Pacífica", 52: "Pacífica", 76: "Pacífica",  # Costa del Pacífico
    50: "Orinoquía", 81: "Orinoquía", 85: "Orinoquía", 99: "Orinoquía",  # Llanos orientales
    18: "Amazonía", 86: "Amazonía", 91: "Amazonía", 94: "Amazonía", 95: "Amazonía", 97: "Amazonía",  # Selva amazónica
}

PERIODOS = {  # Periodo de cada año (según el año de la semana t y de la semana objetivo t + 4)
    **{a: "historia" for a in range(2010, 2015)},  # 2010–2014: solo construyen el canal endémico
    **{a: "entrenamiento" for a in range(2015, cf.ANIO_FIN_ENTRENAMIENTO + 1)},  # 2015–2021
    cf.ANIO_VALIDACION: "validación",  # 2022: umbral de decisión
    **{a: "prueba" for a in range(cf.ANIO_INICIO_PRUEBA, 2025)},  # 2023–2024: se abre una sola vez, al final
}

COLUMNAS_ORIGINALES = [  # Las 21 columnas del archivo original
    "COD_MUN_N", "ANO", "SEMANA", "casos_totales", "casos_0_4", "casos_5_14", "casos_15_64", "casos_65_plus",
    "casos_m", "casos_f", "week_start", "MES", "temp_mean", "prec_total", "ndvi_mean", "oni_anom", "oni_total",
    "Flujo_in", "temp_mean_missing", "time_idx", "poblacion",
]


def municipios():  # Tabla de los 376 municipios con nombre, departamento, región y coordenadas
    d = pd.read_csv(RUTA_DIVIPOLA, dtype=str)  # Lee como texto (los códigos tienen ceros a la izquierda)
    d["COD_MUN_N"] = d["cod_mpio"].astype(int)  # Código numérico, igual al del dataset
    for c in ("latitud", "longitud"):  # Coordenadas con coma decimal
        d[c] = d[c].str.replace(",", ".").astype(float)  # Convierte a número
    d["depto"] = d["COD_MUN_N"] // 1000  # Departamento: dos primeros dígitos del código
    d["region"] = d["depto"].map(REGIONES)  # Región natural
    d["municipio"] = d["nom_mpio"].str.title()  # Nombre legible
    d["departamento"] = d["dpto"].str.title()  # Departamento legible
    presentes = pd.read_csv(cf.RUTA_DATOS, usecols=["COD_MUN_N"])["COD_MUN_N"].unique()  # Municipios del panel
    cols = ["COD_MUN_N", "municipio", "departamento", "depto", "region", "latitud", "longitud"]  # Columnas útiles
    return d.loc[d["COD_MUN_N"].isin(presentes), cols].sort_values("COD_MUN_N").reset_index(drop=True)  # 376 filas


def panel():  # Panel completo con objetivo, variables, coordenadas, región y periodo de cada fila
    df = construir_panel()  # 294 032 filas con canal endémico, brote, objetivo y variables (src/datos.py)
    df["anio_obj"] = df.groupby("COD_MUN_N")["ANO"].shift(-cf.HORIZONTE)  # Año de la semana objetivo t + 4
    df["brote_estacional"] = df.groupby("COD_MUN_N")["brote"].shift(52 - cf.HORIZONTE)  # Brote de t + 4 − 52 (conocido en t)
    df["periodo_t"] = df["ANO"].map(PERIODOS)  # Periodo de la semana t
    df["periodo_obj"] = df["anio_obj"].map(PERIODOS)  # Periodo de la semana objetivo
    df["periodo"] = np.where(df["periodo_t"] == df["periodo_obj"], df["periodo_t"], "purgada")  # Frontera → purgada
    m = municipios()  # Coordenadas y región
    return df.merge(m.drop(columns="depto"), on="COD_MUN_N", how="left")  # Une por municipio


def modelado(df=None):  # Filas aptas para modelar: con objetivo y todas las variables, sin filas purgadas
    df = panel() if df is None else df  # Construye el panel si no se pasa
    ok = df["periodo"].isin(["entrenamiento", "validación", "prueba"]) & df[[cf.TARGET] + cf.FEATURES].notna().all(axis=1)  # Filas completas
    mod = df.loc[ok].reset_index(drop=True)  # Filtra
    mod[cf.TARGET] = mod[cf.TARGET].astype(int)  # Objetivo entero 0/1
    return mod  # Panel de modelado


def particion(mod):  # Entrenamiento, validación y prueba ya purgados
    return (mod[mod["periodo"] == "entrenamiento"], mod[mod["periodo"] == "validación"],  # 2015–2021 y 2022
            mod[mod["periodo"] == "prueba"])  # 2023–2024


def desarrollo(df):  # Datos permitidos para el EDA: todo lo anterior a la prueba (la prueba queda sellada)
    return df[df["ANO"] <= cf.ANIO_VALIDACION]  # 2010–2022 (la validación solo se usa para revisar deriva)


def entrenamiento_eda(df):  # Datos con los que se toman las decisiones del EDA: semanas t de 2010–2021
    return df[df["ANO"] <= cf.ANIO_FIN_ENTRENAMIENTO]  # Historia + entrenamiento
