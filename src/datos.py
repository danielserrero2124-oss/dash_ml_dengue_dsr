"""Construcción del panel municipio × semana: canal endémico, definición de brote, objetivo a 4 semanas y variables.

Reglas para evitar la fuga de datos (data leakage):
  * Todas las variables de la fila t usan información de la semana t o anteriores.
  * El umbral del canal endémico de la semana objetivo (t + 4) solo usa años anteriores, así que ya se conoce en t.
  * La «normal» climática de cada municipio (para las anomalías) se calcula solo con años de entrenamiento.
"""

import numpy as np  # Cálculo numérico vectorizado
import pandas as pd  # Manejo de tablas

from src import config as cf  # Configuración del proyecto


def cargar_crudo():  # Lee el dataset original ordenado por municipio y tiempo
    df = pd.read_csv(cf.RUTA_DATOS)  # 294 032 filas: 376 municipios × 782 semanas (2010–2024)
    df["week_start"] = pd.to_datetime(df["week_start"])  # Fecha de inicio de la semana epidemiológica
    return df.sort_values(["COD_MUN_N", "time_idx"]).reset_index(drop=True)  # Orden: municipio y semana


def canal_endemico(casos, semanas_por_mun):  # Umbral superior del canal endémico (método de cuartiles, Bortman 1999)
    """Q3 de los casos en las mismas semanas (± VENTANA) de los ANIOS_HISTORIA años anteriores, por municipio."""
    n_mun = len(casos) // semanas_por_mun  # Número de municipios
    C = casos.reshape(n_mun, semanas_por_mun)  # Matriz municipios × semanas
    desfases = np.array([-52 * k + d for k in range(1, cf.ANIOS_HISTORIA + 1)  # Misma semana de cada año previo…
                         for d in range(-cf.VENTANA_SEMANAS, cf.VENTANA_SEMANAS + 1)])  # …y sus vecinas (± 2)
    umbral = np.full(C.shape, np.nan)  # Sin historia suficiente queda nulo
    inicio = 52 * cf.ANIOS_HISTORIA + cf.VENTANA_SEMANAS  # Primera semana con 5 años completos de historia
    columnas = np.arange(inicio, semanas_por_mun)  # Semanas con umbral calculable
    indices = columnas[:, None] + desfases[None, :]  # Para cada semana, las 25 semanas históricas de referencia
    umbral[:, columnas] = np.percentile(C[:, indices], 75, axis=2)  # Tercer cuartil por municipio y semana
    return umbral.ravel()  # Vector en el orden original


def media_movil(serie_agrupada, ventana):  # Media móvil dentro de cada municipio
    return serie_agrupada.transform(lambda s: s.rolling(ventana).mean())  # Usa solo la semana actual y las anteriores


def construir_panel():  # Panel completo con objetivo y variables
    df = cargar_crudo()  # Datos ordenados
    semanas = df.groupby("COD_MUN_N").size().iloc[0]  # 782 semanas por municipio (panel balanceado)
    df["umbral"] = canal_endemico(df["casos_totales"].to_numpy(dtype=float), semanas)  # Umbral del canal endémico
    df["brote"] = ((df["casos_totales"] > df["umbral"]) & (df["casos_totales"] >= cf.CASOS_MINIMOS)).astype(float)  # Brote observado
    df.loc[df["umbral"].isna(), "brote"] = np.nan  # Sin canal endémico no se define brote (2010–2014)
    g = df.groupby("COD_MUN_N")  # Agrupación por municipio
    df["y"] = g["brote"].shift(-cf.HORIZONTE)  # Objetivo: brote en t + 4
    df["umbral_obj"] = g["umbral"].shift(-cf.HORIZONTE)  # Umbral de la semana objetivo (conocido en t: usa años previos)
    df["casos_obj"] = g["casos_totales"].shift(-cf.HORIZONTE)  # Casos de la semana objetivo (solo para análisis)
    df["lc"] = np.log1p(df["casos_totales"])  # log(1 + casos): reduce la asimetría de los conteos
    g = df.groupby("COD_MUN_N")  # Reagrupa con las columnas nuevas
    for l in range(4):  # Rezagos del log de casos
        df[f"lc_l{l}"] = g["lc"].shift(l)  # Semana t − l
    for w in (4, 8, 12):  # Medias móviles del log de casos
        df[f"lc_m{w}"] = media_movil(g["lc"], w)  # Tendencia reciente
    df["brote_l0"] = df["brote"]  # Brote en la semana actual
    df["brote_m8"] = media_movil(g["brote"], 8)  # Persistencia del brote en 8 semanas
    df["ratio_umbral"] = df["lc"] - np.log1p(df["umbral_obj"])  # Casos actuales frente al umbral de la semana objetivo (escala log)
    df["l_umbral_obj"] = np.log1p(df["umbral_obj"])  # Nivel del umbral objetivo
    df["incid"] = df["casos_totales"] / df["poblacion"] * 1e5  # Incidencia semanal por 100 000 habitantes
    for v in ("temp_mean", "prec_total", "ndvi_mean"):  # Variables climáticas
        for w in (1, 4, 8, 12):  # Ventanas de acumulación (efectos rezagados del clima sobre el vector)
            df[f"{v}_m{w}"] = media_movil(g[v], w)  # Media móvil
    normal = df[df["ANO"] <= cf.ANIO_FIN_ENTRENAMIENTO].groupby(["COD_MUN_N", "MES"])[["temp_mean", "prec_total", "ndvi_mean"]].mean()  # Normal climática (solo años de entrenamiento)
    norm = df[["COD_MUN_N", "MES"]].join(normal, on=["COD_MUN_N", "MES"])  # Normal de cada fila
    for v, corto in (("temp_mean", "temp"), ("prec_total", "prec"), ("ndvi_mean", "ndvi")):  # Anomalías climáticas
        df[f"{corto}_anom"] = df[v] - norm[v]  # Desviación respecto a lo habitual del municipio en ese mes
        df[f"{corto}_anom_m8"] = media_movil(df.groupby("COD_MUN_N")[f"{corto}_anom"], 8)  # Anomalía media de 8 semanas
    df["oni_l8"] = g["oni_anom"].shift(8)  # El Niño con 8 semanas de rezago
    df["temp_missing"] = df["temp_mean_missing"]  # Bandera de temperatura imputada
    df["sem_sin"] = np.sin(2 * np.pi * df["SEMANA"] / 52)  # Estacionalidad (componente seno)
    df["sem_cos"] = np.cos(2 * np.pi * df["SEMANA"] / 52)  # Estacionalidad (componente coseno)
    df["lpob"] = np.log(df["poblacion"])  # Tamaño del municipio (escala log)
    df["flujo_m4"] = np.log1p(media_movil(g["Flujo_in"], 4))  # Flujo de personas entrantes (media de 4 semanas)
    df["depto"] = df["COD_MUN_N"] // 1000  # Departamento (los dos primeros dígitos del código DANE)
    grupo = df.groupby(["depto", "time_idx"])  # Departamento × semana
    resto_casos = grupo["casos_totales"].transform("sum") - df["casos_totales"]  # Casos del resto del departamento (sin el propio municipio)
    df["esp_lc"] = np.log1p(resto_casos)  # Rezago espacial de casos
    df["esp_lc_d4"] = df["esp_lc"] - df.groupby("COD_MUN_N")["esp_lc"].shift(4)  # Cambio del rezago espacial en 4 semanas
    resto_brotes = grupo["brote"].transform("sum") - df["brote"]  # Brotes del resto del departamento
    resto_n = grupo["brote"].transform("count") - 1  # Municipios restantes del departamento
    df["esp_brote"] = resto_brotes / resto_n.clip(lower=1)  # Proporción de municipios vecinos en brote
    return df  # Panel completo (incluye filas sin objetivo)


def panel_modelado(usar_cache=True):  # Filas válidas para modelar (2015–2024 con objetivo y variables completas)
    if usar_cache and cf.RUTA_PANEL.exists():  # Si ya se construyó
        return pd.read_pickle(cf.RUTA_PANEL)  # Lo carga del disco
    df = construir_panel()  # Construye el panel
    columnas = ["COD_MUN_N", "depto", "ANO", "SEMANA", "MES", "time_idx", "week_start", "casos_totales", "casos_obj",  # Identificación y análisis
                "umbral_obj", "poblacion", cf.TARGET] + cf.FEATURES_ORIGINAL  # Variables y objetivo
    mod = df.loc[df["ANO"] >= 2015, columnas].dropna().reset_index(drop=True)  # Filas con canal endémico, objetivo y variables completas
    mod[cf.TARGET] = mod[cf.TARGET].astype(int)  # Objetivo entero 0/1
    mod.to_pickle(cf.RUTA_PANEL)  # Guarda la caché
    return mod  # Panel listo


def division_temporal(mod):  # Entrenamiento, validación y prueba por años
    tr = mod[mod["ANO"] <= cf.ANIO_FIN_ENTRENAMIENTO]  # 2015–2021
    va = mod[mod["ANO"] == cf.ANIO_VALIDACION]  # 2022
    te = mod[mod["ANO"] >= cf.ANIO_INICIO_PRUEBA]  # 2023–2024
    return tr, va, te  # Tres periodos disjuntos y ordenados en el tiempo
