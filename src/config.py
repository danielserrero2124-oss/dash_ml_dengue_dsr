"""Configuración central del proyecto de predicción de brotes de dengue: rutas, periodos, horizonte y columnas.

Todas las demás piezas (datos, modelos, dashboard, notebooks) importan de aquí, de modo que la definición
del problema y la división temporal viven en un único lugar.
"""

from pathlib import Path  # Manejo de rutas independiente del sistema operativo

RAIZ = Path(__file__).resolve().parent.parent  # Carpeta raíz del proyecto (un nivel por encima de src/)
RUTA_DATOS = RAIZ / "data" / "final_dataset.csv"  # Panel original: municipio × semana epidemiológica
RUTA_PANEL = RAIZ / "data" / "panel_procesado.pkl"  # Panel con canal endémico, objetivo y variables (caché)
RUTA_ARTEFACTOS = RAIZ / "dashboard" / "artefactos.joblib"  # Modelos y métricas para el dashboard
RUTA_PRUEBAS = RAIZ / "dashboard" / "pruebas.joblib"  # Métricas de la literatura, pruebas de significancia y tiempos
SEMILLA = 42  # Semilla aleatoria común para que todo sea reproducible

HORIZONTE = 4  # Semanas de anticipación: con la información de la semana t se predice el brote de la semana t + 4
ANIOS_HISTORIA = 5  # Años previos que definen el canal endémico (método de cuartiles de Bortman)
VENTANA_SEMANAS = 2  # Semanas a cada lado de la semana objetivo dentro del canal endémico
CASOS_MINIMOS = 3  # Un brote exige además al menos 3 casos (evita «brotes» de 1 caso en municipios sin historia)
ANIO_FIN_ENTRENAMIENTO = 2021  # Entrenamiento: 2015–2021 (los años 2010–2014 solo construyen el canal endémico)
ANIO_VALIDACION = 2022  # Validación: umbral de decisión e hiperparámetros del modelo original
ANIO_INICIO_PRUEBA = 2023  # Prueba: 2023–2024 (incluye la epidemia récord de 2024); se usa una sola vez

# Variables predictoras (todas se calculan con información disponible en la semana t)
CASOS = [  # Historia de casos del propio municipio
    "lc_l0", "lc_l1", "lc_l2", "lc_l3",  # log(1 + casos) de la semana actual y las tres anteriores
    "lc_m4", "lc_m8", "lc_m12",  # Media móvil del log de casos en 4, 8 y 12 semanas
    "brote_l0", "brote_m8",  # Brote en la semana actual y proporción de semanas en brote en las últimas 8
    "ratio_umbral", "l_umbral_obj", "incid",  # Casos actuales frente al umbral de la semana objetivo, umbral objetivo e incidencia
]
CLIMA = [  # Condiciones climáticas y ambientales
    "temp_mean_m1", "temp_mean_m4", "temp_mean_m8", "temp_mean_m12",  # Temperatura media (semana y medias móviles)
    "prec_total_m1", "prec_total_m4", "prec_total_m8", "prec_total_m12",  # Precipitación (semana y medias móviles)
    "ndvi_mean_m1", "ndvi_mean_m4", "ndvi_mean_m8", "ndvi_mean_m12",  # Índice de vegetación NDVI
    "temp_anom_m8", "prec_anom_m8", "ndvi_anom_m8",  # Anomalías respecto a la normal del municipio (media de 8 semanas)
    "oni_anom", "oni_l8",  # Índice ONI de El Niño (actual y 8 semanas antes)
    "temp_missing",  # Bandera: temperatura imputada en el dataset original
]
CONTEXTO = ["sem_sin", "sem_cos", "lpob", "flujo_m4"]  # Estacionalidad, población y flujo de personas
ESPACIALES = ["esp_lc", "esp_lc_d4", "esp_brote"]  # Autocorrelación espacial: resto del departamento (solo el modelo original)
FEATURES = CASOS + CLIMA + CONTEXTO  # Variables de los cuatro modelos recomendados
FEATURES_ORIGINAL = FEATURES + ESPACIALES  # Variables de los expertos del modelo original
TARGET = "y"  # Objetivo: 1 si la semana t + 4 es de brote en el municipio
