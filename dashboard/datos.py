"""Datos del dashboard: agregados del EDA precalculados (eda.joblib), artefactos de los modelos y carga diferida de los paneles completos."""

import joblib  # Carga de artefactos
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas

from src import config as cf  # Configuración
from src import datos as dt  # Construcción del panel

DEPARTAMENTOS = {5: "Antioquia", 8: "Atlántico", 11: "Bogotá D. C.", 13: "Bolívar", 15: "Boyacá", 17: "Caldas", 18: "Caquetá", 19: "Cauca",  # Códigos DANE
                 20: "Cesar", 23: "Córdoba", 25: "Cundinamarca", 27: "Chocó", 41: "Huila", 44: "La Guajira", 47: "Magdalena", 50: "Meta",
                 52: "Nariño", 54: "Norte de Santander", 63: "Quindío", 66: "Risaralda", 68: "Santander", 70: "Sucre", 73: "Tolima",
                 76: "Valle del Cauca", 81: "Arauca", 85: "Casanare", 86: "Putumayo", 91: "Amazonas", 94: "Guainía", 95: "Guaviare", 97: "Vaupés", 99: "Vichada"}
MUNICIPIOS = {76001: "Cali", 68001: "Bucaramanga", 5001: "Medellín", 73001: "Ibagué", 54001: "Cúcuta", 50001: "Villavicencio", 8001: "Barranquilla",  # Principales municipios
              41001: "Neiva", 13001: "Cartagena", 11001: "Bogotá", 20001: "Valledupar", 23001: "Montería", 66001: "Pereira", 68276: "Floridablanca",
              76520: "Palmira", 63001: "Armenia", 85001: "Yopal", 73268: "Espinal", 76834: "Tuluá", 41551: "Pitalito", 76111: "Guadalajara de Buga",
              50006: "Acacías", 47001: "Santa Marta", 68081: "Barrancabermeja", 18001: "Florencia", 20011: "Aguachica", 76147: "Cartago",
              68547: "Piedecuesta", 54498: "Ocaña", 41298: "Garzón", 44001: "Riohacha", 68679: "San Gil", 25290: "Fusagasugá", 91001: "Leticia",
              73411: "Líbano", 76275: "Florida", 5088: "Bello", 19001: "Popayán", 73168: "Chaparral", 13430: "Magangué"}


def nombre_municipio(cod):  # Nombre legible de un municipio
    dep = DEPARTAMENTOS.get(int(cod) // 1000, "")  # Departamento
    return f"{MUNICIPIOS[cod]} ({dep})" if cod in MUNICIPIOS else f"Municipio {cod} ({dep})"  # Nombre o código


RUTA_EDA = cf.RAIZ / "dashboard" / "eda.joblib"  # Agregados del EDA precalculados (el dashboard no carga los paneles completos)
VARS_DECILES = ["temp_mean_m8", "prec_total_m8", "ndvi_mean_m8", "temp_anom_m8", "prec_anom_m8", "ndvi_anom_m8", "oni_l8", "flujo_m4"]  # Variables del gráfico de deciles
CONTINUAS = ["lc_m8", "incid", "ratio_umbral", "temp_mean_m8", "prec_total_m8", "ndvi_mean_m8", "temp_anom_m8", "prec_anom_m8", "ndvi_anom_m8", "flujo_m4", "esp_lc"]  # Para los atípicos
N_MUN_SIMULADOR = 25  # Municipios disponibles en el simulador


def _deciles(mod, var):  # Tasa de brote a 4 semanas por decil de una variable
    q = pd.qcut(mod[var].rank(method="first"), 10, labels=False)  # Deciles (rangos para romper empates)
    return mod.groupby(q).agg(tasa=(cf.TARGET, "mean"), n=(cf.TARGET, "size"), lo=(var, "min"), hi=(var, "max"), centro=(var, "median"))  # Resumen


def _correlacion_cruzada(raw, rezagos=range(0, 21)):  # Correlación entre la anomalía climática en t − k y el log de casos en t, media de los municipios
    salida = {"rezago": list(rezagos)}  # Rezagos
    lc = np.log1p(raw["casos_totales"].to_numpy()).reshape(-1, 782)  # Municipios × semanas
    for v in ("temp", "prec", "ndvi"):  # Anomalías
        a = raw[f"{v}_anom"].to_numpy().reshape(-1, 782)  # Municipios × semanas
        medias = []  # Correlación media por rezago
        for k in rezagos:  # Cada rezago
            x, y = (a[:, :-k], lc[:, k:]) if k else (a, lc)  # Clima antes que los casos
            xc, yc = x - x.mean(axis=1, keepdims=True), y - y.mean(axis=1, keepdims=True)  # Centra por municipio
            den = np.sqrt((xc ** 2).sum(axis=1) * (yc ** 2).sum(axis=1))  # Denominador
            medias.append(float(np.nanmean((xc * yc).sum(axis=1) / np.where(den > 0, den, np.nan))))  # Media de las correlaciones por municipio
        salida[v] = medias  # Guarda
    return pd.DataFrame(salida)  # Tabla


def _atipicos(mod):  # Regla de Tukey (1,5 × IQR) y tasa de brote dentro y fuera de los atípicos
    filas = []  # Una fila por variable
    for c in CONTINUAS:  # Variables continuas
        q1, q3 = mod[c].quantile([0.25, 0.75])  # Cuartiles
        at = (mod[c] < q1 - 1.5 * (q3 - q1)) | (mod[c] > q3 + 1.5 * (q3 - q1))  # Atípicos
        filas.append({"variable": c, "pct_atipicos": float(at.mean()), "tasa_atipicos": float(mod.loc[at, cf.TARGET].mean()) if at.any() else 0.0,
                      "tasa_normales": float(mod.loc[~at, cf.TARGET].mean())})  # Resumen
    return sorted(filas, key=lambda f: f["pct_atipicos"], reverse=True)  # Orden


def preparar_eda():  # Calcula todos los agregados del EDA a partir de los paneles completos y los guarda
    raw, mod = dt.construir_panel(), dt.panel_modelado()  # Paneles completos (≈ 220 MB en memoria)
    _, _, te = dt.division_temporal(mod)  # Periodo de prueba
    fechas = raw.drop_duplicates("time_idx").set_index("time_idx")["week_start"]  # Fecha de cada semana
    pob_dep = raw.groupby(["depto", "COD_MUN_N"])["poblacion"].mean().groupby("depto").sum()  # Población por departamento
    dep = pd.DataFrame({"casos": raw.groupby("depto")["casos_totales"].sum(), "pob": pob_dep})  # Casos y población
    dep["incid"] = dep["casos"] / raw["ANO"].nunique() / dep["pob"] * 1e5  # Incidencia media anual por 100 000
    dep["nombre"] = [DEPARTAMENTOS.get(d, str(d)) for d in dep.index]  # Nombres
    top = raw.groupby("COD_MUN_N")["casos_totales"].sum().sort_values(ascending=False).index[:40].tolist()  # 40 municipios con más casos
    canal = raw.loc[raw["COD_MUN_N"].isin(top) & (raw["ANO"] >= 2015), ["COD_MUN_N", "week_start", "casos_totales", "umbral", "brote"]]  # Series del canal endémico
    cols_desc = ["casos_totales", "temp_mean", "prec_total", "ndvi_mean", "oni_anom", "Flujo_in", "poblacion"]  # Variables originales
    desc = raw[cols_desc].describe().T.assign(asimetria=raw[cols_desc].skew())  # Estadística descriptiva
    eda = {
        "NAC": raw.groupby("time_idx").agg(casos=("casos_totales", "sum"), oni=("oni_anom", "first")).assign(fecha=fechas).reset_index(),  # Serie nacional
        "CORTES": {"val": str(raw.loc[raw["ANO"] == cf.ANIO_VALIDACION, "week_start"].min().date()),  # Inicio de 2022
                   "test": str(raw.loc[raw["ANO"] == cf.ANIO_INICIO_PRUEBA, "week_start"].min().date()), "fin": str(raw["week_start"].max().date())},
        "ANUAL": raw.groupby("ANO").agg(casos=("casos_totales", "sum"), tasa_brote=("brote", "mean")),  # Casos y brotes por año
        "EST": pd.DataFrame({"casos": raw.groupby(["ANO", "SEMANA"])["casos_totales"].sum().groupby("SEMANA").mean(),  # Perfil estacional
                             "prec": raw.groupby("SEMANA")["prec_total"].mean()}).loc[1:52],
        "DEP": dep, "TOP_MUN": top, "CANAL": canal.reset_index(drop=True), "DESCRIPTIVA": desc,  # Departamentos, municipios, canal y descriptiva
        "EDAD": {"0–4": int(raw["casos_0_4"].sum()), "5–14": int(raw["casos_5_14"].sum()), "15–64": int(raw["casos_15_64"].sum()), "65 y más": int(raw["casos_65_plus"].sum())},
        "SEXO": {"Hombres": int(raw["casos_m"].sum()), "Mujeres": int(raw["casos_f"].sum())},  # Casos por sexo
        "FALTANTES": raw.groupby("ANO")["temp_mean_missing"].mean(),  # Temperatura imputada por año
        "ONI_ANUAL": raw[raw["ANO"] >= 2015].groupby("ANO").agg(oni=("oni_anom", "mean"), tasa=("brote", "mean")),  # El Niño y brotes por año
        "CCF": _correlacion_cruzada(raw), "CORR_Y": mod[cf.FEATURES_ORIGINAL].corrwith(mod[cf.TARGET]), "CORR": mod[cf.FEATURES_ORIGINAL].corr(),  # Correlaciones
        "DECILES": {v: _deciles(mod, v) for v in VARS_DECILES}, "ATIPICOS": _atipicos(mod),  # Deciles y atípicos
        "TE_SIM": te[te["COD_MUN_N"].isin(top[:N_MUN_SIMULADOR])].reset_index(drop=True),  # Filas de prueba del simulador (25 municipios)
        "RESUMEN": {"n_raw": len(raw), "n_mun": int(raw["COD_MUN_N"].nunique()), "n_semanas": int(raw["time_idx"].nunique()),  # Cifras de cabecera
                    "casos_total": int(raw["casos_totales"].sum()), "n_brotes": int(raw["brote"].sum()), "tasa_brote": float(raw["brote"].mean()),
                    "temp_imputada": float(raw["temp_mean_missing"].mean()), "casos_media": float(raw["casos_totales"].mean()),
                    "casos_mediana": float(raw["casos_totales"].median()), "n_mod": len(mod), "tasa_mod": float(mod[cf.TARGET].mean()),
                    "tasa_tr": float(mod.loc[mod["ANO"] <= cf.ANIO_FIN_ENTRENAMIENTO, cf.TARGET].mean()), "tasa_te": float(te[cf.TARGET].mean())},
    }
    joblib.dump(eda, RUTA_EDA, compress=3)  # Guarda (≈ 2 MB)
    return eda  # Diccionario


_EDA = joblib.load(RUTA_EDA) if RUTA_EDA.exists() else preparar_eda()  # Agregados (se calculan una sola vez)
NAC, CORTES, ANUAL, EST = _EDA["NAC"], _EDA["CORTES"], _EDA["ANUAL"], _EDA["EST"]  # Serie nacional, cortes, casos por año y estacionalidad
DEP, TOP_MUN, CANAL, DESCRIPTIVA = _EDA["DEP"], _EDA["TOP_MUN"], _EDA["CANAL"], _EDA["DESCRIPTIVA"]  # Departamentos, municipios, canal y descriptiva
EDAD, SEXO, FALTANTES, ONI_ANUAL = _EDA["EDAD"], _EDA["SEXO"], _EDA["FALTANTES"], _EDA["ONI_ANUAL"]  # Grupos, faltantes y El Niño
CCF, CORR_Y, CORR, ATIPICOS = _EDA["CCF"], _EDA["CORR_Y"], _EDA["CORR"], _EDA["ATIPICOS"]  # Correlaciones y atípicos
TE_SIM, RESUMEN = _EDA["TE_SIM"], _EDA["RESUMEN"]  # Filas del simulador y cifras de cabecera
TASA = RESUMEN["tasa_mod"]  # Proporción de municipio-semanas con brote a 4 semanas
SEMANAS_TEST = sorted(TE_SIM["week_start"].dt.strftime("%Y-%m-%d").unique())  # Semanas de prueba del simulador


def deciles(var):  # Tabla precalculada de deciles
    return _EDA["DECILES"][var]  # Sin recalcular


def __getattr__(nombre):  # Carga diferida de los paneles completos (solo si alguien los pide, p. ej. los cuadernos)
    if nombre in ("RAW", "MOD", "TR", "VA", "TE"):  # Paneles grandes
        raw, mod = dt.construir_panel(), dt.panel_modelado()  # Construye
        tr, va, te = dt.division_temporal(mod)  # Periodos
        globals().update({"RAW": raw, "MOD": mod, "TR": tr, "VA": va, "TE": te})  # Los deja en caché en el módulo
        return globals()[nombre]  # Devuelve el pedido
    raise AttributeError(nombre)  # Cualquier otro nombre no existe


ART = joblib.load(cf.RUTA_ARTEFACTOS) if cf.RUTA_ARTEFACTOS.exists() else None  # Artefactos del entrenamiento
PRUEBAS = joblib.load(cf.RUTA_PRUEBAS) if cf.RUTA_PRUEBAS.exists() else None  # Pruebas de significancia
RUTA_HETDS = cf.RAIZ / "dashboard" / "hetds.joblib"  # Modelo original para el simulador
RES = ART["resultados"] if ART else {}  # Resultados por modelo
Y_TEST = ART["y_test"] if ART else None  # Etiquetas de prueba
_HETDS = {}  # Caché del modelo original


def hetds():  # Carga el modelo original solo cuando el simulador lo necesita (≈ 180 MB en memoria)
    if "modelo" not in _HETDS and RUTA_HETDS.exists():  # Primera vez
        _HETDS["modelo"] = joblib.load(RUTA_HETDS)  # Carga
    return _HETDS.get("modelo")  # Modelo o None
