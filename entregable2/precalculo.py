"""Precalcula todo lo que muestra el dashboard del Segundo Entregable y lo guarda en tablero/artefactos.joblib.

El dashboard solo lee ese archivo: no reentrena ni recalcula nada al abrirse, así que carga rápido y usa poca memoria.
Reutiliza el código del Primer Entregable (paquete e1) y del proyecto (paquete src).

Uso (desde la carpeta entregable2):  python precalculo.py      (≈ 10 minutos)
"""

import os  # Rutas y variables de entorno
import sys  # Rutas de importación
import time  # Cronómetro
import warnings  # Control de avisos
from pathlib import Path  # Rutas

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # Evita un aviso de joblib en Windows
AQUI = Path(__file__).resolve().parent  # Carpeta entregable2
sys.path.insert(0, str(AQUI.parent))  # Raíz del proyecto (paquete src)
sys.path.insert(0, str(AQUI.parent / "entregable1"))  # Paquete e1 del Primer Entregable
warnings.filterwarnings("ignore")  # Oculta avisos de versiones que no afectan los resultados

import joblib  # noqa: E402  Guardado de artefactos
import numpy as np  # noqa: E402  Cálculo numérico
import pandas as pd  # noqa: E402  Tablas
from scipy import stats  # noqa: E402  Pruebas estadísticas
from sklearn.cluster import DBSCAN, KMeans  # noqa: E402  Agrupamientos
from sklearn.decomposition import PCA  # noqa: E402  Componentes principales
from sklearn.ensemble import IsolationForest  # noqa: E402  Atípicos por aislamiento
from sklearn.feature_selection import mutual_info_classif  # noqa: E402  Información mutua
from sklearn.impute import SimpleImputer  # noqa: E402  Imputación
from sklearn.linear_model import LogisticRegression  # noqa: E402  Escalado de Platt
from sklearn.manifold import TSNE  # noqa: E402  Proyección visual
from sklearn.metrics import average_precision_score, roc_auc_score, silhouette_score  # noqa: E402  Métricas
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold, cross_val_score  # noqa: E402  Validación
from sklearn.pipeline import Pipeline  # noqa: E402  Encadena pasos
from sklearn.preprocessing import StandardScaler  # noqa: E402  Estandarización
from sklearn.svm import LinearSVC  # noqa: E402  Máquina de vectores de soporte lineal
from statsmodels.tsa.seasonal import STL  # noqa: E402  Descomposición
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf  # noqa: E402  Series de tiempo

from e1 import datos as d, espacial as sp, estadistica as es, modelo_base as mb  # noqa: E402  Código del Primer Entregable
from src import config as cf  # noqa: E402  Configuración del proyecto

warnings.simplefilter("ignore")  # statsmodels reactiva sus avisos al importarse
DESTINO = AQUI / "tablero" / "artefactos.joblib"  # Archivo que lee el dashboard
RNG = np.random.default_rng(cf.SEMILLA)  # Generador reproducible


def histograma(x, bins=50, log=False):  # Histograma precalculado (centros y proporciones)
    x = np.log1p(np.asarray(x, float)) if log else np.asarray(x, float)  # Escala opcional
    conteo, bordes = np.histogram(x[~np.isnan(x)], bins=bins)  # Conteos
    return {"x": (bordes[:-1] + bordes[1:]) / 2, "p": conteo / conteo.sum()}  # Centros y proporciones


def caja(x):  # Estadísticos de un diagrama de caja de Tukey
    x = np.asarray(x, float)  # Vector
    q1, me, q3 = np.quantile(x, [0.25, 0.5, 0.75])  # Cuartiles
    iqr = q3 - q1  # Rango intercuartílico
    return {"q1": q1, "mediana": me, "q3": q3, "bajo": max(x.min(), q1 - 1.5 * iqr), "alto": min(x.max(), q3 + 1.5 * iqr),  # Caja y bigotes
            "min": x.min(), "max": x.max()}  # Extremos


# ---------------------------------------------------------------------------
# 1. Datos, partición y contexto
# ---------------------------------------------------------------------------
def contexto(raw, df, tr, va, te, m):
    anual = raw.groupby("ANO")["casos_totales"].sum()  # Casos por año
    res = df.groupby("periodo").agg(filas=("COD_MUN_N", "size"), desde=("week_start", "min"), hasta=("week_start", "max"))  # Partición
    res = res.loc[["historia", "entrenamiento", "validación", "prueba", "purgada"]]  # Orden
    linea = df.groupby(["week_start", "periodo"]).size().unstack(fill_value=0)  # Línea de tiempo de la partición
    return {
        "kpi": {"casos": int(raw["casos_totales"].sum()), "municipios": int(raw["COD_MUN_N"].nunique()), "semanas": int(raw["time_idx"].nunique()),  # Tamaño
                "filas": len(raw), "columnas": raw.shape[1], "anio_record": int(anual.idxmax()), "casos_record": int(anual.max()),  # Récord
                "prev_tr": float(tr[cf.TARGET].mean()), "prev_va": float(va[cf.TARGET].mean()), "n_tr": len(tr), "n_va": len(va), "n_te": len(te),  # Partición
                "brotes_tr": int(tr[cf.TARGET].sum()), "p": len(cf.FEATURES), "departamentos": int(m["depto"].nunique())},  # Clase minoritaria
        "anual": anual, "particion": res, "linea_particion": linea,  # Series
    }


# ---------------------------------------------------------------------------
# 2. Calidad de los datos
# ---------------------------------------------------------------------------
def calidad(raw, eda, m):
    frac = raw.groupby("COD_MUN_N")["temp_mean_missing"].mean()  # Fracción imputada por municipio
    grupo = pd.cut(frac, [-0.01, 0, 0.99, 1], labels=["nunca imputada", "parcialmente", "siempre imputada"])  # Patrón
    matriz = raw.pivot_table(index="COD_MUN_N", columns="ANO", values="temp_mean_missing", aggfunc="mean").loc[frac.sort_values().index]  # Municipio × año
    X = pd.DataFrame({"temp_mean": eda["temp_mean"].where(eda["temp_mean_missing"] == 0), "prec_total": eda["prec_total"],  # Temperatura con faltantes
                      "ndvi_mean": eda["ndvi_mean"], "oni_anom": eda["oni_anom"], "log_poblacion": np.log(eda["poblacion"]),
                      "log_casos": np.log1p(eda["casos_totales"])})
    d2, gl, p_little, _ = es.prueba_little(X.sample(20000, random_state=cf.SEMILLA).to_numpy())  # Prueba de Little
    falta = X["temp_mean"].isna()  # Indicador de faltante
    smd = {c: es.diferencia_estandarizada(X.loc[falta, c], X.loc[~falta, c]) for c in X.columns[1:]}  # Diferencias estandarizadas
    cont = ["casos_totales", "temp_mean", "prec_total", "ndvi_mean", "oni_anom", "Flujo_in", "poblacion"]  # Continuas
    q1, q3 = raw[cont].quantile(0.25), raw[cont].quantile(0.75)  # Cuartiles
    iqr = q3 - q1  # Rango
    atip = (((raw[cont] < q1 - 1.5 * iqr) | (raw[cont] > q3 + 1.5 * iqr)).mean() * 100).rename("atípicos (%)")  # Regla de Tukey
    edades = ["casos_0_4", "casos_5_14", "casos_15_64", "casos_65_plus"]  # Grupos de edad
    chequeos = {"Filas duplicadas exactas": int(raw.duplicated().sum()),  # Duplicados
                "Claves (municipio, semana) repetidas": int(raw.duplicated(["COD_MUN_N", "time_idx"]).sum()),
                "Suma de edades ≠ total": int((raw[edades].sum(axis=1) != raw["casos_totales"]).sum()),  # Consistencia
                "Hombres + mujeres ≠ total": int((raw["casos_m"] + raw["casos_f"] != raw["casos_totales"]).sum()),
                "Temperatura fuera de [0, 40] °C": int((~raw["temp_mean"].between(0, 40)).sum()),  # Rangos físicos
                "NDVI fuera de [−1, 1]": int((~raw["ndvi_mean"].between(-1, 1)).sum()),
                "Conteos o lluvia negativos": int((raw[["casos_totales", "prec_total"]] < 0).sum().sum()),
                "Semanas que no avanzan 7 días": int((raw.sort_values(["COD_MUN_N", "time_idx"]).groupby("COD_MUN_N")["week_start"].diff().dt.days.dropna() != 7).sum())}
    todos = pd.read_csv(d.RUTA_DIVIPOLA, dtype=str)  # Los 1 122 municipios del país
    todos["region"] = todos["cod_dpto"].astype(int).map(d.REGIONES)  # Región
    rep = pd.DataFrame({"país": todos["region"].value_counts(), "panel": m["region"].value_counts()})  # Representatividad
    rep["cobertura"] = rep["panel"] / rep["país"]  # Cobertura
    rep["participación panel"] = rep["panel"] / rep["panel"].sum()  # Composición del panel
    rep["participación país"] = rep["país"] / rep["país"].sum()  # Composición nacional
    imp, obs = raw.loc[raw["temp_mean_missing"] == 1, "temp_mean"], raw.loc[raw["temp_mean_missing"] == 0, "temp_mean"]  # Temperaturas
    bordes = np.linspace(9, 35, 105)  # Intervalos de 0,25 °C
    return {"patron": grupo.value_counts(), "matriz_faltantes": matriz, "frac_imputada": float(raw["temp_mean_missing"].mean()),
            "little": {"d2": d2, "gl": gl, "p": p_little}, "smd": smd, "atipicos": atip, "chequeos": chequeos, "representatividad": rep,
            "pob_cubierta": float(raw.loc[raw["ANO"] == 2024].groupby("COD_MUN_N")["poblacion"].first().sum()),
            "hist_temp": {"x": (bordes[:-1] + bordes[1:]) / 2, "obs": np.histogram(obs, bordes, density=True)[0], "imp": np.histogram(imp, bordes, density=True)[0]},
            "top_casos": raw.nlargest(8, "casos_totales").merge(m, on="COD_MUN_N")[["municipio", "departamento", "week_start", "casos_totales"]]}


# ---------------------------------------------------------------------------
# 3. Variable objetivo
# ---------------------------------------------------------------------------
def objetivo(eda, tr, m):
    e15 = eda[eda["ANO"] >= 2015]  # Periodo con canal endémico
    canal = {mun: g[["week_start", "casos_totales", "umbral", "brote"]].reset_index(drop=True) for mun, g in e15.groupby("COD_MUN_N")}  # Serie por municipio
    nac = e15.groupby("week_start").agg(brote=("brote", "mean"), casos=("casos_totales", "sum"))  # Serie nacional
    mapa = tr.groupby("COD_MUN_N")[cf.TARGET].mean().rename("prevalencia")  # Prevalencia por municipio
    return {"canal": canal, "nacional": nac, "anual": tr.groupby("ANO")[cf.TARGET].agg(["mean", "sum", "size"]),
            "mes": tr.groupby("MES")[cf.TARGET].mean(), "region": tr.groupby("region")[cf.TARGET].mean(), "mapa": mapa}


# ---------------------------------------------------------------------------
# 4. Análisis unidimensional, bidimensional y multivariado
# ---------------------------------------------------------------------------
def univariado(eda, tr):
    num = ["casos_totales", "temp_mean", "prec_total", "ndvi_mean", "oni_anom", "oni_total", "Flujo_in", "poblacion"]  # Originales
    muestra = eda[num].sample(5000, random_state=cf.SEMILLA)  # Muestra para normalidad
    filas = []  # Estadísticos
    for v in num:  # Cada variable
        x = eda[v]  # Serie
        q1, q3 = x.quantile([0.25, 0.75])  # Cuartiles
        filas.append({"variable": v, "media": x.mean(), "mediana": x.median(), "desv. est.": x.std(), "p5": x.quantile(0.05),
                      "p95": x.quantile(0.95), "asimetría": x.skew(), "curtosis": x.kurt(),
                      "atípicos IQR (%)": ((x < q1 - 1.5 * (q3 - q1)) | (x > q3 + 1.5 * (q3 - q1))).mean() * 100,
                      "p normalidad (K²)": stats.normaltest(muestra[v])[1]})
    hist = {}  # Histogramas de variables originales y predictoras
    for v in num:  # Originales
        log = v in ("casos_totales", "Flujo_in", "poblacion")  # Escala log para colas largas
        hist[v] = {**histograma(eda[v], log=log), "log": log, **caja(np.log1p(eda[v]) if log else eda[v])}
    for v in cf.FEATURES:  # Predictoras
        hist[v] = {**histograma(tr[v]), "log": False, **caja(tr[v])}
    der = tr[cf.FEATURES].describe(percentiles=[0.05, 0.5, 0.95]).T  # Predictoras
    der["asimetría"], der["curtosis"] = tr[cf.FEATURES].skew(), tr[cf.FEATURES].kurt()  # Forma
    fase = pd.cut(eda["oni_anom"], [-9, -0.5, 0.5, 9], labels=["La Niña", "Neutral", "El Niño"])  # Fase ENSO
    cat = {"departamento": eda.groupby("departamento")["COD_MUN_N"].nunique().sort_values(),  # Municipios por departamento
           "region": eda.groupby("region")["COD_MUN_N"].nunique(),  # Municipios por región
           "enso": fase[~eda["time_idx"].duplicated()].value_counts()}  # Semanas por fase
    return {"tabla": pd.DataFrame(filas).set_index("variable"), "hist": hist, "derivadas": der, "categoricas": cat}


def bivariado(tr):
    obs = tr[tr["temp_missing"] == 0]  # Temperatura medida
    pares = {}  # Densidades conjuntas
    for x, y, base in [("temp_mean_m4", "prec_total_m4", obs), ("lc_m8", "lc_l0", tr), ("temp_mean_m8", "lc_m8", obs), ("oni_l8", "lc_m8", tr)]:  # Pares
        H, bx, by = np.histogram2d(base[x], base[y], bins=50)  # Conteos
        pares[f"{x}|{y}"] = {"x": (bx[:-1] + bx[1:]) / 2, "y": (by[:-1] + by[1:]) / 2, "z": np.log1p(H.T),  # Densidad
                             "pearson": stats.pearsonr(base[x], base[y])[0], "spearman": stats.spearmanr(base[x], base[y])[0], "n": len(base)}
    corr = tr[cf.FEATURES].corr(method="spearman")  # Correlación de Spearman
    vif = pd.Series(es.vif(tr[cf.FEATURES].sample(50000, random_state=cf.SEMILLA)), index=cf.FEATURES)  # VIF
    muestra = tr.sample(40000, random_state=cf.SEMILLA)  # Muestra para información mutua
    mi = mutual_info_classif(muestra[cf.FEATURES], muestra[cf.TARGET], random_state=cf.SEMILLA)  # Información mutua
    y = tr[cf.TARGET].to_numpy()  # Objetivo
    filas = []  # Asociación con el objetivo
    for j, v in enumerate(cf.FEATURES):  # Cada predictora
        x = tr[v].to_numpy()  # Valores
        rrb, p = es.mann_whitney(x[y == 1], x[y == 0])  # Mann-Whitney
        filas.append({"variable": v, "r_rb": rrb, "p": p, "auc": roc_auc_score(y, x), "mi": mi[j]})  # Resultado
    asoc = pd.DataFrame(filas).set_index("variable")  # Tabla
    asoc["p_holm"], asoc["q_fdr"] = es.holm(asoc["p"]), es.fdr_bh(asoc["p"])  # Correcciones
    deciles = {}  # Tasa de brote por decil de cada predictora
    for v in cf.FEATURES:  # Cada predictora
        base = obs if v.startswith("temp_mean") else tr  # Temperatura: solo medida
        q = pd.qcut(base[v], 10, duplicates="drop")  # Deciles
        t = base.groupby(q, observed=True)[cf.TARGET].mean()  # Tasa por decil
        deciles[v] = {"x": [iv.mid for iv in t.index], "tasa": t.values}  # Curva
    tr2 = tr.assign(fase_enso=pd.cut(tr["oni_anom"], [-9, -0.5, 0.5, 9], labels=["La Niña", "Neutral", "El Niño"]))  # Fase ENSO
    cramer = []  # Categóricas frente al objetivo
    for v in ["MES", "region", "departamento", "fase_enso", "temp_missing"]:  # Variables
        V, chi2, gl, p = es.v_cramer(tr2[v], tr2[cf.TARGET])  # V de Cramér
        cramer.append({"variable": v, "categorías": tr2[v].nunique(), "χ²": chi2, "gl": gl, "p": p, "V de Cramér": V})
    cramer = pd.DataFrame(cramer).set_index("variable")  # Tabla
    cramer["p (Holm)"] = es.holm(cramer["p"])  # Corrección
    regiones = sorted(tr["region"].unique())  # Regiones
    kw, cajas_reg = [], {}  # Kruskal-Wallis y cajas por región
    for v, base in [("lc_l0", tr), ("temp_mean_m8", obs), ("prec_total_m8", tr), ("ndvi_mean_m8", tr)]:  # Variables
        h, p, e2 = es.kruskal([base.loc[base["region"] == r, v] for r in regiones])  # Prueba
        kw.append({"variable": v, "H": h, "p": p, "ε²": e2})  # Resultado
        cajas_reg[v] = {r: caja(base.loc[base["region"] == r, v]) for r in regiones}  # Cajas
    kw = pd.DataFrame(kw).set_index("variable")  # Tabla
    kw["p (Holm)"] = es.holm(kw["p"])  # Corrección
    return {"pares": pares, "corr": corr, "vif": vif, "asociacion": asoc, "deciles": deciles, "cramer": cramer,
            "enso": pd.crosstab(tr2["fase_enso"], tr2[cf.TARGET], normalize="index"), "kruskal": kw, "cajas_region": cajas_reg}


def multivariado(tr, m):
    Z = StandardScaler().fit_transform(tr[cf.FEATURES])  # Estandariza con el entrenamiento
    pca = PCA(random_state=cf.SEMILLA).fit(Z)  # PCA
    idx = RNG.choice(len(Z), 4000, replace=False)  # Muestra para la figura
    P = pca.transform(Z[idx])[:, :2]  # CP1 y CP2
    idx2 = RNG.choice(len(Z), 3000, replace=False)  # Muestra para t-SNE
    emb = TSNE(n_components=2, perplexity=40, init="pca", random_state=cf.SEMILLA).fit_transform(Z[idx2])  # t-SNE
    sub = Z[RNG.choice(len(Z), 30000, replace=False)]  # Muestra para la covarianza
    dif = Z - sub.mean(axis=0)  # Desviaciones
    d2 = np.einsum("ij,jk,ik->i", dif, np.linalg.pinv(np.cov(sub, rowvar=False)), dif)  # Mahalanobis²
    atip_m = d2 > stats.chi2.ppf(0.999, df=Z.shape[1])  # Atípicos de Mahalanobis
    atip_i = IsolationForest(n_estimators=200, random_state=cf.SEMILLA, n_jobs=-1).fit(Z).predict(Z) == -1  # Isolation Forest
    y = tr[cf.TARGET].to_numpy()  # Objetivo
    atip = pd.DataFrame({"atípicos (%)": [atip_m.mean() * 100, atip_i.mean() * 100, (atip_m & atip_i).mean() * 100],
                         "tasa de brote en atípicos": [y[atip_m].mean(), y[atip_i].mean(), y[atip_m & atip_i].mean()],
                         "tasa de brote en el resto": [y[~atip_m].mean(), y[~atip_i].mean(), y[~(atip_m & atip_i)].mean()]},
                        index=["Mahalanobis (χ² 99,9 %)", "Isolation Forest", "Ambos métodos"])
    perfil = tr.groupby("COD_MUN_N").agg(log_incidencia=("incid", lambda s: np.log1p(s.mean())), prevalencia=(cf.TARGET, "mean"),  # Perfil municipal
                                         temp=("temp_mean_m4", "mean"), lluvia=("prec_total_m4", "mean"), ndvi=("ndvi_mean_m4", "mean"),
                                         lpob=("lpob", "mean"), flujo=("flujo_m4", "mean"))
    Zp = StandardScaler().fit_transform(perfil)  # Estandariza
    sil = {k: silhouette_score(Zp, KMeans(k, n_init=20, random_state=cf.SEMILLA).fit_predict(Zp)) for k in range(2, 9)}  # Silueta
    k = max(sil, key=sil.get)  # Mejor k
    perfil["grupo"] = KMeans(k, n_init=20, random_state=cf.SEMILLA).fit_predict(Zp)  # Grupos
    return {"varianza": pca.explained_variance_ratio_, "autovalores": pca.explained_variance_,
            "cargas": pd.DataFrame(pca.components_[:4].T, index=cf.FEATURES, columns=["CP1", "CP2", "CP3", "CP4"]),
            "pca_xy": P, "pca_y": y[idx], "tsne": emb, "tsne_y": y[idx2], "tsne_region": tr["region"].to_numpy()[idx2],
            "atipicos": atip, "atipicos_municipios": tr.loc[atip_m & atip_i, "municipio"].value_counts().head(8),
            "silueta": sil, "k": k, "perfil": perfil.join(m.set_index("COD_MUN_N")[["municipio", "latitud", "longitud"]])}


# ---------------------------------------------------------------------------
# 5. Fuga, componente temporal, espacial y espacio-temporal
# ---------------------------------------------------------------------------
def fuga(tr):
    cand = {**{v: tr[v] for v in cf.FEATURES}, "casos_obj (casos de t+4)": np.log1p(tr["casos_obj"]),  # Candidatas
            "COD_MUN_N (identificador)": tr["COD_MUN_N"], "time_idx (identificador)": tr["time_idx"], "ANO (identificador)": tr["ANO"]}
    auc = pd.Series({k: max(roc_auc_score(tr[cf.TARGET], x), 1 - roc_auc_score(tr[cf.TARGET], x)) for k, x in cand.items()}).sort_values(ascending=False)  # AUC univariado
    pipe = mb.pipeline_base().set_params(clf__C=0.01)  # Mismo Pipeline del modelo base
    con_futuro = tr.assign(lc_obj=np.log1p(tr["casos_obj"]))  # Con información del futuro
    demo = pd.Series({  # Demostración de cuánto infla la fuga
        "Correcto: años hacia adelante con purga": cross_val_score(pipe, tr[cf.FEATURES], tr[cf.TARGET], groups=mb.grupos(tr), cv=mb.ValidacionAnualPurgada(), scoring="roc_auc").mean(),
        "Fuga temporal: semanas mezcladas al azar": cross_val_score(pipe, tr[cf.FEATURES], tr[cf.TARGET], cv=KFold(5, shuffle=True, random_state=cf.SEMILLA), scoring="roc_auc").mean(),
        "Fuga directa: casos de la semana objetivo": cross_val_score(pipe, con_futuro[cf.FEATURES + ["lc_obj"]], con_futuro[cf.TARGET], groups=mb.grupos(tr), cv=mb.ValidacionAnualPurgada(), scoring="roc_auc").mean()})
    return {"auc_univariado": auc, "demo": demo}


def temporal(eda, tr, va):
    nac = eda.groupby("week_start")["casos_totales"].sum()  # Serie nacional semanal
    ln = np.log1p(nac)  # Escala log
    stl = STL(ln, period=52, robust=True).fit()  # Descomposición
    pruebas = []  # Estacionariedad
    for nombre, s in [("Nivel: log casos", ln), ("Primera diferencia", ln.diff().dropna())]:  # Serie y diferencia
        a, k = adfuller(s, autolag="AIC"), kpss(s, regression="c", nlags="auto")  # ADF y KPSS
        pruebas.append({"serie": nombre, "ADF estadístico": a[0], "ADF p": a[1], "KPSS estadístico": k[0], "KPSS p": k[1]})
    p_adf, p_kpss = [], []  # Por municipio
    for _, g in eda[eda["ANO"] >= 2015].groupby("COD_MUN_N"):  # Cada municipio
        x = np.log1p(g["casos_totales"].to_numpy())  # Serie
        if x.std() > 0:  # Si varía
            p_adf.append(adfuller(x, maxlag=8, autolag=None)[1])  # ADF
            p_kpss.append(kpss(x, regression="c", nlags=8)[1])  # KPSS
    base = eda[eda["ANO"] >= 2011].copy()  # Correlación cruzada dentro de los municipios
    base["lc_c"] = base["lc"] - base.groupby("COD_MUN_N")["lc"].transform("mean")  # Casos centrados
    ccf = {}  # Por variable climática
    for v in ["temp_anom", "prec_anom", "ndvi_anom", "oni_anom"]:  # Variables
        xc = base[v] - base.groupby("COD_MUN_N")[v].transform("mean")  # Centrada
        ccf[v] = [np.corrcoef(xc.groupby(base["COD_MUN_N"]).shift(k).fillna(0), base["lc_c"])[0, 1] for k in range(21)]  # corr(x_{t−k}, casos_t)
    cortes = es.segmentacion_binaria(ln.to_numpy(), penalizacion=3 * 2 * np.var(stl.resid) * np.log(len(ln)), min_tam=52)  # Regímenes
    oni = eda.drop_duplicates("week_start").set_index("week_start")["oni_anom"]  # ONI semanal
    nino = list(zip(oni.index[(oni > 0.5) & (oni.shift() <= 0.5)], oni.index[(oni <= 0.5) & (oni.shift() > 0.5)]))  # Episodios de El Niño
    temprano, tardio = tr[tr["ANO"] <= 2018], tr[tr["ANO"] >= 2019]  # Mitades del entrenamiento
    vars_d = ["lc_m4", "brote_m8", "ratio_umbral", "temp_mean_m4", "temp_anom_m8", "prec_total_m4", "ndvi_mean_m4", "oni_anom", "oni_l8", "flujo_m4"]  # Variables
    deriva = pd.DataFrame({v: {"PSI 2015–18 → 2019–21": es.psi(temprano[v], tardio[v]), "PSI entrenamiento → 2022": es.psi(tr[v], va[v])} for v in vars_d}).T  # PSI
    datos_anio = pd.concat([tr, va])  # Entrenamiento y validación
    anios = sorted(datos_anio["ANO"].unique())  # Años
    concepto = pd.DataFrame({v: [roc_auc_score(datos_anio.loc[datos_anio["ANO"] == a, cf.TARGET], datos_anio.loc[datos_anio["ANO"] == a, v]) for a in anios]
                             for v in ["lc_m4", "brote_m8", "ratio_umbral", "oni_l8", "temp_anom_m8"]}, index=anios)  # AUC por año
    orden = d.municipios().sort_values(["region", "latitud"])  # Orden del mapa de calor
    calor = tr.pivot_table(index="COD_MUN_N", columns="week_start", values=cf.TARGET).loc[orden["COD_MUN_N"]]  # Municipio × semana
    tabla_cal = pd.DataFrame({"log_casos": ln.values, "mes": ln.index.month, "semana": ln.index.isocalendar().week.values}, index=ln.index)  # Calendario
    return {"nacional": nac, "mensual": nac.resample("MS").sum(), "stl": pd.DataFrame({"serie": ln, "tendencia": stl.trend, "estacional": stl.seasonal, "residuo": stl.resid}),
            "fuerza": {"tendencia": max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.trend)), "estacional": max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal))},
            "estacionariedad": pd.DataFrame(pruebas).set_index("serie"), "mun_adf": float(np.mean(np.array(p_adf) < 0.05)), "mun_kpss": float(np.mean(np.array(p_kpss) < 0.05)),
            "acf": acf(ln, nlags=104), "pacf": pacf(ln, nlags=52), "n": len(ln), "ccf": pd.DataFrame(ccf), "cortes": [ln.index[i] for i in cortes], "nino": nino,
            "rolling": pd.DataFrame({"media": ln.rolling(52).mean(), "desv": ln.rolling(52).std()}), "deriva": deriva,
            "prevalencias": {"2015–18": temprano[cf.TARGET].mean(), "2019–21": tardio[cf.TARGET].mean(), "2022": va[cf.TARGET].mean()},
            "concepto": concepto, "calor": calor, "calor_regiones": orden["region"].tolist(), "calendario": tabla_cal}


def espacial(tr, m):
    lat, lon = m["latitud"].to_numpy(), m["longitud"].to_numpy()  # Coordenadas
    D, xy = sp.haversine(lat, lon), sp.proyectar(lat, lon)  # Distancias y plano local
    agg = tr.groupby("COD_MUN_N").agg(casos=("casos_totales", "sum"), pob=("poblacion", "mean"), prevalencia=(cf.TARGET, "mean"),
                                      semanas=("time_idx", "size")).loc[m["COD_MUN_N"]]  # Perfil espacial
    agg["log_incidencia"] = np.log1p(agg["casos"] / agg["pob"] / (agg["semanas"] / 52) * 1e5)  # Incidencia anual (log)
    mm = m.join(agg.reset_index(drop=True))  # Une
    clima = tr[tr["temp_missing"] == 0].groupby("COD_MUN_N")["temp_mean_m4"].mean()  # Temperatura medida
    otras = tr.groupby("COD_MUN_N")[["prec_total_m4", "flujo_m4", "lpob"]].mean()  # Lluvia, flujo y población
    mm = mm.set_index("COD_MUN_N").join(clima).join(otras).reset_index()  # Une
    W = sp.pesos_knn(D, 6)  # Matriz de pesos elegida
    sens = []  # Sensibilidad del I de Moran
    for nombre, Wk in [("k = 4", sp.pesos_knn(D, 4)), ("k = 6", W), ("k = 8", sp.pesos_knn(D, 8)), ("k = 12", sp.pesos_knn(D, 12)),
                       ("banda 50 km", sp.pesos_banda(D, 50)), ("banda 100 km", sp.pesos_banda(D, 100)), ("banda 200 km", sp.pesos_banda(D, 200))]:
        r = sp.moran(mm["log_incidencia"].to_numpy(), Wk, 499)  # Moran
        sens.append({"matriz": nombre, "sin vecinos": int((Wk.sum(axis=1) == 0).sum()), "I": r["I"], "z": r["z"], "p": r["p"]})
    glob = {v: sp.moran(mm[v].to_numpy(), W, 999) for v in ["log_incidencia", "prevalencia"]}  # Moran global
    Ii, p_loc, cuad = sp.lisa(mm["log_incidencia"].to_numpy(), W, 999)  # LISA
    mm["LISA"] = np.where(p_loc < 0.05, cuad, "No significativo")  # Clasificación
    mm["Gi"] = sp.getis_ord(mm["log_incidencia"].to_numpy(), D, 6)  # Getis-Ord
    db = DBSCAN(eps=30 / sp.RADIO_TIERRA, min_samples=5, metric="haversine", algorithm="ball_tree").fit(np.radians(m[["latitud", "longitud"]]))  # DBSCAN
    mm["dbscan"] = ["ruido" if l == -1 else f"C{l}" for l in db.labels_]  # Etiquetas
    dist = np.array([10, 25, 50, 75, 100, 150, 200, 300])  # Escalas de Ripley
    L, lo, hi = sp.ripley_l(xy, dist, 99)  # Función L
    iu = np.triu_indices(len(m), 1)  # Pares
    li = mm["log_incidencia"].to_numpy()  # Incidencia
    g = 0.5 * (li[:, None] - li[None, :])[iu] ** 2  # Semivarianzas
    clase = np.digitize(D[iu], np.arange(0, 820, 40))  # Intervalos de 40 km
    vario = pd.DataFrame({"d": D[iu], "g": g, "c": clase}).groupby("c").agg(distancia=("d", "mean"), semivarianza=("g", "mean"), pares=("g", "size"))  # Semivariograma
    todos = pd.read_csv(d.RUTA_DIVIPOLA, dtype=str)  # Municipios del país
    for c in ("latitud", "longitud"):  # Coma decimal
        todos[c] = todos[c].str.replace(",", ".").astype(float)  # A número
    fuera = todos[~todos["cod_mpio"].astype(int).isin(m["COD_MUN_N"])][["latitud", "longitud"]]  # Municipios fuera del panel
    het = tr.groupby("region").agg(municipios=("COD_MUN_N", "nunique"), prevalencia=(cf.TARGET, "mean"))  # Heterogeneidad
    for v in ["lc_m4", "oni_l8", "temp_anom_m8", "prec_total_m8"]:  # AUC por región
        het[f"AUC {v}"] = [roc_auc_score(tr.loc[tr["region"] == r, cf.TARGET], tr.loc[tr["region"] == r, v]) for r in het.index]
    regiones = sorted(tr["region"].unique())  # Regiones
    kw = es.kruskal([mm.loc[mm["region"] == r, "log_incidencia"] for r in regiones])  # Kruskal-Wallis
    dep = tr.groupby("depto").agg(casos=("casos_totales", "sum"))  # Nivel departamento
    dep["pob"] = tr.groupby(["depto", "COD_MUN_N"])["poblacion"].mean().groupby("depto").sum()  # Población
    dep["li"] = np.log1p(dep["casos"] / dep["pob"] / 7 * 1e5)  # Incidencia
    t_dep = tr[tr["temp_missing"] == 0].groupby("depto")["temp_mean_m4"].mean()  # Temperatura departamental
    maup = {"municipios": stats.spearmanr(clima, mm.set_index("COD_MUN_N").loc[clima.index, "log_incidencia"])[0], "n_mun": len(clima),
            "departamentos": stats.spearmanr(t_dep, dep.loc[t_dep.index, "li"])[0], "n_dep": len(t_dep)}
    grados = np.sqrt((lat[:, None] - lat[None, :]) ** 2 + (lon[:, None] - lon[None, :]) ** 2) * 111.32  # Euclidiana sobre grados
    err = np.abs(grados[iu] - D[iu]) / D[iu]  # Error relativo
    return {"mapa": mm, "fuera": fuera, "clark_evans": sp.clark_evans(xy), "ripley": {"d": dist, "L": L, "lo": lo, "hi": hi},
            "sensibilidad": pd.DataFrame(sens).set_index("matriz"), "moran": glob, "fdr_lisa": int((es.fdr_bh(p_loc) < 0.05).sum()),
            "vario": vario, "heterogeneidad": het, "kruskal": kw, "maup": maup, "error_grados": (float(np.median(err)), float(err.max())),
            "k6": np.sort(D, axis=1)[:, 6]}


def espacio_temporal(eda, m):
    e = eda[eda["ANO"] >= 2015]  # Periodo con canal endémico
    D = sp.haversine(m["latitud"].to_numpy(), m["longitud"].to_numpy())  # Distancias
    W = sp.pesos_knn(D, 6)  # Pesos
    anual = e.groupby(["ANO", "COD_MUN_N"]).agg(casos=("casos_totales", "sum"), pob=("poblacion", "mean"), brote=("brote", "mean")).reset_index()  # Año × municipio
    anual["log_incidencia"] = np.log1p(anual["casos"] / anual["pob"] * 1e5)  # Incidencia anual
    anual = anual.merge(m, on="COD_MUN_N")  # Coordenadas
    reg = e.groupby(["week_start", "region"])["brote"].mean().unstack()  # Brote por región y semana
    anios = sorted(anual["ANO"].unique())  # Años
    gi = pd.DataFrame({a: sp.getis_ord(anual[anual["ANO"] == a].set_index("COD_MUN_N").loc[m["COD_MUN_N"], "log_incidencia"].to_numpy(), D, 6) for a in anios},
                      index=m["COD_MUN_N"])  # Gi* por año
    filas = []  # Clasificación de focos
    for mun, z in gi.iterrows():  # Cada municipio
        _, zmk, pmk = es.mann_kendall(z.to_numpy())  # Tendencia
        cal = z > 1.96  # Años como foco
        n = int(cal.sum())  # Conteo
        clase = ("Persistente" if n >= 0.8 * len(anios) else "Emergente / intensificándose" if cal.iloc[-1] and pmk < 0.1 and zmk > 0
                 else "Decreciente" if n > 0 and pmk < 0.1 and zmk < 0 else "Esporádico" if n > 0 else "Nunca foco")  # Categoría
        filas.append({"COD_MUN_N": mun, "categoría": clase, "años como foco": n})
    focos = pd.DataFrame(filas).merge(m, on="COD_MUN_N")  # Une coordenadas
    b = e[["COD_MUN_N", "time_idx", "brote"]].sort_values(["COD_MUN_N", "time_idx"])  # Brotes
    b["inicio"] = (b["brote"] == 1) & (b.groupby("COD_MUN_N")["brote"].shift(1) == 0)  # Inicio de episodio
    ev = b[b["inicio"]].merge(m, on="COD_MUN_N")  # Eventos
    xy_ev = sp.proyectar(ev["latitud"].to_numpy(), ev["longitud"].to_numpy())  # Coordenadas planas
    knox = pd.DataFrame([{"distancia (km)": ds, "tiempo (semanas)": dt, **sp.knox(xy_ev, ev["time_idx"].to_numpy(), ds, dt, 999)}
                         for ds, dt in [(25, 2), (50, 4), (100, 4), (100, 8), (200, 8)]])  # Prueba de Knox
    piv = e.pivot_table(index="time_idx", columns="COD_MUN_N", values="lc").loc[:, m["COD_MUN_N"]]  # Semana × municipio
    anom = piv - piv.mean()  # Centrado por municipio
    I_sem = []  # Moran semanal
    for _, fila in anom.iterrows():  # Cada semana
        z = fila.to_numpy() - fila.mean()  # Desviaciones
        I_sem.append((z @ W @ z) / (z @ z) if z @ z > 0 else np.nan)  # I de Moran
    fechas = e.drop_duplicates("time_idx").set_index("time_idx").loc[anom.index, "week_start"]  # Fechas
    rezago = anom @ pd.DataFrame(W, index=m["COD_MUN_N"], columns=m["COD_MUN_N"]).T  # Rezago espacial
    cruz = pd.DataFrame([{"k": k, "vecinos": np.corrcoef(rezago.iloc[: len(anom) - k].to_numpy().ravel(), anom.shift(-k).iloc[: len(anom) - k].to_numpy().ravel())[0, 1],
                          "propio": np.corrcoef(anom.iloc[: len(anom) - k].to_numpy().ravel(), anom.shift(-k).iloc[: len(anom) - k].to_numpy().ravel())[0, 1]}
                         for k in range(13)]).set_index("k")  # Correlación espacio-temporal
    return {"anual": anual, "regiones": reg, "corr_regiones": reg.corr(), "focos": focos, "knox": knox, "n_eventos": len(ev),
            "moran_semanal": pd.Series(I_sem, index=fechas.values), "cruzada": cruz}


# ---------------------------------------------------------------------------
# 6. Modelos base: regresión logística y SVM lineal
# ---------------------------------------------------------------------------
def pipeline_svm():  # Pipeline de la máquina de vectores de soporte lineal
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler()),  # Mismo preprocesamiento
                     ("clf", LinearSVC(dual=False, max_iter=5000))])  # Bisagra L2 con penalización L2 (forma primal)


def ajustar_svm(tr):  # GridSearchCV de la SVM lineal con la validación purgada
    grid = GridSearchCV(pipeline_svm(), {"clf__C": [0.0001, 0.001, 0.01, 0.1, 1.0], "clf__class_weight": [None, "balanced"]},
                        cv=mb.ValidacionAnualPurgada(), n_jobs=-1, scoring={"auc_pr": "average_precision", "auc_roc": "roc_auc"},
                        refit="auc_pr", return_train_score=True)  # Mismo criterio que la regresión logística
    grid.fit(tr[cf.FEATURES], tr[cf.TARGET], groups=mb.grupos(tr))  # Ajuste
    return grid  # Resultado


def tabla_cv(grid):  # Resultados de la validación cruzada en una tabla
    r = pd.DataFrame(grid.cv_results_)  # Resultados
    t = pd.DataFrame({"C": r["param_clf__C"].astype(float), "class_weight": r["param_clf__class_weight"].astype(str),
                      "AUC-PR": r["mean_test_auc_pr"], "AUC-PR desv.": r["std_test_auc_pr"], "AUC-ROC": r["mean_test_auc_roc"],
                      "AUC-ROC entrenamiento": r["mean_train_auc_roc"]})  # Tabla
    pliegues = pd.DataFrame({"AUC-PR": [r.loc[grid.best_index_, f"split{i}_test_auc_pr"] for i in range(4)],  # Mejor combinación por pliegue
                             "AUC-ROC": [r.loc[grid.best_index_, f"split{i}_test_auc_roc"] for i in range(4)]}, index=[2018, 2019, 2020, 2021])
    return t.sort_values("AUC-PR", ascending=False), pliegues  # Tablas


def modelos(tr, va, te, m):
    t0 = time.time()  # Cronómetro
    g_lr = mb.ajustar_base(tr)  # Regresión logística
    t_lr = time.time() - t0  # Tiempo
    t0 = time.time()  # Cronómetro
    g_svm = ajustar_svm(tr)  # SVM lineal
    t_svm = time.time() - t0  # Tiempo
    lr = g_lr.best_estimator_  # Pipeline logístico
    svm = g_svm.best_estimator_  # Pipeline SVM
    s_va, s_te, s_tr = svm.decision_function(va[cf.FEATURES]), svm.decision_function(te[cf.FEATURES]), svm.decision_function(tr[cf.FEATURES])  # Márgenes
    platt = LogisticRegression().fit(s_va.reshape(-1, 1), va[cf.TARGET])  # Escalado de Platt ajustado en la validación 2022
    prob = lambda s: platt.predict_proba(s.reshape(-1, 1))[:, 1]  # Margen → probabilidad
    P = {"va": {"Regresión logística": lr.predict_proba(va[cf.FEATURES])[:, 1], "SVM lineal": prob(s_va)},  # Probabilidades
         "te": {"Regresión logística": lr.predict_proba(te[cf.FEATURES])[:, 1], "SVM lineal": prob(s_te)},
         "tr": {"Regresión logística": lr.predict_proba(tr[cf.FEATURES])[:, 1], "SVM lineal": prob(s_tr)}}
    lb_va, lb_te = mb.lineas_base(tr, va), mb.lineas_base(tr, te)  # Líneas base
    P["va"].update(lb_va)  # Agrega líneas base
    P["te"].update(lb_te)  # Agrega líneas base
    umbrales = {"Regresión logística": mb.umbral_f1(va[cf.TARGET], P["va"]["Regresión logística"]), "SVM lineal": mb.umbral_f1(va[cf.TARGET], P["va"]["SVM lineal"]),
                "Dummy (prior)": 0.5, "Persistencia": 1.0, "Persistencia (8 sem.)": mb.umbral_f1(va[cf.TARGET], lb_va["Persistencia (8 sem.)"]), "Ingenuo estacional": 1.0}
    ic = {n: mb.bootstrap_metricas(te[cf.TARGET], P["te"][n], umbrales[n], te["COD_MUN_N"], B=1000) for n in P["te"]}  # IC por municipio
    difs = []  # Diferencias pareadas
    for a, b in [("Regresión logística", "Persistencia (8 sem.)"), ("SVM lineal", "Persistencia (8 sem.)"), ("Regresión logística", "Persistencia"),
                 ("SVM lineal", "Regresión logística")]:  # Comparaciones
        for nom, met in [("AUC-ROC", roc_auc_score), ("AUC-PR", average_precision_score)]:  # Métricas
            dif, lo, hi, p = mb.bootstrap_diferencia(te[cf.TARGET], P["te"][a], P["te"][b], te["COD_MUN_N"], metrica=met, B=1000)  # Bootstrap
            difs.append({"comparación": f"{a} − {b}", "métrica": nom, "diferencia": dif, "IC 2,5 %": lo, "IC 97,5 %": hi, "p": p})
    D = sp.haversine(m["latitud"].to_numpy(), m["longitud"].to_numpy())  # Distancias
    diag = {}  # Diagnósticos por modelo
    for n in ["Regresión logística", "SVM lineal"]:  # Modelos base
        p = P["te"][n]  # Probabilidades en prueba
        resid = (te[cf.TARGET] - p).to_numpy()  # Residuos
        acf_r, _ = mb.acf_residuos(te, resid, 12)  # ACF de residuos
        r_mun = te.assign(r=resid).groupby("COD_MUN_N")["r"].mean().loc[m["COD_MUN_N"]]  # Residuo medio por municipio
        f_te, m_te, ece_te = mb.calibracion(te[cf.TARGET].to_numpy(), p)  # Calibración en prueba
        f_va, m_va, ece_va = mb.calibracion(va[cf.TARGET].to_numpy(), P["va"][n])  # Calibración en validación
        diag[n] = {"acf": acf_r, "moran": sp.moran(r_mun.to_numpy(), sp.pesos_knn(D, 6), 999), "residuo_mun": r_mun.to_numpy(),
                   "calibracion": {"te": (m_te, f_te, ece_te), "va": (m_va, f_va, ece_va)},
                   "brecha": pd.DataFrame({c: mb.metricas(yy, pp, umbrales[n]) for c, yy, pp in [("Entrenamiento", tr[cf.TARGET], P["tr"][n]),
                                                                                                 ("Validación 2022", va[cf.TARGET], P["va"][n]),
                                                                                                 ("Prueba 2023–24", te[cf.TARGET], p)]}).T}
    curva = mb.curva_aprendizaje(tr, va, C=g_lr.best_params_["clf__C"], cw=g_lr.best_params_["clf__class_weight"])  # Curva de aprendizaje
    coef_lr = mb.coeficientes_bootstrap(tr, g_lr.best_params_["clf__C"], g_lr.best_params_["clf__class_weight"], B=200)  # Coeficientes con IC
    coef_svm = pd.Series(svm.named_steps["clf"].coef_[0], index=cf.FEATURES)  # Pesos de la SVM
    mun_xy = m.set_index("COD_MUN_N")[["latitud", "longitud"]]  # Coordenadas
    bloques = []  # Validación por bloques espaciales
    for k, (i_tr, i_ev) in enumerate(GroupKFold(5).split(tr, groups=tr["depto"])):  # Bloques de departamentos
        deptos = set(tr.iloc[i_ev]["depto"])  # Departamentos evaluados
        m_ev, m_tr = tr.iloc[i_ev]["COD_MUN_N"].unique(), tr.iloc[i_tr]["COD_MUN_N"].unique()  # Municipios
        Dx = sp.haversine(np.r_[mun_xy.loc[m_tr, "latitud"], mun_xy.loc[m_ev, "latitud"]], np.r_[mun_xy.loc[m_tr, "longitud"], mun_xy.loc[m_ev, "longitud"]])[: len(m_tr), len(m_tr):]
        ent = tr[tr["COD_MUN_N"].isin(m_tr[Dx.min(axis=1) > 50])]  # Zona de separación de 50 km
        fila = {"bloque": k + 1, "departamentos": len(deptos), "municipios evaluados": len(m_ev), "excluidos por 50 km": int((Dx.min(axis=1) <= 50).sum())}
        for nombre, pipe in [("LR", mb.pipeline_base().set_params(clf__C=g_lr.best_params_["clf__C"], clf__class_weight=g_lr.best_params_["clf__class_weight"])),
                             ("SVM", pipeline_svm().set_params(clf__C=g_svm.best_params_["clf__C"], clf__class_weight=g_svm.best_params_["clf__class_weight"]))]:
            mdl = pipe.fit(ent[cf.FEATURES], ent[cf.TARGET])  # Ajuste sin el bloque
            f = (lambda X: mdl.predict_proba(X)[:, 1]) if nombre == "LR" else mdl.decision_function  # Puntaje
            ev1, ev2 = tr.iloc[i_ev], te[te["depto"].isin(deptos)]  # Zonas nuevas (mismo periodo) y zonas nuevas + futuro
            fila[f"{nombre} zonas nuevas"] = roc_auc_score(ev1[cf.TARGET], f(ev1[cf.FEATURES]))  # Esquema 2
            fila[f"{nombre} zonas nuevas + futuro"] = roc_auc_score(ev2[cf.TARGET], f(ev2[cf.FEATURES]))  # Esquema 3
        bloques.append(fila)
    sc = lr.named_steps["scaler"]  # Estandarizador del modelo logístico (para el simulador)
    ref = te.loc[te.groupby("COD_MUN_N")["casos_totales"].idxmax()]  # Semana de máximo de casos de cada municipio en la prueba
    return {"cv": {"Regresión logística": tabla_cv(g_lr), "SVM lineal": tabla_cv(g_svm)},
            "mejores": {"Regresión logística": g_lr.best_params_, "SVM lineal": g_svm.best_params_}, "tiempos": {"Regresión logística": t_lr, "SVM lineal": t_svm},
            "y_va": va[cf.TARGET].to_numpy(), "y_te": te[cf.TARGET].to_numpy(), "P_va": P["va"], "P_te": P["te"], "umbrales": umbrales,
            "ic": ic, "diferencias": pd.DataFrame(difs), "diagnosticos": diag, "curva": curva, "coef_lr": coef_lr, "coef_svm": coef_svm,
            "bloques": pd.DataFrame(bloques).set_index("bloque"), "platt": (float(platt.coef_[0, 0]), float(platt.intercept_[0])),
            "te_mun": te["COD_MUN_N"].to_numpy(), "te_semanas": int(te["time_idx"].nunique()),
            "simulador": {"media": sc.mean_, "escala": sc.scale_, "coef": lr.named_steps["clf"].coef_[0], "intercepto": float(lr.named_steps["clf"].intercept_[0]),
                          "umbral": umbrales["Regresión logística"],
                          "filas": ref[["COD_MUN_N", "municipio", "departamento", "week_start", "casos_totales", cf.TARGET] + cf.FEATURES].reset_index(drop=True)}}


def main():  # Calcula y guarda todos los artefactos
    t = time.time()  # Cronómetro
    raw = pd.read_csv(cf.RUTA_DATOS, parse_dates=["week_start"])  # Archivo original
    df = d.panel()  # Panel con objetivo, variables y periodo
    mod = d.modelado(df)  # Filas aptas para modelar
    tr, va, te = d.particion(mod)  # Particiones purgadas
    eda = d.entrenamiento_eda(df)  # Semanas t de 2010–2021
    m = d.municipios()  # Centroides
    A = {}  # Artefactos
    for nombre, funcion, args in [("contexto", contexto, (raw, df, tr, va, te, m)), ("calidad", calidad, (raw, eda, m)), ("objetivo", objetivo, (eda, tr, m)),
                                  ("univariado", univariado, (eda, tr)), ("bivariado", bivariado, (tr,)), ("multivariado", multivariado, (tr, m)),
                                  ("fuga", fuga, (tr,)), ("temporal", temporal, (eda, tr, va)), ("espacial", espacial, (tr, m)),
                                  ("espacio_temporal", espacio_temporal, (eda, m)), ("modelos", modelos, (tr, va, te, m))]:  # Secciones
        t0 = time.time()  # Cronómetro de la sección
        A[nombre] = funcion(*args)  # Calcula
        print(f"  {nombre}: {time.time() - t0:.0f} s")  # Progreso
    A["municipios"] = m  # Tabla de municipios
    joblib.dump(A, DESTINO, compress=3)  # Guarda comprimido
    print(f"Artefactos guardados en {DESTINO} ({DESTINO.stat().st_size / 1e6:.1f} MB) en {time.time() - t:.0f} s")  # Resumen


if __name__ == "__main__":  # Ejecución directa
    main()  # Calcula
