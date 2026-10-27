"""Modelo base del Primer Entregable: regresión logística dentro de un Pipeline, comparada con líneas base triviales.

* Validación cruzada hacia adelante por años CON PURGA: en cada pliegue el entrenamiento solo usa filas cuya semana
  objetivo (t + 4) es anterior al año de validación, y la validación solo usa filas cuya semana t y semana objetivo
  caen en ese año. Así ninguna etiqueta del año validado se usa al entrenar (intervalo de separación de 4 semanas).
* Intervalos de confianza por bootstrap de MUNICIPIOS (conglomerados): las semanas de un municipio no son independientes.
"""

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from sklearn.calibration import calibration_curve  # Curva de calibración
from sklearn.dummy import DummyClassifier  # Línea base trivial
from sklearn.impute import SimpleImputer  # Imputación
from sklearn.linear_model import LogisticRegression  # Modelo base
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score,  # Métricas
                             matthews_corrcoef, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV  # Búsqueda de hiperparámetros
from sklearn.pipeline import Pipeline  # Encadena preprocesamiento y modelo
from sklearn.preprocessing import StandardScaler  # Estandarización

from src import config as cf  # Configuración del proyecto


class ValidacionAnualPurgada:  # Validación cruzada hacia adelante por años con purga de etiquetas
    """groups = 10 000 · año de la semana t + año de la semana objetivo (se codifican juntos para GridSearchCV)."""

    def __init__(self, anios=(2018, 2019, 2020, 2021)):  # Años de validación
        self.anios = anios  # Guarda los años

    def split(self, X, y=None, groups=None):  # Índices de cada pliegue
        g = np.asarray(groups)  # Códigos
        anio_t, anio_obj = g // 10000, g % 10000  # Decodifica los dos años
        for a in self.anios:  # Un pliegue por año
            entrena = np.flatnonzero((anio_t < a) & (anio_obj < a))  # Etiquetas anteriores al año a (purga)
            valida = np.flatnonzero((anio_t == a) & (anio_obj == a))  # Semana t y objetivo dentro del año a
            yield entrena, valida  # Pliegue

    def get_n_splits(self, X=None, y=None, groups=None):  # Número de pliegues
        return len(self.anios)  # Cuatro


def grupos(df):  # Códigos de año para la validación purgada
    return (df["ANO"].astype(int) * 10000 + df["anio_obj"].astype(int)).to_numpy()  # año_t · 10 000 + año_objetivo


def pipeline_base():  # Pipeline del modelo base: imputación + estandarización + regresión logística
    return Pipeline([("imputer", SimpleImputer(strategy="median")),  # Mediana del entrenamiento (protege ante nulos)
                     ("scaler", StandardScaler()),  # Coeficientes comparables y regularización justa
                     ("clf", LogisticRegression(max_iter=5000))])  # Regresión logística con penalización L2


MALLA = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0],  # Inverso de la fuerza de regularización
         "clf__class_weight": [None, "balanced"]}  # Sin ponderar o ponderando el desbalance


def ajustar_base(tr):  # GridSearchCV del modelo base con la validación purgada
    grid = GridSearchCV(pipeline_base(), MALLA, cv=ValidacionAnualPurgada(), n_jobs=-1,  # Todas las combinaciones
                        scoring={"auc_pr": "average_precision", "auc_roc": "roc_auc", "neg_brier": "neg_brier_score"},  # Métricas
                        refit="auc_pr", return_train_score=True)  # Elige por AUC-PR (desbalance) y reentrena
    grid.fit(tr[cf.FEATURES], tr[cf.TARGET], groups=grupos(tr))  # Ajuste
    return grid  # Resultado


def lineas_base(tr, df_eval):  # Puntajes de las líneas base triviales para las filas de df_eval
    """Dummy (proporción de entrenamiento), persistencia (¿hay brote ahora?) e ingenuo estacional (¿hubo brote en la
    misma semana objetivo del año anterior?). Todas usan solo información disponible en la semana t."""
    dummy = DummyClassifier(strategy="prior").fit(tr[cf.FEATURES], tr[cf.TARGET])  # Siempre la misma probabilidad
    return {
        "Dummy (prior)": dummy.predict_proba(df_eval[cf.FEATURES])[:, 1],  # Constante = prevalencia de entrenamiento
        "Persistencia": df_eval["brote_l0"].to_numpy(float),  # Brote en t → brote en t + 4
        "Persistencia (8 sem.)": df_eval["brote_m8"].to_numpy(float),  # Proporción de semanas en brote en las últimas 8
        "Ingenuo estacional": df_eval["brote_estacional"].to_numpy(float),  # Brote en t + 4 − 52 (hace un año)
    }


def umbral_f1(y, p):  # Umbral que maximiza F1 (se elige en validación, nunca en prueba)
    prec, rec, thr = precision_recall_curve(y, p)  # Curva precisión-recall
    f1 = 2 * prec * rec / (prec + rec + 1e-12)  # F1 en cada umbral
    return float(thr[int(np.argmax(f1[:-1]))])  # Umbral óptimo


def metricas(y, p, umbral):  # Métricas de clasificación con un umbral dado
    y, p = np.asarray(y), np.asarray(p, float)  # Vectores
    pred = (p >= umbral).astype(int)  # Predicción binaria
    unico = len(np.unique(p)) > 1  # Un puntaje constante no ordena (AUC = 0,5)
    return {"AUC-ROC": roc_auc_score(y, p) if unico else 0.5,  # Capacidad de ordenar el riesgo
            "AUC-PR": average_precision_score(y, p) if unico else float(np.mean(y)),  # Desempeño en la clase brote
            "Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),  # Exactitud y precisión
            "Recall": recall_score(y, pred, zero_division=0), "F1": f1_score(y, pred, zero_division=0),  # Sensibilidad y F1
            "MCC": matthews_corrcoef(y, pred), "Brier": brier_score_loss(y, np.clip(p, 0, 1))}  # Correlación de Matthews y Brier


def bootstrap_metricas(y, p, umbral, municipios, B=1000, semilla=cf.SEMILLA):  # IC 95 % por bootstrap de municipios
    y, p, municipios = np.asarray(y), np.asarray(p, float), np.asarray(municipios)  # Vectores
    rng = np.random.default_rng(semilla)  # Generador reproducible
    unicos = np.unique(municipios)  # Municipios
    filas = {m: np.flatnonzero(municipios == m) for m in unicos}  # Filas de cada municipio
    replicas = []  # Métricas de cada réplica
    for _ in range(B):  # Réplicas
        idx = np.concatenate([filas[m] for m in rng.choice(unicos, len(unicos))])  # Municipios con reemplazo
        replicas.append(metricas(y[idx], p[idx], umbral))  # Métricas de la réplica
    R = pd.DataFrame(replicas)  # Tabla de réplicas
    obs = metricas(y, p, umbral)  # Valor observado
    return pd.DataFrame({"valor": obs, "IC 2,5 %": R.quantile(0.025), "IC 97,5 %": R.quantile(0.975)})  # Resultado


def bootstrap_diferencia(y, p_a, p_b, municipios, metrica=roc_auc_score, B=1000, semilla=cf.SEMILLA):  # Diferencia pareada
    y, p_a, p_b, municipios = map(np.asarray, (y, p_a, p_b, municipios))  # Vectores
    rng = np.random.default_rng(semilla)  # Generador reproducible
    unicos = np.unique(municipios)  # Municipios
    filas = {m: np.flatnonzero(municipios == m) for m in unicos}  # Filas por municipio
    difs = np.empty(B)  # Diferencias
    for b in range(B):  # Réplicas
        idx = np.concatenate([filas[m] for m in rng.choice(unicos, len(unicos))])  # Remuestreo
        difs[b] = metrica(y[idx], p_a[idx]) - metrica(y[idx], p_b[idx])  # Diferencia
    dif = metrica(y, p_a) - metrica(y, p_b)  # Observada
    p = max(2 * min((difs <= 0).mean(), (difs >= 0).mean()), 1 / B)  # Valor p bilateral
    return float(dif), float(np.quantile(difs, 0.025)), float(np.quantile(difs, 0.975)), float(p)  # Resultado


def calibracion(y, p, bins=10):  # Curva de calibración y error de calibración esperado (ECE)
    frac, media = calibration_curve(y, p, n_bins=bins, strategy="quantile")  # Por deciles de probabilidad
    contenedores = np.quantile(p, np.linspace(0, 1, bins + 1))  # Bordes de los deciles
    idx = np.clip(np.searchsorted(contenedores, p, side="right") - 1, 0, bins - 1)  # Decil de cada fila
    ece = sum(np.mean(idx == b) * abs(np.mean(y[idx == b]) - np.mean(p[idx == b])) for b in range(bins) if np.any(idx == b))  # ECE
    return frac, media, float(ece)  # Fracción observada, probabilidad media y ECE


def acf_residuos(df, residuo, rezagos=12):  # ACF media de los residuos dentro de cada municipio
    acfs = []  # ACF de cada municipio
    for _, g in df.assign(r=residuo).sort_values("time_idx").groupby("COD_MUN_N"):  # Serie de residuos por municipio
        r = g["r"].to_numpy() - g["r"].mean()  # Centra
        v = r @ r  # Varianza (sin normalizar)
        if v > 0:  # Si hay variación
            acfs.append([r[:-k] @ r[k:] / v for k in range(1, rezagos + 1)])  # Autocorrelación en cada rezago
    return np.mean(acfs, axis=0), len(acfs)  # ACF promedio y número de municipios


def curva_aprendizaje(tr, va, fracciones=(0.1, 0.2, 0.35, 0.5, 0.75, 1.0), C=0.01, cw=None, semilla=cf.SEMILLA):
    """Entrena con fracciones crecientes de MUNICIPIOS del entrenamiento y evalúa en validación (2022)."""
    rng = np.random.default_rng(semilla)  # Generador reproducible
    muni = rng.permutation(tr["COD_MUN_N"].unique())  # Orden aleatorio de municipios
    filas = []  # Resultados
    for f in fracciones:  # Cada fracción
        sub = tr[tr["COD_MUN_N"].isin(muni[: int(round(f * len(muni)))])]  # Municipios incluidos
        m = pipeline_base().set_params(clf__C=C, clf__class_weight=cw).fit(sub[cf.FEATURES], sub[cf.TARGET])  # Ajuste
        filas.append({"fraccion": f, "filas": len(sub),  # Tamaño
                      "AUC-PR entrenamiento": average_precision_score(sub[cf.TARGET], m.predict_proba(sub[cf.FEATURES])[:, 1]),
                      "AUC-PR validación": average_precision_score(va[cf.TARGET], m.predict_proba(va[cf.FEATURES])[:, 1]),
                      "AUC-ROC entrenamiento": roc_auc_score(sub[cf.TARGET], m.predict_proba(sub[cf.FEATURES])[:, 1]),
                      "AUC-ROC validación": roc_auc_score(va[cf.TARGET], m.predict_proba(va[cf.FEATURES])[:, 1])})
    return pd.DataFrame(filas)  # Curva


def coeficientes_bootstrap(tr, C, cw, B=200, semilla=cf.SEMILLA):  # Coeficientes estandarizados con IC por municipios
    rng = np.random.default_rng(semilla)  # Generador reproducible
    unicos = tr["COD_MUN_N"].unique()  # Municipios
    grupos_filas = tr.groupby("COD_MUN_N").indices  # Filas por municipio
    base = pipeline_base().set_params(clf__C=C, clf__class_weight=cw).fit(tr[cf.FEATURES], tr[cf.TARGET])  # Ajuste completo
    coefs = []  # Coeficientes de cada réplica
    for _ in range(B):  # Réplicas
        idx = np.concatenate([grupos_filas[m] for m in rng.choice(unicos, len(unicos))])  # Municipios con reemplazo
        sub = tr.iloc[idx]  # Muestra
        m = pipeline_base().set_params(clf__C=C, clf__class_weight=cw).fit(sub[cf.FEATURES], sub[cf.TARGET])  # Reajuste
        coefs.append(m.named_steps["clf"].coef_[0])  # Coeficientes
    coefs = np.array(coefs)  # Matriz B × p
    return pd.DataFrame({"variable": cf.FEATURES, "coef": base.named_steps["clf"].coef_[0],  # Coeficiente por desviación estándar
                         "IC 2,5 %": np.quantile(coefs, 0.025, axis=0), "IC 97,5 %": np.quantile(coefs, 0.975, axis=0)})
