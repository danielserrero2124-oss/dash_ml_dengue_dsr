"""Evaluación en la prueba: métricas de clasificación y regresión, calibración y recalibración, y pruebas sobre residuos."""

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from scipy import stats  # Pruebas
from sklearn.isotonic import IsotonicRegression  # Recalibración isotónica
from sklearn.linear_model import LogisticRegression  # Escalado de Platt
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,  # Métricas
                             mean_absolute_error, mean_squared_error, precision_recall_curve, precision_score, r2_score,
                             recall_score, roc_auc_score)
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white  # Ljung-Box y White
from statsmodels.tsa.stattools import bds  # Prueba BDS de independencia


def umbral_f1(y, p):
    """Umbral que maximiza F1 (se elige con la validación 2022, nunca con la prueba)."""
    prec, rec, thr = precision_recall_curve(y, p)  # Curva precisión-recall
    f1 = 2 * prec * rec / (prec + rec + 1e-12)  # F1 por umbral
    return float(thr[int(np.argmax(f1[:-1]))])  # Mejor umbral


def metricas_clf(y, p, umbral):
    """Accuracy, Precision, Recall, F1, AUC-ROC, AUC-PR y Brier (si p es una probabilidad) con un umbral dado."""
    pred = (np.asarray(p) >= umbral).astype(int)  # Alertas
    m = {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0), "Recall": recall_score(y, pred),
         "F1": f1_score(y, pred), "AUC-ROC": roc_auc_score(y, p), "AUC-PR": average_precision_score(y, p)}  # Métricas
    if np.min(p) >= 0 and np.max(p) <= 1:  # Probabilidad
        m["Brier"] = brier_score_loss(y, p)  # Error cuadrático de la probabilidad
    return m  # Diccionario


def matriz(y, p, umbral):
    """Matriz de confusión [[VN, FP], [FN, VP]]."""
    return confusion_matrix(y, (np.asarray(p) >= umbral).astype(int), labels=[0, 1])  # Matriz


def ece(y, p, bins=10):
    """Error de calibración esperado sobre deciles de probabilidad: Σ (n_b / n) · |ȳ_b − p̄_b|."""
    y, p = np.asarray(y), np.asarray(p)  # Vectores
    bordes = np.quantile(p, np.linspace(0, 1, bins + 1))  # Deciles
    idx = np.clip(np.searchsorted(bordes, p, side="right") - 1, 0, bins - 1)  # Decil de cada fila
    return float(sum(np.mean(idx == b) * abs(y[idx == b].mean() - p[idx == b].mean()) for b in range(bins) if np.any(idx == b)))  # ECE


def confiabilidad(y, p, bins=10):
    """Puntos del diagrama de confiabilidad: probabilidad media y frecuencia observada por decil."""
    t = pd.DataFrame({"y": y, "p": p})  # Tabla
    t["b"] = pd.qcut(t["p"].rank(method="first"), bins, labels=False)  # Deciles
    return t.groupby("b").agg(predicha=("p", "mean"), observada=("y", "mean"))  # Medias por decil


def recalibrar(p_va, y_va, p_te):
    """Recalibra con la validación 2022 y devuelve las probabilidades de la prueba con Platt y con regresión isotónica.

    * Platt: p̃ = σ(a · logit(p) + b) (o σ(a · margen + b) para la SVM); corrige sesgos monótonos con 2 parámetros.
    * Isotónica: función escalonada no decreciente; más flexible, necesita más datos y puede sobreajustar.
    """
    def a_logit(p):  # Convierte probabilidades a logit (o deja el margen de la SVM)
        p = np.asarray(p, float)  # Vector
        return np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6))) if p.min() >= 0 and p.max() <= 1 else p  # Logit
    platt = LogisticRegression().fit(a_logit(p_va).reshape(-1, 1), y_va)  # Platt
    iso = IsotonicRegression(out_of_bounds="clip").fit(p_va, y_va)  # Isotónica
    return platt.predict_proba(a_logit(p_te).reshape(-1, 1))[:, 1], iso.predict(p_te)  # Prueba recalibrada


def metricas_reg(y, f):
    """RMSE, MAE y R² de un regresor."""
    return {"RMSE": float(np.sqrt(mean_squared_error(y, f))), "MAE": float(mean_absolute_error(y, f)), "R2": float(r2_score(y, f))}  # Métricas


def pruebas_residuos(df, y, f, n_white=5000, semilla=42):
    """Pruebas sobre los residuos de un regresor en la prueba.

    * White (homocedasticidad): regresión auxiliar de e² sobre el ajustado, su cuadrado y la temperatura (muestra).
    * BDS (independencia, Brock-Dechert-Scheinkman): sobre la serie semanal del residuo medio nacional.
    * Ljung-Box (autocorrelación serial): misma serie semanal, rezagos 1, 4 y 8; y proporción de municipios con p < 0,05.
    * Jarque-Bera (normalidad).
    """
    e = np.asarray(y) - np.asarray(f)  # Residuos
    rng = np.random.default_rng(semilla)  # Muestra para White (n grande vuelve todo significativo y es costoso)
    i = rng.choice(len(e), min(n_white, len(e)), replace=False)  # Índices
    exog = np.column_stack([np.ones(len(i)), np.asarray(f)[i], df["temp_mean_m4"].to_numpy()[i]])  # Constante, ajustado y temperatura
    lm, p_white, _, _ = het_white(e[i], exog)  # Prueba de White
    semanal = pd.Series(e, index=df["time_idx"].to_numpy()).groupby(level=0).mean()  # Residuo medio por semana
    bds_est, p_bds = bds(semanal.to_numpy(), max_dim=2)  # BDS con dimensión de inmersión 2
    lb = acorr_ljungbox(semanal.to_numpy(), lags=[1, 4, 8], return_df=True)  # Ljung-Box
    p_mun = [acorr_ljungbox(g.to_numpy(), lags=[4], return_df=True)["lb_pvalue"].iloc[0]  # Ljung-Box por municipio
             for _, g in pd.Series(e, index=df["COD_MUN_N"].to_numpy()).groupby(level=0) if g.std() > 0]
    jb, p_jb = stats.jarque_bera(e)  # Normalidad
    return {"White LM": float(lm), "p White": float(p_white), "BDS": float(np.ravel(bds_est)[0]), "p BDS": float(np.ravel(p_bds)[0]),
            "p Ljung-Box (1)": float(lb["lb_pvalue"].iloc[0]), "p Ljung-Box (4)": float(lb["lb_pvalue"].iloc[1]), "p Ljung-Box (8)": float(lb["lb_pvalue"].iloc[2]),
            "municipios con Ljung-Box p < 0,05": float(np.mean(np.array(p_mun) < 0.05)), "Jarque-Bera": float(jb), "p Jarque-Bera": float(p_jb),
            "asimetría": float(stats.skew(e)), "curtosis": float(stats.kurtosis(e))}
