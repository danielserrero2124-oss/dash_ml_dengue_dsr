"""Funciones reutilizables de modelado: preprocesamiento, validación temporal, catálogo de modelos, GridSearchCV,
umbral de decisión, métricas y pruebas de significancia (DeLong y bootstrap por conglomerados).

Todo el preprocesamiento vive DENTRO de un Pipeline: GridSearchCV lo reajusta en cada pliegue con la parte de
entrenamiento del pliegue, así la validación no se contamina (sin fuga de datos).
"""

import numpy as np  # Cálculo numérico
from scipy import stats  # Distribución normal para DeLong
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # Modelos de ensamble recomendados
from sklearn.impute import SimpleImputer  # Imputación de valores faltantes
from sklearn.linear_model import LogisticRegression  # Regresión logística
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,  # Métricas
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV  # Búsqueda de hiperparámetros
from sklearn.neighbors import KNeighborsClassifier  # K vecinos más cercanos
from sklearn.pipeline import Pipeline  # Encadena preprocesamiento y modelo
from sklearn.preprocessing import StandardScaler  # Estandarización

from src import config as cf  # Configuración del proyecto


class ValidacionTemporalAnual:  # Validación cruzada «hacia adelante» por años (forward chaining)
    """Pliegue k: entrena con todos los años anteriores al año k y valida en el año k (2018, 2019, 2020 y 2021).

    Con datos semanales autocorrelacionados, mezclar semanas al azar entre entrenamiento y validación filtra
    información del futuro; esta validación respeta el orden del tiempo (Bergmeir y Benítez, 2012).
    """

    def __init__(self, anios_validacion=(2018, 2019, 2020, 2021)):  # Años que se usan como validación
        self.anios_validacion = anios_validacion  # Guarda los años

    def split(self, X, y=None, groups=None):  # Genera los índices de cada pliegue; groups = año de cada fila
        anios = np.asarray(groups)  # Año de cada fila
        for a in self.anios_validacion:  # Un pliegue por año de validación
            yield np.flatnonzero(anios < a), np.flatnonzero(anios == a)  # (entrenamiento: años previos, validación: año a)

    def get_n_splits(self, X=None, y=None, groups=None):  # Número de pliegues (requerido por GridSearchCV)
        return len(self.anios_validacion)  # Cuatro pliegues


def make_preprocessor():  # Preprocesamiento común: todas las variables son numéricas
    """Imputa con la mediana del pliegue de entrenamiento y estandariza (media 0, desviación 1)."""
    return Pipeline([("imputer", SimpleImputer(strategy="median")),  # Protege ante nulos en producción
                     ("scaler", StandardScaler())])  # Necesario para KNN (distancias) y la regresión logística (regularización)


def catalogo_modelos():  # Los cuatro modelos recomendados con sus mallas de hiperparámetros
    """Devuelve {nombre: (estimador, malla, usa_muestra)}."""
    return {
        "LogisticRegression": (
            LogisticRegression(max_iter=3000),  # Modelo lineal en el logit
            {"clf__C": [0.01, 0.1, 1.0],  # Inverso de la regularización L2
             "clf__class_weight": [None, "balanced"]},  # Sin ponderar vs. ponderación del desbalance
            False),
        "RandomForest": (
            RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=cf.SEMILLA),  # Bosque de 300 árboles
            {"clf__max_depth": [10, 16],  # Profundidad máxima
             "clf__min_samples_leaf": [20, 60],  # Mínimo de semanas por hoja: evita memorizar brotes aislados
             "clf__class_weight": [None, "balanced_subsample"]},  # Manejo del desbalance
            False),
        "KNN": (
            KNeighborsClassifier(n_jobs=1),  # Clasificación por vecindad
            {"clf__n_neighbors": [51, 151, 301],  # k grande: probabilidades estables
             "clf__weights": ["uniform", "distance"]},  # Voto igual o ponderado por distancia
            True),  # Escala mal con 134 000 filas: se entrena con una muestra estratificada
        "GradientBoosting": (
            GradientBoostingClassifier(subsample=0.8, random_state=cf.SEMILLA),  # Boosting de árboles con submuestreo
            {"clf__n_estimators": [200, 400],  # Número de árboles
             "clf__learning_rate": [0.05],  # Tasa de aprendizaje baja
             "clf__max_depth": [3, 4]},  # Árboles poco profundos
            False),
    }


def train_pipeline(X_train, y_train, anios, model, param_grid):  # Pipeline + GridSearchCV con validación temporal anual
    """Construye Pipeline(preprocesamiento + modelo) y lo ajusta con GridSearchCV y ValidacionTemporalAnual."""
    pipe = Pipeline([("prep", make_preprocessor()), ("clf", model)])  # El preprocesamiento queda dentro del Pipeline
    grid = GridSearchCV(pipe, param_grid, cv=ValidacionTemporalAnual(), n_jobs=-1,  # Prueba todas las combinaciones en paralelo
                        scoring={"auc_pr": "average_precision", "auc_roc": "roc_auc"},  # Registra ambas áreas bajo la curva
                        refit="auc_pr", return_train_score=True)  # Elige por AUC-PR y reentrena con todo el entrenamiento
    grid.fit(X_train, y_train, groups=anios)  # groups = año de cada fila (define los pliegues)
    return grid  # best_estimator_, best_params_, cv_results_


def umbral_max_f1(y_val, p_val):  # Umbral de decisión elegido con datos de VALIDACIÓN (2022), nunca con prueba
    prec, rec, thr = precision_recall_curve(y_val, p_val)  # Precisión y recall para cada umbral
    f1 = 2 * prec * rec / (prec + rec + 1e-9)  # F1 en cada umbral
    return float(thr[int(np.argmax(f1[:-1]))])  # El último punto de la curva no tiene umbral


def metricas(y, p, umbral):  # Métricas de un modelo con un umbral dado
    pred = (p >= umbral).astype(int)  # Predicción binaria
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()  # Matriz de confusión
    return {"auc_roc": float(roc_auc_score(y, p)), "auc_pr": float(average_precision_score(y, p)),  # Áreas bajo las curvas
            "precision": float(precision_score(y, pred, zero_division=0)), "recall": float(recall_score(y, pred, zero_division=0)),  # Precisión y recall
            "f1": float(f1_score(y, pred, zero_division=0)), "accuracy": float(accuracy_score(y, pred)),  # F1 y exactitud
            "matriz": [[int(tn), int(fp)], [int(fn), int(tp)]]}  # [[VN, FP], [FN, VP]]


# ---------------------------------------------------------------------------
# Prueba de DeLong (DeLong et al., 1988; algoritmo rápido de Sun y Xu, 2014)
# ---------------------------------------------------------------------------
def _rangos_medios(x):  # Rangos con empates promediados
    orden = np.argsort(x)  # Orden ascendente
    xs = x[orden]  # Valores ordenados
    n = len(x)  # Tamaño
    t = np.zeros(n)  # Rangos en el orden ordenado
    i = 0  # Inicio de un bloque de empates
    while i < n:  # Recorre los bloques
        j = i  # Fin del bloque
        while j < n and xs[j] == xs[i]:  # Avanza mientras haya empate
            j += 1  # Siguiente
        t[i:j] = 0.5 * (i + j - 1) + 1  # Rango medio (base 1)
        i = j  # Siguiente bloque
    r = np.empty(n)  # Rangos en el orden original
    r[orden] = t  # Deshace el orden
    return r  # Rangos medios


def _delong(y, preds):  # AUC y covarianza de DeLong de varios modelos sobre los mismos datos
    pos, neg = preds[:, y == 1], preds[:, y == 0]  # Puntajes de positivos y negativos
    m, n = pos.shape[1], neg.shape[1]  # Tamaños
    tx = np.array([_rangos_medios(r) for r in pos])  # Rangos entre positivos
    ty = np.array([_rangos_medios(r) for r in neg])  # Rangos entre negativos
    tz = np.array([_rangos_medios(np.r_[pos[k], neg[k]]) for k in range(preds.shape[0])])  # Rangos conjuntos
    auc = (tz[:, :m].sum(axis=1) / m - (m + 1) / 2) / n  # AUC (Mann-Whitney)
    v01 = (tz[:, :m] - tx) / n  # Componentes estructurales de positivos
    v10 = 1 - (tz[:, m:] - ty) / m  # Componentes estructurales de negativos
    return auc, np.atleast_2d(np.cov(v01) / m + np.cov(v10) / n)  # AUC y covarianza


def prueba_delong(y, p_a, p_b):  # Compara dos AUC pareados: devuelve (diferencia, IC 95 %, valor p)
    auc, S = _delong(np.asarray(y), np.vstack([p_a, p_b]))  # AUC y covarianza conjunta
    dif = float(auc[0] - auc[1])  # Diferencia
    ee = float(np.sqrt(max(S[0, 0] + S[1, 1] - 2 * S[0, 1], 1e-12)))  # Error estándar de la diferencia
    return dif, (dif - 1.96 * ee, dif + 1.96 * ee), float(2 * stats.norm.sf(abs(dif / ee)))  # Diferencia, IC y p bilateral


def ic_auc_delong(y, p):  # IC del 95 % de un AUC individual
    auc, S = _delong(np.asarray(y), np.atleast_2d(p))  # AUC y varianza
    ee = float(np.sqrt(S[0, 0]))  # Error estándar
    return float(auc[0] - 1.96 * ee), float(auc[0] + 1.96 * ee)  # Límites


def bootstrap_conglomerados(y, p_a, p_b, grupos, metrica=roc_auc_score, B=1000, semilla=cf.SEMILLA):  # Bootstrap por municipio
    """Remuestrea MUNICIPIOS completos (no filas): respeta la autocorrelación de las semanas de un mismo municipio.

    Devuelve (diferencia observada, IC 95 %, valor p bilateral) de metrica(p_a) − metrica(p_b).
    """
    rng = np.random.default_rng(semilla)  # Generador reproducible
    unicos = np.unique(grupos)  # Municipios
    filas = {g: np.flatnonzero(grupos == g) for g in unicos}  # Filas de cada municipio
    difs = np.empty(B)  # Diferencias bootstrap
    for b in range(B):  # Réplicas
        idx = np.concatenate([filas[g] for g in rng.choice(unicos, len(unicos))])  # Municipios con reemplazo
        difs[b] = metrica(y[idx], p_a[idx]) - metrica(y[idx], p_b[idx])  # Diferencia en la réplica
    dif = float(metrica(y, p_a) - metrica(y, p_b))  # Diferencia observada
    p = 2 * min((difs <= 0).mean(), (difs >= 0).mean())  # Valor p bilateral (percentil)
    return dif, (float(np.quantile(difs, 0.025)), float(np.quantile(difs, 0.975))), float(max(p, 1 / B))  # Resultado
