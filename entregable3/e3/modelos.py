"""Los 7 modelos de cada tarea, sus espacios de búsqueda y las 4 estrategias de balanceo.

Cada combinación se construye como un Pipeline de imbalanced-learn: imputación → estandarización → [remuestreo] → modelo.
El remuestreo (SMOTE o ADASYN) vive DENTRO del Pipeline, así que solo se aplica al ajustar con la parte de entrenamiento
de cada pliegue: las filas sintéticas nunca llegan a la validación ni a la prueba (sin fuga de datos).

Espacios de búsqueda: diccionario {hiperparámetro: (tipo, mínimo, máximo, ¿escala log?)} o (\"cat\", [opciones]).
"""

import numpy as np  # Cálculo numérico
from imblearn.over_sampling import ADASYN, SMOTE  # Sobremuestreo sintético
from imblearn.pipeline import Pipeline  # Pipeline que admite remuestreo
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # Envoltorios propios
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor  # Bosques aleatorios
from sklearn.impute import SimpleImputer  # Imputación
from sklearn.kernel_approximation import Nystroem  # Aproximación del núcleo RBF
from sklearn.linear_model import Lasso, LogisticRegression, Ridge  # Modelos lineales
from sklearn.naive_bayes import GaussianNB  # Naive Bayes
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor  # Vecinos más cercanos
from sklearn.preprocessing import StandardScaler  # Estandarización
from sklearn.svm import LinearSVC, LinearSVR  # Máquinas de vectores de soporte lineales
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor  # Árboles de decisión
from sklearn.utils.class_weight import compute_sample_weight  # Pesos balanceados
from xgboost import XGBClassifier, XGBRegressor  # Boosting de gradiente

MODELOS_CLF = ["KNN", "Naive Bayes", "Regresión logística", "Árbol de decisión", "Random Forest", "XGBoost", "SVM"]  # Etapa 1 (clasificación)
MODELOS_REG = ["KNN", "Ridge", "Lasso", "Árbol de decisión", "Random Forest", "XGBoost", "SVR"]  # Etapa 1 (regresión)
BALANCEOS = ["Sin balanceo", "SMOTE", "ADASYN", "class_weight balanced"]  # Etapa 2 (solo clasificación)

ESPACIOS_CLF = {  # Espacios de búsqueda realistas (los rangos se justifican en el informe)
    "KNN": {"n_neighbors": ("int", 5, 301, True), "weights": ("cat", ["uniform", "distance"])},
    "Naive Bayes": {"var_smoothing": ("float", 1e-12, 1e-2, True)},
    "Regresión logística": {"C": ("float", 1e-4, 10.0, True), "penalty": ("cat", ["l1", "l2"])},
    "Árbol de decisión": {"max_depth": ("int", 2, 20, False), "min_samples_leaf": ("int", 5, 500, True), "criterion": ("cat", ["gini", "entropy"])},
    "Random Forest": {"n_estimators": ("int", 50, 150, False), "max_depth": ("int", 4, 20, False), "min_samples_leaf": ("int", 5, 200, True),
                      "max_features": ("cat", ["sqrt", 0.5])},
    "XGBoost": {"n_estimators": ("int", 50, 400, False), "max_depth": ("int", 2, 8, False), "learning_rate": ("float", 0.01, 0.3, True),
                "subsample": ("float", 0.5, 1.0, False), "min_child_weight": ("float", 1.0, 50.0, True)},
    "SVM": {"C": ("float", 1e-3, 10.0, True), "gamma": ("float", 1e-3, 1.0, True)},
}
ESPACIOS_REG = {
    "KNN": ESPACIOS_CLF["KNN"],
    "Ridge": {"alpha": ("float", 1e-3, 1e3, True)},
    "Lasso": {"alpha": ("float", 1e-4, 1.0, True)},
    "Árbol de decisión": {"max_depth": ("int", 2, 20, False), "min_samples_leaf": ("int", 5, 500, True)},
    "Random Forest": {k: v for k, v in ESPACIOS_CLF["Random Forest"].items()},
    "XGBoost": ESPACIOS_CLF["XGBoost"],
    "SVR": {"C": ("float", 1e-3, 10.0, True), "gamma": ("float", 1e-3, 1.0, True), "epsilon": ("float", 0.01, 0.5, True)},
}


class PesoBalanceado(BaseEstimator, ClassifierMixin):
    """Envoltorio que entrena cualquier clasificador con pesos de muestra «balanced».

    Equivale a ``class_weight='balanced'`` para modelos que no tienen ese parámetro (Naive Bayes, XGBoost):
    cada fila de la clase c pesa n / (2 · n_c).

    Parameters
    ----------
    estimador : estimator
        Clasificador de scikit-learn o compatible que acepta ``sample_weight`` en ``fit``.
    """

    def __init__(self, estimador):  # Guarda el estimador base
        self.estimador = estimador  # Clasificador envuelto

    def fit(self, X, y):  # Ajusta con pesos balanceados
        self.est_ = clone(self.estimador).fit(X, y, sample_weight=compute_sample_weight("balanced", y))  # Pesos inversos a la frecuencia
        self.classes_ = self.est_.classes_  # Clases
        return self  # Encadenable

    def predict_proba(self, X):  # Probabilidades del estimador
        return self.est_.predict_proba(X)  # Delegación

    def predict(self, X):  # Clases predichas
        return self.est_.predict(X)  # Delegación

    def decision_function(self, X):  # Puntajes (si el estimador los tiene)
        return self.est_.decision_function(X)  # Delegación


class KNNBalanceado(BaseEstimator, ClassifierMixin):
    """KNN con corrección por la probabilidad a priori: el análogo de ``class_weight='balanced'`` para vecinos.

    KNN no admite pesos de muestra. Pesar cada clase por 1/π_c equivale a dividir la proporción de vecinos de cada
    clase por su prevalencia y renormalizar: p̃_c ∝ p_c / π_c (regla de Bayes con clases equiprobables).
    """

    def __init__(self, n_neighbors=5, weights="uniform"):  # Hiperparámetros de KNN
        self.n_neighbors = n_neighbors  # Vecinos
        self.weights = weights  # Ponderación

    def fit(self, X, y):  # Ajusta KNN y guarda las prevalencias
        self.knn_ = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights=self.weights, n_jobs=1).fit(X, y)  # KNN
        self.priori_ = np.bincount(y, minlength=2) / len(y)  # Prevalencia de cada clase
        self.classes_ = self.knn_.classes_  # Clases
        return self  # Encadenable

    def predict_proba(self, X):  # Probabilidades corregidas
        p = self.knn_.predict_proba(X) / self.priori_  # Divide por la prevalencia
        return p / p.sum(axis=1, keepdims=True)  # Renormaliza

    def predict(self, X):  # Clase de mayor probabilidad corregida
        return self.predict_proba(X).argmax(axis=1)  # Argmax


def _clasificador(nombre, p, balanceado, semilla):  # Instancia el clasificador con sus hiperparámetros
    cw = "balanced" if balanceado else None  # class_weight nativo
    if nombre == "KNN":  # Vecinos
        return KNNBalanceado(**p) if balanceado else KNeighborsClassifier(n_jobs=1, **p)
    if nombre == "Naive Bayes":  # Bayes ingenuo gaussiano
        return PesoBalanceado(GaussianNB(**p)) if balanceado else GaussianNB(**p)
    if nombre == "Regresión logística":  # L1 o L2 (liblinear resuelve ambas)
        return LogisticRegression(solver="liblinear", class_weight=cw, max_iter=2000, random_state=semilla, **p)
    if nombre == "Árbol de decisión":  # Árbol CART
        return DecisionTreeClassifier(class_weight=cw, random_state=semilla, **p)
    if nombre == "Random Forest":  # Bosque
        return RandomForestClassifier(class_weight=cw, n_jobs=1, random_state=semilla, **p)
    if nombre == "XGBoost":  # Boosting con histogramas
        xgb = XGBClassifier(tree_method="hist", n_jobs=1, random_state=semilla, eval_metric="logloss", **p)
        return PesoBalanceado(xgb) if balanceado else xgb
    if nombre == "SVM":  # Núcleo RBF aproximado (Nyström) + SVM lineal: evita el costo O(n²)–O(n³) del núcleo exacto
        svm = LinearSVC(C=p["C"], dual=False, class_weight=cw, max_iter=5000)  # SVM lineal en el espacio aproximado
        return Pipeline([("nystroem", Nystroem(kernel="rbf", gamma=p["gamma"], n_components=300, random_state=semilla)), ("svm", svm)])
    raise ValueError(nombre)  # Nombre desconocido


def _regresor(nombre, p, semilla):  # Instancia el regresor con sus hiperparámetros
    if nombre == "KNN":
        return KNeighborsRegressor(n_jobs=1, **p)
    if nombre == "Ridge":
        return Ridge(random_state=semilla, **p)
    if nombre == "Lasso":
        return Lasso(max_iter=3000, tol=1e-3, selection="random", random_state=semilla, **p)  # Descenso por coordenadas aleatorio: converge más rápido
    if nombre == "Árbol de decisión":
        return DecisionTreeRegressor(random_state=semilla, **p)
    if nombre == "Random Forest":
        return RandomForestRegressor(n_jobs=1, random_state=semilla, **p)
    if nombre == "XGBoost":
        return XGBRegressor(tree_method="hist", n_jobs=1, random_state=semilla, **p)
    if nombre == "SVR":  # Núcleo RBF aproximado + SVR lineal
        svr = LinearSVR(C=p["C"], epsilon=p["epsilon"], loss="squared_epsilon_insensitive", dual=False, max_iter=5000)
        return Pipeline([("nystroem", Nystroem(kernel="rbf", gamma=p["gamma"], n_components=300, random_state=semilla)), ("svr", svr)])
    raise ValueError(nombre)  # Nombre desconocido


def construir(tarea, nombre, params, balanceo="Sin balanceo", semilla=42):
    """Construye el Pipeline completo de una combinación (preprocesamiento + balanceo + modelo).

    Parameters
    ----------
    tarea : {"clf", "reg"}
        Clasificación o regresión.
    nombre : str
        Modelo (uno de MODELOS_CLF o MODELOS_REG).
    params : dict
        Hiperparámetros del modelo.
    balanceo : str
        Estrategia de balanceo (solo clasificación).
    semilla : int
        Semilla para todos los componentes aleatorios.

    Returns
    -------
    imblearn.pipeline.Pipeline
        Pipeline sin ajustar.
    """
    pasos = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]  # Preprocesamiento (se ajusta en cada pliegue)
    if tarea == "clf":  # Clasificación
        if balanceo == "SMOTE":  # Interpolación entre vecinos de la clase minoritaria
            pasos.append(("sampler", SMOTE(k_neighbors=5, random_state=semilla)))
        elif balanceo == "ADASYN":  # Más muestras sintéticas donde la clase minoritaria es más difícil
            pasos.append(("sampler", ADASYN(n_neighbors=5, random_state=semilla)))
        pasos.append(("modelo", _clasificador(nombre, params, balanceo == "class_weight balanced", semilla)))  # Modelo
    else:  # Regresión (sin balanceo)
        pasos.append(("modelo", _regresor(nombre, params, semilla)))  # Modelo
    return Pipeline(pasos)  # Pipeline de imbalanced-learn


def puntaje(modelo, X):
    """Puntaje continuo de la clase brote: probabilidad si existe, si no el margen de decisión (SVM).

    Parameters
    ----------
    modelo : estimator
        Pipeline ajustado.
    X : array-like
        Predictoras.

    Returns
    -------
    numpy.ndarray
        Puntaje por fila (mayor = más riesgo).
    """
    if hasattr(modelo, "predict_proba"):  # Probabilidad
        try:  # Algunos pipelines exponen el método pero el último paso no lo tiene
            return modelo.predict_proba(X)[:, 1]  # Clase 1
        except AttributeError:  # LinearSVC no tiene probabilidades
            pass  # Usa el margen
    return modelo.decision_function(X)  # Margen de la SVM
