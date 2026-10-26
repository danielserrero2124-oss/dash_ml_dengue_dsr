"""Modelo original: HetDS en línea — selección dinámica de un ensamble heterogéneo con autocorrelación espacial.

Del listado del curso: «HetDS (selección dinámica de ensambles heterogéneos)», grupo «Árboles, boosting y
ensambles propuestos». Base bibliográfica (indexada en Scopus y Web of Science):
  * Cruz, R. M. O., Sabourin, R. y Cavalcanti, G. D. C. (2018). Dynamic classifier selection: Recent advances and
    perspectives. Information Fusion, 41, 195-216. https://doi.org/10.1016/j.inffus.2017.09.010
  * Freund, Y. y Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application
    to boosting. Journal of Computer and System Sciences, 55(1), 119-139. https://doi.org/10.1006/jcss.1997.1504
  * Cesa-Bianchi, N. y Lugosi, G. (2006). Prediction, Learning, and Games. Cambridge University Press.

Idea: K expertos heterogéneos (dos boosting, un bosque, árboles extremos y una regresión logística), entrenados con
las variables del problema MÁS un componente de autocorrelación espacial (casos y brotes del resto del departamento).
Su combinación no es fija: cada semana t se pondera a cada experto según su pérdida reciente con el algoritmo Hedge
(pesos exponenciales), usando SOLO etiquetas ya conocidas en t (las de semanas ≤ t − 4, porque el brote de la semana
s se confirma en s + 4). Así el ensamble se adapta a cambios de régimen, como la epidemia récord de 2024.

    Pesos (Hedge):   w_k(t) = exp(−η · L_k(t)) / Σ_j exp(−η · L_j(t)),   L_k(t) = Σ_{s ∈ W(t)} ℓ(y_s, p_k(s))
    Ventana:         W(t) = filas con semana en (t − r − V, t − r],  r = retraso de la etiqueta, V = ventana
    Combinación:     p(t) = σ( Σ_k w_k(t) · logit(p_k(t)) )
    Pérdida:         ℓ(y, p) = −[y ln p + (1 − y) ln(1 − p)]   (log-loss)

Todo el mecanismo de selección dinámica, la ventana con retraso y la combinación están programados desde cero;
los expertos son estimadores de scikit-learn dentro de Pipelines.
"""

import numpy as np  # Álgebra vectorizada
import pandas as pd  # Historial de pesos
from sklearn.base import BaseEstimator, ClassifierMixin, clone  # Interfaz de scikit-learn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier  # Expertos de árboles
from sklearn.impute import SimpleImputer  # Imputación
from sklearn.linear_model import LogisticRegression  # Experto lineal
from sklearn.pipeline import Pipeline  # Preprocesamiento + experto
from sklearn.preprocessing import StandardScaler  # Estandarización

from src import config as cf  # Configuración del proyecto


def _logit(p):  # Transformación logit con recorte numérico
    p = np.clip(p, 1e-5, 1 - 1e-5)  # Evita infinitos
    return np.log(p / (1 - p))  # ln(p / (1 − p))


def _sigmoide(z):  # Función logística
    return 1 / (1 + np.exp(-np.clip(z, -35, 35)))  # σ(z)


def expertos_por_defecto(semilla=cf.SEMILLA):  # Ensamble heterogéneo: sesgos inductivos distintos
    prep = lambda: Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])  # Preprocesamiento común
    return {
        "Boosting (31 hojas)": Pipeline([("prep", prep()), ("clf", HistGradientBoostingClassifier(  # Boosting flexible
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31, random_state=semilla))]),
        "Boosting (7 hojas)": Pipeline([("prep", prep()), ("clf", HistGradientBoostingClassifier(  # Boosting de árboles pequeños y regularizado
            max_iter=800, learning_rate=0.03, max_leaf_nodes=7, l2_regularization=1.0, random_state=semilla + 1))]),
        "Bosque aleatorio": Pipeline([("prep", prep()), ("clf", RandomForestClassifier(  # Promedio de árboles profundos
            n_estimators=300, max_depth=16, min_samples_leaf=30, max_features=0.3, n_jobs=-1, random_state=semilla))]),
        "Árboles extremos": Pipeline([("prep", prep()), ("clf", ExtraTreesClassifier(  # Cortes aleatorios: más diversidad
            n_estimators=300, max_depth=18, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=semilla))]),
        "Regresión logística": Pipeline([("prep", prep()), ("clf", LogisticRegression(C=0.05, max_iter=3000))]),  # Frontera lineal: extrapola
    }


class HetDSEnLinea(BaseEstimator, ClassifierMixin):
    """Selección dinámica en línea de un ensamble heterogéneo (Hedge con ventana y retraso de etiquetas).

    Parámetros
    ----------
    eta : tasa de aprendizaje de Hedge (qué tan rápido se concentra el peso en el mejor experto).
    ventana : semanas de historia reciente con las que se mide la pérdida de cada experto.
    retraso : semanas que tarda en conocerse la etiqueta (igual al horizonte de predicción).
    columnas : variables que reciben los expertos (por defecto, las del problema más las espaciales).
    """

    def __init__(self, eta=0.05, ventana=8, retraso=cf.HORIZONTE, columnas=None, semilla=cf.SEMILLA):
        self.eta = eta  # Tasa de Hedge
        self.ventana = ventana  # Ventana de pérdida
        self.retraso = retraso  # Retraso de la etiqueta
        self.columnas = columnas  # Variables de los expertos
        self.semilla = semilla  # Semilla

    def fit(self, X, y):  # Entrena cada experto con todo el entrenamiento
        self.columnas_ = list(self.columnas or cf.FEATURES_ORIGINAL)  # Variables usadas
        self.classes_ = np.array([0, 1])  # Clases (requisito de ClassifierMixin)
        self.expertos_ = {}  # Expertos ajustados
        for nombre, est in expertos_por_defecto(self.semilla).items():  # Recorre el ensamble
            self.expertos_[nombre] = clone(est).fit(X[self.columnas_], y)  # Ajusta una copia limpia
        return self  # Convención

    def probas_expertos(self, X):  # Matriz N × K con la probabilidad de cada experto
        return np.column_stack([e.predict_proba(X[self.columnas_])[:, 1] for e in self.expertos_.values()])  # Una columna por experto

    def predict_proba(self, X):  # Predicción estática (pesos iguales): se usa cuando no hay etiquetas recientes
        p = _sigmoide(_logit(self.probas_expertos(X)).mean(axis=1))  # Promedio de logits
        return np.column_stack([1 - p, p])  # Formato de scikit-learn

    def predecir_en_linea(self, X, y, tiempo, P=None):  # Recorre las semanas en orden y adapta los pesos
        """X, y y tiempo describen un flujo de semanas consecutivas. Para predecir la semana t solo se usan las
        etiquetas de filas con tiempo ≤ t − retraso. Devuelve (probabilidades, historial de pesos por semana)."""
        P = self.probas_expertos(X) if P is None else P  # Probabilidades de los expertos
        Z = _logit(P)  # Logits de los expertos
        y = np.asarray(y, dtype=float)  # Etiquetas (solo se leen las ya conocidas)
        tiempo = np.asarray(tiempo)  # Semana de cada fila
        perdida = -(y[:, None] * np.log(np.clip(P, 1e-5, 1)) + (1 - y[:, None]) * np.log(np.clip(1 - P, 1e-5, 1)))  # Log-loss por fila y experto
        semanas = np.unique(tiempo)  # Semanas del flujo, en orden
        orden = np.argsort(tiempo, kind="stable")  # Filas ordenadas por semana
        limites = np.searchsorted(tiempo[orden], semanas)  # Inicio de cada semana en el orden
        acumulada = np.vstack([np.zeros(P.shape[1]), np.cumsum(perdida[orden], axis=0)])  # Pérdida acumulada (para sumar ventanas rápido)
        salida = np.empty(len(tiempo))  # Probabilidades finales
        historial = []  # Pesos de cada semana
        K = P.shape[1]  # Número de expertos
        for i, t in enumerate(semanas):  # Semana a semana
            ini = np.searchsorted(tiempo[orden], t - self.retraso - self.ventana, side="right")  # Primera fila conocida de la ventana
            fin = np.searchsorted(tiempo[orden], t - self.retraso, side="right")  # Última fila conocida (semana ≤ t − retraso)
            L = acumulada[fin] - acumulada[ini]  # Pérdida total de cada experto en la ventana
            w = np.exp(-self.eta * (L - L.min())) if fin > ini else np.ones(K)  # Hedge (se resta el mínimo por estabilidad numérica)
            w = w / w.sum()  # Normaliza
            filas = orden[limites[i]:(limites[i + 1] if i + 1 < len(semanas) else len(orden))]  # Filas de la semana t
            salida[filas] = _sigmoide(Z[filas] @ w)  # Combinación ponderada en el espacio logit
            historial.append(w)  # Guarda los pesos de la semana
        pesos = pd.DataFrame(historial, index=semanas, columns=list(self.expertos_))  # Historial semana × experto
        return salida, pesos  # Probabilidades y pesos
