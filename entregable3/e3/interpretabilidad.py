"""Interpretabilidad global y local del mejor modelo: valores de Shapley (SHAP) y comparación con LIME."""

import pickle  # Lectura de los reajustes finales

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas

from e3 import config as c3, datos as dd, experimento as ex, modelos as mo  # Módulos del entregable


def reajustar(tarea, modelo, balanceo, metodo, semilla=c3.SEMILLA):
    """Vuelve a ajustar una combinación final (mismos hiperparámetros y misma submuestra del reajuste final).

    Returns
    -------
    pipeline : imblearn.pipeline.Pipeline
        Pipeline ajustado.
    estimador : estimator
        Último paso (el modelo), sin envoltorios de balanceo.
    transformar : callable
        Función que aplica la imputación y la estandarización ajustadas (el espacio en que se explica el modelo).
    """
    with open(ex.FINALES / ex._nombre(tarea, modelo, balanceo, metodo), "rb") as f:  # Reajuste final
        params = pickle.load(f)["params"]  # Hiperparámetros
    tr, _, _ = ex.datos()  # Entrenamiento
    ent = dd.submuestra(tr, c3.N_FINAL, semilla)  # Misma submuestra
    y_col = dd.Y_CLF if tarea == "clf" else dd.Y_REG  # Objetivo
    pipe = mo.construir(tarea, modelo, params, balanceo, semilla).fit(ent[dd.X_COLS].to_numpy(), ent[y_col].to_numpy())  # Ajuste
    est = pipe.named_steps["modelo"]  # Modelo
    est = getattr(est, "est_", est)  # Quita el envoltorio de pesos balanceados
    prep = lambda X: pipe.named_steps["scaler"].transform(pipe.named_steps["imputer"].transform(X))  # Preprocesamiento
    return pipe, est, prep  # Piezas


def shap_arbol(est, Xp, tarea="clf"):
    """Valores SHAP exactos para modelos de árboles (TreeSHAP, en escala log-odds en clasificación).

    Propiedades: eficiencia (la suma de los valores más el valor base es la predicción), simetría (variables con igual
    aporte reciben igual valor) y aditividad (en un ensamble, los valores son la suma de los de cada árbol).
    """
    import shap  # Librería SHAP
    explicador = shap.TreeExplainer(est)  # TreeSHAP: polinomial en la profundidad, exacto para árboles
    valores = explicador(Xp)  # Explicación
    if valores.values.ndim == 3:  # Bosques de scikit-learn devuelven una salida por clase
        valores = valores[:, :, 1]  # Clase brote
    valores.feature_names = list(dd.X_COLS)  # Nombres de las variables
    return valores  # Objeto Explanation


def lime_local(est, Xp_entrenamiento, x, semilla=c3.SEMILLA, num=10):
    """Explicación LIME de una fila: regresión lineal ponderada ajustada a perturbaciones alrededor de la fila."""
    from lime.lime_tabular import LimeTabularExplainer  # LIME tabular
    expl = LimeTabularExplainer(Xp_entrenamiento, feature_names=list(dd.X_COLS), class_names=["no brote", "brote"],
                                discretize_continuous=True, random_state=semilla)  # Explicador (perturba con la distribución del entrenamiento)
    e = expl.explain_instance(x, est.predict_proba, num_features=num, num_samples=5000)  # Explicación local
    return pd.DataFrame(e.as_list(), columns=["condición", "peso LIME"]), e.score  # Pesos y R² del modelo sustituto local
