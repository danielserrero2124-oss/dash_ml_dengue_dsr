"""Sección «ML Models», con sub-pestañas organizadas según la rúbrica:
  * Fundamentos teóricos: formulación de HetDS en línea, función objetivo, supuestos, regularización y complejidad.
  * Diseño experimental: partición temporal, validación hacia adelante, Pipeline, código, hiperparámetros y tiempos.
  * Resultados: métricas, comparación con la línea base, pruebas de significancia, ablación, errores y simulador.
  * Conclusiones y preguntas: conclusiones, limitaciones, generalización, trabajo futuro y preguntas de la sustentación.
"""

import inspect  # Código fuente real de las funciones
import textwrap  # Quita la sangría común de los métodos

import numpy as np  # Cálculo numérico
from dash import Input, Output, callback, dcc, html  # Componentes y callbacks

from dashboard import figuras as fg  # Figuras
from dashboard.componentes import (cita, desplegable, formula, grafico, kpi, miles, numero, pasos, pregunta, rubrica, tabla, tarjeta)  # Interfaz
from dashboard.datos import ART, PRUEBAS, RES, SEMANAS_TEST, TE_SIM, TOP_MUN, Y_TEST, hetds, nombre_municipio  # Datos
from src import config as cf  # Configuración
from src import datos as dt  # Canal endémico (código)
from src import entrenamiento as en  # Flujo de entrenamiento (código)
from src import modelado as mod  # Modelos, métricas y pruebas
from src.hetds import HetDSEnLinea, expertos_por_defecto  # Modelo original

ETQ = fg.ETIQUETAS_MODELOS  # Nombres visibles
H = RES["HetDS"]  # Resultados del modelo original
N_BROTES = int(Y_TEST.sum())  # Brotes en la prueba
CODIGO = {  # Fragmentos de código fuente reales
    "Canal endémico y definición de brote": dt.canal_endemico,
    "Preprocesamiento (dentro del Pipeline)": mod.make_preprocessor,
    "Validación cruzada temporal hacia adelante": mod.ValidacionTemporalAnual,
    "Catálogo de modelos y mallas de hiperparámetros": mod.catalogo_modelos,
    "Pipeline + GridSearchCV": mod.train_pipeline,
    "Umbral de decisión en validación": mod.umbral_max_f1,
    "HetDS: expertos heterogéneos": expertos_por_defecto,
    "HetDS: predicción en línea con Hedge": HetDSEnLinea.predecir_en_linea,
    "Flujo completo de entrenamiento y evaluación": en.entrenar_todo,
    "Bootstrap por conglomerados (municipios)": mod.bootstrap_conglomerados,
    "Prueba de DeLong": mod.prueba_delong,
}


def params_texto(params):  # Hiperparámetros en texto corto
    return ", ".join(f"{k} = {v}" for k, v in params.items())  # p. ej. "eta = 0.05, ventana = 8"


def p_texto(p):  # Formato de valores p
    return "< 0,0001" if p < 1e-4 else numero(p, 4)  # Valores muy pequeños se acotan


def veredicto(dif, p):  # Insignia con el resultado de una prueba al 5 %
    if p >= 0.05:  # No se rechaza H0
        return html.Span("Sin diferencia", className="insignia no")  # Empate
    return html.Span("HetDS mejor" if dif > 0 else "HetDS peor", className=f"insignia {'si' if dif > 0 else 'no'}")  # Dirección


# ---------------------------------------------------------------------------
# Sub-pestaña 1: Fundamentos teóricos
# ---------------------------------------------------------------------------
MODELO = r"""
**Expertos.** $K = 5$ clasificadores heterogéneos $f_1,\dots,f_K$ (dos boosting de árboles, un bosque aleatorio, árboles extremos y una regresión
logística), entrenados con 2015–2021 sobre las variables del problema **más** el componente de autocorrelación espacial
$\mathbf{s}_{m,t}$ (casos, cambio de casos y proporción de municipios en brote en el resto del departamento):

$$p_k(m,t) = f_k\big(\mathbf{x}_{m,t},\, \mathbf{s}_{m,t}\big), \qquad \mathbf{s}_{m,t} = \Big(\ln\big(1 + \textstyle\sum_{j \in D(m),\, j \ne m} c_{j,t}\big),\; \Delta_4,\; \bar{b}_{D(m)\setminus m,\,t}\Big).$$

**Pérdida observada con retraso.** La etiqueta de la fila $(m,s)$ se confirma en $s + r$ ($r = 4$). En la semana $t$ solo se conocen las filas con $s \le t - r$.
La pérdida acumulada de cada experto en la ventana reciente $W(t) = \{s : t - r - V < s \le t - r\}$ es

$$L_k(t) = \sum_{s \in W(t)}\sum_{m} \ell\big(y_{m,s},\, p_k(m,s)\big), \qquad \ell(y, p) = -\big[y\ln p + (1 - y)\ln(1 - p)\big].$$

**Pesos Hedge (pesos exponenciales).**

$$w_k(t) = \frac{\exp\big(-\eta\, L_k(t)\big)}{\sum_{j=1}^{K}\exp\big(-\eta\, L_j(t)\big)}.$$

**Combinación en el espacio logit.**

$$P\big(y_{m,t} = 1\big) = \sigma\Big(\sum_{k=1}^{K} w_k(t)\,\operatorname{logit} p_k(m,t)\Big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}.$$
"""

OBJETIVO = r"""
**Función objetivo de cada experto:** riesgo empírico regularizado con log-loss (cada uno con su propia regularización: profundidad, hojas, $L_2$).

**Función objetivo del selector:** los pesos Hedge son la solución exacta de un problema de minimización con regularización entrópica
(*Follow the Regularized Leader*):

$$\mathbf{w}(t) = \arg\min_{\mathbf{w} \in \Delta_K}\; \eta\,\langle \mathbf{w}, \mathbf{L}(t)\rangle + \sum_{k} w_k \ln w_k .$$

**Garantía (arrepentimiento).** Con pérdidas acotadas en $[0, B]$ y $\eta$ adecuado, la pérdida acumulada de Hedge no supera la del mejor experto
en más de $B\sqrt{(T/2)\ln K}$: el ensamble rastrea al mejor experto de cada momento sin saber de antemano cuál será.
"""


def sub_fundamentos():  # Construye la sub-pestaña
    comparacion = [
        ("LogisticRegression", "Log-loss + λ‖w‖²", "C", "Logit lineal en las variables", "O(n·d) por iteración"),
        ("RandomForestClassifier", "Impureza de Gini por corte", "Profundidad, hojas mínimas", "Árboles poco correlacionados", "O(T·n·log n·√d)"),
        ("KNeighborsClassifier", "Ninguna (memoriza)", "k, ponderación", "Vecinos cercanos, misma clase", "O(n·d) por predicción"),
        ("GradientBoostingClassifier", "Log-loss por gradiente funcional", "Árboles, profundidad, tasa", "Aditivo en árboles", "O(T·n·d)"),
        ("HetDS en línea (original)", "Log-loss de expertos + Hedge (FTRL entrópico)", "η, ventana V; regularización de cada experto", "Expertos diversos; cambios de régimen graduales", "Σ expertos + O(K) por fila en línea"),
    ]
    return html.Div([
        rubrica(3),  # Título
        html.Div(html.Div([html.H2(["Modelo original: ", html.Span("HetDS en línea", className="resaltado")]),  # Banner
                           html.P("Selección dinámica de un ensamble heterogéneo con autocorrelación espacial: cinco expertos distintos cuyo peso se "
                                  "actualiza cada semana según qué tan bien han anticipado los brotes ya confirmados.")]), className="banner banner-virus"),
        html.Div([
            tarjeta(
                html.H3("¿Qué es y por qué este modelo?"),
                html.P(["Del listado del curso: ", html.B("HetDS — selección dinámica de ensambles heterogéneos"), " (grupo «Árboles, boosting y ensambles propuestos»). "
                        "Combina tres ideas que los modelos recomendados no tienen:"]),
                html.Ul([
                    html.Li([html.B("Autocorrelación espacial: "), "el dengue se propaga entre municipios vecinos; los expertos ven los casos y brotes del resto del departamento."]),
                    html.Li([html.B("Diversidad: "), "boosting, bosques, árboles extremos y una frontera lineal cometen errores distintos; combinarlos reduce la varianza."]),
                    html.Li([html.B("Adaptación en línea: "), "los pesos cambian semana a semana con las etiquetas ya confirmadas; así el ensamble se reacomoda en epidemias atípicas como la de 2024."]),
                ]),
                html.P(["Implementación propia en ", html.Code("src/hetds.py"), ": la selección dinámica, la ventana con retraso y la combinación están programadas desde cero; los expertos son estimadores de scikit-learn."]),
                cita("Cruz, R. M. O., Sabourin, R. y Cavalcanti, G. D. C. (2018). Dynamic classifier selection: Recent advances and perspectives. Information Fusion, 41, 195–216.", "https://doi.org/10.1016/j.inffus.2017.09.010"),
                cita("Ko, A. H. R., Sabourin, R. y Britto, A. S. (2008). From dynamic classifier selection to dynamic ensemble selection. Pattern Recognition, 41(5), 1718–1731.", "https://doi.org/10.1016/j.patcog.2007.10.015"),
                cita("Freund, Y. y Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences, 55(1), 119–139.", "https://doi.org/10.1006/jcss.1997.1504"),
                cita("Huber, J. H. et al. (2018). Seasonal temperature variation influences climate suitability for dengue, chikungunya, and Zika transmission. PLOS Neglected Tropical Diseases, 12(5), e0006451.", "https://doi.org/10.1371/journal.pntd.0006451"),
                clase="destacada",
            ),
            tarjeta(html.H3("Formulación matemática del algoritmo"), formula(MODELO), clase="destacada"),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Función objetivo y principio de aprendizaje"), formula(OBJETIVO),
                    html.P(["Principio: ", html.B("aprendizaje en línea con expertos"), ". Los expertos se ajustan una vez (aprendizaje por lotes); el selector aprende de forma continua "
                            "y no necesita reentrenar los expertos."], className="nota")),
            tarjeta(
                html.H3("Supuestos del modelo"),
                html.Ul([
                    html.Li([html.B("Etiquetas con retraso conocido: "), "un brote de la semana s se confirma en s + 4; nunca se usa una etiqueta antes de ese momento."]),
                    html.Li([html.B("Cambios de régimen graduales: "), "el experto que funcionó en las últimas semanas sigue siendo útil en las siguientes."]),
                    html.Li([html.B("Diversidad de los expertos: "), "si todos fallaran igual, ponderarlos no ayudaría."]),
                    html.Li([html.B("Dependencia espacial dentro del departamento: "), "los municipios del mismo departamento comparten clima y movilidad."]),
                ]),
                html.H3("Regularización, complejidad y generalización", style={"marginTop": "14px"}),
                html.Ul([
                    html.Li([html.B("η: "), "η → 0 da un promedio fijo (máxima estabilidad); η grande sigue al último líder (máxima adaptación). Se elige en 2022."]),
                    html.Li([html.B("Ventana V: "), "pocas semanas reaccionan rápido pero con ruido; muchas suavizan. El regularizador entrópico impide pesos extremos."]),
                    html.Li([html.B("Costo: "), "entrenar = suma de los cinco expertos; predecir en línea = O(K) por fila más una suma acumulada por semana."]),
                    html.Li([html.B("Generalización: "), f"AUC-ROC {numero(H['val_auc_roc'])} en 2022 y {numero(H['test']['auc_roc'])} en 2023–2024, años nunca vistos al entrenar."]),
                ]),
            ),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Peso de cada experto semana a semana"),
                html.P("Hedge reasigna el peso según la pérdida reciente de cada experto (con 4 semanas de retraso). 2022 sirve para calentar los pesos antes de la prueba.", className="nota"),
                grafico("hd-pesos", fg.fig_pesos(H["pesos"]))),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Expertos individuales, ensamble estático y HetDS en línea"),
                    html.P("AUC-ROC en 2023–2024. El ensamble supera a cada experto y la ponderación dinámica añade un poco más.", className="nota"),
                    grafico("hd-expertos", fg.fig_expertos(H))),
            tarjeta(html.H3("Selección de η y de la ventana (validación 2022)"),
                    html.P(f"AUC-PR en 2022 para cada combinación; se eligió {params_texto({k: v for k, v in H['params'].items() if k != 'retraso'})}.", className="nota"),
                    grafico("hd-seleccion", fg.fig_seleccion(H["seleccion"]))),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Comparación teórica de los cinco modelos"),
                tabla(["Modelo", "Función objetivo", "Control de complejidad", "Supuesto principal", "Costo"], comparacion)),
    ])


# ---------------------------------------------------------------------------
# Sub-pestaña 2: Implementación y diseño experimental
# ---------------------------------------------------------------------------
def tarjeta_tiempos():  # Tiempos de cómputo
    if PRUEBAS is None:  # Sin el script de pruebas
        return tarjeta(html.H3("Tiempos de cómputo"), tabla(["Modelo", "Búsqueda + ajuste"], [(ETQ[n], f"{numero(RES[n]['segundos'], 1)} s") for n in RES], numericas=[1]))
    t = PRUEBAS["tiempos"]  # Tiempos medidos
    filas = [(ETQ[n], f"{numero(t[n]['grid_s'], 1)} s", miles(t[n]["n_fit"]), f"{numero(t[n]['fit_s'], 1)} s", f"{numero(t[n]['pred_ms_1000'], 2)} ms") for n in RES]  # Filas
    return tarjeta(html.H3("Tiempos de cómputo"),
                   html.P("Equipo de 12 núcleos (Windows 11, Python 3.9). «Búsqueda» incluye GridSearchCV con 4 pliegues temporales (o la selección de η y ventana en 2022 para HetDS).", className="nota"),
                   html.Div([grafico("dis-tiempos", fg.fig_tiempos(t)),
                             html.Div([html.Label("Predicción por 1 000 municipio-semanas", className="control-etiqueta"), grafico("dis-pred", fg.fig_prediccion(t))])],
                            className="rejilla cols-2", style={"marginBottom": "0"}),
                   tabla(["Modelo", "Búsqueda + ajuste", "Filas (un ajuste)", "Un ajuste", "Predicción / 1 000"], filas, numericas=[1, 2, 3, 4]))


def sub_diseno():  # Construye la sub-pestaña
    catalogo = mod.catalogo_modelos()  # Mallas usadas
    just = {"LogisticRegression": "C regula la magnitud de los pesos; class_weight compara el manejo del desbalance.",
            "RandomForest": "Profundidad y hojas mínimas limitan la memorización de semanas aisladas.",
            "KNN": "k grande estabiliza la probabilidad; distance pondera vecinos cercanos.",
            "GradientBoosting": "Árboles poco profundos, tasa baja y submuestreo (0,8) regularizan el boosting.",
            "HetDS": "η y la ventana controlan la velocidad de adaptación; se eligen por AUC-PR en 2022."}
    hiper = [(ETQ[n], "; ".join(f"{k.split('__')[-1]} ∈ {v}" for k, v in catalogo[n][1].items()), params_texto(RES[n]["params"]), just[n]) for n in catalogo]  # Recomendados
    hiper.append((ETQ["HetDS"], "η ∈ [0.01, 0.05, 0.2]; ventana ∈ [4, 8, 26] semanas", params_texto(H["params"]), just["HetDS"]))  # Original
    n = ART["n"]  # Tamaños
    return html.Div([
        rubrica(4),  # Título
        tarjeta(html.H3("Separación de los datos: entrenamiento, validación y prueba"),
                html.P("División por años antes de cualquier transformación. Abajo, los cuatro pliegues de la validación cruzada hacia adelante de GridSearchCV.", className="nota"),
                grafico("dis-particion", fg.fig_particion(n))),
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Estrategia de validación"),
                    pasos([f"Entrenamiento 2015–2021 ({miles(n['train'])} filas); 2010–2014 solo alimentan el canal endémico.",
                           "GridSearchCV con validación cruzada hacia adelante por años (2018, 2019, 2020 y 2021 como validación): nunca se entrena con el futuro.",
                           f"Validación 2022 ({miles(n['val'])} filas): umbral de decisión de todos los modelos e hiperparámetros η y ventana de HetDS.",
                           f"Prueba 2023–2024 ({miles(n['test'])} filas, {N_BROTES:,} brotes) usada una sola vez.".replace(",", "."),
                           "Pruebas de significancia pareadas: DeLong, bootstrap por municipio, McNemar y Friedman semanal."])),
            tarjeta(html.H3("Pipeline de cada modelo"),
                    tabla(["Modelo", "Pipeline"], [
                        ("Cuatro recomendados", "SimpleImputer(mediana) → StandardScaler → estimador; GridSearchCV con ValidacionTemporalAnual"),
                        ("HetDS: cada experto", "SimpleImputer(mediana) → StandardScaler → estimador (boosting, bosque, árboles extremos o logística)"),
                        ("HetDS: selector", "Hedge en línea sobre los logits de los expertos, con ventana de V semanas y retraso de 4 semanas"),
                    ]),
                    html.P("Las variables (rezagos, medias móviles, anomalías, rezago espacial) se calculan antes, siempre con semanas ≤ t.", className="nota", style={"marginTop": "10px"})),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Código en Python: Pipeline, GridSearchCV y modelo original"),
                html.P("Código fuente real del proyecto, leído en vivo: lo que se muestra es exactamente lo que se ejecutó.", className="nota"),
                html.Label("Fragmento", className="control-etiqueta"),
                desplegable("dis-codigo", [{"label": k, "value": k} for k in CODIGO], "Pipeline + GridSearchCV"),
                html.Div(id="dis-codigo-texto", className="caja-codigo")),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta(html.H3("Hiperparámetros: malla, valor elegido y justificación"), tabla(["Modelo", "Malla", "Elegido", "Justificación"], hiper)),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta_tiempos(),  # Tiempos
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("Decisiones computacionales"), html.Ul([
                html.Li(f"KNN se entrena con una muestra estratificada de {miles(n['knn'])} filas: predecir exige comparar con todo el conjunto."),
                html.Li("GradientBoostingClassifier usa submuestreo (0,8) y GridSearchCV paraleliza pliegues y combinaciones en 12 núcleos."),
                html.Li("Los expertos de HetDS usan HistGradientBoosting (histogramas) para entrenar en segundos con 134 000 filas."),
                html.Li("El canal endémico se calcula de forma vectorizada (matriz municipios × semanas) en menos de un segundo."),
                html.Li("La importancia por permutación usa una muestra estratificada de 4 000 filas de prueba."),
            ])),
            tarjeta(html.H3("Reproducibilidad"), html.Ul([
                html.Li([html.B("Semilla: "), "42 en particiones, árboles y muestreos."]),
                html.Li([html.B("Entorno: "), "Python 3.9.25, scikit-learn 1.3.0, pandas 2.1.0, numpy 1.25.2, dash 2.18.2, plotly 5.24.1 (mismo que Heart Disease)."]),
                html.Li([html.B("Código modular: "), html.Code("src/config.py"), ", ", html.Code("src/datos.py"), ", ", html.Code("src/modelado.py"), ", ", html.Code("src/hetds.py"), "."]),
                html.Li([html.B("Jupyter Book: "), "los cuadernos ejecutan los modelos y quedan compilados en _build/html."]),
            ]), html.Pre("python -m src.entrenamiento   # entrena los 5 modelos\npython -m src.pruebas         # significancia y tiempos\npython -m dashboard.app       # dashboard", className="codigo")),
        ], className="rejilla cols-2"),
    ])


# ---------------------------------------------------------------------------
# Sub-pestaña 3: Resultados y evaluación crítica
# ---------------------------------------------------------------------------
FORMULAS_PRUEBAS = r"""
**Hipótesis.** $H_0$: HetDS y el modelo comparado tienen el mismo desempeño; $H_1$: difieren (bilateral, $\alpha = 0{,}05$, ajuste de Holm por 4 comparaciones).

**DeLong (AUC-ROC).** $z = \dfrac{\widehat{AUC}_A - \widehat{AUC}_B}{\sqrt{S_{AA} + S_{BB} - 2S_{AB}}} \sim \mathcal{N}(0,1)$, con $\mathbf{S}$ estimada por los componentes estructurales del estadístico U.

**Bootstrap por conglomerados.** DeLong supone filas independientes, pero las semanas de un mismo municipio están correlacionadas. Se remuestrean
**municipios completos** ($B$ = 1 000) y se obtiene la distribución de $\Delta^{*}$; $p = 2\min\{P(\Delta^{*} \le 0), P(\Delta^{*} \ge 0)\}$.

**McNemar (errores con el umbral).** $\chi^2 = \dfrac{(|b - c| - 1)^2}{b + c} \sim \chi^2_1$.

**Friedman + Nemenyi.** Cada semana de prueba es un bloque: $\chi^2_F = \dfrac{12N}{k(k+1)}\Big[\sum_j R_j^2 - \dfrac{k(k+1)^2}{4}\Big]$, $\;\text{DC} = q_\alpha\sqrt{\dfrac{k(k+1)}{6N}}$.
"""

REFERENCIAS = [
    ("DeLong, E. R. et al. (1988). Comparing the areas under two or more correlated ROC curves. Biometrics, 44(3), 837–845.", "https://doi.org/10.2307/2531595"),
    ("Saito, T. y Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432.", "https://doi.org/10.1371/journal.pone.0118432"),
    ("Chicco, D. y Jurman, G. (2020). The advantages of the Matthews correlation coefficient (MCC) over F1 score and accuracy in binary classification evaluation. BMC Genomics, 21, 6.", "https://doi.org/10.1186/s12864-019-6413-7"),
    ("Dietterich, T. G. (1998). Approximate statistical tests for comparing supervised classification learning algorithms. Neural Computation, 10(7), 1895–1923.", "https://doi.org/10.1162/089976698300017197"),
    ("Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. Journal of Machine Learning Research, 7, 1–30.", "https://jmlr.org/papers/v7/demsar06a.html"),
]


def tarjeta_metricas():  # Métricas de la literatura
    if PRUEBAS is None:  # Sin pruebas
        return html.Div()  # Nada
    m = PRUEBAS["metricas"]  # Métricas
    cols = ["auc_roc", "auc_pr", "mcc", "bal_acc", "prec_k", "rec_k", "brier", "logloss"]  # Columnas
    mejor = {c: (min if c in ("brier", "logloss") else max)(m, key=lambda n: m[n][c]) for c in cols}  # Mejor por columna
    cel = lambda n, c, t: html.B(t) if mejor[c] == n else t  # Negrita para el mejor
    filas = [(ETQ[n], cel(n, "auc_roc", f"{numero(m[n]['auc_roc'], 4)} [{numero(m[n]['ic_auc_roc'][0])}–{numero(m[n]['ic_auc_roc'][1])}]"),
              cel(n, "auc_pr", f"{numero(m[n]['auc_pr'], 4)} [{numero(m[n]['ic_auc_pr'][0])}–{numero(m[n]['ic_auc_pr'][1])}]"),
              cel(n, "mcc", numero(m[n]["mcc"])), cel(n, "bal_acc", numero(m[n]["bal_acc"])), cel(n, "prec_k", numero(m[n]["prec_k"])),
              cel(n, "rec_k", numero(m[n]["rec_k"])), cel(n, "brier", numero(m[n]["brier"], 4)), cel(n, "logloss", numero(m[n]["logloss"], 4))) for n in RES]
    return tarjeta(html.H3("Métricas de comparación recomendadas en la literatura (prueba 2023–2024)"),
                   html.P(f"Entre corchetes, IC del 95 % (DeLong para AUC-ROC; bootstrap por municipio para AUC-PR). Precision@5 % y recall@5 %: las "
                          f"{miles(int(round(PRUEBAS['top_k'] * len(Y_TEST))))} municipio-semanas de mayor riesgo, la capacidad típica de una vigilancia focalizada.", className="nota"),
                   tabla(["Modelo", "AUC-ROC [IC]", "AUC-PR [IC]", "MCC", "Exactitud balanceada", "Precision@5 %", "Recall@5 %", "Brier ↓", "Log-loss ↓"], filas, numericas=range(1, 9)),
                   html.Details([html.Summary("Referencias de las métricas y pruebas"), html.Div([cita(t, u) for t, u in REFERENCIAS])], className="pregunta", style={"marginTop": "12px"}))


def bloque_pruebas():  # Pruebas de significancia
    if PRUEBAS is None:  # Sin pruebas
        return tarjeta(html.H3("Prueba de DeLong"), grafico("ml-delong", fg.fig_delong(ART["delong"])))  # Mínimo
    pr, fr = PRUEBAS["pruebas"], PRUEBAS["friedman"]["auc_roc"]  # Resultados
    fmt = lambda x: f"{x:+.4f}".replace(".", ",")  # Diferencia con signo
    filas = [(ETQ[n], fmt(pr[n]["delong"]["dif"]), numero(pr[n]["delong"]["z"], 2), p_texto(pr[n]["delong"]["p_holm"]),
              f"[{fmt(pr[n]['cluster_roc']['ic'][0])}; {fmt(pr[n]['cluster_roc']['ic'][1])}]", p_texto(pr[n]["cluster_roc"]["p_holm"]),
              veredicto(pr[n]["cluster_roc"]["dif"], pr[n]["cluster_roc"]["p_holm"]),
              fmt(pr[n]["cluster_pr"]["dif"]), p_texto(pr[n]["cluster_pr"]["p_holm"]), veredicto(pr[n]["cluster_pr"]["dif"], pr[n]["cluster_pr"]["p_holm"]),
              f"{pr[n]['mcnemar']['b']} / {pr[n]['mcnemar']['c']}", p_texto(pr[n]["mcnemar"]["p_holm"])) for n in pr]
    anual = [(ETQ[n], fmt(pr[n]["anual"][2023]["dif"]), p_texto(pr[n]["anual"][2023]["p"]), fmt(pr[n]["anual"][2024]["dif"]), p_texto(pr[n]["anual"][2024]["p"])) for n in pr]  # Por año
    return html.Div([
        html.Div([
            tarjeta(html.H3("Pruebas de significancia: planteamiento"), formula(FORMULAS_PRUEBAS), clase="destacada"),
            tarjeta(html.H3("HetDS frente a cada modelo: DeLong (AUC-ROC)"),
                    html.P(f"Diferencia de AUC-ROC con IC 95 % sobre {miles(len(Y_TEST))} municipio-semanas de prueba ({N_BROTES:,} brotes). Magenta: significativa.".replace(",", "."), className="nota"),
                    grafico("ml-delong", fg.fig_delong(ART["delong"])),
                    html.H3("Por año de prueba (DeLong)", style={"marginTop": "10px"}),
                    tabla(["HetDS vs.", "Δ 2023", "p", "Δ 2024", "p"], anual, numericas=[1, 2, 3, 4]), clase="destacada"),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Pruebas pareadas en la prueba 2023–2024 (valores p ajustados por Holm)"),
                html.P("La decisión usa el bootstrap por municipio, la prueba más estricta. McNemar: b = filas que solo HetDS clasifica bien; c = las que solo el otro modelo clasifica bien.", className="nota"),
                tabla(["HetDS vs.", "ΔAUC-ROC", "z DeLong", "p DeLong", "IC 95 % (bootstrap)", "p bootstrap", "Resultado", "ΔAUC-PR", "p bootstrap", "Resultado", "b / c", "p McNemar"],
                      filas, numericas=[1, 2, 3, 4, 5, 7, 8, 10, 11])),
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta(html.H3("Friedman + Nemenyi con las semanas de prueba como bloques"),
                html.P(f"χ²_F = {numero(fr['chi2'], 1)}, p = {p_texto(fr['p'])}, {fr['n_semanas']} semanas. HetDS es el mejor (o empata con el mejor) en el "
                       f"{numero(100 * fr['gana_hetds'], 0)} % de las semanas. Dentro de la franja sombreada no hay diferencia con el mejor.", className="nota"),
                grafico("ml-nemenyi", fg.fig_nemenyi(fr))),
    ])


def tarjeta_ablacion():  # De dónde viene la ganancia
    if PRUEBAS is None:  # Sin pruebas
        return html.Div()  # Nada
    a = PRUEBAS["ablacion"]  # Ablación
    filas = [(k, numero(v["auc_roc"], 4), numero(v["auc_pr"], 4) if "auc_pr" in v else "—",
              f"{v['dif_gb']:+.4f}".replace(".", ",") if "dif_gb" in v else "—", p_texto(v["p"]) if "p" in v else "—") for k, v in a.items()]  # Filas
    return tarjeta(html.H3("Ablación: ¿de dónde viene la ventaja de HetDS?"),
                   html.P("Se quita una pieza a la vez y se compara con GradientBoosting (bootstrap por municipio). Así se separa lo que aporta la información espacial, "
                          "lo que aporta la diversidad del ensamble y lo que aporta la ponderación en línea.", className="nota"),
                   grafico("ml-ablacion", fg.fig_ablacion(a)),
                   tabla(["Variante", "AUC-ROC", "AUC-PR", "Δ vs. GB", "p"], filas, numericas=[1, 2, 3, 4]))


PAISES_SIM = TOP_MUN[:25]  # Municipios del simulador


def sub_resultados():  # Construye la sub-pestaña
    semanas = SEMANAS_TEST  # Semanas de prueba
    return html.Div([
        rubrica(5),  # Título
        html.Div([
            tarjeta(html.H3("Modelo y umbral de decisión"),
                    html.Label("Modelo", className="control-etiqueta"),
                    desplegable("ml-modelo", [{"label": ETQ[n], "value": n} for n in RES], "HetDS"),
                    html.Label("Umbral de probabilidad para emitir alerta", className="control-etiqueta"),
                    dcc.Slider(id="ml-umbral", min=0, max=1, step=0.01, value=round(H["umbral"], 2), marks={i / 10: f"{i / 10:.1f}".replace(".", ",") for i in range(0, 11, 2)},
                               tooltip={"placement": "bottom", "always_visible": True}),
                    html.P(id="ml-umbral-nota", className="nota", style={"marginTop": "14px"}), html.Div(id="ml-parametros")),
            tarjeta(html.H3("Selección y justificación de las métricas"), html.Ul([
                html.Li([html.B("AUC-PR (principal): "), "con 10–30 % de brotes, la curva PR muestra mejor el desempeño sobre la clase de interés (Saito y Rehmsmeier, 2015)."]),
                html.Li([html.B("AUC-ROC: "), "ordena municipio-semanas por riesgo; admite DeLong y el bootstrap pareado."]),
                html.Li([html.B("MCC: "), "resume las cuatro celdas de la matriz de confusión sin inflarse con el desbalance (Chicco y Jurman, 2020)."]),
                html.Li([html.B("Precision@5 %: "), "de las alertas que caben en la capacidad de vigilancia, cuántas son brotes reales."]),
                html.Li([html.B("Brier y log-loss: "), "calidad de las probabilidades, importante para priorizar recursos."]),
            ])),
        ], className="rejilla cols-control"),
        html.Div([
            kpi("AUC-ROC prueba", "", "", id_valor="kpi-auc", id_detalle="kpi-auc-ic"), kpi("AUC-PR prueba", "", f"azar = {numero(Y_TEST.mean())}", id_valor="kpi-ap"),
            kpi("Precisión", "", "alertas que fueron brote", id_valor="kpi-prec"), kpi("Recall", "", "brotes anticipados", id_valor="kpi-rec"),
            kpi("F1", "", "balance", id_valor="kpi-f1"), kpi("Exactitud", "", "referencia", id_valor="kpi-acc"),
        ], className="rejilla", style={"gridTemplateColumns": "repeat(auto-fit, minmax(170px, 1fr))"}),
        html.Div([
            tarjeta(html.H3("Curvas ROC (prueba 2023–2024)"), grafico("ml-roc")),
            tarjeta(html.H3("Curvas precisión-recall (prueba 2023–2024)"), grafico("ml-pr")),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Matriz de confusión con el umbral elegido"), html.P(id="ml-conf-nota", className="nota"), grafico("ml-confusion")),
            tarjeta(html.H3("Ranking comparativo frente a la línea base"),
                    html.P("Ordenado por AUC-PR de validación (criterio de selección, sin mirar la prueba). Métricas de umbral con el umbral de 2022 de cada modelo.", className="nota"),
                    html.Div(id="ml-ranking")),
        ], className="rejilla cols-2"),
        tarjeta_metricas(),  # Métricas de la literatura
        html.Div(style={"height": "18px"}),  # Separador
        bloque_pruebas(),  # Pruebas de significancia
        html.Div(style={"height": "18px"}),  # Separador
        tarjeta_ablacion(),  # Ablación
        html.Div(style={"height": "18px"}),  # Separador
        html.Div([
            tarjeta(html.H3("¿Cuánto aporta el clima?"),
                    html.P("Mismo algoritmo (HistGradientBoosting) con distintos bloques de variables. El clima solo predice moderadamente (AUC ≈ 0,71); cuando ya se conoce la historia de casos, el clima casi no añade información: su efecto ya está reflejado en los casos recientes.", className="nota"),
                    grafico("ml-aporte", fg.fig_aporte(ART["aporte_clima"]))),
            tarjeta(html.H3("Desempeño por año de prueba"),
                    html.P("AUC-ROC en 2023 y en la epidemia récord de 2024 (eje truncado: las diferencias son pequeñas).", className="nota"),
                    grafico("ml-anual", fg.fig_auc_anual(RES))),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Diagnóstico de sobreajuste"),
                    html.P("AUC-ROC en entrenamiento, validación cruzada temporal, validación 2022 y prueba (eje truncado en 0,75).", className="nota"),
                    grafico("ml-sobreajuste", fg.fig_sobreajuste(RES))),
            tarjeta(html.H3("Variables más importantes del modelo seleccionado"),
                    html.P("Importancia por permutación: caída del AUC-ROC al desordenar cada variable.", className="nota"), grafico("ml-importancia")),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Simulador de escenarios climáticos (HetDS)"),
                html.P("Elige un municipio y una semana de la prueba y modifica el clima: el simulador recalcula las variables climáticas y la probabilidad de brote a 4 semanas.", className="nota"),
                html.Div([
                    html.Div([
                        html.Div([html.Div([html.Label("Municipio", className="control-etiqueta"), desplegable("s-mun", [{"label": nombre_municipio(m), "value": m} for m in PAISES_SIM], PAISES_SIM[0])]),
                                  html.Div([html.Label("Semana", className="control-etiqueta"), desplegable("s-semana", [{"label": s, "value": s} for s in semanas], semanas[len(semanas) // 2])])],
                                 className="rejilla cols-2", style={"marginBottom": "0"}),
                        html.Label("Cambio de temperatura (°C)", className="control-etiqueta"),
                        dcc.Slider(id="s-temp", min=-2, max=3, step=0.5, value=0, marks={v: f"{v:+g}" for v in range(-2, 4)}),
                        html.Label("Lluvia respecto a lo observado", className="control-etiqueta"),
                        dcc.Slider(id="s-lluvia", min=0.5, max=2, step=0.1, value=1, marks={0.5: "50 %", 1: "100 %", 1.5: "150 %", 2: "200 %"}),
                        html.Label("Cambio del índice ONI (El Niño)", className="control-etiqueta"),
                        dcc.Slider(id="s-oni", min=-1.5, max=1.5, step=0.25, value=0, marks={-1.5: "−1,5", 0: "0", 1.5: "+1,5"}),
                    ]),
                    html.Div([grafico("s-medidor"), html.Div(id="s-veredicto", className="veredicto")], className="resultado"),
                ], className="rejilla", style={"gridTemplateColumns": "minmax(0, 2fr) minmax(0, 1fr)"})),
    ])


# ---------------------------------------------------------------------------
# Sub-pestaña 4: Conclusiones, limitaciones y preguntas
# ---------------------------------------------------------------------------
def sub_conclusiones():  # Construye la sub-pestaña
    t = {n: RES[n]["test"] for n in RES}  # Métricas de prueba
    mejor_rec = max([n for n in RES if n != "HetDS"], key=lambda n: t[n]["auc_roc"])  # Mejor recomendado
    gen = [(ETQ[n], numero(RES[n]["train_auc_roc"]), numero(RES[n]["cv_auc_roc"]), numero(RES[n]["val_auc_roc"]), numero(t[n]["auc_roc"]),
            f"{RES[n]['train_auc_roc'] - t[n]['auc_roc']:+.3f}".replace(".", ",")) for n in RES]  # Generalización
    pr = PRUEBAS["pruebas"] if PRUEBAS else None  # Pruebas
    sig = [ETQ[n] for n in pr if pr[n]["cluster_roc"]["p_holm"] < 0.05 and pr[n]["cluster_roc"]["dif"] > 0] if pr else []  # Superados (bootstrap)
    return html.Div([
        html.Div([rubrica(6), rubrica(7)], className="rejilla cols-2", style={"marginBottom": "0"}),  # Títulos
        html.Div(html.Div([html.H2(["Del modelo a la ", html.Span("acción en salud pública", className="resaltado")]),  # Banner
                           html.P("Una alerta con cuatro semanas de anticipación permite eliminar criaderos, fumigar y preparar hospitales antes del pico.")]),
                 className="banner banner-brazo"),
        html.Div([
            tarjeta(html.H3("Conclusiones sustentadas en los resultados"), html.Ul([
                html.Li([html.B("HetDS es el mejor modelo: "), f"AUC-ROC {numero(t['HetDS']['auc_roc'], 4)} y AUC-PR {numero(t['HetDS']['auc_pr'], 4)} en 2023–2024, frente a "
                         f"{numero(t[mejor_rec]['auc_roc'], 4)} y {numero(t[mejor_rec]['auc_pr'], 4)} del mejor recomendado ({ETQ[mejor_rec]})."]),
                html.Li([html.B("Superioridad significativa: "), f"con la prueba más estricta (bootstrap por municipio, Holm), HetDS supera a {', '.join(sig) or 'ningún modelo'}."]),
                html.Li([html.B("Tamaño del efecto: "), "+0,004 de AUC-ROC y +0,006 de AUC-PR frente a los mejores recomendados, +0,016 frente a LR y KNN: pequeño en valor absoluto porque todos se acercan a un techo de ≈ 0,90, pero consistente (HetDS gana en 2 de cada 3 semanas)."]),
                html.Li([html.B("Origen de la ganancia: "), "la ablación muestra que ambos componentes aportan y se suman: el ensamble en línea sin variables espaciales ya supera a GradientBoosting (+0,0016) y la información espacial añade otro tanto (+0,0028 por sí sola)."]),
                html.Li([html.B("El clima importa, pero no basta: "), f"solo con clima el AUC es ≈ {numero(ART['aporte_clima']['Solo clima y contexto']['auc_roc'], 2)}; con la historia de casos sube a ≈ 0,90 y, a partir de ahí, el clima casi no añade: su efecto ya está en los casos recientes."]),
                html.Li([html.B("Robustez temporal: "), "el desempeño se mantiene en la epidemia récord de 2024, aunque la prevalencia de brotes triplica la del entrenamiento."]),
            ])),
            tarjeta(html.H3("Limitaciones"),
                    html.P(html.B("Teóricas")), html.Ul([
                        html.Li("Hedge garantiza seguir al mejor experto, no superar el techo de información de las variables."),
                        html.Li("El componente espacial usa el departamento, no la distancia real ni la movilidad entre municipios."),
                        html.Li("La ponderación supone cambios graduales: ante un cambio brusco reacciona con 4 semanas de retraso."),
                    ]),
                    html.P(html.B("Empíricas")), html.Ul([
                        html.Li("64 % de las semanas tiene temperatura imputada: la señal térmica está atenuada."),
                        html.Li("El canal endémico depende del subregistro y de cambios en la notificación entre años."),
                        html.Li("Solo dos años de prueba; 2024 fue excepcional, lo que limita la generalización a otros regímenes."),
                        html.Li("Las filas no son independientes: por eso se usan pruebas por conglomerados además de DeLong."),
                    ])),
        ], className="rejilla cols-2"),
        html.Div([
            tarjeta(html.H3("Capacidad de generalización"),
                    html.P("AUC-ROC en cada etapa. Brechas pequeñas entre validación cruzada, validación 2022 y prueba indican que los modelos generalizan a años futuros.", className="nota"),
                    tabla(["Modelo", "Entrenamiento", "CV temporal", "Validación 2022", "Prueba", "Brecha"], gen, numericas=[1, 2, 3, 4, 5])),
            tarjeta(html.H3("Posibles mejoras y líneas futuras"), pasos([
                "Vecindad espacial real (coordenadas, red de movilidad) en lugar del departamento.",
                "Datos climáticos sin imputar (estaciones IDEAM, reanálisis ERA5) y modelos de rezagos distribuidos no lineales.",
                "Serotipos circulantes y datos entomológicos (índices aédicos) como variables.",
                "Expertos reentrenados en línea y umbrales de alerta sensibles al costo de la respuesta.",
                "Evaluación prospectiva con secretarías de salud y monitoreo de deriva de los datos.",
            ])),
        ], className="rejilla cols-2"),
        tarjeta(html.H3("Preguntas esperadas en la sustentación"), html.P("Haz clic en cada pregunta para ver la respuesta.", className="nota"),
                pregunta("¿Cómo se define un brote?", html.P("Con el canal endémico: una semana es de brote si sus casos superan el tercer cuartil de las mismas semanas (± 2) de los cinco años anteriores y hay al menos 3 casos. Es el método de cuartiles (Bortman) que usa la vigilancia en salud pública.")),
                pregunta("¿Por qué predecir a 4 semanas y no la semana actual?", html.P("Porque la utilidad está en anticipar: 4 semanas permiten intervenir criaderos y preparar servicios. Además coincide con el ciclo mosquito–virus–síntomas.")),
                pregunta("¿Qué es HetDS y por qué es original?", html.P("Una selección dinámica de un ensamble heterogéneo: cinco expertos distintos cuyo peso se actualiza cada semana con el algoritmo Hedge, usando solo etiquetas confirmadas. La selección dinámica, la ventana con retraso y la combinación en logit están programadas desde cero; además incorpora autocorrelación espacial.")),
                pregunta("¿Por qué Hedge?", html.P("Porque tiene garantía teórica: su pérdida acumulada no excede la del mejor experto en más de B·√((T/2)·ln K). Es además la solución exacta de minimizar la pérdida con regularización entrópica, lo que evita pesos extremos.")),
                pregunta("¿Usar etiquetas de la prueba en línea no es fuga de datos?", html.P("No, si se respeta el retraso: para predecir la semana t solo se usan brotes de semanas ≤ t − 4, que en la realidad ya estarían confirmados. Es exactamente lo que haría un sistema de vigilancia en operación.")),
                pregunta("¿Es justo que HetDS tenga variables espaciales que los otros no tienen?", html.P("El componente espacial es parte del diseño del modelo original. Para transparencia, la ablación entrega GradientBoosting con las mismas variables espaciales: así se ve cuánto aporta la información espacial y cuánto el ensamble en línea.")),
                pregunta("¿Por qué validación cruzada hacia adelante y no StratifiedKFold?", html.P("Las semanas están autocorrelacionadas: mezclar semanas al azar filtra información del futuro e infla las métricas (la demostración del EDA lo muestra). Entrenar con años anteriores y validar en el siguiente simula el uso real.")),
                pregunta("¿Por qué un bootstrap por municipios además de DeLong?", html.P("DeLong supone filas independientes; las semanas de un municipio no lo son. Remuestrear municipios completos respeta esa dependencia y da intervalos más honestos (más anchos).")),
                pregunta("¿Por qué McNemar no favorece a HetDS frente a GradientBoosting?", html.P("McNemar compara errores totales con el umbral de cada modelo, no el ordenamiento del riesgo. El umbral de HetDS (máximo F1 en 2022) emite más alertas: anticipa más brotes (recall 0,808 frente a 0,771) a cambio de más falsas alarmas (precisión 0,675 frente a 0,696), y por eso suma más errores totales. El AUC, que no depende del umbral, es donde HetDS es superior; en vigilancia suele preferirse perder menos brotes aunque haya más alertas.")),
                pregunta("¿Las diferencias son importantes en la práctica?", html.P("Son estadísticamente significativas pero pequeñas: el techo de ≈ 0,90 lo fija la información disponible. En una vigilancia nacional, unas décimas de AUC-PR se traducen en decenas de brotes anticipados más con el mismo número de alertas.")),
                pregunta("¿Cuál es el aporte del clima?", html.P("Por sí solo da un AUC de ≈ 0,71: modula el riesgo (temperaturas de 25–28 °C, El Niño, lluvias previas) con rezagos de 8–12 semanas. Pero cuando ya se conocen los casos recientes, añadir el clima casi no cambia el AUC (0,8968 → 0,8969): su efecto ya se refleja en los casos. Con un horizonte más largo o sin historia de casos, el clima sería más valioso.")),
                pregunta("¿Qué supuestos podrían fallar?", html.P("Que los cambios de régimen sean graduales (un brote explosivo nuevo tarda 4 semanas en reflejarse en los pesos) y que el departamento represente la vecindad real.")),
                pregunta("¿Cuál es el costo computacional?", html.P("Entrenar HetDS cuesta la suma de sus cinco expertos (≈ 1–2 minutos); predecir en línea es O(K) por fila más una suma acumulada por semana: milisegundos por cada 1 000 municipio-semanas.")),
        ),
    ])


def seccion_modelos():  # Sección con sub-pestañas
    return html.Div([
        dcc.Tabs(id="ml-sub", value="fundamentos", className="subpestanas", children=[
            dcc.Tab(label="Fundamentos teóricos", value="fundamentos", className="subtab", selected_className="subtab--selected"),
            dcc.Tab(label="Diseño experimental", value="diseno", className="subtab", selected_className="subtab--selected"),
            dcc.Tab(label="Resultados", value="resultados", className="subtab", selected_className="subtab--selected"),
            dcc.Tab(label="Conclusiones y preguntas", value="conclusiones", className="subtab", selected_className="subtab--selected"),
        ]),
        html.Div(id="ml-sub-contenido"),
    ])


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(Output("ml-sub-contenido", "children"), Input("ml-sub", "value"))  # Sub-pestaña
def mostrar_sub(v):
    return {"fundamentos": sub_fundamentos, "diseno": sub_diseno, "resultados": sub_resultados, "conclusiones": sub_conclusiones}[v]()  # Construye


@callback(Output("dis-codigo-texto", "children"), Input("dis-codigo", "value"))  # Código fuente
def mostrar_codigo(nombre):
    obj = CODIGO[nombre]  # Función o clase
    archivo = "/".join(inspect.getsourcefile(obj).replace("\\", "/").split("/")[-2:])  # Ruta corta
    return [html.Div(archivo, className="nota"), dcc.Markdown(f"```python\n{textwrap.dedent(inspect.getsource(obj))}\n```")]  # Código con resaltado


@callback(Output("ml-umbral", "value"), Input("ml-modelo", "value"))  # Umbral de validación del modelo elegido
def umbral_defecto(m):
    return round(RES[m]["umbral"], 2)  # Redondeado al paso del deslizador


def fila(n, activo):  # Fila del ranking
    r, t = RES[n], RES[n]["test"]  # Resultados
    return html.Tr([html.Td(ETQ[n]), html.Td(numero(r["val_auc_pr"]), className="num"), html.Td(numero(t["auc_roc"]), className="num"),
                    html.Td(numero(t["auc_pr"]), className="num"), html.Td(numero(t["precision"]), className="num"), html.Td(numero(t["recall"]), className="num"),
                    html.Td(numero(t["f1"]), className="num"), html.Td(numero(t["accuracy"]), className="num")], className="activa" if activo else "")


@callback(Output("kpi-auc", "children"), Output("kpi-auc-ic", "children"), Output("kpi-ap", "children"), Output("kpi-prec", "children"),
          Output("kpi-rec", "children"), Output("kpi-f1", "children"), Output("kpi-acc", "children"), Output("ml-roc", "figure"), Output("ml-pr", "figure"),
          Output("ml-confusion", "figure"), Output("ml-conf-nota", "children"), Output("ml-ranking", "children"), Output("ml-importancia", "figure"),
          Output("ml-parametros", "children"), Output("ml-umbral-nota", "children"), Input("ml-modelo", "value"), Input("ml-umbral", "value"))
def actualizar(m, u):  # Todo lo que depende del modelo y el umbral
    r = RES[m]  # Resultados
    met = mod.metricas(Y_TEST, r["p_test"], u)  # Métricas con el umbral
    (vn, fp), (fn, vp) = met["matriz"]  # Celdas
    nota = f"Anticipa {miles(vp)} de {miles(vp + fn)} brotes con {miles(fp)} falsas alarmas sobre {miles(vn + fp)} municipio-semanas sin brote."  # Lectura
    orden = sorted(RES, key=lambda n: RES[n]["val_auc_pr"], reverse=True)  # Ranking por AUC-PR de validación
    rank = html.Div(html.Table([html.Thead(html.Tr([html.Th(h, className="num" if i else "") for i, h in enumerate(["Modelo", "Val. AUC-PR", "AUC-ROC", "AUC-PR", "Prec.", "Recall", "F1", "Exact."])])),
                                html.Tbody([fila(n, n == m) for n in orden])], className="tabla"), className="tabla-envoltura")
    par = html.Div([html.Label("Hiperparámetros elegidos", className="control-etiqueta"), html.Div([html.Span(f"{k} = {v}", className="chip") for k, v in r["params"].items()])])  # Chips
    ic = r["ic_auc"]  # IC
    return (numero(met["auc_roc"], 4), f"IC 95 %: {numero(ic[0])} – {numero(ic[1])}", numero(met["auc_pr"], 4), numero(met["precision"]), numero(met["recall"]),
            numero(met["f1"]), numero(met["accuracy"]), fg.fig_roc(RES, m), fg.fig_pr(RES, m, float(Y_TEST.mean())), fg.fig_confusion(met["matriz"]), nota, rank,
            fg.fig_importancia(r["importancia"], fg.COLORES_MODELOS[m]), par, f"Umbral elegido en 2022 (máximo F1): {numero(r['umbral'])}.")


@callback(Output("s-medidor", "figure"), Output("s-veredicto", "children"), Input("s-mun", "value"), Input("s-semana", "value"),
          Input("s-temp", "value"), Input("s-lluvia", "value"), Input("s-oni", "value"))  # Simulador climático
def simular(mun, semana, dtemp, lluvia, doni):
    fila_te = TE_SIM[(TE_SIM["COD_MUN_N"] == mun) & (TE_SIM["week_start"].dt.strftime("%Y-%m-%d") == semana)].copy()  # Fila elegida
    modelo = hetds()  # Modelo original (se carga la primera vez que se usa el simulador)
    if fila_te.empty or modelo is None:  # Sin datos o sin modelo
        return fg.fig_medidor(0, 0.5), "Sin datos para esa combinación"  # Aviso
    for c in ["temp_mean_m1", "temp_mean_m4", "temp_mean_m8", "temp_mean_m12", "temp_anom_m8"]:  # Escenario de temperatura
        fila_te[c] = fila_te[c] + dtemp  # Desplaza la temperatura
    for c in ["prec_total_m1", "prec_total_m4", "prec_total_m8", "prec_total_m12"]:  # Escenario de lluvia
        fila_te[c] = fila_te[c] * lluvia  # Escala la lluvia
    fila_te["prec_anom_m8"] = fila_te["prec_anom_m8"] + (lluvia - 1) * fila_te["prec_total_m8"] / lluvia  # Anomalía coherente con el nuevo nivel
    fila_te["oni_anom"] = fila_te["oni_anom"] + doni  # Escenario de El Niño
    fila_te["oni_l8"] = fila_te["oni_l8"] + doni  # También el rezago
    pesos = H["pesos"]  # Historial de pesos
    i = pesos["semana"].index(semana) if semana in pesos["semana"] else -1  # Semana en el historial
    w = np.array([pesos[k][i] for k in pesos if k != "semana"])  # Pesos de esa semana
    P = modelo.probas_expertos(fila_te)  # Probabilidades de los expertos
    z = np.log(np.clip(P, 1e-5, 1 - 1e-5) / (1 - np.clip(P, 1e-5, 1 - 1e-5))) @ w  # Combinación Hedge
    p = float(1 / (1 + np.exp(-z[0])))  # Probabilidad
    u = H["umbral"]  # Umbral de alerta
    real = int(fila_te[cf.TARGET].iloc[0])  # Lo que pasó
    texto = html.Span("🚨 Alerta de brote en 4 semanas" if p >= u else "✅ Sin alerta", style={"color": fg.BROTE if p >= u else "#6ff0c8"})  # Veredicto
    return fg.fig_medidor(p, u), [texto, html.Div(f"Umbral {numero(u)} · Lo que ocurrió: {'brote' if real else 'sin brote'} (escenario observado)", className="nota")]
