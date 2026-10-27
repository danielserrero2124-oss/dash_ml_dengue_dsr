"""Cuaderno 1: ETL y EDA mejorado (sección Data)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Datos: ETL y EDA mejorado\n\n"
           "Este capítulo resume el flujo de datos (ETL) y amplía el análisis exploratorio del Primer Entregable con los hallazgos "
           "nuevos que exige este entregable: el **objetivo de regresión** y el efecto de las **técnicas de balanceo** sobre la "
           "estructura de los datos. El EDA completo (calidad, faltantes, componentes temporal, espacial y espacio-temporal) está en "
           "el libro del Primer Entregable y no se repite."),
        md("## ETL\n\n"
           "| Etapa | Qué se hace | Código |\n"
           "|---|---|---|\n"
           "| **Extracción** | Panel municipio × semana (`data/final_dataset.csv`, 294 032 filas) y centroides DANE (`data/divipola_municipios.csv`) | `e1/datos.py` |\n"
           "| **Transformación** | Canal endémico y brote; objetivo en $t+4$; 34 predictoras con semanas ≤ $t$; partición purgada | `src/datos.py`, `e1/datos.py` |\n"
           "| **Carga** | Entrenamiento 2015–2021, validación 2022 y prueba 2023–2024; submuestras estratificadas para la búsqueda | `e3/datos.py` |\n\n"
           "Nuevo en este entregable: el objetivo de regresión $y^{\\text{reg}}_{m,t} = \\log(1 + c_{m,t+4})$, los casos de la semana objetivo "
           "en escala logarítmica. Comparte horizonte, predictoras y partición con la clasificación."),
        code(PREAMBULO + '''
tr, va, te = dd.cargar()  # Periodos con los dos objetivos
pd.DataFrame({"filas": [len(tr), len(va), len(te)], "brotes (%)": [tr["y"].mean() * 100, va["y"].mean() * 100, te["y"].mean() * 100],
              "media de y_reg": [tr["y_reg"].mean(), va["y_reg"].mean(), te["y_reg"].mean()],
              "semanas con 0 casos en t+4 (%)": [(tr["y_reg"] == 0).mean() * 100, (va["y_reg"] == 0).mean() * 100, (te["y_reg"] == 0).mean() * 100]},
             index=["entrenamiento 2015–2021", "validación 2022", "prueba 2023–2024"])'''),
        md("**Interpretación.** La prueba es mucho más epidémica que el entrenamiento: tres veces más brotes (30 % frente a 10 %) y el "
           "doble de casos por semana en escala log. Para la regresión, el 62 % de las semanas del entrenamiento tiene 0 casos en $t+4$ "
           "(38 % en la prueba): el objetivo está **inflado en ceros** y su distribución cambia entre periodos."),

        md("## Hallazgo nuevo 1: el objetivo de regresión"),
        code('''bordes = np.linspace(0, tr["y_reg"].max(), 60)  # Intervalos
h = np.histogram(tr["y_reg"], bordes)[0] / len(tr)  # Proporciones
fig = go.Figure(go.Bar(x=(bordes[:-1] + bordes[1:]) / 2, y=h, marker_color=AZUL))  # Histograma
fig.update_layout(title="Distribución de log(1 + casos en t + 4) en el entrenamiento", xaxis_title="y_reg", yaxis_title="proporción", bargap=0)  # Formato
mostrar(fig, 340)  # Muestra
pd.Series({"ceros (%)": (tr["y_reg"] == 0).mean() * 100, "media": tr["y_reg"].mean(), "mediana": tr["y_reg"].median(), "desviación": tr["y_reg"].std(),
           "asimetría": tr["y_reg"].skew(), "curtosis": tr["y_reg"].kurt(), "máximo": tr["y_reg"].max()}).to_frame("y_reg").T'''),
        code('''from scipy import stats  # Correlaciones
rho = pd.Series({v: stats.spearmanr(tr[v], tr["y_reg"])[0] for v in dd.X_COLS}).sort_values(key=np.abs, ascending=False)  # Spearman con el objetivo continuo
fig = go.Figure(go.Bar(x=rho.values[::-1], y=rho.index[::-1], orientation="h", marker_color=[ROJO if v > 0 else AZUL for v in rho.values[::-1]]))  # Barras
fig.update_layout(title="Correlación de Spearman de cada predictora con y_reg (entrenamiento)", xaxis_title="ρ")  # Formato
mostrar(fig, 720)  # Muestra
rho.head(8).to_frame("ρ de Spearman")'''),
        code('''from sklearn.metrics import mean_squared_error, mean_absolute_error  # Métricas
lineas = {"Media de entrenamiento": np.full(len(te), tr["y_reg"].mean()), "Persistencia: log casos en t": te["lc_l0"].to_numpy(),
          "Persistencia suavizada: media 4 semanas": te["lc_m4"].to_numpy()}  # Líneas base triviales de regresión
pd.DataFrame({n: {"RMSE": np.sqrt(mean_squared_error(te["y_reg"], f)), "MAE": mean_absolute_error(te["y_reg"], f)} for n, f in lineas.items()}).T'''),
        code('''dec = pd.qcut(tr["lc_l0"].rank(method="first"), 10, labels=False)  # Deciles de los casos actuales
het = tr.groupby(dec).agg(lc_l0=("lc_l0", "mean"), media_y=("y_reg", "mean"), desv_y=("y_reg", "std"))  # Media y dispersión por decil
fig = go.Figure([go.Scatter(x=het["lc_l0"], y=het["media_y"], name="media de y_reg", line=dict(color=AZUL)),
                 go.Scatter(x=het["lc_l0"], y=het["desv_y"], name="desviación de y_reg", line=dict(color=ROJO))])  # Curvas
fig.update_layout(title="y_reg según los casos actuales: la dispersión cambia con el nivel", xaxis_title="log(1 + casos en t), media del decil")  # Formato
mostrar(fig, 340)  # Muestra'''),
        md("**Interpretación.**\n\n"
           "* El objetivo continuo tiene muchos ceros y una cola larga incluso en escala logarítmica: los modelos de regresión "
           "lineales tendrán residuos asimétricos y **heterocedásticos** (la dispersión de $y^{\\text{reg}}$ cambia con el nivel de casos), "
           "lo que anticipa el rechazo de la prueba de White en el análisis de residuos.\n"
           "* La historia de casos domina también en regresión ($\\rho \\approx 0{,}62$–$0{,}69$ con los rezagos y medias del log de casos). "
           "La persistencia ($\\hat y = \\log(1 + c_t)$, RMSE 0,66) y su versión suavizada (media de 4 semanas, RMSE 0,60) son líneas base "
           "exigentes; la media de entrenamiento (RMSE 1,37) es muy pobre por el cambio de régimen.\n"
           "* Consecuencia: el RMSE se reporta junto con el MAE (menos sensible a las semanas epidémicas) y los modelos de árboles "
           "deberían aprovechar mejor las relaciones no lineales y la interacción con el nivel de casos."),

        md("## Hallazgo nuevo 2: qué hacen las técnicas de balanceo con los datos\n\n"
           "SMOTE crea un brote sintético interpolando entre un brote y uno de sus $k$ vecinos brote: $\\mathbf x_{\\text{nuevo}} = "
           "\\mathbf x_i + \\lambda(\\mathbf x_{\\text{vecino}} - \\mathbf x_i)$, $\\lambda \\sim U(0,1)$. ADASYN genera más sintéticos alrededor de "
           "los brotes rodeados de no brotes (los difíciles). `class_weight` no crea filas: pondera los errores."),
        code('''from sklearn.decomposition import PCA  # Proyección para visualizar
from sklearn.preprocessing import StandardScaler  # Estandarización
from imblearn.over_sampling import SMOTE, ADASYN  # Sobremuestreo
m = dd.submuestra(tr, 6000)  # Muestra pequeña para la figura
Z = StandardScaler().fit_transform(m[dd.X_COLS])  # Estandariza (SMOTE usa distancias)
ym = m["y"].to_numpy()  # Clase
pca = PCA(2, random_state=c3.SEMILLA).fit(Z)  # Plano de proyección
fig = make_subplots(1, 3, subplot_titles=("Original", "SMOTE", "ADASYN"))  # Tres paneles
for j, (Zr, yr) in enumerate([(Z, ym), SMOTE(random_state=c3.SEMILLA).fit_resample(Z, ym), ADASYN(random_state=c3.SEMILLA).fit_resample(Z, ym)]):  # Conjuntos
    P = pca.transform(Zr)  # Proyección
    for clase, col in [(0, "#cbd5e1"), (1, ROJO)]:  # Clases
        s = yr == clase  # Filas
        fig.add_scattergl(x=P[s, 0], y=P[s, 1], mode="markers", marker=dict(size=3, color=col, opacity=0.5), showlegend=False, row=1, col=j + 1)  # Puntos
    fig.layout.annotations[j].text += f" — brotes: {yr.mean():.0%} de {len(yr):,} filas"  # Proporción
fig.update_layout(title="Proyección PCA antes y después del sobremuestreo (rojo: brote)")  # Título
mostrar(fig, 380)  # Muestra'''),
        md("**Interpretación.** SMOTE y ADASYN llevan la clase brote al 50 %, rellenando el espacio **entre** brotes reales. En un "
           "panel temporal esto tiene un costo: los brotes sintéticos mezclan semanas de municipios y años distintos, y el modelo ve una "
           "prevalencia ficticia del 50 % (sus probabilidades quedarán desplazadas hacia arriba). ADASYN concentra los sintéticos en la "
           "frontera, donde los brotes se confunden con no brotes. Por eso el balanceo se aplica **solo dentro del Pipeline, en la parte de "
           "entrenamiento de cada pliegue**, y su efecto se mide en la calibración además del AUC."),

        md("## Hallazgos del EDA anterior que guían este entregable\n\n"
           "| Hallazgo (Primer Entregable) | Decisión en este entregable |\n"
           "|---|---|\n"
           "| Autocorrelación temporal; objetivo en $t+4$ | Validación anidada con pliegues **por año** y purga de 4 semanas en ambos bucles |\n"
           "| Prevalencia del brote del 1,7 % al 25 % según el año | Criterio de optimización AUC-PR; balanceo comparado, no impuesto |\n"
           "| VIF ≫ 10 entre medias móviles | Regularización en los lineales (C, α, L1/L2); árboles robustos a la colinealidad |\n"
           "| Relaciones no lineales del clima | Se esperan ventajas de los modelos de árboles (Random Forest, XGBoost) |\n"
           "| Residuos del modelo base con dependencia temporal y espacial | Diagnóstico de residuos (Ljung-Box, BDS) y pruebas para series de pérdidas (MCS, DM) |\n"
           "| Panel de 376 municipios (filas no independientes) | Intervalos bootstrap por municipio y bloques de semanas en las pruebas |"),
    ]
    guardar("1_datos", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
