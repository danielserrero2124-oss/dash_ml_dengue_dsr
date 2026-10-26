"""Figuras de Plotly del dashboard de brotes de dengue, con un tema oscuro común."""

import numpy as np  # Cálculo numérico
import plotly.graph_objects as go  # Figuras
from plotly.subplots import make_subplots  # Paneles apilados (sin doble eje)

# ---------------------------------------------------------------------------
# Paleta (pasos oscuros validados de la paleta categórica de referencia)
# ---------------------------------------------------------------------------
AZUL, NARANJA, AGUA, AMARILLO, MAGENTA = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"  # Ranuras 1–5 en orden fijo
SIN_BROTE, BROTE = AZUL, "#e66767"  # Colores de clase
TEXTO, TEXTO_2, TEXTO_3 = "#f4f2ef", "#c4bfb8", "#8f8a83"  # Tintas de texto
REJILLA, SUPERFICIE = "rgba(255,255,255,0.06)", "#17191b"  # Rejilla recesiva y superficie
COLORES_MODELOS = {"LogisticRegression": AZUL, "RandomForest": NARANJA, "KNN": AGUA, "GradientBoosting": AMARILLO, "HetDS": MAGENTA}  # Color fijo por modelo
ETIQUETAS_MODELOS = {"LogisticRegression": "LogisticRegression", "RandomForest": "RandomForestClassifier", "KNN": "KNeighborsClassifier",
                     "GradientBoosting": "GradientBoostingClassifier", "HetDS": "HetDS en línea (original)"}  # Nombres visibles
COLORES_EXPERTOS = [AMARILLO, "#9c6a00", NARANJA, AGUA, AZUL]  # Expertos del HetDS (dos boosting, bosque, árboles extremos, logística)
DIVERGENTE = [[0, "#3987e5"], [0.5, "#2f3337"], [1, "#e66767"]]  # Azul – gris – rojo
SECUENCIAL = [[0, "#17191b"], [0.5, "#184f95"], [1, "#2a78d6"]]  # Rampa azul
MESES_ES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]  # Meses

ETIQUETAS = {  # Nombres legibles de las variables
    "lc_l0": "log casos (semana t)", "lc_l1": "log casos (t − 1)", "lc_l2": "log casos (t − 2)", "lc_l3": "log casos (t − 3)",
    "lc_m4": "log casos, media 4 sem.", "lc_m8": "log casos, media 8 sem.", "lc_m12": "log casos, media 12 sem.",
    "brote_l0": "Brote en la semana t", "brote_m8": "Semanas en brote (8 sem.)", "ratio_umbral": "Casos actuales / umbral objetivo",
    "l_umbral_obj": "Umbral del canal endémico (t + 4)", "incid": "Incidencia por 100 000 hab.",
    "temp_mean_m1": "Temperatura (semana)", "temp_mean_m4": "Temperatura, media 4 sem.", "temp_mean_m8": "Temperatura, media 8 sem.", "temp_mean_m12": "Temperatura, media 12 sem.",
    "prec_total_m1": "Precipitación (semana)", "prec_total_m4": "Precipitación, media 4 sem.", "prec_total_m8": "Precipitación, media 8 sem.", "prec_total_m12": "Precipitación, media 12 sem.",
    "ndvi_mean_m1": "NDVI (semana)", "ndvi_mean_m4": "NDVI, media 4 sem.", "ndvi_mean_m8": "NDVI, media 8 sem.", "ndvi_mean_m12": "NDVI, media 12 sem.",
    "temp_anom_m8": "Anomalía de temperatura (8 sem.)", "prec_anom_m8": "Anomalía de precipitación (8 sem.)", "ndvi_anom_m8": "Anomalía de NDVI (8 sem.)",
    "oni_anom": "ONI (El Niño), actual", "oni_l8": "ONI hace 8 semanas", "temp_missing": "Temperatura imputada",
    "sem_sin": "Estacionalidad (seno)", "sem_cos": "Estacionalidad (coseno)", "lpob": "log población", "flujo_m4": "Flujo de personas (4 sem.)",
    "esp_lc": "Casos en el resto del departamento", "esp_lc_d4": "Cambio de casos en el departamento", "esp_brote": "Municipios vecinos en brote",
    "temp_mean": "Temperatura media (°C)", "prec_total": "Precipitación (mm)", "ndvi_mean": "NDVI", "oni_anom_raw": "ONI", "Flujo_in": "Flujo de personas",
    "casos_totales": "Casos semanales", "poblacion": "Población",
}


def pct(x, dec=1):  # Porcentaje con coma decimal
    return f"{100 * x:.{dec}f} %".replace(".", ",")  # 0.123 → "12,3 %"


def tema(fig, alto=360, leyenda=True):  # Tema oscuro común
    fig.update_layout(height=alto, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",  # Fondo transparente
                      font=dict(family="Segoe UI, system-ui, sans-serif", color=TEXTO_2, size=12.5),  # Tipografía
                      margin=dict(l=10, r=10, t=30 if leyenda else 10, b=10),  # Márgenes
                      hoverlabel=dict(bgcolor="#202326", bordercolor="#3a3f44", font=dict(color=TEXTO, size=12.5)),  # Globo
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXTO_2)),  # Leyenda
                      showlegend=leyenda, bargap=0.25)  # Separación de barras
    fig.update_xaxes(gridcolor=REJILLA, zeroline=False, linecolor="#2f3337", tickfont=dict(color=TEXTO_3), title_font=dict(color=TEXTO_2), automargin=True)  # Eje X recesivo
    fig.update_yaxes(gridcolor=REJILLA, zeroline=False, linecolor="#2f3337", tickfont=dict(color=TEXTO_3), title_font=dict(color=TEXTO_2), automargin=True)  # Eje Y recesivo
    return fig  # Figura tematizada


def barras(fig):  # Esquinas redondeadas y anillo de superficie entre barras
    fig.update_traces(marker_cornerradius=4, marker_line=dict(color=SUPERFICIE, width=1.5), selector=dict(type="bar"))  # Estilo de barras
    return fig  # Figura


def sombrear_periodos(fig, fechas_corte, fila=None):  # Sombras de validación (2022) y prueba (2023–2024)
    kw = dict(row=fila, col=1) if fila else {}  # Panel opcional
    fig.add_vrect(x0=fechas_corte["val"], x1=fechas_corte["test"], fillcolor="rgba(25,158,112,0.12)", line_width=0, **kw)  # Validación
    fig.add_vrect(x0=fechas_corte["test"], x1=fechas_corte["fin"], fillcolor="rgba(201,133,0,0.14)", line_width=0, **kw)  # Prueba
    return fig  # Figura


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------
def fig_serie_nacional(nac, cortes):  # Casos semanales nacionales y El Niño (ONI) en paneles apilados
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32], vertical_spacing=0.06)  # Dos paneles, un solo eje Y por panel
    fig.add_trace(go.Scatter(x=nac["fecha"], y=nac["casos"], mode="lines", line=dict(color=BROTE, width=1.6), fill="tozeroy", fillcolor="rgba(230,103,103,0.15)",  # Casos
                             name="Casos semanales", hovertemplate="%{x|%d %b %Y}<br>%{y:,} casos<extra></extra>"), row=1, col=1)
    color_oni = [BROTE if v > 0.5 else (AZUL if v < -0.5 else "#5b6066") for v in nac["oni"]]  # El Niño (rojo), La Niña (azul), neutro (gris)
    fig.add_trace(go.Bar(x=nac["fecha"], y=nac["oni"], marker_color=color_oni, name="ONI", hovertemplate="%{x|%b %Y}<br>ONI = %{y:+.2f}<extra></extra>"), row=2, col=1)  # ONI
    sombrear_periodos(fig, cortes, 1)  # Periodos en el panel de casos
    fig.update_yaxes(title_text="Casos", row=1, col=1)  # Eje del panel superior
    fig.update_yaxes(title_text="ONI", row=2, col=1)  # Eje del panel inferior
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))  # Barras finas sin borde
    return tema(fig, 430, leyenda=False)  # Tema


def fig_anual(anual):  # Casos por año y periodo del modelo
    color = [("#5b6066" if a < 2015 else AZUL if a <= 2021 else AGUA if a == 2022 else AMARILLO) for a in anual.index]  # Historia, entrenamiento, validación, prueba
    fig = go.Figure(go.Bar(x=anual.index.astype(str), y=anual["casos"], marker_color=color, customdata=anual["tasa_brote"],  # Barras
                           hovertemplate="%{x}<br>%{y:,} casos<br>Semanas en brote: %{customdata:.1%}<extra></extra>"))
    fig.update_yaxes(title="Casos notificados")  # Eje Y
    fig.update_xaxes(type="category")  # Años como categorías
    return barras(tema(fig, 330, leyenda=False))  # Tema


def fig_estacionalidad(est):  # Perfil por semana epidemiológica: casos y lluvia (paneles apilados)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08)  # Dos paneles
    fig.add_trace(go.Scatter(x=est.index, y=est["casos"], mode="lines", line=dict(color=BROTE, width=2), name="Casos (media nacional)",  # Casos
                             hovertemplate="Semana %{x}<br>%{y:,.0f} casos<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=est.index, y=est["prec"], mode="lines", line=dict(color=AZUL, width=2), name="Precipitación media",  # Lluvia
                             hovertemplate="Semana %{x}<br>%{y:.1f} mm<extra></extra>"), row=2, col=1)
    fig.update_yaxes(title_text="Casos", row=1, col=1)  # Eje superior
    fig.update_yaxes(title_text="Lluvia (mm)", row=2, col=1)  # Eje inferior
    fig.update_xaxes(title_text="Semana epidemiológica", row=2, col=1)  # Eje X
    return tema(fig, 380, leyenda=False)  # Tema


def fig_canal(serie, nombre):  # Canal endémico de un municipio: casos, umbral y semanas de brote
    fig = go.Figure()  # Figura vacía
    fig.add_trace(go.Scatter(x=serie["fecha"], y=serie["umbral"], mode="lines", line=dict(color=AMARILLO, width=1.5, dash="dot"), name="Umbral (Q3 de 5 años)",  # Umbral
                             hovertemplate="%{x|%d %b %Y}<br>Umbral: %{y:.1f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=serie["fecha"], y=serie["casos"], mode="lines", line=dict(color=AZUL, width=1.5), name="Casos semanales",  # Casos
                             hovertemplate="%{x|%d %b %Y}<br>%{y} casos<extra></extra>"))
    b = serie[serie["brote"] == 1]  # Semanas de brote
    fig.add_trace(go.Scatter(x=b["fecha"], y=b["casos"], mode="markers", marker=dict(color=BROTE, size=6, line=dict(color=SUPERFICIE, width=1)), name="Semana de brote",  # Brotes
                             hovertemplate="%{x|%d %b %Y}<br>Brote: %{y} casos<extra></extra>"))
    fig.update_yaxes(title="Casos semanales")  # Eje Y
    fig.update_layout(title=dict(text=nombre, font=dict(color=TEXTO_2, size=13), x=0.99, xanchor="right", y=0.97))  # Nombre del municipio
    return tema(fig, 360)  # Tema


def fig_tasa_por_decil(tabla, var):  # Tasa de brote (en 4 semanas) por decil de una variable climática
    fig = go.Figure(go.Scatter(x=tabla["centro"], y=tabla["tasa"], mode="lines+markers", line=dict(color=BROTE, width=2),  # Curva
                               marker=dict(size=8, color=BROTE, line=dict(color=SUPERFICIE, width=2)), customdata=np.c_[tabla["n"], tabla["lo"], tabla["hi"]],
                               hovertemplate="%{customdata[1]:.2f} – %{customdata[2]:.2f}<br>Tasa de brote: %{y:.1%}<br>%{customdata[0]:,} municipio-semanas<extra></extra>"))
    fig.update_xaxes(title=ETIQUETAS.get(var, var) + " (centro del decil)")  # Eje X
    fig.update_yaxes(title="Brote 4 semanas después", tickformat=".0%")  # Eje Y
    return tema(fig, 360, leyenda=False)  # Tema


def fig_ccf(ccf):  # Correlación cruzada clima → casos según el rezago (media de los municipios)
    fig = go.Figure()  # Figura vacía
    for var, color, nombre in [("temp", BROTE, "Anomalía de temperatura"), ("prec", AZUL, "Anomalía de precipitación"), ("ndvi", AGUA, "Anomalía de NDVI")]:  # Tres variables
        fig.add_trace(go.Scatter(x=ccf["rezago"], y=ccf[var], mode="lines+markers", line=dict(color=color, width=2), marker=dict(size=6), name=nombre,  # Curva
                                 hovertemplate=nombre + "<br>Rezago %{x} semanas<br>r = %{y:+.3f}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=TEXTO_3, width=1))  # Sin correlación
    fig.update_xaxes(title="Rezago (semanas que el clima antecede a los casos)")  # Eje X
    fig.update_yaxes(title="Correlación media (r)")  # Eje Y
    return tema(fig, 360)  # Tema


def fig_oni_anual(tabla):  # El Niño medio del año frente a la proporción de semanas en brote
    fig = go.Figure(go.Scatter(x=tabla["oni"], y=tabla["tasa"], mode="markers+text", text=tabla.index.astype(str), textposition="top center",  # Puntos con el año
                               textfont=dict(color=TEXTO_2, size=11), marker=dict(size=12, color=[BROTE if v > 0.5 else AZUL if v < -0.5 else "#8f8a83" for v in tabla["oni"]],
                                                                                  line=dict(color=SUPERFICIE, width=2)),
                               hovertemplate="%{text}<br>ONI medio: %{x:+.2f}<br>Semanas en brote: %{y:.1%}<extra></extra>"))
    fig.update_xaxes(title="ONI medio del año (> 0,5 El Niño · < −0,5 La Niña)")  # Eje X
    fig.update_yaxes(title="Municipio-semanas en brote", tickformat=".0%")  # Eje Y
    return tema(fig, 360, leyenda=False)  # Tema


def fig_departamentos(dep):  # Incidencia media anual por departamento
    dep = dep.sort_values("incid").tail(15)  # 15 con mayor incidencia
    fig = go.Figure(go.Bar(x=dep["incid"], y=dep["nombre"], orientation="h", marker_color=BROTE, customdata=dep["casos"],  # Barras
                           hovertemplate="%{y}<br>%{x:.0f} casos por 100 000 hab. al año<br>%{customdata:,} casos 2010–2024<extra></extra>"))
    fig.update_xaxes(title="Incidencia media anual por 100 000 habitantes")  # Eje X
    return barras(tema(fig, 440, leyenda=False))  # Tema


def fig_grupos(edad, sexo):  # Casos por grupo de edad y por sexo (dos paneles)
    fig = make_subplots(rows=1, cols=2, column_widths=[0.65, 0.35], subplot_titles=("Grupo de edad", "Sexo"))  # Dos paneles
    fig.add_trace(go.Bar(x=list(edad.keys()), y=list(edad.values()), marker_color=BROTE, hovertemplate="%{x}: %{y:,} casos<extra></extra>"), row=1, col=1)  # Edad
    fig.add_trace(go.Bar(x=list(sexo.keys()), y=list(sexo.values()), marker_color=[AZUL, MAGENTA], hovertemplate="%{x}: %{y:,} casos<extra></extra>"), row=1, col=2)  # Sexo
    fig.update_annotations(font=dict(color=TEXTO_2, size=12.5))  # Títulos de panel
    fig.update_yaxes(title_text="Casos 2010–2024", row=1, col=1)  # Eje Y
    return barras(tema(fig, 340, leyenda=False))  # Tema


def fig_faltantes(tabla):  # Proporción de temperatura imputada por año
    fig = go.Figure(go.Bar(x=tabla.index.astype(str), y=tabla.values, marker_color="#8f8a83", hovertemplate="%{x}: %{y:.1%} de semanas con temperatura imputada<extra></extra>"))  # Barras
    fig.update_yaxes(title="Temperatura imputada", tickformat=".0%", range=[0, 1])  # Eje Y
    fig.update_xaxes(type="category")  # Años
    return barras(tema(fig, 300, leyenda=False))  # Tema


def fig_correlacion_objetivo(corr):  # Correlación de cada variable con el brote a 4 semanas
    c = corr.sort_values()  # Orden
    fig = go.Figure(go.Bar(x=c.values, y=[ETIQUETAS.get(v, v) for v in c.index], orientation="h", marker_color=[BROTE if v > 0 else AZUL for v in c.values],  # Barras
                           hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>"))
    fig.update_xaxes(title="Correlación con el brote en t + 4")  # Eje X
    return barras(tema(fig, 760, leyenda=False))  # Tema


def fig_matriz_correlacion(c):  # Matriz de correlación entre predictoras
    nombres = [ETIQUETAS.get(v, v)[:26] for v in c.columns]  # Etiquetas cortas
    fig = go.Figure(go.Heatmap(z=c.values, x=nombres, y=nombres, zmin=-1, zmax=1, colorscale=DIVERGENTE, xgap=1, ygap=1,  # Mapa de calor
                               colorbar=dict(tickfont=dict(color=TEXTO_3), outlinewidth=0), hovertemplate="%{y} vs. %{x}<br>r = %{z:.2f}<extra></extra>"))
    fig.update_xaxes(showticklabels=False, showgrid=False)  # Sin etiquetas en X
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(size=10))  # Etiquetas en Y
    return tema(fig, 760, leyenda=False)  # Tema


def fig_atipicos(filas):  # Tasa de brote en atípicos frente al resto
    filas = [f for f in filas if f["pct_atipicos"] > 0][::-1]  # Solo variables con atípicos
    n = [ETIQUETAS.get(f["variable"], f["variable"]) for f in filas]  # Etiquetas
    fig = go.Figure([go.Bar(y=n, x=[f["tasa_normales"] for f in filas], orientation="h", name="Sin atípicos", marker_color=SIN_BROTE,  # Normales
                            hovertemplate="%{y}<br>Tasa de brote sin atípicos: %{x:.1%}<extra></extra>"),
                     go.Bar(y=n, x=[f["tasa_atipicos"] for f in filas], orientation="h", name="Atípicos (1,5 × IQR)", marker_color=BROTE,  # Atípicos
                            customdata=[f["pct_atipicos"] for f in filas], hovertemplate="%{y}<br>Tasa de brote en atípicos: %{x:.1%}<br>%{customdata:.1%} de las filas<extra></extra>")])
    fig.update_layout(barmode="group")  # Agrupadas
    fig.update_xaxes(title="Tasa de brote en t + 4", tickformat=".0%")  # Eje X
    return barras(tema(fig, 560))  # Tema


def fig_fuga(fuga):  # Demostración de fuga: partición correcta, aleatoria y con variable del futuro
    casos = [("correcto", "Temporal (correcto)", AZUL), ("aleatoria", "Semanas al azar", AMARILLO), ("futuro", "Con casos de t + 4", BROTE)]  # Escenarios
    fig = go.Figure([go.Bar(x=["AUC-ROC", "AUC-PR"], y=[fuga[k]["auc_roc"], fuga[k]["auc_pr"]], name=n, marker_color=c,  # Barras
                            text=[f"{fuga[k]['auc_roc']:.3f}", f"{fuga[k]['auc_pr']:.3f}"], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False) for k, n, c in casos])
    fig.update_layout(barmode="group")  # Agrupadas
    fig.update_yaxes(range=[0, 1.12], title="Área bajo la curva")  # Eje Y
    return barras(tema(fig, 340))  # Tema


# ---------------------------------------------------------------------------
# ML Models
# ---------------------------------------------------------------------------
def fig_roc(res, sel):  # Curvas ROC en prueba
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=TEXTO_3, dash="dot", width=1), name="Azar", hoverinfo="skip"))  # Diagonal
    for n, r in res.items():  # Una curva por modelo
        a = n == sel  # ¿Seleccionado?
        fig.add_trace(go.Scatter(x=r["roc"][0], y=r["roc"][1], mode="lines", name=f"{ETIQUETAS_MODELOS[n]} · {r['test']['auc_roc']:.3f}",  # Curva
                                 line=dict(color=COLORES_MODELOS[n], width=3 if a else 1.5), opacity=1 if a else 0.45,
                                 hovertemplate=f"{ETIQUETAS_MODELOS[n]}<br>FPR %{{x:.3f}} · TPR %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title="Tasa de falsos positivos", range=[0, 1])  # Eje X
    fig.update_yaxes(title="Recall (brotes anticipados)", range=[0, 1.02])  # Eje Y
    fig.update_layout(legend=dict(orientation="v", x=0.42, y=0.05, yanchor="bottom", font=dict(size=11)))  # Leyenda
    return tema(fig, 400)  # Tema


def fig_pr(res, sel, base):  # Curvas precisión-recall en prueba
    fig = go.Figure()  # Figura vacía
    fig.add_hline(y=base, line=dict(color=TEXTO_3, dash="dot", width=1), annotation_text=f"Azar = {base:.3f}", annotation_font=dict(color=TEXTO_2, size=11))  # Azar
    for n, r in res.items():  # Una curva por modelo
        a = n == sel  # ¿Seleccionado?
        fig.add_trace(go.Scatter(x=r["pr"][0], y=r["pr"][1], mode="lines", name=f"{ETIQUETAS_MODELOS[n]} · {r['test']['auc_pr']:.3f}",  # Curva
                                 line=dict(color=COLORES_MODELOS[n], width=3 if a else 1.5), opacity=1 if a else 0.45,
                                 hovertemplate=f"{ETIQUETAS_MODELOS[n]}<br>Recall %{{x:.3f}} · Precisión %{{y:.3f}}<extra></extra>"))
    fig.update_xaxes(title="Recall", range=[0, 1])  # Eje X
    fig.update_yaxes(title="Precisión (alertas correctas)", range=[0, 1.02])  # Eje Y
    fig.update_layout(legend=dict(orientation="v", x=0.02, y=0.05, yanchor="bottom", font=dict(size=11)))  # Leyenda
    return tema(fig, 400)  # Tema


def fig_confusion(m):  # Matriz de confusión
    m = np.array(m)  # [[VN, FP], [FN, VP]]
    texto = [[f"VN<br><b>{m[0, 0]:,}</b>", f"FP<br><b>{m[0, 1]:,}</b>"], [f"FN<br><b>{m[1, 0]:,}</b>", f"VP<br><b>{m[1, 1]:,}</b>"]]  # Texto
    fila = m / m.sum(axis=1, keepdims=True)  # Proporción por fila
    fig = go.Figure(go.Heatmap(z=fila, x=["Predicho: sin brote", "Predicho: brote"], y=["Real: sin brote", "Real: brote"], text=texto, texttemplate="%{text}",  # Celdas
                               textfont=dict(color=TEXTO, size=15), showscale=False, zmin=0, zmax=1, colorscale=SECUENCIAL, xgap=3, ygap=3,
                               hovertemplate="%{y} · %{x}<br>%{z:.1%} de la fila<extra></extra>"))
    fig.update_yaxes(autorange="reversed", showgrid=False)  # Real sin brote arriba
    fig.update_xaxes(showgrid=False, side="top")  # Etiquetas arriba
    return tema(fig, 330, leyenda=False)  # Tema


def fig_sobreajuste(res):  # AUC en entrenamiento, validación temporal y prueba
    nombres = [ETIQUETAS_MODELOS[n].replace(" (original)", "") for n in res]  # Etiquetas
    fig = go.Figure()  # Figura vacía
    for clave, nombre, color in [("train_auc_roc", "Entrenamiento", "#5b6066"), ("cv_auc_roc", "Validación cruzada temporal", AZUL),  # Series
                                 ("val_auc_roc", "Validación 2022", AGUA), ("test", "Prueba 2023–2024", MAGENTA)]:
        y = [r["test"]["auc_roc"] if clave == "test" else r.get(clave) for r in res.values()]  # Valores
        fig.add_trace(go.Bar(x=nombres, y=y, name=nombre, marker_color=color, hovertemplate="%{x}<br>" + nombre + ": %{y:.3f}<extra></extra>"))  # Barras
    fig.update_yaxes(title="AUC-ROC", range=[0.75, 1.0])  # Eje truncado
    fig.update_layout(barmode="group")  # Agrupadas
    return barras(tema(fig, 380))  # Tema


def fig_importancia(imp, color, top=15):  # Importancia por permutación
    items = list(imp.items())[:top][::-1]  # Las más importantes
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[ETIQUETAS.get(k, k) for k, _ in items], orientation="h", marker_color=color,  # Barras
                           hovertemplate="%{y}<br>Caída del AUC: %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="Caída media del AUC-ROC al permutar")  # Eje X
    return barras(tema(fig, 460, leyenda=False))  # Tema


def fig_auc_anual(res):  # AUC-ROC de cada modelo por año de prueba
    fig = go.Figure()  # Figura vacía
    for n, r in res.items():  # Una serie por modelo
        anios = list(r["auc_anual"])  # Años
        fig.add_trace(go.Bar(x=[str(a) for a in anios], y=[r["auc_anual"][a] for a in anios], name=ETIQUETAS_MODELOS[n], marker_color=COLORES_MODELOS[n],  # Barras
                             hovertemplate=ETIQUETAS_MODELOS[n] + "<br>%{x}: %{y:.4f}<extra></extra>"))
    fig.update_yaxes(title="AUC-ROC", range=[0.8, 0.92])  # Eje truncado (diferencias pequeñas)
    fig.update_layout(barmode="group", legend=dict(font=dict(size=10.5)))  # Agrupadas
    return barras(tema(fig, 360))  # Tema


def fig_delong(delong):  # Gráfico de bosque: ΔAUC de HetDS frente a cada modelo
    nombres = list(delong)[::-1]  # Orden
    fig = go.Figure(go.Scatter(x=[delong[n]["dif"] for n in nombres], y=[ETIQUETAS_MODELOS[n] for n in nombres], mode="markers",  # Puntos
                               marker=dict(size=13, color=[MAGENTA if delong[n]["p"] < 0.05 else "#8f8a83" for n in nombres], line=dict(color=SUPERFICIE, width=2)),
                               error_x=dict(type="data", symmetric=False, array=[delong[n]["ic"][1] - delong[n]["dif"] for n in nombres],
                                            arrayminus=[delong[n]["dif"] - delong[n]["ic"][0] for n in nombres], color=TEXTO_3, thickness=2, width=6),
                               hovertemplate="HetDS vs. %{y}<br>ΔAUC = %{x:+.4f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=TEXTO_3, dash="dot"))  # Sin diferencia
    fig.update_xaxes(title="ΔAUC-ROC = HetDS − modelo (prueba 2023–2024)")  # Eje X
    return tema(fig, 300, leyenda=False)  # Tema


def fig_pesos(pesos):  # Peso de cada experto de HetDS semana a semana (área apilada: suman 1)
    fechas = pesos["semana"]  # Semanas
    fig = go.Figure()  # Figura vacía
    for (k, v), color in zip([(k, v) for k, v in pesos.items() if k != "semana"], COLORES_EXPERTOS):  # Un área por experto
        fig.add_trace(go.Scatter(x=fechas, y=v, mode="lines", stackgroup="uno", name=k, line=dict(width=0.5, color=color),  # Área apilada
                                 hovertemplate=k + "<br>%{x}<br>Peso: %{y:.2f}<extra></extra>"))
    fig.add_vrect(x0=fechas[0], x1="2023-01-02", fillcolor="rgba(255,255,255,0.04)", line_width=0, annotation_text="2022: calentamiento",  # Validación
                  annotation_font=dict(color=TEXTO_2, size=11), annotation_position="top left")
    fig.update_yaxes(title="Peso de Hedge", range=[0, 1])  # Eje Y
    return tema(fig, 380)  # Tema


def fig_expertos(res_h):  # AUC de cada experto, del ensamble estático y de HetDS en línea
    datos = list(res_h["expertos_auc"].items()) + [("Ensamble estático (pesos iguales)", res_h["estatico_auc"]), ("HetDS en línea", res_h["test"]["auc_roc"])]  # Barras
    datos = sorted(datos, key=lambda kv: kv[1])  # Orden ascendente
    fig = go.Figure(go.Bar(x=[v for _, v in datos], y=[k for k, _ in datos], orientation="h",  # Barras
                           marker_color=[MAGENTA if k == "HetDS en línea" else ("#9c3f61" if "estático" in k else "#5b6066") for k, _ in datos],
                           text=[f"{v:.4f}" for _, v in datos], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False,
                           hovertemplate="%{y}<br>AUC-ROC = %{x:.4f}<extra></extra>"))
    fig.update_xaxes(title="AUC-ROC en prueba", range=[0.87, 0.91])  # Eje truncado
    return barras(tema(fig, 340, leyenda=False))  # Tema


def fig_seleccion(seleccion):  # AUC-PR de validación 2022 para cada (η, ventana) de HetDS
    etas = sorted({s["eta"] for s in seleccion})  # Valores de η
    vents = sorted({s["ventana"] for s in seleccion})  # Ventanas
    z = [[next(s["val_auc_pr"] for s in seleccion if s["eta"] == e and s["ventana"] == v) for v in vents] for e in etas]  # Matriz
    fig = go.Figure(go.Heatmap(z=z, x=[f"{v} sem." for v in vents], y=[f"η = {e}" for e in etas], colorscale=SECUENCIAL, texttemplate="%{z:.4f}",  # Mapa
                               textfont=dict(color=TEXTO), xgap=3, ygap=3, showscale=False, hovertemplate="%{y}, ventana %{x}<br>AUC-PR 2022 = %{z:.4f}<extra></extra>"))
    fig.update_xaxes(title="Ventana de pérdida", showgrid=False)  # Eje X
    fig.update_yaxes(showgrid=False)  # Eje Y
    return tema(fig, 260, leyenda=False)  # Tema


def fig_aporte(aporte):  # AUC según el bloque de variables (mismo algoritmo)
    items = list(aporte.items())  # Bloques
    fig = go.Figure([go.Bar(x=[k for k, _ in items], y=[v["auc_roc"] for _, v in items], name="AUC-ROC", marker_color=AZUL,  # AUC-ROC
                            text=[f"{v['auc_roc']:.3f}" for _, v in items], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False),
                     go.Bar(x=[k for k, _ in items], y=[v["auc_pr"] for _, v in items], name="AUC-PR", marker_color=BROTE,  # AUC-PR
                            text=[f"{v['auc_pr']:.3f}" for _, v in items], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False)])
    fig.update_layout(barmode="group")  # Agrupadas
    fig.update_yaxes(range=[0, 1.05], title="Área bajo la curva (prueba)")  # Eje Y
    return barras(tema(fig, 360))  # Tema


def fig_ablacion(abl):  # Ablación: AUC-ROC de cada variante con su IC (bootstrap por municipio) frente a GradientBoosting
    items = [(k, v) for k, v in abl.items() if "ic" in v and k != "GradientBoosting (variables base)"]  # Variantes con IC
    fig = go.Figure(go.Scatter(x=[v["dif_gb"] for _, v in items], y=[k for k, _ in items], mode="markers",  # Puntos
                               marker=dict(size=13, color=[MAGENTA if v["p"] < 0.05 else "#8f8a83" for _, v in items], line=dict(color=SUPERFICIE, width=2)),
                               error_x=dict(type="data", symmetric=False, array=[v["ic"][1] - v["dif_gb"] for _, v in items],
                                            arrayminus=[v["dif_gb"] - v["ic"][0] for _, v in items], color=TEXTO_3, thickness=2, width=6),
                               customdata=[[v["auc_roc"], v["p"]] for _, v in items],
                               hovertemplate="%{y}<br>AUC = %{customdata[0]:.4f}<br>Δ vs. GB = %{x:+.4f}<br>p = %{customdata[1]:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=TEXTO_3, dash="dot"))  # Sin diferencia
    fig.update_xaxes(title="ΔAUC-ROC frente a GradientBoosting (IC 95 %, bootstrap por municipio)")  # Eje X
    return tema(fig, 280, leyenda=False)  # Tema


def fig_tiempos(t):  # Tiempo de GridSearchCV y de un ajuste (segundos, escala log)
    n = list(t)  # Modelos
    e = [ETIQUETAS_MODELOS[k].replace(" (original)", "") for k in n]  # Etiquetas
    fig = go.Figure([go.Bar(x=e, y=[t[k]["grid_s"] for k in n], name="Búsqueda completa (GridSearchCV / selección en 2022)", marker_color=AZUL,  # Búsqueda
                            text=[f"{t[k]['grid_s']:.0f} s" for k in n], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False),
                     go.Bar(x=e, y=[t[k]["fit_s"] for k in n], name="Un ajuste con los mejores hiperparámetros", marker_color=MAGENTA,  # Ajuste único
                            text=[f"{t[k]['fit_s']:.1f} s" for k in n], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False)])
    fig.update_yaxes(title="Segundos (escala log)", type="log")  # Escala log
    fig.update_layout(barmode="group")  # Agrupadas
    return barras(tema(fig, 380))  # Tema


def fig_prediccion(t):  # Latencia de predicción por 1 000 municipio-semanas
    n = sorted(t, key=lambda k: t[k]["pred_ms_1000"])  # Orden
    fig = go.Figure(go.Bar(y=[ETIQUETAS_MODELOS[k] for k in n], x=[t[k]["pred_ms_1000"] for k in n], orientation="h", marker_color=[COLORES_MODELOS[k] for k in n],  # Barras
                           text=[f"{t[k]['pred_ms_1000']:.1f} ms" for k in n], textposition="outside", textfont=dict(color=TEXTO_2), cliponaxis=False,
                           hovertemplate="%{y}<br>%{x:.2f} ms por 1 000 filas<extra></extra>"))
    fig.update_xaxes(title="Milisegundos por 1 000 municipio-semanas (escala log)", type="log")  # Escala log
    return barras(tema(fig, 300, leyenda=False))  # Tema


def fig_nemenyi(fr):  # Diagrama de diferencia crítica (Friedman + Nemenyi)
    r = fr["rangos"]  # Rangos medios
    n = sorted(r, key=r.get)  # Del mejor al peor
    cd, mejor = fr["cd"], r[n[0]]  # Diferencia crítica y mejor rango
    fig = go.Figure()  # Figura vacía
    fig.add_shape(type="rect", x0=mejor, x1=mejor + cd, y0=-0.5, y1=len(n) - 0.5, fillcolor="rgba(213,81,129,0.10)", line=dict(width=0))  # Zona sin diferencia con el mejor
    fig.add_trace(go.Scatter(x=[r[k] for k in n], y=[ETIQUETAS_MODELOS[k] for k in n], mode="markers",  # Puntos
                             marker=dict(size=14, color=[COLORES_MODELOS[k] for k in n], line=dict(color=SUPERFICIE, width=2)),
                             error_x=dict(type="constant", value=cd / 2, color=TEXTO_3, thickness=2, width=6), hovertemplate="%{y}<br>Rango medio: %{x:.2f}<extra></extra>"))
    fig.update_xaxes(title=f"Rango medio del AUC semanal en {fr['n_semanas']} semanas (1 = mejor)", range=[0.5, len(n) + 0.5])  # Eje X
    fig.update_yaxes(autorange="reversed")  # Mejor arriba
    return tema(fig, 300, leyenda=False)  # Tema


def fig_particion(n):  # Esquema de la partición temporal
    fig = go.Figure()  # Figura vacía
    seg = [("Datos", 2010, 5, "#5b6066", "2010–2014: solo canal endémico", "Historia para calcular el umbral"),  # (fila, inicio, ancho, color, texto, ayuda)
           ("Datos", 2015, 7, AZUL, f"Entrenamiento 2015–2021 · {n['train']:,}".replace(",", "."), "GridSearchCV con validación temporal"),
           ("Datos", 2022, 1, AGUA, "Val. 2022", "Umbral y hiperparámetros de HetDS"),
           ("Datos", 2023, 2, AMARILLO, f"Prueba · {n['test']:,}".replace(",", "."), "Se usa una sola vez"),
           ("Pliegue 1", 2015, 3, AZUL, "2015–2017", "Entrena"), ("Pliegue 1", 2018, 1, AGUA, "2018", "Valida"),
           ("Pliegue 2", 2015, 4, AZUL, "2015–2018", "Entrena"), ("Pliegue 2", 2019, 1, AGUA, "2019", "Valida"),
           ("Pliegue 3", 2015, 5, AZUL, "2015–2019", "Entrena"), ("Pliegue 3", 2020, 1, AGUA, "2020", "Valida"),
           ("Pliegue 4", 2015, 6, AZUL, "2015–2020", "Entrena"), ("Pliegue 4", 2021, 1, AGUA, "2021", "Valida")]
    for fila, ini, ancho, color, texto, ayuda in seg:  # Barras
        fig.add_trace(go.Bar(y=[fila], x=[ancho], base=[ini], orientation="h", marker_color=color, text=[texto], textposition="inside", insidetextanchor="middle",
                             textfont=dict(color=TEXTO, size=11.5), hovertemplate=f"{texto}<br>{ayuda}<extra></extra>", showlegend=False))
    fig.update_layout(barmode="overlay")  # Barras por su base
    fig.update_xaxes(tickvals=[a + 0.5 for a in range(2010, 2025)], ticktext=[str(a) for a in range(2010, 2025)], range=[2010, 2025], showgrid=False)  # Años
    fig.update_yaxes(autorange="reversed", showgrid=False)  # Primera fila arriba
    return barras(tema(fig, 300, leyenda=False))  # Tema


def fig_medidor(p, umbral):  # Medidor de probabilidad de brote
    fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number=dict(suffix=" %", valueformat=".1f", font=dict(color=TEXTO, size=40)),
                                 gauge=dict(axis=dict(range=[0, 100], tickcolor=TEXTO_3, tickfont=dict(color=TEXTO_3)), bar=dict(color=BROTE if p >= umbral else AZUL, thickness=0.3),
                                            bgcolor="rgba(255,255,255,0.04)", borderwidth=0, threshold=dict(line=dict(color=TEXTO, width=3), thickness=0.85, value=100 * umbral))))
    return tema(fig, 240, leyenda=False)  # Tema
