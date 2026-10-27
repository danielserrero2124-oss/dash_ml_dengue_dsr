"""Figuras de Plotly del dashboard del Segundo Entregable (tema oscuro común).

Cada función recibe piezas de los artefactos precalculados y devuelve una figura. El Jupyter Book usa estas mismas
funciones, de modo que el informe y el dashboard muestran exactamente las mismas figuras.
"""

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
import plotly.graph_objects as go  # Figuras
from plotly.subplots import make_subplots  # Paneles

# ---------------------------------------------------------------------------
# Paleta y etiquetas
# ---------------------------------------------------------------------------
AZUL, NARANJA, AGUA, AMARILLO, MAGENTA, VIOLETA = "#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#8b6fe0"  # Colores categóricos
BROTE, GRIS = "#e66767", "#6b7075"  # Clase brote y referencia
TEXTO, TEXTO_2, TEXTO_3 = "#f4f2ef", "#c4bfb8", "#8f8a83"  # Tintas
REJILLA, SUPERFICIE = "rgba(255,255,255,0.06)", "#17191b"  # Rejilla y superficie
DIVERGENTE = [[0, "#3987e5"], [0.5, "#2f3337"], [1, "#e66767"]]  # Azul – gris – rojo
CALOR = [[0, "#1d2a3a"], [0.35, "#3987e5"], [0.7, "#f2b134"], [1, "#e66767"]]  # Secuencial para mapas
PERIODOS = {"historia": GRIS, "entrenamiento": AZUL, "validación": AGUA, "prueba": AMARILLO, "purgada": "#3a3f44"}  # Colores de la partición
MODELOS = {"Regresión logística": AZUL, "SVM lineal": MAGENTA, "Persistencia (8 sem.)": AGUA, "Persistencia": AMARILLO,
           "Ingenuo estacional": VIOLETA, "Dummy (prior)": GRIS}  # Color fijo por modelo
CUADRANTES = {"Alto-Alto": "#e66767", "Bajo-Bajo": "#3987e5", "Alto-Bajo": "#f0a3b5", "Bajo-Alto": "#9cc3f2", "No significativo": "#3a3f44"}  # LISA
FOCOS = {"Persistente": "#e66767", "Emergente / intensificándose": "#f2b134", "Decreciente": "#3987e5", "Esporádico": "#f0a3b5", "Nunca foco": "#3a3f44"}  # Focos
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]  # Meses

ETIQUETAS = {  # Nombres legibles de las variables
    "lc_l0": "log casos (t)", "lc_l1": "log casos (t − 1)", "lc_l2": "log casos (t − 2)", "lc_l3": "log casos (t − 3)",
    "lc_m4": "log casos, media 4 sem.", "lc_m8": "log casos, media 8 sem.", "lc_m12": "log casos, media 12 sem.",
    "brote_l0": "Brote en t", "brote_m8": "Semanas en brote (8 sem.)", "ratio_umbral": "Casos / umbral objetivo (log)",
    "l_umbral_obj": "Umbral objetivo (log)", "incid": "Incidencia por 100 000",
    "temp_mean_m1": "Temperatura (semana)", "temp_mean_m4": "Temperatura, 4 sem.", "temp_mean_m8": "Temperatura, 8 sem.", "temp_mean_m12": "Temperatura, 12 sem.",
    "prec_total_m1": "Lluvia (semana)", "prec_total_m4": "Lluvia, 4 sem.", "prec_total_m8": "Lluvia, 8 sem.", "prec_total_m12": "Lluvia, 12 sem.",
    "ndvi_mean_m1": "NDVI (semana)", "ndvi_mean_m4": "NDVI, 4 sem.", "ndvi_mean_m8": "NDVI, 8 sem.", "ndvi_mean_m12": "NDVI, 12 sem.",
    "temp_anom_m8": "Anomalía de temperatura", "prec_anom_m8": "Anomalía de lluvia", "ndvi_anom_m8": "Anomalía de NDVI",
    "oni_anom": "ONI (El Niño)", "oni_l8": "ONI hace 8 semanas", "temp_missing": "Temperatura imputada",
    "sem_sin": "Estacionalidad (seno)", "sem_cos": "Estacionalidad (coseno)", "lpob": "log población", "flujo_m4": "Flujo de personas (log)",
    "casos_totales": "Casos semanales", "temp_mean": "Temperatura (°C)", "prec_total": "Precipitación (mm/semana)", "ndvi_mean": "NDVI",
    "oni_total": "ONI total (°C)", "Flujo_in": "Flujo de personas", "poblacion": "Población",
}


def etiqueta(v):  # Nombre legible de una variable
    return ETIQUETAS.get(v, v)  # Si no está, el nombre de la columna


def tema(fig, alto=360, leyenda=True):  # Tema oscuro común
    fig.update_layout(height=alto, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",  # Fondo transparente
                      font=dict(family="Segoe UI, system-ui, sans-serif", color=TEXTO_2, size=12.5),  # Tipografía
                      margin=dict(l=10, r=10, t=36 if leyenda else 14, b=10),  # Márgenes
                      hoverlabel=dict(bgcolor="#202326", bordercolor="#3a3f44", font=dict(color=TEXTO, size=12.5)),  # Globo
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=TEXTO_2)),  # Leyenda
                      showlegend=leyenda, bargap=0.2)  # Barras
    fig.update_xaxes(gridcolor=REJILLA, zeroline=False, linecolor="#2f3337", tickfont=dict(color=TEXTO_3), automargin=True)  # Eje X
    fig.update_yaxes(gridcolor=REJILLA, zeroline=False, linecolor="#2f3337", tickfont=dict(color=TEXTO_3), automargin=True)  # Eje Y
    fig.update_annotations(font=dict(color=TEXTO_2, size=13))  # Títulos de paneles
    return fig  # Figura


def geo(fig, alto=520):  # Mapa de Colombia con tema oscuro
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#4a5056", showland=True, landcolor="#202326",  # Tierra
                    showocean=True, oceancolor="#111416", showlakes=False, bgcolor="rgba(0,0,0,0)", resolution=50,  # Mar y fondo
                    showcoastlines=True, coastlinecolor="#4a5056", projection_type="mercator")  # Costas
    return tema(fig, alto)  # Tema común


def _puntos(mapa, valor, escala=CALOR, titulo="", rango=None, formato=None, tam=7):  # Mapa de símbolos con escala continua
    fig = go.Figure(go.Scattergeo(lat=mapa["latitud"], lon=mapa["longitud"], text=mapa["municipio"], mode="markers",  # Puntos
                                  marker=dict(size=tam, color=valor, colorscale=escala, cmin=None if rango is None else rango[0],
                                              cmax=None if rango is None else rango[1], line=dict(width=0.3, color="#0d0f10"),
                                              colorbar=dict(title=titulo, tickformat=formato, thickness=12, len=0.8, tickfont=dict(color=TEXTO_3))),
                                  hovertemplate="%{text}<br>" + titulo + ": %{marker.color:.3f}<extra></extra>"))
    return geo(fig)  # Tema de mapa


def _categorias(mapa, columna, colores, tam=7):  # Mapa de símbolos por categoría
    fig = go.Figure()  # Figura
    for cat, col in colores.items():  # Cada categoría
        s = mapa[mapa[columna] == cat]  # Municipios
        if len(s):  # Si hay
            fig.add_scattergeo(lat=s["latitud"], lon=s["longitud"], text=s["municipio"], mode="markers", name=f"{cat} ({len(s)})",
                               marker=dict(size=tam if cat not in ("No significativo", "Nunca foco") else 5, color=col, line=dict(width=0.3, color="#0d0f10")),
                               hovertemplate="%{text}<extra>" + cat + "</extra>")
    return geo(fig)  # Tema de mapa


# ---------------------------------------------------------------------------
# Contexto
# ---------------------------------------------------------------------------
def fig_anual(anual):  # Casos por año
    fig = go.Figure(go.Bar(x=anual.index, y=anual.values, marker_color=[BROTE if a == anual.idxmax() else AZUL for a in anual.index],  # Barras
                           hovertemplate="%{x}: %{y:,.0f} casos<extra></extra>"))
    return tema(fig, 300, leyenda=False)  # Tema


def fig_particion(linea):  # Línea de tiempo de la partición
    fig = go.Figure()  # Figura
    for p in ["historia", "entrenamiento", "validación", "prueba", "purgada"]:  # Periodos
        if p in linea:  # Si existe
            fig.add_bar(x=linea.index, y=linea[p], name=p, marker_color=PERIODOS[p], marker_line_width=0)  # Barras apiladas
    fig.update_layout(barmode="stack", bargap=0, yaxis_title="municipios")  # Apiladas
    return tema(fig, 280)  # Tema


# ---------------------------------------------------------------------------
# EDA: calidad y objetivo
# ---------------------------------------------------------------------------
def fig_matriz_faltantes(matriz):  # Matriz de faltantes municipio × año
    fig = go.Figure(go.Heatmap(z=matriz.values, x=matriz.columns, y=np.arange(len(matriz)), colorscale=[[0, "#202326"], [1, BROTE]],  # Mapa de calor
                               zmin=0, zmax=1, colorbar=dict(title="imputada", thickness=12), hovertemplate="año %{x}: %{z:.0%}<extra></extra>"))
    fig.update_yaxes(showticklabels=False, title="municipios (ordenados)")  # Sin etiquetas
    return tema(fig, 420, leyenda=False)  # Tema


def fig_temp_imputada(h):  # Temperatura observada frente a imputada
    fig = go.Figure([go.Bar(x=h["x"], y=h["obs"], name="Observada", marker_color=AZUL, opacity=0.75),  # Observada
                     go.Bar(x=h["x"], y=h["imp"], name="Imputada", marker_color=BROTE, opacity=0.75)])  # Imputada
    fig.update_layout(barmode="overlay", bargap=0, xaxis_title="°C", yaxis_title="densidad")  # Superpuestas
    return tema(fig, 320)  # Tema


def fig_canal(serie, nombre):  # Canal endémico de un municipio
    fig = go.Figure()  # Figura
    fig.add_scatter(x=serie["week_start"], y=serie["umbral"], name="Umbral (Q3 de 5 años)", line=dict(color=TEXTO_3, dash="dot"))  # Umbral
    fig.add_scatter(x=serie["week_start"], y=serie["casos_totales"], name="Casos", line=dict(color=AZUL, width=1.6))  # Casos
    b = serie[serie["brote"] == 1]  # Semanas de brote
    fig.add_scatter(x=b["week_start"], y=b["casos_totales"], mode="markers", name="Brote", marker=dict(color=BROTE, size=6))  # Brotes
    fig.update_layout(yaxis_title="casos por semana", title=dict(text=nombre, font=dict(color=TEXTO, size=14), x=0.01, y=0.98))  # Título
    return tema(fig, 340)  # Tema


def fig_prevalencia_anual(anual):  # Prevalencia por año
    fig = go.Figure(go.Bar(x=anual.index, y=anual["mean"], text=[f"{int(v):,}" for v in anual["sum"]], textposition="outside",  # Barras
                           marker_color=AZUL, hovertemplate="%{x}: %{y:.1%}<extra></extra>"))
    fig.update_yaxes(tickformat=".0%", range=[0, anual["mean"].max() * 1.25])  # Porcentaje
    return tema(fig, 300, leyenda=False)  # Tema


def fig_objetivo_tiempo(nac):  # Proporción nacional de municipios en brote
    fig = make_subplots(specs=[[{"secondary_y": True}]])  # Dos ejes
    fig.add_bar(x=nac.index, y=nac["casos"], name="Casos (eje derecho)", marker_color="#2f3337", secondary_y=True)  # Casos
    fig.add_scatter(x=nac.index, y=nac["brote"], name="Municipios en brote", line=dict(color=BROTE), secondary_y=False)  # Brote
    fig.update_yaxes(tickformat=".0%", secondary_y=False)  # Porcentaje
    fig.update_layout(bargap=0)  # Sin separación
    return tema(fig, 340)  # Tema


def fig_mes_region(mes, region):  # Prevalencia por mes y por región
    fig = make_subplots(1, 2, column_widths=[0.6, 0.4], subplot_titles=("Por mes", "Por región"))  # Dos paneles
    fig.add_bar(x=MESES, y=mes.values, marker_color=AZUL, row=1, col=1, showlegend=False, hovertemplate="%{x}: %{y:.1%}<extra></extra>")  # Meses
    r = region.sort_values()  # Orden
    fig.add_bar(x=r.values, y=r.index, orientation="h", marker_color=AGUA, row=1, col=2, showlegend=False, hovertemplate="%{y}: %{x:.1%}<extra></extra>")  # Regiones
    fig.update_yaxes(tickformat=".0%", row=1, col=1)  # Porcentaje
    fig.update_xaxes(tickformat=".0%", row=1, col=2)  # Porcentaje
    return tema(fig, 300, leyenda=False)  # Tema


def fig_mapa_valor(mapa, columna, titulo, formato=None, rango=None, escala=CALOR):  # Mapa de una variable municipal
    return _puntos(mapa, mapa[columna], escala, titulo, rango, formato)  # Mapa


# ---------------------------------------------------------------------------
# EDA: uni, bi y multivariado
# ---------------------------------------------------------------------------
def fig_histograma(h, v):  # Histograma y caja de una variable
    fig = make_subplots(2, 1, row_heights=[0.78, 0.22], shared_xaxes=True, vertical_spacing=0.04)  # Histograma + caja
    fig.add_bar(x=h["x"], y=h["p"], marker_color=AZUL, marker_line_width=0, row=1, col=1, hovertemplate="%{x:.2f}: %{y:.1%}<extra></extra>")  # Histograma
    fig.add_box(q1=[h["q1"]], median=[h["mediana"]], q3=[h["q3"]], lowerfence=[h["bajo"]], upperfence=[h["alto"]], y=[""], orientation="h",  # Caja
                marker_color=AMARILLO, row=2, col=1, hoverinfo="skip")
    fig.update_yaxes(tickformat=".0%", row=1, col=1)  # Porcentaje
    fig.update_xaxes(title_text=("log(1 + " + etiqueta(v) + ")") if h["log"] else etiqueta(v), row=2, col=1)  # Eje
    fig.update_layout(bargap=0)  # Sin separación
    return tema(fig, 360, leyenda=False)  # Tema


def fig_densidad(par, x, y):  # Densidad conjunta de dos variables
    fig = go.Figure(go.Heatmap(x=par["x"], y=par["y"], z=par["z"], colorscale=[[0, "#17191b"], [0.4, "#184f95"], [1, "#9cc3f2"]], showscale=False,  # Mapa
                               hovertemplate="x %{x:.2f} · y %{y:.2f}<extra></extra>"))
    fig.update_layout(xaxis_title=etiqueta(x), yaxis_title=etiqueta(y))  # Ejes
    return tema(fig, 340, leyenda=False)  # Tema


def fig_correlacion(corr):  # Matriz de correlación de Spearman
    lab = [etiqueta(v) for v in corr.columns]  # Etiquetas
    fig = go.Figure(go.Heatmap(z=corr.values, x=lab, y=lab, zmin=-1, zmax=1, colorscale=DIVERGENTE, colorbar=dict(title="ρ", thickness=12),  # Mapa
                               hovertemplate="%{y} · %{x}: %{z:.2f}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")  # Diagonal descendente
    fig.update_xaxes(tickangle=-45)  # Etiquetas inclinadas
    return tema(fig, 680, leyenda=False)  # Tema


def fig_asociacion(asoc, metrica):  # Asociación de cada predictora con el objetivo
    col = {"Información mutua": "mi", "|r biserial de rangos|": "r_rb", "AUC univariado": "auc"}[metrica]  # Columna
    vals = asoc[col].abs() if col == "r_rb" else asoc[col].apply(lambda a: max(a, 1 - a)) if col == "auc" else asoc[col]  # Valores
    vals = vals.sort_values()  # Orden
    fig = go.Figure(go.Bar(x=vals.values, y=[etiqueta(v) for v in vals.index], orientation="h",  # Barras
                           marker_color=[BROTE if v.startswith(("lc_", "brote", "ratio", "incid")) else AZUL for v in vals.index],
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title=metrica)  # Eje
    return tema(fig, 720, leyenda=False)  # Tema


def fig_deciles(dec, v):  # Tasa de brote por decil de una predictora
    fig = go.Figure(go.Scatter(x=dec["x"], y=dec["tasa"], mode="lines+markers", line=dict(color=BROTE, width=2), marker=dict(size=8),  # Curva
                               hovertemplate="%{x:.2f}: %{y:.1%}<extra></extra>"))
    fig.update_layout(xaxis_title=etiqueta(v) + " (centro del decil)", yaxis_title="brote en 4 semanas")  # Ejes
    fig.update_yaxes(tickformat=".0%")  # Porcentaje
    return tema(fig, 320, leyenda=False)  # Tema


def fig_cajas_region(cajas, v):  # Cajas por región
    fig = go.Figure()  # Figura
    for r, c in cajas[v].items():  # Regiones
        fig.add_box(name=r, q1=[c["q1"]], median=[c["mediana"]], q3=[c["q3"]], lowerfence=[c["bajo"]], upperfence=[c["alto"]], marker_color=AZUL)  # Caja
    fig.update_layout(yaxis_title=etiqueta(v))  # Eje
    return tema(fig, 320, leyenda=False)  # Tema


def fig_pca(var, xy, y):  # Varianza explicada y proyección CP1–CP2
    fig = make_subplots(1, 2, subplot_titles=("Varianza explicada", "Proyección CP1–CP2 (4 000 filas)"))  # Dos paneles
    k = np.arange(1, len(var) + 1)  # Componentes
    fig.add_bar(x=k, y=var, marker_color=AZUL, name="por componente", row=1, col=1)  # Varianza
    fig.add_scatter(x=k, y=np.cumsum(var), line=dict(color=AMARILLO), name="acumulada", row=1, col=1)  # Acumulada
    for val, col, nom in [(0, "#5b6066", "No brote"), (1, BROTE, "Brote en 4 sem.")]:  # Clases
        s = y == val  # Filas
        fig.add_scattergl(x=xy[s, 0], y=xy[s, 1], mode="markers", name=nom, marker=dict(color=col, size=4, opacity=0.55), row=1, col=2)  # Puntos
    fig.update_yaxes(tickformat=".0%", row=1, col=1)  # Porcentaje
    return tema(fig, 400)  # Tema


def fig_tsne(emb, y, reg, color):  # Proyección t-SNE coloreada por objetivo o región
    fig = go.Figure()  # Figura
    if color == "objetivo":  # Por objetivo
        grupos = [(y == 0, "#5b6066", "No brote"), (y == 1, BROTE, "Brote")]  # Clases
    else:  # Por región
        grupos = [(reg == r, c, r) for r, c in zip(sorted(set(reg)), [AGUA, AZUL, AMARILLO, VIOLETA, NARANJA])]  # Regiones
    for s, c, n in grupos:  # Grupos
        fig.add_scattergl(x=emb[s, 0], y=emb[s, 1], mode="markers", name=n, marker=dict(color=c, size=4, opacity=0.65))  # Puntos
    fig.update_xaxes(showticklabels=False)  # Sin escala (no interpretable)
    fig.update_yaxes(showticklabels=False)  # Sin escala
    return tema(fig, 400)  # Tema


def fig_grupos(perfil):  # Mapa de los grupos de municipios
    p = perfil.assign(grupo="Grupo " + perfil["grupo"].astype(str))  # Etiqueta
    return _categorias(p, "grupo", {g: c for g, c in zip(sorted(p["grupo"].unique()), [AZUL, BROTE, AMARILLO, AGUA, VIOLETA, NARANJA, MAGENTA, GRIS])})  # Mapa


# ---------------------------------------------------------------------------
# EDA: fuga y temporal
# ---------------------------------------------------------------------------
def fig_auc_univariado(auc):  # AUC univariado (auditoría de fuga)
    a = auc.sort_values()  # Orden
    col = [BROTE if "obj" in v else GRIS if "identificador" in v else AZUL for v in a.index]  # Colores
    fig = go.Figure(go.Bar(x=a.values, y=[etiqueta(v) for v in a.index], orientation="h", marker_color=col, hovertemplate="%{y}: %{x:.3f}<extra></extra>"))  # Barras
    fig.add_vline(x=0.9, line_dash="dash", line_color=BROTE, annotation_text="alerta 0,9", annotation_font_color=BROTE)  # Umbral de alerta
    fig.update_xaxes(range=[0.45, 1])  # Rango
    return tema(fig, 760, leyenda=False)  # Tema


def fig_serie(nac, mensual, anual, nivel):  # Serie nacional con agregación elegida
    if nivel == "semana":  # Semanal
        fig = go.Figure(go.Scatter(x=nac.index, y=nac.values, line=dict(color=AZUL, width=1.2)))  # Línea
    elif nivel == "mes":  # Mensual
        fig = go.Figure(go.Scatter(x=mensual.index, y=mensual.values, line=dict(color=AGUA, width=1.8)))  # Línea
    else:  # Anual
        fig = go.Figure(go.Bar(x=anual.index, y=anual.values, marker_color=AMARILLO))  # Barras
    fig.update_layout(yaxis_title="casos")  # Eje
    return tema(fig, 320, leyenda=False)  # Tema


def fig_estacionalidad(cal):  # Cajas por mes y por semana epidemiológica
    fig = make_subplots(1, 2, column_widths=[0.35, 0.65], subplot_titles=("Por mes", "Por semana epidemiológica"))  # Dos paneles
    fig.add_box(x=[MESES[m - 1] for m in cal["mes"]], y=cal["log_casos"], marker_color=AZUL, showlegend=False, row=1, col=1)  # Meses
    fig.add_box(x=cal["semana"], y=cal["log_casos"], marker_color=AGUA, showlegend=False, row=1, col=2)  # Semanas
    fig.update_yaxes(title_text="log(1 + casos nacionales)", row=1, col=1)  # Eje
    return tema(fig, 340, leyenda=False)  # Tema


def fig_stl(stl):  # Descomposición STL
    fig = make_subplots(4, 1, shared_xaxes=True, vertical_spacing=0.04, subplot_titles=("Serie (log)", "Tendencia", "Estacionalidad", "Residuo"))  # Cuatro paneles
    for i, (c, col) in enumerate([("serie", AZUL), ("tendencia", BROTE), ("estacional", AGUA), ("residuo", GRIS)]):  # Componentes
        fig.add_scatter(x=stl.index, y=stl[c], line=dict(color=col, width=1.2), showlegend=False, row=i + 1, col=1)  # Línea
    return tema(fig, 640, leyenda=False)  # Tema


def fig_acf(acf, pacf, n):  # ACF y PACF
    banda = 1.96 / np.sqrt(n)  # Banda aproximada
    fig = make_subplots(1, 2, subplot_titles=("ACF (104 rezagos)", "PACF (52 rezagos)"))  # Dos paneles
    fig.add_bar(x=np.arange(len(acf)), y=acf, marker_color=AZUL, row=1, col=1)  # ACF
    fig.add_bar(x=np.arange(len(pacf)), y=pacf, marker_color=AGUA, row=1, col=2)  # PACF
    for c in (1, 2):  # Bandas
        fig.add_hline(y=banda, line_dash="dot", line_color=TEXTO_3, row=1, col=c)  # Superior
        fig.add_hline(y=-banda, line_dash="dot", line_color=TEXTO_3, row=1, col=c)  # Inferior
    return tema(fig, 320, leyenda=False)  # Tema


def fig_ccf(ccf):  # Correlación cruzada clima → casos
    fig = go.Figure()  # Figura
    nombres = {"temp_anom": "Anomalía de temperatura", "prec_anom": "Anomalía de lluvia", "ndvi_anom": "Anomalía de NDVI", "oni_anom": "ONI"}  # Etiquetas
    for v, c in zip(ccf.columns, [BROTE, AZUL, AGUA, AMARILLO]):  # Variables
        fig.add_scatter(x=ccf.index, y=ccf[v], mode="lines+markers", name=nombres[v], line=dict(color=c))  # Curva
    fig.update_layout(xaxis_title="rezago k (semanas): clima en t − k frente a casos en t", yaxis_title="correlación")  # Ejes
    return tema(fig, 340)  # Tema


def fig_regimenes(stl, cortes, nino):  # Regímenes y episodios de El Niño
    ln = stl["serie"]  # Serie
    fig = go.Figure()  # Figura
    for a, b in nino:  # Episodios de El Niño
        fig.add_vrect(x0=a, x1=b, fillcolor=BROTE, opacity=0.10, line_width=0)  # Franjas
    fig.add_scatter(x=ln.index, y=ln.values, name="log casos", line=dict(color=AZUL, width=1))  # Serie
    limites = [ln.index[0]] + list(cortes) + [ln.index[-1]]  # Límites de los regímenes
    for a, b in zip(limites[:-1], limites[1:]):  # Cada régimen
        nivel = ln[(ln.index >= a) & (ln.index < b)].mean()  # Media del régimen
        fig.add_scatter(x=[a, b], y=[nivel, nivel], mode="lines", line=dict(color=AMARILLO, width=3), showlegend=False)  # Nivel
    return tema(fig, 340)  # Tema


def fig_deriva(concepto):  # AUC univariado por año (concept drift)
    fig = go.Figure()  # Figura
    for v, c in zip(concepto.columns, [AZUL, AGUA, AMARILLO, BROTE, VIOLETA]):  # Variables
        fig.add_scatter(x=concepto.index, y=concepto[v], mode="lines+markers", name=etiqueta(v), line=dict(color=c))  # Curva
    fig.add_hline(y=0.5, line_dash="dot", line_color=TEXTO_3)  # Azar
    fig.update_layout(yaxis_title="AUC univariado")  # Eje
    return tema(fig, 340)  # Tema


def fig_calor(calor, regiones):  # Mapa de calor municipio × semana del objetivo
    fig = go.Figure(go.Heatmap(z=calor.values, x=calor.columns, y=np.arange(len(calor)), colorscale=[[0, "#17191b"], [1, BROTE]], showscale=False,  # Mapa
                               hovertemplate="%{x|%Y-%m-%d}<extra></extra>"))
    cambios = [i for i in range(len(regiones)) if i == 0 or regiones[i] != regiones[i - 1]]  # Inicio de cada región
    fig.update_yaxes(tickvals=cambios, ticktext=[regiones[i] for i in cambios], autorange="reversed")  # Regiones
    return tema(fig, 520, leyenda=False)  # Tema


# ---------------------------------------------------------------------------
# EDA: espacial y espacio-temporal
# ---------------------------------------------------------------------------
def fig_cobertura(mapa, fuera):  # Cobertura del panel frente al país
    fig = go.Figure()  # Figura
    fig.add_scattergeo(lat=fuera["latitud"], lon=fuera["longitud"], mode="markers", name="Fuera del panel", marker=dict(size=3, color="#4a5056"))  # Resto
    lp = mapa["lpob"].to_numpy()  # log población
    fig.add_scattergeo(lat=mapa["latitud"], lon=mapa["longitud"], text=mapa["municipio"], mode="markers", name="Municipios del panel",  # Panel
                       marker=dict(size=3 + 1.6 * (lp - lp.min()), color=AZUL, opacity=0.8, line=dict(width=0.3, color="#0d0f10")))
    return geo(fig)  # Mapa


def fig_ripley(r):  # Función L de Ripley
    fig = go.Figure()  # Figura
    fig.add_scatter(x=r["d"], y=r["hi"], line=dict(width=0), showlegend=False)  # Envolvente superior
    fig.add_scatter(x=r["d"], y=r["lo"], fill="tonexty", fillcolor="rgba(143,138,131,0.2)", line=dict(width=0), name="Envolvente de aleatoriedad")  # Envolvente
    fig.add_scatter(x=r["d"], y=r["L"], mode="lines+markers", line=dict(color=BROTE), name="Observada")  # L observada
    fig.add_hline(y=0, line_dash="dot", line_color=TEXTO_3)  # Aleatoriedad
    fig.update_layout(xaxis_title="distancia d (km)", yaxis_title="L(d) − d (km)")  # Ejes
    return tema(fig, 320)  # Tema


def fig_moran(mapa, W_lag):  # Diagrama de dispersión de Moran
    z = mapa["log_incidencia"] - mapa["log_incidencia"].mean()  # Desviaciones
    pend = np.polyfit(z, W_lag, 1)[0]  # Pendiente = I
    fig = go.Figure(go.Scatter(x=z, y=W_lag, mode="markers", text=mapa["municipio"], marker=dict(color=AZUL, size=6, opacity=0.7),  # Puntos
                               hovertemplate="%{text}<extra></extra>", name="municipios"))
    fig.add_scatter(x=[z.min(), z.max()], y=[pend * z.min(), pend * z.max()], mode="lines", line=dict(color=BROTE), name=f"pendiente = I = {pend:.3f}")  # Recta
    fig.add_hline(y=0, line_color="#3a3f44")  # Eje
    fig.add_vline(x=0, line_color="#3a3f44")  # Eje
    fig.update_layout(xaxis_title="log incidencia (centrada)", yaxis_title="promedio de los 6 vecinos")  # Ejes
    return tema(fig, 380)  # Tema


def fig_lisa(mapa):  # Mapa LISA
    return _categorias(mapa, "LISA", CUADRANTES)  # Mapa por cuadrante


def fig_gi(mapa):  # Mapa de Getis-Ord Gi*
    return _puntos(mapa, mapa["Gi"], DIVERGENTE, "Gi* (z)", rango=(-4, 4))  # Mapa divergente


def fig_vario(vario, varianza):  # Semivariograma empírico
    fig = go.Figure(go.Scatter(x=vario["distancia"], y=vario["semivarianza"], mode="markers+lines", line=dict(color=AZUL),  # Puntos
                               marker=dict(size=np.sqrt(vario["pares"]) / 4 + 4)))
    fig.add_hline(y=varianza, line_dash="dot", line_color=BROTE, annotation_text="meseta (varianza)", annotation_font_color=BROTE)  # Meseta
    fig.update_layout(xaxis_title="distancia (km)", yaxis_title="semivarianza")  # Ejes
    return tema(fig, 320, leyenda=False)  # Tema


def fig_dbscan(mapa):  # Agrupamientos DBSCAN
    cats = sorted([c for c in mapa["dbscan"].unique() if c != "ruido"], key=lambda c: int(c[1:]))  # Clústeres
    paleta = [AZUL, BROTE, AMARILLO, AGUA, VIOLETA, NARANJA, MAGENTA, "#9cc3f2", "#f0a3b5", "#7ee787", "#ffa657", "#d2a8ff", "#79c0ff"]  # Colores
    colores = {c: paleta[i % len(paleta)] for i, c in enumerate(cats)}  # Por clúster
    colores["ruido"] = "#3a3f44"  # Aislados
    return _categorias(mapa, "dbscan", colores)  # Mapa


def fig_anual_mapa(anual, anio):  # Incidencia de un año
    a = anual[anual["ANO"] == anio]  # Año elegido
    return _puntos(a, a["log_incidencia"], CALOR, "log incidencia", rango=(0, anual["log_incidencia"].quantile(0.98)))  # Mapa


def fig_regiones(reg):  # Proporción de municipios en brote por región
    fig = go.Figure()  # Figura
    for r, c in zip(reg.columns, [AGUA, AZUL, AMARILLO, VIOLETA, NARANJA]):  # Regiones
        fig.add_scatter(x=reg.index, y=reg[r], name=r, line=dict(color=c, width=1.4))  # Línea
    fig.update_yaxes(tickformat=".0%")  # Porcentaje
    return tema(fig, 340)  # Tema


def fig_focos(focos):  # Focos calientes por categoría
    return _categorias(focos, "categoría", FOCOS)  # Mapa


def fig_moran_semanal(I):  # I de Moran semanal
    fig = go.Figure(go.Scatter(x=I.index, y=I.values, line=dict(color=AZUL, width=1.2)))  # Serie
    fig.add_hline(y=-1 / 375, line_dash="dot", line_color=TEXTO_3)  # Esperanza
    fig.update_layout(yaxis_title="I de Moran")  # Eje
    return tema(fig, 300, leyenda=False)  # Tema


def fig_cruzada(cruz):  # Correlación espacio-temporal
    fig = go.Figure()  # Figura
    fig.add_scatter(x=cruz.index, y=cruz["vecinos"], mode="lines+markers", name="vecinos en t → municipio en t + k", line=dict(color=BROTE))  # Vecindario
    fig.add_scatter(x=cruz.index, y=cruz["propio"], mode="lines+markers", name="municipio en t → municipio en t + k", line=dict(color=AZUL))  # Propio
    fig.update_layout(xaxis_title="k (semanas)", yaxis_title="correlación")  # Ejes
    return tema(fig, 320)  # Tema


# ---------------------------------------------------------------------------
# Modelos base
# ---------------------------------------------------------------------------
def fig_matriz(cm, umbral):  # Matriz de confusión
    txt = [[f"VN<br>{cm[0][0]:,}", f"FP<br>{cm[0][1]:,}"], [f"FN<br>{cm[1][0]:,}", f"VP<br>{cm[1][1]:,}"]]  # Textos
    fig = go.Figure(go.Heatmap(z=cm, x=["Predicho: sin brote", "Predicho: brote"], y=["Real: sin brote", "Real: brote"], text=txt,  # Mapa
                               texttemplate="%{text}", textfont=dict(size=16, color=TEXTO), colorscale=[[0, "#202326"], [1, "#184f95"]], showscale=False))
    fig.update_yaxes(autorange="reversed")  # Real arriba
    return tema(fig, 330, leyenda=False)  # Tema


def fig_curvas(curvas, prevalencia):  # Curvas ROC y precisión-recall
    fig = make_subplots(1, 2, subplot_titles=("Curva ROC", "Curva precisión-recall"))  # Dos paneles
    for n, c in curvas.items():  # Modelos
        fig.add_scatter(x=c["fpr"], y=c["tpr"], name=f"{n} ({c['auc']:.3f})", line=dict(color=MODELOS[n]), row=1, col=1)  # ROC
        fig.add_scatter(x=c["rec"], y=c["prec"], showlegend=False, line=dict(color=MODELOS[n]), row=1, col=2)  # PR
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=TEXTO_3, dash="dot"), showlegend=False, row=1, col=1)  # Azar
    fig.add_hline(y=prevalencia, line_dash="dot", line_color=TEXTO_3, row=1, col=2)  # Azar PR
    fig.update_xaxes(title_text="tasa de falsos positivos", row=1, col=1)  # Ejes
    fig.update_yaxes(title_text="recall", row=1, col=1)
    fig.update_xaxes(title_text="recall", row=1, col=2)
    fig.update_yaxes(title_text="precisión", row=1, col=2)
    tema(fig, 460)  # Tema
    fig.update_layout(legend=dict(y=-0.22, yanchor="top"), margin=dict(b=90))  # Leyenda debajo (no tapa los títulos)
    return fig  # Figura


def fig_calibracion(cal):  # Curvas de calibración por conjunto
    fig = go.Figure()  # Figura
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=TEXTO_3, dash="dot"), name="Perfecta")  # Diagonal
    for k, (nom, col) in {"va": ("Validación 2022", AGUA), "te": ("Prueba 2023–24", AZUL)}.items():  # Conjuntos
        m_, f_, e_ = cal[k]  # Medias, fracciones y ECE
        fig.add_scatter(x=m_, y=f_, mode="lines+markers", line=dict(color=col), name=f"{nom} (ECE {e_:.3f})")  # Curva
    fig.update_layout(xaxis_title="probabilidad predicha", yaxis_title="proporción observada")  # Ejes
    return tema(fig, 360)  # Tema


def fig_acf_residuos(acf):  # ACF de los residuos
    fig = go.Figure(go.Bar(x=np.arange(1, len(acf) + 1), y=acf, marker_color=AZUL))  # Barras
    fig.add_hline(y=1.96 / np.sqrt(100), line_dash="dot", line_color=TEXTO_3)  # Banda (≈ 100 semanas por municipio)
    fig.add_hline(y=-1.96 / np.sqrt(100), line_dash="dot", line_color=TEXTO_3)  # Banda
    fig.update_layout(xaxis_title="rezago (semanas)", yaxis_title="autocorrelación media")  # Ejes
    return tema(fig, 300, leyenda=False)  # Tema


def fig_residuo_mapa(m, resid):  # Residuo medio por municipio
    return _puntos(m.assign(r=resid), resid, DIVERGENTE, "residuo medio", rango=(-0.4, 0.4))  # Mapa divergente


def fig_curva_aprendizaje(curva, metrica):  # Curva de aprendizaje
    fig = go.Figure()  # Figura
    fig.add_scatter(x=curva["filas"], y=curva[f"{metrica} entrenamiento"], mode="lines+markers", name="Entrenamiento", line=dict(color=AZUL))  # Entrenamiento
    fig.add_scatter(x=curva["filas"], y=curva[f"{metrica} validación"], mode="lines+markers", name="Validación 2022", line=dict(color=AGUA))  # Validación
    fig.update_layout(xaxis_title="filas de entrenamiento", yaxis_title=metrica)  # Ejes
    return tema(fig, 320)  # Tema


def fig_coeficientes(coef):  # Coeficientes estandarizados con IC
    c = coef.sort_values("coef")  # Orden
    fig = go.Figure(go.Bar(x=c["coef"], y=[etiqueta(v) for v in c["variable"]], orientation="h",  # Barras
                           marker_color=np.where(c["coef"] > 0, BROTE, AZUL),
                           error_x=dict(type="data", symmetric=False, array=c["IC 97,5 %"] - c["coef"], arrayminus=c["coef"] - c["IC 2,5 %"], color=TEXTO_3),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="log-odds por 1 desviación estándar")  # Eje
    return tema(fig, 760, leyenda=False)  # Tema


def fig_pesos_svm(pesos):  # Pesos de la SVM lineal
    p = pesos.sort_values()  # Orden
    fig = go.Figure(go.Bar(x=p.values, y=[etiqueta(v) for v in p.index], orientation="h", marker_color=np.where(p.values > 0, BROTE, AZUL),  # Barras
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(xaxis_title="peso del hiperplano (por 1 desviación estándar)")  # Eje
    return tema(fig, 760, leyenda=False)  # Tema


def fig_probabilidad(p, umbral):  # Indicador del simulador
    fig = go.Figure(go.Indicator(mode="gauge+number", value=100 * p, number=dict(suffix=" %", font=dict(color=TEXTO, size=40)),  # Medidor
                                 gauge=dict(axis=dict(range=[0, 100], tickcolor=TEXTO_3), bar=dict(color=BROTE if p >= umbral else AZUL),
                                            bgcolor="#202326", borderwidth=0,
                                            threshold=dict(line=dict(color=AMARILLO, width=3), value=100 * umbral))))
    return tema(fig, 260, leyenda=False)  # Tema
