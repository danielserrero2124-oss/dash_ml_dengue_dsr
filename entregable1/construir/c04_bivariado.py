"""Cuaderno 4: análisis bidimensional (sección 2.3 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Análisis bidimensional\n\n"
           "Relaciones entre pares de variables y de cada predictora con el objetivo, con tamaños de efecto y corrección por "
           "comparaciones múltiples. Se usa el conjunto de entrenamiento (2015–2021)."),
        code(PREAMBULO + '''
from scipy import stats  # Correlaciones y pruebas
from sklearn.feature_selection import mutual_info_classif  # Información mutua
from sklearn.metrics import roc_auc_score  # AUC univariado
df = d.panel()  # Panel completo
mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Solo se usa el entrenamiento
tr = tr.assign(fase_enso=pd.cut(tr["oni_anom"], [-9, -0.5, 0.5, 9], labels=["La Niña", "Neutral", "El Niño"]))  # Fase ENSO
obs = tr[tr["temp_missing"] == 0]  # Filas con temperatura medida (para relaciones con la temperatura)
print(f"Entrenamiento: {len(tr):,} filas · con temperatura medida: {len(obs):,}")  # Tamaños'''),

        md("## Numéricas contra numéricas\n\n"
           "Con más de 100 000 filas un diagrama de dispersión se satura; se usan mapas de densidad (equivalentes a *hexbin*)."),
        code('''def densidad(x, y, nx=60, ny=60):  # Histograma bidimensional precalculado
    H, bx, by = np.histogram2d(x, y, bins=[nx, ny])  # Conteos
    return go.Heatmap(x=(bx[:-1] + bx[1:]) / 2, y=(by[:-1] + by[1:]) / 2, z=np.log1p(H.T), colorscale="Blues", showscale=False)  # log(1 + conteo)

pares = [("temp_mean_m4", "prec_total_m4", obs), ("lc_m8", "lc_l0", tr), ("temp_mean_m8", "lc_m8", obs), ("oni_l8", "lc_m8", tr)]  # Pares
fig = make_subplots(1, 4, subplot_titles=[f"{y} vs {x}" for x, y, _ in pares])  # Cuatro paneles
filas = []  # Correlaciones
for i, (x, y, base) in enumerate(pares):  # Cada par
    fig.add_trace(densidad(base[x], base[y]), row=1, col=i + 1)  # Mapa de densidad
    fig.update_xaxes(title_text=x, row=1, col=i + 1)  # Eje
    filas.append({"par": f"{x} – {y}", "Pearson": stats.pearsonr(base[x], base[y])[0], "Spearman": stats.spearmanr(base[x], base[y])[0], "n": len(base)})  # Correlaciones
fig.update_layout(title="Densidad conjunta (escala log del conteo)")  # Título
mostrar(fig, 360)  # Muestra
pd.DataFrame(filas).set_index("par").round(3)'''),
        md("**Interpretación.** La temperatura y la lluvia medidas están poco correlacionadas entre sí. La historia de casos es "
           "muy persistente (la semana actual y la media de 8 semanas tienen correlación alta). La relación de la temperatura "
           "con los casos es débil y no lineal, y el ONI rezagado se asocia positivamente con los casos: El Niño precede a más "
           "transmisión. En el par de casos, Pearson (0,91) supera a Spearman (0,78) porque la correlación lineal está dominada "
           "por las semanas epidémicas, mientras que en las semanas con pocos casos el orden cambia mucho de una semana a otra."),

        md("## Matriz de correlación y multicolinealidad"),
        code('''corr = tr[cf.FEATURES].corr(method="spearman")  # Correlación de Spearman (robusta a asimetría)
fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale="RdBu_r",  # Mapa de calor
                           colorbar=dict(title="ρ")))
fig.update_layout(title="Correlación de Spearman entre las 34 predictoras (entrenamiento)", yaxis_autorange="reversed")  # Formato
mostrar(fig, 760)  # Muestra
altas = corr.where(np.triu(np.ones(corr.shape, bool), 1)).stack()  # Pares sin repetir
altas = altas[altas.abs() > 0.9].sort_values(ascending=False)  # Correlaciones muy altas
print(f"Pares con |ρ| > 0,9: {len(altas)}")  # Conteo
altas.head(15).round(3).to_frame("ρ")'''),
        code('''vif = pd.Series(es.vif(tr[cf.FEATURES].sample(50000, random_state=cf.SEMILLA)), index=cf.FEATURES, name="VIF")  # Factor de inflación
vif.sort_values(ascending=False).round(1).to_frame().T'''),
        md("**Interpretación.** Hay multicolinealidad severa (VIF ≫ 10) entre las medias móviles de una misma variable "
           "(por ejemplo, la temperatura de 4, 8 y 12 semanas) y entre los rezagos del log de casos. Para predecir no es un "
           "problema grave, pero vuelve inestables e ininterpretables los coeficientes individuales de una regresión logística "
           "sin regularizar. **Consecuencia:** el modelo base usa penalización L2 (se elige su fuerza $C$ en la validación "
           "cruzada) y los coeficientes se interpretan por grupos, con intervalos bootstrap."),

        md("## Categóricas contra numéricas"),
        code('''def caja(x):  # Estadísticos de una caja de Tukey
    q1, me, q3 = np.quantile(x, [0.25, 0.5, 0.75])  # Cuartiles
    iqr = q3 - q1  # Rango
    return dict(q1=[q1], median=[me], q3=[q3], lowerfence=[max(x.min(), q1 - 1.5 * iqr)], upperfence=[min(x.max(), q3 + 1.5 * iqr)])  # Caja

regiones = sorted(tr["region"].unique())  # Regiones
fig = make_subplots(1, 2, subplot_titles=("log(1 + casos) por región", "Temperatura medida (media 8 sem.) por región"))  # Dos paneles
for r in regiones:  # Cada región
    fig.add_box(name=r, **caja(tr.loc[tr["region"] == r, "lc_l0"]), marker_color=AZUL, showlegend=False, row=1, col=1)  # Casos
    fig.add_box(name=r, **caja(obs.loc[obs["region"] == r, "temp_mean_m8"]), marker_color=AMBAR, showlegend=False, row=1, col=2)  # Temperatura
mostrar(fig, 380)  # Muestra
pruebas = []  # Kruskal-Wallis
for v, base in [("lc_l0", tr), ("temp_mean_m8", obs), ("prec_total_m8", tr), ("ndvi_mean_m8", tr)]:  # Variables
    h, p, eps2 = es.kruskal([base.loc[base["region"] == r, v] for r in regiones])  # Prueba por región
    pruebas.append({"variable": v, "H": h, "p": p, "ε²": eps2})  # Resultado
kw = pd.DataFrame(pruebas).set_index("variable")  # Tabla
kw["p (Holm)"] = es.holm(kw["p"])  # Corrección
kw.round(4)'''),
        md("**Interpretación.** Todas las diferencias entre regiones son significativas (con este tamaño de muestra siempre lo "
           "son), pero el tamaño de efecto separa lo importante: la temperatura difiere muchísimo entre regiones (ε² ≈ 0,41, "
           "grande), la lluvia moderadamente (ε² ≈ 0,12) y el NDVI poco (ε² ≈ 0,05), mientras que la diferencia en casos es "
           "despreciable (ε² < 0,01). La región es un fuerte determinante "
           "del clima, lo que anticipa confusión: comparar clima y brotes entre municipios mezcla el efecto del clima con el de "
           "la región. Las anomalías climáticas (desviación respecto a la normal del propio municipio) evitan ese problema."),

        md("## Predictoras contra el objetivo\n\n"
           "Para cada predictora: correlación biserial de rangos (Mann-Whitney, brote frente a no brote), AUC univariado, "
           "correlación punto-biserial e información mutua (captura relaciones no lineales). Se corrigen los 34 valores $p$ "
           "por Holm y por FDR (Benjamini-Hochberg)."),
        code('''y = tr[cf.TARGET].to_numpy()  # Objetivo
muestra = tr.sample(40000, random_state=cf.SEMILLA)  # Muestra para la información mutua
mi = mutual_info_classif(muestra[cf.FEATURES], muestra[cf.TARGET], random_state=cf.SEMILLA)  # Información mutua (nats)
filas = []  # Resultados
for j, v in enumerate(cf.FEATURES):  # Cada predictora
    x = tr[v].to_numpy()  # Valores
    rrb, p = es.mann_whitney(x[y == 1], x[y == 0])  # Brote frente a no brote
    filas.append({"variable": v, "r biserial de rangos": rrb, "p": p, "AUC univariado": roc_auc_score(y, x),  # Asociación
                  "r punto-biserial": stats.pointbiserialr(y, x)[0], "información mutua": mi[j]})  # Lineal y no lineal
asoc = pd.DataFrame(filas).set_index("variable")  # Tabla
asoc["p (Holm)"], asoc["q (FDR)"] = es.holm(asoc["p"]), es.fdr_bh(asoc["p"])  # Correcciones
asoc.sort_values("información mutua", ascending=False).round(4)'''),
        code('''orden = asoc.sort_values("información mutua").index  # Orden
fig = make_subplots(1, 2, shared_yaxes=True, subplot_titles=("Información mutua con el objetivo", "|r biserial de rangos|"))  # Dos paneles
fig.add_bar(x=asoc.loc[orden, "información mutua"], y=orden, orientation="h", marker_color=AZUL, showlegend=False, row=1, col=1)  # MI
fig.add_bar(x=asoc.loc[orden, "r biserial de rangos"].abs(), y=orden, orientation="h", marker_color=VERDE, showlegend=False, row=1, col=2)  # Efecto
fig.update_layout(title="Asociación de cada predictora con el brote en cuatro semanas")  # Título
mostrar(fig, 760)  # Muestra
print(f"Predictoras con q (FDR) < 0,05: {(asoc['q (FDR)'] < 0.05).sum()} de {len(asoc)} · con |r_rb| ≥ 0,1: {(asoc['r biserial de rangos'].abs() >= 0.1).sum()}")  # Resumen'''),
        code('''def tasa_por_decil(base, v):  # Tasa de brote por decil de una predictora
    dec = pd.qcut(base[v], 10, duplicates="drop")  # Deciles
    t = base.groupby(dec, observed=True)[cf.TARGET].mean()  # Tasa por decil
    return [iv.mid for iv in t.index], t.values  # Centro del decil y tasa

fig = make_subplots(1, 4, subplot_titles=("Temperatura medida (8 sem.)", "Lluvia (8 sem.)", "ONI rezagado 8 sem.", "NDVI (8 sem.)"))  # Cuatro paneles
for i, (v, base) in enumerate([("temp_mean_m8", obs), ("prec_total_m8", tr), ("oni_l8", tr), ("ndvi_mean_m8", tr)]):  # Variables
    xc, t = tasa_por_decil(base, v)  # Tasa por decil
    fig.add_scatter(x=xc, y=t, mode="lines+markers", line=dict(color=ROJO), showlegend=False, row=1, col=i + 1)  # Curva
fig.update_yaxes(tickformat=".0%")  # Porcentaje
fig.update_layout(title="Tasa de brote en cuatro semanas por decil de cada variable climática (relaciones no lineales)")  # Título
mostrar(fig, 340)  # Muestra'''),
        md("**Interpretación.** La historia de casos domina: las medias móviles del log de casos, la persistencia del brote y "
           "la incidencia tienen la mayor información mutua y AUC univariados de 0,77–0,88. El flujo de personas (|r_rb| ≈ 0,49), "
           "el ONI y su rezago (≈ 0,30–0,32) y la población aportan una asociación moderada. Las variables climáticas locales tienen efectos pequeños (|r_rb| < 0,15) aunque "
           "casi todas resultan significativas tras corregir por Holm y FDR: el tamaño de efecto, no el valor $p$, dice que "
           "importan poco por sí solas. Las curvas por decil muestran por qué: la relación con la temperatura es **no "
           "lineal** (el riesgo sube hacia los 25–28 °C y no sigue subiendo), algo que una regresión logística lineal no "
           "puede representar sin transformaciones."),

        md("## Categóricas contra categóricas"),
        code('''tablas = []  # Asociación con el objetivo
for v in ["MES", "region", "departamento", "fase_enso", "temp_missing"]:  # Categóricas
    V, chi2, gl, p = es.v_cramer(tr[v], tr[cf.TARGET])  # V de Cramér y χ²
    tablas.append({"variable": v, "categorías": tr[v].nunique(), "χ²": chi2, "gl": gl, "p": p, "V de Cramér": V})  # Resultado
cc = pd.DataFrame(tablas).set_index("variable")  # Tabla
cc["p (Holm)"] = es.holm(cc["p"])  # Corrección
cc.round(4)'''),
        code('''pd.crosstab(tr["fase_enso"], tr[cf.TARGET], normalize="index").rename(columns={0: "no brote", 1: "brote"}).round(3)  # Tabla de contingencia (proporción por fila)'''),
        md("**Interpretación.** Todas las asociaciones son significativas, pero débiles salvo la fase ENSO: durante El Niño la "
           "proporción de brotes en cuatro semanas (16,8 %) es casi cuatro veces la de La Niña (4,5 %), con la V de Cramér "
           "más alta (0,17). El mes y la "
           "región explican poco del brote, porque el canal endémico ya compara cada municipio con su propia estacionalidad. "
           "El departamento tiene una asociación mayor que la región, pero usarlo como predictora equivaldría a memorizar lugares.\n\n"
           "## Selección preliminar de variables\n\n"
           "* **Se conservan:** la historia de casos (señal dominante), el ONI y su rezago, el flujo de personas, la población y "
           "las variables climáticas con sus anomalías (efecto pequeño pero plausible y no lineal), la estacionalidad (seno y coseno) "
           "y la bandera de temperatura imputada.\n"
           "* **No se usan:** los conteos por edad y sexo (suman exactamente el total: redundantes), `oni_total` (casi lineal con "
           "`oni_anom` dentro de cada mes), el departamento y los identificadores.\n"
           "* **A vigilar:** la multicolinealidad entre medias móviles (se maneja con regularización L2)."),
    ]
    guardar("04_bivariado", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
