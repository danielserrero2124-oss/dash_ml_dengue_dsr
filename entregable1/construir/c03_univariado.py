"""Cuaderno 3: análisis unidimensional (sección 2.2 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Análisis unidimensional\n\n"
           "Todas las estadísticas usan las semanas 2010–2021 (historia y entrenamiento); la prueba sigue sellada."),
        code(PREAMBULO + '''
from scipy import stats  # Pruebas de normalidad
df = d.panel()  # Panel completo
eda = d.entrenamiento_eda(df)  # Semanas t de 2010–2021
mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Entrenamiento (para las variables derivadas)
print(f"Filas del EDA: {len(eda):,}")  # Tamaño'''),

        md("## Variables numéricas originales: estadísticos, forma y normalidad\n\n"
           "Las pruebas de normalidad (D'Agostino–Pearson y Jarque–Bera) se aplican a una muestra de 5 000 filas: con cientos de "
           "miles de filas rechazarían cualquier desviación mínima, así que se leen junto con la asimetría y la curtosis."),
        code('''num = ["casos_totales", "casos_0_4", "casos_5_14", "casos_15_64", "casos_65_plus", "casos_m", "casos_f",  # Conteos
       "temp_mean", "prec_total", "ndvi_mean", "oni_anom", "oni_total", "Flujo_in", "poblacion"]  # Continuas
muestra = eda[num].sample(5000, random_state=cf.SEMILLA)  # Muestra para las pruebas
filas = []  # Resultados
for v in num:  # Cada variable
    x = eda[v]  # Serie
    q1, q3 = x.quantile([0.25, 0.75])  # Cuartiles
    iqr = q3 - q1  # Rango intercuartílico
    filas.append({"variable": v, "media": x.mean(), "mediana": x.median(), "desv. est.": x.std(),  # Centro y dispersión
                  "p1": x.quantile(0.01), "p5": x.quantile(0.05), "p25": q1, "p75": q3, "p95": x.quantile(0.95), "p99": x.quantile(0.99),  # Percentiles
                  "asimetría": x.skew(), "curtosis": x.kurt(),  # Forma
                  "atípicos IQR (%)": ((x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)).mean() * 100,  # Regla de Tukey
                  "p normalidad (K²)": stats.normaltest(muestra[v])[1], "p Jarque-Bera": stats.jarque_bera(muestra[v])[1]})  # Normalidad
uni = pd.DataFrame(filas).set_index("variable")  # Tabla
uni.round(3)'''),
        code('''def histograma(x, bins=60, log=False):  # Histograma precalculado (la figura no embebe todos los datos)
    x = np.log1p(x) if log else x  # Escala logarítmica opcional
    conteo, bordes = np.histogram(x, bins=bins)  # Conteos
    return (bordes[:-1] + bordes[1:]) / 2, conteo / conteo.sum()  # Centros y proporciones

paneles = [("casos_totales", True), ("poblacion", True), ("Flujo_in", True), ("temp_mean", False),  # (variable, ¿log?)
           ("prec_total", False), ("ndvi_mean", False), ("oni_anom", False), ("oni_total", False)]
fig = make_subplots(2, 4, subplot_titles=[f"log(1 + {v})" if lg else v for v, lg in paneles])  # Rejilla
for i, (v, lg) in enumerate(paneles):  # Cada variable
    xc, pr = histograma(eda[v], log=lg)  # Histograma
    fig.add_bar(x=xc, y=pr, marker_color=AZUL, showlegend=False, row=i // 4 + 1, col=i % 4 + 1)  # Barras
fig.update_layout(title="Distribuciones de las variables numéricas originales (2010–2021)", bargap=0)  # Formato
mostrar(fig, 520)  # Muestra'''),
        code('''fig = go.Figure()  # Diagramas de caja precalculados
for v in ["temp_mean", "prec_total", "ndvi_mean", "oni_anom"]:  # Variables en escala comparable tras estandarizar
    x = (eda[v] - eda[v].mean()) / eda[v].std()  # Estandariza para un mismo eje
    q1, me, q3 = x.quantile([0.25, 0.5, 0.75])  # Cuartiles
    iqr = q3 - q1  # Rango intercuartílico
    fig.add_box(name=v, q1=[q1], median=[me], q3=[q3], lowerfence=[max(x.min(), q1 - 1.5 * iqr)],  # Caja de Tukey
                upperfence=[min(x.max(), q3 + 1.5 * iqr)], marker_color=AZUL, showlegend=False)
    fig.add_scatter(x=[v, v], y=[x.min(), x.max()], mode="markers", marker=dict(color=ROJO, size=7, symbol="x"), showlegend=False)  # Extremos
fig.update_layout(title="Diagramas de caja estandarizados (x: mínimo y máximo)", yaxis_title="desviaciones estándar")  # Formato
mostrar(fig, 380)  # Muestra'''),
        md("**Interpretación.**\n\n"
           "* **Casos.** Distribución de conteo extremadamente asimétrica (asimetría > 15, curtosis en cientos): la mediana es 0 "
           "y la media unos pocos casos. Es la firma de un fenómeno epidémico con muchas semanas tranquilas y pocas explosivas. "
           "Necesita $\\log(1+x)$; los grupos de edad y sexo tienen la misma forma y son colineales con el total (suman exactamente el total).\n"
           "* **Población y flujo.** Colas largas por las grandes ciudades: también se usan en escala logarítmica.\n"
           "* **Temperatura.** Bimodal por la imputación (un pico artificial en 22,4 °C y la temperatura real de tierra caliente "
           "alrededor de 26–28 °C).\n"
           "* **Precipitación.** Asimétrica a la derecha, con semanas secas frecuentes y lluvias extremas en el Pacífico.\n"
           "* **NDVI.** Ligeramente asimétrico a la izquierda (zonas urbanas y secas con menos vegetación).\n"
           "* **ONI.** Una sola serie nacional repetida en todos los municipios: su variabilidad es solo temporal.\n"
           "* **Normalidad.** Ninguna variable es normal (todas las pruebas rechazan aun con 5 000 filas). No es un problema para "
           "la regresión logística, que no supone normalidad de las predictoras, pero sí justifica las transformaciones y el uso "
           "de pruebas no paramétricas en el análisis bidimensional."),

        md("## Variables predictoras derivadas (entrenamiento)\n\n"
           "Las 34 predictoras se describen en el capítulo de preprocesamiento; aquí se resume su distribución."),
        code('''der = tr[cf.FEATURES].describe(percentiles=[0.05, 0.5, 0.95]).T  # Estadísticos
der["asimetría"] = tr[cf.FEATURES].skew()  # Forma
der["curtosis"] = tr[cf.FEATURES].kurt()  # Colas
der["únicos"] = tr[cf.FEATURES].nunique()  # Cardinalidad
der.round(3)'''),
        md("**Interpretación.** Tras el logaritmo, la historia de casos (`lc_*`) sigue siendo asimétrica porque la mayoría de las "
           "semanas tienen 0 casos, pero las colas se reducen de cientos a unas pocas unidades de curtosis. La incidencia "
           "(`incid`) conserva una cola larga (municipios pequeños con un brote tienen incidencias altísimas); el estandarizado "
           "del Pipeline no la elimina, y por eso el modelo también recibe la versión logarítmica de los casos."),

        md("## Variables categóricas y de calendario"),
        code('''eda = eda.assign(fase_enso=pd.cut(eda["oni_anom"], [-9, -0.5, 0.5, 9], labels=["La Niña", "Neutral", "El Niño"]))  # Fase ENSO
cat = {"región": eda.groupby("region")["COD_MUN_N"].nunique(),  # Municipios por región
       "departamento": eda.groupby("departamento")["COD_MUN_N"].nunique().sort_values(),  # Municipios por departamento
       "fase ENSO (semanas)": eda.drop_duplicates("time_idx")["fase_enso"].value_counts(),  # Semanas por fase
       "temperatura imputada (filas)": eda["temp_mean_missing"].value_counts()}  # Bandera
card = pd.DataFrame({"cardinalidad": {"MES": eda["MES"].nunique(), "SEMANA": eda["SEMANA"].nunique(), "región": eda["region"].nunique(),  # Cardinalidad
                                      "departamento": eda["departamento"].nunique(), "fase ENSO": 3, "temp_mean_missing": 2}})
card'''),
        code('''dep = cat["departamento"]  # Municipios por departamento
fig = make_subplots(1, 3, column_widths=[0.5, 0.25, 0.25], subplot_titles=("Municipios por departamento", "Municipios por región", "Semanas por fase ENSO"))  # Tres paneles
fig.add_bar(x=dep.values, y=dep.index, orientation="h", marker_color=np.where(dep.values < 3, ROJO, AZUL), showlegend=False, row=1, col=1)  # Departamentos (rojo: raros)
fig.add_bar(x=cat["región"].index, y=cat["región"].values, marker_color=VERDE, showlegend=False, row=1, col=2)  # Regiones
fe = cat["fase ENSO (semanas)"]  # Fases
fig.add_bar(x=fe.index.astype(str), y=fe.values, marker_color=[AZUL if f == "La Niña" else GRIS if f == "Neutral" else ROJO for f in fe.index], showlegend=False, row=1, col=3)  # Fases
fig.update_layout(title="Frecuencias de las variables categóricas (rojo: departamentos con menos de 3 municipios)")  # Título
mostrar(fig, 620)  # Muestra
semanas = eda.drop_duplicates("time_idx")["SEMANA"].value_counts().sort_index()  # Semanas por número de semana
print(f"Semana 53: {semanas.get(53, 0)} ocurrencias (años 2015 y 2020) frente a {semanas.drop(53).min()}–{semanas.drop(53).max()} del resto")  # Categoría rara
print("Departamentos con menos de 3 municipios en el panel:", ", ".join(dep[dep < 3].index))  # Raros'''),
        md("**Interpretación y tratamiento de categorías raras.**\n\n"
           "* **Departamento** (32 categorías) tiene varias categorías raras (departamentos amazónicos y San Andrés con 1–2 "
           "municipios). Codificarlo como variable indicadora haría que el modelo memorice lugares y no generalice a zonas "
           "nuevas; **no se usa como predictora**. Se usa como bloque de la validación espacial y para agregar la región.\n"
           "* **Región** (5 categorías) no tiene categorías raras; se usa en el EDA para describir la heterogeneidad espacial.\n"
           "* **Semana 53** solo ocurre en 2015 y 2020. Se codifica con seno y coseno de $2\\pi\\,\\text{semana}/52$, de modo que "
           "la semana 53 queda junto a la semana 1, sin crear una categoría aparte.\n"
           "* **Fase ENSO**: de 626 semanas, 287 son neutrales, 179 de La Niña y 160 de El Niño; ninguna fase es rara. Se "
           "conserva el ONI continuo (no las fases) para no perder información.\n"
           "* **Temperatura imputada**: 64 % de las filas; es la bandera del mecanismo de faltantes y se conserva."),
    ]
    guardar("03_univariado", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
