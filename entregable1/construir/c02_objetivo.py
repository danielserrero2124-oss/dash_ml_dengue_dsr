"""Cuaderno 2: reserva del conjunto de prueba y análisis de la variable objetivo (sección 2.1 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Partición y variable objetivo\n\n"
           "## Orden de trabajo: primero se reserva la prueba\n\n"
           "Antes de cualquier decisión basada en los datos se separa el conjunto de prueba. Como hay tiempo, la partición es "
           "**cronológica**: entrenamiento 2015–2021, validación 2022 (umbral de decisión) y prueba 2023–2024. Los años "
           "2010–2014 solo sirven para construir el canal endémico (necesita cinco años previos).\n\n"
           "**Purga con intervalo de separación.** El objetivo de la fila $t$ es el brote de $t+4$. Una fila de diciembre de 2021 "
           "tiene su etiqueta en enero de 2022; si se dejara en el entrenamiento, una etiqueta de la validación se usaría para "
           "entrenar. Por eso una fila solo pertenece a un periodo si su semana $t$ **y** su semana objetivo $t+4$ caen dentro de "
           "él; las cuatro semanas de cada frontera se descartan."),
        code(PREAMBULO + '''
df = d.panel()  # Panel con objetivo, variables, coordenadas y periodo
resumen = df.groupby("periodo").agg(filas=("COD_MUN_N", "size"), desde=("week_start", "min"), hasta=("week_start", "max"))  # Tamaño por periodo
resumen = resumen.loc[["historia", "entrenamiento", "validación", "prueba", "purgada"]]  # Orden
resumen["desde"], resumen["hasta"] = resumen["desde"].dt.date, resumen["hasta"].dt.date  # Fechas legibles
resumen'''),
        code('''linea = df.groupby(["week_start", "periodo"]).size().reset_index(name="n")  # Filas por semana y periodo
fig = px.bar(linea, x="week_start", y="n", color="periodo", color_discrete_map=PERIODO_COLOR,  # Línea de tiempo
             category_orders={"periodo": ["historia", "entrenamiento", "validación", "prueba", "purgada"]})
fig.update_traces(marker_line_width=0)  # Sin bordes
fig.update_layout(bargap=0, title="Partición cronológica (las franjas claras son las semanas purgadas)", xaxis_title="", yaxis_title="municipios")  # Formato
mostrar(fig, 340)  # Muestra
tr_t = df[df["periodo"] == "entrenamiento"]  # Entrenamiento
va_t = df[df["periodo"] == "validación"]  # Validación
print(f"Última semana objetivo del entrenamiento: {(tr_t['week_start'].max() + pd.Timedelta(weeks=4)).date()}")  # Frontera
print(f"Primera semana t de la validación: {va_t['week_start'].min().date()}")  # Frontera'''),
        md("**A partir de aquí, todo el EDA usa solo semanas hasta 2021** (historia + entrenamiento). La validación de 2022 solo "
           "se mira para medir la deriva (sección 2.6), y la prueba 2023–2024 no se abre hasta evaluar el modelo base."),

        md("## Definición del objetivo: el canal endémico\n\n"
           "El umbral $u_{m,t}$ es el tercer cuartil de los casos de las mismas semanas (± 2) de los cinco años anteriores: 25 "
           "valores. Un brote exige superar el umbral **y** tener al menos 3 casos (en municipios sin historia el umbral es 0 y "
           "un solo caso sería un \"brote\")."),
        code('''eda = d.entrenamiento_eda(df)  # Semanas t de 2010–2021
for mun in [76001, 68001]:  # Cali y Bucaramanga
    s = eda[(eda["COD_MUN_N"] == mun) & (eda["ANO"] >= 2015)]  # Serie del municipio en el entrenamiento
    fig = go.Figure()  # Figura
    fig.add_scatter(x=s["week_start"], y=s["umbral"], name="Umbral (Q3 de 5 años)", line=dict(color=GRIS, dash="dot"))  # Umbral
    fig.add_scatter(x=s["week_start"], y=s["casos_totales"], name="Casos", line=dict(color=AZUL))  # Casos
    b = s[s["brote"] == 1]  # Semanas de brote
    fig.add_scatter(x=b["week_start"], y=b["casos_totales"], mode="markers", name="Brote", marker=dict(color=ROJO, size=5))  # Brotes
    fig.update_layout(title=f"Canal endémico de {s['municipio'].iloc[0]} (2015–2021)", yaxis_title="casos por semana")  # Formato
    mostrar(fig, 340)  # Muestra'''),

        md("## Frecuencia de las clases y desbalance"),
        code('''mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Conjuntos ya purgados
clases = pd.DataFrame({"filas": [len(tr), len(va)], "brotes": [int(tr[cf.TARGET].sum()), int(va[cf.TARGET].sum())]},  # Conteos
                      index=["entrenamiento 2015–2021", "validación 2022"])
clases["no brotes"] = clases["filas"] - clases["brotes"]  # Clase mayoritaria
clases["prevalencia (%)"] = clases["brotes"] / clases["filas"] * 100  # Proporción de brotes
clases["razón no brote : brote"] = clases["no brotes"] / clases["brotes"]  # Grado de desbalance
print(f"Prueba 2023–2024: {len(te):,} filas reservadas (su prevalencia no se consulta en el EDA)")  # Prueba sellada
clases.round(2)'''),
        code('''anual = tr.groupby("ANO")[cf.TARGET].agg(["mean", "sum", "size"]).rename(columns={"mean": "prevalencia", "sum": "brotes", "size": "filas"})  # Por año
fig = px.bar(anual.reset_index(), x="ANO", y="prevalencia", text=anual["brotes"].map("{:,}".format), color_discrete_sequence=[AZUL])  # Barras
fig.update_layout(title="Prevalencia del brote por año de entrenamiento (texto: número de brotes)", yaxis_tickformat=".0%", xaxis_title="")  # Formato
mostrar(fig, 340)  # Muestra
anual'''),
        md("**Interpretación.** La clase brote es minoritaria (≈ 10 % en el entrenamiento, una razón de casi 9 a 1), pero con "
           "más de 13 000 casos de la clase minoritaria hay información suficiente para aprender. El desbalance **cambia "
           "mucho entre años**: del 1,7 % en 2017 al 25 % en 2019. Consecuencias:\n\n"
           "* **Métrica.** La exactitud es engañosa (no alertar nunca acierta el 90 % en el entrenamiento). Se priorizan el "
           "AUC-PR (se concentra en la clase brote y su referencia es la prevalencia), el AUC-ROC, el MCC y la calibración (Brier).\n"
           "* **Validación.** Como la prevalencia depende del año, los pliegues deben ser años completos (validación hacia "
           "adelante por años); un pliegue estratificado al azar ocultaría esa variación.\n"
           "* **Desbalance.** Se compara `class_weight=None` contra `balanced` dentro de la validación cruzada en vez de "
           "decidirlo de antemano; no se usa sobremuestreo sintético porque inventaría semanas que rompen la estructura temporal."),

        md("## El objetivo en el tiempo y en el espacio"),
        code('''nac = eda[eda["ANO"] >= 2015].groupby("week_start").agg(brote=("brote", "mean"), casos=("casos_totales", "sum"))  # Serie nacional
fig = make_subplots(specs=[[{"secondary_y": True}]])  # Dos ejes
fig.add_bar(x=nac.index, y=nac["casos"], name="Casos (eje derecho)", marker_color="#cbd5e1", secondary_y=True)  # Casos
fig.add_scatter(x=nac.index, y=nac["brote"], name="Municipios en brote", line=dict(color=ROJO), secondary_y=False)  # Proporción en brote
fig.update_layout(title="Proporción de municipios en brote y casos nacionales (2015–2021)", bargap=0)  # Formato
fig.update_yaxes(tickformat=".0%", secondary_y=False)  # Porcentaje
mostrar(fig, 360)  # Muestra'''),
        code('''mes = tr.groupby("MES")[cf.TARGET].mean()  # Prevalencia por mes
reg = tr.groupby("region")[cf.TARGET].mean().sort_values()  # Prevalencia por región
fig = make_subplots(1, 2, subplot_titles=("Por mes de la semana t", "Por región"))  # Dos paneles
fig.add_bar(x=mes.index, y=mes.values, marker_color=AZUL, row=1, col=1, showlegend=False)  # Meses
fig.add_bar(x=reg.values, y=reg.index, orientation="h", marker_color=VERDE, row=1, col=2, showlegend=False)  # Regiones
fig.update_yaxes(tickformat=".0%", row=1, col=1)  # Porcentaje
fig.update_xaxes(tickformat=".0%", row=1, col=2)  # Porcentaje
fig.update_layout(title="Prevalencia del brote en cuatro semanas (entrenamiento)")  # Título
mostrar(fig, 340)  # Muestra'''),
        code('''mapa = tr.groupby(["COD_MUN_N", "municipio", "departamento", "latitud", "longitud"])[cf.TARGET].mean().reset_index(name="prevalencia")  # Por municipio
fig = px.scatter_geo(mapa, lat="latitud", lon="longitud", color="prevalencia", hover_name="municipio",  # Mapa de puntos
                     hover_data={"departamento": True, "prevalencia": ":.1%", "latitud": False, "longitud": False},
                     color_continuous_scale="YlOrRd", range_color=(0, 0.3))
fig.update_traces(marker=dict(size=7, line=dict(width=0.3, color="#334155")))  # Tamaño de los puntos
fig.update_layout(title="Proporción de semanas con brote en cuatro semanas, por municipio (2015–2021)", coloraxis_colorbar_tickformat=".0%")  # Formato
mostrar(mapa_base(fig), 560)  # Muestra
print(mapa["prevalencia"].describe())  # Heterogeneidad entre municipios'''),
        md("**Interpretación.** El objetivo tiene dinámica propia en el tiempo: hay ondas epidémicas nacionales (2015–2016 y "
           "2019) separadas por años casi sin brotes (2017–2018), así que la prevalencia de la prueba puede ser muy distinta a "
           "la del entrenamiento. La estacionalidad mensual del objetivo es débil (entre 9 % y 12 % según el mes) comparada con "
           "esas ondas: el canal endémico ya descuenta la estacionalidad habitual de cada municipio. En el espacio, la "
           "proporción de semanas en brote varía entre municipios, pero el canal endémico la normaliza: cada municipio se "
           "compara consigo mismo, así que municipios con muchos casos no tienen necesariamente más brotes. La relación del "
           "objetivo con cada predictora se estudia en el análisis bidimensional."),
    ]
    guardar("02_objetivo", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
