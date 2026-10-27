"""Cuaderno 9: componente espacio-temporal (sección 2.8 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Componente espacio-temporal\n\n"
           "¿Cambian los patrones espaciales con el tiempo? ¿Se desplazan los focos? ¿Los brotes se agrupan a la vez en el "
           "espacio y en el tiempo? Se usan las semanas 2015–2021."),
        code(PREAMBULO + '''
df = d.panel()  # Panel completo
eda = d.entrenamiento_eda(df)  # Semanas t de 2010–2021
e = eda[eda["ANO"] >= 2015].copy()  # Periodo con canal endémico
m = d.municipios()  # Centroides
D = sp.haversine(m["latitud"].to_numpy(), m["longitud"].to_numpy())  # Distancias (km)
W = sp.pesos_knn(D, 6)  # k = 6 vecinos, estandarizada por filas
print(f"Filas: {len(e):,} · municipios: {e['COD_MUN_N'].nunique()} · semanas: {e['time_idx'].nunique()}")  # Tamaño'''),

        md("## Mapas por periodo"),
        code('''anual = e.groupby(["ANO", "COD_MUN_N"]).agg(casos=("casos_totales", "sum"), pob=("poblacion", "mean"), brote=("brote", "mean")).reset_index()  # Año × municipio
anual["log_incidencia"] = np.log1p(anual["casos"] / anual["pob"] * 1e5)  # Incidencia anual (log)
anual = anual.merge(m, on="COD_MUN_N")  # Coordenadas
fig = px.scatter_geo(anual, lat="latitud", lon="longitud", color="log_incidencia", animation_frame="ANO", hover_name="municipio",  # Animación por año
                     color_continuous_scale="YlOrRd", range_color=(0, anual["log_incidencia"].quantile(0.98)))
fig.update_traces(marker=dict(size=7, line=dict(width=0.3)))  # Tamaño
fig.update_layout(title="Log incidencia anual por municipio (use el control para recorrer los años)")  # Título
mostrar(mapa_base(fig), 600)  # Muestra'''),
        code('''fig = px.scatter_geo(anual, lat="latitud", lon="longitud", color="brote", facet_col="ANO", facet_col_wrap=4,  # Paneles por año
                     color_continuous_scale="YlOrRd", range_color=(0, 0.5), hover_name="municipio")
fig.update_traces(marker=dict(size=3))  # Puntos pequeños
fig.update_geos(fitbounds="locations", showcountries=True, landcolor="#f8fafc", showland=True)  # Encuadre
fig.update_layout(title="Proporción de semanas en brote por municipio y año", coloraxis_colorbar_tickformat=".0%")  # Título
mostrar(fig, 620)  # Muestra'''),

        md("## Series por zona y comparación entre zonas"),
        code('''reg = e.groupby(["week_start", "region"]).agg(brote=("brote", "mean")).reset_index()  # Proporción en brote por región y semana
fig = px.line(reg, x="week_start", y="brote", color="region", labels={"brote": "municipios en brote", "week_start": ""})  # Líneas
fig.update_layout(title="Proporción de municipios en brote por región", yaxis_tickformat=".0%")  # Formato
mostrar(fig, 380)  # Muestra
piv = reg.pivot(index="week_start", columns="region", values="brote")  # Semana × región
piv.corr().round(2)  # Sincronía entre regiones'''),
        md("**Interpretación.** Las regiones comparten las grandes ondas (2015–2016 y 2019), pero con distinta intensidad y "
           "desfase. La región Andina, la Amazonía, la Orinoquía y el Caribe están bastante sincronizadas (correlaciones de "
           "0,59 a 0,72), mientras que el Pacífico sigue una dinámica casi propia (0,01 con el Caribe y 0,03 con la Orinoquía). El patrón espacial **cambia año a año**: "
           "en 2017 casi no hay brotes en ninguna parte, en 2019 se extienden por gran parte del país, y en los demás años "
           "aparecen focos localizados que cambian de lugar."),

        md("## ¿Se desplazan los focos? Getis-Ord por año y focos emergentes\n\n"
           "Se calcula Gi* de la log incidencia de cada año y se aplica la prueba de Mann-Kendall a la serie de puntajes z de "
           "cada municipio: un foco **persistente** es caliente casi todos los años; uno **emergente** se vuelve caliente con "
           "tendencia creciente; uno **decreciente** pierde intensidad; uno **esporádico** aparece y desaparece."),
        code('''anios = sorted(anual["ANO"].unique())  # Años
gi = pd.DataFrame({a: sp.getis_ord(anual[anual["ANO"] == a].set_index("COD_MUN_N").loc[m["COD_MUN_N"], "log_incidencia"].to_numpy(), D, k=6)  # Gi* por año
                   for a in anios}, index=m["COD_MUN_N"])
caliente = gi > 1.96  # Foco caliente en el año
filas = []  # Clasificación
for mun, z in gi.iterrows():  # Cada municipio
    s, zmk, pmk = es.mann_kendall(z.to_numpy())  # Tendencia de Gi*
    n_cal = int(caliente.loc[mun].sum())  # Años como foco caliente
    if n_cal >= 0.8 * len(anios):  # Casi siempre caliente
        clase = "Persistente"
    elif caliente.loc[mun].iloc[-1] and pmk < 0.1 and zmk > 0:  # Caliente al final y en aumento
        clase = "Emergente / intensificándose"
    elif n_cal > 0 and pmk < 0.1 and zmk < 0:  # Fue caliente y disminuye
        clase = "Decreciente"
    elif n_cal > 0:  # Caliente algunos años sin tendencia
        clase = "Esporádico"
    else:  # Nunca caliente
        clase = "Nunca foco"
    filas.append({"COD_MUN_N": mun, "categoría": clase, "años como foco": n_cal, "tendencia z (MK)": zmk})  # Resultado
focos = pd.DataFrame(filas).merge(m, on="COD_MUN_N")  # Une coordenadas
print(focos["categoría"].value_counts().to_string())  # Conteo
colores = {"Persistente": ROJO, "Emergente / intensificándose": AMBAR, "Decreciente": AZUL, "Esporádico": "#f9a8d4", "Nunca foco": "#e2e8f0"}  # Colores
fig = px.scatter_geo(focos, lat="latitud", lon="longitud", color="categoría", color_discrete_map=colores, hover_name="municipio",  # Mapa
                     hover_data={"años como foco": True, "latitud": False, "longitud": False})
fig.update_traces(marker=dict(size=6))  # Tamaño
fig.update_layout(title="Clasificación de focos calientes 2015–2021 (Gi* anual + Mann-Kendall)")  # Título
mostrar(mapa_base(fig), 560)  # Muestra'''),
        md("**Interpretación.** De 376 municipios, 193 nunca fueron foco caliente, 159 lo fueron de forma esporádica y solo "
           "8 son persistentes; otros 8 son emergentes y 8 decrecientes. Es decir, los focos **se desplazan** entre años. Esa movilidad es la razón por la que la posición geográfica fija (latitud y longitud) "
           "no basta para predecir: importa qué está pasando ahora en el municipio y en sus vecinos."),

        md("## Interacción espacio-tiempo: prueba de Knox\n\n"
           "Eventos: el **inicio** de cada episodio de brote (primera semana en brote tras una semana sin brote). Se cuentan los "
           "pares de inicios cercanos en el espacio y en el tiempo y se comparan con 999 permutaciones de los tiempos."),
        code('''b = e[["COD_MUN_N", "time_idx", "brote"]].sort_values(["COD_MUN_N", "time_idx"])  # Brotes por municipio
b["inicio"] = (b["brote"] == 1) & (b.groupby("COD_MUN_N")["brote"].shift(1) == 0)  # Primera semana de cada episodio
ev = b[b["inicio"]].merge(m, on="COD_MUN_N")  # Eventos con coordenadas
xy_ev = sp.proyectar(ev["latitud"].to_numpy(), ev["longitud"].to_numpy())  # Coordenadas planas (km)
filas = []  # Resultados
for ds, dt in [(25, 2), (50, 4), (100, 4), (100, 8), (200, 8)]:  # Umbrales de cercanía (km, semanas)
    r = sp.knox(xy_ev, ev["time_idx"].to_numpy(), ds, dt, permutaciones=999)  # Prueba de Knox
    filas.append({"distancia (km)": ds, "tiempo (semanas)": dt, "pares observados": r["observado"], "pares esperados": round(r["esperado"]),  # Conteos
                  "razón obs/esp": r["razon"], "p": r["p"]})
print(f"Inicios de brote 2015–2021: {len(ev):,}")  # Número de eventos
pd.DataFrame(filas).round(4)'''),
        md("**Interpretación.** En todas las escalas hay **más pares de inicios cercanos a la vez en espacio y tiempo de lo "
           "esperado** si fueran independientes (razón > 1, p ≈ 0,001): los brotes se contagian o comparten causas entre "
           "municipios vecinos y en semanas cercanas. Es la evidencia más directa de que un municipio cuyo vecino acaba de "
           "entrar en brote tiene más riesgo, y justifica variables de vecindad rezagadas en el tiempo."),

        md("## Autocorrelación espacio-temporal"),
        code('''piv = e.pivot_table(index="time_idx", columns="COD_MUN_N", values="lc").loc[:, m["COD_MUN_N"]]  # Semana × municipio (log casos)
anom = piv - piv.mean()  # Centra cada municipio (quita el nivel propio)
I_sem = []  # I de Moran de cada semana
for t, fila in anom.iterrows():  # Cada semana
    z = fila.to_numpy() - fila.mean()  # Desviaciones de la semana
    I_sem.append((z @ W @ z) / (z @ z) if z @ z > 0 else np.nan)  # I de Moran (W estandarizada: S0 = n)
I_sem = pd.Series(I_sem, index=e.drop_duplicates("time_idx").set_index("time_idx").loc[anom.index, "week_start"])  # Serie semanal
fig = px.line(x=I_sem.index, y=I_sem.values, labels={"x": "", "y": "I de Moran"})  # Línea
fig.add_hline(y=-1 / (len(m) - 1), line_dash="dot", line_color=GRIS, annotation_text="E[I]")  # Esperanza
fig.update_layout(title="I de Moran semanal del log de casos centrado por municipio")  # Título
mostrar(fig, 340)  # Muestra
print(f"I de Moran semanal: mediana {I_sem.median():.3f}, percentil 10 {I_sem.quantile(0.1):.3f}, percentil 90 {I_sem.quantile(0.9):.3f}")  # Resumen'''),
        code('''Wd = pd.DataFrame(W, index=m["COD_MUN_N"], columns=m["COD_MUN_N"])  # Pesos con índices
rezago = anom @ Wd.T  # Rezago espacial de cada municipio en cada semana (promedio de vecinos)
filas = []  # Correlación cruzada espacio-temporal
for k in range(0, 13):  # Rezagos temporales
    propio = anom.shift(-k)  # Municipio en t + k
    filas.append({"k (semanas)": k, "corr(vecinos en t, municipio en t+k)": np.corrcoef(rezago.iloc[: len(anom) - k].to_numpy().ravel(), propio.iloc[: len(anom) - k].to_numpy().ravel())[0, 1],
                  "corr(municipio en t, municipio en t+k)": np.corrcoef(anom.iloc[: len(anom) - k].to_numpy().ravel(), propio.iloc[: len(anom) - k].to_numpy().ravel())[0, 1]})
cruz = pd.DataFrame(filas).set_index("k (semanas)")  # Tabla
fig = px.line(cruz, markers=True)  # Líneas
fig.update_layout(title="¿Anticipa el vecindario lo que pasará en el municipio?", yaxis_title="correlación", legend_title="")  # Formato
mostrar(fig, 340)  # Muestra
cruz.round(3)'''),
        md("**Interpretación.** La autocorrelación espacial de los casos (ya centrados por municipio) es positiva casi todas las "
           "semanas (I de Moran semanal con mediana 0,15) y sube durante las epidemias: los vecinos se desvían de su normalidad al mismo tiempo. Además, el "
           "promedio de los vecinos en la semana $t$ está correlacionado con lo que pasará en el municipio en $t+4$ "
           "(r ≈ 0,41), aunque menos que la propia historia del municipio (r ≈ 0,62): el vecindario **añade** información, pero no reemplaza la historia propia.\n\n"
           "## Consecuencias para el modelado\n\n"
           "* **Objetivo de predicción.** El uso principal es predecir el **futuro** de los municipios vigilados. Como la "
           "dinámica espacio-temporal también importa para municipios sin historia, se reportan tres esquemas por separado:\n"
           "  1. **Futuro, mismas zonas**: entrenar 2015–2021 y evaluar 2023–2024 en los 376 municipios (esquema principal).\n"
           "  2. **Zonas nuevas, mismo periodo**: bloques por departamento dentro de 2015–2021, con 50 km de separación.\n"
           "  3. **Futuro y zonas nuevas**: entrenar 2015–2021 sin un bloque de departamentos y evaluar ese bloque en 2023–2024.\n"
           "* **Riesgo de fuga por vecindad.** Una variable de vecindad calculada con datos de la semana objetivo de los vecinos "
           "sería fuga; las variables de vecindad deben usar solo semanas ≤ $t$. En la validación por zonas nuevas, los vecinos "
           "del entrenamiento a menos de 50 km de la prueba se excluyen para no filtrar información a través de la autocorrelación."),
    ]
    guardar("09_espacio_temporal", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
