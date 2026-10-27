"""Cuaderno 7: componente temporal (sección 2.6 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Componente temporal\n\n"
           "Validación de la variable de tiempo, estacionalidad, descomposición, estacionariedad, dependencia temporal, "
           "cambios de régimen, deriva y heterogeneidad del panel. Se usan las semanas 2010–2021; la validación 2022 solo "
           "entra en la medición de deriva."),
        code(PREAMBULO + '''
from statsmodels.tsa.seasonal import STL  # Descomposición estacional
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf  # Autocorrelación y estacionariedad
from scipy import stats  # Kolmogórov-Smirnov
from sklearn.metrics import roc_auc_score  # AUC por año
warnings.simplefilter("ignore")  # statsmodels reactiva sus avisos de interpolación al importarse
df = d.panel()  # Panel completo
eda = d.entrenamiento_eda(df)  # Semanas t de 2010–2021
mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Particiones purgadas'''),

        md("## Validación de la variable temporal"),
        code('''fechas = eda["week_start"]  # Fechas de inicio de semana
pasos = eda.sort_values(["COD_MUN_N", "time_idx"]).groupby("COD_MUN_N")["week_start"].diff().dt.days.dropna()  # Paso entre semanas consecutivas
esperadas = (fechas.max() - fechas.min()).days // 7 + 1  # Semanas esperadas en el periodo
anio_iso = fechas.dt.isocalendar().year  # Año ISO de la fecha
pd.Series({
    "Tipo de dato": str(fechas.dtype),
    "Zona horaria": "sin zona (fecha civil de inicio de semana; no hay horas)",
    "Día de la semana de week_start": fechas.dt.day_name().unique()[0],
    "Frecuencia": f"semanal ({pasos.value_counts().index[0]:.0f} días en el {pasos.value_counts(normalize=True).iloc[0]:.0%} de los pasos)",
    "Marcas de tiempo duplicadas (municipio, semana)": int(eda.duplicated(["COD_MUN_N", "week_start"]).sum()),
    "Semanas esperadas / presentes por municipio": f"{esperadas} / {eda.groupby('COD_MUN_N')['week_start'].nunique().min()}",
    "Huecos (semanas faltantes)": int(esperadas - eda.groupby("COD_MUN_N")["week_start"].nunique().min()),
    "Cobertura": f"{fechas.min().date()} a {fechas.max().date()}",
    "Años con semana 53": ", ".join(map(str, sorted(eda.loc[eda["SEMANA"] == 53, "ANO"].unique()))),
    "Filas donde ANO ≠ año ISO de la fecha": int((eda["ANO"] != anio_iso).sum()),
}, name="valor").to_frame()'''),
        md("**Interpretación.** La serie es regular: una fecha por lunes, pasos exactos de 7 días, sin duplicados ni huecos. "
           "El calendario es el de semanas epidemiológicas (coincide con el año ISO), con semana 53 en 2015 y 2020. No hay "
           "zona horaria porque el dato es semanal."),

        md("## Serie completa y agregaciones"),
        code('''nac = eda.groupby("week_start")["casos_totales"].sum()  # Casos nacionales por semana
mensual = nac.resample("MS").sum()  # Por mes
anual = eda.groupby("ANO")["casos_totales"].sum()  # Por año
fig = make_subplots(3, 1, subplot_titles=("Casos por semana", "Casos por mes", "Casos por año"), vertical_spacing=0.09)  # Tres paneles
fig.add_scatter(x=nac.index, y=nac.values, line=dict(color=AZUL, width=1), showlegend=False, row=1, col=1)  # Semanal
fig.add_scatter(x=mensual.index, y=mensual.values, line=dict(color=VERDE), showlegend=False, row=2, col=1)  # Mensual
fig.add_bar(x=anual.index, y=anual.values, marker_color=AMBAR, showlegend=False, row=3, col=1)  # Anual
fig.update_layout(title="Casos de dengue en los 376 municipios (2010–2021)")  # Título
mostrar(fig, 720)  # Muestra'''),
        code('''ln = np.log1p(nac)  # Escala logarítmica
tabla = pd.DataFrame({"log_casos": ln.values, "mes": ln.index.month, "semana": ln.index.isocalendar().week.values}, index=ln.index)  # Calendario
fig = make_subplots(1, 2, column_widths=[0.35, 0.65], subplot_titles=("Por mes", "Por semana epidemiológica"))  # Dos paneles
fig.add_box(x=tabla["mes"], y=tabla["log_casos"], marker_color=AZUL, showlegend=False, row=1, col=1)  # Meses
fig.add_box(x=tabla["semana"], y=tabla["log_casos"], marker_color=VERDE, showlegend=False, row=1, col=2)  # Semanas
fig.update_layout(title="Estacionalidad: log(1 + casos nacionales) por ciclo")  # Título
mostrar(fig, 380)  # Muestra
print(tabla.groupby("mes")["log_casos"].median().round(2).to_dict())  # Mediana por mes'''),

        md("## Descomposición STL (tendencia, estacionalidad y residuo)"),
        code('''stl = STL(ln, period=52, robust=True).fit()  # Descomposición robusta con periodo anual
fig = make_subplots(4, 1, shared_xaxes=True, subplot_titles=("Serie (log)", "Tendencia", "Estacionalidad", "Residuo"), vertical_spacing=0.05)  # Cuatro paneles
for i, (s, col) in enumerate([(ln, AZUL), (stl.trend, ROJO), (stl.seasonal, VERDE), (stl.resid, GRIS)]):  # Componentes
    fig.add_scatter(x=s.index, y=s.values, line=dict(color=col, width=1.2), showlegend=False, row=i + 1, col=1)  # Línea
fig.update_layout(title="Descomposición STL de log(1 + casos nacionales)")  # Título
mostrar(fig, 760)  # Muestra
f_tend = max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.trend))  # Fuerza de la tendencia (Wang et al., 2006)
f_est = max(0, 1 - np.var(stl.resid) / np.var(stl.resid + stl.seasonal))  # Fuerza de la estacionalidad
print(f"Fuerza de la tendencia: {f_tend:.2f} · fuerza de la estacionalidad: {f_est:.2f} (0 = nula, 1 = total)")  # Resultado'''),
        md("**Interpretación.** La serie nacional está dominada por la **tendencia de varios años** (ondas epidémicas de 2010, "
           "2013 y 2019 con valles en 2011–2012 y 2017–2018), no por un ciclo anual. La estacionalidad existe (algo más de "
           "casos en el primer semestre, tras la primera temporada de lluvias) pero su fuerza es muy baja frente a la "
           "tendencia. Para el modelo esto significa que el calendario (seno y coseno de la semana) aporta poco y que la "
           "información relevante está en el **nivel reciente** de la serie: rezagos y medias móviles de casos."),

        md("## Estacionariedad: medias y varianzas móviles, ADF y KPSS"),
        code('''fig = make_subplots(specs=[[{"secondary_y": True}]])  # Dos ejes
fig.add_scatter(x=ln.index, y=ln.rolling(52).mean(), name="Media móvil 52 sem.", line=dict(color=AZUL))  # Media móvil
fig.add_scatter(x=ln.index, y=ln.rolling(52).std(), name="Desviación móvil 52 sem. (eje derecho)", line=dict(color=ROJO), secondary_y=True)  # Desviación móvil
fig.update_layout(title="Media y desviación móviles de log(1 + casos nacionales)")  # Título
mostrar(fig, 340)  # Muestra
filas = []  # Pruebas
for nombre, s in [("Nivel: log casos", ln), ("Primera diferencia", ln.diff().dropna())]:  # Serie y su diferencia
    adf = adfuller(s, autolag="AIC")  # H0: raíz unitaria (no estacionaria)
    kp = kpss(s, regression="c", nlags="auto")  # H0: estacionaria
    filas.append({"serie": nombre, "ADF estadístico": adf[0], "ADF p": adf[1], "KPSS estadístico": kp[0], "KPSS p": kp[1]})  # Resultado
pd.DataFrame(filas).set_index("serie").round(4)'''),
        code('''p_adf, p_kpss = [], []  # Pruebas por municipio
for _, g in eda[eda["ANO"] >= 2015].groupby("COD_MUN_N"):  # Cada municipio (2015–2021)
    x = np.log1p(g["casos_totales"].to_numpy())  # Serie del municipio
    if x.std() > 0:  # Si la serie varía
        p_adf.append(adfuller(x, maxlag=8, autolag=None)[1])  # ADF con 8 rezagos
        p_kpss.append(kpss(x, regression="c", nlags=8)[1])  # KPSS con 8 rezagos
p_adf, p_kpss = np.array(p_adf), np.array(p_kpss)  # Vectores
print(f"Municipios: {len(p_adf)} · ADF rechaza raíz unitaria (p < 0,05): {np.mean(p_adf < 0.05):.0%} · "
      f"KPSS rechaza estacionariedad (p < 0,05): {np.mean(p_kpss < 0.05):.0%}")  # Resumen'''),
        md("**Interpretación.** En la serie nacional las dos pruebas no se contradicen de forma concluyente: ADF no rechaza la "
           "raíz unitaria y KPSS tampoco rechaza la estacionariedad, el caso típico de una serie **muy persistente** con "
           "ondas largas, donde las pruebas tienen poca potencia. La primera diferencia es claramente estacionaria. A nivel "
           "municipal la mayoría de las series rechaza la raíz unitaria, pero muchas también rechazan la estacionariedad. "
           "Consecuencias: (1) las medias y varianzas cambian con el tiempo, así que el desempeño de un año no garantiza el de "
           "otro; (2) se usan variables relativas al propio municipio (log de casos frente al umbral, anomalías climáticas) "
           "en lugar de niveles absolutos; y (3) la validación debe recorrer varios años distintos."),

        md("## Dependencia temporal: ACF, PACF y correlación cruzada"),
        code('''rez = 104  # Dos años de rezagos
a, p_ = acf(ln, nlags=rez), pacf(ln, nlags=52)  # ACF y PACF
banda = 1.96 / np.sqrt(len(ln))  # Banda de confianza aproximada
fig = make_subplots(1, 2, subplot_titles=("ACF de log casos nacionales", "PACF"))  # Dos paneles
fig.add_bar(x=np.arange(rez + 1), y=a, marker_color=AZUL, showlegend=False, row=1, col=1)  # ACF
fig.add_bar(x=np.arange(53), y=p_, marker_color=VERDE, showlegend=False, row=1, col=2)  # PACF
for col in (1, 2):  # Bandas
    fig.add_hline(y=banda, line_dash="dot", line_color=GRIS, row=1, col=col)  # Superior
    fig.add_hline(y=-banda, line_dash="dot", line_color=GRIS, row=1, col=col)  # Inferior
fig.update_layout(title="Autocorrelación de la serie nacional (bandas: ±1,96/√n)")  # Título
mostrar(fig, 360)  # Muestra
print(f"ACF en el rezago 4: {a[4]:.2f} · rezago 52: {a[52]:.2f} · PACF rezago 1: {p_[1]:.2f}, rezago 2: {p_[2]:.2f}")  # Valores clave'''),
        code('''base = eda[eda["ANO"] >= 2011].copy()  # Desde 2011 (la normal y los rezagos ya existen)
base["lc_c"] = base["lc"] - base.groupby("COD_MUN_N")["lc"].transform("mean")  # Casos centrados por municipio
g = base.groupby("COD_MUN_N")  # Agrupación
ccf = {}  # Correlación cruzada media dentro de los municipios
for v in ["temp_anom", "prec_anom", "ndvi_anom", "oni_anom"]:  # Variables climáticas
    xc = base[v] - g[v].transform("mean")  # Centrada por municipio
    ccf[v] = [np.corrcoef(xc.groupby(base["COD_MUN_N"]).shift(k).fillna(0), base["lc_c"])[0, 1] for k in range(0, 21)]  # corr(x_{t−k}, casos_t)
ccf = pd.DataFrame(ccf, index=pd.Index(range(21), name="rezago (semanas)"))  # Tabla
fig = px.line(ccf, markers=True)  # Una línea por variable
fig.update_layout(title="Correlación cruzada dentro de los municipios: clima en t−k frente a casos en t", yaxis_title="correlación", legend_title="")  # Formato
mostrar(fig, 360)  # Muestra
ccf.abs().idxmax().to_frame("rezago de máxima |correlación|").join(ccf.abs().max().to_frame("máxima |correlación|")).round(3)'''),
        md("**Interpretación.** La ACF decae muy lentamente (la semana actual explica buena parte de la semana $t+4$) y la PACF "
           "se concentra en los primeros rezagos: es un proceso autorregresivo persistente, lo que justifica rezagos de 1 a 3 "
           "semanas y medias móviles de 4, 8 y 12. En el rezago 52 la ACF es prácticamente 0: no hay un ciclo anual "
           "fuerte. La correlación cruzada del clima local con los casos es pequeña (|r| < 0,06): la lluvia y el NDVI alcanzan "
           "su máximo con 12–15 semanas de rezago y la temperatura de forma casi contemporánea. El ONI es la señal climática "
           "más fuerte (r ≈ 0,17) y llega con unas 5 semanas de rezago, coherente con el ciclo mosquito–virus–síntomas. Por eso se usan medias móviles largas del clima y el ONI rezagado 8 semanas."),

        md("## Cambios de régimen, eventos y calendario"),
        code('''sigma2 = np.var(stl.resid)  # Varianza del ruido (residuo de STL)
cortes = es.segmentacion_binaria(ln.to_numpy(), penalizacion=3 * 2 * sigma2 * np.log(len(ln)), min_tam=52)  # Regímenes de al menos un año
fechas_corte = [ln.index[i] for i in cortes]  # Fechas
segmentos = np.split(ln.to_numpy(), cortes)  # Segmentos
inicios = [ln.index[0]] + fechas_corte  # Inicio de cada segmento
oni = eda.drop_duplicates("week_start").set_index("week_start")["oni_anom"]  # ONI semanal
fig = go.Figure()  # Figura
for ini, fin in zip(oni.index[(oni > 0.5) & (oni.shift() <= 0.5)], oni.index[(oni <= 0.5) & (oni.shift() > 0.5)]):  # Episodios de El Niño
    fig.add_vrect(x0=ini, x1=fin, fillcolor=ROJO, opacity=0.08, line_width=0)  # Sombreado
fig.add_scatter(x=ln.index, y=ln.values, name="log casos", line=dict(color=AZUL, width=1))  # Serie
for ini, seg in zip(inicios, segmentos):  # Media de cada régimen
    fin = ln.index[min(ln.index.get_loc(ini) + len(seg) - 1, len(ln) - 1)]  # Fin del segmento
    fig.add_scatter(x=[ini, fin], y=[seg.mean()] * 2, mode="lines", line=dict(color=AMBAR, width=3), showlegend=False)  # Nivel
fig.update_layout(title="Regímenes detectados (ámbar) y episodios de El Niño (franjas rojas)")  # Título
mostrar(fig, 380)  # Muestra
print("Puntos de cambio:", ", ".join(str(f.date()) for f in fechas_corte))  # Fechas'''),
        code('''sem = eda.groupby(["ANO", "SEMANA"])["casos_totales"].sum().reset_index()  # Casos por año y semana
sem["vecinas"] = sem.groupby("ANO")["casos_totales"].transform(lambda s: s.rolling(7, center=True, min_periods=3).median())  # Mediana de semanas vecinas
fin_anio = sem[sem["SEMANA"].isin([52, 53, 1])]  # Fin y comienzo de año (festividades)
print(f"Casos en las semanas 52–53 y 1 frente a la mediana de sus vecinas: {(fin_anio['casos_totales'] / fin_anio['vecinas']).median():.2f}")  # Efecto de calendario
covid = sem[sem["ANO"] == 2020].set_index("SEMANA")["casos_totales"]  # 2020
print(f"2020, semanas 1–11 frente a 14–26 (inicio de la cuarentena): {covid.loc[1:11].mean():,.0f} → {covid.loc[14:26].mean():,.0f} casos por semana")  # Evento'''),
        md("**Interpretación.** Los seis puntos de cambio siguen las ondas epidémicas: el fin de la epidemia de 2010 (enero de "
           "2011), el inicio de la epidemia de 2013 (diciembre de 2012), el valle de 2017–2018 (septiembre de 2016), la subida "
           "previa a la epidemia de 2019 (mayo de 2018 y mayo de 2019) y la caída de junio de 2020, que coincide con la "
           "pandemia: entre las semanas 1–11 y 14–26 de 2020 los casos semanales cayeron de ≈ 2 950 a ≈ 1 170 (menos consultas "
           "y notificación). Los episodios de El Niño anteceden a los regímenes altos. El efecto de las festividades de fin "
           "de año es pequeño (≈ 3 % menos casos que en las semanas vecinas)."),

        md("## Deriva temporal: *covariate shift* y *concept drift*"),
        code('''temprano, tardio = tr[tr["ANO"] <= 2018], tr[tr["ANO"] >= 2019]  # Dos mitades del entrenamiento
vars_deriva = ["lc_m4", "brote_m8", "ratio_umbral", "temp_mean_m4", "temp_anom_m8", "prec_total_m4", "ndvi_mean_m4", "oni_anom", "oni_l8", "flujo_m4", "lpob"]  # Variables
filas = []  # Deriva por variable
for v in vars_deriva:  # Cada variable
    filas.append({"variable": v, "PSI 2015–18 → 2019–21": es.psi(temprano[v], tardio[v]), "KS 2015–18 vs 2019–21": stats.ks_2samp(temprano[v], tardio[v])[0],  # Dentro del entrenamiento
                  "PSI entrenamiento → validación 2022": es.psi(tr[v], va[v]), "KS entrenamiento vs 2022": stats.ks_2samp(tr[v], va[v])[0]})  # Hacia la validación
deriva = pd.DataFrame(filas).set_index("variable")  # Tabla
print(f"Prevalencia del objetivo: 2015–18 {temprano[cf.TARGET].mean():.1%} · 2019–21 {tardio[cf.TARGET].mean():.1%} · 2022 {va[cf.TARGET].mean():.1%}")  # Deriva del objetivo
deriva.round(3)'''),
        code('''anios = sorted(tr["ANO"].unique()) + [cf.ANIO_VALIDACION]  # Años
datos_anio = pd.concat([tr, va])  # Entrenamiento y validación
concepto = pd.DataFrame({v: [roc_auc_score(datos_anio.loc[datos_anio["ANO"] == a, cf.TARGET], datos_anio.loc[datos_anio["ANO"] == a, v]) for a in anios]  # AUC por año
                         for v in ["lc_m4", "brote_m8", "ratio_umbral", "oni_l8", "temp_anom_m8"]}, index=anios)
fig = px.line(concepto, markers=True)  # Una línea por variable
fig.add_hline(y=0.5, line_dash="dot", line_color=GRIS)  # Azar
fig.update_layout(title="Concept drift: AUC univariado de cada variable, año por año", yaxis_title="AUC", legend_title="")  # Formato
mostrar(fig, 360)  # Muestra
concepto.round(3)'''),
        md("**Interpretación.** La deriva de las covariables es baja para la historia de casos y el clima local dentro del "
           "entrenamiento (PSI < 0,1), moderada hacia 2022 para las anomalías de temperatura y la lluvia (PSI 0,10–0,20, un año "
           "de La Niña) y **fuerte para el ONI** (PSI ≫ 0,25): cada periodo tiene una fase distinta de El Niño, de modo que su "
           "distribución nunca es estable. La prevalencia del objetivo también cambia (*prior shift*: 7,5 % en 2015–2018, "
           "13,9 % en 2019–2021, 12,4 % en 2022). En cambio, el **concepto** "
           "es estable para la historia de casos: el AUC univariado de `lc_m4` se mantiene entre 0,83 y 0,89 todos los años, "
           "mientras que el del ONI rezagado (0,46–0,68) y el de las anomalías de temperatura (0,46–0,64) varía mucho y llega a "
           "quedar por debajo de 0,5. Consecuencias: la relación casos → brote es la "
           "base confiable; las señales climáticas deben interpretarse con cautela y el umbral de decisión debe revisarse "
           "periódicamente porque la prevalencia cambia."),

        md("## Panel: series por entidad y heterogeneidad"),
        code('''orden = d.municipios().sort_values(["region", "latitud"])  # Orden por región y latitud
mat = tr.pivot_table(index="COD_MUN_N", columns="week_start", values=cf.TARGET).loc[orden["COD_MUN_N"]]  # Municipio × semana
fig = go.Figure(go.Heatmap(z=mat.values, x=mat.columns, y=np.arange(len(mat)), colorscale=[[0, "#f8fafc"], [1, ROJO]], showscale=False))  # Mapa de calor
cambios = orden["region"].ne(orden["region"].shift()).to_numpy().nonzero()[0]  # Inicio de cada región
fig.update_yaxes(tickvals=cambios, ticktext=orden["region"].iloc[cambios].tolist(), autorange="reversed")  # Etiquetas de región
fig.update_layout(title="Brote en cuatro semanas por municipio (filas) y semana (columnas), 2015–2021")  # Título
mostrar(fig, 640)  # Muestra'''),
        code('''ejemplo = {76001: "Cali", 8001: "Barranquilla", 54001: "Cúcuta", 50001: "Villavicencio"}  # Municipios de ejemplo
serie = eda[eda["COD_MUN_N"].isin(ejemplo) & (eda["ANO"] >= 2015)]  # Series
fig = px.line(serie, x="week_start", y="lc", color=serie["COD_MUN_N"].map(ejemplo), labels={"lc": "log(1 + casos)", "color": ""})  # Líneas
fig.update_layout(title="Series de municipios de distintas regiones", xaxis_title="")  # Título
mostrar(fig, 360)  # Muestra
prev = tr.groupby("COD_MUN_N")[cf.TARGET].mean()  # Prevalencia por municipio
print(f"Prevalencia por municipio: mediana {prev.median():.1%}, percentil 10 {prev.quantile(0.1):.1%}, percentil 90 {prev.quantile(0.9):.1%}")  # Heterogeneidad'''),
        md("**Interpretación.** El mapa de calor muestra que los brotes llegan en **ondas sincronizadas** (franjas verticales que "
           "atraviesan muchas regiones a la vez, como en 2019) y también como brotes locales aislados. Las series municipales "
           "son heterogéneas en nivel y en tiempo: una epidemia no golpea a todos los municipios en la misma semana. Esto "
           "justifica normalizar por municipio (canal endémico, anomalías) y evaluar el error por conglomerados.\n\n"
           "## Consecuencias para el modelado\n\n"
           "* **Partición cronológica**, nunca aleatoria (2015–2021 / 2022 / 2023–2024) con purga de 4 semanas en cada frontera.\n"
           "* **Validación cruzada de ventana creciente** por años (2018, 2019, 2020 y 2021), con la misma purga: es el "
           "equivalente a `TimeSeriesSplit` con `gap` igual al horizonte.\n"
           "* **Rezagos y ventanas móviles solo con el pasado**: rezagos de 0–3 semanas y medias de 4, 8 y 12 semanas.\n"
           "* **Componentes cíclicos** (seno y coseno de la semana) en lugar de variables indicadoras de mes.\n"
           "* **Riesgo de deriva**: alto para el ONI y la prevalencia; el umbral debe recalibrarse y la evaluación debe "
           "hacerse en varios años."),
    ]
    guardar("07_temporal", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
