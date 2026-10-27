"""Cuaderno 1: Base de datos (sección 1 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Base de datos\n\n"
           "Este capítulo cubre la sección 1 de la guía: problema de investigación, justificación, fuente y licencia, "
           "diccionario de variables, estructura, tamaño de la muestra, calidad de los datos y consideraciones éticas."),
        code(PREAMBULO + '''
raw = pd.read_csv(cf.RUTA_DATOS)  # Archivo original, sin transformar
raw["week_start"] = pd.to_datetime(raw["week_start"])  # Fecha de inicio de la semana epidemiológica
m = d.municipios()  # 376 municipios con coordenadas DANE y región
print(f"Archivo: {cf.RUTA_DATOS.name} · {raw.shape[0]:,} filas × {raw.shape[1]} columnas")  # Dimensión'''),

        md("## Problema de investigación\n\n"
           "**Pregunta.** ¿Es posible anticipar, con cuatro semanas de margen, si un municipio colombiano entrará en "
           "situación de brote de dengue, usando solo la información disponible en la semana actual (historia de casos, "
           "clima, vegetación, El Niño, población y movilidad)?\n\n"
           "**Definición operativa.** Una semana es de brote si los casos superan el tercer cuartil de las mismas semanas "
           "(± 2) de los cinco años anteriores y hay al menos tres casos (canal endémico, método de cuartiles). El objetivo de "
           "la fila $(m, t)$ es el brote de la semana $t+4$:\n\n"
           "$$y_{m,t} = \\mathbb{1}\\left[c_{m,t+4} > Q_{0.75}\\{c_{m,\\,t+4-52k+j}\\}_{k=1..5,\\;j=-2..2} \\;\\wedge\\; c_{m,t+4} \\ge 3\\right].$$\n\n"
           "**Tipo de tarea.** Clasificación binaria probabilística sobre un panel espacio-temporal (ruta A de la guía, con "
           "las secciones 2.6 a 2.8 por tener tiempo y espacio)."),

        md("## Justificación de la selección del dataset\n\n"
           "* **Relevancia.** El dengue es la arbovirosis de mayor carga en Colombia; 2024 fue el año con más casos de la serie. "
           "Las secretarías de salud deciden cada semana dónde fumigar, eliminar criaderos y reforzar servicios: una alerta con "
           "cuatro semanas de margen tiene uso directo.\n"
           "* **Granularidad.** Municipio × semana epidemiológica, la misma resolución de la vigilancia oficial.\n"
           "* **Riqueza.** Combina la notificación de casos con clima, vegetación (NDVI), el índice ONI de El Niño, población y "
           "flujo de personas, lo que permite estudiar componentes temporales, espaciales y espacio-temporales.\n"
           "* **Tamaño.** Muy por encima del mínimo de 20 000 observaciones exigido por la guía (ver abajo).\n"
           "* **Dificultad adecuada.** El objetivo no es trivial: la persistencia (\"si hay brote hoy, habrá brote en cuatro "
           "semanas\") ya acierta mucho, y el reto es mejorar el **ordenamiento del riesgo**, no la exactitud."),

        md("## Fuente de los datos y licencia\n\n"
           "| Componente | Origen | Licencia / condiciones de uso |\n"
           "|---|---|---|\n"
           "| Panel integrado `final_dataset.csv` | Archivo del proyecto (casos, clima, NDVI, ONI, población y flujo ya integrados por municipio y semana) | Ver nota |\n"
           "| Casos de dengue | Sistema Nacional de Vigilancia en Salud Pública (SIVIGILA), Instituto Nacional de Salud | Datos abiertos del Estado colombiano (Ley 1712 de 2014) |\n"
           "| Índice ONI (El Niño) | NOAA Climate Prediction Center | Dominio público (gobierno de EE. UU.) |\n"
           "| Centroides municipales | DANE, DIVIPOLA, corte 30-dic-2024: [datos.gov.co/resource/gdxc-w37w](https://www.datos.gov.co/Mapas-Nacionales/DIVIPOLA-C-digos-municipios/gdxc-w37w) | CC BY-SA 4.0 |\n\n"
           ":::{note}\n"
           "Las coordenadas no venían en el archivo original: se agregaron desde la DIVIPOLA del DANE (unión por el código "
           "municipal; los 376 municipios del panel se encontraron) para poder hacer el análisis espacial de las secciones 2.7 y 2.8. "
           "El panel integrado se distribuye con el repositorio del proyecto en `data/final_dataset.csv`.\n"
           ":::"),

        md("## Diccionario de variables"),
        code('''dic = pd.DataFrame([  # Tipo, unidad y significado de las 21 columnas originales
    ("COD_MUN_N", "Identificador", "código DANE", "Municipio (5 dígitos; los dos primeros son el departamento)"),
    ("ANO", "Temporal", "año", "Año epidemiológico"),
    ("SEMANA", "Temporal", "semana", "Semana epidemiológica (1–52; 53 en 2015 y 2020)"),
    ("casos_totales", "Conteo", "casos/semana", "Casos de dengue notificados en la semana"),
    ("casos_0_4", "Conteo", "casos/semana", "Casos en menores de 5 años"),
    ("casos_5_14", "Conteo", "casos/semana", "Casos de 5 a 14 años"),
    ("casos_15_64", "Conteo", "casos/semana", "Casos de 15 a 64 años"),
    ("casos_65_plus", "Conteo", "casos/semana", "Casos de 65 años o más"),
    ("casos_m", "Conteo", "casos/semana", "Casos en hombres"),
    ("casos_f", "Conteo", "casos/semana", "Casos en mujeres"),
    ("week_start", "Temporal", "fecha", "Lunes de inicio de la semana epidemiológica"),
    ("MES", "Temporal", "mes", "Mes de la semana (1–12)"),
    ("temp_mean", "Continua", "°C", "Temperatura media semanal (imputada si temp_mean_missing = 1)"),
    ("prec_total", "Continua", "mm/semana", "Precipitación acumulada en la semana"),
    ("ndvi_mean", "Continua", "índice [−1, 1]", "Índice de vegetación de diferencia normalizada (valor mensual)"),
    ("oni_anom", "Continua", "°C", "Anomalía del Índice Oceánico de El Niño (> 0,5 El Niño; < −0,5 La Niña)"),
    ("oni_total", "Continua", "°C", "Temperatura superficial del mar en la región Niño 3.4"),
    ("Flujo_in", "Continua", "personas (escala del dataset)", "Flujo de personas que ingresan al municipio"),
    ("temp_mean_missing", "Binaria", "0/1", "1 si la temperatura de la semana fue imputada"),
    ("time_idx", "Temporal", "índice", "Número de la semana desde enero de 2010 (0–781)"),
    ("poblacion", "Conteo", "habitantes", "Población del municipio en el año"),
], columns=["variable", "tipo", "unidad", "significado"])  # Tabla
dic["dtype"] = dic["variable"].map(raw.dtypes.astype(str))  # Tipo de dato en pandas
dic["valores únicos"] = dic["variable"].map(raw.nunique())  # Cardinalidad
dic'''),
        md("Además del archivo original, el análisis usa tres columnas **derivadas**: el umbral del canal endémico, el "
           "indicador de brote y el objetivo `y` (brote en $t+4$), y las 34 variables predictoras del capítulo de "
           "preprocesamiento. Las coordenadas, el nombre, el departamento y la región vienen de la DIVIPOLA."),

        md("## Estructura de los datos"),
        code('''n_mun, n_sem = raw["COD_MUN_N"].nunique(), raw["time_idx"].nunique()  # Entidades y periodos
balanceado = (raw.groupby("COD_MUN_N").size() == n_sem).all()  # ¿Todas las entidades tienen todas las semanas?
pd.Series({  # Resumen estructural
    "Unidad de observación": "municipio × semana epidemiológica",
    "Nivel de agregación": "conteos semanales por municipio (no hay registros individuales)",
    "Tipo de datos": "panel espacio-temporal (longitudinal con coordenadas)",
    "Municipios (entidades)": n_mun,
    "Departamentos": raw["COD_MUN_N"].floordiv(1000).nunique(),
    "Semanas": n_sem,
    "Periodo": f"{raw['week_start'].min().date()} a {raw['week_start'].max().date()}",
    "Panel balanceado": balanceado,
}, name="valor").to_frame()'''),

        md("## Tamaño de la muestra"),
        code('''df = d.panel()  # Panel con canal endémico, objetivo, variables y periodo de cada fila
mod = d.modelado(df)  # Filas con objetivo y variables completas (2015–2024, sin filas purgadas)
tr, va, te = d.particion(mod)  # Entrenamiento, validación y prueba
p = len(cf.FEATURES)  # Número de variables predictoras
pd.Series({  # Tamaños relevantes
    "Filas del archivo": f"{len(raw):,}",
    "Columnas originales": raw.shape[1],
    "Filas aptas para modelar (2015–2024)": f"{len(mod):,}",
    "Filas de entrenamiento (2015–2021)": f"{len(tr):,}",
    "Variables predictoras (p)": p,
    "Relación n/p (entrenamiento)": f"{len(tr) / p:,.0f}",
    "Brotes en entrenamiento (clase minoritaria)": f"{int(tr[cf.TARGET].sum()):,}",
    "No brotes en entrenamiento": f"{int((1 - tr[cf.TARGET]).sum()):,}",
}, name="valor").to_frame()'''),
        md("### Tamaño efectivo: las filas no son independientes\n\n"
           "Las 782 semanas de un municipio están correlacionadas. Con el coeficiente de correlación intraclase (ICC) del "
           "brote dentro de cada municipio, el **efecto de diseño** $\\text{DEFF} = 1 + (\\bar m - 1)\\,\\text{ICC}$ indica "
           "cuántas filas equivalen a una observación independiente."),
        code('''def icc_anova(valores, grupos):  # ICC(1) por análisis de varianza de un factor
    g = pd.DataFrame({"v": valores, "g": grupos})  # Tabla
    k = g.groupby("g").size().mean()  # Tamaño medio de grupo
    medias = g.groupby("g")["v"].transform("mean")  # Media de cada grupo
    msb = ((medias - g["v"].mean()) ** 2).sum() / (g["g"].nunique() - 1)  # Cuadrado medio entre grupos
    msw = ((g["v"] - medias) ** 2).sum() / (len(g) - g["g"].nunique())  # Cuadrado medio dentro de grupos
    return (msb - msw) / (msb + (k - 1) * msw), k  # ICC y tamaño medio

icc, k = icc_anova(tr[cf.TARGET], tr["COD_MUN_N"])  # ICC del objetivo en entrenamiento
deff = 1 + (k - 1) * icc  # Efecto de diseño
print(f"ICC del brote dentro de cada municipio: {icc:.3f}")  # Correlación intraclase
print(f"Semanas por municipio en entrenamiento: {k:.0f} · efecto de diseño: {deff:.1f}")  # DEFF
print(f"Tamaño efectivo aproximado: {len(tr) / deff:,.0f} observaciones independientes (de {len(tr):,} filas)")  # n efectivo'''),
        md("**Interpretación.** El panel supera con holgura el mínimo de 20 000 observaciones y la relación $n/p$ es muy alta, así "
           "que el sobreajuste por falta de datos no es una preocupación para un modelo lineal. Pero el tamaño efectivo es "
           "mucho menor que el número de filas: las semanas de un mismo municipio se parecen entre sí. Por eso la "
           "incertidumbre de las métricas se estima remuestreando **municipios completos** (bootstrap por conglomerados) y "
           "no filas, y la validación nunca mezcla semanas al azar."),

        md("## Calidad de los datos\n\n### Valores faltantes: patrón"),
        code('''print(f"Celdas nulas en el archivo: {int(raw.isna().sum().sum())}")  # No hay nulos explícitos
print(f"Semanas con temperatura imputada: {raw['temp_mean_missing'].mean():.1%}")  # Faltantes ocultos
frac = raw.groupby("COD_MUN_N")["temp_mean_missing"].mean()  # Fracción imputada por municipio
grupo = pd.cut(frac, [-0.01, 0, 0.99, 1], labels=["nunca imputada", "parcialmente", "siempre imputada"])  # Tres patrones
grupo.value_counts().rename("municipios").to_frame()'''),
        code('''matriz = raw.pivot_table(index="COD_MUN_N", columns="ANO", values="temp_mean_missing", aggfunc="mean")  # Municipio × año
matriz = matriz.loc[frac.sort_values().index]  # Ordena por fracción imputada
fig = px.imshow(matriz, aspect="auto", color_continuous_scale=["#f8fafc", ROJO], zmin=0, zmax=1,  # Mapa de calor
                labels=dict(x="Año", y="Municipio (ordenado)", color="Fracción imputada"))
fig.update_yaxes(showticklabels=False)  # Sin etiquetas de municipio
fig.update_layout(title="Matriz de faltantes: temperatura imputada por municipio y año")  # Título
mostrar(fig, 520)  # Muestra'''),
        code('''imp = raw.loc[raw["temp_mean_missing"] == 1, "temp_mean"]  # Temperaturas imputadas
obs = raw.loc[raw["temp_mean_missing"] == 0, "temp_mean"]  # Temperaturas observadas
bordes = np.linspace(9, 35, 105)  # Intervalos de 0,25 °C (histograma precalculado: la figura no embebe 294 000 valores)
centros = (bordes[:-1] + bordes[1:]) / 2  # Centro de cada intervalo
fig = go.Figure([go.Bar(x=centros, y=np.histogram(obs, bordes, density=True)[0], name="Observada", marker_color=AZUL, opacity=0.6),  # Observada
                 go.Bar(x=centros, y=np.histogram(imp, bordes, density=True)[0], name="Imputada", marker_color=ROJO, opacity=0.6)])  # Imputada
fig.update_layout(barmode="overlay", bargap=0, title="Distribución de la temperatura observada frente a la imputada",  # Formato
                  xaxis_title="°C", yaxis_title="densidad")
mostrar(fig, 380)  # Muestra
print(f"Imputada: media {imp.mean():.2f} °C, desviación {imp.std():.2f} · Observada: media {obs.mean():.2f} °C, desviación {obs.std():.2f}")  # Resumen'''),
        md("**Patrón.** No hay celdas nulas, pero la temperatura está imputada en el 64 % de las semanas y el patrón es "
           "**por municipio**, no por semana: 226 municipios nunca tienen temperatura medida, solo 21 la tienen siempre y 129 "
           "la tienen parcialmente. La fracción imputada es casi idéntica todos los años. Los valores imputados se concentran "
           "alrededor de 22,4 °C con muy poca variación: la imputación original es prácticamente una constante y no "
           "conserva la diferencia climática entre municipios."),

        md("### Valores faltantes: mecanismo (MCAR, MAR o MNAR)\n\n"
           "Se marcan como faltantes las temperaturas imputadas y se aplican tres evidencias: la prueba de Little (H0: MCAR), "
           "la comparación de distribuciones entre registros completos e incompletos (diferencia estandarizada) y un modelo "
           "que intenta predecir qué municipios carecen de temperatura a partir de variables observadas."),
        code('''eda = d.entrenamiento_eda(df)  # Decisiones del EDA: solo semanas 2010–2021 (la prueba queda sellada)
X = pd.DataFrame({"temp_mean": eda["temp_mean"].where(eda["temp_mean_missing"] == 0),  # Temperatura con faltantes
                  "prec_total": eda["prec_total"], "ndvi_mean": eda["ndvi_mean"], "oni_anom": eda["oni_anom"],  # Observadas
                  "log_poblacion": np.log(eda["poblacion"]), "log_casos": np.log1p(eda["casos_totales"])})  # Observadas
muestra = X.sample(20000, random_state=cf.SEMILLA)  # Muestra para la prueba (con n enorme siempre rechaza)
d2, gl, p_little, patrones = es.prueba_little(muestra.to_numpy())  # Prueba de Little
print(f"Prueba de Little (n = 20 000): d² = {d2:,.1f}, gl = {gl}, p = {p_little:.2e}, patrones = {patrones}")  # Resultado
falta = X["temp_mean"].isna()  # Indicador de faltante
pd.DataFrame({"diferencia estandarizada (faltante − observado)": {c: es.diferencia_estandarizada(X.loc[falta, c], X.loc[~falta, c])  # d de Cohen
                                                                 for c in X.columns[1:]}})'''),
        code('''from sklearn.linear_model import LogisticRegression  # Modelo del mecanismo
from sklearn.model_selection import cross_val_score  # Validación cruzada
from sklearn.pipeline import make_pipeline  # Encadena
from sklearn.preprocessing import StandardScaler  # Estandariza
perfil = eda.groupby("COD_MUN_N").agg(log_poblacion=("poblacion", lambda s: np.log(s.mean())), prec=("prec_total", "mean"),  # Perfil del municipio
                                      ndvi=("ndvi_mean", "mean"), log_casos=("casos_totales", lambda s: np.log1p(s.mean())),
                                      falta=("temp_mean_missing", "mean")).join(m.set_index("COD_MUN_N")[["region", "latitud", "longitud"]])
Z = pd.get_dummies(perfil.drop(columns="falta"), columns=["region"], dtype=float)  # Variables observadas
auc_mec = cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)), Z, (perfil["falta"] > 0.5).astype(int),  # ¿Se predice el faltante?
                          cv=5, scoring="roc_auc").mean()
print(f"AUC al predecir qué municipio carece de temperatura con variables observadas: {auc_mec:.3f}")  # 0,5 = imposible de predecir
pd.crosstab(m.set_index("COD_MUN_N")["region"], grupo, margins=True)  # Patrón de faltantes por región (todos los años)'''),
        md("**Mecanismo.** Los faltantes **no son MCAR**: la prueba de Little los rechaza con claridad, y la diferencia más "
           "grande está en el tamaño del municipio (los municipios sin temperatura son mucho más pequeños, $d \\approx -0{,}7$) "
           "y en la carga de casos ($d \\approx -0{,}3$). La falta de dato se explica en parte con variables observadas "
           "(AUC ≈ 0,67 con población, región, coordenadas y clima) y no cambia con el tiempo, lo que es coherente con "
           "**MAR estructural**: los municipios pequeños no tienen estación meteorológica. No se puede descartar un componente "
           "**MNAR** (si las estaciones faltan justo donde el clima es más extremo), y eso no es verificable con estos datos.\n\n"
           "**Consecuencia.** No se reimputa la temperatura con medias de otros municipios (226 municipios no tienen ninguna "
           "medición propia y la altitud cambia mucho entre vecinos). Se conserva la bandera `temp_missing` como variable "
           "(método del indicador de faltante), de modo que el modelo pueda descontar la señal térmica cuando es imputada; el "
           "`SimpleImputer` del Pipeline queda como protección ante nulos futuros."),

        md("### Duplicados exactos y casi duplicados"),
        code('''clima = ["temp_mean", "prec_total", "ndvi_mean"]  # Variables climáticas
dup_clima = raw.duplicated(subset=["time_idx"] + clima, keep=False)  # Misma semana y mismo clima en distintos municipios
pd.Series({
    "Filas duplicadas exactas": int(raw.duplicated().sum()),
    "Claves (municipio, semana) repetidas": int(raw.duplicated(["COD_MUN_N", "time_idx"]).sum()),
    "Filas idénticas salvo identificadores": int(raw.drop(columns=["COD_MUN_N", "time_idx", "week_start", "SEMANA"]).duplicated().sum()),
    "Filas con clima idéntico a otro municipio en la misma semana": f"{int(dup_clima.sum()):,} ({dup_clima.mean():.1%})",
}, name="conteo").to_frame()'''),
        md("**Interpretación.** No hay duplicados exactos, claves (municipio, semana) repetidas ni filas que coincidan en todo "
           "salvo los identificadores. Tampoco hay casi duplicados climáticos: ningún par de municipios comparte exactamente "
           "temperatura, lluvia y NDVI en la misma semana. No se elimina ninguna fila por duplicación."),

        md("### Valores atípicos"),
        code('''cont = ["casos_totales", "temp_mean", "prec_total", "ndvi_mean", "oni_anom", "Flujo_in", "poblacion"]  # Variables continuas
q1, q3 = raw[cont].quantile(0.25), raw[cont].quantile(0.75)  # Cuartiles
iqr = q3 - q1  # Rango intercuartílico
fuera = (raw[cont] < q1 - 1.5 * iqr) | (raw[cont] > q3 + 1.5 * iqr)  # Regla de Tukey
pd.DataFrame({"límite inferior": q1 - 1.5 * iqr, "límite superior": q3 + 1.5 * iqr,  # Límites
              "atípicos (%)": fuera.mean() * 100, "máximo": raw[cont].max()})'''),
        code('''top = raw.nlargest(8, "casos_totales").merge(m, on="COD_MUN_N")  # Semanas con más casos
top[["municipio", "departamento", "week_start", "casos_totales", "poblacion"]]'''),
        md("**Interpretación y tratamiento.** Los casos tienen una cola muy larga: casi el 12 % de las semanas supera el límite "
           "de Tukey (más de 5 casos), y las ocho semanas con más casos son todas de Cali en la epidemia de 2024, no errores "
           "de digitación. **No se eliminan**: son precisamente los eventos que se quieren anticipar. Se tratan con la "
           "transformación $\\log(1 + x)$, que comprime la cola sin borrar información; lo mismo aplica a la población y al "
           "flujo de personas (12–14 % de atípicos por sus colas largas). El 18 % de «atípicos» de temperatura es un "
           "artefacto: la imputación concentra el 64 % de los valores cerca de 22,4 °C, estrecha el rango intercuartílico y "
           "convierte en atípicas temperaturas normales de tierra caliente."),

        md("### Valores imposibles o inconsistentes"),
        code('''edades = ["casos_0_4", "casos_5_14", "casos_15_64", "casos_65_plus"]  # Grupos de edad
chequeos = {  # Reglas de consistencia
    "Conteos negativos": int((raw[["casos_totales"] + edades + ["casos_m", "casos_f"]] < 0).sum().sum()),
    "Suma de edades ≠ total": int((raw[edades].sum(axis=1) != raw["casos_totales"]).sum()),
    "Hombres + mujeres ≠ total": int((raw["casos_m"] + raw["casos_f"] != raw["casos_totales"]).sum()),
    "Temperatura fuera de [0, 40] °C": int((~raw["temp_mean"].between(0, 40)).sum()),
    "Precipitación negativa": int((raw["prec_total"] < 0).sum()),
    "NDVI fuera de [−1, 1]": int((~raw["ndvi_mean"].between(-1, 1)).sum()),
    "Población ≤ 0": int((raw["poblacion"] <= 0).sum()),
    "Semanas que no avanzan 7 días": int((raw.sort_values(["COD_MUN_N", "time_idx"]).groupby("COD_MUN_N")["week_start"].diff().dt.days.dropna() != 7).sum()),
    "Población distinta dentro de un mismo año": int((raw.groupby(["COD_MUN_N", "ANO"])["poblacion"].nunique() > 1).sum()),
    "Casos mayores que la población": int((raw["casos_totales"] > raw["poblacion"]).sum()),
}
pd.Series(chequeos, name="filas con problema").to_frame()'''),
        code('''mayor_lluvia = raw.nlargest(3, "prec_total").merge(m, on="COD_MUN_N")[["municipio", "departamento", "week_start", "prec_total"]]  # Lluvias extremas
mayor_lluvia'''),
        md("**Interpretación.** No hay valores imposibles: los conteos son coherentes entre sí (edades y sexo suman el total), "
           "los rangos físicos son válidos, la frecuencia es estrictamente semanal y la población es anual. Las lluvias "
           "semanales extremas (más de 1 000 mm) corresponden a municipios del Pacífico, una de las regiones más lluviosas "
           "del planeta, así que son plausibles. Las categorías son códigos numéricos, sin errores de escritura."),

        md("### Sesgos de muestreo y representatividad"),
        code('''todos = pd.read_csv(d.RUTA_DIVIPOLA, dtype=str)  # Los 1 122 municipios del país
todos["region"] = todos["cod_dpto"].astype(int).map(d.REGIONES)  # Región
rep = pd.DataFrame({"municipios del país": todos["region"].value_counts(),  # Universo
                    "municipios del panel": m["region"].value_counts()})  # Muestra
rep["cobertura (%)"] = rep["municipios del panel"] / rep["municipios del país"] * 100  # Cobertura
rep["participación en el panel (%)"] = rep["municipios del panel"] / rep["municipios del panel"].sum() * 100  # Composición
rep["participación en el país (%)"] = rep["municipios del país"] / rep["municipios del país"].sum() * 100  # Composición nacional
pob_panel = raw.loc[raw["ANO"] == 2024].groupby("COD_MUN_N")["poblacion"].first().sum()  # Población cubierta en 2024
print(f"Municipios en el panel: {len(m)} de {len(todos)} ({len(m) / len(todos):.1%}) · población cubierta en 2024: {pob_panel / 1e6:.1f} millones")  # Cobertura
rep.round(1)'''),
        md("**Interpretación.** El panel incluye un tercio de los municipios del país pero cerca del 60 % de su población "
           "(32 de unos 53 millones de habitantes), porque contiene las capitales y los municipios con transmisión habitual de "
           "dengue. La composición regional no es la nacional: el Caribe y la Orinoquía están sobrerrepresentados (son "
           "regiones de tierra caliente, endémicas) y la región Andina y la Pacífica, subrepresentadas. Hay tres sesgos "
           "a declarar: (1) **selección**, los municipios de alta montaña sin transmisión no están, así que las conclusiones "
           "aplican a municipios endémicos; (2) **subregistro**, los casos son los notificados al SIVIGILA y su calidad varía "
           "entre municipios y años (por ejemplo, durante la pandemia de 2020); y (3) **cobertura climática**, la medición de "
           "temperatura falta justamente en los municipios pequeños."),

        md("## Consideraciones éticas\n\n"
           "* **Datos personales.** El panel solo contiene conteos agregados por municipio y semana; no hay nombres, documentos ni "
           "direcciones. No requiere consentimiento informado ni aprobación de un comité de ética.\n"
           "* **Riesgo de reidentificación.** Es bajo pero no nulo: en municipios pequeños, una semana con un caso de 65 años o más "
           "y sexo conocido podría asociarse a una persona por conocimiento local. Por eso el informe solo publica agregados y "
           "nunca cruces finos de edad × sexo × semana en municipios pequeños.\n"
           "* **Coordenadas y tiempo.** Las coordenadas son centroides oficiales del municipio, no ubicaciones de viviendas, y el "
           "tiempo está agregado por semana: no permiten ubicar casos individuales.\n"
           "* **Uso del modelo.** Una alerta falsa puede generar gasto o alarma injustificada; un brote no anticipado cuesta "
           "salud. El modelo es una herramienta de apoyo a la vigilancia y no reemplaza el criterio epidemiológico. Además, "
           "si la calidad de la notificación es peor en municipios pobres, el modelo podría subestimar su riesgo (sesgo de equidad)."),
    ]
    guardar("01_base_datos", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
