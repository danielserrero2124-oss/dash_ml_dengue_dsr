"""Cuaderno 8: componente espacial (sección 2.7 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Componente espacial\n\n"
           "Cada municipio se representa por su centroide oficial (DIVIPOLA, DANE). Se validan las coordenadas, se mapean el "
           "objetivo y las variables, se estudia el patrón de puntos, la autocorrelación espacial (Moran, LISA y Getis-Ord), "
           "la heterogeneidad entre regiones, el efecto de la escala (MAUP) y las consecuencias para la validación. Se usa el "
           "periodo de entrenamiento 2015–2021."),
        code(PREAMBULO + '''
from scipy import stats  # Pruebas
from sklearn.cluster import DBSCAN  # Agrupamiento por densidad
df = d.panel()  # Panel completo
mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Solo entrenamiento
m = d.municipios()  # Centroides de los 376 municipios
lat, lon = m["latitud"].to_numpy(), m["longitud"].to_numpy()  # Coordenadas (grados)
D = sp.haversine(lat, lon)  # Distancias de gran círculo (km)
xy = sp.proyectar(lat, lon)  # Coordenadas planas locales (km)'''),

        md("## Validación de coordenadas"),
        code('''pd.Series({
    "Sistema de referencia": "EPSG:4326 (WGS84, grados decimales), el de la DIVIPOLA",
    "Latitud en [−90°, 90°]": bool(np.all(np.abs(lat) <= 90)),
    "Longitud en [−180°, 180°]": bool(np.all(np.abs(lon) <= 180)),
    "Rango de latitud": f"{lat.min():.2f}° a {lat.max():.2f}°",
    "Rango de longitud": f"{lon.min():.2f}° a {lon.max():.2f}°",
    "Dentro de Colombia (lat −4,3 a 13,5; lon −82 a −66,8)": bool(np.all((lat > -4.3) & (lat < 13.5) & (lon > -82) & (lon < -66.8))),
    "Posible inversión lat/lon (|lat| > |lon|)": int(np.sum(np.abs(lat) > np.abs(lon))),
    "Coordenadas nulas o (0, 0)": int(np.sum(np.isnan(lat) | np.isnan(lon) | ((lat == 0) & (lon == 0)))),
    "Decimales de precisión": "6 (≈ 0,1 m; es un centroide, no una ubicación exacta)",
    "Coordenadas duplicadas": int(m.duplicated(["latitud", "longitud"]).sum()),
    "Municipios sin coordenadas": int(m["latitud"].isna().sum()),
}, name="valor").to_frame()'''),
        md("**Interpretación.** Las 376 coordenadas son válidas: dentro de los rangos y de Colombia (de Leticia, a 4,2° S, "
           "hasta La Guajira, a 11,7° N), sin inversiones, sin ceros y sin duplicados. Son **centroides**: representan al municipio "
           "entero, de modo que las distancias entre municipios son aproximadas y no hay información dentro de cada municipio."),

        md("## Mapas: puntos, densidad, objetivo y variables relevantes"),
        code('''todos = pd.read_csv(d.RUTA_DIVIPOLA, dtype=str)  # Los 1 122 municipios del país
for col in ("latitud", "longitud"):  # Coma decimal
    todos[col] = todos[col].str.replace(",", ".").astype(float)  # A número
todos["en_panel"] = todos["cod_mpio"].astype(int).isin(m["COD_MUN_N"])  # ¿Está en el panel?
fig = go.Figure()  # Mapa
fig.add_scattergeo(lat=todos.loc[~todos["en_panel"], "latitud"], lon=todos.loc[~todos["en_panel"], "longitud"], mode="markers",  # Fuera del panel
                   marker=dict(size=4, color="#cbd5e1"), name="Municipios fuera del panel")
pob = tr.groupby("COD_MUN_N")["lpob"].mean().loc[m["COD_MUN_N"]].to_numpy()  # log población
fig.add_scattergeo(lat=lat, lon=lon, mode="markers", text=m["municipio"], name="Municipios del panel",  # Panel
                   marker=dict(size=3 + 1.6 * (pob - pob.min()), color=AZUL, opacity=0.7, line=dict(width=0.3, color="white")))
fig.update_layout(title="Cobertura espacial: municipios del panel (tamaño ∝ log población) frente al resto del país")  # Título
mostrar(mapa_base(fig), 620)  # Muestra'''),
        code('''agg = tr.groupby("COD_MUN_N").agg(casos=("casos_totales", "sum"), pob=("poblacion", "mean"), prevalencia=(cf.TARGET, "mean"),  # Perfil espacial
                                  semanas=("time_idx", "size")).loc[m["COD_MUN_N"]]
agg["incidencia"] = agg["casos"] / agg["pob"] / (agg["semanas"] / 52) * 1e5  # Casos por 100 000 habitantes por año
agg["log_incidencia"] = np.log1p(agg["incidencia"])  # Escala log
mm = m.join(agg.reset_index(drop=True))  # Une al mapa
fig = make_subplots(1, 2, specs=[[{"type": "scattergeo"}, {"type": "scattergeo"}]],  # Dos mapas
                    subplot_titles=("Incidencia anual (log, casos por 100 000)", "Proporción de semanas con brote en 4 semanas"))
fig.add_scattergeo(lat=lat, lon=lon, text=mm["municipio"], mode="markers", row=1, col=1,  # Incidencia
                   marker=dict(size=7, color=mm["log_incidencia"], colorscale="YlOrRd", colorbar=dict(x=0.45, title="log"), line=dict(width=0.3)))
fig.add_scattergeo(lat=lat, lon=lon, text=mm["municipio"], mode="markers", row=1, col=2,  # Prevalencia del brote
                   marker=dict(size=7, color=mm["prevalencia"], colorscale="YlOrRd", colorbar=dict(x=1.0, title="%", tickformat=".0%"), line=dict(width=0.3)))
fig.update_geos(fitbounds="locations", showcountries=True, landcolor="#f8fafc", showland=True)  # Encuadre
fig.update_layout(title="La variable objetivo y la carga de enfermedad en el espacio (2015–2021)", showlegend=False)  # Título
mostrar(fig, 520)  # Muestra'''),
        code('''clima = tr[tr["temp_missing"] == 0].groupby("COD_MUN_N")["temp_mean_m4"].mean()  # Temperatura medida por municipio
otras = tr.groupby("COD_MUN_N")[["prec_total_m4", "flujo_m4"]].mean()  # Lluvia y flujo
mv = m.set_index("COD_MUN_N").join(clima).join(otras).reset_index()  # Une
fig = make_subplots(1, 3, specs=[[{"type": "scattergeo"}] * 3], subplot_titles=("Temperatura medida (°C)", "Lluvia (mm/semana)", "Flujo de personas (log)"))  # Tres mapas
for i, (v, esc) in enumerate([("temp_mean_m4", "RdBu_r"), ("prec_total_m4", "Blues"), ("flujo_m4", "Purples")]):  # Variables
    s = mv[v].notna()  # Municipios con dato
    fig.add_scattergeo(lat=mv.loc[s, "latitud"], lon=mv.loc[s, "longitud"], text=mv.loc[s, "municipio"], mode="markers", row=1, col=i + 1,
                       marker=dict(size=6, color=mv.loc[s, v], colorscale=esc, colorbar=dict(x=0.3 + 0.35 * i, len=0.8), line=dict(width=0.3)))
fig.update_geos(fitbounds="locations", showcountries=True, landcolor="#f8fafc", showland=True)  # Encuadre
fig.update_layout(title="Variables relevantes en el espacio (temperatura solo donde se mide)", showlegend=False)  # Título
mostrar(fig, 460)  # Muestra'''),
        md(":::{note}\n"
           "El dataset no trae polígonos municipales, así que no se dibujan coropletas por municipio: se usan **mapas de "
           "símbolos** sobre los centroides, que cumplen la misma función. La densidad de puntos se analiza a continuación.\n"
           ":::"),

        md("## Patrón de puntos y cobertura"),
        code('''ce = sp.clark_evans(xy)  # Índice del vecino más cercano
distancias = np.array([10, 25, 50, 75, 100, 150, 200, 300])  # Escalas (km)
L, lo, hi = sp.ripley_l(xy, distancias, simulaciones=99)  # Función L con envolvente
print(f"Clark-Evans: R = {ce['R']:.3f} (distancia media al vecino más cercano {ce['r_obs']:.1f} km frente a {ce['r_esp']:.1f} km esperados), z = {ce['z']:.2f}, p = {ce['p']:.1e}")  # Resultado
fig = go.Figure()  # Figura
fig.add_scatter(x=distancias, y=hi, line=dict(color=GRIS, width=0), showlegend=False)  # Envolvente superior
fig.add_scatter(x=distancias, y=lo, fill="tonexty", fillcolor="rgba(100,116,139,0.2)", line=dict(color=GRIS, width=0), name="Envolvente CSR (99 sim.)")  # Envolvente
fig.add_scatter(x=distancias, y=L, mode="lines+markers", line=dict(color=ROJO), name="Observada")  # L observada
fig.add_hline(y=0, line_dash="dot", line_color=GRIS)  # Aleatoriedad
fig.update_layout(title="Función L de Ripley: L(d) − d", xaxis_title="distancia d (km)", yaxis_title="L(d) − d (km)")  # Formato
mostrar(fig, 360)  # Muestra'''),
        code('''db = DBSCAN(eps=30 / sp.RADIO_TIERRA, min_samples=5, metric="haversine", algorithm="ball_tree").fit(np.radians(m[["latitud", "longitud"]]))  # DBSCAN con distancia haversine (30 km)
m_db = m.assign(cluster=["ruido" if l == -1 else f"C{l}" for l in db.labels_])  # Etiquetas
print(f"Clústeres DBSCAN (eps = 30 km, mínimo 5 municipios): {db.labels_.max() + 1} · municipios aislados (ruido): {(db.labels_ == -1).sum()}")  # Resultado
fig = px.scatter_geo(m_db, lat="latitud", lon="longitud", color="cluster", hover_name="municipio")  # Mapa
fig.update_traces(marker=dict(size=6))  # Tamaño
fig.update_layout(title="Agrupamientos de municipios por densidad (DBSCAN, haversine)")  # Título
mostrar(mapa_base(fig), 560)  # Muestra
m_db.groupby("cluster").agg(municipios=("municipio", "size"), ejemplo=("municipio", "first"), departamento=("departamento", "first")).sort_values("municipios", ascending=False).head(14)'''),
        md("**Interpretación.** Los municipios del panel están **agrupados**, no distribuidos al azar: el índice de Clark-Evans es "
           "menor que 1 (R ≈ 0,77: la distancia media al vecino más cercano es 23 km frente a 30 km esperados) y la función L "
           "está por encima de la envolvente de aleatoriedad en todas las escalas analizadas. DBSCAN encuentra 13 núcleos "
           "densos, sobre todo en la región Andina (Eje Cafetero, Cundinamarca, Boyacá, Antioquia) y el Caribe, y deja "
           "aislados a 193 municipios, entre ellos casi todos los de la Orinoquía y la Amazonía, donde la cobertura es escasa. Es un **sesgo de muestreo espacial**: el modelo tendrá menos información para "
           "esas regiones, y cualquier métrica global estará dominada por las zonas densas."),

        md("## Matriz de pesos espaciales\n\n"
           "Como el panel solo tiene un tercio de los municipios, la contigüidad (compartir frontera) no es aplicable: muchos "
           "vecinos reales no están. Se usan **k vecinos más cercanos** (distancia haversine), que garantizan que todo "
           "municipio tenga vecinos, con $k = 6$ y estandarización por filas (el rezago espacial es el promedio de los "
           "vecinos). Se comprueba la sensibilidad a esa elección."),
        code('''k6 = np.sort(D, axis=1)[:, 6]  # Distancia al sexto vecino
print(f"Distancia al 6.º vecino: mediana {np.median(k6):.0f} km, percentil 90 {np.percentile(k6, 90):.0f} km, máximo {k6.max():.0f} km ({m.loc[np.argmax(k6), 'municipio']})")  # Escala
sens = []  # Sensibilidad del I de Moran a la matriz de pesos
for nombre, W in [("k = 4", sp.pesos_knn(D, 4)), ("k = 6", sp.pesos_knn(D, 6)), ("k = 8", sp.pesos_knn(D, 8)), ("k = 12", sp.pesos_knn(D, 12)),  # k vecinos
                  ("banda 50 km", sp.pesos_banda(D, 50)), ("banda 100 km", sp.pesos_banda(D, 100)), ("banda 200 km", sp.pesos_banda(D, 200))]:  # Bandas
    r = sp.moran(mm["log_incidencia"].to_numpy(), W, permutaciones=499)  # Moran
    sens.append({"matriz": nombre, "municipios sin vecinos": int((W.sum(axis=1) == 0).sum()), "I de Moran": r["I"], "z": r["z"], "p": r["p"]})  # Resultado
pd.DataFrame(sens).set_index("matriz").round(4)'''),

        md("## Autocorrelación espacial global y local"),
        code('''W = sp.pesos_knn(D, 6)  # Matriz elegida
glob = pd.DataFrame({v: sp.moran(mm[v].to_numpy(), W, permutaciones=999) for v in ["log_incidencia", "prevalencia"]}).T  # Moran global
glob.index = ["log incidencia anual", "proporción de semanas con brote"]  # Nombres
glob.round(4)'''),
        code('''zx = mm["log_incidencia"] - mm["log_incidencia"].mean()  # Desviaciones
rez = W @ zx.to_numpy()  # Rezago espacial
pend = np.polyfit(zx, rez, 1)[0]  # Pendiente = I de Moran (con W estandarizada)
fig = px.scatter(x=zx, y=rez, hover_name=mm["municipio"], labels={"x": "log incidencia (centrada)", "y": "promedio de los 6 vecinos"})  # Diagrama de Moran
fig.add_scatter(x=[zx.min(), zx.max()], y=[pend * zx.min(), pend * zx.max()], mode="lines", line=dict(color=ROJO), name=f"pendiente = {pend:.3f}")  # Recta
fig.add_hline(y=0, line_color=GRIS)  # Ejes
fig.add_vline(x=0, line_color=GRIS)  # Ejes
fig.update_layout(title="Diagrama de dispersión de Moran (cuadrantes: AA arriba a la derecha, BB abajo a la izquierda)")  # Título
mostrar(fig, 420)  # Muestra'''),
        code('''Ii, p_loc, cuad = sp.lisa(mm["log_incidencia"].to_numpy(), W, permutaciones=999)  # LISA
q_fdr = es.fdr_bh(p_loc)  # Corrección por FDR (376 pruebas)
mm["LISA"] = np.where(p_loc < 0.05, cuad, "No significativo")  # Clasificación con p < 0,05
mm["Gi*"] = sp.getis_ord(mm["log_incidencia"].to_numpy(), D, k=6)  # Getis-Ord
print("LISA (p < 0,05):", mm["LISA"].value_counts().to_dict())  # Conteos
print(f"LISA significativos tras FDR (q < 0,05): {(q_fdr < 0.05).sum()}")  # Con corrección
print(f"Gi*: focos calientes (z > 1,96): {(mm['Gi*'] > 1.96).sum()} · focos fríos (z < −1,96): {(mm['Gi*'] < -1.96).sum()}")  # Hotspots
fig = make_subplots(1, 2, specs=[[{"type": "scattergeo"}] * 2], subplot_titles=("LISA (p < 0,05)", "Getis-Ord Gi* (puntaje z)"))  # Dos mapas
for cat_, col in CUADRANTE_COLOR.items():  # Cuadrantes LISA
    s = mm["LISA"] == cat_  # Municipios
    fig.add_scattergeo(lat=mm.loc[s, "latitud"], lon=mm.loc[s, "longitud"], text=mm.loc[s, "municipio"], mode="markers", name=cat_,
                       marker=dict(size=7 if cat_ != "No significativo" else 4, color=col, line=dict(width=0.3)), row=1, col=1)
fig.add_scattergeo(lat=mm["latitud"], lon=mm["longitud"], text=mm["municipio"], mode="markers", showlegend=False, row=1, col=2,  # Gi*
                   marker=dict(size=6, color=mm["Gi*"], colorscale="RdBu_r", cmid=0, colorbar=dict(title="z", x=1.0), line=dict(width=0.3)))
fig.update_geos(fitbounds="locations", showcountries=True, landcolor="#f8fafc", showland=True)  # Encuadre
fig.update_layout(title="Focos espaciales de la incidencia de dengue (2015–2021)")  # Título
mostrar(fig, 520)  # Muestra
mm.loc[mm["LISA"] == "Alto-Alto", ["municipio", "departamento", "region"]].groupby("departamento").size().sort_values(ascending=False).to_frame("municipios Alto-Alto")'''),
        md("**Interpretación.** La incidencia tiene **autocorrelación espacial positiva clara** (I de Moran ≈ 0,31 con k = 6, "
           "muy por encima de su esperanza −0,003), y el resultado es robusto: con cualquier k o banda de distancia el I es "
           "positivo y significativo. Los municipios vecinos tienen cargas de dengue parecidas. Los indicadores locales señalan "
           "76 municipios Alto-Alto (focos calientes), concentrados en el Meta, el Tolima, el Huila y el Putumayo, y 30 "
           "Bajo-Bajo; Getis-Ord coincide (66 focos calientes y 36 fríos). Tras corregir las 376 pruebas por FDR siguen "
           "siendo significativos 53 municipios, así que los focos no son un artefacto de las comparaciones múltiples.\n\n"
           "En cambio, la **proporción de semanas en brote** tiene una autocorrelación global débil y no significativa "
           "(I ≈ 0,05, p ≈ 0,06): el canal endémico compara "
           "a cada municipio consigo mismo y elimina buena parte de la diferencia de nivel entre zonas. El componente espacial "
           "del brote es **sincrónico** (vecinos que entran en brote al mismo tiempo), lo que se estudia en el capítulo "
           "espacio-temporal."),

        md("## Semivariograma empírico\n\n"
           "Los datos son de área (municipios) y no puntuales y continuos, así que el semivariograma se presenta como "
           "descripción complementaria de la escala de dependencia, no para krigeado."),
        code('''iu = np.triu_indices(len(m), 1)  # Pares sin repetir
dist, dif2 = D[iu], 0.5 * (mm["log_incidencia"].to_numpy()[:, None] - mm["log_incidencia"].to_numpy()[None, :])[iu] ** 2  # Distancia y semivarianza
bordes = np.arange(0, 820, 40)  # Intervalos de 40 km
clase = np.digitize(dist, bordes)  # Intervalo de cada par
vario = pd.DataFrame({"d": dist, "g": dif2, "c": clase}).groupby("c").agg(distancia=("d", "mean"), semivarianza=("g", "mean"), pares=("g", "size"))  # Semivariograma
fig = px.scatter(vario, x="distancia", y="semivarianza", size="pares", labels={"distancia": "distancia (km)"})  # Puntos
fig.add_hline(y=mm["log_incidencia"].var(), line_dash="dot", line_color=ROJO, annotation_text="varianza total (meseta)")  # Meseta
fig.update_layout(title="Semivariograma empírico de la log incidencia")  # Título
mostrar(fig, 360)  # Muestra'''),
        md("**Interpretación.** La semivarianza crece con la distancia y alcanza la meseta (la varianza total) hacia los "
           "100 km: esa es la escala aproximada de la dependencia espacial (el **alcance**). Más allá, conocer a un municipio "
           "dice poco de otro. El efecto pepita es alto (≈ 70 % de la meseta ya en los primeros 40 km): buena parte de la "
           "variación es propia de cada municipio, y la dependencia espacial, aunque clara, es parcial."),

        md("## Heterogeneidad espacial"),
        code('''regiones = sorted(tr["region"].unique())  # Regiones
het = tr.groupby("region").agg(municipios=("COD_MUN_N", "nunique"), prevalencia=(cf.TARGET, "mean"),  # Por región
                               temperatura_medida=("temp_mean_m4", lambda s: s[tr.loc[s.index, "temp_missing"] == 0].mean()))
from sklearn.metrics import roc_auc_score  # AUC por región
for v in ["lc_m4", "oni_l8", "temp_anom_m8", "prec_total_m8"]:  # Relación de cada variable con el objetivo, por región
    het[f"AUC {v}"] = [roc_auc_score(tr.loc[tr["region"] == r, cf.TARGET], tr.loc[tr["region"] == r, v]) for r in het.index]  # AUC univariado
h, p, eps2 = es.kruskal([mm.loc[mm["region"] == r, "log_incidencia"] for r in regiones])  # Incidencia entre regiones
print(f"Kruskal-Wallis de la log incidencia entre regiones: H = {h:.1f}, p = {p:.2e}, ε² = {eps2:.3f}")  # Resultado
het.round(3)'''),
        md("**Interpretación.** Hay heterogeneidad espacial en el **nivel** (la incidencia difiere entre regiones, ε² ≈ 0,17) y "
           "en las **relaciones**: la historia de casos predice bien en todas las regiones (AUC 0,86–0,90), pero la asociación "
           "del ONI (0,62–0,69) y de las anomalías de temperatura (0,52–0,63, más fuerte en el Pacífico) cambia de intensidad, "
           "y la de la lluvia cambia incluso de dirección (AUC por encima de 0,5 en el Caribe y por debajo en las demás regiones). Un "
           "único coeficiente climático para todo el país es una simplificación; modelos con interacciones por región o "
           "no lineales (siguientes entregables) pueden capturarlo."),

        md("## Distancias y escala de agregación (MAUP)"),
        code('''grados = np.sqrt((lat[:, None] - lat[None, :]) ** 2 + (lon[:, None] - lon[None, :]) ** 2) * 111.32  # Distancia euclidiana sobre grados (incorrecta), en km
err = np.abs(grados[iu] - D[iu]) / D[iu]  # Error relativo frente a haversine
print(f"Error de usar distancia euclidiana sobre grados: mediana {np.median(err):.1%}, máximo {err.max():.1%} → se usa haversine")  # Justificación
dep = tr.groupby("depto").agg(casos=("casos_totales", "sum"), pob=("poblacion", "mean"), n=("COD_MUN_N", "nunique"), prevalencia=(cf.TARGET, "mean"))  # Nivel departamento
dep["pob"] = tr.groupby(["depto", "COD_MUN_N"])["poblacion"].mean().groupby("depto").sum()  # Población del departamento (municipios del panel)
dep["log_incidencia"] = np.log1p(dep["casos"] / dep["pob"] / 7 * 1e5)  # Incidencia anual
t_mun = tr[tr["temp_missing"] == 0].groupby("COD_MUN_N")["temp_mean_m4"].mean()  # Temperatura medida por municipio
t_dep = tr[tr["temp_missing"] == 0].groupby("depto")["temp_mean_m4"].mean()  # Por departamento
r_mun = stats.spearmanr(t_mun, mm.set_index("COD_MUN_N").loc[t_mun.index, "log_incidencia"])[0]  # Correlación municipal
r_dep = stats.spearmanr(t_dep, dep.loc[t_dep.index, "log_incidencia"])[0]  # Correlación departamental
print(f"Correlación de Spearman temperatura – incidencia: municipios {r_mun:.2f} (n = {len(t_mun)}) · departamentos {r_dep:.2f} (n = {len(t_dep)})")  # MAUP'''),
        md("**Interpretación.** Cerca del ecuador la distancia euclidiana sobre grados se equivoca poco (mediana 0,2 %, "
           "máximo 2 %), pero el error no es nulo y crece con la latitud (un grado de longitud no mide lo mismo en todas "
           "partes), así que se usa siempre haversine. El **efecto de la escala** "
           "(MAUP) es visible: la correlación entre temperatura e incidencia es 0,10 entre municipios y se duplica (0,22) "
           "entre departamentos, porque la agregación suaviza la variación local y exagera la relación. Por eso el modelo trabaja a escala "
           "municipal y las variables espaciales se construyen con vecinos, no con promedios departamentales."),

        md("## Ingeniería de características espaciales y consecuencias\n\n"
           "* **Latitud y longitud crudas no se usan como predictoras**: el objetivo incluye generalizar a zonas nuevas, y las "
           "coordenadas permitirían memorizar *dónde* hubo brotes. La información geográfica relevante entra por el clima "
           "(que ya depende de la ubicación) y por variables de vecindad.\n"
           "* **Variables de vecindad (rezago espacial)**: el promedio de la log incidencia y de los brotes de los k vecinos en "
           "la semana $t$ (disponibles en $t$). Su utilidad se confirma en el modelo base con el I de Moran de los residuos; "
           "se incorporan en los modelos de los siguientes entregables.\n"
           "* **Validación por bloques espaciales**: como municipios cercanos se parecen (alcance ≈ 100 km), una "
           "validación aleatoria por municipios sería optimista. Se usa `GroupKFold` por **departamento** con una **zona de "
           "separación de 50 km**: se eliminan del entrenamiento los municipios a menos de 50 km de un municipio de prueba."),
    ]
    guardar("08_espacial", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
