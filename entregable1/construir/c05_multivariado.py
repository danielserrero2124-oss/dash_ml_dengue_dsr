"""Cuaderno 5: análisis multivariado (sección 2.4 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Análisis multivariado\n\n"
           "Reducción de dimensionalidad (PCA y t-SNE como apoyo visual), atípicos multivariados (Mahalanobis e Isolation "
           "Forest) y estructura de subpoblaciones (agrupamiento de municipios). Todo se ajusta con el entrenamiento."),
        code(PREAMBULO + '''
from scipy import stats  # Distribución χ²
from sklearn.cluster import KMeans  # Agrupamiento
from sklearn.decomposition import PCA  # Componentes principales
from sklearn.ensemble import IsolationForest  # Atípicos por aislamiento
from sklearn.manifold import TSNE  # Proyección no lineal (solo visual)
from sklearn.metrics import silhouette_score  # Calidad del agrupamiento
from sklearn.preprocessing import StandardScaler  # Estandarización
df = d.panel()  # Panel completo
mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Solo entrenamiento
esc = StandardScaler().fit(tr[cf.FEATURES])  # Estandarización ajustada con el entrenamiento
Z = esc.transform(tr[cf.FEATURES])  # Matriz estandarizada (132 728 × 34)'''),

        md("## Componentes principales (PCA)"),
        code('''pca = PCA(random_state=cf.SEMILLA).fit(Z)  # PCA completo
var = pd.DataFrame({"varianza explicada": pca.explained_variance_ratio_, "acumulada": np.cumsum(pca.explained_variance_ratio_)},  # Varianza
                   index=[f"CP{i + 1}" for i in range(len(cf.FEATURES))])
fig = make_subplots(specs=[[{"secondary_y": True}]])  # Dos ejes
fig.add_bar(x=var.index, y=var["varianza explicada"], name="Por componente", marker_color=AZUL)  # Varianza
fig.add_scatter(x=var.index, y=var["acumulada"], name="Acumulada", line=dict(color=ROJO), secondary_y=True)  # Acumulada
fig.update_yaxes(tickformat=".0%")  # Porcentaje
fig.update_layout(title="Gráfico de sedimentación (scree plot)")  # Título
mostrar(fig, 360)  # Muestra
k80, k90 = int(np.searchsorted(var["acumulada"], 0.8) + 1), int(np.searchsorted(var["acumulada"], 0.9) + 1)  # Componentes necesarios
kaiser = int((pca.explained_variance_ > 1).sum())  # Criterio de Kaiser (autovalor > 1)
print(f"Componentes para 80 % de la varianza: {k80} · para 90 %: {k90} · con autovalor > 1: {kaiser} (de 34 variables)")  # Dimensionalidad efectiva
var.head(8)'''),
        code('''cargas = pd.DataFrame(pca.components_[:4].T, index=cf.FEATURES, columns=["CP1", "CP2", "CP3", "CP4"])  # Cargas
top = pd.concat({c: cargas[c].abs().sort_values(ascending=False).head(6).index.to_series().reset_index(drop=True) for c in cargas}, axis=1)  # Variables dominantes
top.index = [f"#{i + 1}" for i in range(6)]  # Orden
top'''),
        code('''rng = np.random.default_rng(cf.SEMILLA)  # Generador
idx = rng.choice(len(Z), 6000, replace=False)  # Muestra para la figura
P = pca.transform(Z[idx])[:, :2]  # Dos primeras componentes
yb = tr[cf.TARGET].to_numpy()[idx]  # Objetivo
fig = go.Figure()  # Dispersión
for val, col, nom in [(0, "#94a3b8", "No brote"), (1, ROJO, "Brote en 4 semanas")]:  # Por clase
    s = yb == val  # Filas de la clase
    fig.add_scatter(x=P[s, 0], y=P[s, 1], mode="markers", name=nom, marker=dict(color=col, size=4, opacity=0.5))  # Puntos
fig.update_layout(title="Proyección en CP1–CP2 (6 000 filas al azar)", xaxis_title="CP1", yaxis_title="CP2")  # Formato
mostrar(fig, 460)  # Muestra'''),
        md("**Interpretación.** La dimensionalidad efectiva es mucho menor que 34: 8 componentes explican el 80 % de la "
           "varianza, 12 el 90 %, y solo 7 tienen autovalor mayor que 1, porque las predictoras forman **bloques redundantes** "
           "(rezagos y medias móviles de una misma variable). La CP1 (27 %) es el eje de la \"historia de casos\"; la CP2 y la "
           "CP3 son ejes de temperatura (nivel y contraste entre ventanas cortas y largas) junto con el NDVI; la CP4 es el eje "
           "de lluvia y vegetación. Los brotes se concentran hacia un extremo del eje de historia de casos, pero "
           "se superponen con los no brotes: ninguna proyección lineal de dos dimensiones separa las clases."),

        md("## t-SNE (solo como herramienta visual)\n\n"
           "t-SNE preserva vecindades locales pero no distancias globales ni densidades: sirve para mirar, no para inferir."),
        code('''idx2 = rng.choice(len(Z), 3000, replace=False)  # Muestra (t-SNE es costoso)
emb = TSNE(n_components=2, perplexity=40, init="pca", random_state=cf.SEMILLA).fit_transform(Z[idx2])  # Proyección
yb2 = tr[cf.TARGET].to_numpy()[idx2]  # Objetivo
reg = tr["region"].to_numpy()[idx2]  # Región
fig = make_subplots(1, 2, subplot_titles=("Coloreado por objetivo", "Coloreado por región"))  # Dos paneles
for val, col, nom in [(0, "#94a3b8", "No brote"), (1, ROJO, "Brote")]:  # Objetivo
    s = yb2 == val  # Filas
    fig.add_scatter(x=emb[s, 0], y=emb[s, 1], mode="markers", name=nom, marker=dict(color=col, size=4, opacity=0.6), row=1, col=1)  # Puntos
for r, col in zip(sorted(set(reg)), [VERDE, AZUL, AMBAR, MORADO, "#0891b2"]):  # Regiones
    s = reg == r  # Filas
    fig.add_scatter(x=emb[s, 0], y=emb[s, 1], mode="markers", name=r, marker=dict(color=col, size=4, opacity=0.6), row=1, col=2)  # Puntos
fig.update_layout(title="t-SNE de las 34 predictoras estandarizadas (3 000 filas)")  # Título
mostrar(fig, 460)  # Muestra
from sklearn.neighbors import NearestNeighbors  # Vecinos en la proyección
vec = NearestNeighbors(n_neighbors=11).fit(emb).kneighbors(emb, return_distance=False)[:, 1:]  # 10 vecinos de cada punto
for nombre, etiqueta in [("región", reg), ("objetivo", yb2)]:  # Pureza de vecindario frente al azar
    pureza = (etiqueta[vec] == etiqueta[:, None]).mean()  # Proporción de vecinos con la misma etiqueta
    azar = sum((etiqueta == e).mean() ** 2 for e in np.unique(etiqueta))  # Esperado si la etiqueta no tuviera estructura
    print(f"Vecinos t-SNE con igual {nombre}: {pureza:.1%} (esperado al azar {azar:.1%})")  # Resultado'''),
        md("**Interpretación.** t-SNE muestra grupos locales que coinciden más con la región (es decir, con el clima del lugar) "
           "que con el objetivo: el 45 % de los 10 vecinos de cada punto son de la misma región (31 % esperado al azar), "
           "mientras que la coincidencia en el objetivo apenas supera el azar (88 % frente a 81 %); los brotes aparecen como zonas densas dentro de varios grupos y no como un grupo aparte. "
           "Confirma que el brote no es una \"subpoblación\" de municipios sino un estado transitorio que depende de la historia "
           "reciente de casos."),

        md("## Atípicos multivariados"),
        code('''sub = Z[rng.choice(len(Z), 30000, replace=False)]  # Muestra para estimar la covarianza
media, prec = sub.mean(axis=0), np.linalg.pinv(np.cov(sub, rowvar=False))  # Media y precisión (pseudo-inversa por la colinealidad)
dif = Z - media  # Desviaciones
d2 = np.einsum("ij,jk,ik->i", dif, prec, dif)  # Distancia de Mahalanobis al cuadrado de cada fila
corte = stats.chi2.ppf(0.999, df=len(cf.FEATURES))  # Percentil 99,9 de χ²(34)
atip_m = d2 > corte  # Atípicos de Mahalanobis
iso = IsolationForest(n_estimators=200, contamination="auto", random_state=cf.SEMILLA, n_jobs=-1).fit(Z)  # Isolation Forest
atip_i = iso.predict(Z) == -1  # Atípicos por aislamiento
y = tr[cf.TARGET].to_numpy()  # Objetivo
pd.DataFrame({"atípicos (%)": [atip_m.mean() * 100, atip_i.mean() * 100, (atip_m & atip_i).mean() * 100],  # Proporción
              "tasa de brote en atípicos": [y[atip_m].mean(), y[atip_i].mean(), y[atip_m & atip_i].mean()],  # Brote entre atípicos
              "tasa de brote en el resto": [y[~atip_m].mean(), y[~atip_i].mean(), y[~(atip_m & atip_i)].mean()]},  # Brote en el resto
             index=["Mahalanobis (χ² 99,9 %)", "Isolation Forest", "Ambos métodos"])'''),
        code('''cuales = tr.loc[atip_m & atip_i, ["municipio", "region", "ANO"]]  # Filas atípicas por ambos métodos
print("Regiones de los atípicos (ambos métodos):")  # Título
print(cuales["region"].value_counts(normalize=True).round(3).to_string())  # Composición
print("\\nMunicipios más frecuentes:")  # Título
print(cuales["municipio"].value_counts().head(8).to_string())  # Municipios'''),
        md("**Interpretación.** Mahalanobis marca el 6 % de las filas e Isolation Forest el 15 %; las marcadas por ambos "
           "(≈ 5 %) tienen una tasa de brote de ≈ 39 %, más de cuatro veces la del resto. Son semanas epidémicas, grandes "
           "ciudades (Bogotá, Cali, Medellín, con poblaciones y flujos extremos) o municipios de clima extremo (Guamal, "
           "Buenaventura), no errores. Eliminarlos quitaría justamente los eventos de "
           "interés, así que **se conservan**. El dato confirma la decisión del análisis univariado (transformar, no eliminar)."),

        md("## Subpoblaciones: agrupamiento exploratorio de municipios\n\n"
           "Se resume cada municipio con su perfil de 2015–2021 (incidencia, proporción de brotes, clima, vegetación, población "
           "y flujo) y se agrupan con k-medias."),
        code('''perfil = tr.groupby("COD_MUN_N").agg(log_incidencia=("incid", lambda s: np.log1p(s.mean())), prevalencia=(cf.TARGET, "mean"),  # Epidemiología
                                     temp=("temp_mean_m4", "mean"), lluvia=("prec_total_m4", "mean"), ndvi=("ndvi_mean_m4", "mean"),  # Clima
                                     lpob=("lpob", "mean"), flujo=("flujo_m4", "mean"))  # Tamaño y movilidad
Zp = StandardScaler().fit_transform(perfil)  # Estandariza
sil = {k: silhouette_score(Zp, KMeans(k, n_init=20, random_state=cf.SEMILLA).fit_predict(Zp)) for k in range(2, 9)}  # Silueta por k
k = max(sil, key=sil.get)  # Mejor k
perfil["grupo"] = KMeans(k, n_init=20, random_state=cf.SEMILLA).fit_predict(Zp)  # Agrupamiento final
print("Silueta por número de grupos:", {kk: round(v, 3) for kk, v in sil.items()}, "→ k =", k)  # Elección
perfil.groupby("grupo").agg(municipios=("prevalencia", "size"), **{c: (c, "mean") for c in perfil.columns[:-1]}).round(3)'''),
        code('''mm = perfil.join(d.municipios().set_index("COD_MUN_N")).reset_index()  # Une coordenadas
mm["grupo"] = "Grupo " + mm["grupo"].astype(str)  # Etiqueta
fig = px.scatter_geo(mm, lat="latitud", lon="longitud", color="grupo", hover_name="municipio",  # Mapa
                     hover_data={"prevalencia": ":.1%", "temp": ":.1f", "latitud": False, "longitud": False})
fig.update_traces(marker=dict(size=7))  # Tamaño
fig.update_layout(title=f"Agrupamiento de municipios por perfil (k = {k})")  # Título
mostrar(mapa_base(fig), 560)  # Muestra'''),
        md("**Interpretación.** La mejor silueta (0,33, con k = 2) indica una estructura de grupos débil a moderada. Los dos "
           "perfiles son: 72 municipios **grandes y conectados** (población y flujo de personas muy superiores) con una "
           "proporción de semanas en brote de ≈ 19 %, y 304 municipios **pequeños** con ≈ 8 %. El clima medio casi no separa "
           "los grupos: la diferencia está en el tamaño y la movilidad, coherente con que el virus circula más donde llegan "
           "más personas. Esto respalda dos decisiones: incluir población, flujo y clima como predictoras, y validar "
           "también por **bloques espaciales** para comprobar que el modelo funciona en grupos de municipios que no vio."),
    ]
    guardar("05_multivariado", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
