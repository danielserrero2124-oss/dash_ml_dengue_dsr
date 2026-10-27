"""Cuaderno 8: robustez de las métricas y comparación estadística jerárquica."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Robustez y comparación estadística\n\n"
           "Con 140 modelos hay $\\binom{140}{2} = 9\\,730$ pares posibles: probarlos todos con $\\alpha = 0{,}05$ daría ≈ 487 falsos positivos "
           "por azar. La comparación es **jerárquica**: primero una prueba global y solo después comparaciones dirigidas con corrección "
           "por comparaciones múltiples y tamaño del efecto."),
        code(PREAMBULO + '''
import joblib  # Lectura
from sklearn.metrics import average_precision_score, roc_auc_score  # Métricas
from e3 import estadistica as st, evaluacion as ev  # Pruebas y métricas
tabla = pd.read_csv(c3.RESULTADOS / "tabla_maestra.csv")  # Tabla maestra
tr, va, te = dd.cargar()  # Periodos
y_te = te["y"].to_numpy()  # Objetivo de la prueba
semanas = te["time_idx"].to_numpy()  # Semana de cada fila (bloques)
finales = {}  # Predicciones finales
for f in sorted(ex.FINALES.glob("*.pkl")):  # Reajustes finales
    r = pickle.load(open(f, "rb"))  # Predicciones
    finales[(r["tarea"], r["modelo"], r["balanceo"], r["optimizador"])] = r  # Indexado
print(f"Reajustes finales: {len(finales)}")  # 140'''),

        md("## 1. Robustez: media ± desviación en el bucle externo y sensibilidad a la semilla"),
        code('''clf = tabla[tabla["tarea"] == "clf"]  # Clasificación
ext = clf.groupby(["modelo", "balanceo", "optimizador"])["AUC-PR"].agg(["mean", "std"])  # Media y desviación de los 3 pliegues
mej = ext.reset_index().sort_values("mean", ascending=False).groupby("modelo").first()  # Mejor combinación de cada modelo
mej["media ± desv."] = mej.apply(lambda r: f"{r['mean']:.3f} ± {r['std']:.3f}", axis=1)  # Formato
mej.sort_values("mean", ascending=False)[["balanceo", "optimizador", "media ± desv."]]'''),
        code('''rob = joblib.load(c3.RESULTADOS / "robustez_semillas.joblib")  # 5 semillas por modelo
sem = rob.groupby("modelo").agg(**{"AUC-PR media (5 semillas)": ("AUC-PR", "mean"), "desv. por semilla": ("AUC-PR", "std"),
                                   "rango": ("AUC-PR", lambda s: s.max() - s.min()), "AUC-ROC desv. por semilla": ("AUC-ROC", "std")}).sort_values("AUC-PR media (5 semillas)", ascending=False)
sem["desv. entre pliegues externos"] = mej["std"]  # Variabilidad por partición
sem'''),
        md("**Interpretación.** La variabilidad por **semilla** (submuestra, remuestreo e inicialización) es pequeña: una desviación de "
           "0,0006–0,0015 de AUC-PR en la prueba para todos salvo el árbol único (0,008, el más inestable, como se espera de un solo árbol). "
           "La variabilidad **entre pliegues externos** es unas 100 veces mayor (≈ 0,10), porque cada año tiene otra prevalencia. Las "
           "diferencias entre los tres mejores modelos (XGBoost, SVM y Random Forest: 0,821, 0,820 y 0,819) son del mismo orden que la "
           "variabilidad por semilla: **no son mayores que la variabilidad intrínseca**, lo que exige pruebas formales antes de declarar un "
           "ganador."),

        md("## 2. Clasificación — Etapa 1: prueba global de Friedman\n\n"
           "Bloques = las 100 semanas de la prueba (cada semana es un problema de ordenar el riesgo de 376 municipios); en cada semana se "
           "ordenan los modelos por AUC-ROC. Se aplica a los **7 modelos** (mejor combinación de cada uno) y a las **112 combinaciones**."),
        code('''def auc_semanal(claves):  # Matriz semanas × modelos con el AUC-ROC de cada semana
    cols = {}  # Columnas
    for k in claves:  # Modelos
        s = finales[k]["p_te"].astype(float)  # Puntaje
        cols[k] = {w: roc_auc_score(y_te[semanas == w], s[semanas == w]) for w in np.unique(semanas) if 0 < y_te[semanas == w].mean() < 1}  # Semanas con ambas clases
    return pd.DataFrame(cols)  # Tabla

siete = [("clf", m, r["balanceo"], r["optimizador"]) for m, r in mej.iterrows()]  # Mejor combinación de cada modelo
A7 = auc_semanal(siete)  # 100 semanas × 7 modelos
A7.columns = [k[1] for k in siete]  # Nombres
f7 = st.friedman(A7)  # Friedman
pd.DataFrame({"Estadístico de Friedman χ²_F": [f7["chi2"]], "g.l.": [f7["gl"]], "valor p": [f7["p"]], "F de Iman-Davenport": [f7["F"]], "p (F)": [f7["p_F"]],
              "bloques (semanas)": [f7["N"]], "Conclusión": ["Se rechaza H0: hay diferencias" if f7["p"] < 0.05 else "No se rechaza H0"]})'''),
        md("## 3. Clasificación — Etapa 2: Nemenyi y diagrama de Diferencia Crítica"),
        code('''def diagrama_cd(rangos, cd, titulo, alto=None):  # Diagrama de diferencia crítica (una fila por modelo)
    """Cada modelo es un punto en su rango medio con una barra de longitud CD centrada en él (± CD/2): si las barras de dos
    modelos no se tocan, su diferencia de rango supera la CD y son significativamente distintos según Nemenyi. Los modelos
    cuyas barras tocan la franja del mejor (rojo) son estadísticamente indistinguibles de él."""
    r = rangos.sort_values()  # Rangos (mejor arriba)
    nombres = list(r.index)  # Etiquetas
    indist = r - r.iloc[0] < cd  # Indistinguibles del mejor
    fig = go.Figure()  # Figura
    fig.add_vrect(x0=r.iloc[0], x1=r.iloc[0] + cd, fillcolor=ROJO, opacity=0.10, line_width=0)  # Franja de la CD desde el mejor
    fig.add_scatter(x=r.values, y=nombres, mode="markers", marker=dict(size=10, color=np.where(indist, ROJO, AZUL)),  # Rango medio
                    error_x=dict(type="constant", value=cd / 2, color="#94a3b8", thickness=2), showlegend=False,
                    hovertemplate="%{y}: rango %{x:.2f}<extra></extra>")
    fig.update_layout(title=titulo, xaxis_title=f"rango medio (1 = mejor) · barras de longitud CD = {cd:.2f}", yaxis=dict(autorange="reversed"),
                      margin=dict(l=320))  # Formato
    mostrar(fig, alto or max(320, 34 * len(r) + 120))  # Muestra

cd7 = st.diferencia_critica(f7["k"], f7["N"])  # Diferencia crítica
diagrama_cd(f7["rangos"], cd7, f"Diagrama de Diferencia Crítica — 7 modelos (Nemenyi, α = 0,05, N = {f7['N']} semanas)")  # Diagrama
f7["rangos"].to_frame("rango medio")'''),
        code('''todas = [k for k in finales if k[0] == "clf"]  # 112 combinaciones
A112 = auc_semanal(todas)  # 100 semanas × 112
A112.columns = [f"{k[1]} · {k[2].replace('class_weight balanced', 'pesos')} · {k[3].split(' ')[0]}" for k in todas]  # Nombres cortos
f112 = st.friedman(A112)  # Friedman sobre todo el universo
cd112 = st.diferencia_critica(f112["k"], f112["N"])  # CD con k = 112
print(f"112 combinaciones: χ²_F = {f112['chi2']:.1f}, g.l. = {f112['gl']}, p = {f112['p']:.2e} · CD de Nemenyi = {cd112:.2f}")
top = f112["rangos"].head(20)  # Las 20 mejores
lider = top.iloc[0]  # Mejor rango
indist = f112["rangos"][f112["rangos"] - lider < cd112]  # Indistinguibles del líder
diagrama_cd(top, cd112, f"Diferencia Crítica — 20 mejores de 112 combinaciones (CD = {cd112:.2f})")  # Diagrama
print(f"Combinaciones estadísticamente indistinguibles de la mejor ({top.index[0]}): {len(indist)} de 112")'''),
        md("**Interpretación.** Friedman rechaza con claridad que los 7 modelos sean equivalentes ($\\chi^2_F = 437{,}6$, $p \\approx 0$). Con "
           "Nemenyi ($CD = 0{,}90$ para $k = 7$ y $N = 100$), **XGBoost (1,84), SVM (2,10) y Random Forest (2,55) forman un grupo "
           "estadísticamente indistinguible**; regresión logística, KNN y el árbol forman un segundo grupo (4,7–5,2), y Naive Bayes queda solo "
           "al final (6,7). Sobre las 112 combinaciones, la CD crece a ≈ 20 (con $k = 112$ es muy exigente) y **30 combinaciones** son "
           "indistinguibles de la mejor: casi todas de XGBoost, Random Forest y SVM con distintos balanceos y optimizadores. El diagrama "
           "resume en una figura lo que serían 9 730 comparaciones."),

        md("## 4. Comparaciones dirigidas: DeLong con corrección de Holm, bootstrap BCa y delta de Cliff\n\n"
           "Se comparan los **tres mejores modelos** de la Etapa 1 y la regresión logística como referencia lineal."),
        code('''orden = f7["rangos"].index.tolist()  # Modelos por rango
candidatos = orden[:3] + (["Regresión logística"] if "Regresión logística" not in orden[:3] else [])  # Subconjunto reducido
punt = {m: finales[k]["p_te"].astype(float) for k in siete for m in [k[1]] if m in candidatos}  # Puntajes
pares = [(a, b) for i, a in enumerate(candidatos) for b in candidatos[i + 1:]]  # Pares
dl = st.delong_holm(y_te, punt, pares)  # DeLong + Holm
mun = te["COD_MUN_N"].to_numpy()  # Conglomerados
ic, cliff = [], []  # Bootstrap BCa y Cliff
for a, b in pares:  # Cada par
    obs, lo, hi = st.bca_municipios(np.c_[y_te, punt[a], punt[b]], np.zeros(len(y_te)), mun,
                                     lambda M, _: roc_auc_score(M[:, 0], M[:, 1]) - roc_auc_score(M[:, 0], M[:, 2]), B=1000)  # ΔAUC con BCa por municipio
    ic.append((lo, hi))  # IC
    cliff.append(st.delta_cliff(A7[a], A7[b]))  # Dominancia semanal
dl["IC 95 % BCa (municipios)"] = [f"[{lo:.4f}; {hi:.4f}]" for lo, hi in ic]  # IC bootstrap
dl["δ de Cliff (AUC semanal)"] = cliff  # Tamaño del efecto
dl["magnitud"] = [st.magnitud_cliff(x) for x in cliff]  # Etiqueta
dl'''),
        code('''icm = []  # IC BCa de las métricas principales de los 7 modelos
for k in siete:  # Cada modelo
    s = finales[k]["p_te"].astype(float)  # Puntaje
    for nombre, met in [("AUC-ROC", roc_auc_score), ("AUC-PR", average_precision_score)]:  # Métricas
        v, lo, hi = st.bca_municipios(y_te, s, mun, met, B=1000)  # BCa
        icm.append({"modelo": k[1], "métrica": nombre, "valor": v, "IC 95 % inferior": lo, "IC 95 % superior": hi})
pd.DataFrame(icm).pivot(index="modelo", columns="métrica", values=["valor", "IC 95 % inferior", "IC 95 % superior"]).round(4)'''),
        md("**Interpretación.** Entre los finalistas, DeLong solo separa con claridad a los modelos de árboles y la SVM de la regresión "
           "logística (ΔAUC ≈ 0,011–0,012, $p < 0{,}001$ tras Holm, IC BCa que excluyen el 0, delta de Cliff pequeño a mediano). **XGBoost "
           "frente a la SVM no es significativo tras Holm** ($p$ = 0,095) y **la SVM frente a Random Forest tampoco** ($p$ = 0,43). XGBoost "
           "frente a Random Forest sí es significativo ($p < 0{,}001$), pero la diferencia es de 0,0014 de AUC y el delta de Cliff es "
           "**despreciable** (0,045): con 37 600 filas, DeLong detecta diferencias sin relevancia práctica. Los intervalos BCa de AUC-PR de los "
           "tres mejores (≈ ±0,025) se superponen casi por completo. Conclusión: XGBoost es el mejor modelo puntual, pero **XGBoost, SVM y "
           "Random Forest son, en la práctica, equivalentes**; los tres son claramente superiores a los modelos lineales y de vecindad."),

        md("## 5. Regresión: secuencia jerárquica\n\n"
           "Las pérdidas se agregan por semana (error cuadrático medio de los 376 municipios en cada una de las 100 semanas de la prueba), "
           "lo que da series temporales de pérdidas comparables entre modelos.\n\n"
           "1. **Model Confidence Set** sobre las 28 combinaciones (α = 0,10, bootstrap estacionario por bloques).\n"
           "2. **Giacomini-White** (habilidad condicional) entre los finalistas y **Clark-West** para el par anidado (persistencia ⊂ Ridge).\n"
           "3. **Diebold-Mariano con corrección HLN** entre el mejor y el segundo, con **bootstrap estacionario** para el IC.\n"
           "4. **SPA de Hansen**: ¿algún modelo supera genuinamente a la persistencia?\n"
           "5. **d de Cohen** como tamaño del efecto."),
        code('''y_r = te["y_reg"].to_numpy()  # Objetivo de regresión
regs = {(k[1], k[3]): finales[k]["p_te"].astype(float) for k in finales if k[0] == "reg"}  # 28 predicciones
perdidas = pd.DataFrame({f"{m} · {o}": pd.Series((y_r - f) ** 2).groupby(semanas).mean().to_numpy() for (m, o), f in regs.items()})  # 100 semanas × 28
mcs_t = st.mcs(perdidas, alfa=0.10)  # Model Confidence Set
mcs_t'''),
        code('''finalistas = mcs_t[mcs_t["¿en el conjunto?"]].index.tolist()  # Modelos en el MCS
mejor, segundo = mcs_t.index[0], mcs_t.index[1]  # Los dos de menor pérdida
filas = []  # Comparaciones dirigidas
for a in finalistas[:4]:  # Finalistas (hasta 4)
    for b in finalistas[:4]:  # Pares
        if a < b:  # Cada par una vez
            gw, p_gw = st.giacomini_white(perdidas[a], perdidas[b])  # Giacomini-White
            filas.append({"modelo A": a, "modelo B": b, "GW": gw, "p GW": p_gw})  # Fila
gwt = pd.DataFrame(filas)  # Tabla
if len(gwt):  # Holm
    o_ = np.argsort(gwt["p GW"].to_numpy())  # Orden
    gwt.loc[gwt.index[o_], "p Holm"] = np.minimum(1, np.maximum.accumulate((len(gwt) - np.arange(len(gwt))) * gwt["p GW"].to_numpy()[o_]))  # Ajuste
gwt'''),
        code('''ridge = next(f for (m, o), f in regs.items() if m == "Ridge" and f"{m} · {o}" == mcs_t.loc[mcs_t.index.str.startswith("Ridge")].index[0])  # Mejor Ridge
cw, p_cw = st.clark_west(y_r, te["lc_l0"].to_numpy(), ridge)  # Persistencia (anidada: un solo regresor con coeficiente 1) frente a Ridge
d_med, hln, p_dm = st.diebold_mariano_perdidas(perdidas[segundo], perdidas[mejor], h=1)  # DM-HLN (positivo: el mejor tiene menos pérdida)
m_bs, lo_bs, hi_bs = st.ic_bootstrap_estacionario(perdidas[segundo] - perdidas[mejor])  # IC con bootstrap estacionario
pers = pd.Series((y_r - te["lc_l0"].to_numpy()) ** 2).groupby(semanas).mean().to_numpy()  # Pérdida semanal de la persistencia
p_spa = st.spa(pers, perdidas.to_numpy())  # SPA frente a la persistencia
pd.DataFrame({
    "prueba": ["Clark-West (persistencia ⊂ Ridge)", f"Diebold-Mariano HLN ({segundo} vs {mejor})", "SPA de Hansen (benchmark: persistencia)"],
    "estadístico": [cw, hln, np.nan], "valor p": [p_cw, p_dm, float(p_spa["consistent"])],
    "detalle": [f"H1: Ridge predice mejor que la persistencia", f"diferencia media de pérdida {d_med:.4f}, IC 95 % bootstrap estacionario [{lo_bs:.4f}; {hi_bs:.4f}]",
                f"p inferior {float(p_spa['lower']):.3f}, consistente {float(p_spa['consistent']):.3f}, superior {float(p_spa['upper']):.3f}"],
    "d de Cohen": [np.nan, st.d_cohen(perdidas[segundo], perdidas[mejor]), np.nan]})'''),
        md("**Interpretación.**\n\n"
           "1. **MCS** (α = 0,10): el conjunto de confianza retiene solo **3 de 28** combinaciones: XGBoost con el genético y la SVR con el "
           "genético y con la malla. Las demás son significativamente peores que la mejor.\n"
           "2. **Giacomini-White:** entre los finalistas, la habilidad predictiva **condicional** sí difiere (p de Holm < 0,01): aunque en "
           "promedio son indistinguibles, uno es mejor que otro según el estado reciente (por ejemplo, semanas de subida epidémica frente a "
           "semanas tranquilas). **Clark-West** confirma que Ridge supera a la persistencia, el modelo anidado más simple ($p < 0{,}001$).\n"
           "3. **Diebold-Mariano con HLN** entre los dos de menor pérdida media: la diferencia es significativa ($p < 0{,}001$, IC por "
           "bootstrap estacionario que excluye el 0), pero su **tamaño de efecto es mínimo** (d de Cohen 0,07).\n"
           "4. **SPA** rechaza que ningún modelo supere a la persistencia ($p < 0{,}001$): la mejora sobre la línea base trivial es genuina y "
           "no un artefacto de haber probado 28 modelos (*data snooping*).\n\n"
           "Igual que en clasificación: los modelos de árboles y la SVR con núcleo aproximado son los mejores y superan con claridad a la línea "
           "base, pero las diferencias entre los mejores son pequeñas."),
    ]
    guardar("8_estadistica", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
