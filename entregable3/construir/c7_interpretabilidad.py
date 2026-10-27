"""Cuaderno 7: interpretabilidad global y local con SHAP, y comparación con LIME (XGBoost)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Interpretabilidad: valores de Shapley (SHAP) y LIME\n\n"
           "El valor de Shapley de la variable $j$ para la fila $\\mathbf x$ es su contribución marginal promedio sobre todos los "
           "subconjuntos $S$ de las demás variables:\n\n"
           "$$\\phi_j(\\mathbf x) = \\sum_{S \\subseteq F \\setminus \\{j\\}} \\frac{|S|!\\,(|F| - |S| - 1)!}{|F|!}\\Big[f_{S \\cup \\{j\\}}(\\mathbf x) - f_S(\\mathbf x)\\Big].$$\n\n"
           "Es la única atribución que cumple **eficiencia** ($\\sum_j \\phi_j = f(\\mathbf x) - E[f]$), **simetría** (variables con igual "
           "aporte reciben igual valor), **nulidad** y **aditividad** (en un ensamble, los valores son la suma de los de sus árboles). Para "
           "modelos de árboles, TreeSHAP los calcula **exactamente** en tiempo polinomial. En clasificación se expresan en log-odds."),
        code(PREAMBULO + '''
import shap  # Valores de Shapley
import matplotlib.pyplot as plt  # Gráficos de SHAP
from e3 import interpretabilidad as it, evaluacion as ev  # Interpretabilidad
tabla = pd.read_csv(c3.RESULTADOS / "tabla_maestra.csv")  # Tabla maestra
ext = tabla.groupby(["tarea", "modelo", "balanceo", "optimizador"])[["AUC-PR", "RMSE"]].mean().reset_index()  # Bucle externo
mejor_clf = ext[ext["tarea"] == "clf"].sort_values("AUC-PR", ascending=False).iloc[0]  # Mejor clasificador
mejor_xgb = ext[(ext["tarea"] == "clf") & (ext["modelo"] == "XGBoost")].sort_values("AUC-PR", ascending=False).iloc[0]  # Mejor XGBoost (para LIME)
mejor_reg = ext[(ext["tarea"] == "reg") & ext["modelo"].isin(["Random Forest", "XGBoost", "Árbol de decisión"])].sort_values("RMSE").iloc[0]  # Mejor regresor de árboles
pd.DataFrame([mejor_clf, mejor_xgb, mejor_reg], index=["mejor clasificador", "mejor XGBoost (clasificación)", "mejor regresor de árboles"])'''),

        md("## Clasificación: interpretabilidad global"),
        code('''tr, va, te = dd.cargar()  # Periodos
pipe, est, prep = it.reajustar("clf", mejor_clf["modelo"], mejor_clf["balanceo"], mejor_clf["optimizador"])  # Mejor clasificador
muestra = te.sample(3000, random_state=c3.SEMILLA)  # Filas de la prueba a explicar
Xp = prep(muestra[dd.X_COLS].to_numpy())  # Espacio en que trabaja el modelo (estandarizado)
sv = it.shap_arbol(est, Xp)  # Valores SHAP exactos
sv.data = muestra[dd.X_COLS].to_numpy()  # Se muestran los valores originales (no estandarizados) en el eje de color
plt.figure()  # Figura nueva
shap.plots.beeswarm(sv, max_display=15, show=False)  # Gráfico resumen
plt.title(f"SHAP global — {mejor_clf['modelo']} ({mejor_clf['balanceo']}, {mejor_clf['optimizador']})")  # Título
plt.tight_layout()  # Ajuste
plt.show()  # Muestra
imp = pd.Series(np.abs(sv.values).mean(axis=0), index=dd.X_COLS).sort_values(ascending=False)  # Importancia media |SHAP|
imp.head(10).to_frame("media |SHAP| (log-odds)")'''),
        md("**Interpretación.** El modelo se apoya sobre todo en la **historia reciente del brote**: la proporción de semanas en brote en las "
           "últimas 8 (`brote_m8`) es, de lejos, la variable con mayor impacto, seguida del log de casos de 4 semanas, los casos frente al "
           "umbral objetivo y el brote actual. Después aparecen el **flujo de personas** y **El Niño** (`oni_anom`). En el gráfico de "
           "enjambre, los valores altos (rojo) de estas variables empujan hacia el brote. Las variables climáticas locales tienen impactos "
           "pequeños, coherente con todo el EDA."),

        md("## Clasificación: interpretabilidad local\n\n"
           "Dos filas de la prueba: un brote **bien anticipado** (verdadero positivo con probabilidad alta) y un brote **no anticipado** "
           "(falso negativo con probabilidad baja, el error más costoso)."),
        code('''p = ev.metricas_clf  # (solo para recordar la función de métricas)
proba = pipe.predict_proba(muestra[dd.X_COLS].to_numpy())[:, 1]  # Probabilidades de la muestra
y_m = muestra["y"].to_numpy()  # Objetivo
i_bien = int(np.argmax(np.where(y_m == 1, proba, -1)))  # Brote con la probabilidad más alta
i_mal = int(np.argmin(np.where(y_m == 1, proba, 2)))  # Brote con la probabilidad más baja
for i, titulo in [(i_bien, "Brote bien anticipado"), (i_mal, "Brote no anticipado (falso negativo)")]:  # Dos casos
    fila = muestra.iloc[i]  # Fila
    print(f"{titulo}: {fila['municipio']} ({fila['departamento']}), semana del {fila['week_start'].date()} · probabilidad {proba[i]:.3f} · casos en t: {int(fila['casos_totales'])}")
    plt.figure()  # Figura
    shap.plots.waterfall(sv[i], max_display=10, show=False)  # Cascada
    plt.title(titulo)  # Título
    plt.tight_layout()  # Ajuste
    plt.show()  # Muestra'''),
        md("**Interpretación.** En el **brote bien anticipado** (Floridablanca, septiembre de 2023, probabilidad 0,93) casi todas las "
           "variables de historia empujan hacia el brote: el municipio ya venía con semanas en brote y casos sobre su umbral. En el **brote "
           "no anticipado** (Arenal, febrero de 2023, probabilidad 0,009) el municipio tenía **0 casos** en la semana de la predicción y "
           "ninguna semana reciente en brote: todas las variables de historia empujan hacia «no brote». Es un brote que aparece sin señal "
           "previa en el propio municipio; solo información externa (vecinos en brote, movilidad) podría anticiparlo, lo que vuelve a apuntar "
           "a las variables de vecindad."),

        md("## XGBoost: SHAP frente a LIME\n\n"
           "LIME perturba la fila, pondera las perturbaciones por cercanía y ajusta una **regresión lineal local**; sus pesos dependen de "
           "la discretización, del tamaño del vecindario y de la muestra aleatoria. SHAP reparte la predicción con una regla axiomática "
           "(eficiencia, simetría, aditividad) y, para árboles, es exacto."),
        code('''pipe_x, est_x, prep_x = it.reajustar("clf", "XGBoost", mejor_xgb["balanceo"], mejor_xgb["optimizador"])  # Mejor XGBoost
Xp_x = prep_x(muestra[dd.X_COLS].to_numpy())  # Muestra transformada
sv_x = it.shap_arbol(est_x, Xp_x)  # SHAP
ent_x = prep_x(dd.submuestra(tr, 5000)[dd.X_COLS].to_numpy())  # Datos de referencia para LIME
comparacion = []  # SHAP frente a LIME
for i, nombre in [(i_bien, "brote bien anticipado"), (i_mal, "brote no anticipado")]:  # Mismas dos filas
    pesos, r2 = it.lime_local(est_x, ent_x, Xp_x[i])  # LIME
    pesos["variable"] = pesos["condición"].str.extract(r"([a-z][a-z_0-9]*)")[0]  # Variable de cada condición
    top_shap = pd.Series(sv_x.values[i], index=dd.X_COLS).abs().sort_values(ascending=False).head(10)  # 10 variables con mayor |SHAP|
    comun = len(set(top_shap.index) & set(pesos["variable"]))  # Coincidencias
    display(Markdown(f"**{nombre}** · R² del modelo lineal local de LIME: {r2:.2f} · variables en común entre los 10 principales: {comun} de 10"))
    display(pesos.assign(**{"SHAP de la variable": pesos["variable"].map(pd.Series(sv_x.values[i], index=dd.X_COLS))}).round(3))  # Tabla
    comparacion.append({"fila": nombre, "R² LIME": r2, "coincidencias top 10": comun})
pd.DataFrame(comparacion)'''),
        code('''import scipy.stats as ss  # Correlación de rangos
rho = []  # Estabilidad de LIME frente a SHAP en 20 filas
for i in range(20):  # Filas
    pesos, _ = it.lime_local(est_x, ent_x, Xp_x[i], num=34)  # Todas las variables
    pesos["variable"] = pesos["condición"].str.extract(r"([a-z][a-z_0-9]*)")[0]  # Variable
    l = pesos.groupby("variable")["peso LIME"].sum().reindex(dd.X_COLS).fillna(0)  # Peso por variable
    rho.append(ss.spearmanr(np.abs(l), np.abs(sv_x.values[i]))[0])  # Concordancia de importancias
print(f"Correlación de Spearman entre |LIME| y |SHAP| por fila (20 filas): mediana {np.median(rho):.2f}, rango {np.min(rho):.2f} a {np.max(rho):.2f}")'''),
        md("**Interpretación.** LIME y SHAP coinciden mejor en el brote bien anticipado que en el falso negativo (las coincidencias y el R² "
           "del modelo lineal local de LIME están en las tablas), y la concordancia general entre las importancias es solo moderada "
           "(correlación de rangos por fila en el rango reportado arriba). Las diferencias metodológicas lo explican: LIME **discretiza** las "
           "variables en cuartiles y ajusta una regresión lineal a perturbaciones aleatorias, así que depende del tamaño del vecindario, de la "
           "semilla y de la discretización, y no garantiza que sus pesos sumen la predicción; SHAP (TreeSHAP) es **exacto** para árboles y "
           "cumple eficiencia, simetría y aditividad. Divergen más cuando el modelo es muy no lineal alrededor de la fila (como en el falso "
           "negativo, en una zona plana de probabilidad casi 0, donde el R² de LIME cae) o cuando hay variables colineales, entre las que LIME "
           "reparte el peso de forma inestable."),

        md("## Regresión: SHAP del mejor regresor de árboles"),
        code('''pipe_r, est_r, prep_r = it.reajustar("reg", mejor_reg["modelo"], "—", mejor_reg["optimizador"])  # Mejor regresor
Xp_r = prep_r(muestra[dd.X_COLS].to_numpy())  # Muestra transformada
sv_r = it.shap_arbol(est_r, Xp_r, tarea="reg")  # SHAP (en escala log(1 + casos))
sv_r.data = muestra[dd.X_COLS].to_numpy()  # Valores originales
plt.figure()  # Figura
shap.plots.beeswarm(sv_r, max_display=15, show=False)  # Resumen
plt.title(f"SHAP global — regresión, {mejor_reg['modelo']} ({mejor_reg['optimizador']})")  # Título
plt.tight_layout()  # Ajuste
plt.show()  # Muestra
pred_r = pipe_r.predict(muestra[dd.X_COLS].to_numpy())  # Predicciones
err = np.abs(muestra["y_reg"].to_numpy() - pred_r)  # Error absoluto
for i, titulo in [(int(np.argmin(err + (muestra["y_reg"].to_numpy() < 2) * 10)), "Semana epidémica bien predicha"), (int(np.argmax(err)), "Mayor error de la muestra")]:  # Dos casos
    fila = muestra.iloc[i]  # Fila
    print(f"{titulo}: {fila['municipio']}, semana del {fila['week_start'].date()} · observado {fila['y_reg']:.2f} · predicho {pred_r[i]:.2f}")
    plt.figure()  # Figura
    shap.plots.waterfall(sv_r[i], max_display=10, show=False)  # Cascada
    plt.title(titulo)  # Título
    plt.tight_layout()  # Ajuste
    plt.show()  # Muestra'''),
        md("**Interpretación.** En regresión el patrón es el mismo: las medias y rezagos del log de casos dominan, y El Niño y el flujo de "
           "personas vienen después. El error más grande de la muestra (Uribia, octubre de 2024) es otra vez un brote súbito: el modelo "
           "predice 0,55 (≈ 0,7 casos) y ocurren ≈ 18. La cascada muestra que las variables de historia, bajas en ese momento, arrastran la "
           "predicción hacia abajo."),
    ]
    guardar("7_interpretabilidad", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
