"""Cuaderno 6: resultados de regresión — métricas, ajuste por subconjunto y análisis de residuos."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Resultados de regresión: log(1 + casos) en $t + 4$\n\n"
           "Siete modelos × cuatro optimizadores = 28 combinaciones, con la misma validación anidada (criterio: −RMSE)."),
        code(PREAMBULO + '''
from e3 import evaluacion as ev  # Métricas y pruebas de residuos
tabla = pd.read_csv(c3.RESULTADOS / "tabla_maestra.csv")  # Tabla maestra
reg = tabla[tabla["tarea"] == "reg"]  # Regresión
externo = reg.groupby(["modelo", "optimizador"]).agg(**{"RMSE media": ("RMSE", "mean"), "RMSE desv.": ("RMSE", "std"), "MAE media": ("MAE", "mean"),
                                                      "MAE desv.": ("MAE", "std"), "R² media": ("R2", "mean"), "segundos": ("segundos_optimizacion", "sum")}).sort_values("RMSE media")
externo'''),
        code('''tr, va, te = dd.cargar()  # Periodos
finales = {}  # Predicciones de las 28 combinaciones
for f in sorted(ex.FINALES.glob("reg__*.pkl")):  # Reajustes finales
    r = pickle.load(open(f, "rb"))  # Predicciones
    finales[(r["modelo"], r["optimizador"])] = r  # Indexado
mejores = externo.reset_index().groupby("modelo").first().sort_values("RMSE media")  # Mejor optimizador de cada modelo (bucle externo)
sel = [(m, o) for m, o in mejores["optimizador"].items()]  # Las 7 seleccionadas
prueba = pd.DataFrame({f"{m} · {o}": {**ev.metricas_reg(te["y_reg"], finales[(m, o)]["p_te"]), **{f"{k} (validación 2022)": v for k, v in ev.metricas_reg(va["y_reg"], finales[(m, o)]["p_va"]).items()}}
                       for m, o in sel}).T  # Métricas en la prueba y en 2022
prueba["RMSE persistencia (prueba)"] = np.sqrt(np.mean((te["y_reg"] - te["lc_l0"]) ** 2))  # Línea base trivial
prueba.sort_values("RMSE")'''),
        md("**Interpretación.** En el bucle externo XGBoost (RMSE 0,481 ± 0,079), SVR con núcleo aproximado y Random Forest quedan "
           "prácticamente empatados; Ridge y Lasso, a menos de 0,01; el árbol único y KNN, detrás. Las diferencias entre optimizadores son "
           "mínimas (milésimas). En la prueba 2023–2024 XGBoost logra RMSE 0,573, MAE 0,438 y $R^2$ 0,79, **13 % menos error que la "
           "persistencia** (0,659). Las métricas empeoran de 2022 (RMSE 0,455) a la prueba porque 2023–2024 tiene más casos y más variación."),

        md("## Ajuste en entrenamiento, validación y prueba\n\n"
           "Para ver el ajuste en cada subconjunto se promedian, por semana, el valor observado y el predicho de los 376 municipios "
           "(arriba), y se muestra la serie de un municipio (abajo)."),
        code('''m, o = sel[0]  # Mejor regresor
r = finales[(m, o)]  # Sus predicciones
ent = tr.loc[r["idx_tr"]]  # Submuestra de entrenamiento del reajuste final
partes = [("entrenamiento", ent, r["p_tr"], AZUL), ("validación 2022", va, r["p_va"], VERDE), ("prueba 2023–2024", te, r["p_te"], ROJO)]  # Subconjuntos
fig = make_subplots(2, 1, subplot_titles=(f"Promedio semanal de los municipios — {m} ({o})", "Santiago de Cali"), vertical_spacing=0.12)  # Dos paneles
for nombre, df_, pred, col in partes:  # Cada subconjunto
    s = pd.DataFrame({"fecha": df_["week_start"].to_numpy(), "obs": df_["y_reg"].to_numpy(), "pred": pred}).groupby("fecha").mean()  # Semana
    fig.add_scatter(x=s.index, y=s["obs"], line=dict(color="#94a3b8", width=1), name="observado" if nombre == "entrenamiento" else None, showlegend=nombre == "entrenamiento", row=1, col=1)  # Observado
    fig.add_scatter(x=s.index, y=s["pred"], line=dict(color=col, width=2), name=f"predicho ({nombre})", row=1, col=1)  # Predicho
    cali = df_["COD_MUN_N"].to_numpy() == 76001  # Cali
    fig.add_scatter(x=df_["week_start"].to_numpy()[cali], y=df_["y_reg"].to_numpy()[cali], mode="markers", marker=dict(color="#94a3b8", size=4), showlegend=False, row=2, col=1)  # Observado
    fig.add_scatter(x=df_["week_start"].to_numpy()[cali], y=np.asarray(pred)[cali], mode="markers", marker=dict(color=col, size=4), showlegend=False, row=2, col=1)  # Predicho
fig.update_yaxes(title_text="log(1 + casos en t + 4)")  # Eje
mostrar(fig, 620)  # Muestra'''),
        md("**Interpretación.** El modelo sigue bien el nivel medio en los tres subconjuntos, incluida la epidemia de 2024, aunque "
           "**suaviza los picos** y llega con algo de retraso a los cambios bruscos: predice a 4 semanas con la información de la semana "
           "actual. En Cali (abajo) se ve lo mismo a escala municipal: buen seguimiento de la tendencia, subestimación de los máximos."),

        md("## Análisis de residuos\n\n"
           "* **White** (homocedasticidad): regresión auxiliar de $e^2$ sobre el valor ajustado, la temperatura, sus cuadrados y su producto.\n"
           "* **BDS** (independencia, Brock-Dechert-Scheinkman): sobre la serie semanal del residuo medio; detecta dependencia lineal y no lineal.\n"
           "* **Ljung-Box**: autocorrelación serial de la misma serie (rezagos 1, 4 y 8) y proporción de municipios con dependencia.\n"
           "* **Jarque-Bera**: normalidad (asimetría y curtosis)."),
        code('''res = pd.DataFrame({f"{m} · {o}": ev.pruebas_residuos(te, te["y_reg"].to_numpy(), finales[(m, o)]["p_te"]) for m, o in sel}).T  # Pruebas
res[["RMSE"]] = prueba[["RMSE"]]  # RMSE junto a las pruebas
res[["RMSE", "p White", "p BDS", "p Ljung-Box (1)", "p Ljung-Box (4)", "municipios con Ljung-Box p < 0,05", "p Jarque-Bera", "asimetría", "curtosis"]]'''),
        code('''from statsmodels.tsa.stattools import acf  # Autocorrelación
fig = make_subplots(2, 2, subplot_titles=("Residuos frente al valor ajustado (muestra)", "Histograma de residuos", "ACF del residuo medio semanal", "ACF media dentro de los municipios"))  # Paneles
m, o = sel[0]  # Mejor regresor
f_te = finales[(m, o)]["p_te"].astype(float)  # Predicción
e = te["y_reg"].to_numpy() - f_te  # Residuos
i = np.random.default_rng(c3.SEMILLA).choice(len(e), 6000, replace=False)  # Muestra
fig.add_scattergl(x=f_te[i], y=e[i], mode="markers", marker=dict(size=3, color=AZUL, opacity=0.4), showlegend=False, row=1, col=1)  # Dispersión
h = np.histogram(e, 80)  # Histograma
fig.add_bar(x=(h[1][:-1] + h[1][1:]) / 2, y=h[0], marker_color=AZUL, showlegend=False, row=1, col=2)  # Barras
semanal = pd.Series(e, index=te["time_idx"].to_numpy()).groupby(level=0).mean()  # Serie semanal
a = acf(semanal, nlags=12)  # ACF
fig.add_bar(x=np.arange(1, 13), y=a[1:], marker_color=AMBAR, showlegend=False, row=2, col=1)  # ACF semanal
for cc in (1, 2):  # Bandas
    fig.add_hline(y=1.96 / np.sqrt(len(semanal)), line_dash="dot", line_color=GRIS, row=2, col=cc)
    fig.add_hline(y=-1.96 / np.sqrt(len(semanal)), line_dash="dot", line_color=GRIS, row=2, col=cc)
acf_mun = np.nanmean([acf(g.to_numpy(), nlags=12)[1:] for _, g in pd.Series(e, index=te["COD_MUN_N"].to_numpy()).groupby(level=0) if g.std() > 0], axis=0)  # ACF por municipio
fig.add_bar(x=np.arange(1, 13), y=acf_mun, marker_color=VERDE, showlegend=False, row=2, col=2)  # ACF media
fig.update_layout(title=f"Diagnóstico de residuos en la prueba — {m} ({o})")  # Título
mostrar(fig, 620)  # Muestra'''),
        md("**Interpretación.** Los supuestos de independencia y homocedasticidad **no se cumplen** en ningún modelo: White rechaza "
           "la homocedasticidad (la varianza del error crece con el nivel de casos, como anticipó el EDA), BDS rechaza la independencia y "
           "Ljung-Box encuentra autocorrelación en la serie semanal y en ≈ 89 % de los municipios. Los residuos son levemente asimétricos y "
           "con colas un poco más pesadas que la normal. Es la firma de información estructural sin capturar (rachas epidémicas y dinámica "
           "de vecindad). Para la predicción puntual no invalida el modelo, pero obliga a usar pruebas que toleran dependencia en la "
           "comparación (bootstrap estacionario, MCS) y sugiere modelos con rezagos espaciales o del propio error."),
    ]
    guardar("6_regresion", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
