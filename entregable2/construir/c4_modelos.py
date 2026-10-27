"""Cuaderno 4: Modelos base (pestaña 3 del dashboard) con detalles adicionales."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Modelos base: regresión logística y SVM lineal\n\n"
           "Acompaña la pestaña **Modelos base** del dashboard. Los dos modelos comparten el preprocesamiento (`SimpleImputer` → "
           "`StandardScaler`), la malla de hiperparámetros ($C$ y `class_weight`), la validación cruzada hacia adelante por años con purga "
           "y el criterio de selección (AUC-PR). Se comparan con cuatro líneas base triviales y se evalúan una sola vez en 2023–2024."),
        code(PREAMBULO + '''
import inspect  # Código fuente
pd.DataFrame({n: {"mejores hiperparámetros": MOD["mejores"][n], "tiempo de GridSearchCV (s)": round(MOD["tiempos"][n], 1),
                  "umbral (máximo F1 en 2022)": round(MOD["umbrales"][n], 3)} for n in ["Regresión logística", "SVM lineal"]}).T'''),
        md("## Detalle: el código de los dos Pipelines\n\n"
           "La regresión logística viene del Primer Entregable (`e1/modelo_base.py`); la SVM lineal y su escalado de Platt están en "
           "`precalculo.py`. `LinearSVC(dual=False)` resuelve el problema primal, más eficiente cuando hay muchas más filas que variables."),
        code('''import precalculo  # Script de precálculo (SVM)
from e1 import modelo_base as mb  # Modelo base logístico
print(inspect.getsource(mb.pipeline_base))  # Pipeline logístico
print(inspect.getsource(precalculo.pipeline_svm))  # Pipeline SVM'''),
        md("## Validación cruzada (GridSearchCV)"),
        code('''for n in ["Regresión logística", "SVM lineal"]:  # Modelos
    tabla, pliegues = MOD["cv"][n]  # Todas las combinaciones y desempeño por pliegue
    display(Markdown(f"**{n}**"))  # Título
    display(tabla.round(4))  # Malla completa (en el dashboard aparece resumida)
    display(pliegues.round(4))  # Mejor combinación por pliegue'''),
        md("**Interpretación.** Los dos modelos eligen $C = 0{,}01$ sin ponderar clases. El AUC-PR casi no cambia con $C$ (con 130 000 "
           "filas la regularización importa poco para el desempeño, pero estabiliza los coeficientes ante la colinealidad). Ponderar clases "
           "(`balanced`) baja el AUC-PR. El AUC-ROC de entrenamiento (≈ 0,901) y de validación (≈ 0,899) son casi iguales: no hay "
           "sobreajuste. El desempeño cambia más entre años (AUC-PR de 0,56 en 2021 a 0,78 en 2019) que entre modelos."),

        md("## Detalle: el escalado de Platt de la SVM\n\n"
           "La SVM produce un margen $s = \\mathbf w^\\top \\mathbf x + b$, no una probabilidad. Con la validación 2022 se ajusta una "
           "regresión logística de una variable, $\\hat p = \\sigma(a\\,s + c)$ (Platt, 1999). Se hace con 2022 y no con el entrenamiento "
           "para no reutilizar los mismos datos que definieron el hiperplano."),
        code('''a, c_ = MOD["platt"]  # Parámetros de Platt
print(f"p = σ({a:.3f} · margen + {c_:.3f})")  # Transformación
for s in (-2, -1, 0, 1):  # Ejemplos
    print(f"margen {s:+d} → probabilidad {1 / (1 + np.exp(-(a * s + c_))):.3f}")  # Probabilidad'''),
        md("**Interpretación.** Un margen 0 (justo en el hiperplano) equivale a una probabilidad de ≈ 0,57; la pendiente alta ($a \\approx 3{,}5$) "
           "indica que el margen ordena bien: pequeñas diferencias de margen cambian mucho la probabilidad."),

        md("## Resultados en la prueba 2023–2024\n\n"
           ":::{admonition} Detalle: intervalos de confianza por bootstrap de conglomerados\n"
           "En cada una de las 1 000 réplicas se eligen 376 municipios **con reemplazo**, se juntan todas sus semanas y se recalcula la "
           "métrica. El intervalo es el de los percentiles 2,5 y 97,5. Remuestrear municipios, no filas, respeta la dependencia entre las "
           "semanas de un mismo municipio y da intervalos honestos.\n"
           ":::"),
        code('''tabla = pd.concat({n: r.apply(lambda f: f"{f['valor']:.3f} [{f['IC 2,5 %']:.3f}; {f['IC 97,5 %']:.3f}]", axis=1) for n, r in MOD["ic"].items()}, axis=1).T  # Valor [IC]
tabla'''),
        md("**Interpretación.** Los dos modelos base superan a todas las líneas base en AUC-ROC (0,886–0,887 frente a 0,859 de la mejor), "
           "AUC-PR (0,81 frente a 0,73), F1 y MCC, con intervalos de unas ± 0,01. La exactitud no los distingue de la persistencia simple "
           "(≈ 0,82–0,83), y el Dummy ya llega a 0,70: por eso no se usa para decidir. El ingenuo estacional es casi inútil (AUC 0,54)."),
        code('''MOD["diferencias"].round(4)  # Diferencias pareadas'''),
        md("**Interpretación.** Frente a la persistencia de 8 semanas, ambos modelos ganan ≈ 0,026–0,027 de AUC-ROC y ≈ 0,08 de AUC-PR, con "
           "intervalos que excluyen el 0. Entre ellos, la SVM gana 0,001 de AUC-ROC ($p \\approx 0{,}02$): significativo por el tamaño de "
           "muestra, irrelevante en la práctica; en AUC-PR no hay diferencia. Son, en la práctica, el mismo modelo."),
        code('''for n in ["Regresión logística", "SVM lineal"]:  # Matriz de confusión de cada modelo
    cm, met = metricas_umbral(n, MOD["umbrales"][n])  # Con su umbral de 2022
    mostrar(fg.fig_matriz(cm.tolist(), MOD["umbrales"][n]))  # Matriz
    print(n, {k: round(v, 3) for k, v in met.items()})  # Métricas'''),
        md("**Interpretación.** Con el umbral de máximo F1 en 2022, la regresión logística anticipa ≈ 74 % de los brotes con precisión ≈ 70 % "
           "(unas 121 alertas por semana en todo el país). La SVM tiene un umbral distinto (sus probabilidades vienen de Platt) pero un "
           "balance muy parecido. En el dashboard el umbral se mueve con un control deslizante."),
        code('''mostrar(fg.fig_curvas(curvas(["Regresión logística", "SVM lineal", "Persistencia (8 sem.)", "Persistencia", "Ingenuo estacional", "Dummy (prior)"]), MOD["y_te"].mean()))'''),
        md("**Interpretación.** Las curvas de los dos modelos base se superponen y dominan a las líneas base en casi todo el rango. La "
           "persistencia simple es un solo punto en la ROC (es binaria)."),

        md("## Diagnóstico"),
        code('''for n in ["Regresión logística", "SVM lineal"]:  # Calibración y brecha
    dg = MOD["diagnosticos"][n]  # Diagnóstico
    display(Markdown(f"**{n}**"))  # Título
    mostrar(fg.fig_calibracion(dg["calibracion"]))  # Calibración
    display(dg["brecha"].round(4))  # Mismas métricas en los tres conjuntos'''),
        md("**Interpretación.** En 2022 el error de calibración es pequeño (ECE ≈ 0,03); en 2023–2024 crece (ECE 0,04–0,05) y los modelos "
           "subestiman, porque la prevalencia triplica la del entrenamiento (*prior shift*). La brecha de AUC-ROC entre entrenamiento y "
           "prueba es de ≈ 0,02: no hay sobreajuste; las diferencias en AUC-PR, precisión y Brier se deben a la prevalencia.\n\n"
           ":::{admonition} Detalle: error de calibración esperado\n"
           "$\\text{ECE} = \\sum_{b=1}^{10} \\frac{n_b}{n}\\,\\big|\\bar y_b - \\bar p_b\\big|$ sobre deciles de probabilidad: el promedio, ponderado "
           "por tamaño, de la distancia entre lo predicho y lo observado.\n"
           ":::"),
        code('''for n in ["Regresión logística", "SVM lineal"]:  # Residuos
    dg = MOD["diagnosticos"][n]  # Diagnóstico
    mostrar(fg.fig_acf_residuos(dg["acf"]))  # ACF de residuos
    print(n, "· ACF rezagos 1–4:", np.round(dg["acf"][:4], 3), "· Moran del residuo medio:", round(dg["moran"]["I"], 3), "p =", dg["moran"]["p"])  # Resumen
mostrar(fg.fig_residuo_mapa(M, MOD["diagnosticos"]["Regresión logística"]["residuo_mun"]))  # Mapa de residuos'''),
        md("**Interpretación.** Los residuos tienen autocorrelación temporal en los rezagos 1–3 (≈ 0,34, 0,26 y 0,19; desaparece en el rezago "
           "4) y espacial (I de Moran ≈ 0,16–0,17, $p = 0{,}001$). Hay información estructural sin capturar: los episodios de brote duran "
           "varias semanas y se propagan entre vecinos. Es la motivación para agregar rezagos espaciales y modelos no lineales."),
        code('''mostrar(fg.fig_curva_aprendizaje(MOD["curva"], "AUC-ROC"))  # Curva de aprendizaje
MOD["curva"].round(4)'''),
        md("**Interpretación.** Con el 10 % de los municipios (13 000 filas) el AUC de validación ya es ≈ 0,90: la curva es plana y las "
           "curvas de entrenamiento y validación convergen. Más datos no mejorarían un modelo lineal; el límite es de sesgo."),
        code('''MOD["bloques"].round(4)  # Validación por bloques de departamentos (zona de separación de 50 km)'''),
        md("**Interpretación.** En departamentos no vistos el AUC-ROC es 0,88–0,93 (mismo periodo) y 0,86–0,91 (zonas nuevas y futuro): los "
           "modelos generalizan a zonas nuevas porque sus predictoras son relativas a cada municipio. Los dos modelos se comportan igual en "
           "cada bloque."),

        md("## Interpretación de los coeficientes"),
        code('''mostrar(fg.fig_coeficientes(MOD["coef_lr"]))  # Regresión logística
coef = MOD["coef_lr"].assign(odds_ratio=lambda t: np.exp(t["coef"]))  # Razón de odds
coef.reindex(coef["coef"].abs().sort_values(ascending=False).index).round(3)  # Tabla completa'''),
        md("**Interpretación.** La proporción reciente de semanas en brote (odds × 1,83 por desviación estándar), el log de casos de la semana "
           "anterior (× 1,54) y los casos frente al umbral objetivo (× 1,54) son los factores de riesgo más fuertes. Las medias largas de "
           "casos tienen signo negativo condicional a las cortas: lo que anticipa un brote es la **aceleración** reciente. El ONI multiplica "
           "las odds por ≈ 1,26. Los coeficientes de temperatura incluyen el 0 (el 64 % está imputada) y la lluvia tiene signos mezclados "
           "entre ventanas: efectos no lineales que un modelo lineal no capta."),
        code('''mostrar(fg.fig_pesos_svm(MOD["coef_svm"]))  # SVM lineal
lr = MOD["coef_lr"].set_index("variable")["coef"]  # Coeficientes logísticos
print(f"Correlación entre los pesos de la SVM y los coeficientes logísticos: {np.corrcoef(MOD['coef_svm'][lr.index], lr)[0, 1]:.2f}")  # Similitud
pd.DataFrame({"SVM": MOD["coef_svm"], "logística": lr}).sort_values("SVM", key=np.abs, ascending=False).head(10).round(3)'''),
        md("**Interpretación.** Los pesos de la SVM se parecen a los coeficientes logísticos (correlación ≈ 0,87) y la variable más pesada "
           "es la misma. La diferencia más visible está en la temperatura: la SVM reparte su efecto entre las ventanas de 8 y 12 semanas con "
           "signos opuestos, un síntoma de la colinealidad entre medias móviles. Las escalas no son comparables (la bisagra no mide log-odds)."),

        md("## Simulador\n\n"
           "El simulador del dashboard toma la semana con más casos de cada municipio en 2023–2024, cambia la temperatura (suma $\\Delta$ a "
           "todas sus ventanas y a su anomalía), la lluvia (la multiplica) y el ONI, y recalcula $\\hat p = \\sigma(\\boldsymbol\\beta^\\top \\mathbf z + \\beta_0)$ "
           "con la estandarización y los coeficientes del Pipeline logístico."),
        code('''from tablero.datos import simular  # Misma función del dashboard
filas = MOD["simulador"]["filas"]  # Escenarios
probs = np.array([simular(f) for _, f in filas.iterrows()])  # Probabilidad observada de cada escenario
cali = filas.iloc[int(np.argmin(np.abs(probs - MOD["simulador"]["umbral"])))]  # Municipio más cercano al umbral de alerta
print(f"Escenario: {cali['municipio']} ({cali['departamento']}), semana del {cali['week_start'].date()} · Cali en su pico: {simular(filas[filas['COD_MUN_N'] == 76001].iloc[0]):.3f}")
pd.DataFrame({esc: {"probabilidad": simular(cali, *args)} for esc, args in {
    "clima observado": (0, 1, None), "+2 °C": (2, 1, None), "lluvia × 1,5": (0, 1.5, None), "El Niño fuerte (ONI 2)": (0, 1, 2.0),
    "La Niña fuerte (ONI −1,5)": (0, 1, -1.5)}.items()}).T.round(3)'''),
        md("**Interpretación.** En una semana epidémica como el pico de Cali la probabilidad está saturada (≈ 0,995) y el clima no la "
           "mueve. Cerca del umbral sí importa: en Cáqueza (probabilidad 0,241, justo debajo del umbral 0,244) un El Niño fuerte la sube a "
           "0,258 y activa la alerta, y La Niña la reduce casi a la mitad. En cambio, +2 °C o 50 % más lluvia apenas la mueven (la "
           "temperatura incluso la baja un poco), reflejo de coeficientes climáticos pequeños y de signos mezclados. El simulador muestra una asociación aprendida, no un efecto causal.\n\n"
           "## Nota crítica\n\n"
           "La exactitud en la prueba (≈ 0,83) está en el rango 80–90 % que la guía pide revisar: (1) la auditoría no encontró fuga; (2) el "
           "Dummy acierta 0,70 y la persistencia 0,82, así que la exactitud no mide el mérito del modelo; (3) la validación respeta el tiempo y "
           "el espacio; y (4) el problema no es trivial: la mejor regla simple llega a AUC-PR 0,73 y los modelos a 0,81, con residuos que aún "
           "tienen estructura. El desempeño alto en exactitud se explica por el desbalance y la persistencia natural de las epidemias."),
    ]
    guardar("4_modelos_base", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
