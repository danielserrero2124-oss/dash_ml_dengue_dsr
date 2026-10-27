"""Cuaderno 2: diseño experimental — pipeline combinatorio, espacios de búsqueda, validación anidada y optimizadores."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Diseño experimental: 140 modelos con validación anidada\n\n"
           "## El pipeline combinatorio\n\n"
           "Cada modelo base se combina con cada técnica de balanceo (solo en clasificación) y cada combinación se ajusta con cada "
           "método de optimización. El total es un **producto**, no una suma:\n\n"
           "$$\\underbrace{7}_{\\text{modelos}} \\times \\underbrace{4}_{\\text{balanceo}} \\times \\underbrace{4}_{\\text{optimizadores}} = 112 \\text{ (clasificación)}, \\qquad "
           "7 \\times 4 = 28 \\text{ (regresión)}, \\qquad 112 + 28 = 140.$$\n\n"
           "Como cada combinación se evalúa en 3 pliegues externos, el experimento tiene **420 tareas** de optimización, más 140 "
           "reajustes finales. Todo se programa con bucles sobre funciones parametrizadas (modelo → balanceo → optimizador) y se ejecuta "
           "en paralelo con `joblib` (10 procesos)."),
        code(PREAMBULO + '''
pd.DataFrame({"Etapa 1 · modelos (clasificación)": mo.MODELOS_CLF, "Etapa 1 · modelos (regresión)": mo.MODELOS_REG,
              "Etapa 2 · balanceo": mo.BALANCEOS + ["", "", ""], "Etapa 3 · optimizadores": list(op.METODOS) + ["", "", ""]})'''),

        md("## Espacios de búsqueda\n\n"
           "Los rangos se eligieron por la escala natural de cada hiperparámetro y por lo aprendido en el Primer Entregable; los que "
           "cambian por órdenes de magnitud ($C$, $\\alpha$, tasa de aprendizaje, `var_smoothing`, vecinos, mínimo de filas por hoja) se "
           "exploran en **escala logarítmica**."),
        code('''filas = []  # Tabla de espacios
for tarea, espacios in [("clasificación", mo.ESPACIOS_CLF), ("regresión", mo.ESPACIOS_REG)]:  # Tareas
    for modelo, esp in espacios.items():  # Modelos
        for h, spec in esp.items():  # Hiperparámetros
            rango = ", ".join(map(str, spec[1])) if spec[0] == "cat" else f"[{spec[1]:g}, {spec[2]:g}]" + (" (log)" if spec[3] else "")  # Rango legible
            filas.append({"tarea": tarea, "modelo": modelo, "hiperparámetro": h, "tipo": spec[0], "rango": rango})  # Fila
pd.DataFrame(filas)'''),
        md("**Justificación de los rangos.**\n\n"
           "| Modelo | Decisión | Razón |\n"
           "|---|---|---|\n"
           "| KNN | 5 a 301 vecinos (log), voto uniforme o por distancia | Con 20 000 filas y 10 % de brotes, pocos vecinos dan probabilidades ruidosas; el Primer Entregable eligió k ≈ 150 |\n"
           "| Naive Bayes | `var_smoothing` de 1e−12 a 1e−2 (log) | Es el único hiperparámetro: estabiliza varianzas casi nulas (variables binarias y con ceros) |\n"
           "| Regresión logística | C de 1e−4 a 10 (log), L1 o L2 | L1 selecciona variables entre bloques colineales; L2 reparte el peso |\n"
           "| Árbol de decisión | profundidad 2–20, 5–500 filas por hoja (log), Gini o entropía | Controla la varianza de un árbol único |\n"
           "| Random Forest | 50–150 árboles, profundidad 4–20, 5–200 por hoja (log), √p o 50 % de variables | Más árboles reducen varianza con rendimientos decrecientes; acotado por costo |\n"
           "| XGBoost | 50–400 árboles, profundidad 2–8, tasa 0,01–0,3 (log), submuestreo 0,5–1, peso mínimo 1–50 (log) | Árboles poco profundos y tasas bajas: el régimen habitual del boosting |\n"
           "| SVM / SVR | C de 1e−3 a 10 y γ de 1e−3 a 1 (log); ε de 0,01 a 0,5 en SVR | Núcleo RBF aproximado con Nyström (300 componentes) para evitar el costo cuadrático del núcleo exacto |\n"
           "| Ridge / Lasso | α de 1e−3 a 1e3 (Ridge) y de 1e−4 a 1 (Lasso), log | Lasso con α grande anula todas las variables: el rango se acota |"),

        md("## Balanceo de clases\n\n"
           "| Estrategia | Cómo actúa | Implementación |\n"
           "|---|---|---|\n"
           "| Sin balanceo | Línea base | — |\n"
           "| SMOTE | Interpola brotes sintéticos entre vecinos brote ($k = 5$) hasta igualar las clases | `imblearn` dentro del Pipeline |\n"
           "| ADASYN | Como SMOTE, pero más sintéticos donde un brote tiene más vecinos no brote | `imblearn` dentro del Pipeline |\n"
           "| class_weight balanced | Cada error de la clase $c$ pesa $n/(2n_c)$ | Parámetro nativo; `sample_weight` para Naive Bayes y XGBoost; corrección por la probabilidad a priori en KNN |\n\n"
           "KNN no acepta pesos de muestra. Su equivalente es dividir la proporción de vecinos de cada clase por su prevalencia y "
           "renormalizar, $\\tilde p_c \\propto p_c / \\pi_c$: la regla de Bayes con clases equiprobables."),
        code('''import inspect  # Código fuente
print(inspect.getsource(mo.construir))  # El Pipeline de cada combinación'''),

        md("## Validación cruzada anidada\n\n"
           "Reportar la métrica del mejor modelo sobre el mismo esquema que lo eligió es optimista (sesgo de selección). Por eso hay dos "
           "bucles, ambos **por años y con purga de 4 semanas** (el objetivo de la fila $t$ está en $t+4$):\n\n"
           "* **Bucle externo (3 pliegues):** cada año $a \\in \\{2019, 2020, 2021\\}$ se evalúa con un modelo entrenado solo con los años "
           "anteriores. Es la métrica que se reporta (media ± desviación de los 3 pliegues).\n"
           "* **Bucle interno (2 pliegues):** dentro del entrenamiento de cada pliegue externo, los dos últimos años validan; el optimizador "
           "solo ve el promedio de estos pliegues.\n\n"
           "**¿Por qué 3 y 2?** El tiempo no se puede barajar: los pliegues son años completos, y un pliegue externo necesita al menos 4 años "
           "previos (2 para entrenar y 2 para validar dentro). 2019–2021 cubre un año epidémico (2019), uno de pandemia (2020) y uno de baja "
           "transmisión (2021), así que la desviación entre pliegues refleja regímenes reales. Para que las 420 tareas quepan en el presupuesto, "
           "el entrenamiento externo se submuestrea a 20 000 filas estratificadas por año y clase; el año evaluado se usa **completo**."),
        code('''tr, va, te = dd.cargar()  # Periodos
filas = []  # Línea de tiempo de los pliegues
for a in c3.ANIOS_EXTERNOS:  # Pliegues externos
    ent, ev = dd.pliegue_externo(tr, a)  # Externo
    sub = dd.submuestra(ent, c3.N_EXTERNO)  # Submuestra
    for k, (i_tr, i_va) in enumerate(dd.pliegues_internos(sub), 1):  # Internos
        filas.append({"pliegue externo": a, "pliegue interno": k, "entrena (años)": f"{sub['ANO'].iloc[i_tr].min()}–{sub['ANO'].iloc[i_tr].max()}",
                      "filas internas de entrenamiento": len(i_tr), "valida (año)": int(sub["ANO"].iloc[i_va].iloc[0]), "filas internas de validación": len(i_va)})
    filas.append({"pliegue externo": a, "pliegue interno": "externo", "entrena (años)": f"{ent['ANO'].min()}–{a - 1}", "filas internas de entrenamiento": len(sub),
                  "valida (año)": a, "filas internas de validación": len(ev)})
pd.DataFrame(filas)'''),
        code('''fig = go.Figure()  # Diagrama de la validación anidada
y0 = 0  # Posición vertical
for a in c3.ANIOS_EXTERNOS:  # Pliegues externos
    for k, (ini_val) in enumerate([a - 2, a - 1]):  # Internos
        fig.add_bar(x=[ini_val - 2015], base=[2015], y=[f"externo {a} · interno {k + 1}"], orientation="h", marker_color=AZUL, showlegend=False)  # Entrena interno
        fig.add_bar(x=[1], base=[ini_val], y=[f"externo {a} · interno {k + 1}"], orientation="h", marker_color=AMBAR, showlegend=False)  # Valida interno
    fig.add_bar(x=[a - 2015], base=[2015], y=[f"externo {a}"], orientation="h", marker_color="#1e3a8a", showlegend=False)  # Entrena externo
    fig.add_bar(x=[1], base=[a], y=[f"externo {a}"], orientation="h", marker_color=ROJO, showlegend=False)  # Evalúa externo
fig.add_bar(x=[7], base=[2015], y=["reajuste final"], orientation="h", marker_color="#1e3a8a", showlegend=False)  # Final
fig.add_bar(x=[1], base=[2022], y=["reajuste final"], orientation="h", marker_color=VERDE, showlegend=False)  # Umbral y calibración
fig.add_bar(x=[2], base=[2023], y=["reajuste final"], orientation="h", marker_color=GRIS, showlegend=False)  # Prueba
fig.update_layout(barmode="overlay", title="Validación anidada: azul = entrena, ámbar = valida (interno), rojo = evalúa (externo), verde = 2022, gris = prueba",
                  xaxis=dict(dtick=1, title="año"), yaxis=dict(autorange="reversed"))
mostrar(fig, 520)  # Muestra'''),
        md("**Reajuste final.** Con los hiperparámetros elegidos en el pliegue externo más reciente (2021), cada combinación se ajusta "
           "sobre 2015–2021 (40 000 filas). La validación 2022 fija el umbral de decisión y la recalibración, y la prueba 2023–2024 se usa una "
           "sola vez para las métricas de los capítulos de resultados y para las pruebas estadísticas."),

        md("## Métrica de validación\n\n"
           "El criterio que maximizan los cuatro optimizadores es el **AUC-PR** (precisión promedio) en clasificación y el **−RMSE** en "
           "regresión. Un brote no anticipado (falso negativo) cuesta más que una falsa alarma, pero la capacidad de respuesta de una "
           "secretaría también es limitada: el AUC-PR resume el compromiso precisión–recall en todos los umbrales y, a diferencia del AUC-ROC, "
           "se concentra en la clase brote y no se infla con los verdaderos negativos (90 % de las filas). El umbral concreto se elige después "
           "maximizando F1 en 2022; con él se reportan Accuracy, Precision, Recall y F1."),

        md("## Los cuatro optimizadores (y uno adicional)\n\n"
           "| Método | Cómo propone el siguiente candidato | Exploración / explotación |\n"
           "|---|---|---|\n"
           "| **Grid Search** | Malla regular: la mayor que cabe en 20 evaluaciones (las dimensiones sin niveles quedan en el centro del rango) | Solo exploración, fija de antemano; sufre con muchas dimensiones |\n"
           "| **Random Search** | Muestreo independiente, uniforme o log-uniforme | Solo exploración; cubre mejor las dimensiones importantes que la malla (Bergstra y Bengio, 2012) |\n"
           "| **Bayesiana (Optuna, TPE)** | Modelo sustituto *Tree-structured Parzen Estimator*: estima $l(x) = p(x \\mid y \\text{ bueno})$ y $g(x) = p(x \\mid y \\text{ malo})$ con el cuantil γ de los mejores resultados, y propone el candidato que maximiza $l(x)/g(x)$ | La razón $l/g$ es equivalente a maximizar la **Mejora Esperada (EI)**; 5 evaluaciones aleatorias iniciales exploran y después explota, con exploración residual por los 24 candidatos muestreados de $l(x)$ |\n"
           "| **Genética (DEAP)** | Población de 6 configuraciones codificadas en $[0, 1]^d$; selección por **torneo** (3), **cruce** de mezcla (α = 0,5, p = 0,7), **mutación** gaussiana (σ = 0,15, p = 0,3); estrategia (μ + λ) con **elitismo** | El cruce explota combinaciones de buenos padres; la mutación y los inmigrantes mantienen la **diversidad**, que se registra por generación |\n"
           "| **Successive Halving** (complementario) | 27 candidatos con 1/9 de las filas, los mejores 9 con 1/3 y los mejores 3 con todas (factor 3) | Gasta poco en las configuraciones malas y reasigna el presupuesto a las competitivas |\n\n"
           "**Sin fuga de datos en ningún método.** Optuna y DEAP no integran validación cruzada: la función objetivo implementada a mano "
           "ajusta el **Pipeline completo** (imputación, estandarización, remuestreo y modelo) dentro de cada pliegue interno, con los datos de "
           "entrenamiento del pliegue."),
        code('''print(inspect.getsource(op.bayesiana))  # Optuna con TPE
print(inspect.getsource(op.genetica))  # DEAP'''),
        code('''print(inspect.getsource(ex.ejecutar_tarea))  # Una tarea de la validación anidada'''),

        md("## Tabla maestra de experimentos\n\n"
           "Cada una de las 420 tareas (140 combinaciones × 3 pliegues externos) se registra con su modelo, balanceo, optimizador, semilla, "
           "hiperparámetros finales, métricas del bucle externo, número de evaluaciones y tiempos. Se guarda en `resultados/tabla_maestra.csv`."),
        code('''tabla = pd.read_csv(c3.RESULTADOS / "tabla_maestra.csv")  # Tabla maestra
print(f"Registros: {len(tabla)} · combinaciones: {tabla.groupby(['tarea', 'modelo', 'balanceo', 'optimizador']).ngroups} · "
      f"evaluaciones de hiperparámetros: {int(tabla['evaluaciones'].sum()):,} · ajustes de modelos en el bucle interno: {int(tabla['evaluaciones'].sum()) * c3.PLIEGUES_INTERNOS:,}")
tabla.drop(columns=["mejores_params"]).head(8)'''),
        md("**Reproducibilidad.** Una sola semilla global (`SEMILLA = 42`, `e3/config.py`) se propaga a NumPy, scikit-learn, XGBoost, "
           "imbalanced-learn, Optuna (`TPESampler(seed)`) y DEAP (`random.seed`). Las versiones exactas están en `entregable3/requirements.txt`. "
           "El código está separado en módulos (datos, modelos, optimizadores, experimento, evaluación, estadística, complejidad) con "
           "*docstrings* estilo NumPy en las funciones compartidas."),
    ]
    guardar("2_diseno", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
