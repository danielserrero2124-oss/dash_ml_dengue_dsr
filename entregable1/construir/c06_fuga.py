"""Cuaderno 6: auditoría de fuga de datos (sección 2.5 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Auditoría de fuga de datos\n\n"
           "Una variable tiene fuga si contiene información que no estaría disponible en la semana $t$, cuando se emite la "
           "alerta. Se revisa la disponibilidad de cada familia de variables, el desempeño univariado de cada una, los "
           "identificadores y el solapamiento entre particiones, y se demuestra cuánto se inflaría el desempeño con fuga."),
        code(PREAMBULO + '''
from sklearn.metrics import roc_auc_score  # AUC univariado
from sklearn.model_selection import KFold, cross_val_score  # Validación aleatoria (para la demostración)
df = d.panel()  # Panel completo
mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Particiones purgadas
print(f"Entrenamiento: {len(tr):,} filas")  # Tamaño'''),

        md("## Disponibilidad de cada variable en la semana de la predicción"),
        code('''familias = pd.DataFrame([  # Familia, variables y justificación de disponibilidad
    ("Historia de casos", "lc_l0–lc_l3, lc_m4–lc_m12, incid", "Sí", "Casos notificados en las semanas t, t−1, …; se conocen al cierre de la semana t."),
    ("Brote actual", "brote_l0, brote_m8", "Sí", "Se calcula con los casos hasta t y el umbral de t, que usa años anteriores."),
    ("Umbral objetivo", "l_umbral_obj, ratio_umbral", "Sí", "u(t+4) es el Q3 de semanas t+4−52k±2 (k ≥ 1): la más reciente es t−46."),
    ("Clima", "temp/prec/ndvi_mean_m1–m12", "Sí", "Medias móviles de semanas ≤ t (rolling hacia atrás)."),
    ("Anomalías", "temp/prec/ndvi_anom_m8", "Sí", "La normal climática se estima solo con 2010–2021 (entrenamiento)."),
    ("El Niño", "oni_anom, oni_l8", "Sí*", "El ONI mensual se publica con unos días de rezago; oni_l8 (8 semanas antes) no tiene ningún riesgo."),
    ("Calendario", "sem_sin, sem_cos", "Sí", "La semana del año se conoce de antemano."),
    ("Contexto", "lpob, flujo_m4, temp_missing", "Sí", "Población anual proyectada; flujo de semanas ≤ t; bandera de imputación."),
    ("Objetivo y futuro", "y, casos_obj (casos de t+4)", "No", "Son el objetivo o información de t+4: nunca se usan como predictoras."),
    ("Identificadores", "COD_MUN_N, ANO, time_idx, departamento", "Sí, pero se excluyen", "Permitirían memorizar lugares o épocas y no generalizan."),
], columns=["familia", "variables", "¿disponible en t?", "justificación"])
familias'''),
        code('''# Verificación numérica del umbral objetivo: semanas históricas que usa u(t+4)
desfases = [cf.HORIZONTE - 52 * k + j for k in range(1, cf.ANIOS_HISTORIA + 1) for j in range(-cf.VENTANA_SEMANAS, cf.VENTANA_SEMANAS + 1)]  # Desfases respecto a t
print(f"El umbral objetivo usa las semanas t{min(desfases):+d} a t{max(desfases):+d}: todas anteriores a t → sin fuga")  # Rango'''),
        md(":::{note}\n"
           "El ONI de un mes se publica a inicios del mes siguiente, así que en las primeras semanas de cada mes el valor "
           "contemporáneo `oni_anom` podría no estar publicado todavía. Por eso se conserva también `oni_l8`, que no tiene "
           "ningún riesgo, y la variable contemporánea queda **en observación**: en la auditoría de desempeño no se comporta "
           "como un proxy del objetivo (AUC univariado ≈ 0,65).\n"
           ":::"),

        md("## Desempeño univariado: señal de alerta si se acerca a 1"),
        code('''filas = []  # AUC univariado de cada candidata
candidatas = {**{v: tr[v] for v in cf.FEATURES},  # Las 34 predictoras
              "casos_obj (casos de t+4)": np.log1p(tr["casos_obj"]),  # Información del futuro (para contraste)
              "COD_MUN_N (identificador)": tr["COD_MUN_N"], "time_idx (identificador)": tr["time_idx"], "ANO (identificador)": tr["ANO"]}  # Identificadores
for nombre, x in candidatas.items():  # Cada candidata
    a = roc_auc_score(tr[cf.TARGET], x)  # AUC univariado
    filas.append({"variable": nombre, "AUC univariado": max(a, 1 - a)})  # Simétrico (la dirección no importa)
auc_uni = pd.DataFrame(filas).set_index("variable").sort_values("AUC univariado", ascending=False)  # Tabla
auc_uni["alerta (> 0,9)"] = np.where(auc_uni["AUC univariado"] > 0.9, "⚠ fuga probable", "")  # Bandera
auc_uni.round(3).head(15)'''),
        code('''orden = auc_uni.sort_values("AUC univariado").index  # Orden
colores = [ROJO if "obj" in v else GRIS if "identificador" in v else AZUL for v in orden]  # Rojo: futuro; gris: identificadores
fig = go.Figure(go.Bar(x=auc_uni.loc[orden, "AUC univariado"], y=orden, orientation="h", marker_color=colores))  # Barras
fig.add_vline(x=0.9, line_dash="dash", line_color=ROJO, annotation_text="alerta 0,9")  # Umbral de alerta
fig.update_layout(title="AUC univariado de cada variable (rojo: información del futuro; gris: identificadores)", xaxis_range=[0.45, 1])  # Formato
mostrar(fig, 820)  # Muestra'''),
        md("**Interpretación.** Ninguna de las 34 predictoras supera 0,9: la más fuerte es la media de 4 semanas del log de "
           "casos (≈ 0,88), coherente con la persistencia de las epidemias y no con una fuga. Los casos de la semana objetivo, "
           "en cambio, alcanzan ≈ 0,96: es exactamente el patrón que delata una fuga, y por eso esa columna se descarta. Los "
           "identificadores tienen AUC moderados (los brotes se concentran en ciertos años y municipios); no se usan porque el "
           "modelo aprendería *cuándo* y *dónde* hubo brotes en el entrenamiento en lugar de *por qué*."),

        md("## Duplicados y entidades repetidas entre particiones"),
        code('''clave = lambda x: set(zip(x["COD_MUN_N"], x["time_idx"]))  # Clave única (municipio, semana)
ultima_obj_tr = (tr["time_idx"] + cf.HORIZONTE).max()  # Última semana objetivo del entrenamiento
pd.Series({
    "Filas compartidas entrenamiento ∩ validación": len(clave(tr) & clave(va)),
    "Filas compartidas entrenamiento ∩ prueba": len(clave(tr) & clave(te)),
    "Filas compartidas validación ∩ prueba": len(clave(va) & clave(te)),
    "Última semana objetivo del entrenamiento (time_idx)": int(ultima_obj_tr),
    "Primera semana t de la validación (time_idx)": int(va["time_idx"].min()),
    "Última semana objetivo de la validación (time_idx)": int((va["time_idx"] + cf.HORIZONTE).max()),
    "Primera semana t de la prueba (time_idx)": int(te["time_idx"].min()),
    "Filas purgadas en las fronteras": int((df["periodo"] == "purgada").sum()),
    "Municipios presentes en las tres particiones": len(set(tr["COD_MUN_N"]) & set(va["COD_MUN_N"]) & set(te["COD_MUN_N"])),
}, name="valor").to_frame()'''),
        md("**Interpretación.** No hay filas compartidas y ninguna etiqueta de un periodo se usa en el anterior: la última "
           "semana objetivo del entrenamiento es anterior a la primera semana de la validación, y lo mismo entre validación y "
           "prueba (la purga quita 4 semanas en cada frontera). Los 376 municipios aparecen en las tres particiones **por "
           "diseño**: el objetivo principal es predecir el futuro de municipios conocidos. Como las semanas de un municipio "
           "están correlacionadas, esto exige (1) que la partición sea temporal y (2) evaluar aparte la generalización a "
           "**zonas nuevas** con bloques espaciales (capítulo del modelo base)."),

        md("## Demostración: cuánto infla la fuga el desempeño\n\n"
           "Se entrena la misma regresión logística del modelo base en dos escenarios incorrectos y en el correcto."),
        code('''pipe = mb.pipeline_base().set_params(clf__C=0.01)  # Mismo Pipeline del modelo base
temporal = cross_val_score(pipe, tr[cf.FEATURES], tr[cf.TARGET], groups=mb.grupos(tr), cv=mb.ValidacionAnualPurgada(), scoring="roc_auc").mean()  # Correcto
aleatorio = cross_val_score(pipe, tr[cf.FEATURES], tr[cf.TARGET], cv=KFold(5, shuffle=True, random_state=cf.SEMILLA), scoring="roc_auc").mean()  # Semanas mezcladas
con_futuro = tr.assign(lc_obj=np.log1p(tr["casos_obj"]))  # Agrega los casos de t+4
futuro = cross_val_score(pipe, con_futuro[cf.FEATURES + ["lc_obj"]], con_futuro[cf.TARGET], groups=mb.grupos(tr),  # Con información del futuro
                         cv=mb.ValidacionAnualPurgada(), scoring="roc_auc").mean()
pd.DataFrame({"AUC-ROC de validación cruzada": [temporal, aleatorio, futuro]},  # Comparación
             index=["Correcto: años hacia adelante con purga", "Fuga temporal: semanas mezcladas al azar",
                    "Fuga directa: casos de la semana objetivo"]).round(4)'''),
        md("**Interpretación.** Con la validación correcta el AUC ronda 0,90. Mezclar semanas al azar cambia poco en este "
           "modelo lineal (la regresión logística no puede memorizar interacciones finas entre semanas vecinas), pero la fuga directa lo lleva casi a 1: un modelo «perfecto» e inútil. La auditoría no "
           "solo revisa columnas; también comprueba que el esquema de validación respete el tiempo, porque con modelos más "
           "flexibles (los de los siguientes entregables) la mezcla aleatoria sí infla el desempeño.\n\n"
           "## Resultado de la auditoría\n\n"
           "| Decisión | Variables | Motivo |\n"
           "|---|---|---|\n"
           "| **Descartadas** | `casos_obj` y cualquier dato de $t+1$ a $t+4$ | Información del futuro (AUC ≈ 0,96) |\n"
           "| **Descartadas** | `COD_MUN_N`, `ANO`, `time_idx`, departamento | Identificadores: memorizan lugar y época |\n"
           "| **Descartadas** | Casos por edad y sexo | Redundantes (suman el total), no son fuga |\n"
           "| **En observación** | `oni_anom` | Posible rezago de publicación; se conserva `oni_l8` como respaldo |\n"
           "| **En observación** | `brote_m8`, `lc_m4`, `ratio_umbral` | Señal fuerte (AUC 0,81–0,88) pero legítima: semanas ≤ t |\n"
           "| **En observación** | `temp_missing` | Proxy del tamaño del municipio, no del objetivo |"),
    ]
    guardar("06_fuga", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
