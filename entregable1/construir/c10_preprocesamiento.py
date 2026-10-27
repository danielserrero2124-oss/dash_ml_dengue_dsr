"""Cuaderno 10: preprocesamiento (sección 2.9 de la guía)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas del cuaderno
    c = [
        md("# Preprocesamiento\n\n"
           "Cada decisión se rastrea a un hallazgo del EDA. Hay dos niveles:\n\n"
           "1. **Ingeniería de características** (`src/datos.py`): rezagos, medias móviles, anomalías y componentes cíclicos. "
           "Ninguna se *ajusta* con datos de otras filas, salvo la normal climática, que se estima solo con 2010–2021. Todas "
           "usan semanas ≤ $t$.\n"
           "2. **Pipeline de scikit-learn**: imputación y estandarización, ajustadas **solo con el entrenamiento** (dentro de cada "
           "pliegue de la validación cruzada, el Pipeline se reajusta con la parte de entrenamiento del pliegue)."),
        code(PREAMBULO + '''
df = d.panel()  # Panel completo
mod = d.modelado(df)  # Filas aptas para modelar
tr, va, te = d.particion(mod)  # Particiones purgadas
print(f"Entrenamiento {len(tr):,} · validación {len(va):,} · prueba {len(te):,} filas · {len(cf.FEATURES)} predictoras")  # Tamaños'''),

        md("## Decisiones y hallazgos que las motivan"),
        code('''decisiones = pd.DataFrame([  # (paso, decisión, hallazgo del EDA, capítulo)
    ("Partición", "Cronológica 2015–21 / 2022 / 2023–24 con purga de 4 semanas", "Autocorrelación temporal; objetivo en t+4", "Objetivo, Temporal"),
    ("Validación", "Ventana creciente por años (2018–2021) con purga", "Prevalencia y régimen cambian por año", "Objetivo, Temporal"),
    ("Imputación", "Se conserva temp_missing (método del indicador); SimpleImputer(mediana) como protección", "Faltantes MAR estructurales por municipio", "Base de datos"),
    ("Transformación", "log(1 + x) de casos, flujo y población", "Asimetría > 14 y curtosis en cientos", "Univariado"),
    ("Escalamiento", "StandardScaler dentro del Pipeline", "Escalas muy distintas; penalización L2", "Univariado, Bivariado"),
    ("Codificación categórica", "Semana → seno y coseno; departamento y región no se codifican", "Estacionalidad débil y cíclica; departamentos raros", "Univariado, Temporal"),
    ("Atípicos", "No se eliminan; se comprimen con el logaritmo", "Atípicos = epidemias reales con mucho más brote", "Base de datos, Multivariado"),
    ("Rezagos", "log casos en t, t−1, t−2, t−3", "PACF concentrada en los primeros rezagos", "Temporal"),
    ("Ventanas móviles", "Medias de 4, 8 y 12 semanas de casos y clima", "ACF persistente; clima actúa con 8–12 semanas", "Temporal"),
    ("Anomalías climáticas", "Desviación respecto a la normal mensual del municipio (2010–2021)", "El clima difiere mucho entre regiones (ε² ≈ 0,41)", "Bivariado, Espacial"),
    ("Canal endémico", "Casos frente al umbral objetivo (ratio_umbral) y umbral objetivo", "Brote se define relativo a la historia propia", "Objetivo"),
    ("El Niño", "ONI actual y rezagado 8 semanas", "Deriva fuerte del ONI; efecto rezagado", "Bivariado, Temporal"),
    ("Regularización", "L2 con C elegido por validación cruzada", "VIF ≫ 10 entre medias móviles", "Bivariado"),
    ("Desbalance", "class_weight None vs balanced en la malla; sin SMOTE", "≈ 10 % de brotes; prevalencia cambia por año", "Objetivo"),
    ("Variables espaciales", "No en el modelo base; propuestas para los siguientes entregables", "Moran ≈ 0,31; Knox significativo", "Espacial, Espacio-temporal"),
    ("Coordenadas", "Latitud y longitud no se usan como predictoras", "Focos que se desplazan; generalizar a zonas nuevas", "Espacio-temporal"),
], columns=["paso", "decisión", "hallazgo del EDA", "capítulo"])
decisiones'''),

        md("## Las 34 variables predictoras"),
        code('''familia = {**{v: "Historia de casos" for v in cf.CASOS}, **{v: "Clima y ambiente" for v in cf.CLIMA}, **{v: "Contexto" for v in cf.CONTEXTO}}  # Familia
formula = {  # Definición de cada variable (todas con semanas ≤ t)
    "lc_l0": "log(1 + casos_t)", "lc_l1": "log(1 + casos_{t−1})", "lc_l2": "log(1 + casos_{t−2})", "lc_l3": "log(1 + casos_{t−3})",
    "lc_m4": "media de log casos en t−3…t", "lc_m8": "media en t−7…t", "lc_m12": "media en t−11…t",
    "brote_l0": "brote en t (canal endémico)", "brote_m8": "proporción de semanas en brote en t−7…t",
    "ratio_umbral": "log(1 + casos_t) − log(1 + u_{t+4})", "l_umbral_obj": "log(1 + u_{t+4}) (años anteriores)", "incid": "casos_t / población × 10⁵",
    "temp_anom_m8": "media 8 sem. de (temp − normal del mes)", "prec_anom_m8": "media 8 sem. de (lluvia − normal)", "ndvi_anom_m8": "media 8 sem. de (NDVI − normal)",
    "oni_anom": "ONI de la semana", "oni_l8": "ONI de t−8", "temp_missing": "1 si la temperatura de t es imputada",
    "sem_sin": "sen(2π·semana/52)", "sem_cos": "cos(2π·semana/52)", "lpob": "log(población)", "flujo_m4": "log(1 + media 4 sem. del flujo)",
}
for v in ("temp_mean", "prec_total", "ndvi_mean"):  # Medias móviles del clima
    for w in (1, 4, 8, 12):  # Ventanas
        formula[f"{v}_m{w}"] = f"media de {v} en t−{w - 1}…t" if w > 1 else f"{v} de la semana t"
pd.DataFrame({"familia": familia, "definición": formula}).loc[cf.FEATURES]'''),

        md("## El Pipeline"),
        code('''from sklearn import set_config  # Diagrama del Pipeline
set_config(display="diagram")  # Muestra el Pipeline como diagrama
pipe = mb.pipeline_base()  # SimpleImputer → StandardScaler → LogisticRegression
pipe'''),
        code('''prep = mb.pipeline_base()[:-1].fit(tr[cf.FEATURES])  # Solo el preprocesamiento, ajustado con el entrenamiento
Xtr = prep.transform(tr[cf.FEATURES])  # Entrenamiento transformado
Xva = prep.transform(va[cf.FEATURES])  # Validación transformada con los parámetros del entrenamiento
chequeo = pd.DataFrame({"media entrenamiento": Xtr.mean(axis=0), "desv. entrenamiento": Xtr.std(axis=0),  # ≈ 0 y 1
                        "media validación": Xva.mean(axis=0), "desv. validación": Xva.std(axis=0)}, index=cf.FEATURES)  # No exactamente 0 y 1
print(f"Nulos tras el preprocesamiento: {int(np.isnan(Xtr).sum())} · medianas de imputación aprendidas solo con {len(tr):,} filas de entrenamiento")  # Verificación
chequeo.round(3).head(12)'''),
        md("**Interpretación.** En el entrenamiento, cada variable queda con media 0 y desviación 1; en la validación no "
           "exactamente, porque se transforma con la media y la desviación **del entrenamiento**. Esa pequeña diferencia es la "
           "prueba de que el preprocesamiento no vio la validación. No hay nulos: el `SimpleImputer` solo actuaría si en "
           "producción llegara una semana sin dato. La normal climática de las anomalías se estimó con 2010–2021, así que la "
           "prueba tampoco influye en ella."),
    ]
    guardar("10_preprocesamiento", c)  # Escribe el cuaderno


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
