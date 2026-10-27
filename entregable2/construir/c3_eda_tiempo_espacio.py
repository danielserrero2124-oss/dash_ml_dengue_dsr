"""Cuaderno 3: EDA II — componentes temporal, espacial y espacio-temporal y preprocesamiento."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# EDA II: tiempo, espacio y preprocesamiento\n\n"
           "Acompaña las sub-pestañas **Temporal, Espacial, Espacio-temporal y Preprocesamiento** del dashboard."),
        code(PREAMBULO + '''
t = A["temporal"]  # Componente temporal
pd.Series({"fuerza de la tendencia": t["fuerza"]["tendencia"], "fuerza estacional": t["fuerza"]["estacional"],
           "municipios con ADF significativo": t["mun_adf"], "municipios con KPSS significativo": t["mun_kpss"]}).to_frame("valor")'''),

        md("## Componente temporal"),
        code('''mostrar(fg.fig_serie(t["nacional"], t["mensual"], A["contexto"]["anual"].loc[:2021], "semana"))  # Serie semanal
mostrar(fg.fig_estacionalidad(t["calendario"]))  # Estacionalidad
mostrar(fg.fig_stl(t["stl"]))  # STL'''),
        md("**Interpretación.** La serie la dominan ondas de varios años (fuerza de la tendencia 0,66) y la estacionalidad anual es casi nula "
           "(0,02): la información útil está en el nivel reciente de la serie.\n\n"
           ":::{admonition} Detalle: fuerza de la tendencia y de la estacionalidad (Wang, Smith y Hyndman, 2006)\n"
           "Con la descomposición STL $y_t = T_t + S_t + R_t$: $F_T = \\max\\big(0, 1 - \\frac{\\text{Var}(R_t)}{\\text{Var}(T_t + R_t)}\\big)$ y "
           "$F_S = \\max\\big(0, 1 - \\frac{\\text{Var}(R_t)}{\\text{Var}(S_t + R_t)}\\big)$.\n"
           ":::"),
        code('''t["estacionariedad"]  # ADF (H0: raíz unitaria) y KPSS (H0: estacionaria)'''),
        md("**Interpretación.** En nivel, ADF no rechaza la raíz unitaria y KPSS tampoco rechaza la estacionariedad: una serie muy persistente "
           "en la que ambas pruebas pierden potencia. La primera diferencia es estacionaria. Por municipio, el 62 % rechaza la raíz unitaria y el "
           "49 % rechaza la estacionariedad."),
        code('''mostrar(fg.fig_acf(t["acf"], t["pacf"], t["n"]))  # ACF y PACF
mostrar(fg.fig_ccf(t["ccf"]))  # Correlación cruzada
t["ccf"].abs().agg(["idxmax", "max"]).T.rename(columns={"idxmax": "rezago de máxima |r|", "max": "máxima |r|"})'''),
        md("**Interpretación.** ACF persistente y PACF concentrada en los primeros rezagos: se usan rezagos de 1–3 semanas y medias de 4, 8 y "
           "12. El clima local llega con rezago (lluvia y NDVI 12–15 semanas) y correlaciones pequeñas; el ONI es la señal más fuerte."),
        code('''mostrar(fg.fig_regimenes(t["stl"], t["cortes"], t["nino"]))  # Regímenes
print("Puntos de cambio:", ", ".join(str(c.date()) for c in t["cortes"]))  # Fechas
display(t["deriva"])  # PSI
mostrar(fg.fig_deriva(t["concepto"]))  # Concept drift
t["concepto"]'''),
        md("**Interpretación.** Los regímenes siguen las epidemias y la caída de 2020 (pandemia). La deriva es baja para la historia de casos, "
           "moderada hacia 2022 en la temperatura anómala y la lluvia, y muy fuerte en el ONI; el concepto casos → brote es estable "
           "(AUC 0,83–0,89 cada año), el del clima no.\n\n"
           ":::{admonition} Detalle: índice de estabilidad poblacional y segmentación binaria\n"
           "$\\text{PSI} = \\sum_i (a_i - r_i)\\ln(a_i/r_i)$ sobre los deciles de la referencia (< 0,1 estable; > 0,25 deriva fuerte). Los "
           "puntos de cambio se buscan dividiendo recursivamente la serie donde más baja la suma de cuadrados, con segmentos de al menos 52 "
           "semanas y una penalización tipo BIC $3 \\cdot 2\\hat\\sigma^2 \\ln n$, con $\\hat\\sigma^2$ la varianza del residuo de STL.\n"
           ":::"),
        code('''mostrar(fg.fig_calor(t["calor"], t["calor_regiones"]))  # Panel municipio × semana'''),
        md("**Interpretación.** Ondas sincronizadas (franjas verticales) y focos locales: heterogeneidad entre municipios en nivel y en tiempo."),

        md("## Componente espacial"),
        code('''s = A["espacial"]  # Componente espacial
mapa = s["mapa"]  # Municipios
mostrar(fg.fig_cobertura(mapa, s["fuera"]))  # Cobertura
mostrar(fg.fig_mapa_valor(mapa, "log_incidencia", "log incidencia"))  # Incidencia
pd.Series(s["clark_evans"]).to_frame("Clark-Evans")'''),
        md("**Interpretación.** Las coordenadas DIVIPOLA (EPSG:4326) son válidas. El panel deja vacíos en la Orinoquía y la Amazonía (sesgo "
           "de muestreo espacial) y los municipios están agrupados ($R \\approx 0{,}77$)."),
        code('''mostrar(fg.fig_ripley(s["ripley"]))  # Ripley
mostrar(fg.fig_dbscan(mapa))  # DBSCAN
mapa["dbscan"].value_counts().head(8).to_frame("municipios")'''),
        md(":::{admonition} Detalle: índice de Clark-Evans y función K de Ripley\n"
           "$R = \\bar r_{\\text{obs}} / \\bar r_{\\text{esp}}$ con $\\bar r_{\\text{esp}} = 0{,}5/\\sqrt{n/A}$ (aleatoriedad completa); "
           "$K(d) = \\frac{A}{n(n-1)}\\sum_{i \\ne j}\\mathbb 1(d_{ij} \\le d)$ y $L(d) = \\sqrt{K(d)/\\pi}$. La envolvente se obtiene con 99 "
           "simulaciones de puntos uniformes en la misma envolvente convexa, así que el efecto de borde afecta igual a lo observado y a lo simulado.\n"
           ":::"),
        code('''display(s["sensibilidad"])  # Moran con distintas matrices de pesos
mostrar(fg.fig_moran(mapa, rezago_espacial(mapa["log_incidencia"])))  # Diagrama de Moran
pd.DataFrame(s["moran"]).T'''),
        md("**Interpretación.** La incidencia tiene autocorrelación espacial clara (I ≈ 0,31 con k = 6) y robusta a la matriz de pesos. La "
           "proporción de semanas en brote no (I ≈ 0,05, p ≈ 0,06): el canal endémico normaliza el nivel de cada zona.\n\n"
           ":::{admonition} Detalle: I de Moran, LISA y Getis-Ord\n"
           "$I = \\frac{n}{S_0}\\frac{\\mathbf z^\\top \\mathbf W \\mathbf z}{\\mathbf z^\\top \\mathbf z}$ con $E[I] = -1/(n-1)$; significancia por 999 "
           "permutaciones. LISA: $I_i = \\frac{z_i}{m_2}\\sum_j w_{ij} z_j$ con permutación condicional. Getis-Ord: "
           "$G_i^* = \\frac{\\sum_j w_{ij}x_j - \\bar x \\sum_j w_{ij}}{s\\sqrt{(n\\sum_j w_{ij}^2 - (\\sum_j w_{ij})^2)/(n-1)}}$, que ya es un puntaje z.\n"
           ":::"),
        code('''mostrar(fg.fig_lisa(mapa))  # LISA
mostrar(fg.fig_gi(mapa))  # Gi*
print(f"LISA significativos tras FDR: {s['fdr_lisa']}")  # Corrección
mapa.loc[mapa["LISA"] == "Alto-Alto", "departamento"].value_counts().to_frame("municipios Alto-Alto")'''),
        md("**Interpretación.** 76 municipios Alto-Alto (Meta, Tolima, Huila, Putumayo) y 53 significativos tras FDR: los focos no son un "
           "artefacto de las comparaciones múltiples."),
        code('''mostrar(fg.fig_vario(s["vario"], float(mapa["log_incidencia"].var())))  # Semivariograma
display(s["heterogeneidad"])  # Por región
pd.Series({"MAUP: correlación temperatura–incidencia (municipios)": s["maup"]["municipios"], "MAUP: (departamentos)": s["maup"]["departamentos"],
           "error mediano de la distancia euclidiana sobre grados": s["error_grados"][0]}).to_frame("valor")'''),
        md("**Interpretación.** Alcance ≈ 100 km con efecto pepita alto. La relación del clima con el brote cambia entre regiones, y al agregar "
           "a departamentos la correlación con la temperatura se duplica (MAUP): se trabaja a escala municipal con haversine."),

        md("## Componente espacio-temporal"),
        code('''st = A["espacio_temporal"]  # Espacio-temporal
for a in (2017, 2019):  # Un año sin brotes y uno epidémico
    mostrar(fg.fig_anual_mapa(st["anual"], a))  # Mapa del año
mostrar(fg.fig_focos(st["focos"]))  # Focos
st["focos"]["categoría"].value_counts().to_frame("municipios")'''),
        md("**Interpretación.** El patrón espacial cambia cada año y los focos se desplazan: 159 municipios son focos esporádicos y solo 8 "
           "persistentes.\n\n"
           ":::{admonition} Detalle: clasificación de focos\n"
           "Para cada municipio se calcula $G_i^*$ de la log incidencia de cada año (2015–2021) y la prueba de Mann-Kendall sobre esa serie: "
           "**persistente** si es foco ≥ 80 % de los años; **emergente** si es foco el último año con tendencia creciente ($p < 0{,}1$); "
           "**decreciente** si fue foco con tendencia decreciente; **esporádico** si fue foco sin tendencia.\n"
           ":::"),
        code('''mostrar(fg.fig_regiones(st["regiones"]))  # Series por región
display(st["corr_regiones"].round(2))  # Sincronía
st["knox"]  # Prueba de Knox'''),
        md("**Interpretación.** Las regiones están sincronizadas salvo el Pacífico. La prueba de Knox encuentra en todas las escalas más inicios "
           "de brote cercanos en espacio y tiempo de lo esperado (razón 1,1–1,3; p = 0,001).\n\n"
           ":::{admonition} Detalle: prueba de Knox\n"
           "Se cuentan los pares de inicios de brote a menos de $d_s$ km y $d_t$ semanas, y se compara con 999 permutaciones de los tiempos "
           "entre lugares (que rompen la interacción y conservan los patrones puramente espaciales y temporales).\n"
           ":::"),
        code('''mostrar(fg.fig_moran_semanal(st["moran_semanal"]))  # Moran semanal
mostrar(fg.fig_cruzada(st["cruzada"]))  # Vecindario → municipio
st["cruzada"]'''),
        md("**Interpretación.** La autocorrelación espacial de los casos es positiva casi todas las semanas y el vecindario en $t$ anticipa al "
           "municipio en $t+4$ (r ≈ 0,41), aunque menos que su propia historia (r ≈ 0,62). Consecuencia: variables de vecindad rezagadas "
           "para los siguientes modelos, y validación por zonas nuevas con zona de separación.\n\n"
           "## Preprocesamiento\n\n"
           "| Paso | Decisión | Hallazgo que la motiva |\n"
           "|---|---|---|\n"
           "| Partición | Cronológica con purga de 4 semanas | Autocorrelación; objetivo en $t+4$ |\n"
           "| Validación | Ventana creciente por años con purga | Prevalencia y régimen cambian por año |\n"
           "| Imputación | `temp_missing` + `SimpleImputer(mediana)` | Faltantes MAR estructurales |\n"
           "| Transformación | $\\log(1+x)$ de casos, flujo y población | Asimetría > 14 |\n"
           "| Escalamiento | `StandardScaler` en el Pipeline | Penalización L2 y escalas distintas |\n"
           "| Codificación | Seno y coseno de la semana; sin departamento | Estacionalidad cíclica; categorías raras |\n"
           "| Atípicos | Se conservan | Son epidemias reales |\n"
           "| Rezagos y ventanas | $t \\dots t-3$; medias 4, 8 y 12 | ACF y PACF |\n"
           "| Anomalías | Respecto a la normal mensual del municipio (2010–2021) | Clima muy distinto entre regiones |\n"
           "| Regularización | L2 con $C$ por validación cruzada | VIF ≫ 10 |\n"
           "| Desbalance | `class_weight` en la malla; sin SMOTE | ≈ 10 % de brotes, variable |\n"
           "| Coordenadas | No se usan como predictoras | Focos que se desplazan |"),
        code('''import inspect  # Código fuente
from e1 import modelo_base as mb  # Modelo base del Primer Entregable
print(inspect.getsource(mb.pipeline_base))  # Pipeline de la regresión logística
print(inspect.getsource(mb.ValidacionAnualPurgada))  # Validación cruzada purgada'''),
        md("**Interpretación.** Todo el preprocesamiento ajustable (mediana e imputación, media y desviación) vive dentro del Pipeline, así que "
           "en cada pliegue se aprende solo con la parte de entrenamiento. La validación cruzada purgada descarta las filas cuyo objetivo "
           "cruza la frontera del año validado."),
    ]
    guardar("3_eda_tiempo_espacio", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
