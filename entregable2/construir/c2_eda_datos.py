"""Cuaderno 2: EDA I — calidad, objetivo, análisis uni, bi y multivariado y fuga (pestaña EDA del dashboard)."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# EDA I: datos, objetivo y relaciones entre variables\n\n"
           "Acompaña las sub-pestañas **Datos y calidad, Variable objetivo, Univariado, Bivariado, Multivariado y Fuga de datos** del "
           "dashboard. Cada figura es la misma del dashboard; después de cada una va su interpretación y, en los recuadros *Detalle*, lo que "
           "el dashboard no muestra: las fórmulas de las pruebas y las tablas completas. Todas las decisiones usan las semanas 2010–2021."),
        code(PREAMBULO + '''
cal = A["calidad"]  # Calidad de los datos
print(f"Temperatura imputada: {cal['frac_imputada']:.1%} de las semanas")  # Faltante oculto
cal["patron"].to_frame("municipios")  # Patrón por municipio'''),

        md("## Datos y calidad\n\n### Patrón de los faltantes"),
        code('''mostrar(fg.fig_matriz_faltantes(cal["matriz_faltantes"]))  # Matriz municipio × año
mostrar(fg.fig_temp_imputada(cal["hist_temp"]))  # Observada frente a imputada'''),
        md("**Interpretación.** No hay celdas nulas, pero la temperatura está imputada en el 64 % de las semanas. El patrón es por "
           "**municipio** (226 nunca medidos, 21 siempre medidos, 129 parcialmente) y estable en el tiempo. La imputación original es casi "
           "una constante cerca de 22,4 °C: no conserva la diferencia climática entre municipios."),
        md("### Mecanismo de los faltantes\n\n"
           ":::{admonition} Detalle: prueba de Little\n"
           "Con $J$ patrones de faltantes, $\\boldsymbol\\mu$ y $\\boldsymbol\\Sigma$ estimadas por máxima verosimilitud (algoritmo EM) y "
           "$\\bar{\\mathbf x}_j$ la media de las variables observadas en el patrón $j$:\n\n"
           "$$d^2 = \\sum_{j=1}^{J} n_j\\,(\\bar{\\mathbf x}_j - \\boldsymbol\\mu_j)^\\top \\boldsymbol\\Sigma_j^{-1} (\\bar{\\mathbf x}_j - \\boldsymbol\\mu_j) \\sim \\chi^2\\Big(\\sum_j p_j - p\\Big)\\ \\text{bajo MCAR}.$$\n\n"
           "La implementación (EM incluido) está en `entregable1/e1/estadistica.py`.\n"
           ":::"),
        code('''li = cal["little"]  # Prueba de Little
print(f"Little: d² = {li['d2']:,.1f}, gl = {li['gl']}, p = {li['p']:.1e}")  # Resultado
pd.Series(cal["smd"], name="diferencia estandarizada (faltante − observado)").to_frame()  # Tamaños de efecto'''),
        md("**Interpretación.** Little rechaza MCAR. La diferencia más grande es el tamaño del municipio ($d \\approx -0{,}7$): faltan las "
           "mediciones en municipios pequeños, sin estación meteorológica. Es **MAR estructural**; un componente MNAR no es verificable. "
           "Decisión: conservar la bandera `temp_missing` (método del indicador) en lugar de reimputar con vecinos."),
        md("### Valores imposibles, atípicos y representatividad"),
        code('''pd.Series(cal["chequeos"], name="filas con problema").to_frame()  # Consistencia'''),
        code('''display(cal["atipicos"].to_frame())  # Atípicos por la regla de Tukey
cal["top_casos"]  # Las semanas con más casos'''),
        code('''rep = cal["representatividad"]  # Cobertura por región
print(f"Población cubierta en 2024: {cal['pob_cubierta'] / 1e6:.1f} millones")  # Cobertura poblacional
rep'''),
        md("**Interpretación.** No hay valores imposibles ni duplicados. Los atípicos de casos son epidemias reales (Cali en 2024) y se "
           "conservan con transformación logarítmica; el 18 % de \"atípicos\" de temperatura es un artefacto de la imputación. El panel cubre "
           "un tercio de los municipios pero ≈ 60 % de la población, con el Caribe y la Orinoquía sobrerrepresentados."),

        md("## Variable objetivo"),
        code('''o = A["objetivo"]  # Objetivo
mostrar(fg.fig_prevalencia_anual(o["anual"]))  # Prevalencia por año
mostrar(fg.fig_mes_region(o["mes"], o["region"]))  # Por mes y región
o["anual"].rename(columns={"mean": "prevalencia", "sum": "brotes", "size": "filas"})'''),
        md("**Interpretación.** La prevalencia del brote va del 1,7 % (2017) al 25 % (2019): el desbalance depende del año. La exactitud no "
           "sirve para decidir (no alertar nunca acierta el 90 %); se usan AUC-PR, MCC y Brier, y la validación recorre años completos. La "
           "estacionalidad mensual del objetivo es débil porque el canal endémico ya la descuenta."),
        code('''mostrar(fg.fig_objetivo_tiempo(o["nacional"]))  # Municipios en brote
mapa = M.merge(o["mapa"].reset_index(), on="COD_MUN_N")  # Prevalencia por municipio
mostrar(fg.fig_mapa_valor(mapa, "prevalencia", "brote", ".0%", (0, 0.3)))  # Mapa
mapa["prevalencia"].describe().to_frame().T'''),
        md("**Interpretación.** Los brotes llegan en ondas nacionales (2015–2016 y 2019). En el espacio la prevalencia varía entre municipios "
           "(mediana ≈ 8 %), pero no sigue el nivel de casos, porque cada municipio se compara consigo mismo."),

        md("## Análisis unidimensional"),
        code('''u = A["univariado"]  # Univariado
for v in ["casos_totales", "temp_mean", "prec_total", "ndvi_mean"]:  # Variables representativas
    mostrar(fg.fig_histograma(u["hist"][v], v))  # Histograma + caja
u["tabla"]  # Estadísticos, forma, atípicos y normalidad'''),
        md("**Interpretación.** Los casos tienen asimetría > 18 y curtosis en cientos (mediana 0): se usa $\\log(1+x)$. La temperatura es "
           "bimodal por la imputación; la lluvia es asimétrica a la derecha; el NDVI, levemente a la izquierda. Ninguna variable es normal "
           "(todas las pruebas rechazan aun con 5 000 filas), lo que justifica pruebas no paramétricas."),
        code('''pd.DataFrame({"departamentos raros (< 3 municipios)": [", ".join(u["categoricas"]["departamento"][u["categoricas"]["departamento"] < 3].index)],
              "semanas por fase ENSO": [u["categoricas"]["enso"].to_dict()], "municipios por región": [u["categoricas"]["region"].to_dict()]}).T'''),
        md("**Interpretación.** El departamento tiene categorías raras y no se codifica (haría memorizar lugares); la semana 53 (2015 y 2020) "
           "se resuelve con seno y coseno; ninguna fase ENSO es rara."),

        md("## Análisis bidimensional\n\n"
           ":::{admonition} Detalle: medidas de asociación usadas\n"
           "* **Correlación biserial de rangos** (Mann-Whitney): $r_{rb} = 2U/(n_1 n_0) - 1 \\in [-1, 1]$; es $2\\,\\text{AUC} - 1$.\n"
           "* **Kruskal-Wallis** con tamaño de efecto $\\varepsilon^2 = H/(n-1)$ (0,01 pequeño, 0,08 mediano, 0,26 grande).\n"
           "* **V de Cramér** con corrección de sesgo: $V = \\sqrt{\\tilde\\varphi^2/\\min(\\tilde k - 1, \\tilde r - 1)}$.\n"
           "* **Información mutua** $I(X;Y) = \\sum p(x,y)\\ln\\frac{p(x,y)}{p(x)p(y)}$, estimada por vecinos más cercanos; capta relaciones no lineales.\n"
           "* **VIF** $= 1/(1-R_j^2) = [\\mathbf R^{-1}]_{jj}$, con $\\mathbf R$ la matriz de correlación.\n"
           "* **Holm** controla la tasa de error de la familia; **Benjamini-Hochberg**, la tasa de falsos descubrimientos.\n"
           ":::"),
        code('''b = A["bivariado"]  # Bivariado
for p in b["pares"]:  # Pares numéricos
    x, y = p.split("|")  # Variables
    mostrar(fg.fig_densidad(b["pares"][p], x, y))  # Densidad conjunta
pd.DataFrame({p: {"Pearson": d["pearson"], "Spearman": d["spearman"], "n": d["n"]} for p, d in b["pares"].items()}).T'''),
        md("**Interpretación.** Temperatura y lluvia están poco relacionadas; la historia de casos es muy persistente (Pearson 0,91 frente a "
           "Spearman 0,78: la correlación lineal la dominan las semanas epidémicas); la temperatura casi no se correlaciona con los casos; el "
           "ONI rezagado sí, de forma positiva."),
        code('''mostrar(fg.fig_correlacion(b["corr"]))  # Spearman
b["vif"].sort_values(ascending=False).to_frame("VIF").T  # Multicolinealidad'''),
        md("**Interpretación.** Las medias móviles de una misma variable forman bloques con $|\\rho| > 0{,}9$ y VIF ≫ 10 (hasta ≈ 1 000 en la "
           "temperatura de 8 semanas). La predicción no sufre, pero los coeficientes individuales son inestables: se usa penalización L2."),
        code('''mostrar(fg.fig_asociacion(b["asociacion"], "Información mutua"))  # Asociación
b["asociacion"].sort_values("mi", ascending=False)  # Tabla completa (no aparece en el dashboard)'''),
        md("**Interpretación.** La historia de casos domina (AUC univariados de 0,77–0,88). El flujo, el ONI y la población tienen asociación "
           "moderada. El clima local tiene efectos pequeños aunque casi todo es significativo tras Holm y FDR: con $n$ grande el tamaño de efecto "
           "manda."),
        code('''for v in ["temp_mean_m8", "prec_total_m8", "oni_l8", "lc_m4"]:  # Relaciones no lineales
    mostrar(fg.fig_deciles(b["deciles"][v], v))  # Tasa de brote por decil'''),
        md("**Interpretación.** Con la temperatura medida, el riesgo sube hacia los 25–28 °C y se estabiliza; con la historia de casos crece de "
           "forma monótona y fuerte. Las formas no lineales anticipan una limitación de los modelos base lineales."),
        code('''display(b["kruskal"])  # Por región
display(b["cramer"])  # Categóricas
b["enso"]  # Brote por fase ENSO'''),
        md("**Interpretación.** La temperatura difiere muchísimo entre regiones ($\\varepsilon^2 \\approx 0{,}41$) y la historia de casos casi nada. "
           "La fase ENSO es la categórica más asociada al brote (V ≈ 0,17): 16,8 % en El Niño frente a 4,5 % en La Niña."),

        md("## Análisis multivariado"),
        code('''mu = A["multivariado"]  # Multivariado
mostrar(fg.fig_pca(mu["varianza"], mu["pca_xy"], mu["pca_y"]))  # PCA
acum = np.cumsum(mu["varianza"])  # Varianza acumulada
print(f"Componentes para 80 %: {np.searchsorted(acum, 0.8) + 1} · para 90 %: {np.searchsorted(acum, 0.9) + 1} · autovalor > 1: {(mu['autovalores'] > 1).sum()}")
mu["cargas"].round(3)  # Cargas completas de las 4 primeras componentes'''),
        md("**Interpretación.** 8 componentes explican el 80 % de la varianza: la dimensionalidad efectiva es mucho menor que 34. CP1 es la "
           "historia de casos; CP2–CP3, temperatura y NDVI; CP4, lluvia. Los brotes no se separan linealmente en dos dimensiones."),
        code('''mostrar(fg.fig_tsne(mu["tsne"], mu["tsne_y"], mu["tsne_region"], "region"))  # t-SNE por región
display(mu["atipicos"])  # Atípicos multivariados
mu["atipicos_municipios"].to_frame("filas atípicas")'''),
        md("**Interpretación.** t-SNE agrupa más por región que por objetivo: el brote es un estado transitorio, no una subpoblación. Los "
           "atípicos multivariados tienen una tasa de brote cuatro veces mayor (semanas epidémicas, grandes ciudades): se conservan.\n\n"
           ":::{admonition} Detalle: distancia de Mahalanobis\n"
           "$D^2(\\mathbf x) = (\\mathbf x - \\boldsymbol\\mu)^\\top \\boldsymbol\\Sigma^{-1}(\\mathbf x - \\boldsymbol\\mu) \\sim \\chi^2_{34}$ bajo normalidad; se marca "
           "atípico por encima del percentil 99,9. Por la colinealidad se usa la pseudo-inversa de $\\boldsymbol\\Sigma$.\n"
           ":::"),
        code('''mostrar(fg.fig_grupos(mu["perfil"]))  # Grupos de municipios
print("Silueta por k:", {k: round(v, 3) for k, v in mu["silueta"].items()})  # Elección de k
mu["perfil"].groupby("grupo")[["prevalencia", "temp", "lluvia", "lpob", "flujo"]].mean()  # Perfil de cada grupo'''),
        md("**Interpretación.** La mejor silueta es 0,33 con $k = 2$: estructura débil a moderada. Los grupos se separan por tamaño y movilidad, "
           "no por clima; los municipios grandes y conectados tienen más del doble de semanas en brote."),

        md("## Fuga de datos"),
        code('''f = A["fuga"]  # Auditoría
mostrar(fg.fig_auc_univariado(f["auc_univariado"]))  # AUC univariado
f["demo"].to_frame("AUC-ROC de validación cruzada")  # Demostración'''),
        md("**Interpretación.** Ninguna predictora supera 0,9 de AUC univariado; los casos de la semana objetivo (información del futuro) "
           "llegan a ≈ 0,96 y se descartan. Con la validación correcta el AUC es ≈ 0,90; mezclando semanas sube un poco, y con la fuga directa "
           "llega a ≈ 0,99. El umbral objetivo $u_{t+4}$ usa semanas $t-258$ a $t-46$, todas anteriores a $t$: no es fuga."),
    ]
    guardar("2_eda_datos", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
