"""Cuaderno 1: Contexto del problema (pestaña 1 del dashboard) con detalles adicionales."""

from comun import PREAMBULO, code, guardar, md  # Piezas comunes


def construir():  # Celdas
    c = [
        md("# Contexto del problema\n\n"
           "Este capítulo acompaña la pestaña **Contexto del problema** del dashboard: interpreta cada elemento y agrega detalles que "
           "no caben en el dashboard (la construcción del objetivo, la comparación de las funciones de pérdida de los dos modelos base y "
           "la verificación de la partición)."),
        code(PREAMBULO + '''
k = A["contexto"]["kpi"]  # Indicadores de la portada
pd.Series({"Casos 2010–2024": f"{k['casos']:,}", "Municipios": k["municipios"], "Departamentos": k["departamentos"], "Semanas": k["semanas"],
           "Año récord": f"{k['anio_record']} ({k['casos_record']:,} casos)", "Prevalencia de brote (entrenamiento)": f"{k['prev_tr']:.1%}",
           "Prevalencia de brote (validación 2022)": f"{k['prev_va']:.1%}", "Predictoras": k["p"]}, name="valor").to_frame()'''),
        md("**Interpretación de los indicadores.** El panel reúne 1,26 millones de casos notificados en 376 municipios durante 782 "
           "semanas. 2024 es el año récord y queda dentro de la prueba, así que los modelos se evalúan en un régimen más extremo que el de "
           "entrenamiento. La clase brote es minoritaria (≈ 10 %), lo que orienta la elección de métricas.\n\n"
           "## Pregunta y objetivos\n\n"
           "**Pregunta.** ¿Es posible anticipar, con cuatro semanas de margen, si un municipio entrará en brote de dengue usando solo la "
           "información disponible en la semana actual?\n\n"
           "El dashboard presenta cuatro objetivos: describir la base de datos y su estructura; definir el brote sin información del futuro; "
           "explorar la relación de la historia de casos, el clima y El Niño con los brotes; y entrenar dos modelos base lineales comparados "
           "con líneas base triviales."),

        md("## Casos por año"),
        code('''mostrar(fg.fig_anual(A["contexto"]["anual"]))  # Misma figura de la pestaña Contexto
A["contexto"]["anual"].to_frame("casos").T'''),
        md("**Interpretación.** Las epidemias de 2010, 2013, 2019 y 2023–2024 dominan la serie y coinciden con episodios de El Niño. 2017 "
           "y 2018 son años de muy baja transmisión. Esta alternancia es la razón de que la prevalencia del objetivo cambie tanto entre años."),

        md("## Detalle: construcción del objetivo con el canal endémico\n\n"
           "El dashboard muestra la fórmula; aquí está el código que la implementa (`src/datos.py`). El umbral se calcula de forma "
           "vectorizada: para cada semana se toman los 25 valores históricos (5 años × 5 semanas) y su percentil 75."),
        code('''import inspect  # Código fuente
from src import datos as sd  # Módulo del canal endémico
print(inspect.getsource(sd.canal_endemico))  # Función que calcula el umbral'''),
        code('''serie = A["objetivo"]["canal"][76001]  # Cali
mostrar(fg.fig_canal(serie, "Santiago de Cali"))  # Canal endémico
print(f"Semanas de brote en Cali 2015–2021: {int(serie['brote'].sum())} de {len(serie)} ({serie['brote'].mean():.1%})")  # Resumen'''),
        md("**Interpretación.** El umbral (línea punteada) sigue la historia de cada municipio: en Cali es alto porque la transmisión es "
           "habitual. Un brote no es \"muchos casos\", sino **más casos de lo normal para esa época del año en ese municipio**. El requisito "
           "de al menos 3 casos evita que un solo caso en un municipio sin historia cuente como brote."),

        md("## Detalle: las funciones de pérdida de los dos modelos base\n\n"
           "La regresión logística minimiza la pérdida logarítmica; la SVM lineal (`LinearSVC`) minimiza la bisagra al cuadrado. Ambas son "
           "funciones convexas del margen $z = \\tilde y\\,(\\mathbf w^\\top \\mathbf x + b)$ con $\\tilde y \\in \\{-1, 1\\}$:\n\n"
           "$$\\ell_{\\text{log}}(z) = \\ln(1 + e^{-z}), \\qquad \\ell_{\\text{bisagra}^2}(z) = \\max(0,\\, 1 - z)^2.$$\n\n"
           "La diferencia clave: la bisagra es **exactamente cero** para los puntos bien clasificados con margen ($z \\ge 1$), así que la SVM "
           "solo aprende de los puntos cercanos a la frontera (vectores de soporte); la logística sigue aprendiendo de todos los puntos y "
           "produce probabilidades directamente. Por eso la SVM necesita un paso adicional (escalado de Platt) para dar probabilidades."),
        code('''import plotly.graph_objects as go  # Figura
z = np.linspace(-3, 3, 300)  # Márgenes
fig = go.Figure()  # Comparación de pérdidas
fig.add_scatter(x=z, y=np.log1p(np.exp(-z)), name="Logística: ln(1 + e^(−z))", line=dict(color=fg.AZUL, width=3))  # Logística
fig.add_scatter(x=z, y=np.maximum(0, 1 - z) ** 2, name="Bisagra al cuadrado: max(0, 1 − z)²", line=dict(color=fg.MAGENTA, width=3))  # SVM
fig.add_scatter(x=z, y=(z < 0).astype(float), name="Pérdida 0-1", line=dict(color=fg.GRIS, dash="dot"))  # Referencia
fig.update_layout(xaxis_title="margen z = ỹ (wᵀx + b)", yaxis_title="pérdida", yaxis_range=[0, 5])  # Ejes
mostrar(fg.tema(fig, 380))  # Muestra'''),
        md("**Interpretación.** Las dos pérdidas son sustitutos convexos de la pérdida 0-1 (que no se puede optimizar directamente). Para "
           "márgenes negativos (errores) ambas crecen; para márgenes grandes la logística decae suavemente y la bisagra se anula. Con datos "
           "tabulares, muchas filas y las mismas variables, ambas suelen encontrar hiperplanos muy parecidos, como se verá en el capítulo de "
           "modelos base."),

        md("## Detalle: la partición se verifica, no se supone"),
        code('''mostrar(fg.fig_particion(A["contexto"]["linea_particion"]))  # Línea de tiempo
A["contexto"]["particion"]  # Filas y fechas de cada periodo'''),
        md("**Interpretación.** Las 6 016 filas *purgadas* son las cuatro semanas de cada frontera, cuya etiqueta cae en el periodo siguiente "
           "(por ejemplo, diciembre de 2021, cuyo objetivo es enero de 2022). Con la purga, la última etiqueta del entrenamiento es anterior "
           "a la primera semana de la validación, y lo mismo entre validación y prueba: ninguna etiqueta se usa fuera de su periodo.\n\n"
           "## Detalle: consideraciones éticas\n\n"
           "El panel contiene solo conteos agregados; no requiere consentimiento ni comité de ética. Los cruces finos (edad × sexo × semana en "
           "municipios pequeños) no se publican para evitar cualquier riesgo de reidentificación. El riesgo más relevante es de **equidad**: si "
           "la notificación es peor donde hay menos recursos, el modelo aprenderá menos riesgo justo donde la vigilancia es más débil."),
    ]
    guardar("1_contexto", c)  # Escribe


if __name__ == "__main__":  # Ejecución directa
    construir()  # Genera
