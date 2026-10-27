"""Genera los 8 cuadernos del Jupyter Book del Tercer Entregable (se ejecutan después con nbconvert).

Uso (desde la carpeta entregable3/construir):  python construir_todo.py
"""

import c1_datos, c2_diseno, c3_optimizacion, c4_computacional, c5_clasificacion, c6_regresion, c7_interpretabilidad, c8_estadistica  # Capítulos

for modulo in (c1_datos, c2_diseno, c3_optimizacion, c4_computacional, c5_clasificacion, c6_regresion, c7_interpretabilidad, c8_estadistica):  # Cada capítulo
    modulo.construir()  # Escribe el cuaderno
