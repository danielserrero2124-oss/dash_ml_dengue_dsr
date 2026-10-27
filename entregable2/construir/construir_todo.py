"""Genera los 4 cuadernos del Jupyter Book del Segundo Entregable (se ejecutan después con nbconvert).

Uso (desde la carpeta entregable2/construir):  python construir_todo.py
"""

import c1_contexto, c2_eda_datos, c3_eda_tiempo_espacio, c4_modelos  # Capítulos

for modulo in (c1_contexto, c2_eda_datos, c3_eda_tiempo_espacio, c4_modelos):  # Cada capítulo
    modulo.construir()  # Escribe el cuaderno
