"""Genera los 11 cuadernos del Primer Entregable (se ejecutan después con nbconvert).

Uso (desde la carpeta entregable1/construir):  python construir_todo.py
"""

import c01_base_datos, c02_objetivo, c03_univariado, c04_bivariado, c05_multivariado, c06_fuga  # Capítulos 1–6
import c07_temporal, c08_espacial, c09_espacio_temporal, c10_preprocesamiento, c11_modelo_base  # Capítulos 7–11

for modulo in (c01_base_datos, c02_objetivo, c03_univariado, c04_bivariado, c05_multivariado, c06_fuga,  # Todos los capítulos
               c07_temporal, c08_espacial, c09_espacio_temporal, c10_preprocesamiento, c11_modelo_base):
    modulo.construir()  # Escribe el cuaderno
