"""Lanza la validación anidada de las 140 combinaciones y el reajuste final.

Se ejecuta desde un script aparte (y no con ``python -m e3.experimento``) para que, en Windows, los procesos de joblib
importen las funciones desde el módulo ``e3.experimento`` y no desde ``__main__``.

Uso (desde la carpeta entregable3):  python correr_experimento.py
"""

import os  # Rutas
import sys  # Rutas de importación

AQUI = os.path.dirname(os.path.abspath(__file__))  # Carpeta entregable3
sys.path[:0] = [AQUI, os.path.join(AQUI, ".."), os.path.join(AQUI, "..", "entregable1")]  # e3, src y e1

if __name__ == "__main__":  # Protección necesaria en Windows para procesos hijos
    from e3 import experimento  # Módulo del experimento
    experimento.main()  # Corre lo pendiente
