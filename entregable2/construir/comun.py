"""Piezas comunes para generar los cuadernos del Jupyter Book del Segundo Entregable."""

from pathlib import Path  # Rutas

import nbformat as nbf  # Construcción de cuadernos

CARPETA = Path(__file__).resolve().parent.parent / "cuadernos"  # entregable2/cuadernos
md = nbf.v4.new_markdown_cell  # Atajo: celda de texto
code = nbf.v4.new_code_cell  # Atajo: celda de código

PREAMBULO = '''import os  # Rutas
import sys  # Configuración del intérprete
sys.path.insert(0, os.path.abspath(".."))  # Carpeta entregable2 (paquete tablero)
sys.path.insert(0, os.path.abspath("../.."))  # Raíz del proyecto (paquete src)
sys.path.insert(0, os.path.abspath("../../entregable1"))  # Paquete e1 (código fuente que se muestra)
import warnings  # Control de avisos
warnings.filterwarnings("ignore")  # Oculta avisos de versiones
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from IPython.display import HTML, Markdown, display  # Incrustar figuras y texto
from tablero import figuras as fg  # Las mismas figuras del dashboard
from tablero.datos import A, M, MOD, FEATURES, curvas, metricas_umbral, rezago_espacial  # Los mismos artefactos del dashboard
pd.set_option("display.width", 160)  # Ancho de las tablas
pd.set_option("display.max_columns", 30)  # Columnas visibles
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")  # Cuatro decimales


def mostrar(fig):  # Muestra una figura del dashboard como HTML interactivo
    """Pone el fondo oscuro de las tarjetas del dashboard y carga Plotly desde internet."""
    fig.update_layout(paper_bgcolor=fg.SUPERFICIE, plot_bgcolor=fg.SUPERFICIE, margin=dict(l=40, r=20, t=50, b=40))  # Fondo oscuro
    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displaylogo": False})))  # Figura interactiva'''


def guardar(nombre, celdas):  # Escribe un cuaderno
    nb = nbf.v4.new_notebook()  # Cuaderno vacío
    nb["cells"] = celdas  # Celdas
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}  # Kernel
    CARPETA.mkdir(parents=True, exist_ok=True)  # Carpeta
    nbf.write(nb, CARPETA / f"{nombre}.ipynb")  # Guarda
    print(f"Cuaderno generado: {nombre}.ipynb")  # Aviso
