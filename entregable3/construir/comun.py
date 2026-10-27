"""Piezas comunes para generar los cuadernos del Jupyter Book del Tercer Entregable."""

from pathlib import Path  # Rutas

import nbformat as nbf  # Construcción de cuadernos

CARPETA = Path(__file__).resolve().parent.parent / "cuadernos"  # entregable3/cuadernos
md = nbf.v4.new_markdown_cell  # Atajo: celda de texto
code = nbf.v4.new_code_cell  # Atajo: celda de código

PREAMBULO = '''import os  # Rutas
import sys  # Configuración del intérprete
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # Evita un aviso de joblib en Windows
sys.path[:0] = [os.path.abspath(".."), os.path.abspath("../.."), os.path.abspath("../../entregable1")]  # e3, src y e1
import warnings  # Control de avisos
warnings.filterwarnings("ignore")  # Oculta avisos de versiones
import contextlib, io  # Silenciar un diagnóstico de joblib en Windows
from joblib.externals.loky import cpu_count  # Conteo de núcleos
with contextlib.redirect_stderr(io.StringIO()):  # joblib imprime un diagnóstico al no encontrar «wmic»
    cpu_count(only_physical_cores=True)  # Se calcula una vez (queda en caché)
import pickle  # Lectura de resultados
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
import plotly.express as px  # Figuras rápidas
import plotly.graph_objects as go  # Figuras detalladas
from plotly.subplots import make_subplots  # Paneles
from IPython.display import Markdown, display  # Texto enriquecido
from e1.figuras import mostrar, AZUL, ROJO, VERDE, AMBAR, GRIS, MORADO  # Estilo de figuras (tema claro del informe)
from e3 import config as c3, datos as dd, modelos as mo, optimizadores as op, experimento as ex  # Código del entregable
c3.fijar_semilla()  # Semilla global (42)
pd.set_option("display.width", 170)  # Ancho de las tablas
pd.set_option("display.max_columns", 30)  # Columnas visibles
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")  # Cuatro decimales
COLORES = {"Grid Search": GRIS, "Random Search": AMBAR, "Bayesiana (Optuna TPE)": AZUL, "Genética (DEAP)": VERDE, "Successive Halving": MORADO}  # Color por optimizador'''


def guardar(nombre, celdas):  # Escribe un cuaderno
    nb = nbf.v4.new_notebook()  # Cuaderno vacío
    nb["cells"] = celdas  # Celdas
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}  # Kernel
    CARPETA.mkdir(parents=True, exist_ok=True)  # Carpeta
    nbf.write(nb, CARPETA / f"{nombre}.ipynb")  # Guarda
    print(f"Cuaderno generado: {nombre}.ipynb")  # Aviso
