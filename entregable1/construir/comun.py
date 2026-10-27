"""Piezas comunes para generar los cuadernos del Primer Entregable con nbformat."""

from pathlib import Path  # Rutas

import nbformat as nbf  # Construcción de cuadernos

CARPETA = Path(__file__).resolve().parent.parent / "cuadernos"  # entregable1/cuadernos
md = nbf.v4.new_markdown_cell  # Atajo: celda de texto
code = nbf.v4.new_code_cell  # Atajo: celda de código

PREAMBULO = '''import os  # Variables de entorno y rutas
import sys  # Configuración del intérprete
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # Evita un aviso de joblib en Windows
sys.path.insert(0, os.path.abspath("../.."))  # Raíz del proyecto (paquete src)
sys.path.insert(0, os.path.abspath(".."))  # Carpeta entregable1 (paquete e1)
import warnings  # Control de avisos
warnings.filterwarnings("ignore")  # Oculta avisos de versiones que no afectan los resultados
import contextlib, io  # Silenciar una salida de diagnóstico
from joblib.externals.loky import cpu_count  # Conteo de núcleos de joblib
with contextlib.redirect_stderr(io.StringIO()):  # En Windows 11 joblib imprime un diagnóstico al no encontrar «wmic»
    cpu_count(only_physical_cores=True)  # Se calcula una vez (queda en caché) sin ensuciar las salidas
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
import plotly.express as px  # Figuras rápidas
import plotly.graph_objects as go  # Figuras detalladas
from plotly.subplots import make_subplots  # Figuras con varios paneles
from src import config as cf  # Configuración del proyecto (horizonte, periodos, variables)
from e1 import datos as d, estadistica as es, espacial as sp, modelo_base as mb  # Código del entregable
from e1.figuras import mostrar, mapa_base, AZUL, ROJO, VERDE, AMBAR, GRIS, MORADO, PERIODO_COLOR, CUADRANTE_COLOR  # Estilo
np.random.seed(cf.SEMILLA)  # Semilla global (42)
pd.set_option("display.width", 160)  # Ancho de las tablas
pd.set_option("display.max_columns", 40)  # Columnas visibles
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")  # Cuatro decimales'''


def guardar(nombre, celdas):  # Escribe un cuaderno con las celdas dadas
    nb = nbf.v4.new_notebook()  # Cuaderno vacío
    nb["cells"] = celdas  # Celdas
    nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}  # Kernel
    CARPETA.mkdir(parents=True, exist_ok=True)  # Crea la carpeta si no existe
    nbf.write(nb, CARPETA / f"{nombre}.ipynb")  # Guarda
    print(f"Cuaderno generado: {nombre}.ipynb")  # Aviso
