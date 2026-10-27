"""Une la portada, los 4 cuadernos ejecutados, el capítulo del dashboard y las conclusiones en un solo cuaderno.

Uso (desde la carpeta entregable3/construir):  python unir_cuadernos.py
Después se ejecuta de principio a fin con nbconvert para comprobar que corre como un único cuaderno.
"""

from pathlib import Path  # Rutas

import nbformat as nbf  # Lectura y escritura de cuadernos

RAIZ = Path(__file__).resolve().parent.parent  # Carpeta entregable3
CUADERNOS = sorted((RAIZ / "cuadernos").glob("[1-8]_*.ipynb"))  # 1_datos … 8_estadistica
DESTINO = RAIZ / "cuadernos" / "Entregable3_DanielSerranoRomero.ipynb"  # Misma carpeta: las rutas relativas siguen valiendo

celdas = [nbf.v4.new_markdown_cell((RAIZ / "intro.md").read_text(encoding="utf-8"))]  # Portada
for ruta in CUADERNOS:  # Cada capítulo en orden
    celdas.extend(nbf.read(ruta, as_version=4).cells)  # Sus celdas con salidas
for md in ("conclusiones.md",):  # Capítulos de texto
    celdas.append(nbf.v4.new_markdown_cell((RAIZ / md).read_text(encoding="utf-8")))  # Agrega

unido = nbf.v4.new_notebook(cells=celdas)  # Cuaderno único
unido["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}  # Kernel
for i, celda in enumerate(unido.cells):  # Identificadores únicos
    celda["id"] = f"celda-{i:04d}"  # Uno por celda
nbf.write(unido, DESTINO)  # Guarda
print(f"Cuaderno unificado: {DESTINO.name} · {len(celdas)} celdas de {len(CUADERNOS)} capítulos")  # Aviso
