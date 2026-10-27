"""Une la portada, los 11 cuadernos ejecutados y las conclusiones en un solo cuaderno del Primer Entregable.

Uso (desde la carpeta entregable1/construir):  python unir_cuadernos.py
Después se ejecuta de principio a fin con nbconvert para comprobar que corre como un único cuaderno.
"""

from pathlib import Path  # Rutas

import nbformat as nbf  # Lectura y escritura de cuadernos

RAIZ = Path(__file__).resolve().parent.parent  # Carpeta entregable1
CUADERNOS = sorted((RAIZ / "cuadernos").glob("[01][0-9]_*.ipynb"))  # 01_base_datos … 11_modelo_base
DESTINO = RAIZ / "cuadernos" / "Entregable1_DanielSerranoRomero.ipynb"  # Misma carpeta: las rutas relativas siguen valiendo

celdas = [nbf.v4.new_markdown_cell((RAIZ / "intro.md").read_text(encoding="utf-8"))]  # Portada
for ruta in CUADERNOS:  # Cada capítulo en orden
    nb = nbf.read(ruta, as_version=4)  # Cuaderno ejecutado
    for celda in nb.cells:  # Sus celdas
        celdas.append(celda)  # Agrega con sus salidas
celdas.append(nbf.v4.new_markdown_cell((RAIZ / "conclusiones.md").read_text(encoding="utf-8")))  # Cierre

unido = nbf.v4.new_notebook(cells=celdas)  # Cuaderno único
unido["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}  # Kernel
for i, celda in enumerate(unido.cells):  # Identificadores únicos
    celda["id"] = f"celda-{i:04d}"  # Uno por celda
nbf.write(unido, DESTINO)  # Guarda
print(f"Cuaderno unificado: {DESTINO.name} · {len(celdas)} celdas de {len(CUADERNOS)} capítulos")  # Aviso
