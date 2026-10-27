"""Imprime las salidas de texto y tablas de un cuaderno ejecutado (para revisar las interpretaciones)."""
import sys  # Argumentos

import nbformat as nbf  # Cuadernos

nb = nbf.read(sys.argv[1], as_version=4)  # Cuaderno
for i, c in enumerate(nb.cells):  # Celdas
    for o in c.get("outputs", []) if c.cell_type == "code" else []:  # Salidas
        if o.get("output_type") == "stream":  # Texto impreso
            print(f"[{i}]", o["text"][:2500])  # Muestra
        elif o.get("output_type") == "error":  # Error
            print(f"[{i}] ERROR", o["ename"], o["evalue"][:300])  # Muestra
        elif "text/plain" in o.get("data", {}) and "<table" in o["data"].get("text/html", ""):  # Tabla
            print(f"[{i}] TABLA")  # Encabezado
            print(o["data"]["text/plain"][:7000])  # Texto de la tabla
