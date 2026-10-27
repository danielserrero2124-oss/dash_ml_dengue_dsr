"""Imprime las salidas de texto de un cuaderno ejecutado (para revisar las interpretaciones)."""
import sys  # Argumentos
import nbformat as nbf  # Lectura de cuadernos
import re  # Limpieza de HTML

nb = nbf.read(sys.argv[1], as_version=4)  # Cuaderno
for i, c in enumerate(nb.cells):  # Celdas
    if c.cell_type != "code":  # Solo código
        continue  # Salta
    for o in c.get("outputs", []):  # Salidas
        if o.get("output_type") == "stream":  # Texto impreso
            print(f"[{i}]", o["text"][:3000])  # Muestra
        elif o.get("output_type") == "error":  # Error
            print(f"[{i}] ERROR", o["ename"], o["evalue"])  # Muestra
        elif "text/plain" in o.get("data", {}) and "text/html" not in o.get("data", {}) or ("text/html" in o.get("data", {}) and "<table" in o["data"]["text/html"]):  # Tablas
            print(f"[{i}]", o["data"].get("text/plain", "")[:3000])  # Muestra
