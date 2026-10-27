"""Análisis complementarios que se ejecutan después del experimento principal:

1. Estudio de eficiencia de los optimizadores (XGBoost, 40 evaluaciones, 3 semillas, con Successive Halving).
2. Sensibilidad a la semilla del mejor ajuste de cada modelo de clasificación (5 semillas).
3. Optimización computacional (complejidad, variantes, paralelización, perfilamiento), al final y sin otros procesos.

Uso (desde la carpeta entregable3):  python correr_analisis.py
"""

import os  # Rutas
import sys  # Rutas de importación
import time  # Cronómetro

AQUI = os.path.dirname(os.path.abspath(__file__))  # Carpeta entregable3
sys.path[:0] = [AQUI, os.path.join(AQUI, ".."), os.path.join(AQUI, "..", "entregable1")]  # e3, src y e1

if __name__ == "__main__":  # Protección para los procesos de joblib en Windows
    import pandas as pd  # Tablas
    from e3 import complejidad, config as c3, estudio_optimizadores, robustez  # Análisis
    t0 = time.perf_counter()  # Cronómetro
    if not (c3.RESULTADOS / "estudio_optimizadores.joblib").exists():  # Estudio de optimizadores
        estudio_optimizadores.main()
        print(f"Estudio de optimizadores: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if not (c3.RESULTADOS / "robustez_semillas.joblib").exists():  # Sensibilidad a la semilla
        t = pd.read_csv(c3.RESULTADOS / "tabla_maestra.csv")  # Tabla maestra
        ext = t[t["tarea"] == "clf"].groupby(["modelo", "balanceo", "optimizador"])["AUC-PR"].mean().reset_index()  # Bucle externo
        mejores = ext.sort_values("AUC-PR", ascending=False).groupby("modelo").first().reset_index()  # Mejor de cada modelo
        robustez.main([tuple(x) for x in mejores[["modelo", "balanceo", "optimizador"]].to_numpy()])
        print(f"Robustez: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if not (c3.RESULTADOS / "complejidad.joblib").exists():  # Complejidad (al final, sin competencia por los núcleos)
        complejidad.main()
        print(f"Complejidad: {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
