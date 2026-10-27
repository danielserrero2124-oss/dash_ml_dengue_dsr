"""Código del Tercer Entregable: 140 modelos (7 × 4 × 4 en clasificación y 7 × 4 en regresión) con validación anidada.

Módulos:
  config        semilla global, periodos de la validación anidada, tamaños de muestra y presupuestos
  datos         panel, objetivo de regresión y particiones anidadas con purga
  modelos       los 7 modelos de cada tarea, sus espacios de búsqueda y las 4 estrategias de balanceo
  optimizadores Grid, Random, Bayesiana (Optuna, TPE), Genética (DEAP) y Successive Halving
  experimento   bucle externo / interno, tabla maestra de las 140 combinaciones y reajuste final
  evaluacion    métricas, calibración y pruebas sobre los residuos
  estadistica   Friedman, Nemenyi, DeLong, bootstrap BCa, delta de Cliff, MCS, Giacomini-White, Clark-West, DM-HLN, SPA
  complejidad   escalamiento empírico, variantes eficientes de cada modelo, paralelización y perfilamiento
"""
