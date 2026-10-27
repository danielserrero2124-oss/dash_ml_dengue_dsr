# Conclusiones

## Hallazgos

1. **Los modelos de árboles y la SVM con núcleo aproximado son los mejores, y son equivalentes entre sí.** En la prueba 2023–2024,
   XGBoost sin balanceo logra AUC-PR 0,823 y AUC-ROC 0,898, la SVM 0,820 y Random Forest 0,818. Friedman–Nemenyi los agrupa como
   estadísticamente indistinguibles; DeLong con Holm no separa XGBoost de la SVM, y la diferencia de XGBoost con Random Forest
   (0,0014 de AUC) tiene un delta de Cliff despreciable. Los tres superan con claridad a la regresión logística, KNN, el árbol único y
   Naive Bayes.
2. **La ganancia sobre el modelo base es real pero moderada:** de AUC-PR 0,809 (logística del Primer Entregable) a 0,823. El techo lo
   fija la información disponible, no el algoritmo: la historia de casos domina y los brotes súbitos (sin casos previos) no se
   pueden anticipar con las variables del propio municipio.
3. **El balanceo no ayuda en este problema.** SMOTE, ADASYN y `class_weight` no mejoran el AUC-PR (lo reducen 0,005–0,024) y
   deterioran la calibración (ECE de 0,07 a 0,14–0,18). El desbalance se maneja mejor eligiendo el umbral en 2022 y, si se necesitan
   probabilidades, recalibrando con Platt o regresión isotónica (ECE ≤ 0,06).
4. **El optimizador importa menos que el modelo.** Con 20 evaluaciones los cuatro métodos llegan al mismo desempeño externo. La
   búsqueda bayesiana (TPE) tiene la mejor curva de desempeño en cualquier momento y, en el estudio con más presupuesto, es el único
   método que en las tres semillas llega al casi óptimo (≈ 14 evaluaciones; Grid y Random no lo logran en 40). Successive Halving
   ofrece la mejor calidad por minuto. Las mejoras internas por debajo de ≈ 0,005 no se trasladan al bucle externo.
5. **Regresión:** XGBoost y la SVR aproximada son los mejores (RMSE 0,573 y $R^2$ 0,79 en la prueba, 13 % menos error que la
   persistencia; el MCS retiene 3 de 28 combinaciones). Los residuos son heterocedásticos y dependientes (White, BDS y Ljung-Box
   rechazan): queda estructura temporal y espacial sin capturar.
6. **Eficiencia computacional verificada, no supuesta:** `hist` acelera XGBoost 16× y la parada temprana mejora el modelo con 125
   árboles; FAISS HNSW acelera KNN 13× con 93 % de vecinos exactos; Nyström evita el costo cuadrático de la SVM (exponente empírico 2,0
   del núcleo exacto); `partial_fit` reduce la memoria de Naive Bayes 10×. SAGA no es más rápido en este conjunto denso de 34
   variables, y se reporta así.

## Limitaciones

* La búsqueda usa submuestras estratificadas (20 000 filas en el bucle externo, 40 000 en el reajuste final) para que 420 tareas
  anidadas quepan en el presupuesto de cómputo; el año evaluado siempre se usa completo.
* Tres pliegues externos (2019, 2020 y 2021) cubren regímenes distintos pero dan una estimación de la variabilidad con pocos puntos;
  la desviación entre pliegues (≈ 0,10) refleja la prevalencia de cada año más que la inestabilidad del modelo.
* El algoritmo genético trabaja con una población pequeña (6) por el presupuesto de 20 evaluaciones y converge prematuramente en la
  mayoría de las corridas.
* No hubo GPU disponible para medir `device='cuda'` en XGBoost.
* Las variables de vecindad espacial, que el EDA y los residuos señalan como la información faltante, no se incluyeron para mantener
  la comparación con los entregables anteriores.

## Trabajo futuro

* Agregar rezagos espaciales (promedio de vecinos en $t$) y su interacción con El Niño.
* Optimización multiobjetivo (AUC-PR y calibración) y umbrales sensibles al costo de las alertas.
* Ensamblar los tres modelos equivalentes y recalibrar periódicamente con datos recientes.

## Reproducibilidad

* **Semilla global:** `SEMILLA = 42` en `e3/config.py`, propagada a NumPy, scikit-learn, XGBoost, imbalanced-learn, Optuna y DEAP.
* **Entorno:** `entregable3/requirements.txt` (Python 3.9.25, versiones exactas).
* **Registro de experimentos:** `resultados/tabla_maestra.csv` (420 filas: combinación, pliegue, semilla, hiperparámetros, métricas
  externas, evaluaciones y tiempos) y un archivo por tarea en `resultados/tareas/`.
* **Pasos** (desde `entregable3`):

```bash
python correr_experimento.py
```

```bash
python correr_analisis.py
```

```bash
cd construir && python construir_todo.py
```

Después se ejecutan los cuadernos de `cuadernos/` con `nbconvert` y se compila el libro con `jupyter-book build entregable3`.
