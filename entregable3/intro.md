# Tercer Entregable: modelos de Machine Learning, optimización y comparación estadística

**Predicción de brotes de dengue en Colombia con cuatro semanas de anticipación**

**Maestría en Matemáticas · Proyecto de Aprendizaje Automático · Daniel Serrano Romero**

Este entregable entrena, optimiza y compara **140 modelos** con rigor estadístico y eficiencia computacional:

* **Clasificación** (¿habrá brote en $t+4$?): 7 modelos × 4 estrategias de balanceo × 4 optimizadores = **112**.
* **Regresión** ($\log(1 + \text{casos})$ en $t+4$): 7 modelos × 4 optimizadores = **28**.

Cada combinación se evalúa con **validación cruzada anidada por años** (3 pliegues externos, 2 internos, con purga de 4 semanas), se
reajusta con 2015–2021 y se evalúa una sola vez en la prueba 2023–2024. Todos los cuadernos son ejecutables de principio a fin.

## Resumen de resultados

| Pregunta | Respuesta |
|---|---|
| Mejor clasificador | XGBoost sin balanceo (optimizado con el algoritmo genético): AUC-PR 0,823 y AUC-ROC 0,898 en la prueba; supera al modelo base logístico (0,809) |
| ¿Sirve el balanceo? | No mejora la discriminación (ADASYN −0,024, SMOTE −0,010, `class_weight` −0,005 de AUC-PR) y deteriora la calibración (ECE de 0,07 a 0,14–0,18); el umbral elegido en 2022 maneja mejor el desbalance |
| Mejor regresor | XGBoost: RMSE 0,573 y $R^2$ 0,79 en la prueba, 13 % menos error que la persistencia |
| Optimizadores | Con 20 evaluaciones alcanzan desempeños externos casi iguales; la búsqueda bayesiana (TPE) logra la mejor curva de desempeño en cualquier momento y Successive Halving la mejor calidad por minuto |
| Costo | 420 tareas anidadas, 8 250 evaluaciones de hiperparámetros y 16 500 ajustes en 105 minutos con 10 procesos |

## Cómo leer este libro

| Capítulo | Contenido | Sección de la guía |
|---|---|---|
| 1. Datos | ETL y EDA mejorado: objetivo de regresión y efecto del balanceo | 7.1 |
| 2. Diseño experimental | Pipeline combinatorio, espacios de búsqueda, validación anidada, teoría de los optimizadores, tabla maestra | 2, 3.2, 3.3, 7.3 |
| 3. Optimización | Métricas externas, tiempos, presupuesto, curvas *anytime*, diversidad genética, estabilidad y multi-fidelidad | 3.3, 3.4 |
| 4. Optimización computacional | Complejidad teórica frente a empírica, KNN/FAISS, SAGA, `partial_fit`, XGBoost `hist`, SVM, paralelización, perfilamiento | 4 |
| 5. Clasificación | Bucle externo, balanceo, prueba, matrices de confusión, ROC y calibración | 5.1 |
| 6. Regresión | RMSE, MAE, ajuste por subconjunto y pruebas de residuos (White, BDS, Ljung-Box) | 5.2 |
| 7. Interpretabilidad | SHAP global y local; comparación con LIME en XGBoost | 5.3 |
| 8. Robustez y estadística | Media ± desviación, semillas, Friedman, Nemenyi y diagrama CD, DeLong + Holm, BCa, Cliff; MCS, Giacomini-White, Clark-West, DM-HLN, SPA | 5.4, 6 |
| 9. Conclusiones | Hallazgos, limitaciones y reproducibilidad | 9 |
