# Predicción de brotes de dengue en Colombia: informe del desarrollo

**Proyecto de Aprendizaje Automático · Alerta temprana a 4 semanas según las condiciones climáticas**

Este libro acompaña al dashboard interactivo. Documenta la formulación del problema, el análisis exploratorio,
los fundamentos del modelo original y, sobre todo, **la ejecución real de los modelos**: los cuadernos de los
capítulos 2, 4 y 5 se ejecutaron completos y sus resultados son los mismos que muestra el dashboard.

## Resumen

| Elemento | Resultado |
|---|---|
| Problema | Clasificación binaria: ¿el municipio estará en brote dentro de 4 semanas? |
| Definición de brote | Canal endémico: casos > tercer cuartil de las mismas semanas de los 5 años previos (y ≥ 3 casos) |
| Datos | 376 municipios × 782 semanas (2010–2024): casos, temperatura, lluvia, NDVI, ONI, población y flujo |
| Validación | Entrenamiento 2015–2021 (validación cruzada hacia adelante), validación 2022, prueba 2023–2024 |
| Modelos recomendados | LogisticRegression (AUC 0,885), RandomForest (0,898), KNN (0,885), GradientBoosting (0,898) |
| Modelo original | **HetDS en línea**: AUC-ROC **0,902**, AUC-PR **0,829** en 2023–2024 |
| Significancia | Supera a los cuatro en AUC con bootstrap por municipio (p = 0,004, Holm), DeLong y Friedman–Nemenyi; McNemar (errores con el umbral) no lo favorece frente a GB |
| Aporte del clima | Solo con clima AUC ≈ 0,71; con la historia de casos ya incluida, el clima casi no añade |

## Cómo leer este informe

| Capítulo | Contenido | Tipo |
|---|---|---|
| 1. Problema y formulación matemática | Importancia, objetivos, datos, tipo de tarea y ecuaciones | Texto |
| 2. EDA, datos y preprocesamiento | Figuras del dashboard con su interpretación | Cuaderno ejecutado |
| 3. Fundamentos teóricos del modelo original | HetDS, Hedge, FTRL, garantía de arrepentimiento, supuestos | Texto |
| 4. Entrenamiento y resultados | `GridSearchCV` de los cuatro recomendados y HetDS en línea | Cuaderno ejecutado |
| 5. Pruebas de significancia, ablación y tiempos | DeLong, bootstrap por municipio, McNemar, Friedman–Nemenyi | Cuaderno ejecutado |
| 6. El dashboard | Ejecución, estructura según la rúbrica y arquitectura | Texto |
| 7. Conclusiones | Hallazgos, limitaciones, trabajo futuro y preguntas | Texto |

Las figuras son las mismas del dashboard y conservan la interactividad: al pasar el cursor se muestran los valores.
