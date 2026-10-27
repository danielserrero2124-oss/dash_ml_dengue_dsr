# Segundo Entregable: dashboard y modelos base

**Predicción de brotes de dengue en Colombia con cuatro semanas de anticipación**

**Maestría en Matemáticas · Proyecto de Aprendizaje Automático · Daniel Serrano Romero**

Este entregable tiene dos partes:

1. **Un dashboard en Dash + Plotly** con tres pestañas: **Contexto del problema**, **EDA** y **Modelos base** (regresión logística y
   SVM lineal). Cada resultado del dashboard trae su propia interpretación.
2. **Este Jupyter Book**, el informe del desarrollo: un capítulo por pestaña con las mismas figuras del dashboard (importadas del mismo
   código), su interpretación y los **detalles que no aparecen en el dashboard**: fórmulas de las pruebas, tablas completas, código de
   los Pipelines, el escalado de Platt de la SVM, la comparación de las funciones de pérdida y la arquitectura de la aplicación.

## Resumen de resultados

| Elemento | Resultado |
|---|---|
| Datos | 376 municipios × 782 semanas (2010–2024); temperatura imputada en el 64 % de las semanas (MAR estructural) |
| Objetivo | Brote en $t+4$ según el canal endémico; ≈ 10 % de brotes en entrenamiento, del 1,7 % al 25 % según el año |
| EDA | La historia de casos domina; el clima actúa con rezago y de forma no lineal; El Niño cuadruplica el riesgo; Moran ≈ 0,31 y Knox significativo |
| Modelos base | Regresión logística: AUC-ROC 0,885 y AUC-PR 0,809 · SVM lineal: 0,887 y 0,808 (prueba 2023–2024) |
| Líneas base | Persistencia de 8 semanas 0,859 / 0,728 · persistencia 0,787 / 0,588 · Dummy 0,500 / 0,303 |
| Diagnóstico | Sin sobreajuste; residuos con autocorrelación temporal y espacial; probabilidades subestimadas por el cambio de prevalencia |

## Cómo leer este libro

| Capítulo | Pestaña del dashboard |
|---|---|
| 1. Contexto del problema | Contexto del problema |
| 2. EDA I: datos, objetivo y relaciones | EDA › Datos y calidad, Variable objetivo, Univariado, Bivariado, Multivariado, Fuga de datos |
| 3. EDA II: tiempo, espacio y preprocesamiento | EDA › Temporal, Espacial, Espacio-temporal, Preprocesamiento |
| 4. Modelos base | Modelos base (todas las sub-pestañas) |
| 5. El dashboard | Arquitectura, componentes interactivos y ejecución |
| 6. Conclusiones | Hallazgos, limitaciones y reproducibilidad |

El análisis completo paso a paso (todas las pruebas con su código) está en el libro del **Primer Entregable**, del que este
entregable reutiliza el código (`entregable1/e1`).
