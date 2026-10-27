# Primer Entregable: base de datos, EDA y modelo base

**Predicción de brotes de dengue en Colombia con cuatro semanas de anticipación**

**Maestría en Matemáticas · Proyecto de Aprendizaje Automático · Daniel Serrano Romero**

Este libro establece las bases del proyecto: (i) la selección y documentación de la base de datos, (ii) un análisis
exploratorio exhaustivo que guía cada decisión de modelado, y (iii) un modelo base (regresión logística) comparado con
líneas base triviales. Todos los capítulos son cuadernos ejecutados de principio a fin; sus figuras son interactivas.

## Ruta de trabajo

El objetivo es categórico (brote / no brote en la semana $t+4$), así que se sigue la **ruta A (clasificación)** de la
guía. Como el dataset tiene tiempo y, tras agregar los centroides municipales del DANE, también coordenadas, se añaden
las secciones **2.6 (temporal), 2.7 (espacial) y 2.8 (espacio-temporal)** con la partición correspondiente: cronológica
para el futuro y por bloques espaciales para zonas nuevas.

**Dataset y problema** → **reserva de la prueba 2023–2024** (purga de 4 semanas) → **EDA** del objetivo, uni-, bi- y
multivariado → **auditoría de fuga** → componentes **temporal, espacial y espacio-temporal** → **preprocesamiento** en
`Pipeline` → **regresión logística frente a líneas base** → **nota crítica** y diagnóstico de residuos.

## Resumen

| Elemento | Resultado |
|---|---|
| Datos | Panel de 376 municipios × 782 semanas (2010–2024): 294 032 filas, 21 columnas, sin nulos ni duplicados |
| Objetivo | Brote en $t+4$ según el canal endémico; ≈ 10 % de brotes en el entrenamiento, con fuerte variación anual |
| Partición | Entrenamiento 2015–2021 · validación 2022 · prueba 2023–2024, con purga de 4 semanas en cada frontera |
| Hallazgos clave | La historia de casos domina; el clima actúa con rezago y de forma no lineal; El Niño cuadruplica el riesgo; autocorrelación espacial (Moran ≈ 0,31) e interacción espacio-tiempo (Knox) |
| Faltantes | La temperatura está imputada en el 64 % de las semanas (MAR estructural: municipios pequeños sin estación) |
| Modelo base | Regresión logística L2: AUC-ROC ≈ 0,885 y AUC-PR ≈ 0,81 en la prueba, frente a 0,79 / 0,59 de la persistencia y 0,50 / 0,30 del Dummy |

## Correspondencia con la guía del entregable

| Requisito de la guía | Dónde está |
|---|---|
| 1. Problema, justificación, fuente y licencia, diccionario, estructura | Capítulo 1 |
| 1. Tamaño de la muestra, $n/p$, casos por clase, entidades independientes | Capítulo 1 (incluye tamaño efectivo por ICC) |
| 1. Calidad: faltantes (patrón y mecanismo, prueba de Little), duplicados, atípicos, valores imposibles, sesgos | Capítulo 1 |
| 1. Consideraciones éticas | Capítulo 1 |
| Reservar la prueba antes de cualquier decisión | Capítulo 2 |
| 2.1 Variable objetivo | Capítulo 2 |
| 2.2 Análisis unidimensional | Capítulo 3 |
| 2.3 Análisis bidimensional (con tamaños de efecto, Holm y FDR, información mutua, VIF) | Capítulo 4 |
| 2.4 Análisis multivariado (PCA, t-SNE, Mahalanobis, Isolation Forest, agrupamiento) | Capítulo 5 |
| 2.5 Auditoría de fuga | Capítulo 6 |
| 2.6 Componente temporal (STL, ADF/KPSS, ACF/PACF, regímenes, deriva) | Capítulo 7 |
| 2.7 Componente espacial (CRS, mapas, Clark-Evans, Ripley, DBSCAN, Moran, LISA, Gi*, MAUP) | Capítulo 8 |
| 2.8 Componente espacio-temporal (focos emergentes, Knox) | Capítulo 9 |
| 2.9 Preprocesamiento en Pipeline | Capítulo 10 |
| 3. Modelo base, líneas base, métricas con IC, residuos, curva de aprendizaje, coeficientes | Capítulo 11 |
| Nota crítica (desempeño ≥ 80 %) | Capítulo 11 |
| Entregables: Jupyter Book, base de datos, dependencias y semillas | Conclusiones |
