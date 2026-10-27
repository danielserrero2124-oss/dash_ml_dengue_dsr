# Conclusiones, limitaciones y reproducibilidad

## Hallazgos del EDA que guían el modelado

| Hallazgo | Evidencia | Decisión |
|---|---|---|
| La historia de casos domina la predicción | AUC univariados de 0,77–0,88; información mutua máxima; concepto estable año a año | Rezagos y medias móviles del log de casos; persistencia como línea base exigente |
| El clima actúa con rezago y de forma no lineal | Correlación cruzada débil con máximo a 12–15 semanas (lluvia, NDVI); riesgo máximo hacia 25–28 °C | Medias móviles largas y anomalías; modelos no lineales en los siguientes entregables |
| El Niño eleva el riesgo | Brote en 4 semanas: 16,8 % en El Niño frente a 4,5 % en La Niña (V de Cramér 0,17) | ONI actual y rezagado 8 semanas |
| Desbalance que cambia por año | Prevalencia del 1,7 % (2017) al 25 % (2019); 30 % en la prueba | AUC-PR, MCC y Brier; validación por años; umbral en validación |
| Estacionalidad débil, ondas de varios años | Fuerza estacional 0,02 frente a 0,66 de la tendencia (STL) | Seno y coseno de la semana; variables relativas al municipio |
| Autocorrelación espacial | Moran ≈ 0,31 (incidencia), 76 municipios Alto-Alto, alcance ≈ 100 km | Validación por bloques de departamentos con 50 km de separación |
| Interacción espacio-tiempo | Knox: 13–28 % más pares cercanos en espacio y tiempo de lo esperado (p = 0,001) | Variables de vecindad rezagadas para los siguientes modelos |
| Temperatura imputada en el 64 % | MAR estructural: municipios pequeños sin estación (d ≈ −0,7 en población) | Indicador `temp_missing`; no se reimputa con vecinos |
| Atípicos = epidemias | Atípicos multivariados con tasa de brote de ≈ 39 % frente a ≈ 9 % | No se eliminan; logaritmo |
| Multicolinealidad | VIF > 10 entre medias móviles | Regularización L2 |

## Modelo base

La regresión logística con penalización L2 ($C = 0{,}01$, sin ponderar clases) alcanza en la prueba 2023–2024 un
**AUC-ROC de 0,885** y un **AUC-PR de 0,81**, con intervalos de confianza por municipio de unas ± 0,01, y supera de
forma significativa a la mejor línea base trivial (persistencia de 8 semanas: 0,86 y 0,73). La exactitud (≈ 0,83) no
es informativa: el Dummy llega a 0,70 y la persistencia simple a 0,82.

El diagnóstico marca el camino de los siguientes entregables:

* **Sin sobreajuste y con muestra suficiente**: curva de aprendizaje plana y brecha pequeña entre entrenamiento y prueba.
* **Sesgo del modelo**: los residuos conservan autocorrelación temporal (rachas de error) y espacial (I de Moran
  significativo), y el clima tiene efectos no lineales que un modelo lineal no captura.
* **Cambio de prevalencia**: el modelo ordena bien el riesgo en 2023–2024, pero sus probabilidades quedan
  subestimadas; hay que recalibrar el umbral o la intersección con datos recientes.
* **Generaliza a zonas nuevas**: con bloques de departamentos y 50 km de separación, el desempeño se mantiene.

## Limitaciones

* **Representatividad.** 376 de 1 122 municipios (≈ 60 % de la población), con sobrerrepresentación del Caribe y la
  Orinoquía; los resultados aplican a municipios con transmisión habitual.
* **Calidad de la notificación.** Los casos son los notificados al SIVIGILA; hay subregistro variable y caídas como la de 2020
  por la pandemia.
* **Clima imputado.** La temperatura falta en los municipios pequeños; el efecto real de la temperatura puede estar
  atenuado.
* **Escala espacial.** Las coordenadas son centroides; no hay información dentro de cada municipio ni polígonos para
  vecindades por contigüidad.
* **Pocos años de prueba.** Dos años, uno de ellos (2024) excepcional.
* **Riesgos éticos.** Datos agregados sin información personal; el riesgo de reidentificación es bajo. El principal
  riesgo es de equidad: si la notificación es peor en municipios pobres, el modelo puede subestimar su riesgo.

## Reproducibilidad

* **Semilla:** `SEMILLA = 42` en `src/config.py`, usada en todos los muestreos, permutaciones, bootstrap y modelos.
* **Entorno:** Python 3.9.25 con las versiones fijadas en `entregable1/requirements.txt`.
* **Código:** `entregable1/e1/` (datos, estadística, análisis espacial, modelo base y figuras) y `src/` (canal endémico y
  variables). Todo el código está comentado línea por línea.
* **Ejecución de principio a fin** (desde la raíz del proyecto):

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r entregable1/requirements.txt
cd entregable1/construir && ../../.venv/Scripts/python construir_todo.py && cd ../cuadernos
for n in *.ipynb; do ../../.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace $n; done
cd ../.. && .venv/Scripts/jupyter-book build entregable1
```

## Entregables

| Entregable | Ubicación |
|---|---|
| Informe en Jupyter Book | `entregable1/_build/html/index.html` |
| Base de datos | `data/final_dataset.csv` (panel) y `data/divipola_municipios.csv` (centroides DANE, CC BY-SA 4.0) |
| Dependencias y semillas | `entregable1/requirements.txt` y `src/config.py` |
