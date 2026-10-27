# Conclusiones y reproducibilidad

## Conclusiones

1. **Los dos modelos base lineales son equivalentes.** La regresión logística (AUC-ROC 0,885, AUC-PR 0,809) y la SVM lineal (0,887 y
   0,808) llegan casi al mismo hiperplano con las mismas variables: la diferencia en AUC-ROC (+0,001) es estadísticamente detectable pero
   irrelevante en la práctica, y en AUC-PR no hay diferencia.
2. **Superan con claridad a las líneas base triviales.** Frente a la persistencia de 8 semanas ganan ≈ 0,027 de AUC-ROC y ≈ 0,08 de
   AUC-PR (intervalos por municipio que excluyen el 0); frente a la persistencia simple, ≈ 0,10 y ≈ 0,22.
3. **La exactitud no sirve para juzgarlos.** Con ≈ 30 % de brotes en la prueba, el Dummy acierta 70 % y la persistencia 82 %.
4. **La historia de casos domina** y lo que anticipa un brote es su aceleración reciente; El Niño aporta; la temperatura local casi no
   aporta en un modelo lineal (64 % imputada y relación no lineal).
5. **No hay sobreajuste; el límite es de sesgo.** Curva de aprendizaje plana, brecha pequeña entre entrenamiento y prueba, pero residuos
   con autocorrelación temporal y espacial: falta la dinámica de vecindad (Knox, Moran) y la no linealidad del clima.
6. **Generalizan a zonas nuevas** (bloques de departamentos con 50 km de separación) porque sus variables son relativas a cada municipio.
7. **Las probabilidades se desplazan con la prevalencia.** En 2023–2024 quedan subestimadas: el umbral o la intersección deben
   recalibrarse con datos recientes.

## Limitaciones

* Representatividad: 376 de 1 122 municipios (≈ 60 % de la población), con el Caribe y la Orinoquía sobrerrepresentados.
* Subregistro variable de la notificación y caída de 2020 por la pandemia.
* Temperatura imputada en los municipios pequeños; coordenadas como centroides, sin polígonos.
* Dos años de prueba, uno de ellos (2024) excepcional.

## Siguientes pasos

* Variables de rezago espacial (vecinos en $t$) y de interacción con El Niño.
* Modelos no lineales y ensambles; recalibración periódica del umbral.

## Reproducibilidad

* **Semilla:** `SEMILLA = 42` (`src/config.py`).
* **Entorno:** Python 3.9.25 con las versiones de `entregable2/requirements.txt`.
* **Pasos** (desde `entregable2`):

```bash
python precalculo.py
```

```bash
python -m tablero.app
```

```bash
cd construir && python construir_todo.py && cd ../cuadernos
```

```bash
for n in *.ipynb; do jupyter nbconvert --to notebook --execute --inplace $n; done
```

```bash
cd ../.. && jupyter-book build entregable2
```
