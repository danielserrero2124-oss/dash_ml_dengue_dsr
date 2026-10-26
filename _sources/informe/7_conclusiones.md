# Conclusiones, limitaciones y preguntas

## Resultados en la prueba 2023–2024

| Modelo | AUC-ROC | AUC-PR | MCC | Brier ↓ | Ajuste |
|---|---|---|---|---|---|
| **HetDS en línea (original)** | **0,9017** | **0,8288** | **0,610** | **0,1117** | 78 s |
| GradientBoostingClassifier | 0,8977 | 0,8226 | 0,607 | 0,1142 | 284 s |
| RandomForestClassifier | 0,8976 | 0,8226 | 0,607 | 0,1144 | 173 s |
| LogisticRegression | 0,8854 | 0,8090 | 0,593 | 0,1218 | 1 s |
| KNeighborsClassifier | 0,8854 | 0,8030 | 0,583 | 0,1235 | 0,1 s |

## Significancia de la superioridad de HetDS

| HetDS vs. | ΔAUC-ROC | IC 95 % (bootstrap por municipio) | p (Holm) | ΔAUC-PR | p (Holm) |
|---|---|---|---|---|---|
| GradientBoosting | +0,0040 | [+0,0029; +0,0050] | 0,004 | +0,0062 | 0,004 |
| RandomForest | +0,0041 | [+0,0031; +0,0051] | 0,004 | +0,0062 | 0,004 |
| LogisticRegression | +0,0162 | [+0,0138; +0,0189] | 0,004 | +0,0198 | 0,004 |
| KNN | +0,0162 | [+0,0143; +0,0184] | 0,004 | +0,0258 | 0,004 |

* **DeLong:** todas las diferencias con $p < 10^{-24}$, en 2023 y en 2024 por separado.
* **Friedman–Nemenyi (100 semanas como bloques):** $\chi^2_F = 274{,}8$, $p \approx 3\times10^{-58}$; HetDS tiene el mejor rango
  medio (1,49) y su distancia a GradientBoosting (2,24) supera la diferencia crítica (0,61). Es el mejor en el 66 % de las semanas.
* **McNemar** (errores totales con el umbral de 2022 de cada modelo): es la única prueba que no favorece a HetDS. Frente a
  KNN, HetDS comete significativamente menos errores; frente a LogisticRegression y RandomForest no hay diferencia; y frente a
  GradientBoosting comete **más** errores totales ($b = 719$ frente a $c = 893$, $p < 0{,}001$). La razón es el umbral: el de HetDS
  (máximo F1) emite más alertas, así que anticipa más brotes (recall 0,808 frente a 0,771) a cambio de más falsas alarmas
  (precisión 0,675 frente a 0,696). McNemar mide exactitud con un umbral fijo; el AUC mide la calidad del ordenamiento del riesgo,
  que es donde HetDS es superior.

## Ablación: ¿de dónde viene la ventaja?

| Variante | AUC-ROC | Δ vs. GradientBoosting | p (bootstrap por municipio) |
|---|---|---|---|
| GradientBoosting (variables base) | 0,8977 | — | — |
| HetDS sin variables espaciales | 0,8994 | +0,0016 | 0,002 |
| GradientBoosting + variables espaciales | 0,9005 | +0,0028 | 0,002 |
| HetDS estático (pesos iguales) | 0,9011 | +0,0034 | — |
| **HetDS completo en línea** | **0,9017** | **+0,0040** | 0,002 |

Los dos componentes aportan y se suman: el ensamble heterogéneo en línea supera a GradientBoosting aun sin
variables espaciales, y la autocorrelación espacial añade información que ningún modelo recomendado tenía.

## Conclusiones

1. **HetDS en línea es el mejor modelo** y su superioridad es estadísticamente significativa frente a los cuatro
   modelos recomendados con la prueba más estricta (bootstrap por municipio con ajuste de Holm).
2. **El tamaño del efecto es moderado:** +0,004 de AUC-ROC y +0,006 de AUC-PR frente a los mejores recomendados,
   +0,016 frente a LR y KNN. Todos los modelos se acercan a un techo de ≈ 0,90 que fija la información disponible;
   la ventaja de HetDS es consistente (gana 2 de cada 3 semanas) más que grande.
3. **La historia de casos domina:** los casos recientes frente al umbral del canal endémico son el mejor predictor.
4. **El clima anticipa, pero ya está en los casos:** solo con clima el AUC es 0,71 (temperaturas de 25–28 °C, El Niño
   y lluvias previas aumentan el riesgo con 8–12 semanas de rezago), pero cuando ya se conocen los casos recientes el
   clima casi no añade (0,8968 → 0,8969).
5. **Robustez temporal:** el desempeño se mantiene en la epidemia récord de 2024, aunque la proporción de brotes
   triplica la del entrenamiento.

## Limitaciones

* **Teóricas:** Hedge garantiza seguir al mejor experto, no superar el techo de información; la vecindad espacial se
  aproxima por el departamento; ante cambios bruscos la ponderación reacciona con el retraso de 4 semanas.
* **Empíricas:** el 64 % de las semanas tiene temperatura imputada; el canal endémico depende del subregistro; solo
  hay dos años de prueba y 2024 fue excepcional; las filas no son independientes (por eso se usan pruebas por conglomerados).

## Trabajo futuro

* Vecindad espacial real (coordenadas, red de movilidad) en lugar del departamento.
* Datos climáticos sin imputar (estaciones IDEAM, reanálisis ERA5) y rezagos distribuidos no lineales.
* Serotipos circulantes e índices entomológicos como variables.
* Expertos reentrenados en línea y umbrales de alerta sensibles al costo de la respuesta.
* Evaluación prospectiva con secretarías de salud y monitoreo de deriva de los datos.
