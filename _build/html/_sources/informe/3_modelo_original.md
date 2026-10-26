# Fundamentos teóricos del modelo original: HetDS en línea

## Origen y motivación

Del listado de modelos del curso se eligió **HetDS — selección dinámica de ensambles heterogéneos** (grupo
«Árboles, boosting y ensambles propuestos»). La selección dinámica parte de una idea sencilla: ningún
clasificador es el mejor en todas las regiones del espacio ni en todos los momentos, así que conviene
combinar varios **expertos** y decidir, caso a caso, cuánto confiar en cada uno (Ko, Sabourin y Britto, 2008;
Cruz, Sabourin y Cavalcanti, 2018).

En la vigilancia del dengue el «momento» importa: el régimen epidémico cambia entre años (2024 triplicó la
proporción de brotes del entrenamiento). Por eso la competencia de cada experto se mide **en línea**, con las
etiquetas que se van confirmando, y se combina con el algoritmo **Hedge** de pesos exponenciales
(Freund y Schapire, 1997), que tiene garantías teóricas de desempeño.

El modelo agrega un componente de **autocorrelación espacial**: el dengue se propaga entre municipios vecinos
porque comparten clima, vectores y movilidad de personas.

## Formulación

**Expertos.** $K = 5$ clasificadores heterogéneos entrenados con 2015–2021:

| Experto | Sesgo inductivo |
|---|---|
| Boosting de árboles (31 hojas) | Interacciones complejas, corrección secuencial de errores |
| Boosting de árboles (7 hojas, $L_2$) | Efectos más suaves y regularizados |
| Bosque aleatorio | Promedio de árboles profundos: baja varianza |
| Árboles extremos | Cortes aleatorios: máxima diversidad |
| Regresión logística | Frontera lineal: extrapola fuera del rango de entrenamiento |

Cada experto recibe las variables del problema $\mathbf{x}_{m,t}$ y las espaciales

$$
\mathbf{s}_{m,t} = \Big(\ln\big(1 + \textstyle\sum_{j \in D(m),\, j\ne m} c_{j,t}\big),\;
\Delta_4 \ln\big(1 + \textstyle\sum_{j \in D(m),\, j\ne m} c_{j,t}\big),\;
\bar b_{D(m)\setminus m,\,t}\Big),
$$

donde $D(m)$ es el departamento del municipio $m$: casos del resto del departamento, su cambio en cuatro
semanas y la proporción de municipios vecinos en brote. Cada experto produce $p_k(m,t) = f_k(\mathbf{x}_{m,t}, \mathbf{s}_{m,t})$.

**Pérdida con retraso de la etiqueta.** El brote de la semana $s$ se confirma en $s + r$ ($r = 4$). Para predecir
la semana $t$ solo se usan las filas con $s \le t - r$. La pérdida de cada experto en la ventana
$W(t) = \{s : t - r - V < s \le t - r\}$ es

$$
L_k(t) = \sum_{s\in W(t)}\sum_m \ell\big(y_{m,s}, p_k(m,s)\big), \qquad
\ell(y,p) = -\big[y\ln p + (1-y)\ln(1-p)\big].
$$

**Pesos Hedge.**

$$
w_k(t) = \frac{\exp\big(-\eta L_k(t)\big)}{\sum_{j=1}^K \exp\big(-\eta L_j(t)\big)} .
$$

**Combinación en el espacio logit.**

$$
P\big(y_{m,t}=1\big) = \sigma\Big(\sum_{k=1}^K w_k(t)\,\operatorname{logit}p_k(m,t)\Big), \qquad \sigma(z) = \frac{1}{1+e^{-z}} .
$$

## Función objetivo y principio de aprendizaje

* **Expertos:** cada uno minimiza el riesgo empírico con log-loss y su propia regularización (profundidad,
  hojas mínimas, $L_2$).
* **Selector:** los pesos Hedge son la solución exacta del problema *Follow the Regularized Leader* con
  regularización entrópica sobre el símplex $\Delta_K$:

$$
\mathbf{w}(t) = \arg\min_{\mathbf{w}\in\Delta_K}\; \eta\,\langle\mathbf{w}, \mathbf{L}(t)\rangle + \sum_k w_k\ln w_k .
$$

* **Garantía (arrepentimiento):** con pérdidas acotadas en $[0, B]$ (las probabilidades se recortan a
  $[10^{-5}, 1-10^{-5}]$), la pérdida acumulada de Hedge excede la del mejor experto en, a lo sumo,
  $B\sqrt{(T/2)\ln K}$ para $\eta$ adecuado (Cesa-Bianchi y Lugosi, 2006).

## Supuestos

1. **Retraso conocido de las etiquetas:** nunca se usa un brote antes de que esté confirmado.
2. **Cambios de régimen graduales:** el experto que funcionó en las últimas semanas sigue siendo útil.
3. **Diversidad:** los expertos cometen errores distintos (si fallaran igual, ponderarlos no ayudaría).
4. **Dependencia espacial dentro del departamento.**

## Regularización, complejidad y generalización

| Elemento | Papel |
|---|---|
| $\eta$ | $\eta\to0$: promedio fijo (estable); $\eta$ grande: sigue al último líder (adaptativo) |
| Ventana $V$ | Pocas semanas: reacción rápida y ruidosa; muchas: más suave |
| Regularizador entrópico | Evita pesos extremos (equivale a un prior uniforme sobre los expertos) |
| Expertos | Cada uno con su regularización interna |

* **Costo:** entrenar = suma de los cinco expertos; predecir en línea = $O(K)$ por fila más una suma acumulada por semana.
* **Hiperparámetros:** $\eta \in \{0{,}01; 0{,}05; 0{,}2\}$ y $V \in \{4, 8, 26\}$ semanas se eligen por AUC-PR en 2022.
* **Generalización:** se evalúa en 2023–2024, años nunca usados para entrenar ni para elegir hiperparámetros.

## Referencias

* Cruz, R. M. O., Sabourin, R. y Cavalcanti, G. D. C. (2018). Dynamic classifier selection: Recent advances and perspectives. *Information Fusion*, 41, 195–216. <https://doi.org/10.1016/j.inffus.2017.09.010>
* Ko, A. H. R., Sabourin, R. y Britto, A. S. (2008). From dynamic classifier selection to dynamic ensemble selection. *Pattern Recognition*, 41(5), 1718–1731. <https://doi.org/10.1016/j.patcog.2007.10.015>
* Freund, Y. y Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. *Journal of Computer and System Sciences*, 55(1), 119–139. <https://doi.org/10.1006/jcss.1997.1504>
* Cesa-Bianchi, N. y Lugosi, G. (2006). *Prediction, Learning, and Games*. Cambridge University Press.
* Huber, J. H. et al. (2018). Seasonal temperature variation influences climate suitability for dengue, chikungunya, and Zika transmission. *PLOS Neglected Tropical Diseases*, 12(5), e0006451. <https://doi.org/10.1371/journal.pntd.0006451>
