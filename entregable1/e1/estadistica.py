"""Herramientas estadísticas del EDA: faltantes (Little), asociación (V de Cramér, información mutua, tamaños de
efecto), multicolinealidad (VIF), corrección por comparaciones múltiples, deriva (PSI) y puntos de cambio.

Con cientos de miles de filas casi toda diferencia resulta «significativa»; por eso cada prueba se reporta junto
con un tamaño de efecto que dice si la diferencia importa.
"""

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from scipy import stats  # Distribuciones y pruebas


# ---------------------------------------------------------------------------
# Faltantes: prueba MCAR de Little (1988) con estimación EM de la media y la covarianza
# ---------------------------------------------------------------------------
def _em_normal(X, iteraciones=200, tol=1e-6):  # Media y covarianza de máxima verosimilitud con faltantes (algoritmo EM)
    n, p = X.shape  # Filas y columnas
    mu = np.nanmean(X, axis=0)  # Media inicial con los valores observados
    S = np.diag(np.nanvar(X, axis=0))  # Covarianza inicial diagonal
    faltante = np.isnan(X)  # Máscara de faltantes
    patrones = {}  # Filas agrupadas por patrón de faltantes
    for i, fila in enumerate(map(tuple, faltante)):  # Recorre las filas
        patrones.setdefault(fila, []).append(i)  # Agrupa
    for _ in range(iteraciones):  # Iteraciones EM
        suma_x = np.zeros(p)  # Suma de los valores esperados
        suma_xx = np.zeros((p, p))  # Suma de los productos esperados
        for patron, filas in patrones.items():  # Cada patrón comparte la misma estructura
            m = np.array(patron)  # Columnas faltantes
            o = ~m  # Columnas observadas
            Xp = X[filas].copy()  # Filas del patrón
            C = np.zeros((p, p))  # Covarianza condicional de los faltantes
            if m.any():  # Paso E: esperanza condicional de los faltantes dado lo observado
                B = S[np.ix_(m, o)] @ np.linalg.inv(S[np.ix_(o, o)])  # Coeficientes de regresión
                Xp[:, m] = mu[m] + (Xp[:, o] - mu[o]) @ B.T  # Valores esperados
                C[np.ix_(m, m)] = S[np.ix_(m, m)] - B @ S[np.ix_(o, m)]  # Varianza que no explica lo observado
            suma_x += Xp.sum(axis=0)  # Acumula
            suma_xx += Xp.T @ Xp + len(filas) * C  # Acumula
        mu_nueva = suma_x / n  # Paso M: nueva media
        S_nueva = suma_xx / n - np.outer(mu_nueva, mu_nueva)  # Paso M: nueva covarianza
        listo = np.abs(mu_nueva - mu).max() < tol  # Convergencia
        mu, S = mu_nueva, S_nueva  # Actualiza
        if listo:  # Si convergió
            break  # Termina
    return mu, S, patrones  # Estimaciones y patrones


def prueba_little(X):  # Prueba MCAR de Little: H0 = los faltantes son completamente al azar
    """d² = Σ_j n_j (x̄_j − μ_j)ᵀ Σ_j⁻¹ (x̄_j − μ_j) ~ χ²(Σ_j p_j − p). Devuelve (d², gl, valor p, n.º de patrones)."""
    X = np.asarray(X, dtype=float)  # Matriz numérica
    mu, S, patrones = _em_normal(X)  # Estimaciones EM
    d2, gl = 0.0, 0  # Estadístico y grados de libertad
    for patron, filas in patrones.items():  # Cada patrón
        o = ~np.array(patron)  # Columnas observadas
        dif = X[np.ix_(filas, o)].mean(axis=0) - mu[o]  # Media del patrón frente a la media global
        d2 += len(filas) * dif @ np.linalg.inv(S[np.ix_(o, o)]) @ dif  # Distancia de Mahalanobis ponderada
        gl += int(o.sum())  # Variables observadas en el patrón
    gl -= X.shape[1]  # Resta p
    return float(d2), gl, float(stats.chi2.sf(d2, gl)), len(patrones)  # Resultado


def diferencia_estandarizada(a, b):  # Tamaño de efecto entre dos grupos (d de Cohen con desviación combinada)
    a, b = np.asarray(a, float), np.asarray(b, float)  # Vectores
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)  # Desviación combinada
    return float((a.mean() - b.mean()) / s) if s > 0 else 0.0  # |d| ≥ 0,1 suele considerarse un desbalance relevante


# ---------------------------------------------------------------------------
# Asociación y tamaños de efecto
# ---------------------------------------------------------------------------
def v_cramer(x, y):  # V de Cramér con corrección de sesgo (Bergsma, 2013) y prueba χ²
    tabla = pd.crosstab(x, y)  # Tabla de contingencia
    chi2, p, gl, _ = stats.chi2_contingency(tabla, correction=False)  # Prueba χ² de independencia
    n = tabla.to_numpy().sum()  # Total
    r, k = tabla.shape  # Filas y columnas
    phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))  # φ² corregido
    rc, kc = r - (r - 1) ** 2 / (n - 1), k - (k - 1) ** 2 / (n - 1)  # Dimensiones corregidas
    return float(np.sqrt(phi2 / max(min(rc - 1, kc - 1), 1e-12))), float(chi2), int(gl), float(p)  # (V, χ², gl, p)


def mann_whitney(a, b):  # Mann-Whitney U con correlación biserial de rangos como tamaño de efecto
    u, p = stats.mannwhitneyu(a, b, alternative="two-sided")  # Prueba no paramétrica
    return float(2 * u / (len(a) * len(b)) - 1), float(p)  # r_rb ∈ [−1, 1] (positivo si a tiende a ser mayor que b) y valor p


def kruskal(grupos):  # Kruskal-Wallis con ε² como tamaño de efecto
    h, p = stats.kruskal(*grupos)  # Prueba no paramétrica de k grupos
    n = sum(len(g) for g in grupos)  # Tamaño total
    return float(h), float(p), float(h / (n - 1))  # (H, p, ε²); ε² ≈ 0,01 pequeño, 0,08 mediano, 0,26 grande


def holm(p):  # Valores p ajustados por Holm (controla la tasa de error de la familia)
    p = np.asarray(p, float)  # Vector
    orden = np.argsort(p)  # De menor a mayor
    m = len(p)  # Número de pruebas
    ajust = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * p[orden]))  # Multiplica y fuerza monotonía
    salida = np.empty(m)  # Vuelve al orden original
    salida[orden] = ajust  # Reordena
    return salida  # p ajustados


def fdr_bh(p):  # Valores q de Benjamini-Hochberg (controla la tasa de falsos descubrimientos)
    p = np.asarray(p, float)  # Vector
    m = len(p)  # Número de pruebas
    orden = np.argsort(p)[::-1]  # De mayor a menor
    q = np.minimum(1, np.minimum.accumulate(p[orden] * m / (m - np.arange(m))))  # p·m/rango, monótono
    salida = np.empty(m)  # Orden original
    salida[orden] = q  # Reordena
    return salida  # q


def vif(X):  # Factor de inflación de la varianza: VIF_j = 1 / (1 − R²_j)
    """R²_j es el de la regresión de la variable j sobre las demás; se obtiene de la diagonal de la inversa de la
    matriz de correlación: VIF_j = [R⁻¹]_jj."""
    R = np.corrcoef(np.asarray(X, float), rowvar=False)  # Matriz de correlación
    return np.diag(np.linalg.pinv(R))  # VIF de cada variable (> 10: multicolinealidad severa)


# ---------------------------------------------------------------------------
# Deriva y cambios de régimen
# ---------------------------------------------------------------------------
def psi(referencia, actual, cortes=10):  # Índice de estabilidad poblacional entre dos periodos
    """PSI = Σ (a_i − r_i) ln(a_i / r_i) con cortes por deciles de la referencia. < 0,1 estable; 0,1–0,25 moderada; > 0,25 fuerte."""
    referencia, actual = np.asarray(referencia, float), np.asarray(actual, float)  # Vectores
    bordes = np.unique(np.quantile(referencia, np.linspace(0, 1, cortes + 1)))  # Deciles de la referencia
    bordes[0], bordes[-1] = -np.inf, np.inf  # Abre los extremos
    r = np.histogram(referencia, bordes)[0] / len(referencia)  # Proporción por intervalo (referencia)
    a = np.histogram(actual, bordes)[0] / len(actual)  # Proporción por intervalo (actual)
    r, a = np.clip(r, 1e-4, None), np.clip(a, 1e-4, None)  # Evita log(0)
    return float(np.sum((a - r) * np.log(a / r)))  # PSI


def segmentacion_binaria(x, penalizacion=None, min_tam=26):  # Puntos de cambio en la media (segmentación binaria)
    """Divide recursivamente la serie donde más se reduce la suma de cuadrados; acepta el corte si la reducción
    supera la penalización (por defecto, tipo BIC: 2·σ²·ln n). Devuelve los índices de los puntos de cambio."""
    x = np.asarray(x, float)  # Serie
    n = len(x)  # Longitud
    sigma2 = np.var(np.diff(x)) / 2  # Varianza del ruido estimada con diferencias (robusta a cambios de nivel)
    pen = 2 * sigma2 * np.log(n) if penalizacion is None else penalizacion  # Penalización
    cortes = []  # Puntos de cambio encontrados

    def costo(a, b):  # Suma de cuadrados del segmento [a, b)
        s = x[a:b]  # Segmento
        return float(((s - s.mean()) ** 2).sum())  # Costo

    def dividir(a, b):  # Busca el mejor corte en [a, b)
        if b - a <= 2 * min_tam:  # Segmento demasiado corto para dos partes de min_tam
            return  # No divide
        base = costo(a, b)  # Costo sin dividir
        mejores = [(base - costo(a, k) - costo(k, b), k) for k in range(a + min_tam, b - min_tam)]  # Ganancia de cada corte
        ganancia, k = max(mejores)  # Mejor corte
        if ganancia > pen:  # Si compensa la penalización
            cortes.append(k)  # Guarda
            dividir(a, k)  # Izquierda
            dividir(k, b)  # Derecha

    dividir(0, n)  # Toda la serie
    return sorted(cortes)  # Índices ordenados


def mann_kendall(x):  # Prueba de tendencia monótona de Mann-Kendall
    x = np.asarray(x, float)  # Serie
    n = len(x)  # Longitud
    s = sum(np.sign(x[j] - x[i]) for i in range(n - 1) for j in range(i + 1, n))  # Estadístico S
    var = n * (n - 1) * (2 * n + 5) / 18  # Varianza de S sin empates
    z = (s - np.sign(s)) / np.sqrt(var) if s != 0 else 0.0  # z con corrección de continuidad
    return float(s), float(z), float(2 * stats.norm.sf(abs(z)))  # (S, z, p)
