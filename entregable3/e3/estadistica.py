"""Comparación estadística jerárquica de modelos.

Clasificación: Friedman (ómnibus) → Nemenyi y diagrama de Diferencia Crítica → DeLong en un subconjunto reducido con
corrección de Holm → intervalos bootstrap BCa por municipio → tamaño del efecto (delta de Cliff).

Regresión: Model Confidence Set (MCS) → Giacomini-White / Clark-West entre finalistas → Diebold-Mariano con corrección
HLN y bootstrap estacionario → tamaño del efecto (d de Cohen); además SPA de Hansen frente a un benchmark.
"""

import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from arch.bootstrap import MCS, SPA, StationaryBootstrap  # Pruebas basadas en bootstrap para series
from scipy import stats  # Distribuciones

from src.modelado import prueba_delong  # DeLong (implementación del proyecto final)


# ---------------------------------------------------------------------------
# Clasificación
# ---------------------------------------------------------------------------
def friedman(matriz, mayor_mejor=True):
    """Prueba de Friedman sobre una matriz bloques × modelos.

    Parameters
    ----------
    matriz : pandas.DataFrame
        Filas = bloques (por ejemplo, semanas de la prueba); columnas = modelos; valores = métrica.
    mayor_mejor : bool
        Si True, el rango 1 es el valor más alto.

    Returns
    -------
    dict
        Estadístico χ²_F, grados de libertad, valor p, estadístico F de Iman-Davenport y rangos medios.
    """
    R = (-matriz if mayor_mejor else matriz).rank(axis=1, method="average")  # Rangos dentro de cada bloque
    N, k = R.shape  # Bloques y modelos
    medios = R.mean()  # Rango medio de cada modelo
    chi2 = 12 * N / (k * (k + 1)) * (np.sum(medios ** 2) - k * (k + 1) ** 2 / 4)  # Estadístico de Friedman
    F = (N - 1) * chi2 / (N * (k - 1) - chi2) if N * (k - 1) > chi2 else np.inf  # Corrección de Iman-Davenport (menos conservadora)
    return {"chi2": float(chi2), "gl": k - 1, "p": float(stats.chi2.sf(chi2, k - 1)), "F": float(F),
            "p_F": float(stats.f.sf(F, k - 1, (k - 1) * (N - 1))) if np.isfinite(F) else 0.0, "N": N, "k": k,
            "rangos": medios.sort_values()}


def diferencia_critica(k, N, alfa=0.05):
    """Diferencia crítica de Nemenyi: CD = q_α · √(k(k+1)/(6N)), con q_α el cuantil del rango estudentizado / √2."""
    q = stats.studentized_range.ppf(1 - alfa, k, np.inf) / np.sqrt(2)  # Valor crítico (grados de libertad infinitos)
    return float(q * np.sqrt(k * (k + 1) / (6 * N)))  # Diferencia crítica


def grupos_cd(rangos, cd):
    """Grupos maximales de modelos cuya diferencia de rango medio es menor que la CD (las barras del diagrama)."""
    r = rangos.sort_values()  # Rangos ordenados
    nombres, valores = list(r.index), r.to_numpy()  # Listas
    grupos = []  # Barras
    for i in range(len(valores)):  # Desde cada modelo
        j = i  # Extiende mientras la diferencia sea menor que la CD
        while j + 1 < len(valores) and valores[j + 1] - valores[i] < cd:
            j += 1  # Siguiente
        if j > i and not any(a <= i and j <= b for a, b in grupos):  # Grupo nuevo y no contenido en otro
            grupos.append((i, j))  # Agrega
    return [(nombres[a], nombres[b], valores[a], valores[b]) for a, b in grupos]  # (primero, último, rango inicial, rango final)


def delong_holm(y, puntajes, pares):
    """DeLong para un conjunto reducido de pares, con valores p crudos y ajustados por Holm.

    Parameters
    ----------
    y : array-like
        Objetivo.
    puntajes : dict
        {modelo: puntaje}.
    pares : list of (str, str)
        Pares a comparar.

    Returns
    -------
    pandas.DataFrame
        Diferencia de AUC, IC de DeLong, p crudo y p de Holm.
    """
    filas = []  # Resultados
    for a, b in pares:  # Cada par
        dif, (lo, hi), p = prueba_delong(np.asarray(y), np.asarray(puntajes[a]), np.asarray(puntajes[b]))  # DeLong
        filas.append({"modelo A": a, "modelo B": b, "ΔAUC (A − B)": dif, "IC 2,5 %": lo, "IC 97,5 %": hi, "p crudo": p})  # Fila
    t = pd.DataFrame(filas)  # Tabla
    orden = np.argsort(t["p crudo"].to_numpy())  # Holm: de menor a mayor
    m = len(t)  # Número de pruebas
    ajust = np.minimum(1, np.maximum.accumulate((m - np.arange(m)) * t["p crudo"].to_numpy()[orden]))  # Ajuste monótono
    t.loc[t.index[orden], "p Holm"] = ajust  # Asigna
    return t  # Tabla


def bca_municipios(y, s, municipios, metrica, B=1000, semilla=42):
    """Intervalo bootstrap BCa (bias-corrected and accelerated) remuestreando municipios completos.

    BCa corrige el sesgo (proporción de réplicas por debajo del valor observado) y la asimetría (aceleración estimada
    por jackknife sobre municipios), así que es adecuado cuando la distribución de la métrica es asimétrica.

    Returns
    -------
    (float, float, float)
        Valor observado, límite inferior y superior del IC 95 %.
    """
    y, s, municipios = np.asarray(y), np.asarray(s), np.asarray(municipios)  # Vectores
    unicos = np.unique(municipios)  # Municipios
    filas = [np.flatnonzero(municipios == m) for m in unicos]  # Filas de cada municipio
    est = lambda idx: metrica(y[np.concatenate([filas[i] for i in idx])], s[np.concatenate([filas[i] for i in idx])])  # Métrica de una muestra de municipios
    theta = est(np.arange(len(unicos)))  # Observado
    rng = np.random.default_rng(semilla)  # Generador
    rep = np.array([est(rng.integers(0, len(unicos), len(unicos))) for _ in range(B)])  # Réplicas bootstrap
    z0 = stats.norm.ppf(np.clip(np.mean(rep < theta), 1e-6, 1 - 1e-6))  # Corrección de sesgo
    jack = np.array([est(np.delete(np.arange(len(unicos)), i)) for i in range(len(unicos))])  # Jackknife
    dj = jack.mean() - jack  # Desviaciones
    a = np.sum(dj ** 3) / (6 * np.sum(dj ** 2) ** 1.5 + 1e-12)  # Aceleración
    cuant = [stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) for z in stats.norm.ppf([0.025, 0.975])]  # Cuantiles ajustados
    return float(theta), float(np.quantile(rep, cuant[0])), float(np.quantile(rep, cuant[1]))  # Observado e IC


def delta_cliff(a, b):
    """Delta de Cliff: P(A > B) − P(A < B) ∈ [−1, 1]. |δ| < 0,147 despreciable, < 0,33 pequeño, < 0,474 mediano, si no grande."""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None, :]  # Todas las parejas
    return float(np.mean(a > b) - np.mean(a < b))  # Dominancia estocástica


def magnitud_cliff(d):  # Etiqueta de magnitud (Romano et al., 2006)
    d = abs(d)  # Valor absoluto
    return "despreciable" if d < 0.147 else "pequeño" if d < 0.33 else "mediano" if d < 0.474 else "grande"  # Umbrales


# ---------------------------------------------------------------------------
# Regresión (series de pérdidas semanales)
# ---------------------------------------------------------------------------
def diebold_mariano_hln(e1, e2, h=1, potencia=2):
    """Diebold-Mariano con la corrección de Harvey, Leybourne y Newbold (1997) para muestras finitas.

    d_t = |e1_t|^potencia − |e2_t|^potencia. DM = d̄ / √(V̂/T), con V̂ la varianza de largo plazo con h − 1 autocovarianzas.
    HLN multiplica por √((T + 1 − 2h + h(h − 1)/T) / T) y usa la t de Student con T − 1 grados de libertad.

    Returns
    -------
    (float, float, float)
        d̄ (positivo: el modelo 2 es mejor), estadístico corregido y valor p bilateral.
    """
    d = np.abs(np.asarray(e1)) ** potencia - np.abs(np.asarray(e2)) ** potencia  # Diferencial de pérdida
    T = len(d)  # Longitud
    dc = d - d.mean()  # Centrado
    gamma = [np.sum(dc[k:] * dc[: T - k]) / T for k in range(h)]  # Autocovarianzas
    V = gamma[0] + 2 * sum(gamma[1:])  # Varianza de largo plazo
    dm = d.mean() / np.sqrt(V / T)  # Estadístico DM
    hln = dm * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)  # Corrección HLN
    return float(d.mean()), float(hln), float(2 * stats.t.sf(abs(hln), T - 1))  # Resultado


def diebold_mariano_perdidas(l1, l2, h=1):
    """DM-HLN directamente sobre series de pérdidas (d_t = l1_t − l2_t)."""
    d = np.asarray(l1) - np.asarray(l2)  # Diferencial
    T = len(d)  # Longitud
    dc = d - d.mean()  # Centrado
    V = sum((1 if k == 0 else 2) * np.sum(dc[k:] * dc[: T - k]) / T for k in range(h))  # Varianza de largo plazo
    hln = d.mean() / np.sqrt(V / T) * np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T)  # Estadístico corregido
    return float(d.mean()), float(hln), float(2 * stats.t.sf(abs(hln), T - 1))  # Resultado


def giacomini_white(l1, l2):
    """Prueba de habilidad predictiva condicional de Giacomini y White (2006), horizonte de un paso.

    H0: E[d_{t+1} | h_t] = 0 con instrumentos h_t = (1, d_t). Z_t = h_t · d_{t+1}; GW = T · Z̄ᵀ Ω̂⁻¹ Z̄ ~ χ²(2).
    Detecta superioridad que depende del estado (por ejemplo, del régimen epidémico), no solo en promedio.
    """
    d = np.asarray(l1) - np.asarray(l2)  # Diferencial de pérdida
    Z = np.column_stack([np.ones(len(d) - 1), d[:-1]]) * d[1:, None]  # Instrumentos × diferencial siguiente
    T = len(Z)  # Observaciones
    zbar = Z.mean(axis=0)  # Media
    Om = np.cov(Z, rowvar=False, ddof=0)  # Covarianza
    gw = float(T * zbar @ np.linalg.pinv(Om) @ zbar)  # Estadístico de Wald
    return gw, float(stats.chi2.sf(gw, Z.shape[1]))  # Estadístico y p


def clark_west(y, f_restringido, f_general):
    """Prueba de Clark y West (2007) para modelos anidados (H0: igual capacidad predictiva poblacional).

    f_t = (y − f₁)² − [(y − f₂)² − (f₁ − f₂)²]; el término (f₁ − f₂)² corrige el ruido que agrega estimar los
    parámetros extra del modelo general. Unilateral: H1 el modelo general es mejor.
    """
    y, f1, f2 = map(np.asarray, (y, f_restringido, f_general))  # Vectores
    f = (y - f1) ** 2 - ((y - f2) ** 2 - (f1 - f2) ** 2)  # Diferencial ajustado
    t = f.mean() / (f.std(ddof=1) / np.sqrt(len(f)))  # Estadístico t
    return float(t), float(stats.norm.sf(t))  # Estadístico y p unilateral


def mcs(perdidas, alfa=0.10, bloque=4, reps=1000, semilla=42):
    """Model Confidence Set (Hansen, Lunde y Nason, 2011) con bootstrap estacionario por bloques.

    Parameters
    ----------
    perdidas : pandas.DataFrame
        T × M pérdidas (filas = semanas, columnas = modelos).

    Returns
    -------
    pandas.DataFrame
        Valor p del MCS de cada modelo y si queda en el conjunto de confianza.
    """
    m = MCS(perdidas, size=alfa, block_size=bloque, method="R", reps=reps, bootstrap="stationary", seed=semilla)  # Procedimiento
    m.compute()  # Eliminación secuencial
    p = m.pvalues.iloc[:, 0].rename("p MCS")  # Valores p
    t = pd.DataFrame({"pérdida media": perdidas.mean(), "rango medio": perdidas.rank(axis=1).mean()}).join(p)  # Tabla
    t["¿en el conjunto?"] = t.index.isin(m.included)  # Incluido
    return t.sort_values("pérdida media")  # Ordenada


def spa(benchmark, modelos, bloque=4, reps=1000, semilla=42):
    """Superior Predictive Ability (Hansen, 2005): ¿algún modelo supera genuinamente al benchmark (control de data snooping)?"""
    s = SPA(benchmark, modelos, block_size=bloque, reps=reps, bootstrap="stationary", seed=semilla)  # Prueba
    s.compute()  # Calcula
    return s.pvalues  # Valores p (inferior, consistente, superior)


def ic_bootstrap_estacionario(d, bloque=4, reps=2000, semilla=42):
    """IC 95 % de la media de una serie dependiente con bootstrap estacionario (Politis y Romano, 1994)."""
    bs = StationaryBootstrap(bloque, np.asarray(d), seed=semilla)  # Bloques de longitud geométrica con media «bloque»
    ic = bs.conf_int(np.mean, reps=reps, method="percentile")  # Percentiles
    return float(np.mean(d)), float(ic[0, 0]), float(ic[1, 0])  # Media e IC


def d_cohen(a, b):
    """d de Cohen para dos muestras de pérdidas (desviación combinada)."""
    a, b = np.asarray(a), np.asarray(b)  # Vectores
    s = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)  # Desviación combinada
    return float((a.mean() - b.mean()) / s)  # Diferencia estandarizada
