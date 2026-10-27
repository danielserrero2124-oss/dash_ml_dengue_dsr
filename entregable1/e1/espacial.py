"""Análisis espacial desde cero: distancias haversine, matrices de pesos, I de Moran global, LISA, Getis-Ord Gi*,
índice del vecino más cercano (Clark-Evans), función K de Ripley y prueba espacio-temporal de Knox.

Las coordenadas son centroides municipales en grados (EPSG:4326). Nunca se usa distancia euclidiana sobre grados:
las distancias se calculan con la fórmula haversine o en una proyección local en kilómetros.
"""

import numpy as np  # Cálculo numérico
from scipy import stats  # Distribución normal
from scipy.spatial import ConvexHull, cKDTree  # Envolvente convexa y búsqueda de vecinos
from matplotlib.path import Path  # Prueba de punto dentro de un polígono

RADIO_TIERRA = 6371.0088  # Radio medio de la Tierra en km


def haversine(lat, lon):  # Matriz de distancias de gran círculo (km) entre todos los pares de puntos
    la, lo = np.radians(lat)[:, None], np.radians(lon)[:, None]  # Columnas en radianes
    a = np.sin((la - la.T) / 2) ** 2 + np.cos(la) * np.cos(la.T) * np.sin((lo - lo.T) / 2) ** 2  # Fórmula haversine
    return 2 * RADIO_TIERRA * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Distancia en km


def proyectar(lat, lon):  # Proyección equirectangular local a km (válida a la escala de Colombia para áreas y vecinos)
    lat0 = np.radians(np.mean(lat))  # Latitud de referencia
    x = np.radians(lon) * RADIO_TIERRA * np.cos(lat0)  # Este-oeste en km
    y = np.radians(lat) * RADIO_TIERRA  # Norte-sur en km
    return np.column_stack([x, y])  # Coordenadas planas


def pesos_knn(D, k=6, estandarizar=True):  # Matriz de pesos de los k vecinos más cercanos
    """Se usa k vecinos porque el panel no incluye todos los municipios del país (no hay contigüidad completa):
    con k vecinos todo municipio tiene vecinos y la matriz no depende de la densidad local."""
    n = D.shape[0]  # Número de municipios
    W = np.zeros((n, n))  # Matriz de pesos
    vecinos = np.argsort(D + np.eye(n) * 1e12, axis=1)[:, :k]  # Los k más cercanos (excluye a sí mismo)
    W[np.arange(n)[:, None], vecinos] = 1.0  # Vecino = 1
    return W / W.sum(axis=1, keepdims=True) if estandarizar else W  # Estandarizada por filas: el rezago es un promedio


def pesos_banda(D, distancia):  # Matriz de pesos por banda de distancia (vecino si está a menos de «distancia» km)
    W = ((D > 0) & (D <= distancia)).astype(float)  # Binaria
    s = W.sum(axis=1, keepdims=True)  # Número de vecinos
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)  # Estandarizada por filas (aislados quedan en cero)


def moran(x, W, permutaciones=999, semilla=42):  # I de Moran global con prueba de permutaciones
    """I = (n/S0) · zᵀWz / zᵀz, con z = x − x̄. E[I] = −1/(n−1). p = proporción de permutaciones con I ≥ I_obs."""
    x = np.asarray(x, float)  # Vector
    z = x - x.mean()  # Desviaciones
    n, S0 = len(x), W.sum()  # Tamaño y suma de pesos
    I = n / S0 * (z @ W @ z) / (z @ z)  # Estadístico observado
    rng = np.random.default_rng(semilla)  # Generador reproducible
    sim = np.empty(permutaciones)  # I bajo aleatoriedad espacial
    for b in range(permutaciones):  # Permutaciones
        zp = rng.permutation(z)  # Reasigna los valores a otros lugares
        sim[b] = n / S0 * (zp @ W @ zp) / (zp @ zp)  # I permutado
    p = (np.sum(sim >= I) + 1) / (permutaciones + 1)  # Valor p unilateral (autocorrelación positiva)
    return {"I": float(I), "E_I": -1 / (n - 1), "z": float((I - sim.mean()) / sim.std()), "p": float(p)}  # Resultado


def lisa(x, W, permutaciones=999, semilla=42):  # Indicadores locales de asociación espacial (Anselin, 1995)
    """I_i = (z_i / m2) · Σ_j w_ij z_j. Permutación condicional: se fija el valor del municipio i y se sortean los
    valores de sus vecinos entre los demás municipios. Devuelve I_i, valor p y cuadrante (AA, BB, AB, BA)."""
    x = np.asarray(x, float)  # Vector
    z = x - x.mean()  # Desviaciones
    m2 = (z @ z) / len(z)  # Varianza
    rezago = W @ z  # Promedio de los vecinos
    Ii = z / m2 * rezago  # I local observado
    n = len(z)  # Tamaño
    k = int(round((W > 0).sum(axis=1).mean()))  # Vecinos por municipio (k vecinos)
    rng = np.random.default_rng(semilla)  # Generador reproducible
    p = np.empty(n)  # Valores p
    for i in range(n):  # Cada municipio
        otros = np.delete(z, i)  # Valores de los demás
        idx = np.argpartition(rng.random((permutaciones, n - 1)), k, axis=1)[:, :k]  # k vecinos al azar sin reemplazo
        Isim = z[i] / m2 * otros[idx].mean(axis=1)  # I local simulado
        mas = np.sum(Isim >= Ii[i]) if Ii[i] >= 0 else np.sum(Isim <= Ii[i])  # Simulaciones tan extremas como la observada
        p[i] = (mas + 1) / (permutaciones + 1)  # Valor p (pseudo)
    cuadrante = np.select([(z > 0) & (rezago > 0), (z < 0) & (rezago < 0), (z > 0) & (rezago < 0)],  # Tipo de asociación
                          ["Alto-Alto", "Bajo-Bajo", "Alto-Bajo"], "Bajo-Alto")  # Alto-Alto = foco caliente
    return Ii, p, cuadrante  # Resultado


def getis_ord(x, D, k=6):  # Gi* de Getis-Ord con los k vecinos más el propio municipio (puntaje z)
    x = np.asarray(x, float)  # Vector
    n = len(x)  # Tamaño
    W = pesos_knn(D, k, estandarizar=False) + np.eye(n)  # Binaria e incluye al propio municipio (la estrella de Gi*)
    xb, s = x.mean(), x.std()  # Media y desviación global
    sw = W.sum(axis=1)  # Suma de pesos (k + 1)
    num = W @ x - xb * sw  # Suma local menos la esperada
    den = s * np.sqrt((n * (W ** 2).sum(axis=1) - sw ** 2) / (n - 1))  # Error estándar
    return num / den  # z > 1,96: foco caliente; z < −1,96: foco frío


def clark_evans(xy):  # Índice del vecino más cercano: R < 1 agrupado, R ≈ 1 aleatorio, R > 1 disperso
    n = len(xy)  # Número de puntos
    A = ConvexHull(xy).volume  # Área de la envolvente convexa (en 2D, «volume» es el área), km²
    d, _ = cKDTree(xy).query(xy, k=2)  # Distancia al vecino más cercano (la primera es el propio punto)
    r_obs = d[:, 1].mean()  # Distancia media observada
    r_esp = 0.5 / np.sqrt(n / A)  # Distancia esperada bajo aleatoriedad completa (Poisson)
    ee = 0.26136 / np.sqrt(n ** 2 / A)  # Error estándar
    z = (r_obs - r_esp) / ee  # Estadístico
    return {"R": float(r_obs / r_esp), "r_obs": float(r_obs), "r_esp": float(r_esp), "z": float(z),  # Resultado
            "p": float(2 * stats.norm.sf(abs(z))), "area": float(A)}


def _aleatorios_en_poligono(poligono, n, rng):  # n puntos uniformes dentro de un polígono (muestreo por rechazo)
    camino = Path(poligono)  # Polígono
    lo, hi = poligono.min(axis=0), poligono.max(axis=0)  # Caja envolvente
    puntos = []  # Acumulador
    while sum(len(p) for p in puntos) < n:  # Hasta tener suficientes
        c = rng.uniform(lo, hi, size=(2 * n, 2))  # Candidatos en la caja
        puntos.append(c[camino.contains_points(c)])  # Conserva los que caen dentro
    return np.vstack(puntos)[:n]  # Primeros n


def ripley_l(xy, distancias, simulaciones=99, semilla=42):  # Función L de Ripley con envolvente de aleatoriedad completa
    """K(d) = A/(n(n−1)) · Σ_{i≠j} 1(d_ij ≤ d); L(d) = √(K/π). Bajo aleatoriedad L(d) − d ≈ 0; por encima de la
    envolvente simulada hay agrupamiento a esa escala. No se corrige el efecto de borde (se compara con simulaciones
    en la misma región, que sufren el mismo efecto)."""
    hull = ConvexHull(xy)  # Envolvente convexa
    A = hull.volume  # Área
    poligono = xy[hull.vertices]  # Vértices

    def L(pts):  # L(d) − d de un conjunto de puntos
        n = len(pts)  # Tamaño
        dd = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))[np.triu_indices(n, 1)]  # Distancias por pares
        K = np.array([2 * np.sum(dd <= d) for d in distancias]) * A / (n * (n - 1))  # K de Ripley
        return np.sqrt(K / np.pi) - distancias  # L(d) − d

    rng = np.random.default_rng(semilla)  # Generador reproducible
    sims = np.array([L(_aleatorios_en_poligono(poligono, len(xy), rng)) for _ in range(simulaciones)])  # CSR simulada
    return L(xy), sims.min(axis=0), sims.max(axis=0)  # Observada y envolvente


def knox(xy, t, d_espacio, d_tiempo, permutaciones=999, semilla=42):  # Prueba de interacción espacio-tiempo de Knox (1964)
    """Cuenta los pares de eventos cercanos a la vez en el espacio (≤ d_espacio km) y en el tiempo (≤ d_tiempo
    semanas) y la compara con la distribución obtenida al permutar los tiempos entre los lugares."""
    pares = np.array(sorted(cKDTree(xy).query_pairs(d_espacio)))  # Pares cercanos en el espacio
    t = np.asarray(t, float)  # Tiempos
    observado = int(np.sum(np.abs(t[pares[:, 0]] - t[pares[:, 1]]) <= d_tiempo))  # Pares cercanos en espacio y tiempo
    rng = np.random.default_rng(semilla)  # Generador reproducible
    sim = np.empty(permutaciones)  # Conteos bajo independencia espacio-tiempo
    for b in range(permutaciones):  # Permutaciones
        tp = rng.permutation(t)  # Reasigna los tiempos
        sim[b] = np.sum(np.abs(tp[pares[:, 0]] - tp[pares[:, 1]]) <= d_tiempo)  # Conteo permutado
    return {"observado": observado, "esperado": float(sim.mean()), "razon": float(observado / sim.mean()),  # Resultado
            "p": float((np.sum(sim >= observado) + 1) / (permutaciones + 1)), "pares_espacio": int(len(pares))}
