"""Optimizadores de hiperparámetros con una interfaz común: Grid, Random, Bayesiano (Optuna, TPE), Genético (DEAP)
y Successive Halving (multi-fidelidad).

Todos reciben un ``objetivo(params, fidelidad=1.0) -> float`` que se MAXIMIZA y que internamente ajusta el Pipeline completo
(preprocesamiento + balanceo + modelo) en cada pliegue interno, de modo que ningún método ve datos de validación al
preprocesar. Todos devuelven el historial de evaluaciones para construir curvas de desempeño en cualquier momento.
"""

import itertools  # Producto cartesiano de la malla
import math  # Funciones matemáticas
import random  # Aleatoriedad de DEAP
import time  # Cronómetro

import numpy as np  # Cálculo numérico
import optuna  # Optimización bayesiana
from deap import base, creator, tools  # Algoritmos genéticos

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Silencia el registro de Optuna
if not hasattr(creator, "Aptitud"):  # Evita redefinir las clases de DEAP al recargar el módulo
    creator.create("Aptitud", base.Fitness, weights=(1.0,))  # Maximización
    creator.create("Individuo", list, fitness=creator.Aptitud)  # Individuo = lista de genes en [0, 1]


# ---------------------------------------------------------------------------
# Utilidades del espacio de búsqueda
# ---------------------------------------------------------------------------
def decodificar(espacio, genes):
    """Convierte un vector en [0, 1]^d en hiperparámetros (escala log cuando corresponde).

    Parameters
    ----------
    espacio : dict
        {nombre: ("int"|"float", min, max, log) o ("cat", opciones)}.
    genes : sequence of float
        Coordenadas en [0, 1].

    Returns
    -------
    dict
        Hiperparámetros listos para el modelo.
    """
    p = {}  # Hiperparámetros
    for (nombre, spec), g in zip(espacio.items(), genes):  # Cada dimensión
        g = min(max(float(g), 0.0), 1.0)  # Recorta a [0, 1]
        if spec[0] == "cat":  # Categórica
            p[nombre] = spec[1][min(int(g * len(spec[1])), len(spec[1]) - 1)]  # Índice de la categoría
        else:  # Numérica
            lo, hi, log = spec[1], spec[2], spec[3]  # Rango
            v = math.exp(math.log(lo) + g * (math.log(hi) - math.log(lo))) if log else lo + g * (hi - lo)  # Interpolación (log o lineal)
            p[nombre] = int(round(v)) if spec[0] == "int" else float(v)  # Entero o real
    return p  # Hiperparámetros


def _clave(p):  # Clave hashable de una configuración (para la caché)
    return tuple(sorted((k, str(v)) for k, v in p.items()))  # Tupla ordenada


def _registrar(historial, p, valor, t0, fidelidad=1.0):  # Agrega una evaluación al historial
    historial.append({"eval": len(historial) + 1, "params": p, "valor": float(valor), "segundos": time.perf_counter() - t0, "fidelidad": fidelidad})


# ---------------------------------------------------------------------------
# Métodos
# ---------------------------------------------------------------------------
def grid(espacio, objetivo, presupuesto, semilla=42):
    """Búsqueda en malla con la mayor malla regular que cabe en el presupuesto.

    Los niveles se reparten por dimensión en orden (las primeras dimensiones son las más influyentes); una dimensión
    sin niveles adicionales queda fija en el centro de su rango.

    Returns
    -------
    list of dict
        Historial de evaluaciones.
    """
    niveles = {k: 1 for k in espacio}  # Empieza con un nivel (el centro) por dimensión
    tam = lambda: math.prod(len(espacio[k][1]) if espacio[k][0] == "cat" and niveles[k] > 1 else niveles[k] for k in espacio)  # Tamaño de la malla
    cambio = True  # ¿Se pudo agrandar?
    while cambio:  # Agrega niveles mientras quepa
        cambio = False  # Supone que no
        for k in espacio:  # Dimensión por dimensión
            if espacio[k][0] == "cat" and niveles[k] > 1:  # Categórica ya completa
                continue  # Siguiente
            niveles[k] += 1  # Intenta un nivel más
            if tam() <= presupuesto:  # Cabe
                cambio = True  # Hubo cambio
            else:  # No cabe
                niveles[k] -= 1  # Revierte
    ejes = []  # Valores de cada eje en [0, 1]
    for k, spec in espacio.items():  # Cada dimensión
        if spec[0] == "cat":  # Categórica: todas las opciones o la primera
            n = len(spec[1]) if niveles[k] > 1 else 1  # Número de opciones
            ejes.append([(i + 0.5) / len(spec[1]) for i in range(n)])  # Centro de cada categoría
        else:  # Numérica: niveles equiespaciados (en escala log si corresponde)
            ejes.append([0.5] if niveles[k] == 1 else list(np.linspace(0.05, 0.95, niveles[k])))  # Evita los bordes exactos
    historial, t0 = [], time.perf_counter()  # Historial y cronómetro
    for genes in itertools.product(*ejes):  # Cada punto de la malla
        p = decodificar(espacio, genes)  # Hiperparámetros
        _registrar(historial, p, objetivo(p), t0)  # Evalúa
    return historial  # Historial


def aleatoria(espacio, objetivo, presupuesto, semilla=42):
    """Búsqueda aleatoria: muestreo uniforme (o log-uniforme) independiente de cada hiperparámetro."""
    rng = np.random.default_rng(semilla)  # Generador reproducible
    historial, t0 = [], time.perf_counter()  # Historial y cronómetro
    for _ in range(presupuesto):  # Presupuesto de evaluaciones
        p = decodificar(espacio, rng.random(len(espacio)))  # Punto aleatorio
        _registrar(historial, p, objetivo(p), t0)  # Evalúa
    return historial  # Historial


def bayesiana(espacio, objetivo, presupuesto, semilla=42, arranque=5):
    """Optimización bayesiana con Optuna: modelo sustituto TPE (Tree-structured Parzen Estimator).

    TPE modela p(x | y bueno) = l(x) y p(x | y malo) = g(x) con estimadores de densidad de Parzen y propone el candidato
    que maximiza l(x)/g(x), que es equivalente a maximizar la Mejora Esperada (EI). Las ``arranque`` primeras
    evaluaciones son aleatorias (exploración); después TPE explota las regiones prometedoras.
    """
    historial, t0 = [], time.perf_counter()  # Historial y cronómetro

    def f(trial):  # Función objetivo de Optuna
        p = {}  # Hiperparámetros sugeridos
        for k, spec in espacio.items():  # Cada dimensión
            if spec[0] == "cat":  # Categórica
                p[k] = trial.suggest_categorical(k, spec[1])
            elif spec[0] == "int":  # Entera
                p[k] = trial.suggest_int(k, spec[1], spec[2], log=spec[3])
            else:  # Real
                p[k] = trial.suggest_float(k, spec[1], spec[2], log=spec[3])
        v = objetivo(p)  # Evalúa
        _registrar(historial, p, v, t0)  # Registra
        return v  # Optuna maximiza

    muestreador = optuna.samplers.TPESampler(seed=semilla, n_startup_trials=arranque, multivariate=True)  # TPE multivariado
    optuna.create_study(direction="maximize", sampler=muestreador).optimize(f, n_trials=presupuesto)  # Optimiza
    return historial  # Historial


def genetica(espacio, objetivo, presupuesto, semilla=42, poblacion=6, p_cruce=0.7, p_mutacion=0.3, torneo=3):
    """Algoritmo genético (DEAP) con codificación real en [0, 1]^d y estrategia (μ + λ).

    * Selección: torneo de tamaño ``torneo`` (presión selectiva moderada, robusta a la escala de la aptitud).
    * Cruce: mezcla (blend, α = 0,5) con probabilidad ``p_cruce``: explora entre y alrededor de los padres.
    * Mutación: gaussiana (σ = 0,15, probabilidad por gen 0,3) con probabilidad ``p_mutacion``: mantiene diversidad.
    * Elitismo: (μ + λ) conserva los μ mejores entre padres e hijos, así que el mejor nunca se pierde.
    * Caché: una configuración ya evaluada no vuelve a gastar presupuesto.

    Returns
    -------
    historial : list of dict
        Evaluaciones (solo las nuevas).
    diversidad : list of float
        Distancia media entre individuos de cada generación (normalizada por √d): si cae a ~0, hay convergencia prematura.
    """
    random.seed(semilla)  # DEAP usa el módulo random: se siembra una vez (cada tarea corre en su propio proceso)
    rnd = random  # Mismo generador para los operadores y las decisiones
    d = len(espacio)  # Dimensión del genotipo
    tb = base.Toolbox()  # Caja de herramientas de DEAP
    tb.register("individuo", lambda: creator.Individuo([rnd.random() for _ in range(d)]))  # Individuo aleatorio
    historial, cache, t0 = [], {}, time.perf_counter()  # Historial, caché y cronómetro

    def aptitud(ind):  # Evalúa un individuo (con caché)
        p = decodificar(espacio, ind)  # Fenotipo
        k = _clave(p)  # Clave
        if k not in cache:  # Nueva configuración
            cache[k] = objetivo(p)  # Evalúa
            _registrar(historial, p, cache[k], t0)  # Registra
        return (cache[k],)  # Tupla de aptitud

    def diversidad_de(pob):  # Distancia media entre pares de individuos
        G = np.array(pob)  # Matriz de genes
        dist = [np.linalg.norm(G[i] - G[j]) for i in range(len(G)) for j in range(i + 1, len(G))]  # Distancias
        return float(np.mean(dist) / math.sqrt(d))  # Normalizada

    pob = [tb.individuo() for _ in range(poblacion)]  # Población inicial
    for ind in pob:  # Evalúa
        ind.fitness.values = aptitud(ind)
    diversidad = [diversidad_de(pob)]  # Diversidad inicial
    while len(historial) < presupuesto:  # Generaciones hasta agotar el presupuesto
        hijos = []  # Descendencia
        while len(hijos) < poblacion:  # λ = μ hijos
            a, b = [creator.Individuo(x) for x in tools.selTournament(pob, 2, tournsize=torneo)]  # Dos padres por torneo
            if rnd.random() < p_cruce:  # Cruce
                tools.cxBlend(a, b, alpha=0.5)  # Mezcla
            for h in (a, b):  # Mutación de cada hijo
                if rnd.random() < p_mutacion:  # ¿Muta?
                    tools.mutGaussian(h, mu=0, sigma=0.15, indpb=0.3)  # Ruido gaussiano
                h[:] = [min(max(g, 0.0), 1.0) for g in h]  # Recorta a [0, 1]
                del h.fitness.values  # Invalida la aptitud
                hijos.append(h)  # Agrega
        antes = len(historial)  # Evaluaciones antes de la generación
        for h in hijos:  # Evalúa los hijos
            h.fitness.values = aptitud(h)
            if len(historial) >= presupuesto:  # Presupuesto agotado
                break  # Termina
        hijos = [h for h in hijos if h.fitness.valid]  # Hijos evaluados
        pob = tools.selBest(pob + hijos, poblacion)  # (μ + λ): elitismo
        diversidad.append(diversidad_de(pob))  # Diversidad de la generación
        if len(historial) == antes:  # Solo configuraciones repetidas: población convergida
            pob[-1] = tb.individuo()  # Inmigrante aleatorio para no estancarse
            pob[-1].fitness.values = aptitud(pob[-1])  # Evalúa
    return historial, diversidad  # Historial y diversidad


def halving(espacio, objetivo, n_inicial=27, factor=3, fidelidad_min=1 / 9, semilla=42):
    """Successive Halving (multi-fidelidad): muchos candidatos con poco presupuesto, pocos con presupuesto completo.

    Ronda r: n_inicial / factor^r candidatos evaluados con fidelidad fidelidad_min · factor^r (fracción de las filas de
    entrenamiento de cada pliegue interno); pasan a la siguiente ronda los 1/factor mejores. Con 27, 3 y 1/9: 27
    candidatos con 1/9 de los datos, 9 con 1/3 y 3 con todos; costo ≈ 27/9 + 9/3 + 3 = 9 evaluaciones completas.
    """
    rng = np.random.default_rng(semilla)  # Generador reproducible
    candidatos = [decodificar(espacio, rng.random(len(espacio))) for _ in range(n_inicial)]  # Candidatos iniciales
    historial, t0, fid = [], time.perf_counter(), fidelidad_min  # Historial, cronómetro y fidelidad inicial
    while candidatos:  # Rondas
        puntajes = []  # Puntajes de la ronda
        for p in candidatos:  # Cada candidato
            v = objetivo(p, fidelidad=min(fid, 1.0))  # Evalúa con la fidelidad de la ronda
            _registrar(historial, p, v, t0, fidelidad=min(fid, 1.0))  # Registra
            puntajes.append(v)  # Guarda
        if fid >= 1.0 or len(candidatos) <= 1:  # Última ronda
            break  # Termina
        orden = np.argsort(puntajes)[::-1]  # Mejores primero
        candidatos = [candidatos[i] for i in orden[: max(1, len(candidatos) // factor)]]  # Pasa 1/factor
        fid *= factor  # Más datos en la siguiente ronda
    return historial  # Historial


METODOS = {"Grid Search": grid, "Random Search": aleatoria, "Bayesiana (Optuna TPE)": bayesiana, "Genética (DEAP)": genetica}  # Los cuatro exigidos
