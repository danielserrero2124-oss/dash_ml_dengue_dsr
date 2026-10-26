"""Genera los tres cuadernos del Jupyter Book (EDA, modelos y pruebas) con nbformat.

Los cuadernos se ejecutan después con nbconvert, así que el libro publica la ejecución real de los modelos.
Uso (desde la raíz del proyecto):  python notebooks/construir_cuadernos.py
"""

from pathlib import Path  # Rutas

import nbformat as nbf  # Construcción de cuadernos

CARPETA = Path(__file__).resolve().parent  # Carpeta notebooks/
md = nbf.v4.new_markdown_cell  # Atajo: celda de texto
code = nbf.v4.new_code_cell  # Atajo: celda de código

PREAMBULO = '''import os  # Variables de entorno
import sys  # Configuración del intérprete
os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # Evita un aviso de joblib en Windows
sys.path.insert(0, "..")  # Agrega la raíz del proyecto para importar src y dashboard
import warnings  # Control de avisos
warnings.filterwarnings("ignore")  # Oculta avisos de versiones que no afectan los resultados
import numpy as np  # Cálculo numérico
import pandas as pd  # Tablas
from IPython.display import HTML, display  # Incrustar figuras interactivas
from dashboard import figuras as fg  # Mismas figuras del dashboard
from src import config as cf  # Configuración del proyecto
pd.set_option("display.width", 160)  # Ancho de las tablas
pd.set_option("display.max_columns", 30)  # Columnas visibles


def mostrar(fig):  # Muestra una figura del dashboard como HTML interactivo
    """Aplica el fondo oscuro de las tarjetas y la incrusta con Plotly cargado desde internet."""
    fig.update_layout(paper_bgcolor=fg.SUPERFICIE, plot_bgcolor=fg.SUPERFICIE)  # Fondo oscuro
    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False)))  # Figura interactiva
'''


def cuaderno_eda():  # Capítulo 2: EDA, datos y preprocesamiento
    c = [md("# EDA, datos y preprocesamiento\n\nEste cuaderno reproduce el análisis exploratorio del dashboard con las mismas funciones "
            "(`dashboard/datos.py` y `dashboard/figuras.py`). Todas las figuras son interactivas."),
         code(PREAMBULO + '''from dashboard import datos as dd  # Panel, agregados del EDA y artefactos
print(f"Panel completo: {dd.RAW.shape[0]:,} filas · {dd.RAW['COD_MUN_N'].nunique()} municipios · {dd.RAW['time_idx'].nunique()} semanas")  # Dimensión
print(f"Filas para modelar (2015–2024): {len(dd.MOD):,}")  # Filas útiles'''),
         md("## Fuente, estructura y estadística descriptiva\n\nPanel balanceado municipio × semana (2010–2024) que integra la notificación "
            "semanal de dengue con clima, NDVI, el índice ONI, población y flujo de personas."),
         code('''cols = ["casos_totales", "temp_mean", "prec_total", "ndvi_mean", "oni_anom", "Flujo_in", "poblacion"]  # Variables originales
desc = dd.RAW[cols].describe().T  # Media, desviación, cuartiles, mínimo y máximo
desc["asimetria"] = dd.RAW[cols].skew()  # Asimetría
desc.round(2)  # Tabla'''),
         md("**Interpretación.** Los casos semanales son muy asimétricos (mediana 0, máximo de más de 1 400): justifican el logaritmo. "
            "El flujo de personas y la población también tienen colas largas."),
         md("## Casos por año y serie nacional"),
         code('''mostrar(fg.fig_anual(dd.ANUAL))  # Casos por año con el periodo del modelo
mostrar(fg.fig_serie_nacional(dd.NAC, dd.CORTES))  # Casos semanales y El Niño'''),
         md("**Interpretación.** Las epidemias de 2010, 2013, 2019 y 2023–2024 coinciden con episodios de El Niño (ONI > 0,5). 2024 es un "
            "régimen excepcional: la prueba evalúa a los modelos en condiciones más extremas que las del entrenamiento."),
         md("## Estacionalidad y grupos de población"),
         code('''mostrar(fg.fig_estacionalidad(dd.EST))  # Perfil semanal de casos y lluvia
mostrar(fg.fig_grupos(dd.EDAD, dd.SEXO))  # Casos por edad y sexo'''),
         md("## Canal endémico: definición del brote\n\nUna semana es de brote si sus casos superan el tercer cuartil de las mismas semanas (± 2) "
            "de los cinco años anteriores y hay al menos 3 casos."),
         code('''for mun in [76001, 68001]:  # Cali y Bucaramanga
    s = dd.RAW[(dd.RAW["COD_MUN_N"] == mun) & (dd.RAW["ANO"] >= 2015)]  # Serie del municipio
    mostrar(fg.fig_canal(pd.DataFrame({"fecha": s["week_start"], "casos": s["casos_totales"], "umbral": s["umbral"], "brote": s["brote"]}), dd.nombre_municipio(mun)))  # Figura
print(f"Semanas de brote 2015–2024: {int(dd.RAW['brote'].sum()):,} ({dd.RAW['brote'].mean():.1%})")  # Prevalencia'''),
         md("## Clima de hoy y brote en cuatro semanas"),
         code('''for v in ["temp_mean_m8", "prec_total_m8", "oni_l8"]:  # Tres variables climáticas
    mostrar(fg.fig_tasa_por_decil(dd.deciles(v), v))  # Tasa de brote por decil
mostrar(fg.fig_ccf(dd.CCF))  # Correlación cruzada clima → casos
mostrar(fg.fig_oni_anual(dd.ONI_ANUAL))  # El Niño y los años epidémicos'''),
         md("**Interpretación.** La tasa de brote crece en los deciles de temperatura de 25–28 °C, la zona óptima para *Aedes aegypti*. "
            "La correlación cruzada es débil (|r| ≈ 0,05) y llega con 8–12 semanas de rezago: el clima modula el riesgo, pero no lo determina solo."),
         md("## Departamentos, faltantes y atípicos"),
         code('''mostrar(fg.fig_departamentos(dd.DEP))  # Incidencia por departamento
mostrar(fg.fig_faltantes(dd.FALTANTES))  # Temperatura imputada por año
print(f"Temperatura imputada: {dd.RAW['temp_mean_missing'].mean():.1%} de las semanas")  # Faltantes
from dashboard.sec_eda import tabla_atipicos  # Regla de Tukey sobre las variables del modelo
mostrar(fg.fig_atipicos(tabla_atipicos()))  # Tasa de brote en atípicos'''),
         md("**Interpretación.** No hay nulos, pero la temperatura está imputada en casi dos tercios de las semanas (concentrada cerca de 22 °C), "
            "lo que atenúa la señal térmica; se conserva la bandera `temp_missing`. Los atípicos son epidemias reales y no se eliminan."),
         md("## Relaciones entre variables"),
         code('''mostrar(fg.fig_correlacion_objetivo(dd.CORR_Y))  # Correlación con el brote en t + 4
mostrar(fg.fig_matriz_correlacion(dd.CORR))  # Matriz de correlación'''),
         md("## Prevención de la fuga de datos\n\n* Todas las variables usan semanas $\\le t$; el umbral objetivo solo usa años anteriores.\n"
            "* La normal climática (anomalías) se calcula con 2010–2021.\n* La validación respeta el tiempo (hacia adelante por años).\n"
            "* El preprocesamiento vive dentro del `Pipeline`.\n\nLa demostración siguiente compara la partición correcta con dos fugas típicas."),
         code('''if dd.ART:  # Requiere el entrenamiento
    mostrar(fg.fig_fuga(dd.ART["fuga"]))  # Fugas: partición aleatoria y variable del futuro
    display(pd.DataFrame(dd.ART["fuga"]).drop(columns="azar").T.round(4))  # Tabla'''),
         ]
    return c  # Celdas


def cuaderno_modelos():  # Capítulo 4: entrenamiento y resultados
    c = [md("# Entrenamiento y resultados de los cinco modelos\n\nEste cuaderno **ejecuta** el flujo completo de `src/entrenamiento.py`: "
            "los cuatro modelos recomendados con `Pipeline` + `GridSearchCV` y validación temporal hacia adelante, y el modelo original HetDS en línea. "
            "Al final guarda los artefactos que usa el dashboard."),
         code(PREAMBULO + '''import time  # Tiempo total
import joblib  # Guardado de artefactos
from src import datos as dt  # Panel de datos
from src import modelado as mod  # Modelos y pruebas
from src import entrenamiento as en  # Flujo de entrenamiento
panel = dt.panel_modelado()  # Panel procesado
tr, va, te = dt.division_temporal(panel)  # Periodos
pd.DataFrame({"filas": [len(tr), len(va), len(te)], "brotes": [tr.y.mean(), va.y.mean(), te.y.mean()]},  # Tamaños y prevalencias
             index=["Entrenamiento 2015–2021", "Validación 2022", "Prueba 2023–2024"]).style.format({"filas": "{:,}", "brotes": "{:.1%}"})'''),
         md("## Mallas de hiperparámetros y validación temporal\n\n`GridSearchCV` usa `ValidacionTemporalAnual`: cuatro pliegues que entrenan con "
            "los años anteriores y validan en 2018, 2019, 2020 y 2021. El criterio de selección es el AUC-PR."),
         code('''pd.DataFrame([(n, str(m[1])) for n, m in mod.catalogo_modelos().items()], columns=["Modelo", "Malla"])  # Mallas'''),
         md("## Ejecución del entrenamiento\n\nLa celda siguiente entrena todo (varios minutos: GradientBoosting es el más costoso)."),
         code('''t0 = time.time()  # Inicio
art, hetds, modelos = en.entrenar_todo()  # Entrena los cuatro recomendados y HetDS
joblib.dump(art, cf.RUTA_ARTEFACTOS, compress=3)  # Artefactos para el dashboard
joblib.dump(hetds, cf.RAIZ / "dashboard" / "hetds.joblib", compress=3)  # Modelo original para el simulador
print(f"Tiempo total: {(time.time() - t0) / 60:.1f} minutos")  # Duración'''),
         md("## Resultados en la prueba 2023–2024"),
         code('''res = art["resultados"]  # Resultados por modelo
tabla = pd.DataFrame({fg.ETIQUETAS_MODELOS[n]: {"Val. AUC-PR": r["val_auc_pr"], "AUC-ROC": r["test"]["auc_roc"], "AUC-PR": r["test"]["auc_pr"],  # Métricas
                      "Precisión": r["test"]["precision"], "Recall": r["test"]["recall"], "F1": r["test"]["f1"], "Umbral": r["umbral"],
                      "Hiperparámetros": ", ".join(f"{k}={v}" for k, v in r["params"].items())} for n, r in res.items()}).T
tabla.sort_values("AUC-ROC", ascending=False)  # Ranking'''),
         code('''for n in ["LogisticRegression", "RandomForest", "KNN", "GradientBoosting"]:  # Resumen de GridSearchCV de cada modelo
    print(fg.ETIQUETAS_MODELOS[n])  # Nombre
    display(pd.DataFrame(res[n]["cv_tabla"]).head(4).round(4))  # Mejores 4 combinaciones'''),
         code('''mostrar(fg.fig_roc(res, "HetDS"))  # Curvas ROC
mostrar(fg.fig_pr(res, "HetDS", float(art["y_test"].mean())))  # Curvas PR
mostrar(fg.fig_confusion(res["HetDS"]["test"]["matriz"]))  # Matriz de confusión de HetDS'''),
         md("## Sobreajuste y desempeño por año"),
         code('''mostrar(fg.fig_sobreajuste(res))  # AUC en entrenamiento, validación y prueba
mostrar(fg.fig_auc_anual(res))  # AUC por año de prueba'''),
         md("## Modelo original: HetDS en línea"),
         code('''h = res["HetDS"]  # Resultados del modelo original
display(pd.DataFrame(h["seleccion"]).round(4))  # Selección de η y ventana en 2022
mostrar(fg.fig_seleccion(h["seleccion"]))  # Mapa de calor
mostrar(fg.fig_pesos(h["pesos"]))  # Pesos de los expertos semana a semana
mostrar(fg.fig_expertos(h))  # Expertos, ensamble estático y HetDS'''),
         md("## ¿Cuánto aporta el clima? e importancia de variables"),
         code('''display(pd.DataFrame(art["aporte_clima"]).T.round(4))  # AUC por bloque de variables
mostrar(fg.fig_aporte(art["aporte_clima"]))  # Figura
mostrar(fg.fig_importancia(h["importancia"], fg.MAGENTA))  # Importancia por permutación de HetDS'''),
         md("## Comparación con DeLong"),
         code('''display(pd.DataFrame(art["delong"]).T)  # Diferencias, IC y valores p
mostrar(fg.fig_delong(art["delong"]))  # Gráfico de bosque'''),
         ]
    return c  # Celdas


def cuaderno_pruebas():  # Capítulo 5: pruebas de significancia, ablación y tiempos
    c = [md("# Pruebas de significancia, ablación y tiempos de cómputo\n\nEste cuaderno ejecuta `src/pruebas.py`: métricas recomendadas en la literatura, "
            "DeLong, bootstrap por conglomerados (municipios), McNemar, Friedman + Nemenyi con las semanas como bloques, la ablación del modelo original "
            "y los tiempos de cómputo."),
         code(PREAMBULO + '''import joblib  # Carga de resultados
from src import pruebas as pb  # Pruebas
pb.main()  # Ejecuta todo (varios minutos)
P = joblib.load(cf.RUTA_PRUEBAS)  # Resultados'''),
         md("## Métricas de comparación"),
         code('''m = pd.DataFrame(P["metricas"]).T.drop(columns=["ic_auc_roc", "ic_auc_pr"]).rename(index=fg.ETIQUETAS_MODELOS)  # Métricas
m.round(4).sort_values("auc_pr", ascending=False)  # Tabla'''),
         md("## Pruebas pareadas (HetDS frente a cada modelo, p ajustados por Holm)"),
         code('''filas = []  # Una fila por modelo
for n, d in P["pruebas"].items():  # Resultados
    filas.append({"HetDS vs.": fg.ETIQUETAS_MODELOS[n], "ΔAUC-ROC": d["delong"]["dif"], "p DeLong": d["delong"]["p_holm"],  # DeLong
                  "IC 95 % bootstrap": tuple(round(x, 4) for x in d["cluster_roc"]["ic"]), "p bootstrap (ROC)": d["cluster_roc"]["p_holm"],  # Bootstrap ROC
                  "ΔAUC-PR": d["cluster_pr"]["dif"], "p bootstrap (PR)": d["cluster_pr"]["p_holm"], "McNemar b/c": f"{d['mcnemar']['b']}/{d['mcnemar']['c']}",
                  "p McNemar": d["mcnemar"]["p_holm"]})
pd.DataFrame(filas).set_index("HetDS vs.")  # Tabla'''),
         code('''mostrar(fg.fig_nemenyi(P["friedman"]["auc_roc"]))  # Friedman + Nemenyi semanal
print({k: v for k, v in P["friedman"]["auc_roc"].items() if k != "rangos"})  # Estadístico, p y DC'''),
         md("## Ablación: ¿de dónde viene la ganancia?"),
         code('''display(pd.DataFrame(P["ablacion"]).T)  # Variantes
mostrar(fg.fig_ablacion(P["ablacion"]))  # Figura'''),
         md("## Tiempos de cómputo"),
         code('''display(pd.DataFrame(P["tiempos"]).T.rename(index=fg.ETIQUETAS_MODELOS).round(2))  # Tabla
mostrar(fg.fig_tiempos(P["tiempos"]))  # Búsqueda y un ajuste
mostrar(fg.fig_prediccion(P["tiempos"]))  # Latencia de predicción'''),
         ]
    return c  # Celdas


for nombre, celdas in [("1_eda", cuaderno_eda()), ("2_modelos", cuaderno_modelos()), ("3_pruebas", cuaderno_pruebas())]:  # Tres cuadernos
    nb = nbf.v4.new_notebook(cells=celdas, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3 (dengue)", "language": "python"}})  # Cuaderno
    nbf.write(nb, CARPETA / f"{nombre}.ipynb")  # Guarda
    print("Creado", nombre)  # Confirmación
