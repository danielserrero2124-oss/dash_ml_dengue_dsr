"""Estilo común de las figuras interactivas del informe (Plotly) y función para incrustarlas en el Jupyter Book."""

import plotly.graph_objects as go  # Figuras de bajo nivel
import plotly.io as pio  # Plantillas
from IPython.display import HTML, display  # Incrustar HTML en el cuaderno

AZUL, ROJO, VERDE, AMBAR, GRIS, MORADO = "#2563eb", "#dc2626", "#16a34a", "#d97706", "#64748b", "#7c3aed"  # Paleta
PERIODO_COLOR = {"historia": GRIS, "entrenamiento": AZUL, "validación": VERDE, "prueba": AMBAR, "purgada": "#cbd5e1"}  # Colores por periodo
CUADRANTE_COLOR = {"Alto-Alto": ROJO, "Bajo-Bajo": AZUL, "Alto-Bajo": "#f9a8d4", "Bajo-Alto": "#93c5fd", "No significativo": "#e2e8f0"}  # LISA

plantilla = go.layout.Template(pio.templates["plotly_white"])  # Parte de la plantilla blanca
plantilla.layout.font = dict(family="Inter, Segoe UI, sans-serif", size=13)  # Tipografía
plantilla.layout.colorway = [AZUL, ROJO, VERDE, AMBAR, MORADO, GRIS, "#0891b2", "#db2777"]  # Colores por defecto
plantilla.layout.margin = dict(l=60, r=30, t=60, b=50)  # Márgenes
pio.templates["informe"] = plantilla  # Registra la plantilla
pio.templates.default = "informe"  # La usa por defecto


def mostrar(fig, alto=None):  # Incrusta una figura interactiva (Plotly se carga desde internet)
    if alto:  # Altura opcional
        fig.update_layout(height=alto)  # Ajusta
    display(HTML(fig.to_html(include_plotlyjs="cdn", full_html=False, config={"displaylogo": False})))  # HTML interactivo


def mapa_base(fig):  # Encuadre geográfico de Colombia para scatter_geo
    fig.update_geos(fitbounds="locations", showcountries=True, countrycolor="#94a3b8", showland=True,  # Fronteras y tierra
                    landcolor="#f8fafc", showocean=True, oceancolor="#e0f2fe", showlakes=False,  # Fondo
                    resolution=50, projection_type="mercator")  # Resolución y proyección
    return fig  # Figura ajustada
