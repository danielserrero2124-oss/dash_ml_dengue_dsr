"""Dashboard del Segundo Entregable (Dash + Plotly): Contexto del problema · EDA · Modelos base.

Antes de abrirlo:  python precalculo.py   (genera tablero/artefactos.joblib)
Ejecución (desde la carpeta entregable2):  python -m tablero.app   →   http://localhost:8050
"""

import os  # Variables de entorno

from dash import Dash, Input, Output, callback, dcc, html  # Aplicación, componentes y callbacks

app = Dash(__name__, title="Brotes de Dengue · Modelos base", suppress_callback_exceptions=True)  # Crea la aplicación (carga assets/)
server = app.server  # Servidor Flask (para gunicorn)

from tablero.sec_contexto import seccion_contexto  # noqa: E402  Pestaña 1
from tablero.sec_eda import seccion_eda  # noqa: E402  Pestaña 2 y sus callbacks
from tablero.sec_modelos import seccion_modelos  # noqa: E402  Pestaña 3 y sus callbacks

app.layout = html.Div([
    html.Div([  # Encabezado con imagen de fondo
        html.Div([html.Div("🦟", className="marca-icono"),
                  html.Div([html.H1("Predicción de Brotes de Dengue en Colombia"),
                            html.P("Contexto, análisis exploratorio y modelos base · alerta a 4 semanas · 376 municipios · 2010–2024")])], className="marca"),
        html.Div([html.Span("Dash + Plotly", className="chip"), html.Span("scikit-learn 1.3", className="chip"), html.Span("Segundo entregable", className="chip")]),
    ], className="encabezado"),
    dcc.Tabs(id="pestanas", value="contexto", className="pestanas", children=[  # Navegación principal
        dcc.Tab(label="Contexto del problema", value="contexto", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="EDA", value="eda", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="Modelos base", value="modelos", className="tab", selected_className="tab--selected"),
    ]),
    dcc.Loading(html.Div(id="contenido"), type="dot", color="#ff6b5a"),  # Contenido de la pestaña activa (con indicador de carga)
    html.Div("Maestría en Matemáticas · Proyecto de Aprendizaje Automático · Daniel Serrano Romero", className="pie"),  # Pie de página
], className="contenedor")


@callback(Output("contenido", "children"), Input("pestanas", "value"))  # Cambia de pestaña
def mostrar(p):
    return {"contexto": seccion_contexto, "eda": seccion_eda, "modelos": seccion_modelos}[p]()  # Construye la pestaña


if __name__ == "__main__":  # Ejecución directa
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))  # http://localhost:8050 (o el puerto de PORT)
