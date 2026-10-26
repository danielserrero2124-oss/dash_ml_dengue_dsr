"""Dashboard interactivo del proyecto de predicción de brotes de dengue en Colombia (Dash + Plotly).

Secciones: Contexto · EDA · ML Models (con sub-pestañas organizadas según la rúbrica de sustentación).
Antes de abrirlo: python -m src.entrenamiento  y  python -m src.pruebas
Ejecución (desde la raíz del proyecto):  python -m dashboard.app   →   http://localhost:8050
"""

import os  # Variables de entorno

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count()))  # Evita un aviso de joblib en Windows

from dash import Dash, Input, Output, callback, dcc, html  # noqa: E402  Aplicación, componentes y callbacks

app = Dash(__name__, title="Brotes de Dengue · ML Dashboard", suppress_callback_exceptions=True)  # Crea la aplicación (carga assets/)
server = app.server  # Servidor Flask (para gunicorn)

from dashboard.sec_contexto import seccion_contexto  # noqa: E402  Sección Contexto
from dashboard.sec_eda import seccion_eda  # noqa: E402  Sección EDA y sus callbacks
from dashboard.sec_modelos import seccion_modelos  # noqa: E402  Sección ML Models y sus callbacks

app.layout = html.Div([
    html.Div([  # Encabezado con imagen de fondo
        html.Div([html.Div("🦟", className="marca-icono"),
                  html.Div([html.H1("Predicción de Brotes de Dengue en Colombia"),
                            html.P("Alerta temprana a 4 semanas según las condiciones climáticas · 376 municipios · 2010–2024")])], className="marca"),
        html.Div([html.Span("Dash + Plotly", className="chip"), html.Span("scikit-learn 1.3", className="chip"), html.Span("HetDS en línea desde cero", className="chip")]),
    ], className="encabezado"),
    dcc.Tabs(id="pestanas", value="contexto", className="pestanas", children=[  # Navegación principal
        dcc.Tab(label="Contexto", value="contexto", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="EDA", value="eda", className="tab", selected_className="tab--selected"),
        dcc.Tab(label="ML Models", value="ml", className="tab", selected_className="tab--selected"),
    ]),
    html.Div(id="contenido"),  # Contenido de la pestaña activa
    html.Div("Maestría en Matemáticas · Proyecto de Aprendizaje Automático · Daniel Serrano Romero", className="pie"),  # Pie de página
], className="contenedor")


@callback(Output("contenido", "children"), Input("pestanas", "value"))  # Cambia de sección
def mostrar(p):
    return {"contexto": seccion_contexto, "eda": seccion_eda, "ml": seccion_modelos}[p]()  # Construye la sección


if __name__ == "__main__":  # Solo al ejecutar el archivo directamente
    app.run(debug=False, port=int(os.environ.get("PORT", 8050)))  # http://localhost:8050 (o el puerto de la variable PORT)
