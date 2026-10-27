"""Piezas reutilizables de la interfaz: tarjetas, indicadores, tablas, fórmulas, notas e interpretaciones."""

import pandas as pd  # Tablas
from dash import dcc, html  # Componentes de Dash

CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"]}  # Barra de herramientas simplificada


def numero(valor, dec=3):  # Número con coma decimal
    return f"{valor:.{dec}f}".replace(".", ",")  # 0.8612 → "0,861"


def miles(valor):  # Entero con punto de miles
    return f"{valor:,.0f}".replace(",", ".")  # 100000 → "100.000"


def pct(valor, dec=1):  # Porcentaje con coma decimal
    return f"{100 * valor:.{dec}f} %".replace(".", ",")  # 0.123 → "12,3 %"


def tarjeta(*hijos, clase=""):  # Contenedor con fondo, borde y sombra
    return html.Div(list(hijos), className=f"tarjeta {clase}".strip())  # div con la clase «tarjeta»


def kpi(etiqueta, valor, detalle=""):  # Indicador numérico
    return tarjeta(html.Div(etiqueta, className="etiqueta"), html.Div(valor, className="valor"), html.Div(detalle, className="detalle"), clase="kpi")  # Etiqueta + cifra + detalle


def grafico(id_=None, figura=None):  # Gráfico de Plotly con la configuración común
    props = {"id": id_} if id_ else {}  # Identificador si se actualiza por callback
    return dcc.Graph(figure=figura or {}, config=CONFIG, **props)  # Figura


def desplegable(id_, opciones, valor):  # Lista desplegable
    return dcc.Dropdown(id=id_, options=opciones, value=valor, clearable=False)  # Siempre con una opción


def control(etiqueta, componente):  # Control con su rótulo
    return html.Div([html.Label(etiqueta, className="control-etiqueta"), componente])  # Rótulo + control


def lectura(*texto):  # Recuadro de interpretación debajo de un resultado
    return html.Div([html.Span("Interpretación", className="lectura-titulo"), html.P(list(texto))], className="lectura")  # Texto


def seccion(titulo, subtitulo=""):  # Encabezado de un bloque
    return html.Div([html.H2(titulo), html.P(subtitulo)] if subtitulo else [html.H2(titulo)], className="seccion")  # Título y subtítulo


def tabla(encabezados, filas, numericas=()):  # Tabla HTML; numericas = índices de columnas alineadas a la derecha
    cab = html.Thead(html.Tr([html.Th(h, className="num" if i in numericas else "") for i, h in enumerate(encabezados)]))  # Encabezado
    cuerpo = html.Tbody([html.Tr([html.Td(c, className="num" if i in numericas else "") for i, c in enumerate(f)]) for f in filas])  # Filas
    return html.Div(html.Table([cab, cuerpo], className="tabla"), className="tabla-envoltura")  # Con desplazamiento horizontal


def tabla_df(df, dec=3, indice=True, pct_cols=()):  # Tabla a partir de un DataFrame
    filas = []  # Filas formateadas
    for idx, fila in df.iterrows():  # Cada fila
        celdas = [str(idx)] if indice else []  # Índice
        for col, v in fila.items():  # Cada celda
            if isinstance(v, (float, int)) and not isinstance(v, bool) and pd.notna(v):  # Numérica
                celdas.append(pct(v) if col in pct_cols else miles(v) if isinstance(v, int) or (float(v).is_integer() and abs(v) >= 100) else numero(v, dec))
            else:  # Texto u otro
                celdas.append("—" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
        filas.append(celdas)  # Agrega
    enc = ([df.index.name or ""] if indice else []) + [str(c) for c in df.columns]  # Encabezados
    return tabla(enc, filas, numericas=tuple(range(1 if indice else 0, len(enc))))  # Tabla


def formula(texto):  # Bloque con fórmulas en LaTeX (MathJax)
    return dcc.Markdown(texto, mathjax=True, className="formula")  # $...$ y $$...$$


def pasos(lista):  # Lista de pasos numerados
    return html.Div([html.Div([html.Div(str(i), className="paso-numero"), html.Div(t, className="paso-texto")], className="paso")
                     for i, t in enumerate(lista, 1)])  # Número + texto


def chips(textos):  # Etiquetas en línea
    return html.Div([html.Span(t, className="chip") for t in textos])  # Una etiqueta por texto


def pestanas_internas(id_, opciones, valor):  # Sub-pestañas en forma de píldoras
    return dcc.Tabs(id=id_, value=valor, className="subpestanas",
                    children=[dcc.Tab(label=l, value=v, className="subtab", selected_className="subtab--selected") for v, l in opciones])  # Píldoras
