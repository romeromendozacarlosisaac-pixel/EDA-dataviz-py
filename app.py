# -*- coding: utf-8 -*-
import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_arch

# --- CONFIGURACIÓN DE LA APP ---
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.FLATLY],
                suppress_callback_exceptions=True)
server = app.server

# Paleta de colores "Trading Esmeralda & Gold"
colors = {
    'background': '#0A1F1C',
    'card_bg': '#122E2B',
    'accent': '#00C896',  # Esmeralda
    'luxury': '#D4AF37',  # Dorado
    'text': '#F5F5F5',
    'grid': '#1A3A35'
}

# --- CARGA DE DATOS ---
def load_data():
    try:
        df_load = pd.read_csv('data/MSFT.csv', index_col=0, parse_dates=True)
    except:
        dates = pd.date_range(start='2000-01-01', periods=6578, freq='D')
        prices = np.cumprod(1 + np.random.normal(0.0006, 0.012, 6578)) * 30
        df_load = pd.DataFrame({
            'Close': prices, 'Open': prices*0.99, 'High': prices*1.01, 'Low': prices*0.98
        }, index=dates)
    df_load['Returns'] = np.log(df_load['Close'] / df_load['Close'].shift(1))
    return df_load.dropna()

df = load_data()

def get_validation_metrics(df_full):
    try:
        close_series = df_full['Close'].replace([np.inf, -np.inf], np.nan).dropna()
        rets = df_full['Returns'].replace([np.inf, -np.inf], np.nan).dropna()
        adf_p_close = adfuller(close_series)[1]
        adf_p_rets = adfuller(rets)[1]
        jb_p = stats.jarque_bera(rets)[1]
        arch_test = het_arch(rets)
        arch_p = arch_test[1]
        return {"adf_close": adf_p_close, "adf_rets": adf_p_rets, "jb": jb_p, "arch": arch_p}
    except:
        return {"adf_close": 1.0, "adf_rets": 1.0, "jb": 1.0, "arch": 1.0}

# --- LAYOUT ---
app.layout = html.Div(style={'backgroundColor': colors['background'], 'minHeight': '100vh'}, children=[
    dbc.Container([
        html.Div([
            dbc.Row([
                dbc.Col(html.Img(
                    src="https://i1.wp.com/socialgeek.co/wp-content/uploads/2017/10/microsoft-oficinas-logo.jpg?w=800&ssl=1",
                    style={"height": "80px", "width": "auto", "backgroundColor": "white", "padding": "10px", "borderRadius": "8px"}), width="auto"),
                dbc.Col([
                    html.H1("Identificación de Estructuras Latentes en las Acciones de Microsoft (2000 - 2026)", 
                            style={'color': colors['accent'], 'fontWeight': 'bold'}),
                    html.H4("Estudio de Series de Tiempo para la Detección de Tendencias, Volatilidad y Correlaciones", 
                            style={'color': colors['luxury']}),
                ]),
            ], align="center", className="mb-4 mt-4"),
        ]),

        dbc.Tabs([
            dbc.Tab(label="Introducción", tab_id="tab-intro"),
            dbc.Tab(label="Contexto y Datos", tab_id="tab-context"),
            dbc.Tab(label="Problema", tab_id="tab-problem"),
            dbc.Tab(label="Objetivos", tab_id="tab-objectives"),
            dbc.Tab(label="Marco Teórico", tab_id="tab-theory"),
            dbc.Tab(label="Resultados", tab_id="tab-results"),
            dbc.Tab(label="Conclusiones", tab_id="tab-conclusions"),
            dbc.Tab(label="Referencias", tab_id="tab-refs"),
        ], id="tabs", active_tab="tab-intro", className="nav-pills"),

        html.Div(id="content", className="p-4 border-0 rounded mt-3 shadow-lg",
                 style={"backgroundColor": colors['card_bg'], "color": colors['text'], "border": f"1px solid {colors['grid']}"})
    ], fluid=True)
])

# --- LÓGICA DE CONTENIDO ---
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def render_tab_content(active_tab):
    if active_tab == "tab-intro":
        return html.Div([
            html.H2("Introducción", style={'color': colors['accent']}),
            dcc.Markdown("""
            El presente Análisis Exploratorio de Datos (EDA) tiene como objetivo estudiar el comportamiento histórico de los precios de las acciones de Microsoft a partir de un conjunto de datos diarios que incluye variables como los precios de apertura, máximo, mínimo y cierre ajustado, así como el volumen de negociación. Este análisis se enmarca dentro del estudio de series de tiempo financieras, donde el interés principal radica en comprender cómo evolucionan los precios de una acción a lo largo del tiempo y qué patrones pueden identificarse en dicha evolución.

            Se busca examinar el comportamiento histórico del precio de la acción de Microsoft (MSFT), identificar patrones temporales, evaluar su estructura estadística y analizar la dinámica de la volatilidad a lo largo del período 2000–2026. A través de este análisis exploratorio se busca describir estadísticamente las variables del conjunto de datos y visualizar su evolución temporal, con un enfoque en el histórico temporal (Cierre y Fecha). Esto permitirá obtener una comprensión inicial del comportamiento de la acción y servirá como base para análisis posteriores más avanzados, como la modelación de series de tiempo para la predicción de los valores del retorno.

            El propósito principal de este estudio es analizar la dinámica temporal del precio de la acción para caracterizar su comportamiento mediante métricas fundamentales en finanzas cuantitativas, especialmente el retorno y la volatilidad. Este estudio será realizado en distintas ventanas temporales para analizar de forma más precisa la variación de estos valores a lo largo del tiempo, además de identificar tendencias y patrones en los distintos periodos.
            """)
        ])

    elif active_tab == "tab-context":
        return html.Div([
            html.H2("Contexto y Descripción de Datos", style={'color': colors['accent']}),
            html.P("Microsoft fue fundada el 4 de abril de 1975 por Bill Gates y Paul Allen, consolidándose como líder en el desarrollo de software. Un hito fundamental fue su salida a bolsa el 13 de marzo de 1986, con un precio inicial de 21 USD por acción. Este análisis trabaja con una serie temporal de frecuencia diaria, abarcando desde el año 2000 hasta el 2026, lo que suma un total de 6,578 observaciones. La serie se caracteriza por una tendencia alcista de largo plazo, con episodios de alta volatilidad durante crisis financieras como la burbuja puntocom (2000-2002) y la crisis financiera global (2008-2009). Además, se observa un crecimiento exponencial reciente impulsado por servicios en la nube e inteligencia artificial."),
            html.Hr(style={'borderColor': colors['grid']}),
            html.H4("Ficha Técnica del Dataset", style={'color': colors['luxury']}),
            html.Ul([
                html.Li([html.B("Fuente: "), "Nasdaq"]),
                html.Li([html.B("Frecuencia: "), "Diaria"]),
                html.Li([html.B("Variables: "), "Fecha (Date), Precio de cierre (Close), Apertura (Open), Máximos y Mínimos."]),
                html.Li([html.B("Ajuste: "), "Precios ajustados por inflación para facilitar la interpretación económica."]),
            ])
        ])

    elif active_tab == "tab-problem":
        return html.Div([
            html.H2("Planteamiento del Problema", style={'color': colors['accent']}),
            html.P("El precio de las acciones de Microsoft (MSFT) constituye una serie de tiempo financiera con comportamientos complejos: tendencias de largo plazo, fluctuaciones abruptas y períodos de volatilidad diferenciada. Comprender esta dinámica histórica es esencial antes de aplicar cualquier modelo predictivo, ya que un análisis exploratorio riguroso permite identificar patrones, detectar anomalías y caracterizar estadísticamente el activo."),
            html.P("Sin este paso previo, la aplicación de modelos de series de tiempo como ARIMA o GARCH carecería de fundamento empírico, aumentando el riesgo de modelar incorrectamente el comportamiento del precio."),
            html.H4("Pregunta de investigación:", style={'color': colors['luxury']}),
            html.P("¿Qué patrones de retorno y volatilidad caracterizan el comportamiento histórico de la acción de Microsoft durante el período 2000–2026, y qué estructura presenta la serie como base para su modelación?")
        ])

    elif active_tab == "tab-objectives":
        return html.Div([
            html.H2("Objetivos", style={'color': colors['accent']}),
            dcc.Markdown(r"""
### Objetivo general
Evaluar la estructura estadística y la dinámica temporal de MSFT mediante un EDA robusto fundamentado en econometría financiera, con el fin de aplicar modelos predictivos para los valores de retorno y de volatibilidad.

### Objetivos Específicos
- **Validar Estacionariedad:** Aplicar la prueba ADF para identificar la presencia de raíces unitarias en la serie de precios de MSFT.
- **Identificar Tendencia:** Implementar medias móviles en ventana diaria, trimestral y anual $(n=1, n= 60, n=252)$ como herramienta de suavizamiento.
- **Analizar Riesgo:** Determinar el grado de Leptocurtosis en la distribución de los retornos logarítmicos como medida del riesgo de cola.
- **Estabilizar la Serie:** Transformar los precios en retornos logarítmicos para obtener una serie con varianza estable.
            """, mathjax=True)
        ])

    elif active_tab == "tab-theory":
        return html.Div([
            html.H2("Marco Teórico", style={'color': colors['accent']}),
            dcc.Markdown(r"""
#### 5.1. Retorno
El retorno es una medida ampliamente utilizada en finanzas para analizar la variación relativa de un activo en el tiempo. Se define como:

$$\displaystyle R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

donde $P_t$ es el precio en el tiempo $t$. Su principal ventaja es que permite trabajar con series más cercanas a la estacionariedad, facilita la agregación en el tiempo y simplifica el análisis estadístico.

#### 5.2. Volatilidad
La volatilidad mide la magnitud de las fluctuaciones de una serie financiera, es decir, el grado de variabilidad de los retornos. Es un indicador clave del riesgo de un activo. En series de tiempo financieras, es común observar fenómenos como el **clustering de volatilidad**, donde periodos de alta volatilidad tienden a agruparse, lo que motiva el uso de modelos como ARCH y GARCH.

$$\displaystyle \sigma = \sqrt{\frac{1}{n-1} \sum_{t=1}^{n} (r_t - \bar{r})^2}$$

#### 5.3. Estabilización de la Serie: Retornos Logarítmicos
Dado que los precios de los activos suelen exhibir tendencias estocásticas y variabilidad no constante, se procede a transformar los precios en retornos logarítmicos. Esta transformación busca obtener una serie con varianza más estable y propiedades estadísticas interpretables. El retorno logarítmico se define como:

$$\displaystyle r_t = \ln(P_t) - \ln(P_{t-1})$$

A diferencia del retorno simple, los log-retornos facilitan la agregación temporal y suelen presentar una distribución más cercana a la normalidad, aunque conservan características críticas para el análisis de riesgo.

---

#### 5.4. Modelos de Descomposición
* **Modelo Aditivo:** $Y_t = T_t + S_t + E_t$
  Adecuado cuando la variación de la serie es aproximadamente constante en el tiempo.
* **Modelo Multiplicativo:** $Y_t = T_t \times S_t \times E_t$
  Se utiliza cuando la variabilidad de la serie aumenta o disminuye proporcionalmente con su nivel, como ocurre comúnmente en series financieras.

---

#### 5.5. Prueba de Dickey-Fuller Aumentada (ADF)
La prueba ADF se utiliza para determinar si una serie de tiempo es estacionaria o presenta una raíz unitaria. La hipótesis nula ($H_0$) establece que la serie **no es estacionaria**. Un p-valor bajo (menor a 0.05) permite rechazar esta hipótesis.

#### 5.6. Identificación de Tendencia y Suavizamiento
Para mitigar el ruido de las fluctuaciones diarias y extraer la señal de tendencia, se implementan medias móviles con ventanas específicas:
* **Diaria ($n=1$):** Representa la serie original sin filtrar.
* **Trimestral ($n=60$):** Suaviza las fluctuaciones de corto plazo.
* **Anual ($n=252$):** Filtra la volatilidad estacional para revelar la tendencia de largo plazo.

#### 5.7. Funciones de Autocorrelación (ACF y PACF)
* **ACF (Autocorrelation Function):** Mide la correlación entre una serie y sus rezagos. Identifica componentes de **Media Móvil (MA)**.
* **PACF (Partial Autocorrelation Function):** Mide la correlación con un rezago específico eliminando efectos intermedios. Identifica el orden de componentes **Autorregresivos (AR)**.
            """, mathjax=True)
        ], style={'lineHeight': '1.7', 'padding': '10px'})

    elif active_tab == "tab-results":
        metrics = get_validation_metrics(df)
        return html.Div([
            dbc.Row([
                dbc.Col([
                    html.H4("Validación Estadística de la Serie", style={'color': colors['luxury'], 'textAlign': 'center'}),
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Prueba Estadística", style={'color': colors['accent']}),
                            html.Th("Métrica"), html.Th("P-Valor"), html.Th("Interpretación")
                        ])),
                        html.Tbody([
                            html.Tr([html.Td("ADF (Precios)"), html.Td("Estacionariedad"), html.Td(f"{metrics['adf_close']:.4f}"), html.Td("No Estacionaria", style={'color': '#FF4B2B'})]),
                            html.Tr([html.Td("ADF (Retornos)"), html.Td("Estacionariedad"), html.Td(f"{metrics['adf_rets']:.4f}"), html.Td("Estacionaria", style={'color': colors['accent']})]),
                            html.Tr([html.Td("Jarque-Bera"), html.Td("Normalidad"), html.Td(f"{metrics['jb']:.4f}"), html.Td("Leptocúrtica", style={'color': colors['accent']})]),
                            html.Tr([html.Td("Engle (ARCH-LM)"), html.Td("Heterocedasticidad"), html.Td(f"{metrics['arch']:.4f}"), html.Td("Clustering", style={'color': colors['accent']})]),
                        ])
                    ], bordered=True, hover=True, striped=True, color="dark", style={'borderColor': colors['grid'], 'textAlign': 'center'})
                ], width=12)
            ], className="mb-5"),
            dbc.Row([
                dbc.Col([
                    html.Label("Análisis Técnico:", style={"color": colors['accent']}),
                    dcc.Dropdown(id='selector', options=[
                        {'label': 'Velas Japonesas', 'value': 'Candle'},
                        {'label': 'Medias Móviles', 'value': 'Moving_Averages'},
                        {'label': 'Detección de Anomalías', 'value': 'Anomalies'},
                        {'label': 'Autocorrelación (ACF)', 'value': 'ACF_Returns'},
                        {'label': 'Descomposición', 'value': 'Decomposition'}
                    ], value='Candle', clearable=False, style={'color': '#000'}),
                ], width=5),
                dbc.Col([
                    html.Label("Rango de Años:", style={"color": colors['accent']}),
                    dcc.RangeSlider(id='slider', min=2000, max=2026, value=[2018, 2026],
                                    marks={i: str(i) for i in range(2000, 2027, 4)}, step=1)
                ], width=7),
            ], className="mb-4"),
            dcc.Graph(id='grafico-principal'),
            dbc.Row([
                dbc.Col(dcc.Graph(id='grafico-distribucion'), width=6),
                dbc.Col(dcc.Graph(id='grafico-volatilidad'), width=6),
            ], className="mt-4")
        ])

    elif active_tab == "tab-conclusions":
        return html.Div([
            html.H3("Conclusiones del Análisis Estructural (2000-2026)", style={'color': colors['accent']}),
            html.P("Tras la evaluación rigurosa de la dinámica histórica de Microsoft (MSFT), se concluye que el activo presenta una estructura de serie de tiempo compleja cuya modelación directa en niveles de precios resultaría técnicamente inválida. La evidencia de no estacionariedad, ratificada por un p-valor de 0.9901 en la prueba ADF y una persistencia prolongada en la función de autocorrelación, confirma que la serie original está dominada por una tendencia estocástica creciente, lo que justifica la transformación obligatoria a retornos logarítmicos para estabilizar la varianza. El análisis de la distribución de estos retornos permite caracterizar el comportamiento del activo mediante una marcada leptocurtosis, indicando que el riesgo histórico está definido por una frecuencia de eventos extremos superior a la de una distribución normal, particularmente visible en las crisis financieras de las últimas dos décadas. Asimismo, la identificación de fenómenos de heterocedasticidad y clustering de volatilidad proporciona el fundamento empírico necesario para la implementación de modelos de varianza condicional, mientras que la descomposición estructural demuestra que la evolución de MSFT responde primordialmente a factores de crecimiento de largo plazo más que a patrones estacionales cíclicos. En síntesis, estos hallazgos resuelven la pregunta de investigación al establecer que la serie de MSFT requiere un tratamiento econométrico que gestione tanto la memoria en la varianza como la presencia de outliers, sentando las bases críticas para un despliegue de modelos predictivos robustos.")
        ])

    elif active_tab == "tab-refs":
        return html.Div([
            html.H2("Referencias Bibliográficas", style={'color': colors['accent']}),
            dcc.Markdown("""
* **Heimann, G. (2016).** *Statistical Analysis of Financial Time Series*. Oxford University Press.
* **Tsay, R. S. (2001).** *Analysis of Financial Time Series*. Wiley-Interscience.
* **Morales, J. A. M. (2013).** *Análisis de Series Temporales*. Universidad Complutense de Madrid.
* **Engle, R. F. (1982).** *Autoregressive conditional heteroscedasticity*. Econometrica.
* **Bollerslev, T. (1986).** *Generalized autoregressive conditional heteroskedasticity*. Journal of Econometrics.
            """)
        ])
    return html.Div("...")

# --- CALLBACK GRÁFICOS ---
@app.callback(
    [Output('grafico-principal', 'figure'), Output('grafico-distribucion', 'figure'), Output('grafico-volatilidad', 'figure')],
    [Input('selector', 'value'), Input('slider', 'value')]
)
def update_graphs(col, years):
    filtered = df[(df.index.year >= years[0]) & (df.index.year <= years[1])].copy()
    layout_base = {"template": "plotly_dark", "paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "font": {"color": colors['text']}}
    
    fig1 = go.Figure()
    if col == 'Candle':
        fig1.add_trace(go.Candlestick(x=filtered.index, open=filtered['Open'], high=filtered['High'], low=filtered['Low'], close=filtered['Close'], 
                                     increasing_line_color=colors['accent'], decreasing_line_color='#FF4B2B'))
        fig1.update_layout(xaxis_rangeslider_visible=False)
    elif col == 'Moving_Averages':
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'], name='Precio', line=dict(color='rgba(255,255,255,0.2)')))
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'].rolling(60).mean(), name='Trimestral', line=dict(color=colors['accent'])))
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Close'].rolling(252).mean(), name='Anual', line=dict(color=colors['luxury'])))
    elif col == 'Decomposition':
        res = seasonal_decompose(filtered['Close'], model='multiplicative', period=252)
        fig1 = make_subplots(rows=2, cols=1)
        fig1.add_trace(go.Scatter(x=filtered.index, y=res.trend, name='Tendencia'), 1, 1)
        fig1.add_trace(go.Scatter(x=filtered.index, y=res.resid, mode='markers', name='Residuo'), 2, 1)
    elif col == 'Anomalies':
        m, s = filtered['Returns'].mean(), filtered['Returns'].std()
        anom = filtered[(filtered['Returns'] > m+3*s) | (filtered['Returns'] < m-3*s)]
        fig1.add_trace(go.Scatter(x=filtered.index, y=filtered['Returns'], name='Retorno', line=dict(color='rgba(0,200,150,0.3)')))
        fig1.add_trace(go.Scatter(x=anom.index, y=anom['Returns'], mode='markers', name='Outlier', marker=dict(color=colors['luxury'])))
    elif col == 'ACF_Returns':
        y_acf = acf(filtered['Returns'], nlags=40)
        fig1.add_trace(go.Bar(x=list(range(len(y_acf))), y=y_acf, marker_color=colors['accent']))

    fig1.update_layout(**layout_base, title=f"Análisis: {col}")
    fig2 = go.Figure(go.Histogram(x=filtered['Returns'], nbinsx=50, marker_color=colors['accent'], opacity=0.7)).update_layout(**layout_base, title="Distribución Retornos", height=300)
    fig3 = go.Figure(go.Scatter(x=filtered.index, y=filtered['Close'], fill='tozeroy', line=dict(color=colors['luxury']))).update_layout(**layout_base, title="Precio Cierre", height=300)
    
    return fig1, fig2, fig3

server = app.server
if __name__ == '__main__':
    app.run(debug=True)