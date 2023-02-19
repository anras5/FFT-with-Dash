import dash_bootstrap_components as dbc
import plotly.express as px
import pandas as pd
import numpy as np
from dash import Dash, html, dcc, Output, Input

app = Dash(__name__, external_stylesheets=[dbc.themes.SIMPLEX])
server = app.server


# sin function
def sin(a, t, f):
    return a * np.sin(2 * np.pi * t * f)


app.layout = dbc.Container([
    html.H1('FFT with Dash Plotly'),

    dbc.Row([
        dbc.Col([
            dbc.Row([
                html.Div('Choose amplitude for signal 1:'),
                dcc.Slider(0, 3, value=1,
                           marks={i: str(i) for i in range(4)},
                           id='slider_amp1', tooltip={"placement": "bottom"})
            ]),
            dbc.Row([
                html.Div('Choose frequency for signal 1:'),
                dcc.Slider(0, 4, value=1, id='slider_freq1', tooltip={"placement": "bottom", "always_visible": True}),
            ]),
        ]),
        dbc.Col([
            dbc.Row([
                html.Div('Choose amplitude for signal 2:'),
                dcc.Slider(0, 3, value=1,
                           marks={i: str(i) for i in range(4)},
                           id='slider_amp2', tooltip={"placement": "bottom"})
            ]),
            dbc.Row([
                html.Div('Choose frequency for signal 2:'),
                dcc.Slider(0, 4, value=1, id='slider_freq2', tooltip={"placement": "bottom", "always_visible": True}),
            ]),
        ]),
    ]),

    html.Div('Choose sampling frequency for (signal1 + signal2):'),
    dcc.Slider(0, 16, value=8, id='slider_sample_freq', tooltip={"placement": "bottom"}),

    dbc.Col([
        dbc.Row([
            dcc.Graph(id='signal_graph'),
        ]),
        dbc.Row([
            dcc.Graph(id='fft_graph')
        ])
    ])
])


@app.callback(
    Output('signal_graph', 'figure'),
    Output('fft_graph', 'figure'),
    Input('slider_amp1', 'value'),
    Input('slider_freq1', 'value'),
    Input('slider_amp2', 'value'),
    Input('slider_freq2', 'value'),
    Input('slider_sample_freq', 'value')
)
def update_signal(amp1, freq1, amp2, freq2, sampling_freq):
    # prepare dataframe with signals
    df = pd.DataFrame({
        "t": (x for x in np.arange(0, 10, 1 / 200)),
        "signal1": (sin(amp1, x, freq1) for x in np.arange(0, 10, 1 / 200)),
        "signal2": (sin(amp2, x, freq2) for x in np.arange(0, 10, 1 / 200))
    })

    # signal1 + signal2
    df['signal12'] = df['signal1'] + df['signal2']

    # get samples
    t_samples = np.arange(0, 10, 1 / sampling_freq)
    samples = [sin(amp1, t, freq1) + sin(amp2, t, freq2) for t in t_samples]

    # prepare signal graph
    fig_signals = px.line(df,
                          x='t',
                          y=df.columns[1:],
                          labels={'x': 't', 'value': 'sin(2*pi*t*f)'},
                          title='Signals'
                          )

    # add samples to the graph
    fig_signals.add_scatter(x=t_samples, y=samples,
                            mode='markers',
                            name="sampled signal12",
                            hovertemplate='variable=sample<br>t=%{x}<br>sin(2*pi*t*f)=%{y}<extra></extra>')

    # customize the signal graph
    fig_signals.update_yaxes(range=(-6, 6))
    fig_signals.update_layout(legend_title_text='Signals')

    # FFT
    n = len(samples)
    signal12_fft = abs(np.fft.fft(samples)) / n * 2
    fig_fft = px.scatter(x=[x / n * sampling_freq for x in range(n)],
                         y=signal12_fft,
                         labels={'x': 'Frequency', 'y': 'Amplitude'},
                         color_discrete_sequence=['crimson'],
                         title='FFT')
    fig_fft.update_layout(showlegend=False)

    fig_fft.add_bar(x=[x / n * sampling_freq for x in range(n)],
                    y=signal12_fft,
                    width=[0.05 for _ in range(n)],
                    marker_color='crimson')

    return fig_signals, fig_fft


if __name__ == '__main__':
    print("Visit http://localhost:8050/ to see results")
    app.run(host='0.0.0.0', debug=True)
