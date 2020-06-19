import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from utils import process_text, predict_source

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
means = pickle.load(open('data/means.p', 'rb'))
options = means.index

app.layout = html.Div([
    html.H1(children='News Article Classifier', style={'textAlign': 'center'},),
    dcc.Input(id='input-1-state', type='text', value=None),
    html.Button(id='submit-button-state', children='Submit'),
    html.Div(id='article-value', style={'display': 'none'}),
    html.Div(dcc.Dropdown(id='radars', options=[ {'label': i, 'value': i} for i in options ], value='ABC')),
    dcc.Graph(id='probs', style={"width": "80%", "display": "inline-block"}),
    dcc.Graph(id='radar-chart', style={"width": "80%", "display": "inline-block"}),
    ])


@app.callback(
    [
        Output('radar-chart', 'figure'),
        Output('probs', 'figure'),
    ],
    [
        Input('submit-button-state', 'n_clicks'),
        Input('radars', 'value'),
    ],
    [
        State('input-1-state', 'value'),
    ],
    )
def update_output(n_clicks, value, input1):
    if n_clicks is None:
        raise PreventUpdate

    else:
        categories = list(means.columns) + list([means.columns[0]])
        r = means.loc[means.index == value].values.flatten().tolist()
        r.append(r[0])
        fig = go.Figure()
        fig.add_trace(
            go.Scatterpolar(
                name=f'{value}',
                r=r,
                theta=categories,
                fill='toself',
            )
        )
        fig.update_layout(
            title=f'Article Composition and Mean {value} Article Composition',
            polar=dict(
                radialaxis=dict(
                    angle=180,
                    tickangle=-180,
                    visible=True,
                    range=[0, 0.5]
                )
            ),
        )

        topics, vec = process_text(input1)
        r2 = [ topics[i][1] for i in range(len(topics)) ]
        r2.append(r2[0])
        preds = np.round(predict_source(vec)[0], 3)*100

        fig.add_trace(
            go.Scatterpolar(
                name='article',
                r=r2,
                theta=categories,
                fill='toself',
            )
        )

        fig2 = go.Figure()
        sources = ['ABC', 'CCTV', 'CNN', 'Reuters']
        fig2.add_trace(go.Bar(x=preds, y=sources, text=preds, orientation='h'))
        fig2.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig2.update_layout(
            title='Probability of Source',
        )

        return fig, fig2


if __name__ == '__main__':
    app.run_server(debug=True)