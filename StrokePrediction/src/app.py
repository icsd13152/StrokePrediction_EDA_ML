
import pandas as pd
import numpy as np
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State


import plotly.graph_objs as go
import shap

import math
import joblib

# Load model
current_folder = os.path.dirname(__file__)
hd_model_obj = joblib.load('GB.sav')

hdpred_model = hd_model_obj
hd_pipeline = []

# Start Dashboard
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Layout
app.layout = html.Div([
    html.Div([html.H2('Example Stroke Risk Prediction Tool',
                      style={'marginLeft': 20, 'color': 'white'})],
             style={'borderBottom': 'thin black solid',
                    'backgroundColor': '#24a0ed',
                    'padding': '10px 5px'}),
    dbc.Row([
        dbc.Col([html.Div("Patient information",
                          style={'font-weight': 'bold', 'font-size': 20}),
                 dbc.Row([html.Div("Patient Inofrmation",
                                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('Patient Age (years): '),
                         dcc.Input(
                             type="number",
                             debounce=True,
                             value='55',
                             id='age'
                         )
                     ]), width={"size": 3}),
                     # dbc.Col(html.Div([
                     #     html.Label('Sex: '),
                     #     dcc.Dropdown(
                     #         options=[
                     #             {'label': 'Female', 'value': '0'},
                     #             {'label': 'Male', 'value': '1'}
                     #         ],
                     #         value='0',
                     #         id='sex_male'
                     #     )
                     # ]), width={"size": 3}),
                 ], style={'padding': '10px 25px'}),
                 dbc.Row([
                     # dbc.Col(html.Div([
                     #     html.Label('Patient Age (years): '),
                     #     dcc.Input(
                     #         type="number",
                     #         debounce=True,
                     #         value='55',
                     #         id='age'
                     #     )
                     # ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Sex: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Female', 'value': '0'},
                                 {'label': 'Male', 'value': '1'}
                             ],
                             value='0',
                             id='sex_male'
                         )
                     ]), width={"size": 3}),
                 ], style={'padding': '10px 25px'}),
                 dbc.Row([html.Div("Patient health",
                                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('BMI: '),
                         dcc.Input(
                             type="number",
                             debounce=True,
                             value='132',
                             id='resting_bp'
                         )
                     ]), width={"size": 3}, style={'padding': '10px 10px'}),
                     dbc.Col(html.Div([
                         html.Label('hypertension: '),
                         dcc.Input(
                             type="number",
                             debounce=True,
                             value='151',
                             id='maximum_hr'
                         )
                     ]), width={"size": 3}, style={'padding': '10px 10px'}),
                     dbc.Col(html.Div([
                         html.Label('heart_disease: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'No', 'value': '0'},
                                 {'label': 'Yes', 'value': '1'}
                             ],
                             value='0',
                             id='ht'
                         )
                     ]), width={"size": 3}, style={'padding': '10px 10px'}),
                     dbc.Col(html.Div([
                         html.Label('Ever Married: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'No', 'value': '0'},
                                 {'label': 'Yes', 'value': '1'}
                             ],
                             value='0',
                             id='married'
                         )
                     ]), width={"size": 3}, style={'padding': '10px 10px'}),
                 ], style={'padding': '10px 25px'}),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('Work Type: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Asymptomatic', 'value': '0'},
                                 {'label': 'Angina', 'value': '1'},
                                 {'label': 'Non-anginal', 'value': '2'},
                                 {'label': 'Non-anginal', 'value': '3'}
                             ],
                             value='0',
                             id='work_type'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Smoking Status: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'xx', 'value': '0'},
                                 {'label': 'xx2', 'value': '1'},
                                 {'label': 'x', 'value': '2'},
                                 {'label': 'g', 'value': '3'}
                             ],
                             value='0',
                             id='smoke_status'
                         )
                     ]), width={"size": 3}),
                 ], style={'padding': '10px 25px'}),
                 dbc.Row([html.Div("ECG results",
                                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row([
                     dbc.Col(html.Div([
                         html.Label('Residence Type: '),
                         dcc.Dropdown(
                             options=[
                                 {'label': 'Normal', 'value': '0'},
                                 {'label': 'Not normal', 'value': '1'}
                             ],
                             value='0',
                             id='res_type'
                         )
                     ]), width={"size": 3}),
                     dbc.Col(html.Div([
                         html.Label('Avg_glucose_level: '),
                         dcc.Input(
                             type="number",
                             debounce=True,
                             value='1',
                             id='glucose_lvl'
                         )
                     ]), width={"size": 3}),

                 ], style={'padding': '10px 25px'}),
                 # dbc.Row([html.Div("Thallium stress test results",
                 #                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),

                 ], style={'padding': '10px 25px'}
                ),

        # Right hand column containing the summary information for predicted heart disease risk
        dbc.Col([html.Div("Predicted stroke risk",
                          style={'font-weight': 'bold', 'font-size': 20}),
                 dbc.Row(dcc.Graph(
                     id='Metric_1',
                     style={'width': '100%', 'height': 80},
                     config={'displayModeBar': False}
                 ), style={'marginLeft': 15}),
                 dbc.Row([html.Div(id='main_text', style={'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row([html.Div(["The figure below indicates the impact (magnitude of increase or decrease in "
                                    "log-odds) of factors on the model prediction of the patient's Stroke likelihood."
                                    " The figure calculates the odds (the ratio of something happening to something not happening)."
                                    " Finally in this figure we can see the log-odds which is equal to log(p/1-p)." 
                                    " Max value of log-odds is ln(100/1)=4.605 which is high risk with probability=p=100%"],
                                   style={'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row(dcc.Graph(
                     id='Metric_2',
                     config={'displayModeBar': False}
                 ), style={'marginLeft': 15}),
                 dbc.Row([html.Div(id='action_header',
                                   style={'font-weight': 'bold', 'font-size': 16, 'padding': '10px 25px'})]),
                 dbc.Row(
                     dbc.Col([html.Div(id='recommended_action')], width={"size": 11},
                             style={'font-size': 16, 'padding': '10px 25px',
                                    'backgroundColor': '#E2E2E2', 'marginLeft': 25})),
                 ],
                style={'padding': '10px 25px'}

                ),
    ]),

    html.Div(id='data_patient', style={'display': 'none'}),
]
)




# Responsive element: create X matrix for input to model estimation
@app.callback(
    Output('data_patient', 'children'),
    [Input('res_type', 'value'),
     Input('glucose_lvl', 'value'),
     Input('resting_bp', 'value'),
     Input('married', 'value'),
     Input('sex_male', 'value'),
     Input('age', 'value'),
     Input('ht', 'value'),
     Input('maximum_hr', 'value'),
     Input('work_type', 'value'),
     Input('smoke_status', 'value'),


     ]
)
def generate_feature_matrix(res_type,glucose_lvl, resting_bp, married,sex_male,age,ht, maximum_hr,
                            work_type, smoke_status
                             ):

    # generate a new X_matrix for use in the predictive models
    column_names = ['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level', 'bmi','smoking_status']



    values = [sex_male,age,maximum_hr,ht,married,work_type,res_type,glucose_lvl, resting_bp,smoke_status]

    x_patient = pd.DataFrame(data=[values],
                             columns=column_names,
                             index=[0])
    print(x_patient)
    return x_patient.to_json()


@app.callback(
    [Output('Metric_1', 'figure'),
     Output('main_text', 'children'),
     Output('action_header', 'children'),
     Output('recommended_action', 'children'),
     Output('Metric_2', 'figure')],
    [Input('data_patient', 'children')]
)
def predict_hd_summary(data_patient):

    # read in data and predict likelihood of heart disease
    x_new = pd.read_json(data_patient)
    pred = hdpred_model.predict(x_new)
    print(pred)
    y_val = hdpred_model.predict_proba(x_new)[:,1]*100
    print(y_val)
    print(hdpred_model.predict_proba(x_new)*100)
    # maxiY = np.argmax(y_val)
    # y = y_val[maxiY]*100
    text_val = str(np.round(y_val, 1)) + "%"
    print(text_val)
    # assign a risk group
    if y_val/100 <= 0.275685:
        risk_grp = 'low risk'
    elif y_val/100 <= 0.795583:
        risk_grp = 'medium risk'
    else:
        risk_grp = 'high risk'

    # assign an action related to the risk group
    rg_actions = {'low risk': ['Discuss with patient any single large risk factors they may have, and otherwise '
                               'continue supporting healthy lifestyle habits. Follow-up in 12 months'],
                  'medium risk': ['Discuss lifestyle with patient and identify changes to reduce risk. '
                                  'Schedule follow-up with patient in 3 months on how changes are progressing. '
                                  'Recommend performing simple tests to assess positive impact of changes.'],
                  'high risk': ['Immediate follow-up with patient to discuss next steps including additional '
                                'follow-up tests, lifestyle changes and medications.']}

    next_action = rg_actions[risk_grp][0]

    # create a single bar plot showing likelihood of heart disease
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=[''],
        x=y_val,
        marker_color='rgb(112, 128, 144)',
        orientation='h',
        width=1,
        text=text_val,
        textposition='auto',
        hoverinfo='skip'
    ))

    # add blocks for risk groups
    bot_val = 0.5
    top_val = 1

    fig1.add_shape(
        type="rect",
        x0=0,
        y0=bot_val,
        x1=0.275686 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="green"
    )
    fig1.add_shape(
        type="rect",
        x0=0.275686 * 100,
        y0=bot_val,
        x1=0.795584 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="orange"
    )
    fig1.add_shape(
        type="rect",
        x0=0.795584 * 100,
        y0=bot_val,
        x1=1 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="red"
    )
    fig1.add_annotation(
        x=0.275686 / 2 * 100,
        y=0.75,
        text="Low risk",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.add_annotation(
        x=0.53 * 100,
        y=0.75,
        text="Medium risk",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.add_annotation(
        x=0.9 * 100,
        y=0.75,
        text="High risk",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.update_layout(margin=dict(l=0, r=50, t=10, b=10), xaxis={'range': [0, 100]})
    # do shap value calculations for basic waterfall plot
    explainer_patient = shap.TreeExplainer(hdpred_model)
    shap_values_patient = explainer_patient.shap_values(x_new)

    updated_fnames = x_new.T.reset_index()

    updated_fnames.columns = ['feature', 'value']
    updated_fnames['shap_original'] = pd.Series(shap_values_patient[0])

    updated_fnames['shap_abs'] = updated_fnames['shap_original'].abs()
    updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)

    # need to collapse those after first 9, so plot always shows 10 bars
    show_features = 10
    num_other_features = updated_fnames.shape[0] - show_features
    col_other_name = f"{num_other_features} other features"
    f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
    f_group['feature'] = col_other_name
    plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])
    print(plot_data)
    # additional things for plotting
    plot_range = plot_data['shap_original'].cumsum().max() - plot_data['shap_original'].cumsum().min()
    plot_data['text_pos'] = np.where(plot_data['shap_original'].abs() > (1/9)*plot_range, "inside", "outside")
    plot_data['text_col'] = "white"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] < 0), 'text_col'] = "#3283FE"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] > 0), 'text_col'] = "#F6222E"

    fig2 = go.Figure(go.Waterfall(
        name="",
        orientation="h",
        measure=['absolute'] + ['relative']*show_features,
        base=explainer_patient.expected_value[0],
        textposition=plot_data['text_pos'],
        text=plot_data['shap_original'],
        textfont={"color": plot_data['text_col']},
        texttemplate='%{text:+.2f}',
        y=plot_data['feature'],
        x=plot_data['shap_original'],
        connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
        decreasing={"marker": {"color": "#3283FE"}},
        increasing={"marker": {"color": "#F6222E"}},
        hoverinfo="skip"
    ))
    fig2.update_layout(
        waterfallgap=0.2,
        autosize=False,
        width=800,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor='black',
            tickcolor='black',
            ticks='outside',
            ticklen=5
        ),
        margin={'t': 25, 'b': 50},
        shapes=[
            dict(
                type='line',
                yref='paper', y0=0, y1=1.02,
                xref='x', x0=plot_data['shap_original'].sum(),#+explainer_patient.expected_value,
                x1=plot_data['shap_original'].sum(),#+explainer_patient.expected_value,
                layer="below",
                line=dict(
                    color="black",
                    width=1,
                    dash="dot")
            )
        ]
    )
    fig2.update_yaxes(automargin=True)
    # print(explainer_patient.expected_value)
    # fig2.add_annotation(
    #     yref='paper',
    #     xref='x',
    #     x=explainer_patient.expected_value,
    #     y=-0.12,
    #     text="E[f(x)] = " +str(explainer_patient.expected_value),
    #     showarrow=False,
    #     font=dict(color="black", size=14)
    # )

    fig2.add_annotation(
        yref='paper',
        xref='x',
        x=plot_data['shap_original'].sum(),#+explainer_patient.expected_value,
        y=1.075,
        text="odds ratio = " +str(plot_data['shap_original'].sum()),#+explainer_patient.expected_value)),
        showarrow=False,
        font=dict(color="black", size=14)
    )

    return fig1, \
           f"Based on the patient's profile, the predicted likelihood of Stroke is {text_val}. " \
           f"This patient is in the {risk_grp} group.", \
           f"Recommended action(s) for a patient in the {risk_grp} group", \
           next_action, \
           fig2

    #
    # return fig1, \
    #        f"Based on the patient's profile, the predicted likelihood of Stroke is {text_val}. " \
    #        f"This patient is in the {risk_grp} group."


if __name__ == '__main__':
    app.run_server(debug=False)