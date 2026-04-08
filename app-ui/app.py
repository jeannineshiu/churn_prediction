import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'app-ml' / 'src'))
os.chdir(project_root)

from common.utils import read_config, load_model
from common.data_manager import DataManager
from pipelines.preprocessing import PreprocessingPipeline
from pipelines.feature_engineering import FeatureEngineeringPipeline

# ─── Data & Model Setup ───────────────────────────────────────────────────────
config = read_config(project_root / 'config' / 'config.yaml')
model = load_model(config['pipeline_runner']['model_path'])
data_manager = DataManager(config)
preprocessing = PreprocessingPipeline(config)
feature_eng = FeatureEngineeringPipeline(config)

raw_df = data_manager.load_data(data_manager.get_raw_data_path())
processed_df = preprocessing.run(raw_df.copy())   # categoricals as strings, Churn as 0/1
encoded_df = feature_eng.run(processed_df.copy()) # all numeric

X_train, X_test, y_train, y_test = data_manager.split_data(encoded_df.copy())

# Raw test customers (not encoded) for "Random Customer" feature
raw_test = processed_df.drop(columns=['Churn']).loc[X_test.index]
y_test_labels = processed_df['Churn'].loc[X_test.index]
test_indices = list(X_test.index)

# Fit LabelEncoders on the full processed data (same behaviour as feature_eng pipeline)
cat_cols = processed_df.select_dtypes(include='object').columns.tolist()
encoders = {col: LabelEncoder().fit(processed_df[col]) for col in cat_cols}
feat_names = list(X_test.columns)

# Pre-compute test set performance
y_pred_all = model.predict(X_test).flatten().astype(int)
y_prob_all = model.predict_proba(X_test)[:, 1]
cm_vals = confusion_matrix(y_test, y_pred_all)
fpr, tpr, _ = roc_curve(y_test, y_prob_all)
roc_auc_val = auc(fpr, tpr)
accuracy = (y_pred_all == y_test.values).mean()
churn_recall = cm_vals[1, 1] / cm_vals[1].sum()

feat_imp_df = (
    pd.DataFrame({'feature': feat_names, 'importance': model.get_feature_importance()})
    .sort_values('importance', ascending=True)
    .tail(15)
)

# ─── Encoding helper ──────────────────────────────────────────────────────────
def encode_customer(raw: dict) -> pd.DataFrame:
    """Encode a single customer's raw values into the model's expected format."""
    row = {}
    for col, val in raw.items():
        if col in encoders:
            row[col] = int(encoders[col].transform([val])[0])
        else:
            row[col] = float(val) if val is not None else 0.0
    return pd.DataFrame([row])[feat_names]

# ─── Form helpers ─────────────────────────────────────────────────────────────
CAT_OPTIONS = {
    'gender':           ['Female', 'Male'],
    'Partner':          ['No', 'Yes'],
    'Dependents':       ['No', 'Yes'],
    'PhoneService':     ['No', 'Yes'],
    'MultipleLines':    ['No', 'No phone service', 'Yes'],
    'InternetService':  ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity':   ['No', 'No internet service', 'Yes'],
    'OnlineBackup':     ['No', 'No internet service', 'Yes'],
    'DeviceProtection': ['No', 'No internet service', 'Yes'],
    'TechSupport':      ['No', 'No internet service', 'Yes'],
    'StreamingTV':      ['No', 'No internet service', 'Yes'],
    'StreamingMovies':  ['No', 'No internet service', 'Yes'],
    'Contract':         ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['No', 'Yes'],
    'PaymentMethod':    [
        'Bank transfer (automatic)', 'Credit card (automatic)',
        'Electronic check', 'Mailed check'
    ],
}

def dropdown(fid, options, value=None):
    return dcc.Dropdown(
        id=fid,
        options=[{'label': str(o), 'value': o} for o in options],
        value=value if value is not None else options[0],
        clearable=False,
        style={'fontSize': '13px'}
    )

def form_row(label, component):
    return dbc.Row([
        dbc.Col(
            html.Label(label, style={
                'fontSize': '12px', 'fontWeight': '500',
                'color': '#555', 'lineHeight': '34px', 'marginBottom': 0
            }),
            width=5
        ),
        dbc.Col(component, width=7),
    ], className='mb-1 align-items-center')

INPUT_IDS = [
    'input-gender', 'input-SeniorCitizen', 'input-Partner', 'input-Dependents',
    'input-tenure', 'input-Contract', 'input-PaperlessBilling', 'input-PaymentMethod',
    'input-MonthlyCharges', 'input-TotalCharges', 'input-PhoneService',
    'input-MultipleLines', 'input-InternetService', 'input-OnlineSecurity',
    'input-OnlineBackup', 'input-DeviceProtection', 'input-TechSupport',
    'input-StreamingTV', 'input-StreamingMovies',
]
FEATURE_COLS = [fid.replace('input-', '') for fid in INPUT_IDS]

# ─── Static figures ───────────────────────────────────────────────────────────
def build_cm_figure():
    labels = ['No Churn', 'Churn']
    fig = go.Figure(go.Heatmap(
        z=cm_vals[::-1], x=labels, y=labels[::-1],
        colorscale=[[0, '#eaf4fb'], [1, '#1a6fa8']],
        text=cm_vals[::-1], texttemplate='%{text}',
        textfont={'size': 22}, showscale=False,
    ))
    fig.update_layout(
        xaxis_title='Predicted', yaxis_title='Actual',
        margin=dict(l=10, r=10, t=10, b=10),
        height=260, plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig

def build_roc_figure():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'AUC = {roc_auc_val:.3f}',
        line=dict(color='#1a6fa8', width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines',
        name='Random', line=dict(color='#aaa', dash='dash', width=1.5)
    ))
    fig.update_layout(
        xaxis_title='False Positive Rate', yaxis_title='True Positive Rate',
        margin=dict(l=10, r=10, t=10, b=10),
        height=260, plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(x=0.55, y=0.05, font=dict(size=11))
    )
    return fig

def build_feat_imp_figure():
    fig = go.Figure(go.Bar(
        x=feat_imp_df['importance'],
        y=feat_imp_df['feature'],
        orientation='h',
        marker_color='#1a6fa8',
        hovertemplate='%{y}: %{x:.1f}<extra></extra>'
    ))
    fig.update_layout(
        xaxis_title='Importance Score',
        margin=dict(l=10, r=10, t=10, b=10),
        height=380, plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig

# ─── Layout ───────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title='Churn Prediction Demo'
)

# --- Form Card ---
form_card = dbc.Card([
    dbc.CardHeader(html.Strong('Customer Profile')),
    dbc.CardBody([
        html.Small('DEMOGRAPHICS', className='text-muted fw-bold'),
        form_row('Gender',        dropdown('input-gender', CAT_OPTIONS['gender'])),
        form_row('Senior Citizen', dropdown('input-SeniorCitizen', [0, 1], value=0)),
        form_row('Partner',       dropdown('input-Partner', CAT_OPTIONS['Partner'])),
        form_row('Dependents',    dropdown('input-Dependents', CAT_OPTIONS['Dependents'])),
        html.Hr(className='my-2'),

        html.Small('ACCOUNT & BILLING', className='text-muted fw-bold'),
        form_row('Tenure (months)', dcc.Input(
            id='input-tenure', type='number', value=12, min=1, max=72,
            debounce=True, style={'width': '100%', 'fontSize': '13px', 'height': '34px'}
        )),
        form_row('Contract',          dropdown('input-Contract', CAT_OPTIONS['Contract'])),
        form_row('Paperless Billing', dropdown('input-PaperlessBilling', CAT_OPTIONS['PaperlessBilling'])),
        form_row('Payment Method',    dropdown('input-PaymentMethod', CAT_OPTIONS['PaymentMethod'])),
        form_row('Monthly Charges', dcc.Input(
            id='input-MonthlyCharges', type='number', value=65.0,
            debounce=True, style={'width': '100%', 'fontSize': '13px', 'height': '34px'}
        )),
        form_row('Total Charges', dcc.Input(
            id='input-TotalCharges', type='number', value=780.0,
            debounce=True, style={'width': '100%', 'fontSize': '13px', 'height': '34px'}
        )),
        html.Hr(className='my-2'),

        html.Small('SERVICES', className='text-muted fw-bold'),
        form_row('Phone Service',     dropdown('input-PhoneService', CAT_OPTIONS['PhoneService'])),
        form_row('Multiple Lines',    dropdown('input-MultipleLines', CAT_OPTIONS['MultipleLines'])),
        form_row('Internet Service',  dropdown('input-InternetService', CAT_OPTIONS['InternetService'])),
        form_row('Online Security',   dropdown('input-OnlineSecurity', CAT_OPTIONS['OnlineSecurity'])),
        form_row('Online Backup',     dropdown('input-OnlineBackup', CAT_OPTIONS['OnlineBackup'])),
        form_row('Device Protection', dropdown('input-DeviceProtection', CAT_OPTIONS['DeviceProtection'])),
        form_row('Tech Support',      dropdown('input-TechSupport', CAT_OPTIONS['TechSupport'])),
        form_row('Streaming TV',      dropdown('input-StreamingTV', CAT_OPTIONS['StreamingTV'])),
        form_row('Streaming Movies',  dropdown('input-StreamingMovies', CAT_OPTIONS['StreamingMovies'])),
    ], style={'overflowY': 'auto', 'maxHeight': 'calc(100vh - 230px)', 'padding': '12px'}),
])

buttons_row = dbc.Row([
    dbc.Col(dbc.Button('🎲  Random Customer', id='btn-random',
                       color='secondary', className='w-100 mt-2'), width=6),
    dbc.Col(dbc.Button('🔮  Predict',         id='btn-predict',
                       color='primary',   className='w-100 mt-2'), width=6),
])

# --- Prediction tab ---
prediction_tab = dbc.Row([
    dbc.Col([form_card, buttons_row], width=4),
    dbc.Col([
        html.Div(id='true-label-badge'),
        dbc.Card(
            dbc.CardBody(
                id='prediction-result',
                children=[html.Div(
                    '← Fill in a customer profile and click Predict',
                    style={'color': '#aaa', 'textAlign': 'center',
                           'paddingTop': '50px', 'fontSize': '15px'}
                )]
            ),
            id='result-card',
            style={'minHeight': '190px'}
        ),
        dbc.Card([
            dbc.CardHeader(html.Strong('Churn Probability')),
            dbc.CardBody(dcc.Graph(
                id='prob-gauge',
                config={'displayModeBar': False},
                style={'height': '170px'}
            )),
        ], className='mt-3'),
    ], width=8),
], className='mt-3 g-3')

# --- Overview tab ---
def metric_card(value, label):
    return dbc.Card(dbc.CardBody([
        html.H4(value, className='text-center mb-0 fw-bold'),
        html.P(label, className='text-center text-muted mb-0', style={'fontSize': '13px'})
    ]), color='light')

overview_tab = html.Div([
    dbc.Row([
        dbc.Col(metric_card(f'{accuracy:.1%}',    'Accuracy'),     width=3),
        dbc.Col(metric_card(f'{churn_recall:.1%}', 'Churn Recall'), width=3),
        dbc.Col(metric_card(f'{roc_auc_val:.3f}',  'ROC AUC'),      width=3),
        dbc.Col(metric_card(str(len(X_test)),      f'Test Samples ({int(y_test.sum())} churned)'), width=3),
    ], className='mt-3 mb-3'),

    dbc.Row([
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Strong('Confusion Matrix')),
            dbc.CardBody(dcc.Graph(
                figure=build_cm_figure(), config={'displayModeBar': False}
            )),
        ]), width=6),
        dbc.Col(dbc.Card([
            dbc.CardHeader(html.Strong('ROC Curve')),
            dbc.CardBody(dcc.Graph(
                figure=build_roc_figure(), config={'displayModeBar': False}
            )),
        ]), width=6),
    ], className='mb-3'),

    dbc.Card([
        dbc.CardHeader(html.Strong('Feature Importance (Top 15)')),
        dbc.CardBody(dcc.Graph(
            figure=build_feat_imp_figure(), config={'displayModeBar': False}
        )),
    ]),
])

# --- App layout ---
app.layout = dbc.Container([
    dbc.Row(dbc.Col(
        html.Div([
            html.H3('Churn Prediction Dashboard', className='fw-bold mb-0 me-3', style={'display': 'inline'}),
            html.Small(
                'Telco Customer Churn  ·  CatBoostClassifier + Optuna  ·  IBM Telco Dataset',
                className='text-muted',
                style={'verticalAlign': 'middle'}
            ),
        ])
    ), className='py-2 border-bottom'),

    dbc.Tabs([
        dbc.Tab(prediction_tab, label='Single Prediction', tab_id='tab-predict'),
        dbc.Tab(overview_tab,   label='Model Overview',    tab_id='tab-overview'),
    ], active_tab='tab-predict', className='mt-2'),
], fluid=True, style={'backgroundColor': '#f8f9fa', 'minHeight': '100vh', 'padding': '0 20px'})


# ─── Callbacks ────────────────────────────────────────────────────────────────
@app.callback(
    [Output(fid, 'value') for fid in INPUT_IDS] + [Output('true-label-badge', 'children')],
    Input('btn-random', 'n_clicks'),
    prevent_initial_call=True
)
def load_random_customer(n_clicks):
    idx = int(np.random.choice(test_indices))
    row = raw_test.loc[idx]
    actual = y_test_labels.loc[idx]
    label, color = ('Churned ✗', 'danger') if actual == 1 else ('Retained ✓', 'success')
    badge = dbc.Alert(
        [html.Strong('True Label: '), label],
        color=color, className='py-2 mb-2', style={'fontSize': '13px'}
    )
    return [row[col] for col in FEATURE_COLS] + [badge]


@app.callback(
    [Output('prediction-result', 'children'),
     Output('result-card', 'style'),
     Output('prob-gauge', 'figure')],
    Input('btn-predict', 'n_clicks'),
    [State(fid, 'value') for fid in INPUT_IDS],
    prevent_initial_call=True
)
def predict(n_clicks, *values):
    raw = dict(zip(FEATURE_COLS, values))
    X = encode_customer(raw)
    prob = float(model.predict_proba(X)[0, 1])
    pred = int(prob >= 0.5)

    if pred == 1:
        content = [
            html.Div('⚠️', style={'fontSize': '60px', 'textAlign': 'center', 'paddingTop': '12px'}),
            html.H3('CHURN RISK', className='text-center fw-bold'),
            html.H5(f'{prob:.1%} churn probability', className='text-center opacity-75'),
        ]
        card_style = {'backgroundColor': '#dc3545', 'color': 'white', 'minHeight': '190px'}
    else:
        content = [
            html.Div('✅', style={'fontSize': '60px', 'textAlign': 'center', 'paddingTop': '12px'}),
            html.H3('LIKELY TO STAY', className='text-center fw-bold'),
            html.H5(f'{1 - prob:.1%} retention probability', className='text-center opacity-75'),
        ]
        card_style = {'backgroundColor': '#198754', 'color': 'white', 'minHeight': '190px'}

    gauge = go.Figure(go.Indicator(
        mode='gauge+number',
        value=prob * 100,
        number={'suffix': '%', 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1},
            'bar': {'color': '#dc3545' if pred else '#198754'},
            'steps': [
                {'range': [0, 50],  'color': '#d1e7dd'},
                {'range': [50, 100], 'color': '#f8d7da'},
            ],
            'threshold': {
                'line': {'color': '#333', 'width': 2},
                'value': 50, 'thickness': 0.75
            },
        },
        title={'text': 'Churn Probability (%)', 'font': {'size': 13}}
    ))
    gauge.update_layout(
        margin=dict(l=15, r=15, t=30, b=5),
        paper_bgcolor='white',
        height=165
    )

    return content, card_style, gauge


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=8050)
    args = parser.parse_args()
    app.run(debug=True, host='0.0.0.0', port=args.port)
