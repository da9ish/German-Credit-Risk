import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools



def load_data():
    columns = ["checking_ac", "duration", "credit_history", "purpose", "amount", "saving_ac",
               "employment_status", "installment_rate", 'personal_status_sex', "debtor_guarantor", "residence_since",
               "property", "age", "installment_plan", "housing", "existing_credits", "job", "liable_count", "telephone",
               "foreign_worker", "target"]
    df = pd.read_csv("../data/german.data2.csv", delimiter=' ',
                     index_col=False, names=columns)
    cat_dict = {
        "A11": "-0",
        "A12": "0-200",
        "A13": "200+",
        "A14": "no checking acc",
        "A30": "no credit",
        "A31": "all paid duly",
        "A32": "existing paid duly",
        "A33": "payment delay",
        "A34": "critical acc",
        "A40": "car (new)",
        "A41": "car (old)",
        "A42": "furniture/equipment",
        "A43": "radio/television",
        "A44": "domestic appliances",
        "A45": "repairs",
        "A46": "education",
        "A47": "vacation",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
        "A61": "-100",
        "A62": "100-500",
        "A63": "500-1000",
        "A64": "1000+",
        "A65": "no acc",
        "A71": "unemployed",
        "A72": "-1",
        "A73": "1-4",
        "A74": "4-7",
        "A75": "7+",
        "A91": "male-div/sep",
        "A92": "male-single",
        "A93": "male-married",
        "A94": "female-div/sep/mar",
        "A95": "female-single",
        "A101": "none",
        "A102": "co-applicant",
        "A103": "guarantor",
        "A121": "real est",
        "A122": "building",
        "A123": "car",
        "A124": "none",
        "A141": "bank",
        "A142": "store",
        "A143": "none",
        "A151": "rent",
        "A152": "own",
        "A153": "free",
        "A171": "unempl-no res",
        "A172": "unempl-res",
        "A173": "empl",
        "A174": "high empl",
        "A191": "yes",
        "A191": "no",
        "A201": "yes",
        "A202": "no",
    }

    df.replace(cat_dict, inplace=True)
    df.target.replace({2: 0}, inplace=True)
    return df


def get_cols(df):
    cat_cols = []
    for col in df.columns:
        if len(df[col].unique()) < 11:
            cat_cols.append(col)
            df[col] = df[col].astype('category')
    return cat_cols

def count_plot(col_name):
    return {
            'data':  [
                {
                    'x': df[col_name],
                    # 'xbins': dict(start=np.min(df.amount), size=0.25, end=np.max(df.amount)),
                    # 'text': ,
                    # 'customdata': ,
                    # 'name': '',
                    'type': 'histogram'
                }],
            'layout':{
                'title':"Histogram Frequency Counts"
            }
        }

def box_plot(col_name):
    trace = go.Box(
    y = list(df[col_name]),
    name = "Whiskers and Outliers",
    boxpoints = 'outliers',
    marker = dict(
        color = 'rgb(107,174,214)'),
    line = dict(
        color = 'rgb(107,174,214)')
    )
    data = [trace]
    layout = go.Layout(
    title = "Box Plot Styling Outliers"
    )
    fig = go.Figure(data=data,layout=layout)
    return fig

def distribution_plot(col_name): 
    hist_data = [df[col_name].tolist()]
    group_labels = [col_name]

    fig = ff.create_distplot(hist_data, group_labels)
    return fig

def qq_plot(col_name):
    return {
            "data": [
            {
            "name": "Sample",
            "marker": {
                "size": 8
            },
            "mode": "markers",
            "x": list(df[col_name]),
            "y": [1 for i in range(1000)],
            "type": "scatter"
            }
        ],
        "layout": {
            "yaxis": {
            "rangemode": "tozero",
            "nticks": 11,
            "title": "Y Axis"
            },
            "xaxis": {
            "rangemode": "tozero",
            "title": "X Axis"
            },
            "title": "QQ Plot"
        }
    }


df = load_data()
app = dash.Dash()

x = np.random.randn(1000)  
hist_data = [x]
group_labels = ['distplot']

fig = ff.create_distplot(hist_data, group_labels)

app.layout = html.Div([
    html.H1('Hello Dash'),
    dcc.Dropdown(
        options=[
            {'label': 'Numerical Columns', 'value': 'NC'},
            {'label': 'Categorical Columns', 'value': 'CC'},
        ],
        value='NC',
        id='category_dd'
    ),
    dcc.Dropdown(
        options=[
            {'label': [], 'value': []}
        ],
        id='col_dd',
        value='duration'
    ),
    dcc.Graph(
        id='count-plot',
    ),
    dcc.Graph(
        id='qq-plot',
    ),
    dcc.Graph(
        id='dist-plot', 
    ),
    dcc.Graph(
        id='box-plot', 
        
    ),
])


@app.callback(
    Output(component_id='col_dd', component_property='options'),
    [Input(component_id='category_dd', component_property='value')]
)
def get_columns(category):
    columns = get_cols(df)
    num_cols = list(df._get_numeric_data().columns)
    cat_cols = list(set(columns) - set(num_cols))
    f_list = []
    if category == 'NC':
        for i in num_cols:
            f_list.append({'label': i, 'value': i})
        return f_list
    else:
        for i in cat_cols:
            f_list.append({'label': i, 'value': i})
        return f_list

@app.callback(
    Output(component_id='count-plot', component_property='figure'),
    [Input(component_id='col_dd', component_property='value')]
)
def show_count_plot(col_name):
    return count_plot(col_name)

@app.callback(
    Output(component_id='qq-plot', component_property='figure'),
    [Input(component_id='col_dd', component_property='value')]
)
def show_qq_plot(col_name):
    return qq_plot(col_name)


@app.callback(
    Output(component_id='dist-plot', component_property='figure'),
    [Input(component_id='col_dd', component_property='value')]
)
def show_dist_plot(col_name):
    return distribution_plot(col_name)

@app.callback(
    Output(component_id='box-plot', component_property='figure'),
    [Input(component_id='col_dd', component_property='value')]
)
def show_box_plot(col_name):
    return box_plot(col_name)

if __name__ == '__main__':
    app.run_server(debug=True)
