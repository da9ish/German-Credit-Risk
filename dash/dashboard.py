import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff
from plotly import tools
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import preprocessing
from sklearn import svm, tree
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_curve, auc
# from gunicorn import serve

checking_ac = {
    'A11': 0,
    'A14': 1,
    'A12': 2,
    'A13': 3
}

saving_ac = {
    'A61': 0,
    'A64': 1,
    'A62': 2,
    'A63': 3,
    'A64': 4,
    'A65': 5
}

employment_status = {
    'A71' : 0,
    'A72' : 1,
    'A73' : 2,
    'A74' : 3,
    'A75' : 4
}

job = {
    'A171': 0,
    'A172': 1,
    'A173': 2,
    'A174': 3
}

telephone = {
    'A191': 0,
    'A192': 1
}

foreign_worker = {
    'A201': 0,
    'A202': 1
}

num_cols = ["duration", "amount", "age"]

def load_data():
    columns = ["checking_ac", "duration", "credit_history", "purpose", "amount", "saving_ac",
               "employment_status", "installment_rate", 'personal_status_sex', "debtor_guarantor", "residence_since",
               "property", "age", "installment_plan", "housing", "existing_credits", "job", "liable_count", "telephone",
               "foreign_worker", "target"]
    df = pd.read_csv("../data/german.data2.csv", delimiter=' ',
                     index_col=False, names=columns)
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
                    'type': 'histogram'
                }],
            'layout':{
                'title':"Count Plot"
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
    fig.layout.title = "Distribution Plot"
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

def roc_plot(fpr, tpr, roc_auc):
    lw = 2
    trace1 = go.Scatter(x=fpr, y=tpr, 
                    mode='lines', 
                    line=dict(color='darkorange', width=lw),
                    name='ROC curve (area = %0.2f)' % roc_auc
                   )

    trace2 = go.Scatter(x=[0, 1], y=[0, 1], 
                        mode='lines', 
                        line=dict(color='navy', width=lw, dash='dash'),
                        showlegend=False)

    layout = go.Layout(title='Receiver operating characteristic example',
                    xaxis=dict(title='False Positive Rate'),
                    yaxis=dict(title='True Positive Rate'))

    fig = go.Figure(data=[trace1, trace2], layout=layout)
    return fig

def splitting(df):
    X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df.iloc[:, 20:21], test_size=0.2, random_state=3)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=3)
    return X_train, y_train, X_val, y_val, X_test, y_test

def get_encoding_cols(X_train):
    cat_cols_X_train = list(set(X_train.columns.values) - set(num_cols))
    cat_cols = []
    for col in X_train.columns:
        if len(X_train[col].unique()) < 11:
            cat_cols.append(col)
            X_train[col]=X_train[col].astype('category')

    level_encoding_cols = ["checking_ac", "saving_ac", "employment_status", "installment_rate", "job", "residence_since", "liable_count", "existing_credits", 'telephone', 'foreign_worker']
    one_hot_encoding_cols = list(set(cat_cols_X_train) - set(level_encoding_cols))
    return level_encoding_cols, one_hot_encoding_cols

def label_encoding(X_train, X_test, X_val):
    x_col = {'checking_ac': checking_ac, 'saving_ac': saving_ac, 'employment_status': employment_status, 'job': job, 'telephone': telephone, 'foreign_worker': foreign_worker}
    for indx, val in x_col.items():
        col = indx
        rep_dict = val
        X_train[col].replace(rep_dict, inplace=True)
        X_val[col].replace(rep_dict, inplace=True)
        X_test[col].replace(rep_dict, inplace=True)

def one_hot_encoding(train, val, test, col_name):
    x = pd.get_dummies(train[col_name])
    y = pd.get_dummies(test[col_name])
    z = pd.get_dummies(val[col_name])
    for i in x.columns:
        train[col_name + ' is ' + i + '?'] = x[i]
        
    for i in z.columns:
        val[col_name + ' is ' + i + '?'] = z[i]
        
    for i in y.columns:
        test[col_name + ' is ' + i + '?'] = y[i]

    train.drop(col_name, axis=1, inplace=True)
    val.drop(col_name, axis=1, inplace=True)
    test.drop(col_name, axis=1, inplace=True)

def log_reg(X_train, y_train, X_val, y_val, X_test, y_test):
    reg = LogisticRegression()
    reg.fit(X_train, y_train.values.ravel())
    predict = reg.predict(X_val)
    acc = accuracy_score(y_val, predict)
    print("Accuracy on Validation Set: " + str(acc))
    predict = reg.predict(X_test)
    acc = accuracy_score(y_test, predict)
    print("Accuracy on Test Set: " + str(acc))
    return reg

def get_roc_metrics(model, X_test, y_test):
    probs = model.predict_proba(X_test)
    # y_pred = model.predict(X_test)
    preds = probs[:,1]
    print(preds, y_test)
    fpr, tpr, treshold = roc_curve(y_test, preds)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc
    
df = load_data()
df.target.replace({2: 0}, inplace=True)
X_train, y_train, X_val, y_val, X_test, y_test = splitting(df)
print("X Train: " + str(X_train.shape))
print("Y Train: " + str(y_train.shape))

print("X Test: " + str(X_test.shape))
print("Y Test: " + str(y_test.shape))

print("X Val: " + str(X_val.shape))
print("Y Val: " + str(y_val.shape))
level_encoding_cols, one_hot_encoding_cols = get_encoding_cols(X_train)
label_encoding(X_train, X_test, X_val)
for i in one_hot_encoding_cols:
    one_hot_encoding(X_train, X_val, X_test, i)

reg = log_reg(X_train, y_train, X_val, y_val, X_test, y_test)
fpr, tpr, roc_auc = get_roc_metrics(reg, X_test, y_test)
roc_fig = (roc_plot(fpr, tpr, roc_auc))

app = dash.Dash()
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True

x = np.random.randn(1000)  
hist_data = [x]
group_labels = ['distplot']

fig = ff.create_distplot(hist_data, group_labels)

app.layout = html.Div([
    html.H1('German Credit Scoring'),
    dcc.Dropdown(
        options=[
            {'label': 'Select Category', 'value': ''},
            {'label': 'Numerical Columns', 'value': 'NC'},
            {'label': 'Categorical Columns', 'value': 'CC'},
        ],
        value='',
        id='category_dd'
    ),
    dcc.Dropdown(
        options=[
            {'label': 'Select Column', 'value': ''},
        ],
        id='col_dd',
        value=''
    ),
    html.Div([
        dcc.Graph(
        id='dist-plot', 
        ),
        dcc.Graph(
            id='box-plot',    
        )
    ], className='none', id='num_plots'),
    html.Div([
        dcc.Graph(
            id='count-plot',
        ),
        dcc.Graph(
            id='box-plot', 
        ),
    ], className='none', id='cat_plots'),
    html.Div([
        html.H2('ROC Curve'),
        dcc.Graph(
            id='roc-curve',
            figure=roc_fig
        )
    ])
], className="container")


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

# @app.callback(
#     Output(component_id='qq-plot', component_property='figure'),
#     [Input(component_id='col_dd', component_property='value')]
# )
# def show_qq_plot(col_name):
#     return qq_plot(col_name)

@app.callback(
    Output(component_id='num_plots', component_property='className'),
    [Input(component_id='category_dd', component_property='value')]
)
def set_display_block_num(category):
    print(category)
    if category == "NC":
        return 'block'
    else:
        return 'none'

@app.callback(
    Output(component_id='cat_plots', component_property='className'),
    [Input(component_id='category_dd', component_property='value')]
)
def set_display_block_cat(category):
    print(category)
    if category == "CC":
        return 'block'
    else:
        return 'none'

@app.callback(
    Output(component_id='count-plot', component_property='figure'),
    [Input(component_id='col_dd', component_property='value')]
)
def show_count_plot(col_name):
    return count_plot(col_name)

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

@app.server.route('/static/<path:path>')
def static_file(path):
    static_folder = os.path.join(os.getcwd(), 'static')
    return send_from_directory(static_folder, path)

if __name__ == '__main__':
    app.run_server(debug=True)
