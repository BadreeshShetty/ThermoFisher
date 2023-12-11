import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

from datetime import datetime
import plotly.express as px
import matplotlib.pyplot as plt

from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.varmax import VARMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from prophet import Prophet

import warnings
warnings.filterwarnings('ignore')

import plotly.graph_objects as go
import streamlit as st



st.set_page_config(
    page_title="Time Series Approach",
    page_icon="👾",
    layout="wide"
)

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_percentage_error.html
# https://stackoverflow.com/questions/47648133/mape-calculation-in-python
def percentage_error(actual, predicted):
    res = np.empty(actual.shape)
    for j in range(actual.shape[0]):
        if actual[j] != 0:
            res[j] = (actual[j] - predicted[j]) / actual[j]
        else:
            res[j] = predicted[j] / np.mean(actual)
    return res

def mape(y_true, y_pred): 
    return np.mean(np.abs(percentage_error(np.asarray(y_true), np.asarray(y_pred)))) * 100

# https://stackoverflow.com/questions/49604269/run-all-regressors-against-the-data-in-scikit
def evaluate_model(model, train, test, model_name):
    
#     exog_cols=train.columns.tolist()
#     exog_cols.remove('amount_p$_sum')
#     exog_cols.remove('reference_date')
    # fit the model
    model=model.fit()
    
    if model_name!='Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)':
    # predict the values using training data
        train_pred = model.predict(start=train.index[0], end=train.index[-1])
    else:
        train_pred = model.predict(start=train.index[0], exog=train[exog_cols], end=train.index[-1])

    
    # evaluate using training data
    train_mape = mape(train['amount_p$_sum'], train_pred)
    
    if model_name!='Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)':
    # predict the values using testing data
        test_pred = model.predict(start=test.index[0], end=test.index[-1])
    else:
        test_pred = model.predict(start=test.index[0], exog=test[exog_cols], end=test.index[-1])
        
    exog=test[exog_cols]
    # evaluate using testing data
    test_mape = mape(test['amount_p$_sum'], test_pred)
    
    plot_model(train,test,train_pred,test_pred,model_name)
    
    print("Mean Absolute Percentage Error: {:.2f}\n".format(train_mape))
    print("Mean Absolute Percentage Error: {:.2f}\n".format(test_mape))
            
    return train_mape, test_mape



def plot_model(train, test, train_pred, test_pred, model_name):
    # Combine train and test data for Plotly express
    plot_data = pd.concat([train.assign(dataset='Train', prediction=train_pred),
                           test.assign(dataset='Test', prediction=test_pred)])

    # Create an interactive Plotly Express line plot
    fig = px.line(plot_data, x=plot_data.index, y='amount_p$_sum', color='dataset',
                  labels={'amount_p$_sum': 'Amount in Millions($)', 'index': 'Date in Weeks'},
                  title=str(model_name)+' - Train and Test Data vs. Predictions')

    # Add prediction line for both train and test
    fig.add_trace(px.line(plot_data, x=plot_data.index, y='prediction', color='dataset').data[0])

    fig.update_layout(height=800)
    # Show the plot in Streamlit
    st.plotly_chart(fig,use_container_width=True,height=800)



def plot_model(train, test, train_pred, test_pred, model_name):
    # Create a Plotly figure
    fig = go.Figure()

    # Plot training data
    fig.add_trace(go.Scatter(x=train.index, y=train['amount_p$_sum'], mode='lines', name='Actual Train Data', line=dict(color='blue')))

    # Plot actual test data
    fig.add_trace(go.Scatter(x=test.index, y=test['amount_p$_sum'], mode='lines', name='Actual Test Data', line=dict(color='green')))

    # Plot predicted training data
    fig.add_trace(go.Scatter(x=train.index, y=train_pred, mode='lines', name=f'{model_name} Train', line=dict(color='red')))

    # Plot predicted test data
    fig.add_trace(go.Scatter(x=test.index, y=test_pred, mode='lines', name=f'{model_name} Test', line=dict(color='orange')))

    # Update layout
    fig.update_layout(
        # legend_title=f'{model_name} - Train and Test Data vs. Predictions',
        title=str(model_name) + ' - Train and Test Data vs. Predictions',
        xaxis_title='Date in Weeks',
        yaxis_title='Amount in Millions($)',
    )

    # Show the Plotly figure using Streamlit
    fig.update_layout(height=800)

    st.plotly_chart(fig,use_container_width=True,height=800)
    

invoice_df=pd.read_excel("InvoiceData_concate_cleaned_grouped_flu_stock_weekly.xlsx")


normal_df = invoice_df.reset_index()
normal_df.head()


train_size=18
train, test = normal_df.iloc[:-train_size], normal_df.iloc[-train_size:]

normal_df['amount_p$_sum'].plot(figsize=(12, 6), label='Original Data')
plt.title('Time Series Data')
plt.show()


exog_cols=invoice_df.columns.tolist()
exog_cols.remove('amount_p$_sum')
exog_cols.remove('reference_date')


base_models = [
    ('Simple Exponential Smoothing (SES)',SimpleExpSmoothing(train['amount_p$_sum'])),
    ("Holt/'s Simple Exponential Smoothing (HWES)",Holt(train['amount_p$_sum'])),
    ('Auto Regression (AR)',AutoReg(train['amount_p$_sum'],lags=1)),
    ('Moving Average (MA)',ARIMA(train['amount_p$_sum'], order=(0, 0, 1))),
    ('Autoregressive Moving Average (ARMA)',ARIMA(train['amount_p$_sum'], order=(2, 0, 1))),
    ('Autoregressive Integrated Moving Average (ARIMA)',ARIMA(train['amount_p$_sum'], order=(1, 1, 1))),
    ('Seasonal Autoregressive Integrated Moving-Average (SARIMA)',SARIMAX(train['amount_p$_sum'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))),
    ('Seasonal Autoregressive Integrated Moving-Average with Exogenous Regressors (SARIMAX)',SARIMAX(train['amount_p$_sum'], exog=train[exog_cols], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12)))  
]

train_mape_l, test_mape_l, model_name_l =[], [], []

for name, model_n in base_models:
#     start_time = time.time()
    train_mape, test_mape=\
    evaluate_model(model_n, train, test,name)
    train_mape_l.append(train_mape)
    test_mape_l.append(test_mape)
    model_name_l.append(name)
#     end_time = time.time()
#     print("Computation Time: {}".format(end_time - start_time))
#     print("----------------------------------\n")
    
