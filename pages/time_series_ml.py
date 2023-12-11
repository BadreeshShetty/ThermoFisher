import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
plt.rcParams["figure.figsize"] = (20,7)

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregCustom import ForecasterAutoregCustom
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import grid_search_forecaster
from skforecast.model_selection import backtesting_forecaster
from skforecast.utils import save_forecaster
from skforecast.utils import load_forecaster

import streamlit as st
import plotly.express as px

st.set_page_config(
    page_title="Time Series ML Approach",
    page_icon="💹",
    layout="wide"
)


# https://towardsdatascience.com/choosing-the-correct-error-metric-mape-vs-smape-5328dec53fac
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# def mape(y_true, y_pred):
#     return mean_absolute_error(y_true, y_pred) * 100


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
def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, if_plot):
    # fit the model
    model.fit(X_train, y_train)

    # predict the values using training data
    train_pred = model.predict(X_train)

    # evaluate using training data
    train_rmse = rmse(y_train, train_pred)
    train_mae = mean_absolute_error(y_train, train_pred)
    train_mape = mape(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    # print the results of the training data
    print("---Training data results---\n")
    print("Root Mean Squared Error: {:.2f}\n".format(train_rmse))
    print("Mean Absolute Error: {:.2f}\n".format(train_mae))
    print("Mean Absolute Percentage Error: {:.2f}\n".format(train_mape))
    print("R Square: {:.2f}\n".format(train_r2))
    
    # predict the values using testing data
    test_pred = model.predict(X_test)

    # evaluate using testing data
    test_rmse = rmse(y_test, test_pred)
    test_mae = mean_absolute_error(y_test, test_pred)
    test_mape = mape(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
        # print the results of the testing data
    print("-----Testing data results-----\n")
    print("Root Mean Squared Error: {:.2f}\n".format(test_rmse))
    print("Mean Absolute Error: {:.2f}\n".format(test_mae))
    print("Mean Absolute Percentage Error: {:.2f}\n".format(test_mape))
    print("R Square: {:.2f}\n".format(test_r2))
    
    if if_plot=="Yes":
        plot_model(y_train,y_test,train_pred,test_pred,model_name)
    else:
        print("No plot")
        
    tree_explained=['EXTR','Gradient Boosting','Random Forest','Light GBM','Xgboost','Catboost']
    
    if model_name in tree_explained:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type='bar')
        shap.summary_plot(shap_values, X_train)

        shap.initjs()

#         for i in range(len(X_test)):
#             display(shap.force_plot(explainer.expected_value, shap_values[i], features=X_train.loc[i], 
#             feature_names=X_train.columns))
            
    return train_rmse, train_mae, train_mape, train_r2, test_rmse, test_mae, test_mape, test_r2

def plot_model(y_train,y_test,train_pred,test_pred,model_name):
    
    # Assuming you have a list of time points or indices
    time_points_train = range(len(y_train))
    time_points_test = range(len(y_train), len(y_train) + len(y_test))

    # Create a figure and axis for the time series plot
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(time_points_train, y_train, label='Actual Train Data', color='blue')

    # Plot testing data
    plt.plot(time_points_test, y_test, label='Actual Test Data', color='green')

    # Plot training predictions
    plt.plot(time_points_train, train_pred, label=str(model_name)+' Train Predictions', color='red')

    # Plot testing predictions
    plt.plot(time_points_test, test_pred, label=str(model_name)+' Test Predictions', color='orange')

    # Add labels and legend
    plt.xlabel('Time/Iterations')
    plt.ylabel('Values')
    plt.legend()

    # Set a title for the plot
    plt.title(str(model_name)+' - Train and Test Data vs. Predictions')

    # Show the plot
    plt.grid(True)
    plt.show()

def feature_importance(model, X_train, model_name):
    features = X_train.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)

    # customized number 
    num_features = 25

    plt.figure(figsize=(10,10))
    plt.title('Feature Importances')

    # only plot the customized number of features
    plt.barh(range(num_features), importances[indices[-num_features:]], color='r', align='center')
    plt.yticks(range(num_features), [features[i] for i in indices[-num_features:]])
    plt.xlabel(str(model_name)+'s Feature Importance')
    plt.show();




timed_df=pd.read_excel("InvoiceData_concate_cleaned_grouped_flu_stock_weekly.xlsx")
# timed_df.drop('Date', axis = 1,inplace=True)

timed_df=timed_df.fillna(0)


steps = 18
data_train = timed_df[:-steps]
data_test = timed_df[-steps:]

# Create a line plot using Plotly Express
fig = px.line()
fig.add_scatter(x=data_train.index, y=data_train['amount_p$_sum'], mode='lines', name='train')
fig.add_scatter(x=data_test.index, y=data_test['amount_p$_sum'], mode='lines', name='test')

# Update layout
fig.update_layout(title='Train and Test Data',
                  xaxis_title='Date in Weeks',
                  yaxis_title='Amount in Millions($)',
                #   legend_title='Data Type',
                  height=400, width=800)

# Show the plot
# fig.show()

st.plotly_chart(fig,use_container_width=True,height=800)

exog_cols=timed_df.columns.tolist()
exog_cols.remove('amount_p$_sum')
exog_cols.remove('reference_date')

forecaster = ForecasterAutoreg(
                 regressor = XGBRegressor(random_state=42),
                 lags = 4 )

param_grid = {
    'n_estimators': [100, 500],
    'max_depth': [3, 5, 10],
    'learning_rate': [0.01, 0.1]
}

# Lags used as predictors
lags_grid = [4, 20]

results_grid = grid_search_forecaster(
                   forecaster         = forecaster,
                   y                  = data_train['amount_p$_sum'],
                   exog               = data_train[exog_cols],
                   param_grid         = param_grid,
                   lags_grid          = lags_grid,
                   steps              = 18,
                   refit              = False,
                   metric             = mape,
#                    initial_train_size = len(timed_df.loc[:end_train]),
                   initial_train_size = int(len(data_train)*0.8),
                   fixed_train_size   = False,
                   return_best        = True,
                   n_jobs             = 'auto',
                   verbose            = False
               )



predictions = forecaster.predict(steps=18,exog= data_test[exog_cols])

fig = px.line()
fig.add_scatter(x=data_train.index, y=data_train['amount_p$_sum'], mode='lines', name='train')
fig.add_scatter(x=data_test.index, y=data_test['amount_p$_sum'], mode='lines', name='test')
fig.add_scatter(x=predictions.index, y=predictions.values, mode='lines', name='predictions')

# Update layout
fig.update_layout(title='Sk-Forecast (Hyper Parameter Tuned) Train, Test, and Predictions',
                  xaxis_title='Date in Weeks',
                  yaxis_title='Amount in Millions($)',
                #   legend_title='Legend',
                  height=400, width=800)

# Show the plot
# fig.show()
fig.update_layout(height=800)
st.plotly_chart(fig,use_container_width=True,height=800)

error_mape = mape(
                y_true = data_test['amount_p$_sum'],
                y_pred = predictions
            )

st.write(f"Test error (mape): {error_mape}")

feature_importances = forecaster.get_feature_importances().sort_values('importance', ascending=False)

fig = px.bar(feature_importances.head(20), y='feature', x='importance', orientation='h',
             title=f'Top 20 Feature Importances for SkForecast',
             labels={'importance': 'SkForecast Feature Importance'})

fig.update_layout(yaxis_title='Feature', xaxis_title='Importance', xaxis=dict(tickformat='%'), yaxis={'categoryorder':'total ascending'})

# Format x-axis ticks to display only two decimals
fig.update_xaxes(tickformat=".2%")

# Set bar color to red
fig.update_traces(marker_color='red')

fig.update_layout(height=800)
st.plotly_chart(fig,use_container_width=True,height=800)


