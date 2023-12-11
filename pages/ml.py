import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler,MinMaxScaler,MaxAbsScaler

from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import ExtraTreeRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.svm import LinearSVR

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

import time
import shap

pd.options.mode.chained_assignment = None

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

st.set_page_config(
    page_title="Machine Learning Approach",
    page_icon="👩‍💻",
    layout="wide"
)
st.set_option('deprecation.showPyplotGlobalUse', False)


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
def evaluate_model(mas, model, X_train, y_train, X_test, y_test, model_name, if_plot):
    # fit the model
    model.fit(X_train, y_train)

    # predict the values using training data
    train_pred = model.predict(X_train)

    # evaluate using training data
    train_rmse = rmse(y_train, train_pred)
#     train_mae = mean_absolute_error(y_train, train_pred)
    train_mape = mape(y_train, train_pred)
    train_r2 = r2_score(y_train, train_pred)
    
    # print the results of the training data
    st.write("---Model: "+str(model_name)+"---\n")
    st.write("---Training data results---\n")
    st.write("Root Mean Squared Error: {:.2f}\n".format(train_rmse))
    st.write("R Square: {:.2f}\n".format(train_r2))
#     print("Mean Absolute Error: {:.2f}\n".format(train_mae))
    st.write("Mean Absolute Percentage Error: {:.2f}\n".format(train_mape))
    
    # predict the values using testing data
    test_pred = model.predict(X_test)

    # evaluate using testing data
    test_rmse = rmse(y_test, test_pred)
#     test_mae = mean_absolute_error(y_test, test_pred)
    test_mape = mape(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)
    
        # print the results of the testing data
    print("-----Testing data results-----\n")
    print("Root Mean Squared Error: {:.2f}\n".format(test_rmse))
    print("R Square: {:.2f}\n".format(test_r2))
#     print("Mean Absolute Error: {:.2f}\n".format(test_mae))
    print("Mean Absolute Percentage Error: {:.2f}\n".format(test_mape))
    
    if if_plot=="Yes":
        plot_model(mas, y_train,y_test,train_pred,test_pred,model_name)
    else:
        print("No plot")
        
    tree_explained=['EXTR','Gradient Boosting','Random Forest','Light GBM','Xgboost','Catboost']
    
    if model_name in tree_explained:

        feature_importance_plotly(model, X_train, model_name)
        
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(X_train)
        # fig_bar = shap.summary_plot(shap_values, X_train, plot_type='bar', show=False)

        # # Display the plot using Streamlit
        # st.pyplot(fig_bar)

#         for i in range(len(X_test)):
#             display(shap.force_plot(explainer.expected_value, shap_values[i], features=X_train.loc[i], 
#             feature_names=X_train.columns))
            
    return train_rmse, train_mape, train_r2, test_rmse, test_mape, test_r2

# https://stackoverflow.com/questions/48973140/how-to-interpret-mse-in-keras-regressor/49009442#49009442


def plot_model(mas, y_train, y_test, train_pred, test_pred, model_name):
    # Inverse transform predictions and target values
    train_pred = mas.inverse_transform(train_pred.reshape(-1, 1)).ravel()
    y_train = mas.inverse_transform(y_train.reshape(-1, 1)).ravel()
    test_pred = mas.inverse_transform(test_pred.reshape(-1, 1)).ravel()
    y_test = mas.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # Create a DataFrame for the data
    data = {
        'Time': list(range(len(y_train) + len(y_test))),
        'Values': list(y_train) + list(y_test),
        'Type': ['Actual Train Data'] * len(y_train) + ['Actual Test Data'] * len(y_test),
        'Predictions': list(train_pred) + list(test_pred),
        'Model': [str(model_name) + ' Train Predictions'] * len(y_train) + [str(model_name) + ' Test Predictions'] * len(y_test)
    }

    df = pd.DataFrame(data)

    # Plot using Plotly Express
    fig = px.line(df, x='Time', y='Values', color='Type', labels={'Values': 'Amount in Millions($)'},
                  title=str(model_name) + ' - Train and Test Data vs. Predictions')

    fig.add_scatter(x=df[df['Type'] == 'Actual Train Data']['Time'], y=df[df['Type'] == 'Actual Train Data']['Predictions'],
                    mode='lines', line=dict(color='red'), name=str(model_name) + ' Train Predictions')

    fig.add_scatter(x=df[df['Type'] == 'Actual Test Data']['Time'], y=df[df['Type'] == 'Actual Test Data']['Predictions'],
                    mode='lines', line=dict(color='orange'), name=str(model_name) + ' Test Predictions')

    # Show the plot
    fig.update_layout(xaxis_title='Date in Weeks', yaxis_title='Amount in Millions($)')
                    #   legend_title='Legend')
    
    fig.update_layout(height=800)

    st.plotly_chart(fig,use_container_width=True,height=800)
    # fig.show()
    

def feature_importance_plotly(model, X_train, model_name):
    features = X_train.columns
    importances = model.feature_importances_
    indices = importances.argsort()

    # customized number 
    num_features = 25

    # Create a DataFrame for easier manipulation
    data = pd.DataFrame({'Feature': [features[i] for i in indices[-num_features:]],
                         'Importance': importances[indices[-num_features:]]})

    fig = px.bar(data, y='Feature', x='Importance', orientation='h',
                 title=f'{model_name}s Feature Importance',
                 labels={'Importance': f'{model_name}s Feature Importance'},
                 height=600, width=800,
                 color='Importance',  # Set color to Importance values
                 color_continuous_scale='Reds')  # Set color scale to Reds

    fig.update_layout(yaxis_title='Feature', xaxis_title='Importance', xaxis=dict(tickformat='%'), yaxis={'categoryorder':'total ascending'})

    # Format x-axis ticks to display only two decimals
    fig.update_xaxes(tickformat=".2%")

    fig.update_layout(height=800)

    st.plotly_chart(fig,use_container_width=True,height=800)


timed_df=pd.read_excel("InvoiceData_concate_cleaned_grouped_flu_stock_weekly.xlsx")
# timed_df.drop('Date', axis = 1,inplace=True)

train=timed_df[timed_df['reference_date']<'2023-05-01']
test=timed_df[timed_df['reference_date']>='2023-05-01']

train.drop('reference_date', axis = 1,inplace=True)
test.drop('reference_date', axis = 1,inplace=True)


mas_features= MaxAbsScaler()

train_array_features = mas_features.fit_transform(train.drop('amount_p$_sum', axis = 1))
test_array_features = mas_features.transform(test.drop('amount_p$_sum', axis = 1))

train_features = pd.DataFrame(train_array_features,columns = train.drop('amount_p$_sum', axis = 1).columns)
test_features = pd.DataFrame(test_array_features,columns = test.drop('amount_p$_sum', axis = 1).columns)


X_train = train_features
X_test = test_features

mas_outcome= MaxAbsScaler()

y_train = mas_outcome.fit_transform(train['amount_p$_sum'].values.reshape(-1, 1))
y_test = mas_outcome.transform(test['amount_p$_sum'].values.reshape(-1, 1))

base_models = [
    ('Linear Regression',LinearRegression()),
    ('Stochastic Gradient Descent Regression',SGDRegressor()),
    ('Decision Tree Regression',DecisionTreeRegressor()),
    ('Extra Tree Regressor',ExtraTreeRegressor()),
    ('Support Vector Regression',SVR()),
    ('Nu Support Vector Regression',NuSVR()),
    ('Linear Support Vector Regression',LinearSVR()),
    ('K-Nearest Neighbors',KNeighborsRegressor()),
    ('Multi Layer Perceptron',MLPRegressor()),
    ('Extra Trees Regressor',ExtraTreesRegressor()),
    ('Bagging Regressor',BaggingRegressor()),
    ('Gradient Boosting',GradientBoostingRegressor()),
    ('Adaboost Regressor',AdaBoostRegressor()),
    ('Random Forest Regressor',RandomForestRegressor()),
    ('Light GBM Regressor',LGBMRegressor()),
    ('Xgboost',XGBRegressor()),
    ('Catboost',CatBoostRegressor()),
]

train_rmse_l, train_mape_l, train_r2_l, test_rmse_l, test_mape_l, test_r2_l, model_name_l =\
[], [], [], [], [], [], []

for name, model_n in base_models:
#     start_time = time.time()
    print("---------------------------------------------------- Model: {}".format(name))
    train_rmse, train_mape, train_r2, test_rmse, test_mape, test_r2=\
    evaluate_model(mas_outcome, model_n, X_train, y_train, X_test, y_test,name,if_plot="Yes")
    train_rmse_l.append(train_rmse)
    train_mape_l.append(train_mape)
    train_r2_l.append(train_r2)
    test_rmse_l.append(test_rmse)
    test_mape_l.append(test_mape)
    test_r2_l.append(test_r2)
    model_name_l.append(name)
#     end_time = time.time()
#     print("Computation Time: {}".format(end_time - start_time))
    print("----------------------------------\n")
    
