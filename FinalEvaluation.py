from tkinter import *
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") # ignoring annoying warnings


#loading data
features = pd.read_csv('features.csv')
train = pd.read_csv('train.csv')
stores = pd.read_csv('stores.csv')
test = pd.read_csv('test.csv')
#merging fetures and stores data 
feature_store = features.merge(stores, how='inner', on='Store')
#checking the types of dataframe(feature_store)
pd.DataFrame(feature_store.dtypes, columns=['Type'])
#train.head(5)
pd.DataFrame({'Type_Train': train.dtypes, 'Type_Test': test.dtypes})
#converting datetime
feature_store.Date = pd.to_datetime(feature_store.Date)
train.Date = pd.to_datetime(train.Date)
test.Date = pd.to_datetime(test.Date)
feature_store['Week'] = feature_store.Date.dt.week 
feature_store['Year'] = feature_store.Date.dt.year
#combining the data we will be needed 
to_train = train.merge(feature_store,on=['Store','Date','IsHoliday']).sort_values(by=['Store','Dept', 'Date']).reset_index(drop=True)
to_test = test.merge(feature_store,on=['Store','Date','IsHoliday']).sort_values(by=['Store','Dept','Date']).reset_index(drop=True)
del features, stores, train,test


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


to_train = to_train.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
to_test = to_test.drop(columns=['Fuel_Price','MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5'])
to_train.Type = to_train.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))
to_test.Type = to_test.Type.apply(lambda x: 3 if x == 'A' else(2 if x == 'B' else 1))
X_train = to_train[['Store','Dept','IsHoliday','Size','Week','Type','Year']]
Y_train = to_train['Weekly_Sales']

x_train, X_test, y_train, y_test = train_test_split(X_train,Y_train, test_size=0.3);



RFRegressor = RandomForestRegressor(n_estimators = 50)
RFRegressor.fit(x_train, y_train)
predictions = RFRegressor.predict(X_test)
error = y_test - predictions

def weekly_sales_per_year():
    weekly_sales_2010 = to_train[to_train.Year==2010]['weekly_sales'].groupby(to_train['week']).mean()
    weekly_sales_2011 = to_train[to_train.Year==2011]['weekly_sales'].groupby(to_train['week']).mean()
    weekly_sales_2012 = to_train[to_train.Year==2012]['weekly_sales'].groupby(to_train['week']).mean()
    plt.figure()
    sns.lineplot(weekly_sales_2010.index, weekly_sales_2010.values)
    sns.lineplot(weekly_sales_2011.index, weekly_sales_2011.values)
    sns.lineplot(weekly_sales_2012.index, weekly_sales_2012.values)
    plt.xticks(np.arange(1, 53, step=1))
    plt.legend(['2010', '2011', '2012'])
    plt.title('average weekly sales - per Year')
    plt.ylabel('sales')
    plt.xlabel('week')
    plt.show()

def mean_median_graph():
    weekly_sales_mean = to_train['Weekly_Sales'].groupby(to_train['Date']).mean()
    weekly_sales_median = to_train['Weekly_Sales'].groupby(to_train['Date']).median()
    plt.figure()
    sns.lineplot(weekly_sales_mean.index, weekly_sales_mean.values)
    sns.lineplot(weekly_sales_median.index, weekly_sales_median.values)
    plt.legend(['Mean', 'Median'])
    plt.title('Weekly Sales - Mean and Median')
    plt.ylabel('sales')
    plt.xlabel('date')
    plt.show()

def weekly_sales_per_store():
    weekly_sales = to_train['Weekly_Sales'].groupby(to_train['Store']).mean()
    plt.figure()
    sns.barplot(weekly_sales.index, weekly_sales.values)
    plt.title('Average Sales - per Store')
    plt.ylabel('sales')
    plt.xlabel('store')
    plt.show()

def weekly_sales_per_department():
    weekly_sales = to_train['Weekly_Sales'].groupby(to_train['Dept']).mean()
    plt.figure()
    sns.barplot(weekly_sales.index, weekly_sales.values)
    plt.title('Average Sales - per Dept')
    plt.ylabel('sales')
    plt.xlabel('dept')
    plt.show()



import pickle
filename = 'store.pkl'
pickle.dump(RFRegressor, open(filename, 'wb'))
