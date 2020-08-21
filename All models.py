# -*- coding: utf-8 -*-
"""
Created on Sat Jan 25 18:12:23 2020

@author: Shivam
"""

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt


df = pd.read_csv("AQI.csv")

"""Check for the missing data""" 
df.isnull().sum()


""" imputing the missing values with mean"""
df['PM 2.5'].fillna(df['PM 2.5'].mean(), inplace=True)
"""Check again for the missing data""" 
df.isnull().sum()



#plotting the dataframe
sns.pairplot(df)

#checking the corellation
corelation = df.corr()

"""plotting the correlation"""
sns.heatmap(corelation,annot=True,cmap='winter_r')


X=df.iloc[:,:8] ## independent features

y=df.iloc[:,8] ## dependent features : "pm25"

#splitting the train data and test data into 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)




#checking the coefficient value for each independent variable
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df


#cross validation  for linear regression
score=cross_val_score(lm,X,y,cv=5)
score
score.mean()


#checking the coefficient value for each independent variable
coeff_df = pd.DataFrame(lm.coef_,X.columns,columns=['Coefficient'])
coeff_df




#fitting the linear model
lm=LinearRegression()
lm.fit(X_train,y_train)


#checking the train score, R^2 value
lm.score(X_train,y_train)

#checking the test scores, R^2 value
lm.score(X_test,y_test)

#predicting the test data
y_pred_lm = lm.predict(X_test)

""" Mean Absolute error & Mean square error for linear model """
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_lm))

print('MSE:', metrics.mean_squared_error(y_test, y_pred_lm))

mae_l = metrics.mean_absolute_error(y_test, y_pred_lm)
mse_l = metrics.mean_squared_error(y_test, y_pred_lm)

#from sklearn.metrics import confusion_matrix
#conf_matrix_lm = confusion_matrix(y_test,y_pred_lm)
#print(classification_report(y_test_lm,y_pred_lm))

#plotted y-pred and y-actual
plt.scatter(y_test,y_pred_lm)
plt.title('Actual y - Predicted y ')
plt.xlabel('Actual Y')
plt.ylabel('Pred Y')


#sns.heatmap(conf_matrix_lm, annot=True,xticklabels=x_axis_labels, yticklabels=y_axis_labels)


"""-----------------------------------------------------------------"""







from sklearn.neighbors import KNeighborsRegressor
accuracy_rate = []
# Will take some time
for i in range(1,20):    
    knn = KNeighborsRegressor(n_neighbors=i)
    score=cross_val_score(knn,X,y,cv=10,scoring="neg_mean_squared_error")
    accuracy_rate.append(score.mean())
    

#plotting the elow graph for all k values 
plt.figure(figsize=(10,6))
plt.plot(range(1,20),accuracy_rate,color='black', marker='.',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


knn = KNeighborsRegressor(n_neighbors=15)
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)

""" Mean Absolute error & Mean square error for linear model """
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_knn))

print('MSE:', metrics.mean_squared_error(y_test, y_pred_knn))


mae_k = metrics.mean_absolute_error(y_test, y_pred_knn)
mse_k = metrics.mean_squared_error(y_test, y_pred_knn)

#plotted y-pred and y-actual
plt.scatter(y_test,y_pred_knn)
plt.title('Actual y - Predicted y ')
plt.xlabel('Actual Y')
plt.ylabel('Pred Y')



"""-------------------------------------------------------------------------"""



"""Decision Tree"""

from sklearn.tree import DecisionTreeRegressor
dtree=DecisionTreeRegressor(criterion="mse",random_state=0)
dtree.fit(X_train,y_train)

y_pred_dtree=dtree.predict(X_test)

""" Mean Absolute error & Mean square error for linear model """
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_dtree))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_dtree))


mae_d = metrics.mean_absolute_error(y_test, y_pred_dtree)
mse_d = metrics.mean_squared_error(y_test, y_pred_dtree)

#plotted y-pred and y-actual
plt.scatter(y_test,y_pred_dtree)
plt.title('Actual y - Predicted y ')
plt.xlabel('Actual Y')
plt.ylabel('Pred Y')



"""---------------------------------------------------------------------"""


"""Random Forest """

from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor(n_estimators=100,random_state=0)
rf.fit(X_train,y_train)

y_pred_rf = rf.predict(X_test)
 

""" Mean Absolute error & Mean square error for linear model """
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_rf))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_rf))

mae_r = metrics.mean_absolute_error(y_test, y_pred_rf)
mse_r = metrics.mean_squared_error(y_test, y_pred_rf)

#plotted y-pred and y-actual
plt.scatter(y_test,y_pred_rf)
plt.title('Actual y - Predicted y ')
plt.xlabel('Actual Y')
plt.ylabel('Pred Y')


"""-----------------------------------------------------------"""


"""Xgboost"""


import xgboost as xgb   #conda install py-xgboost


xgboost =xgb.XGBRegressor(n_estimators=1000)
xgboost.fit(X_train,y_train)

y_pred_xgb=xgboost.predict(X_test)

""" Mean Absolute error & Mean square error for linear model """
print('MAE:', metrics.mean_absolute_error(y_test, y_pred_xgb))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_xgb))


mae_x = metrics.mean_absolute_error(y_test, y_pred_xgb)
mse_x = metrics.mean_squared_error(y_test, y_pred_xgb)

#plotted y-pred and y-actual
plt.scatter(y_test,y_pred_xgb)
plt.title('Actual y - Predicted y ')
plt.xlabel('Actual Y')
plt.ylabel('Pred Y')




"""----------------------------------------------------------------"""

"""ANN"""

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

NN_model = Sequential()

# The Input Layer :
NN_model.add(Dense(8, kernel_initializer='normal',input_dim = X_train.shape[1], activation='relu'))

# The Hidden Layers :
NN_model.add(Dense(250, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dropout(rate=0.2))
NN_model.add(Dense(50, kernel_initializer='normal',activation='relu'))
#NN_model.add(Dropout(rate=0.5))
NN_model.add(Dense(26, kernel_initializer='normal',activation='relu'))

# The Output Layer :
NN_model.add(Dense(1, kernel_initializer='normal',activation='linear'))

# Compile the network :
NN_model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
NN_model.summary()


NN_model.fit(X_train, y_train, nb_epoch = 100)

y_pred_ANN=NN_model.predict(X_test)

print('MAE:', metrics.mean_absolute_error(y_test, y_pred_ANN))
print('MSE:', metrics.mean_squared_error(y_test, y_pred_ANN))


mae_nn = metrics.mean_absolute_error(y_test, y_pred_ANN)
mse_nn = metrics.mean_squared_error(y_test, y_pred_ANN)


#plotted y-pred and y-actual
plt.scatter(y_test,y_pred_ANN)
plt.title('Actual y - Predicted y ')
plt.xlabel('Actual Y')
plt.ylabel('Pred Y')





"""---------------------------------------------------------"""


"""Model Comparision"""




model = {'Linear Regression' : [mae_l,mse_l],
            'KNN' : [mae_k,mse_k],
            'Decision Tree' : [mae_d,mse_d],
            'Random Forest' : [mae_r,mse_r],
            'XgBoost' : [mae_x,mse_x],
            'ANN' : [mae_nn,mse_nn]
        }


compare = pd.DataFrame.from_dict(model,orient="index",columns=['MAE','MSE'])
