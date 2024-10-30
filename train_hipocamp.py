import sklearn as sk1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import sklearn as sk1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#Import training dataset
df_train = pd.read_csv('train_radiomics_hipocamp.csv')

#Import test dataset 
df_test = pd.read_csv('test_radiomics_hipocamp.csv')

pd.set_option('display.max_columns',None)

print(df_train.dtypes)

df_train.drop(df_train.select_dtypes(include='object').drop(columns=['Transition'], errors='ignore').columns, axis=1, inplace=True)

df_test.drop(df_test.select_dtypes(include=object), axis = 1, inplace = True) 

#print(df_train.dtypes)

#print(df_train.describe())

#print(df_train.isnull().any)

#print(df_train.duplicated().any())

X = df_train.drop('Transition', axis = 1)
y = df_train['Transition'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2022)
"""
rf_model = RandomForestClassifier(bootstrap= False, max_depth = 2, verbose = 1)
rf_model.fit(X_train, y_train)

rf_score = rf_model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (rf_score*100))

test_predictions = rf_model.predict(df_test)
"""
xgb_model = XGBClassifier(max_depth = 1, objecive = 'reg:squarederror')
xgb_model.fit(X_train, y_train)

xgb_score = xgb_model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (xgb_score * 100))

test_predictions = xgb_model.predict(df_test)

#Daqui para baixo não mexer

output = pd.DataFrame(columns=['RowId', 'Result'])

# Save predictions to a CSV file
# Using list comprehension to construct the DataFrame more efficiently
output = pd.DataFrame({'RowId': range(1, len(test_predictions) +1), 'Result': test_predictions})
  
output.to_csv('test_predictions.csv', index=False)
#Import training dataset
df_train = pd.read_csv('train_radiomics_hipocamp.csv')

#Import test dataset 
df_test = pd.read_csv('test_radiomics_hipocamp.csv')

pd.set_option('display.max_columns',None)

print(df_train.dtypes)

df_train.drop(df_train.select_dtypes(include='object').drop(columns=['Transition'], errors='ignore').columns, axis=1, inplace=True)

df_test.drop(df_test.select_dtypes(include=object), axis = 1, inplace = True) 

#print(df_train.dtypes)

#print(df_train.describe())

#print(df_train.isnull().any)

#print(df_train.duplicated().any())

X = df_train.drop('Transition', axis = 1)
y = df_train['Transition'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2022)
"""
rf_model = RandomForestClassifier(bootstrap= False, max_depth = 2, verbose = 1)
rf_model.fit(X_train, y_train)

rf_score = rf_model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (rf_score*100))

test_predictions = rf_model.predict(df_test)
"""
xgb_model = XGBClassifier(max_depth = 1, objecive = 'reg:squarederror')
xgb_model.fit(X_train, y_train)

xgb_score = xgb_model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (xgb_score * 100))

test_predictions = xgb_model.predict(df_test)

#Daqui para baixo não mexer

output = pd.DataFrame(columns=['RowId', 'Result'])

# Save predictions to a CSV file
# Using list comprehension to construct the DataFrame more efficiently
output = pd.DataFrame({'RowId': range(1, len(test_predictions) +1), 'Result': test_predictions})
  
output.to_csv('test_predictions.csv', index=False)
