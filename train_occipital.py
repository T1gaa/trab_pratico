import sklearn as sk1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import time

"""
Dados de controlo que são necessários ajustar para o checkpoint 1
"""
#Import dataset
df = pd.read_csv('train_radiomics_occipital_CONTROL.csv')
pd.set_option('display.max_columns',None)
#Check data 

#print(df.head())
#print(df.dtypes)
#print(df.describe())

#print(df.isna().any())

#Dar drop a todas as colunas que são objetos excepto a coluna 'Transition'
df.drop(df.select_dtypes(include='object').drop(columns=['Transition'], errors='ignore').columns, axis=1, inplace=True)

X = df_train.drop('Transition', axis = 1)
y = df_train['Transition']

#BLOCO DE CÓDIGO PARA FAZER O RANDOMFORESTCLASSIFFIER
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2022)

rf_model = RandomForestClassifier(bootstrap= False, max_depth = 2, verbose = 1)
rf_model.fit(X_train, y_train)

rf_score = rf_model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (rf_score*100))

test_predictions = rf_model.predict(df_test)


"""
#BLOCO DE CÓDIGO PARA FAZER O XGBoost
#Para poder usar o XGBoost nestes dados é preciso fazer primeiro label encoding da variável y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.3, random_state=2022)

xgb_model = XGBClassifier(max_depth = 1, objective = 'multi:softmax')
xgb_model.fit(X_train, y_train)

xgb_score = xgb_model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (xgb_score * 100))

test_predictions = xgb_model.predict(df_test)

test_predictions_text = label_encoder.inverse_transform(test_predictions)
"""

