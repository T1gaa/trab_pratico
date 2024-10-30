import sklearn as sk1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

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
df_train.drop(df_train.select_dtypes(include='object').drop(columns=['Transition'], errors='ignore').columns, axis=1, inplace=True)
