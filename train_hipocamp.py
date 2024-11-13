import sklearn as sk1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import shap
import time

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

"""# BLOCO DE CÓDIGO PARA FAZER O RANDOMFORESTCLASSIFFIER
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2022)


pd.Series
rf_model = RandomForestClassifier(bootstrap= False, max_depth = 2, verbose = 1)
rf_model.fit(X_train, y_train)

rf_score = rf_model.score(X_test, y_test)
print("Accuracy: %.2f%%" % (rf_score*100))

test_predictions = rf_model.predict(df_test)

#RANDOM FOREST FEATURE IMPORTANCE
start_time = time.time()

mdi_importances = pd.Series(rf_model.feature_importances_, index=X_test.columns)

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

count = 0

for feature_name, mdi_importance in mdi_importances.items():
    if mdi_importance > 0.000:
        print(f"Feature name {feature_name}: {mdi_importance:.4f}")
        count += 1
print(f"Total features with importance bigger than 0.000: {count}")
"""


# BLOCO DE CÓDIGO PARA FAZER O XGBoost
#Para poder usar o XGBoost nestes dados é preciso fazer primeiro label encoding da variável y
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.327, random_state=2022)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)   # Transform test set using training parameters
X_val_scaled = scaler.transform(df_test) 


selector = SelectKBest(score_func=f_classif, k=900)
X_train_selected = selector.fit_transform(X_train_scaled, y_train)
X_test_selected = selector.transform(X_test_scaled)
X_val_selected = selector.transform(X_val_scaled)
    
# Get selected feature indices
selected_features = selector.get_support()

xgb_model = XGBClassifier(
    n_estimators = 200,
    max_depth = 6,
    learning_rate = 0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softprob',
    use_label_encoder=False,
    eval_metric='mlogloss'
)

xgb_model.fit(X_train_selected, y_train)

y_pred = xgb_model.predict(X_test_selected)
y_pred_proba = xgb_model.predict_proba(X_test_selected)

 # Print evaluation metrics
print("\nModel Performance on Test Set:")
print("-----------------------------")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

test_predictions = xgb_model.predict(X_val_selected)

"""#Definir os parametros da GridSearchCV
param_grid={
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [100, 200, 300],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

#Aplicar GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    verbose=1,
    n_jobs=-1
)

#Fitting GridSearchCV aos dados de treino
start_time = time.time()
grid_search.fit(X_train, y_train)
elapsed_time = time.time()
#Os melhores parâmetros e ao modelo
best_model = grid_search.best_estimator_
print("Best parameters found: ", grid_search.best_params_)

#Avaliação do model nos dados de teste
test_predictions = best_model.predict(df_test)
accuracy = accuracy_score(y_test, test_predictions)
print(f"Test Accuracy: {accuracy:.2f}")
"""

#Descodificar as predictions para os labels originais
test_predictions_text = label_encoder.inverse_transform(test_predictions)

"""#XGBOOST FEATURE IMPORTANCE
start_time = time.time()

mdi_importances = pd.Series(best_model.feature_importances_, index=X_test.columns)

elapsed_time = time.time() - start_time

print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

count = 0

for feature_name, mdi_importance in mdi_importances.items():
    if mdi_importance > 0.000:
        #print(f"Feature name {feature_name}: {mdi_importance:.4f}")
        count += 1
print(f"Total features with importance bigger than 0.000: {count}")
"""

#Daqui para baixo não mexer

output = pd.DataFrame(columns=['RowId', 'Result'])

# Save predictions to a CSV file
# Using list comprehension to construct the DataFrame more efficiently
output = pd.DataFrame({'RowId': range(1, len(test_predictions_text) +1), 'Result': test_predictions_text}) #ATENÇÃO aqui podemos mudar para test_predictions_text por causa do decoding da label
  
output.to_csv('test_predictions.csv', index=False)