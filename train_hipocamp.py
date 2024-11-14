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

def data_analysis(df_train, df_test):
    pd.set_option('display.max_columns',None)

    print(df_train.dtypes)

    #print(df_train.dtypes)

    #print(df_train.describe())

    #print(df_train.isnull().any)

    #print(df_train.duplicated().any())

def random_forest_classifier(X, y, df_test):
    # BLOCO DE CÓDIGO PARA FAZER O RANDOMFORESTCLASSIFFIER
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2022)

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
    
    return test_predictions

def xgboost(X,y, df_test):
    # BLOCO DE CÓDIGO PARA FAZER O XGBoost
    #Para poder usar o XGBoost nestes dados é preciso fazer primeiro label encoding da variável y
    print("Encoding labels of y variable...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("Splitting values into train and test")
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size = 0.327, random_state=2022)

    xgb_model = XGBClassifier(
        n_estimators = 80, #best n_estimators = 80
        min_child_weight = 8, #best min_child_weight = 8
        max_depth = 10, #best max_depth = 10
        learning_rate = 0.02, #best learning_rate = 0.02
        subsample=0.8, #best subsample = 0.8
        colsample_bytree=0.8, #best colsample_bytree = 0.8
        colsample_bylevel = 1.0, #best colsample_bylevel = 1.0 -> default
        colsample_bynode = 1.0, #best colsample_bynode = 1.0 -> default
        gamma = 0, #best gamma = 0 -> default
        alpha = 0, #best alpha = 0
        early_stopping_rounds = 30, #no difference made with different values
        objective='multi:softprob', #no difference made with different values
        eval_metric='mlogloss' #best eval_metric = 'mlogloss' with eval_set defined
    )

    print('Fitting model...')
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)],verbose=1)

    print('Predicting results...')
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)

    # Print evaluation metrics
    print("\nModel Performance on Test Set:")
    print("-----------------------------")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    test_predictions = xgb_model.predict(df_test)

    """
    #Definir os parametros da GridSearchCV
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
    print('Decoding predictions to original classes...')
    test_predictions_text = label_encoder.inverse_transform(test_predictions)

    #XGBOOST FEATURE IMPORTANCE
    print('Computing feature importances...')
    start_time = time.time()

    mdi_importances = pd.Series(xgb_model.feature_importances_, index=X_test.columns)

    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    count = 0

    for feature_name, mdi_importance in mdi_importances.items():
        if mdi_importance > 0.000:
            #print(f"Feature name {feature_name}: {mdi_importance:.4f}")
            count += 1
    print(f"Total features with importance bigger than 0.000: {count}")
    
    return test_predictions_text

def main():
    
    #Import training dataset
    print('Importing train dataset...')
    df_train = pd.read_csv('train_radiomics_hipocamp.csv')

    #Import test dataset
    print('Importing test dataset...')
    df_test = pd.read_csv('test_radiomics_hipocamp.csv')
    
    print('Dropping all features that are type object except Transition from train dataset...')
    df_train.drop(df_train.select_dtypes(include='object').drop(columns=['Transition'], errors='ignore').columns, axis=1, inplace=True)

    print('Dropping all features that are type object from test dataset...')
    df_test.drop(df_test.select_dtypes(include=object), axis = 1, inplace = True) 
    
    print('Creating X variable...')
    X = df_train.drop('Transition', axis = 1)
    print('Creating y variable...')
    y = df_train['Transition']

    #print('Applying Random Forest Classifier model to the train dataset...')
    #test_predictions = random_forest_classifier(X, y, df_test)
    
    print('Applying XGBoost model to the train dataset...')
    test_predictions = xgboost(X, y, df_test)
    
    output = pd.DataFrame(columns=['RowId', 'Result'])

    # Save predictions to a CSV file
    # Using list comprehension to construct the DataFrame more efficiently
    print('Saving predictions to the test_predictions.csv...')
    output = pd.DataFrame({'RowId': range(1, len(test_predictions) +1), 'Result': test_predictions}) #ATENÇÃO aqui podemos mudar para test_predictions_text por causa do decoding da label
  
    output.to_csv('test_predictions.csv', index=False)
    
if __name__ == '__main__':
    main()
