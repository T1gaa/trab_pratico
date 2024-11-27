import sklearn as sk1
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, RandomizedSearchCV
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, precision_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
import shap
import time
from scipy.stats import uniform, randint

def data_analysis(df_train, df_test):
    pd.set_option('display.max_columns',None)

    print(df_train.dtypes)

    print(df_train.dtypes)

    print(df_train.describe())

    print(df_train.isnull().any)

    print(df_train.duplicated().any())

def random_forest_classifier(X, y, df_test):
    # BLOCO DE CÓDIGO PARA FAZER O RANDOMFORESTCLASSIFFIER
    print('Splitting values...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.327, random_state=2022)
    
    print('Training using RandomForestClassifier...')
    rf_model = RandomForestClassifier(
        bootstrap= True, #best bootstrap= True
        max_depth = 10, #best max_depth=
        n_estimators= 100, #best n_estimators = 
        min_samples_split= 2, #best min_samples_split = 
        min_samples_leaf= 2, #best min_samples_leaf = 
        max_features= 'sqrt', #best max_features = 
        criterion='gini', #best criterion= 
        verbose = 1)
    
    print('Fitting the values...')
    rf_model.fit(X_train, y_train)

    test_predictions = rf_model.predict(df_test)
    
    rf_score = rf_model.score(X_test, test_predictions)
    print("Accuracy: %.2f%%" % (rf_score*100))
    
    print('Precision score:')
    precision = precision_score(y_true = y_test, y_pred = test_predictions, average='macro')
    print(precision)

    # Print evaluation metrics
    print("\nModel Performance on Test Set:")
    print("-----------------------------")
    print("\nClassification Report:")
    print(classification_report(y_test, test_predictions))
    
    print('Precision scores: ')
    precision_scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
    print(f"Average precision: {precision_scores.mean():.3f} (+/- {precision_scores.std() * 2:.3f})")

    #RANDOM FOREST FEATURE IMPORTANCE
    start_time = time.time()

    mdi_importances = pd.Series(rf_model.feature_importances_, index=X_test.columns)

    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    count = 0

    for feature_name, mdi_importance in mdi_importances.items():
        if mdi_importance > 0.000:
            #print(f"Feature name {feature_name}: {mdi_importance:.4f}")
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

    xgb_classifier = XGBClassifier(random_state = 2022)
    
    # Define the parameter distributions for RandomizedSearchCV
    random_param_dist = {
    # Number of boosting rounds (trees)
    'n_estimators': randint(50, 120),
    
    # Learning rate
    'learning_rate': uniform(0.01, 0.1),
    
    # Maximum depth of trees
    'max_depth': randint(3, 10),
    
    # Minimum child weight
    'min_child_weight': randint(1, 10),
  
    # Booster type
    'booster': ['gbtree', 'gblinear', 'dart']
    }

    # Perform RandomizedSearchCV
    print('Applying RandomSearch')
    random_search = RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=random_param_dist,
        n_iter=5,  # Number of parameter settings sampled
        cv=5,        # 5-fold cross-validation
        scoring='accuracy',
        random_state=42,
        n_jobs=-1    # Use all available cores
    )

    # Fit RandomizedSearchCV
    random_search.fit(X_test, y_test)

    # Print best parameters and score from RandomSearch
    print("Best parameters from RandomSearch:")
    print(random_search.best_params_)
    print(f"Best cross-validation score: {random_search.best_score_:.4f}")

    # Define GridSearchCV parameters based on RandomSearch results
    # Narrow down the search space around the best parameters
    grid_param_grid = {
        'n_estimators': [
            max(50, random_search.best_params_['n_estimators'] - 50),
            random_search.best_params_['n_estimators'],
            min(500, random_search.best_params_['n_estimators'] + 50)
        ],
        'learning_rate': np.linspace(
            max(0.01, random_search.best_params_['learning_rate'] - 0.1),
            min(0.3, random_search.best_params_['learning_rate'] + 0.1),
            5
        ),
        'max_depth': [
            max(3, random_search.best_params_['max_depth'] - 1),
            random_search.best_params_['max_depth'],
            min(10, random_search.best_params_['max_depth'] + 1)
        ],
        'min_child_weight': [
            max(1, random_search.best_params_['min_child_weight'] - 2),
            random_search.best_params_['min_child_weight'],
            min(10, random_search.best_params_['min_child_weight'] + 2)
        ],
        'booster': [random_search.best_params_['booster']]
    }

    # Perform GridSearchCV
    print('Applying GridSearch')
    grid_search = GridSearchCV(
        estimator=xgb_classifier,
        param_grid=grid_param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    # Fit GridSearchCV
    grid_search.fit(X_test, y_test)

    # Print best parameters and score from GridSearch
    print("\nBest parameters from GridSearch:")
    print(grid_search.best_params_)
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    print('Predicting results...')
    y_pred = grid_search.predict(df_test)
    y_pred_proba = grid_search.predict_proba(df_test)
    
    best_model = grid_search.best_estimator_

    # Print evaluation metrics
    print("\nModel Performance on Test Set:")
    print("-----------------------------")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division= 1))

    test_predictions = best_model.predict(df_test)

    #Avaliação do model nos dados de teste
    test_predictions = best_model.predict(df_test)
    accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    #XGBOOST FEATURE IMPORTANCE
    print('Computing feature importances...')
    start_time = time.time()

    mdi_importances = pd.Series(best_model.feature_importances_, index=X_test.columns)

    elapsed_time = time.time() - start_time

    print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")

    count = 0

    features_to_drop = []
    
    for feature_name, mdi_importance in mdi_importances.items():
        if mdi_importance < 0.00:
            #print(f"Feature name {feature_name}: {mdi_importance:.4f}")
            count += 1         
            features_to_drop.append(feature_name)
    print(f"Total features with importance lower than 0.000: {count}")
    
    print('Build second dataset using the most important features...')
    X2 = X.drop(columns = features_to_drop)
    dftest2 = df_test.drop(columns = features_to_drop)
    
    print("Splitting values into train and test ind the second dataset")
    X_train, X_test, y_train, y_test = train_test_split(X2, y_encoded, test_size = 0.327, random_state=2022)
    
    print('Fitting second model...')
    best_model.fit(X_train, y_train,verbose=1)
    
    print('Predicting results...')
    y_pred = best_model.predict(dftest2)
    y_pred_proba = best_model.predict_proba(dftest2)

    # Print evaluation metrics
    print("\nModel Performance on Test Set:")
    print("-----------------------------")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division= 1))

    test_predictions = best_model.predict(df_test)

    #Avaliação do model nos dados de teste
    test_predictions2 = best_model.predict(dftest2)
    accuracy = accuracy_score(y_test, test_predictions)
    print(f"Test Accuracy: {accuracy:.2f}")
    
    
    #Descodificar as predictions para os labels originais
    print('Decoding predictions to original classes...')
    test_predictions_text = label_encoder.inverse_transform(test_predictions)
    
    #Descodificar as predictions para os labels originais
    print('Decoding predictions to original classes...')
    test_predictions_text2 = label_encoder.inverse_transform(test_predictions2)
    
    return test_predictions_text, test_predictions_text2

def svm(X, y, df_test):
    #Apply the Support-Vector Machine Model
    return

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

    #data_analysis(df_train, df_test)
    
    #print('Applying Random Forest Classifier model to the train dataset...')
    #test_predictions = random_forest_classifier(X, y, df_test)
    
    print('Applying XGBoost model to the train dataset...')
    test_predictions, test_predictions2 = xgboost(X, y, df_test)
    
    output = pd.DataFrame(columns=['RowId', 'Result'])

    # Save predictions to a CSV file
    # Using list comprehension to construct the DataFrame more efficiently
    print('Saving predictions to the test_predictions.csv...')
    output = pd.DataFrame({'RowId': range(1, len(test_predictions) +1), 'Result': test_predictions}) #ATENÇÃO aqui podemos mudar para test_predictions_text por causa do decoding da label
  
    output.to_csv('test_predictions.csv', index=False)
    
    output = pd.DataFrame({'RowId': range(1, len(test_predictions2) +1), 'Result': test_predictions2})
    output.to_csv('test_predictions2.csv', index = False)
    
    
if __name__ == '__main__':
    main()
