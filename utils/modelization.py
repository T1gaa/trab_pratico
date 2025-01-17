import pickle
import pandas as pd

# filename without path and extension
def saveModel(model, filename):
    with open( "best_models/" + filename + ".pkl", 'wb') as file:
        pickle.dump(model, file)

# filename without path and extension
def loadModel( filename):
    with open("best_models/" + filename + ".pkl", 'rb') as file:
        loaded_model = pickle.load(file)
    return loaded_model

def submitModel(preds, filename):
    output = pd.DataFrame({'RowId': range(1, len(preds) +1), 'Result': preds}) #ATENÇÃO aqui podemos mudar para test_predictions_text por causa do decoding da label
    output.to_csv("submissions/" + filename + ".csv", index=False) 


