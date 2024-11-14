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
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from torch.utils.data import Dataset, random_split, DataLoader
from torch.nn import Module, Linear,  ReLU, Softmax, CrossEntropyLoss
from torch.nn.init import xavier_uniform_, kaiming_uniform_
from torch.optim import SGD, Adam
from torchsummary import summary
from torch import Tensor
from livelossplot import PlotLosses
import shap
import time
import torch

#Import training dataset
df_train = pd.read_csv('train_radiomics_hipocamp.csv')

#Import test dataset 
df_test = pd.read_csv('test_radiomics_hipocamp.csv')

df_train.drop(df_train.select_dtypes(include='object').drop(columns=['Transition'], errors='ignore').columns, axis=1, inplace=True)

df_test.drop(df_test.select_dtypes(include=object), axis = 1, inplace = True) 

X = df_train.drop('Transition', axis = 1)
y = df_train['Transition']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

t_X = pd.DataFrame(X)
filename = "features_X.csv"
t_X.to_csv(filename, index=False, encoding='utf-8')

t_y = pd.DataFrame(y_encoded)
filename = "label_y.csv"
t_y.to_csv(filename, index=False, encoding='utf-8')

class CSVDataset(Dataset):
    def __init__(self,path):
        #define inputs and outputs
        df_X = pd.read_csv("features_X.csv", header=0)
        df_y = pd.read_csv("label_y.csv",header=0)
        #convert to numpy array
        self.X = df_X.values
        self.y = df_y.values[:,0]-1
        #ensure X and Y are float32 and Long tensor
        self.X= self.X.astype('float32')
        self.y= torch.tensor(self.y, dtype=torch.long, device="cpu") #cude se utilizar gpu, se é cpu
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        #get an instance
        return [self.X[idx], self.y[idx]]
    
    def get_splits(self, n_test):
        #define test and train size
        test_size = round(n_test * len(self.X))
        train_size  = len(self.X) - test_size
        #return holdout split
        return random_split(self, [train_size, test_size])

def prepare_data(df_train, n_test):
    #create on instance of the dataset
    dataset = CSVDataset(df_train)
    #Calculate the split
    train, test = dataset.get_splits(n_test)
    #prepare the dataloaders
    train_dl = DataLoader(train, batch_size=len(train), shuffle=True)
    test_dl = DataLoader(test, batch_size=len(train), shuffle=True)
    return train_dl, test_dl

train_dl, test_dl = prepare_data(df_train, 0.33)

def visualize_dataset(train_dl, test_dl):
    print(f"Train size: {len(train_dl.dataset)}")
    print(f"Test size: {len(test_dl.dataset)}")
    x, y = next(iter(train_dl)) #iterate through the Loaders to fetch a batch of cases
    print(f"Shape tensor train data batch - input: {x.shape}, output: {y.shape}")
    x, y = next(iter(test_dl))
    print(f"Shape tensor test data batch - input: {x.shape}, output: {y.shape}")

visualize_dataset(train_dl, test_dl)

def visualize_holdout_balance(train_dl, test_dl):
    _, y_train = next(iter(train_dl))
    _, y_test = next(iter(test_dl))
    
    sns.set_style('whitegrid')
    train_df = len(y_train)
    test_df = len(y_test)
    
    Class_1_train = np.count_nonzero(y_train == 0)
    Class_2_train = np.count_nonzero(y_train == 1)
    Class_3_train = np.count_nonzero(y_train == 2)
    print("Train data: ", train_df)
    print("Class 1: ", Class_1_train)
    print("Class 2: ", Class_2_train)
    print("Class 3: ", Class_3_train)
    print("Values mean (train): ", np.mean(y_train.numpy()))
    
    Class_1_test = np.count_nonzero(y_test == 0)
    Class_2_test = np.count_nonzero(y_test == 1)
    Class_3_test = np.count_nonzero(y_test == 2)
    print("Test data: ", test_df)
    print("Class 1: ", Class_1_test)
    print("Class 2: ", Class_2_test)
    print("Class 3: ", Class_3_test)
    print("Values mean (test): ", np.mean(y_test.numpy()))
    
    """graph = sns.pairplot({'Class 1 train': Class_1_train,
                         'Class 2 train': Class_2_train,
                         'Class 3 train': Class_3_train,
                         'Class_1_test' : Class_1_test,
                         'Class_2_test' : Class_2_test,
                         'Class_3_test' : Class_3_test})
    
    graph.set_title('Data balance by class')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.savefig('data_balance_MLP.png')
    plt.show()
    
    graph = sns.barplot(x=['Train data average', 'Test data average'],
                        y=[np.mean(y_train.numpy()), np.mean(y_test.numpy())])
    graph.set_title('Data balance by mean')
    plt.xticks(rotation=70)
    plt.tight_layout()
    plt.show()"""

visualize_holdout_balance(train_dl, test_dl)

EPOCHS = 200
LEARNING_RATE = 0.001

class MLP_1(Module):
    def __init__(self, n_inputs):
        super(MLP_1, self).__init__()
        #1st layer input
        self.hidden1 = Linear(n_inputs, 24)
        kaiming_uniform_(self.hidden1.weight, nonlinearity ='relu') # He initialization
        self.act1 = ReLU()
        #2nd layer
        self.hidden2 = Linear(24, 12)
        kaiming_uniform_(self.hidden2.weight, nonlinearity ='relu') # He initialization
        self.act2 = ReLU()
        #3rd layer
        self.hidden3 = Linear(12, 4) #one node for the predicted value output, also second value is number of classes to predict
        xavier_uniform_(self.hidden3.weight) #GLorot initialization
        self.act3 = Softmax(dim=1) #softmax since it is multiclass
        
    def forward(self, X):
        #input for the 1st layer
        X = self.hidden1(X)
        X = self.act1(X)
        #2nd layer
        X = self.hidden2(X)
        X = self.act2(X)
        #3rd layer and output
        X = self.hidden3(X)
        X = self.act3(X)
        return X
    
model = MLP_1(2161) #number of input features goes into this function

print(summary(model, input_size=(len(train_dl.dataset), len(df_train.columns)), verbose=0))
model.to("cpu")

def train_model(train_dl, val_dl, model):
    #to visualize training process
    liveloss = PlotLosses()
    #define loss function and optimization
    criterion = CrossEntropyLoss() #sparse_categorical_crossentropy
    optimizer = SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9) #stochastic gradient descent
    #iterate the epochs
    for epoch in range(EPOCHS):
        logs = {} #
        #train phase
        model.train()
        running_loss = 0.0
        running_corrects = 0.0
        for inputs, labels in train_dl: #backpropagation
            inputs = inputs.to("cpu")
            labels = labels.to("cpu")
            #calculate model output
            outputs = model(inputs)
            #calculate the loss
            loss = criterion(outputs, labels)
            optimizer.zero_grad() #sets the gradients of all parameters to zero
            loss.backward()
            #update model weights
            optimizer.step()
            running_loss += loss.detach() * inputs.size(0)
            _, preds = torch.max(outputs, 1) #Get predictions from the maximum value
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_dl.dataset)
        epoch_acc = running_corrects.float() / len(train_dl.dataset)
        logs['loss'] = epoch_loss.item()
        logs['accuracy'] = epoch_acc.item()
        
        #Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0.0
        for inputs, labels in val_dl: #backpropagation
            inputs = inputs.to("cpu")
            labels = labels.to("cpu")
            #calculate model output
            outputs = model(inputs)
            #calculate the loss
            loss = criterion(outputs, labels)
            running_loss += loss.detach() * inputs.size(0)
            _, preds = torch.max(outputs, 1) #Get predictions from the maximum value
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(val_dl.dataset)
        epoch_acc = running_corrects.float() / len(val_dl.dataset)
        logs['val_loss'] = epoch_loss.item()
        logs['val_accuracy'] = epoch_acc.item()
        liveloss.update(logs)
        liveloss.send()
        
train_model(train_dl, test_dl, model)

def evaluate_model(test_dl, model):
    predictions = list()
    actual_values = list()
    for i,( inputs, labels) in enumerate(test_dl):
        #evaluate the model with test cases
        yprev = model(inputs)
        #remove numpy array
        yprev = yprev.detach().numpy()
        actual = labels.numpy()
        #convert to labels class
        yprev = np.argmax(yprev, axis = 1)
        #reshape for stacking
        actual = actual.reshape((len(actual), 1))
        yprev = yprev.reshape((len(yprev), 1))
        #save
        predictions.append(yprev)
        actual_values.append(actual)
        break
    predictions, actual_values = np.vstack(predictions), np.vstack(actual_values)
    return predictions, actual_values

def display_confusion_matrix(cm):
    plt.figure(figsize=(16,8))
    sns.heatmap(cm, annot=True,xticklabels=['Class 1', 'Class 2', 'Class 3'],
                yticklabels=['Class 1','Class 2','Class 3'],
                annot_kws={"size":12}, fmt='g', linewidths=.5)
    plt.xlabel('True label')
    plt.ylabel('Predicted label')
    plt.show()
    
predictions, actual_values = evaluate_model(test_dl, model)
               
sucess = 0 
failure = 0
for r,p in zip(actual_values, predictions):
    print(f'real: {r+1} prediction: {p+1}')
    if r==p: sucess +=1
    else: failure+=1

acc = accuracy_score(actual_values, predictions)
print(f'Accuracy: {acc:0.3f}\n')
print(f'sucess:{sucess} failure:{failure}')

print(classification_report(actual_values, predictions))
cm = confusion_matrix(actual_values, predictions)
print(cm)
display_confusion_matrix(cm)

def predict(row, model):
    #convert row to tensor
    row = Tensor([row])
    #make a prediction
    yprev = model(row)
    #remove the numpy array
    yprev = yprev.detach().numpy()
    return yprev

row = [5, 0, 1, 34, 0, 1, 8, 1]
yprev = predict(row, model)
print('Predicted: %s (class=%d)' % (yprev, np.argmax(yprev)+1))