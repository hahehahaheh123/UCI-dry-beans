import csv
import pandas as pd

headers = ['A', 'P', 'L', 'l', 'K', 'Ec', 'C', 'Ed', 'Ex', 'S', 'R', 'CO', 'SF1',
'SF2', 'SF3', 'SF4', 'Class']

beans_df = pd.read_csv('Dry_Bean_Dataset.dat', sep=',', names=headers)

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

X = beans_df.drop(columns=['Class'])

beans_df['Class'] = beans_df['Class'].replace('SEKER', 0)
beans_df['Class'] = beans_df['Class'].replace('BARBUNYA', 1)
beans_df['Class'] = beans_df['Class'].replace('BOMBAY', 2)
beans_df['Class'] = beans_df['Class'].replace('CALI', 3)
beans_df['Class'] = beans_df['Class'].replace('HOROZ', 4)
beans_df['Class'] = beans_df['Class'].replace('SIRA', 5)
beans_df['Class'] = beans_df['Class'].replace('DERMASON', 5)

y_label = beans_df['Class'].values.reshape(X.shape[0], 1)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y_label, test_size=0.2, random_state=2)

sc = StandardScaler()
sc.fit(Xtrain)
Xtrain = sc.transform(Xtrain)
Xtest = sc.transform(Xtest)

class NeuralNet():
    def __init__(self, layers=[16, 10, 10, 7], learning_rate=0.01, iterations=400, mb_size=32):
        self.layers = layers
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.mb_size = mb_size
        self.loss = []
        self.params = {}
        self.sample_size = None
        self.X = None
        self.y = None
       
    def init_weights(self):
        np.random.seed(1)
        self.params['W1'] = np.random.randn(self.layers[0], self.layers[1])
        self.params['b1'] = np.random.randn(self.layers[1],)
        self.params['W2'] = np.random.randn(self.layers[1], self.layers[2])
        self.params['b2'] = np.random.randn(self.layers[2],)
        self.params['W3'] = np.random.randn(self.layers[2], self.layers[3])
        self.params['b3'] = np.random.randn(self.layers[3],)
        
    def sigmoid(self, Z):
        return 1/(1+np.exp(-Z))
        
    def one_arr(self, x):
        X = np.array([0, 0, 0, 0, 0, 0, 0])
        X[x] = 1
        return X
        
    def out_vec(self, y):
        l = []
        for x in y:
            l.append(self.one_arr(x))
        l = np.asarray(l)
        return l
     
    def MSE(self, y, yhat):
        loss = 1/len(y) * np.sum(np.sum(np.square(yhat - self.out_vec(y)), axis=1))
        return loss
      
    def forward_propagation(self, X_mbr, y_mbr):
        Z1 = X_mbr.dot(self.params['W1']) + self.params['b1']
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.sigmoid(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        yhat = self.sigmoid(Z3)
        loss = self.MSE(y_mbr, yhat)
        
        self.params['Z1'] = Z1
        self.params['A1'] = A1
        self.params['Z2'] = Z2
        self.params['A2'] = A2
        self.params['Z3'] = Z3
        
        return yhat, loss
        
    def back_propagation(self, X_mbr, y_mbr, yhat):
        avgd = 1/len(X_mbr)
        dl_wrt_yhat = 2*(yhat - self.out_vec(y_mbr))
        dl_wrt_Z3 = dl_wrt_yhat * (yhat * (1 - yhat))
        dl_wrt_b3 = avgd * np.sum(dl_wrt_Z3, axis=0, keepdims=True)
        dl_wrt_W3 = avgd * self.params['A2'].T.dot(dl_wrt_Z3)
        
        dl_wrt_A2 = self.params['W3'].dot(dl_wrt_Z3.T).T
        dl_wrt_Z2 = dl_wrt_A2 * (self.params['A2'] * (1 - self.params['A2']))
        dl_wrt_b2 = avgd * np.sum(dl_wrt_Z2, axis=0, keepdims=True)
        dl_wrt_W2 = avgd * self.params['A1'].T.dot(dl_wrt_Z2)
        
        dl_wrt_A1 = self.params['W2'].dot(dl_wrt_Z2.T).T
        dl_wrt_Z1 = dl_wrt_A1 * (self.params['A1'] * (1 - self.params['A1']))
        dl_wrt_b1 = avgd * np.sum(dl_wrt_Z1, axis=0, keepdims=True)
        dl_wrt_W1 = avgd * X_mbr.T.dot(dl_wrt_Z1)
        
        self.params['W1'] = self.params['W1'] - self.learning_rate * dl_wrt_W1
        self.params['W2'] = self.params['W2'] - self.learning_rate * dl_wrt_W2
        self.params['W3'] = self.params['W3'] - self.learning_rate * dl_wrt_W3
        self.params['b1'] = self.params['b1'] - self.learning_rate * dl_wrt_b1
        self.params['b2'] = self.params['b2'] - self.learning_rate * dl_wrt_b2
        self.params['b3'] = self.params['b3'] - self.learning_rate * dl_wrt_b3
        
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.init_weights()
        split_div = round(len(self.X) / self.mb_size)
        
        X_mb = np.array_split(self.X, split_div)
        y_mb = np.array_split(self.y, split_div)
        
        for i in range(self.iterations):
            for j in range(0, split_div):
                total_sample_loss = 0
                yhat, loss = self.forward_propagation(X_mb[j], y_mb[j])
                self.back_propagation(X_mb[j], y_mb[j], yhat)
                total_sample_loss += loss
                
            self.loss.append(total_sample_loss)
        
    def predict_nr(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.sigmoid(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        pred = self.sigmoid(Z3)
        return pred
        
    def predict_r(self, X):
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.sigmoid(Z1)
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.sigmoid(Z2)
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        pred = self.sigmoid(Z3)
        return np.round(pred)
        
    def row_equals(self, w, z):
        width = range(0, 6)
        if all(w[i] == z[i] for i in width) == True:
          return 1
        else:
          return 0
        
    def re_list(self, y, yhat):
        list = []
        for i in range(0, len(y)):
            list.append(self.row_equals(y[i], yhat[i]))
        list = np.asarray(list)
        return list
        
    def acc(self, y, yhat):
        re_counter = sum(self.re_list(y, yhat))
        return re_counter / len(y) * 100
        
    def plot_loss(self):
        plt.plot(self.loss)
        plt.xlabel("Iteration")
        plt.ylabel("loss")
        plt.title("Loss curve after fitting")
        plt.show()
