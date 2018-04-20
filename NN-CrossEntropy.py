import numpy as np
import pickle
import os
import math
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from pylab import imshow, show, get_cmap
import logging

#-------------------------------------------------------------------------------------------------------#

##class Perceptron:
##    X_s = None
##    W_
    
class NN_MLP(object):
    X = None
    used_X = []
    used_Y = []
    validation_X = []
    validation_Y = []
    Y = None
    W_s = []
    Z_s = []
    hyperparameters = []
    actual_hyper = {}

    def __init__(self):
        logging.basicConfig(filename='log.log',level=logging.DEBUG)

    def addHyperparameter(self, cantCapas, cantH1, cantH2, len_batch, learningRate, dropout):
        """Recibe cantidad de capas de la NN, cantidad de perceptrones para cada capa oculta,
           tamaño de cada batch, learning rate, porcentaje de dropout."""
        nHyperParameter = {}
        nHyperParameter["cantCapas"] = cantCapas        
        nHyperParameter["cantH1"] = cantH1
        nHyperParameter["cantH2"] = cantH2
        nHyperParameter["len_batch"] = len_batch
        nHyperParameter["learningRate"] = learningRate
        nHyperParameter["dropout"] = dropout

        self.hyperparameters.append(nHyperParameter)

    def getRandomElements(self, cant, lista_X, lista_Y):
        r_X = []
        r_Y = []
        for i in range(cant):
            ind = np.random.randint(0,len(lista_X))
            r_X.append(lista_X[ind])
            r_Y.append(lista_Y[ind])
            del lista_X[ind]
            del lista_Y[ind]
            
        return r_X, r_Y, np.array(lista_X), np.array(lista_Y)

    def softmax(self):
        pass

    def crossEntropy(self, X, Y):
        return X

    def ReLU(self, X):
        return np.maximum(X,0)

    def dropout(self, X):
        for i in range(int(X.shape[0] * self.actual_hyper["dropout"])):
            ind = np.random.randint(0,len(X))
            X[ind].fill(0)
        return X
                

    def genW_s(self):
        cant = self.actual_hyper["cantCapas"] + 1
        self.W_s = [[]] * cant

        # W1: input to H1
        H1 = []
        self.W_s[0] = np.random.randn(self.actual_hyper["cantH1"], 784)

        # W2: H1 to H2

        if(self.actual_hyper["cantCapas"] == 2):
            H2 = []
            self.W_s[1] = np.random.randn(self.actual_hyper["cantH2"], self.actual_hyper["cantH1"])

        # W3: H2 o H1 to Loss

        if(self.actual_hyper["cantCapas"] == 2):
            self.W_s[2] = np.random.randn(10, self.actual_hyper["cantH2"]) 
        else:
            self.W_s[1] = np.random.randn(10, self.actual_hyper["cantH1"]) #

        self.W_s = np.array(self.W_s)

    def forward(self, X, Y):
        
        h1 = np.dot(self.W_s[0], np.transpose(X))
        self.Z_s.append(self.dropout(self.ReLU(h1)))
        
        
##        print("Shape del batch: ",np.transpose(X).shape)
##        print("Shape de los W's de H1: ",self.W_s[0].shape)
##        print("Shape del resultado: ",h1.shape)
##        print("\n------------\n")

        if(self.actual_hyper["cantCapas"] == 2):
            print(self.Z_s[-1].dtype)
            h2 = np.dot(self.W_s[1], self.Z_s[-1])
            self.Z_s.append(self.dropout(self.ReLU(h2)))
        
            L = np.transpose(np.dot(self.W_s[2], self.Z_s[-1]))
        else:
            L = np.transpose(np.dot(self.W_s[1], self.Z_s[-1]))

##        print("Shape del batch: ",h1.shape)
##        print("Shape de los W's de H1: ",self.W_s[1].shape)
##        print("Shape del resultado: ",L.shape)
##        print("\n------------\n")

        self.Z_s.append(self.crossEntropy(L, Y))

        print(self.Z_s[-1])
        

    def train(self, X, Y):
        """Función que realiza el proceso de entrenamiento, recibe el vector de datos de entrenamiento y el vector con
           la clase a la que corresponde cada uno de los datos. Aquí se entrena el Algoritmo Genético."""
        self.X = X
        self.Y = Y
        
        self.validation_X, self.validation_Y, self.X, self.Y = self.getRandomElements(int(self.X.shape[0]*0.2), list(self.X), list(self.Y))
                
        batch_X, batch_Y, self.X, self.Y = self.getRandomElements(self.actual_hyper["len_batch"], list(self.X), list(self.Y))

        iteracion = 0
        if(self.W_s == []):
            self.genW_s()

        self.forward(batch_X, batch_Y)
        #while(len(X) != 0):
         #   if(iteracion)


        ## logging.debug('This message should go to the log file')
        

    def classifyTrain(self, X, W):
        """Función que implementa el proceso de clasificación una vez obtenida la W definiva, recibe el vector con los datos de prueba.
           Retorna un vector con las respuetas predecidas correspondientes a cada dato de prueba"""
        predict_Y = []
        for i in range(X.shape[0]):
            predict_Y.append(np.argmax(np.dot(W,X[i])))

        return np.array(predict_Y)

    def classify(self, X):
        """Función que implementa el proceso de clasificación una vez obtenida la W definiva, recibe el vector con los datos de prueba.
           Retorna un vector con las respuetas predecidas correspondientes a cada dato de prueba"""
        predict_Y = []
        for i in range(X.shape[0]):
            predict_Y.append(np.argmax(np.dot(self.W["w"],X[i])))

        return np.array(predict_Y)

#-------------------------------------------------------------------------------------------------------#

def unPickle(file):
    """Función para descomprimir los archivos de imágenes de CIFAR-10"""
    with open(file, 'rb') as fo:
        x = pickle.load(fo, encoding='bytes')
    return x

def toPickle(file):
    """Función para descomprimir los archivos de imágenes de CIFAR-10"""
    with open(file, 'rb') as fo:
        x = pickle.load(fo, encoding='bytes')
    return x

def plotGrayImage(img, m = 28, n = 28):
    imshow(img.reshape(m,n), cmap="gray")
    show()

def main():
    
    
    mnist = fetch_mldata('MNIST original')    
    train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
    train_lbl = train_lbl.astype(int)
    test_lbl = test_lbl.astype(int)

    nn = NN_MLP()
    nn.addHyperparameter(2, 64, 32, 16, 0.01, 0.5)
    nn.addHyperparameter(1, 64, 32, 16, 0.01, 0.5)
    nn.actual_hyper = nn.hyperparameters[0]
    
    nn.train(train_img,train_lbl)
   


main()













