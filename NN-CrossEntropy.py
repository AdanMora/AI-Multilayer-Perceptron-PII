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
            
        return np.array(r_X), np.array(r_Y), np.array(lista_X), np.array(lista_Y)

    def softmax(self, X):
        e_x = np.exp(X - np.max(X))
        e_x += np.finfo(float).eps
        suma = np.sum(e_x, 1)
        return  e_x / suma[:,None]

    def crossEntropy(self, p):
        loss = -np.log(p)
        return np.sum(loss) / self.actual_hyper["len_batch"]

    def ReLU(self, X):
        return np.maximum(X,0)

    def dropout(self, X):
        for i in range(int(X.shape[0] * self.actual_hyper["dropout"])):
            ind = np.random.randint(0,len(X))
            X[ind].fill(0)
        return X

    def ReLU_Prime(self, X):
        pass

    def softmax_Prime(self, X):
        pass

    def crossEntropy_Prime(self, p):
        pass

    def one_hot_encode(self, Y):
        y = np.zeros(self.actual_hyper["len_batch"])
        y[np.arange(Y.size), Y] = 1
        return y
                

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
        
        h1 = self.ReLU(np.dot(self.W_s[0], np.transpose(X)))
        self.Z_s.append(h1)
        h1 = self.dropout(h1)

        if(self.actual_hyper["cantCapas"] == 2):
            h2 = self.ReLU(np.dot(self.W_s[1], h1))
            self.Z_s.append(h2)
            h2 = self.dropout(h2)
        
            L = np.transpose(np.dot(self.W_s[2], h2))
        else:
            L = np.transpose(np.dot(self.W_s[1], h1))

##        print("Shape del batch: ",h1.shape)
##        print("Shape de los W's de H1: ",self.W_s[1].shape)
##        print("Shape del resultado: ",L.shape)
##        print("\n------------\n")

        o = self.softmax(L)

##        print(o)
##
##        print(np.sum(o,1))
##
##        print(self.crossEntropy(o))
        return o

    def backward(self,X,Y,o):

        loss = self.crossEntropy(o)

        o_error = loss - o
        print(o_error)

        o_delta = o_error * self.crossEntropy_Prime(o)

        print(o_delta.shape)
        print(o_delta)

        

        
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

        o = self.forward(batch_X, batch_Y)

        self.backward(batch_X, batch_Y, o)
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
    nn.addHyperparameter(2, 32, 16, 8, 0.01, 0.5)
    nn.addHyperparameter(1, 32, 16, 8, 0.01, 0.5)
    nn.actual_hyper = nn.hyperparameters[0]
    
    nn.train(train_img,train_lbl)
   


main()













