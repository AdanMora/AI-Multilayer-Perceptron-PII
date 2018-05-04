import numpy as np
import pickle
import os
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
    Hist_Loss = []

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
        print("\nLoss: ",loss)
        return np.sum(loss) / self.actual_hyper["len_batch"]

    def ReLU(self, X):
        return np.maximum(X,0)

    def dropout(self, X):
        return X * ((np.random.rand(*X.shape) < self.actual_hyper["dropout"]) / self.actual_hyper["dropout"])

    def ReLU_Prime(self, X):
        return np.heaviside(X,0)

    def softmax_Prime(self, X, Y):
        for i in range(X.shape[0]):
            yi = np.argmax(Y[i])
            y = X[i][yi]
            X[i] *= -y
            X[i][yi] = y*(1 - y)

        print("\nSoftmax: ",X)

        return X
        

    def crossEntropy_Prime(self, p, Y):
##        y = self.one_hot_encode(p.shape,Y)
##        p_i = np.sum(p*y, axis=1)
##        return self.softmax_Prime(p, y) / p_i[:,None]

        p[range(Y.size),Y] =- 1
        return p

    def one_hot_encode(self, shape, Y):
        y = np.zeros(shape)
        y[np.arange(Y.shape[0]), Y] = 1
        return y
                

    def genW_s(self):
        # np.random.randn(n) * sqrt(2.0/n)
        
        cant = self.actual_hyper["cantCapas"] + 1
        self.W_s = []

        n = self.actual_hyper["len_batch"]

        # W1: input to H1
        H1 = []
        self.W_s.append(np.random.randn(self.actual_hyper["cantH1"], 784) * np.sqrt(self.actual_hyper["cantH1"]))

        # W2: H1 to H2

        if(self.actual_hyper["cantCapas"] == 2):
            H2 = []
            self.W_s.append(np.random.randn(self.actual_hyper["cantH2"], self.actual_hyper["cantH1"]) * np.sqrt(self.actual_hyper["cantH2"]))
            self.W_s.append(np.random.randn(10, self.actual_hyper["cantH2"]) * np.sqrt(10))
            
        else:
            self.W_s.append(np.random.randn(10, self.actual_hyper["cantH1"]) * np.sqrt(10))

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

        print("\nSoftmax: ",o)

        print("\nSoftmax - 1: ",o - 1)

        print("\nSoftmax - 1 ++: ",np.sum(o - 1, axis = 1))

        print("\nSoftmax ++",np.sum(o, axis = 1))

##        print(o)
##
##        print(np.sum(o,1))
##
##        print(self.crossEntropy(o))
        return o

    def backward(self,X,Y,o):

        loss = self.crossEntropy(o)

        self.Hist_Loss.append(loss)

        o_error = loss - o
        print("\no_error: ",o_error)

        d = self.crossEntropy_Prime(o, Y)

        print("\nd: ",d)

        print("\nCross entropy ++",np.sum(d, axis = 1))

        o_delta = o_error * d

        print("\no_delta: ",o_delta)

        #z3_error = np.dot(o_delta, self.W_s[2])
        #z3_delta = z3_error * self.Z_s[2]

        


        
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

        o = self.forward(batch_X / 255, batch_Y)

        self.backward(batch_X, batch_Y, o)

        #while(len(X) != 0):
         #   if(iteracion)


        ## logging.debug('This message should go to the log file')

    def plotGraphic(self, titulo):
        import matplotlib.pyplot as plt
        fig1 = plt.figure(figsize = (8,8))
        plt.subplots_adjust(hspace=0.4)
        
##        p1 = plt.subplot(2,1,1)
##        l1 = plt.plot(list(range(self.CantTotal_Gen)), self.Hist_Eficiencia, 'g-')
##        xl = plt.xlabel('Generación n')
##        yl = plt.ylabel('% Exactitud')
##        grd = plt.grid(True)

        p2 = plt.subplot(2,1,2)
        ll2 = plt.plot(list(range(self.CantTotal_Gen)), self.Hist_Loss, 'c-')
        xxl = plt.xlabel('Generación n')
        yyl = plt.ylabel('% Loss')
        grd1 = plt.grid(True)

        sttl = plt.suptitle(titulo)
        plt.savefig(os.path.join(os.environ["HOMEPATH"], "Desktop\\Pruebas-CIFAR\\" + titulo + " - Tipo " + str(self.actual_hyper["tipo"]) + '.png'))
        fig1.clf()
        #plt.show()
        

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

##    a = np.array([[0.2,0.1,0.7],[0.7,0.9,0.6],[0.1,0.05,0.8]])
##    Y = np.array([1,2,0])
##
##    y = np.zeros(a.shape)
##    y[np.arange(Y.shape[0]), Y] = 1
##
##    c = a * y
##    print(c)
##    
##    b = np.sum(c, axis = 1)
##
##    print(b)
##
##    U1 = np.random.rand(*a.shape) < 0.5
##    U2 = (np.random.rand(*a.shape) < 0.5) / 0.5
##
##    print(*a.shape)
##
##    print(a)
##
##    print(U1)
##    print(U2)
##    
##    b = a * U1
##    c = a * U2
##
##    print(b)
##    print(c)
    
main()













