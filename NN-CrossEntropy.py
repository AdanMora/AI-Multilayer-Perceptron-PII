import numpy as np
import pickle
import os
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split

#-------------------------------------------------------------------------------------------------------#

##class Perceptron:
##    X_s = None
##    W_
    
class NN_MultiLayer:
    X = None
    Y = None
    W_s = []
    hyperparameters = []
    actual_hyper = {}

    def addHyperparameter(self, cantCapas, cantH1, cantH2, dropout):
        """Recibe cantidad de capas de la NN, cantidad de perceptrones para cada capa oculta, porcenteje
           de Dropout."""
        nHyperParameter = {}
        nHyperParameter["cantCapas"] = cantCapas        
        nHyperParameter["cantH1"] = cantH1
        nHyperParameter["cantH2"] = cantH2
        nHyperParameter["dropout"] = dropout

        self.hyperparameters.append(nHyperParameter)

    def softmax(self):
        pass

    def crossEntropy(self):
        pass

    def genW_s(self):
        # tipo 0 -> iris
        if not (self.actual_hyper["tipo"]):
            self.W_s = wiris.generateWs(self.actual_hyper["cant_W's"])
            self.tipoD = wiris.tipo
        else:
            self.W_s = wifar.generateWs(self.actual_hyper["cant_W's"],4)
            self.tipoD = wifar.tipo

    def train(self, X, Y):
        """Función que realiza el proceso de entrenamiento, recibe el vector de datos de entrenamiento y el vector con
           la clase a la que corresponde cada uno de los datos. Aquí se entrena el Algoritmo Genético."""
        self.X = X
        self.Y = Y


        # Generar W's

        self.genW_s()

        # Calcular Loss, general y por clase
        ## ...
        count = 0
        for i in range(self.actual_hyper["cant_Gen"]):
            count += 1
            newW_s = []

            for i in range(self.W_s.shape[0]):
                self.hingeLoss_W(self.W_s[i])
                self.W_s[i]["E"] = (np.sum(np.equal(self.classifyTrain(self.X, self.W_s[i]["w"]), self.Y)) / len(self.Y))                
                
            self.W_s = np.sort(self.W_s, order="E")

            self.W_s = self.W_s[::-1]

            masAptos = self.W_s[:int(self.W_s.shape[0]*0.5)]

            print(self.W_s.shape[0])

            if (self.W_s[i]["E"] >= self.actual_hyper["min_Aceptacion"]) or (self.W_s.shape[0] < 10):

                break

            indMenosAptos = self.W_s.shape[0] - int(self.W_s.shape[0]*self.actual_hyper["cant_MenosAptos"])
            menosAptos = self.W_s[indMenosAptos:]

            N_masAptos = masAptos.shape[0]
            N_menosAptos = menosAptos.shape[0]

            diferencia = N_masAptos - N_menosAptos

            cruce1 = masAptos[:diferencia]
            cruce2 = masAptos[diferencia:]

            
            for i in range(cruce1.shape[0] - 1):
                newW_s.append(self.mkCruce(cruce1[i], cruce1[i + 1]))
                
            for i in range(N_menosAptos):
                newW_s.append(self.mkCruce(cruce2[i], menosAptos[i]))


            self.W_s = np.array(newW_s)
            
        self.W = masAptos[0]

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

def plotGrayImage(img):
    from pylab import imshow, show, get_cmap
    imshow(img.reshape(28,28), cmap="gray")
    show()

def main():
    
    
    mnist = fetch_mldata('MNIST original')    
    train_img, test_img, train_lbl, test_lbl = train_test_split(mnist.data, mnist.target, test_size=1/7.0, random_state=0)
    train_lbl = train_lbl.astype(int)
    test_lbl = test_lbl.astype(int)

    plotGrayImage(test_img[100])

    print(test_lbl[100])



main()













