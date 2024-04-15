import sys
import numpy as np
import pandas as pd

# USE_DATASET_SPLIT 2

def sigmoid(v):
    #print(v)
    return 1.0/(1+ np.exp(-v))

def derivative_sigmoid(v):
    #print(v)
    return v * (1.0 - v)

# 2 inputs and 5 hidden layer, 1 output
class NeuralNetwork:
    def __init__(self, X, Y, alpha):
        self.X = X.astype(float)
        self.Y = Y.astype(float)
        self.alpha = alpha
        self.W1 = np.random.rand(5, 1) - 0.5
        self.b1 = np.random.rand(5, 1) - 0.5
        self.W2 = np.random.rand(1, 5) - 0.5
        self.b2 = np.random.rand(1, 1) - 0.5

    def feedforward(self):
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = sigmoid(self.Z2)
        #print(self.A2)

    def backpropagate(self):
        dZ2 = self.A2 - (self.Y/50)
        dW2 = 1/self.Y.size * dZ2.dot(self.A1.T)
        db2 = 1/self.Y.size * np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * derivative_sigmoid(self.A1)
        dW1 = 1/self.Y.size * dZ1.dot(self.X.T)
        db1 = 1/self.Y.size * np.sum(dZ1)

        self.W1 = self.W1 - self.alpha * dW1
        self.b1 = self.b1 - self.alpha * db1
        self.W2 = self.W2 - self.alpha * dW2
        self.b2 = self.b2 - self.alpha * db2

    def get_predictions(self):
        convertedA2 = np.round(self.A2 * 50)
        return convertedA2

    def get_accuracy(self):
        convertedA2 = self.get_predictions()
        #print(convertedA2, self.Y)
        return np.sum(convertedA2 == self.Y) / self.Y.size

    def setX(self, value):
        # Here, you can add any validation or processing logic before setting the attribute
        self.X = value.astype(float)

    def setY(self, value):
        # Here, you can add any validation or processing logic before setting the attribute
        self.Y = value.astype(float)

if __name__ == "__main__":
    training_data = np.array(pd.read_csv('train_data.csv'))
    training_labels = np.array(pd.read_csv('train_label.csv'))
    test_data = np.array(pd.read_csv('test_data.csv'))
    batchSize = 100
    epoch = 10

    X = training_data.T[3:4]
    test_X = test_data.T[3:4]
    Y = training_labels.T
    np.set_printoptions(threshold=sys.maxsize)    

    #print(Y.astype(float))
    nn = NeuralNetwork(X,Y,0.2)
    yaxisb = [[] for _ in range(np.ceil(X.size/batchSize).astype(int))]
    yaxis = [[] for _ in range(epoch)]
    xaxis = [i for i in range(1, epoch+1)]

    for i in range(epoch):
        k = 0
        for j in range(0, X.size, batchSize):
            if j+batchSize<=X.size:
                temp_x = X[:,j:j+batchSize]
                temp_y = Y[:,j:j+batchSize]
            else:
                temp_x = X[:,j:X.size]
                temp_y = Y[:,j:X.size]
            
            nn.setX(temp_x)
            nn.setY(temp_y)
            nn.feedforward()
            nn.backpropagate()

            yaxisb[k] = np.append(yaxisb[k], nn.get_accuracy())

            #if i == epoch-1 or i == 0 or i == 5:
            #    print('epoch i:' + str(i) + ' '+ str(nn.get_accuracy()))
            k+=1
        nn.setX(X)
        nn.setY(Y)
        nn.feedforward()
        yaxis[i] = np.append(yaxis[i], nn.get_accuracy())

    nn.setX(test_X)
    nn.setY(Y)
    nn.feedforward()
    #print(nn.get_predictions())
    data = {'BEDS': nn.get_predictions().astype(int)[0]}
    df = pd.DataFrame(data)
    df.to_csv('output.csv', index=False)
        

    #plt.plot(xaxis, yaxis, label='Epoch Wise Total Accuracy')
    #for i in range(k):
        #plt.plot(xaxis, yaxisb[i], label='Line'+str(i+1))
    
    #plt.legend()
    #plt.show()

