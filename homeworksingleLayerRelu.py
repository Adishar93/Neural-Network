import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(v):
    return np.where(v >= 0, v, 0.01 * v)


def derivative_sigmoid(v):
    return np.where(v >= 0, 1, 0.01)


class NeuralNetwork:
    def __init__(self, X, Y, alpha, input_size, middle_layer_neurons):
        self.X = X.astype(float)
        self.Y = Y.astype(float)
        self.alpha = alpha
        self.W1 = np.random.rand(middle_layer_neurons, input_size) - 0.5
        self.b1 = np.random.rand(middle_layer_neurons, 1) - 0.5
        self.W2 = np.random.rand(1, middle_layer_neurons) - 0.5
        self.b2 = np.random.rand(1, 1) - 0.5

    def feedforward(self):
        self.Z1 = np.dot(self.W1, self.X) + self.b1
        self.A1 = sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1) + self.b2
        self.A2 = sigmoid(self.Z2)
        # print(self.A2)

    def backpropagate(self):
        dZ2 = self.A2 - self.Y
        dW2 = 1 / self.Y.size * dZ2.dot(self.A1.T)
        db2 = 1 / self.Y.size * np.sum(dZ2)
        dZ1 = self.W2.T.dot(dZ2) * derivative_sigmoid(self.A1)
        dW1 = 1 / self.Y.size * dZ1.dot(self.X.T)
        db1 = 1 / self.Y.size * np.sum(dZ1)

        self.W1 = self.W1 - self.alpha * dW1
        self.b1 = self.b1 - self.alpha * db1
        self.W2 = self.W2 - self.alpha * dW2
        self.b2 = self.b2 - self.alpha * db2

    def get_predictions(self):
        convertedA2 = np.round(self.A2)
        return convertedA2

    def get_accuracy(self):
        convertedA2 = self.get_predictions()
        # print(convertedA2, self.Y)
        return np.sum(convertedA2 == self.Y) / self.Y.size

    def setX(self, value):
        self.X = value.astype(float)

    def setY(self, value):
        self.Y = value.astype(float)


def normalizeAllRows(arr):
    min_values = np.min(arr, axis=1, keepdims=True)
    max_values = np.max(arr, axis=1, keepdims=True)
    normalized_array = (arr - min_values) / (max_values - min_values)
    return normalized_array


def oneHotEncode(referenceRow, row):
    categories = np.unique(referenceRow)
    one_hot_encoded = np.zeros((len(row), len(categories)), dtype=int)
    category_to_index = {category: i for i, category in enumerate(categories)}
    for i, value in enumerate(row):
        one_hot_encoded[i, category_to_index[value]] = 1
    return one_hot_encoded


if __name__ == "__main__":
    np.set_printoptions(threshold=sys.maxsize)
    input_data = np.array(pd.read_csv("train_data3.csv"))
    input_labels = np.array(pd.read_csv("train_label3.csv"))
    test_data = np.array(pd.read_csv("test_data1.csv"))

    # tuning parameters
    split_index = int(len(input_data) * 0.8)
    batchSize = 4
    epoch = 10
    middle_layer_neurons = 8
    learningRate = 0.005

    maxAccuracy = 0
    W1 = []
    b1 = []
    W2 = []
    b2 = []

    for i in range(20):
        # Shuffle data
        shuffled_indices = np.random.permutation(len(input_data))
        input_data = input_data[shuffled_indices]
        input_labels = input_labels[shuffled_indices]

        # Initializing training and validation datasets
        training_data = input_data[:split_index]
        training_labels = input_labels[:split_index]
        validation_data = input_data[split_index:]
        validation_labels = input_labels[split_index:]

        allInputData = np.vstack((input_data, test_data))

        # Updating model inputs
        X = normalizeAllRows(training_data.T[[2, 3, 14, 15]])
        X = np.vstack(
            (X, oneHotEncode(allInputData.T[[1]][0], training_data.T[[1]][0]).T)
        )
        validation_X = normalizeAllRows(validation_data.T[[2, 3, 14, 15]])
        validation_X = np.vstack(
            (
                validation_X,
                oneHotEncode(allInputData.T[[1]][0], validation_data.T[[1]][0]).T,
            )
        )
        test_X = normalizeAllRows(test_data.T[[2, 3, 14, 15]])
        test_X = np.vstack(
            (test_X, oneHotEncode(allInputData.T[[1]][0], test_data.T[[1]][0]).T)
        )
        validation_Y = validation_labels.T
        Y = training_labels.T

        # print(Y.astype(float))
        nn = NeuralNetwork(X, Y, learningRate, X.shape[0], middle_layer_neurons)
        yaxisb = [[] for _ in range(np.ceil(X.shape[1] / batchSize).astype(int))]
        yaxis = [[] for _ in range(epoch)]
        yaxisV = [[] for _ in range(epoch)]
        xaxis = [i for i in range(1, epoch + 1)]

        for i in range(epoch):
            k = 0
            for j in range(0, X.shape[1], batchSize):
                if j + batchSize <= X.shape[1]:
                    temp_x = X[:, j : j + batchSize]
                    temp_y = Y[:, j : j + batchSize]
                else:
                    temp_x = X[:, j : X.shape[1]]
                    temp_y = Y[:, j : X.shape[1]]

                nn.setX(temp_x)
                nn.setY(temp_y)
                nn.feedforward()
                nn.backpropagate()

                yaxisb[k] = np.append(yaxisb[k], nn.get_accuracy())

                # if i == epoch-1 or i == 0 or i == 5:
                #    print('epoch i:' + str(i) + ' '+ str(nn.get_accuracy()))
                k += 1
            # generating graph data for accuracy of whole dataset
            nn.setX(X)
            nn.setY(Y)
            nn.feedforward()
            yaxis[i] = np.append(yaxis[i], nn.get_accuracy())
            # generating graph data for accuracy of validation set
            nn.setX(validation_X)
            nn.setY(validation_Y)
            nn.feedforward()
            currentAccuracy = nn.get_accuracy()
            yaxisV[i] = np.append(yaxisV[i], nn.get_accuracy())

        if currentAccuracy > maxAccuracy:
            maxAccuracy = currentAccuracy
            W1 = nn.W1
            b1 = nn.b1
            W2 = nn.W2
            b2 = nn.b2

    # writing predictions for the test data
    nn.setX(test_X)
    nn.setY(Y)
    nn.W1 = W1
    nn.b1 = b1
    nn.W2 = W2
    nn.b2 = b2
    print("Max accuracy:" + str(maxAccuracy))
    nn.feedforward()
    # print(nn.get_predictions())
    data = {"BEDS": nn.get_predictions().astype(int)[0]}
    df = pd.DataFrame(data)
    df.to_csv("output.csv", index=False)

    # plt.plot(xaxis, yaxis, label='Epoch Wise Total Accuracy')
    plt.plot(xaxis, yaxisV, label="Epoch Wise Validation Accuracy")
    # for i in range(k):
    # plt.plot(xaxis, yaxisb[i], label='Line'+str(i+1))

    plt.legend()
    plt.show()
