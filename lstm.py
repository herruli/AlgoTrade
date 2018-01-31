import time
import warnings
import numpy as np
from numpy import newaxis
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers.noise import GaussianNoise
from keras.models import Sequential
from keras import metrics
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")
#This will load first 20% data for testing and the last 80% for training
def load_data(stock, seq_len):

    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])

    result = np.array(result)
    row = round(0.8 * result.shape[0])
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, -1][:,-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
    return [x_train, y_train, x_test, y_test]
#This will load first 80% of data for training data and use the last 20% of data for testing
def load_dataforward(stock, seq_len):

    amount_of_features = len(stock.columns)
    data = stock.as_matrix()
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    result = np.array(result)
    row = round(0.2 * result.shape[0])
    train = result[int(row):, :]
    x_train = train[:, :-1]
    y_train = train[:, -1][:,-1]
    x_test = result[:int(row), :-1]
    y_test = result[:int(row), -1][:,-1]
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
    return [x_train, y_train, x_test, y_test]


def model(layers):
    #SETTINGS
    #d  is for dropout rate refer to Bengio journal about drpoout rate can prevent overfitting
    #nodes is for how many nodes in each hidden layer
    #activationChoice is to set different activation function. Keras has softmax, relu, elu, tanh and others
        d = 0.2
        nodes = 500
        activationChoice = 'relu'
        model = Sequential()
        model.add(LSTM(nodes, input_shape=(layers[1], layers[0]),return_sequences=True))
        model.add(Dropout(d))
        model.add(LSTM(nodes, input_shape=(layers[1], layers[0]),return_sequences=False))
        model.add(Dropout(d))
        model.add(Dense(nodes,init='uniform',activation=activationChoice))
        model.add(Dense(nodes,init='uniform',activation=activationChoice))
        model.add(Dense(nodes,init='uniform',activation=activationChoice))
        model.add(Dense(1,init='uniform',activation='linear'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model
