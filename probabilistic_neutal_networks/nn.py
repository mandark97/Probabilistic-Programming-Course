from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons, load_iris
from tensorflow.keras.utils import to_categorical


class NeuralNetwork(object):
    def __init__(self, mode, n_hidden=5):
        if mode not in ['binary', 'multiclass']:
            raise Exception
        self.mode = mode
        self.n_hidden = n_hidden

    def model(self, X, y):
        if self.mode == 'binary':
            loss = 'binary_crossentropy'
            activation = 'sigmoid'
            classes = 1
        else:
            loss = 'categorical_crossentropy'
            activation = 'softmax'
            classes = y.shape[1]

        model = Sequential([
            layers.Dense(self.n_hidden, input_dim=(X.shape[1]), activation='tanh',
                         kernel_initializer='ones'),
            layers.Dense(classes, activation=activation)
        ])
        model.compile(loss=loss,
                      optimizer='adam', metrics=['acc'])

        return model

    def fit(self, X, y, **kwargs):
        if self.mode == 'multiclass':
            y = to_categorical(y)
        model = self.model(X, y)
        model.fit(X, y, **kwargs)

    def evaluate(self, X, y, **kwargs):
        if self.mode == 'multiclass':
            y = to_categorical(y)
        model = self.model(X, y)
        return model.evaluate(X, y, **kwargs)
