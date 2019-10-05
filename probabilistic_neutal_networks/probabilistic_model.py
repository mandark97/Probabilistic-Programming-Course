import pymc as pm
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return (1. / (1. + np.exp(-x)))


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)


class ProbabilisticModel(object):
    def __init__(self, mode, n_hidden=5):
        if mode not in ['binary', 'multiclass']:
            raise Exception
        self.mode = mode
        self.n_hidden = n_hidden
        self.mcmc = None

    def _base_model(self, X, y):
        weights_in_1 = pm.Normal('w_in_1', 0, 1,
                                 size=(X.shape[1], self.n_hidden))

        # Weights from 1st to 2nd layer
        weights_1_2 = pm.Normal('w_1_2', 0, 1,
                                size=(self.n_hidden, self.n_hidden))

        # Weights from hidden layer to output
        output_size = (self.n_hidden, ) if self.mode == 'binary' else (
            self.n_hidden, len(np.unique(y)))
        weights_2_out = pm.Normal('w_2_out', 0, 1,
                                  size=output_size)

        @pm.deterministic
        def act_1(x=X, weights_in_1=weights_in_1):
            return np.tanh(np.dot(x, weights_in_1))

        @pm.deterministic
        def act_2(act_1=act_1, weights_1_2=weights_1_2):
            return np.tanh(np.dot(act_1, weights_1_2))

        @pm.deterministic
        def act_out(act_2=act_2, weights_2_out=weights_2_out):
            x = np.dot(act_2, weights_2_out)
            if self.mode == 'binary':
                return sigmoid(x)
            else:
                return softmax(x)

        self.weights = [weights_in_1, weights_1_2, weights_2_out]
        return act_out

    def binary(self, X, y):
        act_out = self._base_model(X, y)
        # Binary classification -> Bernoulli likelihood
        out = pm.Bernoulli("y", p=act_out,
                           value=y,
                           observed=True)
        return pm.Model(self.weights + [out])

    def multiclass(self, X, y):
        act_out = self._base_model(X, y)

        # Binary classification -> Bernoulli likelihood
        out = pm.Categorical("y", p=act_out,
                             value=y,
                             observed=True)
        return pm.Model(self.weights + [out])

    def fit(self, X, y, steps=200000, cut=100000):
        if self.mode == 'binary':
            model = self.binary(X, y)
        else:
            model = self.multiclass(X, y)

        self.mcmc = pm.MCMC(model)
        self.mcmc.sample(steps, cut, 1)

    def evaluate(self, X, y):
        w1, w12, w2 = map(lambda x: x.value, self.weights)
        act1 = np.tanh(np.dot(X, w1))
        act2 = np.tanh(np.dot(act1, w12))
        x = np.dot(act2, w2)
        if self.mode == 'binary':
            out = sigmoid(x)
            pred = out > 0.5
        else:
            out = softmax(x)
            pred = np.argmax(out, axis=1)
        return (pred == y).mean()
