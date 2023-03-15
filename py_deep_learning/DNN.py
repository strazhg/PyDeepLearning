from numpy import exp, array, random, dot, save, load
import os

from .Layers import *

class DNN():
    def __init__(self, name : str, learning_rate : float):
        self.learning_rate = learning_rate
        random.seed(1)
        if os.path.exists(name):
            self.layers = load(name, allow_pickle=True)
        else:
            self.shape = tuple()
            self.layers = 2 * random.random(self.shape) - 1

    def add_layer(self, layer : Input):
        self.shape += layer.shape
        self.layers = 2 * random.random(self.shape) - 1

    def sigmoid(self, x):
        return 1 / (1 + exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def think(self, inputs, training : bool = False):
        output = self.sigmoid(dot(array(inputs), self.layers))
        if training:
            adjustment = dot(array(inputs).T, self.learning_rate * self.sigmoid_derivative(output))
            self.layers += adjustment
        return output

    def save_model(self, name : str):
        save(name, self.layers)        

    def learn(self, inputs, learning_rate, iterations):
        for i in range(iterations):
            output = self.think(inputs, True)
            print('{} / {} iterations with output -> {}'.format(i + 1, iterations, output))
        return self
