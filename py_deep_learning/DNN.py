import numpy as np
import os

class DNN():
    def __init__(self):
        self.layers = []
        np.random.seed(1)
    
    # activation function
    def sigmoid(self, x):
        return(1/(1 + np.exp(-x)))
    
    def f_forward(self, x):
        # Output
        for layer in self.layers:
            z = x.dot(layer, out=None)# input from layer
            a = self.sigmoid(z)# output of layer
        
        return(a)

    # initializing the weights randomly
    def add_layer(self, shape : tuple[int, int]):
        layer = 2 * np.random.random(shape) - 1
        self.layers.append(layer)
        
    # for loss we will be using mean square error(MSE)
    def loss(self, out, Y):
        s = (np.square(out-Y))
        s = np.sum(s)/len(Y)
        return(s)

    # Back propagation of error
    def back_prop(self, x, y, alpha):
        
        # Output layer
        for layer in self.layers:
            z = x.dot(layer, out=None)# input from layer
            a = self.sigmoid(z)# output of layer
        
            # error in output layer
            d2 =(a-y)
            d1 = np.multiply((layer.dot((d2.transpose()))).transpose(),
                                        (np.multiply(a, 1-a)))

            # Gradient for w1 and w2
            layer_adj = x.transpose().dot(d1)
            
            # Updating parameters
            layer = layer-(alpha*(layer_adj))

    def train(self, Y, alpha = 0.01, epoch = 10):
        acc =[]
        loss =[]
        for j in range(epoch):
            l = []
            for i in range(len(self.layers)):
                out = self.f_forward(np.asarray(self.layers[i]))
                l.append(self.loss(out, Y))
                self.back_prop(l[i], Y, alpha)
            print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(self.layers)))*100)
            acc.append((1-(sum(l)/len(self.layers)))*100)
            loss.append(sum(l)/len(self.layers))
        return(acc, loss, self.layers)
