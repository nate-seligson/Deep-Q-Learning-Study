#neural network
import numpy as np
from random import uniform
#sigmoid and sigmoid derivative
sig = np.vectorize(lambda n: (1 + (np.exp(-n))) ** -1)
sigderiv = np.vectorize(lambda n: sig(n) * (1 - sig(n)))
class Layer:
    def __init__(self, rows, columns):
        self.weights = np.array([[0 for _ in range(rows)] for _ in range(columns)]) 
        self.bias = np.array([0 for _ in range(rows)])
    def randomize(self):
        random = np.vectorize(lambda _: uniform(-1,1))
        self.weights = random(self.weights)
        self.bias = random(self.bias)
class Network:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, layers):
        self.layers = []
        #learning rate
        self.lr = 0.01

        self.layers.append(Layer(hidden_nodes,input_nodes))
        for _ in range(layers):
            self.layers.append(Layer(hidden_nodes, hidden_nodes))
        self.layers.append(Layer(output_nodes, hidden_nodes))
    def randomize(self):
        for layer in self.layers:
            layer.randomize()
    def forward_prop(self,input):
        res = input
        #iterate through each layer, res keeps the input to each layer
        for layer in self.layers:
            res = np.add(res.dot(layer.weights), layer.bias)
            res = sig(res)
        return res
    def back_prop(self, input, expected):
        layerOuts = []
        res = input
        #get ouput from each layer
        for layer in self.layers:
            z = np.add(res.dot(layer.weights), layer.bias)
            a = sig(z)
            layerOuts.append((res, z, a))
            res = a
    
        changes = []
        #initialize cost from output to expected
        dc = 2 * (layerOuts[-1][2]-expected)
        #go backwards through each layer
        for i in range(len(self.layers)-1, -1, -1):
            layer_input, z, a = layerOuts[i]
            #change in layer over change in activated result
            dL_da = dc
            #change in activated result over change in unactivated result
            da_dz = sigderiv(z)
            #multiplied out to get change in layer over change in unactiated result
            dL_dz = dL_da* da_dz

            change = Layer(len(self.layers[i].weights[0]), len(self.layers[i].weights))
            #get outer product from layer weights and change in layer over unactivated result
            change.weights = np.outer(layer_input, dL_dz)
            #no activation from bias, change is same as dL_dz
            change.bias = dL_dz
            changes.append(change)

            if i > 0:
                #reset dc to be output from previous layer
                dc = dL_dz.dot(self.layers[i].weights.T)
        return changes[::-1]
    def apply_changes(self, changes):
        for i, change in enumerate(changes):
            self.layers[i].weights -= self.lr * change.weights
            self.layers[i].bias -= self.lr * change.bias
