import random
import math

def activation_fun(x):
    return 1.0 / (1.0 + math.exp(-x))

def activation_derivative(x):
    return activation_fun(x)*(1-activation_fun(x))

class Neuron:
    def __init__(self, inputs) -> None:
        self.weights = [(random.uniform(-1, 1)) for i in range(inputs + 1)]
    
    def predict(self, inputs):
        result = 0
        inputs = inputs.copy()
        inputs.append(1)
        for i in range(len(self.weights)):
            result += self.weights[i]*inputs[i]
        result = activation_fun(result)
        return result

class Layer:
    def __init__(self, neurons, inputs) -> None:
        self.neurons = [Neuron(inputs) for i in range(neurons)]
    
    def predict(self, inputs):
        return [neuron.predict(inputs) for neuron in self.neurons]
    
class Network:
    def __init__(self, structure) -> None:
        self.layers = [Layer(structure[i], structure[i-1]) for i in range(1, len(structure))]

    def predict(self, inputs):
        result = inputs
        for layer in self.layers:
            result = layer.predict(result)
        return result

dataset = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0],
]

structure = [2, 2, 1]
network = Network(structure)

for datapoint in dataset:
    print("for inputs", datapoint[0], " expected ", datapoint[1], ", predicted ", network.predict(datapoint[0]))