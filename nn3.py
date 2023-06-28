import random
import math

def activation_fun(x):
    return 1.0 / (1.0 + math.exp(-x))

def activation_derivative(x):
    return activation_fun(x)*(1-activation_fun(x))

class Neuron:
    def __init__(self, inputs) -> None:
        self.weights = [(random.uniform(-1, 1)) for i in range(inputs + 1)]
        self.last_result = 0
        self.last_inputs = []
    
    def predict(self, inputs):
        result = 0
        inputs = inputs.copy()
        inputs.append(1)
        self.last_inputs = inputs
        for i in range(len(self.weights)):
            result += self.weights[i]*inputs[i]
        self.last_result = result
        return activation_fun(result)
    
    def learn(self, correction):
        correction = correction * activation_derivative(self.last_result)
        back_corrections = []
        for i in range(len(self.weights)):
            back_corrections.append(correction * self.weights[i])
            self.weights[i] += correction * self.last_inputs[i]
        return back_corrections

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
        return result[0]
    
    def learn_rec(self, neuron: Neuron, correction, layer):
        corrections = neuron.learn(correction)
        if layer == 0:
            return
        for i in range(len(neuron.weights) -1):
            self.learn_rec(self.layers[layer-1].neurons[i], corrections[i], layer - 1)

    def learn(self, datapoint, alpha):
        [inputs, target] = datapoint
        output = self.predict(inputs)
        error = target - output
        correction = error * alpha
        layer = len(self.layers) - 1
        self.learn_rec(self.layers[layer].neurons[0], correction, layer)
        return abs(error)

dataset = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0],
]

def learn(network: Network, dataset, epochs, alpha):
    for i in range(epochs):
        random.shuffle(dataset)
        errors = 0
        for datapoint in dataset:
            errors += network.learn(datapoint, alpha)
        if i % 1000 == 0:
            print("epoch: ", i, " error: ", errors/4)

structure = [2, 2, 1]
network = Network(structure)
learn(network, dataset, 10000, 0.2)

# network.learn(dataset[1], 100)

for datapoint in dataset:
    print("for inputs", datapoint[0], " expected ", datapoint[1], ", predicted ", network.predict(datapoint[0]))