import random
import math

def activation_fun(x):
    return 1.0 / (1.0 + math.exp(-x))

def activation_derivative(x):
    return activation_fun(x)*(1-activation_fun(x))



dataset = [
    [[0, 0], 0],
    [[0, 1], 1],
    [[1, 0], 1],
    [[1, 1], 0],
]