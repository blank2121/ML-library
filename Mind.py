# ================================================= |
#                                                   |
#                   ML-Library                      |
#               by: Winston Walter                  |
#                                                   |
#                                                   |
#                  description:                     |
#   This module will be to help you make simple     |
#   machine learning models. This is more for       |
#   myself to test my ML knowledge from             |
#   Explo.                                          |
#                                                   |
# ================================================= |


from numpy import array, exp, dot

class Neurons:
    def __init__(self, value, weight, bias = None):
        self.value = value
        self.weight = weight
        self.bias = bias
class Training:
    def __init__(self, data: array, solution: array, neurons: Neurons):
        self.data = data
        self.solution = solution
        self.neurons = neurons
    
    def train(self, epochs):
        """this function will train with the data given when defining\
            the Training class"""
        for i in range(epochs):
            input_layer = self.data
            output = Operations.sigmoid(dot(input_layer, ))
            error = output - self.solution

            adjustment = error * Operations.sigmoid_deriv(output)

            self.neurons.bias += dot(input_layer.T, adjustment)
        
        return self.neurons.bias

        
class Operations:
    def sigmoid(x):
        return 1 / (exp(-x))

    def sigmoid_der(x):
        return x * (1 - x)
    

    def reLu(x):
        return max(0,x)

    def reLu_der(x):
        if x > 0:
            return 1
        elif x <= 0:
            return 0

    def error(target, output):
        return (target-output)

