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


from numpy import array, exp


class TrainingData:
    def __init__(self, data: array, solution: array):
        self.data = data
        self.solution = solution
    

class Neurons:
    def __init__(self, value, weight, bias = None):
        self.value = value
        self.weight = weight
        self.bias = bias


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


print(Operations.reLu(100000))