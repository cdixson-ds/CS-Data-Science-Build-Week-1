import sys
import numpy as np
import matplotlib

#initialize weights and biases
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #randn, gausian distribution bounded around zero
        #np.random.randn is passed parameters which are the shape
        #shaped in this way, inputs then neurons, so that we don't need to 
        #transpose when we do a forward pass
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        #input to np.zeros to pass the shape is a tuple of the shape
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        #outputs of the previous layer
        self.output = np.dot(inputs, self.weights) + self.biases

#activation function class
class Activation_ReLU:
    def forward(self, inputs):
        #another example of ReLU
        self.output = np.maximum(0, inputs)




#create data
def create_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points) #radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)

#after running through ReLU function there are no negative values, more zeros
activation1.forward(layer1.output)
print(activation1.output)

#softmax activation function

#Euler's number, e, could also use math.e?
E = 2.71828182846

layer_outputs = [4.8, 1.21, 2.385]

#calculate exponential value for each value in a vector
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
print('exponential values: ', exp_values) 