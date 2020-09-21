import sys
import numpy as np
import matplotlib

# print("Python:", sys.version)
# print("Numpy:", np.__version__)
# print("Matplotlib:", matplotlib.__version__)

#code first neuron

# inputs = [1, 2, 3, 2.5] #three neurons in the previous layer

# weights1 = [0.2, 0.8, -0.5, 1.0]
# weights2 = [0.5, -0.91, 0.26, -0.5]
# weights3 = [-0.26, -0.27, 0.17, 0.87]

# bias1 = 2 #every unique neuron has a unique bias
# bias2 = 3
# bias3 = 0.5

# biases = [2, 3, 0.5]

# weights = [[0.2, 0.8, -0.5, 1.0],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]

#add up the inputs x the weights + the bias

# output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias
# print(output)

#model three neurons with four inputs, 3 unique weight sets

# output = [inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + bias1,
#           inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + bias2,
#           inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + bias3]
# print(output)


#zip together weights and biases, combines two lists element wise
# layer_outputs = []  #output of current layer
# for neuron_weights, neuron_bias in zip(weights, biases):
#     neuron_output = 0 #output of given neuron
#     for n_input, weight in zip(inputs, neuron_weights):
#         neuron_output += n_input*weight
#     neuron_output += neuron_bias
#     layer_outputs.append(neuron_output)

#print(layer_outputs)


inputs = [1, 2, 3, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2

#multiply matrices and vectors element wise
#in this case inputs and weights are interchangeable, 
#both are vectors and the order doesn't change the dot product
# output = np.dot(inputs, weights) + bias
#print(output)

#dot product for a layer of neurons

inputs = [1, 2, 3, 2.5]
biases = [2, 3, 0.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

#weights has to come first because weights is a matrix
#iterate through a matrix of vectors
# output = np.dot(weights, inputs) + biases
# #print(output)

#pass a batch of inputs
inputs = [[1, 2, 3, 2.5],
          [2.0, 5.0, -1.0,2.0],
          [-1.5, 2.7, 3.3, -0.8]]
         
#don't need to change weights and biases yet, still only 3 neurons  
#the size at index 1 of the first element in the dot product needs to 
#match index 0 of the second element passed, need to transpose weights
biases = [2, 3, 0.5]                     
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

output = np.dot(inputs, np.array(weights).T) + biases
#print(output)

##

biases = [2, 3, 0.5]   

weights2 = [[0.1, -0.14, 0.5],
           [-0.5, 0.12, -0.33],
           [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]

#outputs for layer 1 become inputs for layer 2
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

#print(layer2_outputs)

#convert into an object

#input data denoted by capital X

np.random.seed(0)

X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0,2.0],
    [-1.5, 2.7, 3.3, -0.8]]

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
        
#input size is 4, number of neurons as many as you want
# layer1 = Layer_Dense(4,5)
#careful input size dawg
# layer2 = Layer_Dense(5,2)

#pass data through the layer objects

# layer1.forward(X)
# print(layer1.output)

# layer2.forward(layer1.output)
# print(layer2.output)

#work on activation functions, start with a step function
#if input is greater than zero, output will be 1, otherwise 0

#inputs X weights plus bias fet through activation function
#the output will always be a zero or one for this activation

#sigmoid has more granular output than the step function

#Rectified linear activation function ReLU, fast, less complicated than sigmoid
#most popular for hidden layers


X = [[1, 2, 3, 2.5],
    [2.0, 5.0, -1.0,2.0],
    [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

#example of RelU function
for i in inputs:
    if i > 0:
        output.append(i)
    elif i <= 0:
        output.append(0)

print(output)

#Create ReLU object

class Activation_ReLU:
    def forward(self, inputs):
        #another example of ReLU
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

#forward pass with ReLU function
layer1.forward(X)
layer2.forward(layer1.output)
print(layer2.output)

#function to creates a spiral dataset with 3 classes
#https://cs231n.github.io/neural-networks-case-study
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

# import matplotlib.pyplot as plt

# print("wuddup dawg")
# X, y = create_data(100, 3)
            
# plt.scatter(X[:,0], X[:,1], c=y, cmap="brg")
# plt.show()

#in this case we have 2 inputs
layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
#print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

