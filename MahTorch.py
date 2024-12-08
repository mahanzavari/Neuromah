import numpy as np
import os 
import cv2
import pickle 
import copy
import networkx
"""
Notes :
- The linear activation is mostly used in the output layer of regression problems
- Softmax is a better choice for the output layer of a classification problem ( model's confidence )since it's output is normalized and dependant on each other 
unlike ReLU
- A categorical Cross Entropy accounts for the model's confidence and outputs a lower loss when the confidence is higher
- when there is a flashy wiggle in the model's prediction visualization, it means that learning rate is too high
-  Remember, even if we desire
the best accuracy out of our model, the optimizer’s task is to decrease loss, not raise accuracy
directly. Loss is the mean value of all of the sample losses, and some of them could drop
significantly, while others might rise just slightly, changing the prediction for them from a correct
to an incorrect class at the same time
- a solution to the above problem is to use a learning rate decay
- AdaGrad provides a way to normalize parameter updates by keeping a history of previous updates — the bigger the sum of the updates is, in either
direction (positive or negative), the smaller updates are made further in training. This lets less-frequently updated parameters to keep-up with changes,
effectively utilizing more neurons for training.
- One general rule to follow when selecting initial model hyperparameters is to find the smallest model possible
that still learns.
- The process of trying different model settings is called hyperparameter searching
- . It is usually fine to scale datasets that consist of larger numbers than the training data using a scaler prepared on the training data. If the 
resulting numbers are slightly outside of the -1 to 1 range, it does not affect validation or testing negatively, since we do not train on these data.
-be aware that non-linear scaling can leak the information from other datasets to the training dataset and, in this case,
the scaler should be prepared on the training data only
"""
class Dense_Layer:
     def __init__(self, num_inputs, num_neurons, weight_regularizer_l1=0, weight_regularizer_l2=0, bias_regularizer_l1=0, bias_regularizer_l2=0):
          self.weights = 0.01 * np.random.randn(num_inputs , num_neurons) # 0.01 == so as to speed up the training
          self.biases = np.zeros((1 , num_neurons)) # ensuring that each neuron fires
          # self.weight_regularizer_l1 = weight_regularizer_l1
          # self.weight_regularizer_l2 = weight_regularizer_l2
          # self.bias_regularizer_l1 = bias_regularizer_l1
          # self.bias_regularizer_l2 = bias_regularizer_l2
     def forward(self , inputs , training):
          self.inputs = inputs # so we would remember what the inputs were
          self.outputs = np.dot(self.weights , inputs) + self.biases
     def backward(self , dvalues):
          self.dweights = np.dot(self.inputs.T , dvalues)
          self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
          if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1
          if self.weight_regularizer_l2 > 0:
              self.dweights += 2 * self.weight_regularizer_l2 * \
                               self.weights
          if self.bias_regularizer_l1 > 0:
              dL1 = np.ones_like(self.biases)
              dL1[self.biases < 0] = -1
              self.dbiases += self.bias_regularizer_l1 * dL1
          if self.bias_regularizer_l2 > 0:
              self.dbiases += 2 * self.bias_regularizer_l2 * \
                              self.biases
          self.dinputs = np.dot(dvalues, self.weights.T)
     def get_parameter(self): # getter
          return self.weights , self.biases
     def set_parametes(self , weights , biases):
          self.weights = weights
          self.biases = biases
     def load_parameters(self):
          return self.weights , self.biases
     def save_parameters(self , weights , biases):
          self.biases = biases
          self.weights = weights

class Layer_Dropout:
     def __init__(self , rate):
          self.rate = 1 - rate
     def forward(self , inputs , training):
          self.inputs = inputs
          if not training :
               print()
class Layer_Input:
     def forward(self, inputs, training):
        self.output = inputs
class ReLU_Activation:
     def forward(self , inputs):
          self.inputs = inputs
          self.output = np.maximum(0, inputs)
     def backward(self , dvalues):
          # copy for modifying
          self.dinputs = dvalues.copy()
          self.dinputs[self.inputs <= 0 ] = 0 # zero grad where inputs  
class Softmax_Activation:
     def forward(self , inputs):
          exp_values = np.exp(inputs - np.max(inputs , axis=1 , keepdims=True))
          probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
          self.output = probabilities
     def backward(self, dvalues):
         self.dinputs = np.empty_like(dvalues)
         for index, (single_output, single_dvalues) in \
                 enumerate(zip(self.output, dvalues)):
             single_output = single_output.reshape(-1, 1)
             jacobian_matrix = np.diagflat(single_output) - \
                               np.dot(single_output, single_output.T)
             self.dinputs[index] = np.dot(jacobian_matrix,
                                          single_dvalues)
     def predictions(self, outputs):
         return np.argmax(outputs, axis=1)
class Loss:
     def calculate(self , output , y):
          sample_losses = self.forward(output , y)
          data_loss = np.mean(sample_losses)
          return data_loss
class Loss_categoricalCrossEntropy(Loss):
     def forward(self , y_pred , y_true):
          samples = len(y_pred)
          # clipping the data to prevent division by 0 and prevent to drag mean toward any data
          y_pred_clipped = np.clip(y_pred , 1e-7 , 1 - 1e-7)
          if len(y_true.shape) == 1:
               correct_confidences = y_pred_clipped[
                    range(samples),
                    y_true
                    ]
          elif len(y_true.shape)== 2:
               correct_confidences = np.sum(
                    y_pred_clipped * y_true,
                    axis=1
               )
          negative_log_likelihoods = -np.log(correct_confidences)
          return negative_log_likelihoods
     def backward(self , dvalues , y_true):
          samples_num = len(dvalues)
          label_num = len(dvalues[0])
          # If sparse turn them into onr-hot
          if len(y_true.shape) == 1:
               y_true = np.eye(label_num)[y_true] 
          # Gradient
          self.dinputs = -y_true / dvalues # np.divide[x , y]
          self.dinputs = self.dinputs / samples_num
class Activation_Softmax:
     def forward(self , inputs , training):
          self.inputs = inputs
          exp_values = np.exp(inputs - np.max(inputs, axis=1,keepdims=True))
          probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
     def backward(self , dvalues):
          self.inputs = np.empty_like(dvalues)
          for i , (single_output , single_dvalues) in enumerate(zip(self.output , dvalues)):
               # Flatten
               single_output = single_output.reshape(-1 , 1)
               jacobian_mat = np.diagflat(single_output) - np.dot(single_output , single_output.T)
               self.dinputs[i] = np.dot(jacobian_mat , single_dvalues)
class Activation_Softmax_Loss_CategoricalCrossEntropy():
     def __init__(self):
          self.activation = Activation_Softmax()
          self.loss = Loss_categoricalCrossEntropy()
          
     def forward(self , inputs , y_true):
          self.activation.forward(inputs)
          self.output = self.activation.output
          return self.loss.calculate(self.output , y_true)
     def backward(self , dvalues , y_true):
          sample_num = len(dvalues)
          if len(y_true.shape) == 2: # turn one-hot values into discrete ones
               y_true = np.argmax(y_true , axis =1)
          self.dinputs = dvalues.copy()
          self.dinputs[range(sample_num) , y_true] -= 1
          self.dinputs = self.dinputs / sample_num
class Optimizer_SGD:
     def __init__(self , LEARNING_RATE = 1.0 , decay=0. , momentum = 0):
          self.LEARNING_RATE = LEARNING_RATE
          self.decay = decay
          self.current_learning_rate = LEARNING_RATE
          self.iter_num = 0
          self.momentum  = momentum
     # called before any update
     def pre_update_params(self):
          if self.decay:
               self.current_learning_rate = self.LEARNING_RATE*(1/(1 + self.decay*self.iter_num)) # exponential decay
          
     def update_params(self , layer):
          if self.momentum:
               if not hasattr(layer , 'weight_momentums'): # if the layer does not already have a momentum
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.biases_momentums = np.zeros_like(layer.biases)
               weight_updates = self.momentum * self.weight_momentums - self.current_learning_rate* layer.dweights
               layer.weight_momentums = weight_updates
               biases_updates = self.momentum * layer.biases_momentums - self.current_learning_rate* layer.dbiases
               layer.biases_momentums = biases_updates
          else:
               weight_updates = -self.current_learning_rate * layer.dweight
               biases_updates = -self.current_learning_rate * layer.dbiases
          
          layer.weights += -self.LEARNING_RATE * layer.dweights
          layer.biases += -self.LEARNING_RATE * layer.dbiases

     def post_update_params(self):
          self.iter_num += 1
class AdaGrad:
     def __init__(self , learning_rate =1.0 , decay = 0. , epsilon = 1e-7): # a momentum maybe?
          self.learning_rate = learning_rate
          self.decay = decay
          self.epsilon = epsilon
          self.current_lr = learning_rate
          self.iter_num = 0
     def pre_update_params(self):
          if self.decay:
               self.current_lr = self.learning_rate / (1 + self.decay*self.iter_num)
     def update_params(self, layer):
          if not hasattr(layer , 'weight_cache'):
               layer.weight_cache = np.zeros_like(layer.weights)
               layer.bias_cache = np.zeros_like(layer.biases)
          # cache update
          layer.weight_cache += layer.dweights**2
          layer.bias_cache += layer.dbiasas**2
          # epsilon is used for preventing division by zero
          layer.weights += -self.current_lr*layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
          layer.biases += -self.current_lr*layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
     def post_update_params(self):
          self.iter_num += 1
class Optimizer_RMSProp:
     def __init__(self , learning_rate = 0.001, epsilon = 1e-7 , decay= 0.  , rho = 0.9):
          self.learning_rate = learning_rate
          self.current_learning_rate = learning_rate
          self.decay = decay
          self.epsilon = epsilon
          self.rho = rho
          self.iter_num = 0
     def pre_update_params(self):
          if self.decay:
               self.current_learning_rate = self.current_learning_rate * (1 / 1 + self.decay * self.iter_num)
     def update_params(self , layer):
          if not hasattr(layer , 'weight_cache'):
               layer.weight_cache = np.zeros_like(layer.weights)
               layer.bias_cache = np.zeros_like(layer.biases)
          layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2 
          layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases**2 
          
          layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
          layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
     def post_update_params(self):
          self.iter_num += 1
class Optimizer_Adam:
     def __init__(self , learning_rate = 0.001 , decay = 0. , epsilon = 1e-7 , beta_1 = 0.9 , beta_2 = 0.999):
          self.learning_rate = learning_rate
          self.current_learning_rate = learning_rate
          self.decay = decay
          self.epsilon = epsilon
          self.iter_num = 0
          self.beta_1 = beta_1
          self.beta_2 = beta_2
     def pre_update_params(self):
          if self.decay:
               self.current_learning_rate = self.current_learning_rate * (1 / (1 + self.decay * self.iter_num))
     def update_params(self, layer):
          if not hasattr(layer , 'weight_cache'):
               layer.weight_cache = np.zeros_like(layer.weights)
               layer.bias_cache = np.zeros_like(layer.biases)
               layer.weight_momentums = np.zeros_like(layer.weights)
               layer.bias_momentums = np.zeros_like(layer.biases)
               # updating momentum with current gradients
          layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
          layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
          # get corrected momentum
          weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iter_num + 1))
          bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iter_num + 1))
          # updating the cache
          layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
          layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases**2
          # get corrected cache
          weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iter_num + 1))
          bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iter_num + 1))
          # Vanilla SGD
          layer.weights += self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
          layer.biases += self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
          
     def post_update(self):
          self.iter_num += 1
"""
# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
The next step is to create the optimizer’s object:
# Create optimizer
optimizer = Optimizer_SGD()
Then perform a forward pass of our sample data:
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)
# Let's print loss value
print('loss:', loss)
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
print('acc:', accuracy)
Next, we do our backward pass, which is also called backpropagation:
# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
Then we finally use our optimizer to update weights and biases:
# Update weights and biases
optimizer.update_params(dense1)
optimizer.update_params(dense2)

""" 
               
"""
def spiral_data(points_per_class, num_classes):
    X = np.zeros((points_per_class * num_classes, 2))  # Data matrix (each row is a point)
    y = np.zeros(points_per_class * num_classes, dtype='uint8')  # Class labels
    for j in range(num_classes):
        ix = range(points_per_class * j, points_per_class * (j + 1))
        r = np.linspace(0.0, 1, points_per_class)  # Radius
        t = np.linspace(j * 4, (j + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2  # Theta
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j
    return X, y
# Create dataset
X, y = spiral_data(samples=100, classes=3)
# Create Dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)
# Create ReLU activation (to be used with Dense layer):
activation1 = Activation_ReLU()
# Create second Dense layer with 64 input features (as we take output
# of previous layer here) and 3 output values (output values)
dense2 = Layer_Dense(64, 3)
# Create Softmax classifier's combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# Create optimizer
optimizer = Optimizer_SGD()
# Train in loop
for epoch in range(10001):
# Perform a forward pass of our training data through this layer
dense1.forward(X)
# Perform a forward pass through activation function
# takes the output of first dense layer here
activation1.forward(dense1.output)
# Perform a forward pass through second Dense layer
# takes outputs of activation function of first layer as inputs
dense2.forward(activation1.output)
# Perform a forward pass through the activation/loss function
# takes the output of second dense layer here and returns loss
loss = loss_activation.forward(dense2.output, y)Chapter 10 - Optimizers - Neural Networks from Scratch in Python
11
# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(loss_activation.output, axis=1)
if len(y.shape) == 2:
y = np.argmax(y, axis=1)
accuracy = np.mean(predictions==y)
if not epoch % 100:
print(f'epoch: {epoch}, ' +
f'acc: {accuracy:.3f}, ' +
f'loss: {loss:.3f}')
# Backward pass
loss_activation.backward(loss_activation.output, y)
dense2.backward(loss_activation.dinputs)
activation1.backward(dense2.dinputs)
dense1.backward(activation1.dinputs)
# Update weights and biases
optimizer.update_params(dense1)
optimizer.update_params(dense2)
"""     