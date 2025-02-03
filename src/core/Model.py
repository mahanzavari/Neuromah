import numpy as np

from ..layers import Layer_Input
from tqdm import tqdm
import time
import pickle
import copy

class Model:
    xp = np # numpy is the default array module (CPU)
    
    # def get_array_module():
    #     return Model.xp
    
    def __init__(self , device = None):
        import numpy as np
        Model.xp = np#if Cupy fails or CPU is chosen
        
        if device is None:
            try:
                import cupy as cp
                Model.xp = cp
                self.device = 'gpu'
                print("Using NVIDIA GPU with CuPy (auto-device-selection)")
            except ImportError:
                import numpy as np
                Model.xp = np
                self.device = 'cpu'
                print("CuPy not found, using CPU with NumPy (auto-device-selection)")
        else:
            device = device.lower()
            if device == 'gpu':
                try:
                    import cupy as cp
                    Model.xp = cp
                    self.device = 'gpu'
                    print("Using NVIDIA GPU with Cupy (user-specified)")
                except ImportError:
                    print("Cupy not found, but GPU device was requested.\n \
                          Falling back to CPU with NumPy.")
                    import numpy as np
                    Model.xp = np
                    self.device = 'cpu'
            elif device == 'cpu':
                import numpy as np
                Model.xp = np
                self.device = 'cpu'
                print("Using CPU with NumPy (user-specified)")
            else:
                raise ValueError(f"Invalid device choice: '{device}'. Choose 'cpu' or 'gpu'.")            
        
        self.layers = []
        # Activation_Softmax classifier's output object
        self.softmax_classifier_output = None
        self.loss = None 
        self.accuracy = None
        
        
    # adding layers
    def add(self, layer):
        layer.xp = Model.xp
        # print(layer)
        self.layers.append(layer)

    # Set loss, optimizer and accuracy
    # * specifies that all following arguments must be a keyword arguments
    def set(self, *, loss=None, optimizer=None, accuracy=None):
        if loss is not None:
            self.loss = loss
        else:
            raise ValueError("A loss object needs to be passed to the set contructor")

        if optimizer is not None:
            self.optimizer = optimizer
        else:
            raise ValueError("An optimizer object needs to be passed to the set contructor")
        if accuracy is not None:
            self.accuracy = accuracy
        else:
            raise ValueError("A metrics object needs to be passed to the set contructor")

    def finalize(self , xp = np):

        # instantiate and set the input layer
        self.input_layer = Layer_Input()
        # Count all the layer objects
        layer_count = len(self.layers)
        self.trainable_layers = []
        for layer in self.layers:
            if hasattr(layer , 'weights'):
                self.trainable_layers.append(layer)
                
                
        # Iterate the objects(layer)
        for i in range(layer_count):
            # If it's the first layer, the previous layer is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            # For the last layer - the next object is the loss
            # we save the last layer cause it's output is the model's output
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]


            # If layer contains an attribute called "weights",
            # it's a trainable layer
            # bias check is not needed
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])
                

        # Update loss object with the model reference
        if self.loss is not None:
            # calling loss with self as its arg
            # self.loss = self.loss(self)
            if isinstance(self.loss , type): # is a class?
                self.loss = self.loss(model = self)
            # aleady an instace, leave it 
        if self.accuracy is not None:
            # self.accuracy = self.accuracy(self)
            if isinstance(self.accuracy , type):
                self.accuracy = self.accuracy(model = self)
                
        # If output activation is Activation_Softmax and
        # loss function is Categorical Cross-Entropy
        # use Activation_Softmax_Loss_CategoricalCrossentropy for
        # faster gradient calculation
        from ..losses.Activation_Softmax_Loss_CategoricalCrossentropy import Activation_Softmax_Loss_CategoricalCrossentropy
        from ..activations import Activation_Softmax
        from ..losses import Loss_CategoricalCrossentropy 

        if isinstance(self.layers[-1], Activation_Softmax) and \
           isinstance(self.loss, Loss_CategoricalCrossentropy):
            self.softmax_classifier_output = \
                Activation_Softmax_Loss_CategoricalCrossentropy()
    # model train func
        # model train func
    def train(self, X, y, *, epochs=1, batch_size=None,
              verbose=1, validation_data=None , xp = np):
        # xp = Model.get_array_module()
        
        # Initialize accuracy
        self.accuracy.init(y)

        # Calculate number of steps
        train_steps = 1
        if batch_size is not None:
            train_steps = len(X) // batch_size
            if train_steps * batch_size < len(X):
                train_steps += 1

        # Main training loop
        for epoch in range(1, epochs + 1):
            start_time = time.time()  # Start time for the epoch

            # Reset loss and accuracy accumulators
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Use tqdm for steps within the epoch
            with tqdm(total=train_steps, desc=f'Epoch {epoch}/{epochs}', 
                      bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}', 
                      dynamic_ncols=True, unit='step',
                      colour='green') as pbar_steps:
                for step in range(train_steps):
                    # Get batch
                    if batch_size is None:
                        batch_X = X
                        batch_y = y
                    else:
                        batch_X = X[step * batch_size:(step + 1) * batch_size]
                        batch_y = y[step * batch_size:(step + 1) * batch_size]

                    # Forward pass
                    output = self.forward(batch_X, training=True)

                    # Calculate loss
                    losses = self.loss.calculate(output, batch_y, include_regularization=True)
                    data_loss = losses['data_loss']
                    regularization_loss = losses.get('regularization_loss', 0)
                    loss = data_loss + regularization_loss

                    # Get predictions and accuracy
                    predictions = self.output_layer_activation.predictions(output)
                    accuracy = self.accuracy.calculate(predictions, batch_y)

                    # Backward pass
                    self.backward(output, batch_y)

                    # Optimization
                    self.optimizer.pre_update_params()
                    for layer in self.trainable_layers:
                        self.optimizer.update_params(layer)
                    self.optimizer.post_update_params()

                    # Update step progress bar
                    pbar_steps.update(1)

            # Calculate epoch time
            epoch_time = time.time() - start_time

            # Calculate epoch loss and accuracy
            epoch_losses = self.loss.calculate_accumulated(include_regularization=True)
            epoch_data_loss = epoch_losses["data_loss"]
            epoch_regularization_loss = epoch_losses.get("regularization_loss", 0)
            epoch_loss = epoch_data_loss + epoch_regularization_loss

            output = self.forward(X, training=False)
            predictions = self.output_layer_activation.predictions(output)
            epoch_accuracy = self.accuracy.calculate(predictions, y)

            # Print epoch summary
            print(
                #   f'Epoch {epoch}/{epochs}, '
                  f'time: {epoch_time:.2f}s, '
                  f'acc: {epoch_accuracy:.3f}, '
                  f'loss: {epoch_loss:.3f} ('
                  f'data_loss: {epoch_data_loss:.3f}, '
                  f'reg_loss: {epoch_regularization_loss:.3f}), '
                  f'lr: {self.optimizer.current_learning_rate:.7f}')

            # Validation (if validation data is provided)
            if validation_data is not None:
                self.evaluate(*validation_data, batch_size=batch_size)            

    # Evaluates the model using passed-in dataset
    def evaluate(self, X_val, y_val, *, batch_size=None):
        xp = Model.xp
        
        # Default value when batch is None
        validation_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # by using Floor(//) some data might get unused so an increment is needed
            # to include them
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1
        # Reset accumulated values
        self.loss.new_pass()
        self.accuracy.new_pass()

        # Iterate over steps
        for step in range(validation_steps):

            # If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val

            # Otherwise slice a batch
            else:
                batch_X = X_val[step*batch_size:(step+1)*batch_size]
                batch_y = y_val[step*batch_size:(step+1)*batch_size]

            # Perform forward pass
            output = self.forward(batch_X, training=False)
            # Calculate loss
            self.loss.calculate(output, batch_y)
            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Get and print validation loss and accuracy
        # validation_loss = self.loss.calculate_accumulated()
        validation_losses = self.loss.calculate_accumulated()
        validation_data_loss = validation_losses['data_loss']
        validation_accuracy = self.accuracy.calculate_accumulated()

        # Print a summary
        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_data_loss:.3f}')

    # Predicts on the samples
    def predict(self, X, *, batch_size=None):
        xp = Model.xp
        
        # Default value batch is None
        prediction_steps = 1
        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            # by using Floor(//) some data might get unused so an increment is needed
            # to include them
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1
        # Model outputs
        output = []
        # Iterate over steps
        for step in range(prediction_steps):

            # If batch size is not set - train using one step and full dataset
            if batch_size is None:
                batch_X = X
            # Otherwise slice a batch
            else:
                batch_X = X[step*batch_size:(step+1)*batch_size]
            # Perform forward pass
            batch_output = self.forward(batch_X, training=False)
            # Append batch prediction to the output list
            output.append(batch_output)
        # Stack and return results
        return xp.vstack(output)

    # Performs forward pass
    def forward(self, X, training):

        
        # Call forward method on the input layer
        # this will set the output property that the first layer in "prev" object is expecting
        self.input_layer.forward(X, training)
        # Call forward method of every object in a chain
        # Pass output of the previous object as a parameter
        for layer in self.layers:
            layer.forward(layer.prev.output , training)

        # layer is not the last layer, return it's output
        # since it's output is the model's output
        return layer.output

    # performs backward pass
    def backward(self, output, y):
        xp = Model.xp
        # If softmax classifier
        if self.softmax_classifier_output is not None:
            # call backward method on the combined activation/loss
            # this will set dinputs property
            self.softmax_classifier_output.backward(output, y)
            # Since the backward method of the last layer won't be called
            # which is Activation_Softmax activation
            # as we used combined activation/loss object
            # let's set dinputs in this object
            self.layers[-1].dinputs = \
                self.softmax_classifier_output.dinputs

            # Call backward method going through
            # all the objects but one last
            # in reversed order passing dinputs as a parameter
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        # this will set dinputs property that the last
        # layer will try to access shortly
        self.loss.backward(output, y)
        
        # Call backward method going through all the objects
        # in reversed order passing dinputs as a parameter
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    # Retrieves and returns parameters of trainable layers
    def get_parameters(self):
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_parameters())
        return parameters
    
    # Updates the model with new parameters
    def set_parameters(self, parameters):

        # Iterate over the parameters and layers
        # and update each layers with each set of the parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)
    # Saves the parameters to a file
    def save_parameters(self, path):
        # write-binary
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    # Loads the weights and updates a model instance with them
    def load_parameters(self, path):
        # read-binary
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))
    # Saves the model
    def save(self, path):
        # deep copy of current model instance
        model = copy.deepcopy(self)
        # Reset accumulated values
        model.loss.new_pass()
        model.accuracy.new_pass()
        # Remove data from the input layer
        # and gradients from the loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)
        # For each layer remove inputs, output and dinputs properties
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs',
                             'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)
        with open(path, 'wb') as f:
            pickle.dump(model, f)

    # Loads and returns a model
    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            model = pickle.load(f)
        return model
