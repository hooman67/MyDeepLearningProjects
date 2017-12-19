import numpy as np


class NeuralNetwork(object):
    #constructor
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate, verbose=False):
        # Set number of nodes in input, hidden and output layers.
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        # Initialize weights
        self.weights_input_to_hidden = np.random.normal(0.0, self.input_nodes**-0.5, 
                                       (self.input_nodes, self.hidden_nodes))

        self.weights_hidden_to_output = np.random.normal(0.0, self.hidden_nodes**-0.5, 
                                       (self.hidden_nodes, self.output_nodes))
        self.lr = learning_rate
        
        #### TODO: Set self.activation_function to your implemented sigmoid function ####
        #
        # Note: in Python, you can define a function with a lambda expression,
        # as shown below.
        self.activation_function = lambda x : 1 / (1 + np.exp(-x))  # Replace 0 with your sigmoid calculation.
        
        ### If the lambda code above is not something you're familiar with,
        # You can uncomment out the following three lines and put your 
        # implementation there instead.
        #
        #def sigmoid(x):
        #    return 0  # Replace 0 with your sigmoid calculation here
        #self.activation_function = sigmoid

        self.verbose = verbose
                    

    def train(self, features, targets):
        ''' Trains the network on batch of features and targets. 
        
            Arguments
            ---------
            
            features: 2D array, each row is one data record, each column is a feature
            targets: 1D array of target values
        
        '''
        n_records = features.shape[0]
        delta_weights_i_h = np.zeros(self.weights_input_to_hidden.shape)
        delta_weights_h_o = np.zeros(self.weights_hidden_to_output.shape)
        for X, y in zip(features, targets):
            
            final_outputs, hidden_outputs = self.forward_pass_train(X)  # Implement the forward pass function below
            # Implement the backproagation function below
            delta_weights_i_h, delta_weights_h_o = self.backpropagation(final_outputs, hidden_outputs, X, y, 
                                                                        delta_weights_i_h, delta_weights_h_o)
        self.update_weights(delta_weights_i_h, delta_weights_h_o, n_records)


    def forward_pass_train(self, X):
        ''' Performs a forward pass to calculate the output of each layer
            given input X.
        '''
        # TODO: Hidden layer - Replace these values with your calculations.
        hidden_inputs = np.dot(X, self.weights_input_to_hidden) # signals into hidden layer
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer

        # TODO: Output layer - Replace these values with your calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) # signals into final output layer
        #hs no activation for the output layer to output regression.
        final_outputs = final_inputs # signals from final output layer.
        
        return final_outputs, hidden_outputs

    def backpropagation(self, final_outputs, hidden_outputs, X, y, delta_weights_i_h, delta_weights_h_o):
        ''' Performs a backwards pass over the network, to calculate the update term for 
            each weight given the ouput, and lastWeightUpdate (i.e last delta_weight)
            Arguments
            ---------
            final_outputs: output from forward pass
            y: target (i.e. label) batch
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
        '''

        # TODO: Output error - Replace this value with your calculations.
        error = y - final_outputs # Output layer error is the difference between desired target and actual output.
        
        ''' hs note: ouput_error_term is the error on the output node which basically, 
            passed through the last activation to contribute to our error.
            output_error_term = err*DerivativeOfActivation(final_outputs)
            But here we are not using an activation on the output node
            node so (activation(x) = x and derivative of activation(x) is 1) so:
        '''
        output_error_term = error * 1

        ''' hs node: This is the error of the hidden layer after the sigmoid 
            btw hidden layer and output layer. So we just multiply the weights
        '''
        #print(output_error_term*self.weights_hidden_to_output)
        #hidden_error = np.dot(output_error_term, self.weights_hidden_to_output)
        #hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)
        if self.verbose:
            print("output_error_term:")
            print(output_error_term)
            print("\n")
            print("weights_hidden_to_output:")
            print(self.weights_hidden_to_output)
            print("\n")
        # works: hidden_error = np.dot(self.weights_hidden_to_output, output_error_term)[:,None]
        # works (better style): hidden_error = np.dot(self.weights_hidden_to_output, output_error_term).reshape(np.dot(self.weights_hidden_to_output, output_error_term).shape[0],1)
        #works (but below is better): hidden_error = output_error_term*self.weights_hidden_to_output
        hidden_error = np.multiply(output_error_term,self.weights_hidden_to_output)
        if self.verbose:
            print("hiddenError:")
            print(hidden_error)
            print("\n")
        
        ''' hs note: This is the error of the hidden layer before it passed through
            the sigmoid so we have to multiply by derivative of sigmoid.
            hidden_error_term = hidden_error * derivativeOfSigmoid(hidden_outputs).
            Below is for sigmoid
        '''
        if self.verbose:
            print("hidden_outputs[:,None]:")
            print(hidden_outputs[:,None])
            print("\n")
        # works: hidden_error_term = hidden_error * hidden_outputs[:,None] * (1 - hidden_outputs[:,None])
        # works: hidden_error_term = hidden_error * hidden_outputs.reshape(hidden_outputs.shape[0],1) * (1 - hidden_outputs.reshape(hidden_outputs.shape[0],1))
        hidden_error_term = np.multiply(hidden_error, np.multiply( hidden_outputs.reshape(hidden_outputs.shape[0],1), (1 - hidden_outputs.reshape(hidden_outputs.shape[0],1)) ))
        if self.verbose:
            print("hidden_error_term:")
            print(hidden_error_term)
            print("\n")

        ''' hs note: This is the change in weights  GD should make == 
            update term (i.e weight step) for the w(s)from last hidden unit, 
            to output node. (small w). For each w the requiredChangeInW. 
            NOTE: 
            the update term for w of each node, is the last update for the w(s) 
            of that node (last delta w) + theErrorTheNodeMade*TheInputToTheNode
        '''
        if self.verbose:
            print("delta_weights_h_o:")
            print(delta_weights_h_o)
            print("\n")
            print("output_error_term * hidden_outputs[:,None]")
            print(output_error_term * hidden_outputs[:,None])
            print("\n")
        #works: delta_weights_h_o += output_error_term * hidden_outputs[:, None]
        delta_weights_h_o += np.multiply(output_error_term, hidden_outputs.reshape(hidden_outputs.shape[0], 1))

        ''' hs note: 
            Weight step (input to hidden).
            NOTE: here we gotta turn x into a column vector because default of numpy
            is row vector, but we always perform all calculations in terms of column
            vectors. 
        '''
        if self.verbose:
            print("delta_weights_i_h:")
            print(delta_weights_i_h)
            print("\n")
            print("X:")
            print(X)
            print("\n")
            print("(hidden_error_term * X).T :")
            print((hidden_error_term * X).T  )
            print("\n")
        #works: delta_weights_i_h += (hidden_error_term * X).T
        delta_weights_i_h += np.multiply(hidden_error_term, X).T 
        
        return delta_weights_i_h, delta_weights_h_o
        


    def update_weights(self, delta_weights_i_h, delta_weights_h_o, n_records):
        ''' Updates weights on gradient descent step
         
            Arguments
            ---------
            delta_weights_i_h: change in weights from input to hidden layers
            delta_weights_h_o: change in weights from hidden to output layers
            n_records: number of records

        '''
        #works: self.weights_input_to_hidden += self.lr * delta_weights_i_h / n_records # update input-to-hidden weights with gradient descent step
        self.weights_input_to_hidden += np.divide( np.multiply(self.lr, delta_weights_i_h), n_records ) # update input-to-hidden weights with gradient descent step
        # works: self.weights_hidden_to_output += self.lr * delta_weights_h_o / n_records # update hidden-to-output weights with gradient descent step
        self.weights_hidden_to_output += np.divide( np.multiply(self.lr, delta_weights_h_o), n_records )# update hidden-to-output weights with gradient descent step

    def run(self, features):
        ''' Run a forward pass through the network with input features
            This is basically the test method. 
        
            Arguments
            ---------
            features: 1D array of feature values (i.e. x_i  == 1 example)
        '''
        
        #### Implement the forward pass here ####
        # TODO: Hidden layer - replace these values with the appropriate calculations.
        hidden_inputs = np.dot(features, self.weights_input_to_hidden)
        hidden_outputs = self.activation_function(hidden_inputs) # signals from hidden layer
        
        # TODO: Output layer - Replace these values with the appropriate calculations.
        final_inputs = np.dot(hidden_outputs, self.weights_hidden_to_output) 
        final_outputs = final_inputs # No activation in the last node
        
        return final_outputs


#########################################################
# Set your hyperparameters here
##########################################################
iterations = 5000
learning_rate = 0.5
hidden_nodes = 12
output_nodes = 1
