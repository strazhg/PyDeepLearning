from numpy import exp, array, random, dot


class DNN():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        random.seed(1)

        # We model a single neuron, with 3 input connections and 1 output connection.
        # We assign random weights to a 3 x 1 matrix, with values in the range -1 to 1
        # and mean 0.
        self.synaptic_weights = []
    
    def add(self, shape : tuple):
        self.synaptic_weights.append(2 * random.random(shape) - 1)
    # The Sigmoid function, which describes an S shaped curve.
    # We pass the weighted sum of the inputs through this function to
    # normalise them between 0 and 1.
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))

    # The derivative of the Sigmoid function.
    # This is the gradient of the Sigmoid curve.
    # It indicates how confident we are about the existing weight.
    def __sigmoid_derivative(self, x):
        return x * (1 - x)

    # We train the neural network through a process of trial and error.
    # Adjusting the synaptic weights each time.
    def train(self, training_set_inputs, number_of_training_iterations):
        for iteration in range(number_of_training_iterations):
            # Pass the training set through our neural network (a single neuron).
            output = self.think(training_set_inputs, 0.01, training=True)

    # The neural network thinks.
    def think(self, inputs, learning_rate : float, training : bool):
        # Pass inputs through our neural network.
        output = self.__sigmoid(dot(inputs, self.synaptic_weights[0]))
        for i in range(1, len(self.synaptic_weights)):
            output = self.__sigmoid(dot(output, self.synaptic_weights[i]))
        
        if training:
            # Calculate the error (The difference between the desired output
            # and the predicted output).
            error = inputs - output

            # Multiply the error by the input and again by the gradient of the Sigmoid curve.
            # This means less confident weights are adjusted more.
            # This means inputs, which are zero, do not cause changes to the weights.
            adjustment = dot(inputs.T, error * self.__sigmoid_derivative(output))

            # Adjust the weights.
            for layer in self.synaptic_weights:
                layer += learning_rate * adjustment

        return output
