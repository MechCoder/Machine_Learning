import numpy as np
r"""
Logistic Regression is a supervised machine learning technique, used to fit a linear model
(binary or multinomial) to a given set of data.
Tested with the two example datasets given in Andrew Ng's Stanford Class.
"""

class LinearRegression(object):
    def __init__(self, input_data, output_data, alpha=0.01,
        iterations=1500):
        """
        Input should be in the form of a numpy array.
        """
        in_shape = input_data.shape
        self.output_data = output_data
        self.alpha = alpha
        self.iterations = iterations
        # Just one feature
        if len(in_shape) == 1:
            self.n_samples = in_shape[0]
            self.n_features = 1
            self.new_input = np.ones([self.n_samples, 2])

            # Normalise the given array, by subtracting it with the mean
            # and dividing it by the standard deviation.
            self.mean = np.mean(input_data, axis=0)
            self.std = np.std(input_data, axis=0)
            self.new_input[:, 1] = (input_data - self.mean)/self.std

        else:
            self.n_samples, self.n_features = in_shape
            self.new_input = np.ones([self.n_samples, self.n_features+1])
            self.mean = np.mean(input_data, axis=0)
            self.std = np.std(input_data, axis=0)
            self.new_input[: , 1: ] = (input_data - self.mean)/self.std
        out_shape = output_data.shape
        if len(out_shape) != 1:
            raise ValueError("Output array should be "
                "one dimensional")
        if out_shape[0] != self.n_samples:
            raise ValueError("Number of samples of input "
                "and output array should be same")

        self.n_features += 1  # Dummy feature

    def fit_batch_grad(self):
        r"""
        Batch gradient algorithm (Iterative).
        This is a vectorised implementation of it.
        """
        theta_init = np.zeros(self.n_features)
        for i in xrange(self.iterations):
            hypothesis = np.dot(self.new_input, theta_init)
            theta_init = theta_init - (self.alpha/self.n_samples)*(np.dot((
                hypothesis - self.output_data).T, self.new_input))
        self.theta = theta_init

    def fit_stochastic_grad(self):
        r"""
        Stochastic gradient algorithm.
        """
        theta_init = np.zeros(self.n_features)
        for i in xrange(self.iterations):
            for j in xrange(self.n_samples):
                test = self.new_input[j]
                theta_init = theta_init - self.alpha*(
                    np.dot(test, theta_init.T) -self.output_data[j])*test
        self.theta = theta_init

    def predict(self, predict_input):
        r"""
        Function used to predict the given output.
        """
        shape = predict_input.shape
        if len(shape) != 1 or shape[0] != self.n_features - 1:
            raise ValueError("The array for which output needs "
                "to be predicted should have same number "
                "of features as input array") 
        return np.dot(np.insert((predict_input -
            self.mean)/self.std, 0, 1), self.theta)
