import numpy as np

r"""
Regression is a method to establish a relationship or a model between the independent and
dependent variables, and the model is used to predict the value of the dependent
variable, for given independent variables.

Methods currently implemented:
1. Linear Regression. (Both binomial and multinomial)
2. Locally Weighted Regression.
Tested with the example datasets given in Andrew Ng's Stanford Class.
"""

class LinearRegression(object):
    r"""
    Base Class, for different types of regression.
    Linear Regression is used to fit a linear model, which is called a hypothesis
    function, like theta = theta0 + theta1*x1 + theta2*x2 + ... where x1 and x2
    are the input features.
    """
    def __init__(self, input_data, output_data, alpha=0.01,
        iterations=1500):
        """
        Input should be in the form of a numpy array. Bas
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
        Stochastic gradient algorithm. Does not sum up all the terms while minimising
        the cost function.
        """
        theta_init = np.zeros(self.n_features)
        for i in xrange(self.iterations):
            for j in xrange(self.n_samples):
                test = self.new_input[j]
                theta_init = theta_init - self.alpha*(
                    np.dot(test, theta_init.T) -self.output_data[j])*test
        self.theta = theta_init

    def hypothesis(self):
        r"""
        Returns value predicted by the hypothesis.
        """
        return self.theta

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


class LWR(LinearRegression):
    r"""
    Locally weighted regression is a non-parametric Regression
    model. It is done by fitting a separate model for each point
    in the given data set. Since that may be computationally
    expensive, we are limiting ourselves to predicting a hypothesis
    for a given data point.
    """
    def __init__(self, input_data, output_data, alpha=0.01,
        tau=1, iterations=100):
        super(LWR, self).__init__(input_data, output_data, alpha=alpha,
            iterations=iterations)
        self.tau = tau

    def fit_batch_grad(self):
        raise NotImplementedError("Fitting a hypothesis function to each and "
            "every point would be computationally expensive. Use LWR "
            "to predict the output for a given input.")

    def fit_stochastic_grad(self):
        raise ValueError("Sorry, no stochastic gradient"
            "algorithm for LWR")

    def predict(self, predict_input):
        r"""
        Prediction is made fitting a weighted hypothesis function for the given input.
        The weight given is exp(-1(x - xi)**2/2*sigma**2)
        """
        shape = predict_input.shape
        if len(shape) != 1 or shape[0] != self.n_features - 1:
            raise ValueError("The array for which output needs "
                "to be predicted should have same number "
                "of features as input array")
        predict_input = np.insert((predict_input - self.mean)/self.std, 0, 1)
        theta_init = np.zeros([self.n_features])


        # Helpful in calculating the weights.
        # Points which are near the given point are given a higher weight, and the
        # points which are farther away are given a lesser weight.
        temp = self.new_input - predict_input
        weights = np.exp(-np.sum(temp**2, axis=1)/(2*self.tau**2))
        for i in xrange(self.iterations):
             hypothesis = np.dot(self.new_input, theta_init)
             theta_init = theta_init - self.alpha*(np.dot((weights*(hypothesis - self.output_data)).T, self.new_input))
        return np.dot(theta_init.T, predict_input)
