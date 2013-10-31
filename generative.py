r"""
Implementation a few generative algorithms as given in Andrew Ng's
Stanford class.
"""

import os
import math
import numpy as np
import sys


class GaussianDiscriminant(object):
    def __init__(self, inputdata, outputdata):
        r"""
        Input data and output data should be in the form of numpy
        arrays and preprocessing.
        """
        self.inputdata = inputdata
        self.outputdata = outputdata
        self.input, self.features = self.inputdata.shape

        # Checking if the output data is binary or not.
        self.classes = {}
        for index, label in enumerate(self.outputdata):
            if str(label) not in self.classes.keys():
                if len(self.classes) == 2:
                    raise ValueError("GDA is a binary classification problem "
                        "enter valid output data.")
                else:
                    self.classes[str(label)] = [index]
            else:
                self.classes[str(label)].append(index)

        # Normalising by subtracting with mean and divinding by standard deviation.
        self.mean = np.mean(self.inputdata, axis=0)
        self.std = np.std(self.inputdata, axis=0)
        self.inputdata = (self.inputdata - self.mean)/self.std

    def fit(self):
        r"""
        Three parameters have to be estimated.
        1] P(x | y = 0) which follows a multivariate normal distribution with mean u0 and 
            covariance sigma
        2] P(x | y = 1) which follows a multivariate normal distribution with mean u1 and 
            covariance sigma
        """
        classes = self.classes.keys()
        mean_param = []
        input_values = []   # For storing the input values.

        # Modelling the mean for predicting x given y = first label and y = second label.
        for index_, class_ in enumerate(classes):
            input_values.append(self.inputdata[self.classes[class_]]) 
            mean_param.append(np.mean(input_values[index_], axis=0))
        self.mean_param0 = mean_param[0]
        self.mean_param1 = mean_param[1]

        # Modelling the covariance matrix.
        covariance_matrix = np.zeros([self.features, self.features])
        for index_, class_ in enumerate(input_values):
            variance = input_values[index_] - mean_param[index_]
            covariance_matrix += np.dot(variance.T, variance)

        self.covariance_matrix = np.matrix(covariance_matrix/self.input)

    def predict(self, predict_value):
        r"""
        Find which has a higher probability, for both models and output that particular label.
        """
        predict_value = (predict_value - self.mean)/self.std
        cov_inverse = self.covariance_matrix.I
        mag = np.abs(np.linalg.det(cov_inverse))**0.5
        common_coeff = 1/(mag*((2*math.pi)**(0.5*self.features)))

        temp1 = predict_value - self.mean_param0
        temp2 = predict_value - self.mean_param1
        model1 = common_coeff*math.exp(-0.5*np.dot(np.dot(temp1.T, cov_inverse), temp1))
        model2 = common_coeff*math.exp(-0.5*np.dot(np.dot(temp2.T, cov_inverse), temp2))
        if model1 > model2:
            return float(self.classes.keys()[0])
        return float(self.classes.keys()[1])
