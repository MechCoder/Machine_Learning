# TODO: There definitely seems to be wrong in the implementation. If you do know
# please let me know.

r"""
Implementation a few generative algorithms as given in Andrew Ng's
Stanford class.
"""

import os
import math
import numpy as np
import sys


class GaussianDiscriminant(object):

    def train(self, inputdata, outputdata):
        r"""
        Input data and output data should be in the form of numpy
        arrays and preprocessing.
        Three parameters have to be estimated.
        1] P(x | y = 0) which follows a multivariate normal distribution with mean u0 and
            covariance sigma
        2] P(x | y = 1) which follows a multivariate normal distribution with mean u1 and
            covariance sigma
        3] Covariance matrix of the distribution.
        4] P(y) which is assumed to follow a Bernoulli distribution with p.
        """
        self.inputdata = inputdata
        self.outputdata = outputdata
        self.examples, self.features = self.inputdata.shape

        # Normalising by subtracting with mean and divinding by standard deviation.
        self.mean = np.mean(self.inputdata, axis=0)
        self.std = np.std(self.inputdata, axis=0)
        self.normalized = (self.inputdata - self.mean)/self.std

        # Checking if the output data is binary or not.
        # This can be done using a combination of np.unique and np.where but it should
        # break as soon as it is found that y is multilabel.
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

        # Modelling y
        self.label1, self.label2 = sorted(self.classes.keys())
        self.p = len(self.classes[self.label1])/float(self.examples)

        # Modelling mean0 and mean1
        x_label1 = self.normalized[self.classes[self.label1]]
        x_label2 = self.normalized[self.classes[self.label2]]
        self.mean1 = np.mean(x_label1, axis=0)
        self.mean2 = np.mean(x_label2, axis=0)

        # Modelling covariance
        var1 = np.dot((x_label1 - self.mean1).T, (x_label1 - self.mean1))
        var2 = np.dot((x_label2 - self.mean2).T, (x_label2 - self.mean2))
        self.covariance = np.matrix((var1 + var2)/self.examples)
        print self.covariance

    def get_params(self):
        if not hasattr(self, "covariance"):
            raise ValueError("GaussianDiscriminant has to be trained")

        return self.p, self.mean1, self.mean2, self.covariance

    def predict(self, x):
        r"""
        Outputs the label or array of labels for the input that needs to
        be predicted.
        """
        if not hasattr(self, "covariance"):
            raise ValueError("GaussianDiscriminant has to be trained")
        x = np.atleast_2d(x)
        examples, features = x.shape

        p = self.p
        covariance = self.covariance
        mean1 = self.mean1
        mean2 = self.mean2
        labels = []
        cov_inverse = covariance.I

        for ind, exam in enumerate(x):
            temp1 = exam[ind] - mean1
            temp2 = exam[ind] - mean2
            model1 = math.exp(np.dot(np.dot(temp1.T, cov_inverse), temp1)*-0.5)*p
            model2 = math.exp(np.dot(np.dot(temp2.T, cov_inverse), temp2)*-0.5)*(1-p)
            if model1 > model2:
                labels.append(int(float(self.label1)))
            else:
                labels.append(int(float(self.label2)))

        return labels
