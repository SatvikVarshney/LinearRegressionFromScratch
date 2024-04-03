#!/usr/bin/env python
# coding: utf-8

# ### Created by: Satvik Varshney

# Required Libraries
import numpy as np
from numpy import loadtxt
from numpy.linalg import inv
import json
import sys

class DataProcessor:
    def __init__(self, inputFile: str, configFile: str):
        # Define the output file name based on input file
        self.outputFile = "data/" + inputFile.split('/')[-1].replace('in', 'out')
        # Load the dataset and split into features (X) and target variable (y)
        self.data = loadtxt(inputFile, delimiter=" ")
        self.dimensions = self.data.shape
        self.features = self.data[:, :-1]
        self.target = self.data[:, -1]

        # Load configuration from JSON file for hyperparameters
        with open(configFile, 'r') as file:
            config = json.load(file)
        self.learningRate = config['learning rate']
        self.iterations = config['num iter']

class LinearModel:
    # Linear Regression using Gradient Descent
    def computeGradientDescent(self, X, y, lr, iterations):
        samples = X.shape[0]
        X_bias = np.hstack((np.ones((samples, 1)), X))
        weights = np.zeros(X_bias.shape[1])
        for i in range(iterations):
            predictions = X_bias.dot(weights)
            errors = y - predictions
            for j in range(len(weights)):
                weights[j] += (lr / samples) * np.dot(errors, X_bias[:, j])
        return weights
    
    # Linear Regression using Analytic Solution
    def computeAnalyticalSolution(self, X, y):
        samples = len(X)
        X_bias = np.hstack((np.ones((samples, 1)), X))
        weights = inv(X_bias.T.dot(X_bias)).dot(X_bias.T).dot(y)
        return weights

if __name__ == "__main__":
    if len(sys.argv) == 3:
        inputDataFile = sys.argv[1]
        configJsonFile = sys.argv[2]
    else:
        # Default filenames if not provided
        inputDataFile = "data/1.in"
        configJsonFile = "data/1.json"

    processor = DataProcessor(inputDataFile, configJsonFile)
    model = LinearModel()

    # Calculate weights using both methods
    weightsGD = model.computeGradientDescent(processor.features, processor.target, processor.learningRate, processor.iterations)
    weightsAnalytic = model.computeAnalyticalSolution(processor.features, processor.target)

    # Format the weights to four decimal places
    weightsGDFormatted = ['%.4f' % elem for elem in weightsGD]
    weightsAnalyticFormatted = ['%.4f' % elem for elem in weightsAnalytic]

    # Output results to file
    with open(processor.outputFile, 'w') as outputFile:
        outputFile.write("-------" + processor.outputFile.split('/')[-1] + "-------\n")
        np.savetxt(outputFile, weightsAnalyticFormatted, fmt="%s", newline="\n")
        outputFile.write("\n")
        np.savetxt(outputFile, weightsGDFormatted, fmt="%s", newline="\n")
        outputFile.write("--------EOF--------")
