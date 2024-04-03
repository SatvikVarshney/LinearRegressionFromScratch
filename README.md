# Simple Linear Regression Implementation

## Overview
This project is an investigation into the fundamental ideas that underpin machine learning, with a particular emphasis on the use of a straightforward linear regression model. 
The analytic solution and gradient descent are the two primary approaches that are incorporated into the project in order to locate the line that provides the greatest fit for a particular dataset.
The primary goal is to compare the effectiveness of different strategies on synthetic datasets and to gain an understanding of how these methods operate.

## Features
- **Analytic Solution**: Calculates the optimal weights for the regression model analytically using the normal equation.
- **Gradient Descent**: Iteratively adjusts the weights to minimize the cost function, showcasing the practical application of this widely used optimization technique.

## Getting Started

### Prerequisites
- Python 3.x
- numpy

### Installation
Clone this repository to your local machine:
```bash
git clone https://github.com/<your-username>/SimpleLinearRegression.git

## Data Files

This project uses synthetic datasets provided in `.in` files for training the linear regression models. Each `.in` file contains multiple lines, each representing a data point.
A line consists of space-separated real numbers, where the last number is the target variable (y) and the preceding numbers are the feature variables (x1, x2, ..., xM).

### Example:
14 20 69
16 3 -1
24 30 99
11 62 240
30 -4 -43

In this example, each line represents a data point with two features and one target variable.

### Configuration Files

Hyperparameters for the gradient descent method are specified in `.json` files. Each `.json` file corresponds to an `.in` file and contains the learning rate and the number of iterations.

#### 1.json Example:
```json
{
	"learning rate": 0.0001,
	"num iter": 1000
}

#### 2.json Example:
{
	"learning rate": 0.01,
	"num iter": 1000
}

