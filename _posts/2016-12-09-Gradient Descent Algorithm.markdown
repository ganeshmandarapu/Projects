---
layout: post
title: "Gradient Descent Algorithm"
img: new-york.jpg # Add image post (optional)
date: 2016-12-09 12:55:00 +0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)

---

In my previous blog, I have presented syntactical differences and similar functions between R & Python. Now, I want to take it to next level and write some machine learning algorithms using both R and Python. Here, one may use direct functions from the packages available. However, here I am presenting the way to write your own functions for algorithms.  In this series, I am starting with Gradient decent algorithm.  I will briefly explain what is gradient decent, then, simulate data for simple linear regression. We apply gradient decent algorithm for a linear regression to identify parameters. I will show how you can write your own functions for simple linear regression using gradient decent in both R and Python. I will show the results of both R and Python codes. 

#Gradient descent algorithm

Gradient descent is one of the popular optimization algorithms. We use it to minimize the first order derivative of the cost function.  You can define a cost function using an equation (example: f(x) = 3x2 + 2x+3) , then do the first order derivative for it ( in this case, f `(x)= 6x+2). Then you can randomly initialize value of x and, iterate the value with some learning rate (slope), until we reach convergence (we reach local minimum point).  For more details, you can refer simple computational example, with python code, given in wikepedia .  

I am taking the same concept of Gradient Decent and applying it for simple linear regression. In the above example, we have the equation to represent the line and we are finding the minimum value. However, in the case of linear regression, we have data points, and we come up with the linear equation, that fits these data points. Here, it works by efficiently searching the parameter space to identify optimal parameters i.e, intercept(theta0) and slope(theta1). We initialize the algorithm with some theta values and iterate the algorithm to update theta values simultaneously until we reach convergence. These final theta values (parameters or coefficients), when we put it in equation format, represent best regression line. 

In machine learning, the regression function y = theta0 + theta1x is referred to as a 'hypothesis' function. The idea, just like in Least squares method, here is to minimize the sum of squared errors, represented as a 'cost function'. We minimize cost function to achieve the best regression line.

Let us start with simulating data for simple linear regression.  


##R

Importing the dummy data we created.

```{r cars}

setwd("S:\\ANALYTICS\\R_Programming\\Gradient_Descent")

N      <- 100
x      <- rnorm(n = 100, m=48.9, sd=9.7)
beta   <- 0.4
errors <- rweibull(N, shape=3, scale=10)
errors <- errors - factorial(1/1.5)  ----- this centers the error distribution on 0
y      <- 1 + x*beta + errors

data <- data.frame(x,y)

colnames(data) <- c("x", "y")

write.csv(data, file = "Data.csv", row.names = FALSE)

```

Below we are considering the above simulated data for the python coding purpose.

```{r}

colnames(data) <- NULL

write.csv(data, file = "Data_Python.csv", row.names = FALSE)

```

Looking at the dimension of data this is because we are working on matrix method.

Now we have to change the dimension of x(predictor) and y(response) accordigly to make matrix multiplication possible. 

create the x- matrix of explanatory variables. Here we are producing a matrix with 1's in 100 rows to make the matrix dimension is such a way that the multiplicatioin is possible.

```{r, message=FALSE, warning=FALSE}

Data <- read.csv("Data.csv", header = TRUE)

head(Data)

dim(Data)
head(Data)
attach(Data)

a0 = rep(1,100)
a1 = Data[,1]
b = Data[,2]
x <- as.matrix(cbind(a0,a1))
y <- as.matrix(b)


head(x)
head(y)
```

#Derivative Gradient: 

Derivative gradient is the next step after simulating the data in which we obtain the initial parameters. For initializing the parameter of hypothesis function we considered the first order derivative of the cost function in the below function.  

where m is the total number of training examples and h(x(i)) is the hypothesis function defined like this:
For determining the initial gradient values we have to consider the first derivative of the cost function which shown below. Cost function

```{r}
Initial_Gradient <- function(x, y, theta) {
  m <- nrow(y)
  hypothesis = x %*% t(theta)
  loss =  hypothesis - y
  x_transpose = t(x)
  gradient <- (1/m)* (t(x) %*% (loss))
  return(t(gradient))
}

```

Here 'i' represents each observation from the data.

For the derivative gradient we are determining the first derivative of the cost function which gives initial coefficients to update through the gradient runner below. 
This derivative gradient is called by the gradient runner inside the for loop.

From the above function results we will get the matrix of a0 and a1. Now using the gradient runner we have to iterate through the theta0 and theta1 untill they converge. 

#gradient runner:

Gradient runner is the iterative process to update the theta values through the below function untill it converge to its local minimum of the cost function.

Here in this function additional element we are using is learning rate(alpha) which help to converge to local minimum. For our requirement we considered alpha as 0.0001.

Gradient descent is used to minimize the cost function to get the converged theta value matrix which fits the regression line in data. 

```{r}
gradient_descent <- function(x, y, Numiterrations){
  # Initialize the coefficients as matrix

  theta <- matrix(c(0, 0), nrow=1) 
  #learning rate
  alpha = 0.0001 
  for (i in 1:Numiterrations) {
    theta <- theta - alpha  * Initial_Gradient(x, y, theta)
  }
  return(theta)
}

```


```{r}

print(gradient_descent(x, y, 1000))

```

To speed up the convergence change learning rate or number of iteratioins. If you consider learning rate as too small it will take more time to converge and if it is high also it will never converge. 

#Python

Import all the required packages and the dummy data to run the algorithm. For python also we are doing it in matrix method.

Same steps to be followed in python as we did using R. 

```{r engine='python'}

from numpy import *
import numpy as np
import random
from sklearn.datasets.samples_generator import make_regression 
import pylab
from scipy import stats
import pandas as pd
import os

path = "S:\\ANALYTICS\\R_Programming\\Gradient_Descent"
os.chdir(path)

points = genfromtxt("Data_Python.csv", delimiter=",", )

a= points[:,0]
b= points[:,1]

x= np.reshape(a,(len(a),1))
x = np.c_[ np.ones(len(a)), x] # insert column
y = b


###For determining the initial gradient values we have to consider the first derivative of the cost function which shown below. Cost function

def Initial_Gradient(x, y, theta):
    m = x.shape[0]
    hypothesis = np.dot(x, theta)
    loss = hypothesis - y
    x_transpose = x.transpose()
    gradient = (1/m) * np.dot(x_transpose, loss) 
    return(gradient) 

###Below gradient descent function is to update the theta0 and theta1 values(coefficients of hypothesis equation) with the initial values untill they converge. We are calling the initial theta value matrix inside the for loop.      
    
def gradient_descent(alpha, x, y, numIterations):
    m = x.shape[0] # number of samples
    theta = np.zeros(2)
    x_transpose = x.transpose()
    for iter in range(0, numIterations):
        hypothesis = np.dot(x, theta)
        loss = hypothesis - y
        J = np.sum(loss ** 2) / (2 * m)  # cost
        theta = theta - alpha * Initial_Gradient(x, y, theta)  # update
    return theta

###Now to verify the algorithm in R and Python we are providing learning rate and number of iteration to gradient descent to perform.          
    
print(gradient_descent(0.0001,x,y,1000))
    
```

If you observe from the both algorithms we can see they given the same theta values(coefficients) for hypothesis. 
