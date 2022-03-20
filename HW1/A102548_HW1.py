"""
2022PR homework1 created by Pei Hsuan Tsai.
    Implement linear regression by using only NumPy, then train your implemented model using Gradient Descent
    by the provided dataset and test the performance with testing data.
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


def model(data, a, b):
    'Model y=ax+b for data.\nOutput : predicted y'
    return data*a+b


def MSE(data, answer, size):
    'Loss function by Mean Square Error.\nOutput : error'
    error = 0
    for i in range(size):
        difference = data[i] - answer[i]
        error += difference ** 2
    error /= size
    return error


def Gradient(data, answer, a, b):
    'Gradient of loss function.'
    # Loss function w.r.t a
    new_a = 0.
    new_b = 0.
    for i in range(DataSize):
        new_a += 2 * data[i] * (model(data[i], a, b)-answer[i])
        new_b += 2 * (model(data[i], a, b)-answer[i])
    new_a /= DataSize
    new_b /= DataSize
    return new_a, new_b


if __name__ == '__main__':
    # Load data
    train_df = pd.read_csv("./train_data.csv")
    x_train, y_train = train_df['x_train'], train_df['y_train']

    # Train model
    # 1. Random initialize the weights, intercepts of the linear model
    DataSize = y_train.size
    meanY = sum(y_train) / DataSize
    LearnRate = 0.1
    iteration = 100
    converge_limit = 15
    converge_range = 0.00001
    loss_log = []
    count = 0
    # Assume normal diatribution, N(0, variance)
    Var = sum(x_train**2) / DataSize
    A = np.random.normal(0, Var)
    B = np.random.normal(0, Var)

    for i in range(iteration):
        # 2. Feed foward the training data into the model, get the output prediction
        y_pred = model(x_train, A, B)
        # 3. Calculating training loss by Mean Square Error of predcition and ground truth data
        loss = MSE(y_pred, y_train, DataSize)
        if i > 0:
            diff = abs(loss-loss_log[-1])
            if diff < converge_range:
                count += 1
        loss_log.append(loss)
        # 4. Calculating the gradients
        grad_a, grad_b = Gradient(x_train, y_train, A, B)
        # 5. Updating the weights and intercepts by the gradients * learning rate
        A = A - grad_a * LearnRate
        B = B - grad_b * LearnRate
        # End of training. When loss is same as past for converge_limit time, stop the training
        if count > converge_limit:
            break
    
    # Test the performance on the testing data
    # Inference the test data (x_test) by your model and calculate the MSE of (y_test, y_pred)
    test_data = pd.read_csv("test_data.csv")
    x_test, y_test = test_data['x_test'], test_data['y_test']
    y_pred = model(x_test, A, B)
    error = MSE(y_pred, y_test, y_test.size)

    # Print information
    print("Mean Square Error of prediction and ground truth :", error)
    print("The weights of the linear model :", A)
    print("The intercepts of the linear model :", B)

    # Plot the learning curve of the training
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(loss_log)
    plt.show()    