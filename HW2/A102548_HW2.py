"""
2022PR homework2 created by Pei Hsuan Tsai.
    Implement Fisher's linear discriminant by using only NumPy,
    then train your model on the provided dataset, and evaluate the performance on testing data.
"""
#!/usr/bin/env python
# coding: utf-8

# ## HW2: Linear Discriminant Analysis
# In hw2, you need to implement Fisher’s linear discriminant by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
# 
# Please note that only **NUMPY** can be used to implement your model, you will get no points by simply calling sklearn.discriminant_analysis.LinearDiscriminantAnalysis 

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# as the maximun vale -> 10^9
MAX = 1000000000


def Mean(data):
    'Calculate the mean of data set.\nOutput : value of the mean'
    m = [0, 0]
    for i in data:
        m += i
    if len(data) != 0:
        m /= len(data)
    return m


def KNN(test, trainX, trainY, K):
    'Use K nearest-neighbor rule to predicte the class. K = 1.\nOutput : the prediction'
    predict = []
    for x in test:
        # calculate distance between x and traindata, dis = ||x-train||^2
        minX = [MAX]
        minY = [0]
        for i in range(len(trainX)):
            dis = abs(x-trainX[i])
            dis = (dis[0]**2) + (dis[1]**2)
            if dis < max(minX):
                if len(minX) >= K:
                    # remove max value of min
                    index = np.argmax(minX)
                    minX.pop(index)
                    minY.pop(index)
                    # push new min vale
                    minX.append(dis)
                    minY.append(trainY[i])
                else:
                    minX.append(dis)
                    minY.append(trainY[i])
        # sum k minY to predicte the class (because y = 0 or 1, if sum >= K/2, it can be class2)
        tmp = sum(minY)
        if tmp >= (K/2):
            y = 1
        else:
            y = 0
        predict.append(y)
    return predict


if __name__ == "__main__":
    # ## Load data. x=[ , ], y= 0 or 1. train data -> 750, test data -> 250
    x_train = pd.read_csv("x_train.csv").values
    y_train = pd.read_csv("y_train.csv").values[:,0]
    x_test = pd.read_csv("x_test.csv").values
    y_test = pd.read_csv("y_test.csv").values[:,0]

    # ## 1. Compute the mean vectors mi, (i=1,2) of each 2 classes
    # seperate the class of training data in N1, N2
    N1 = []
    N2 = []
    for i in range(x_train.shape[0]):
        # class1
        if y_train[i] == 0:
            N1.append(x_train[i])
        # class2    
        elif y_train[i] == 1:
            N2.append(x_train[i])
    # calculate mean vector m1 and m2 of class1 and class2, mean = sum(x)/N
    m1 = Mean(N1)
    m2 = Mean(N2)
    assert m1.shape == (2,)
    print(f"mean vector of class 1: {m1}", f"\nmean vector of class 2: {m2}")


    # ## 2. Compute the Within-class scatter matrix SW = sum((x1-m1) * (x1-m1)T) + sum((x2-m2) * (x2-m2)T)
    sw1 = np.zeros([2, 2])
    sw2 = np.zeros([2, 2])
    # for class1
    for x in N1:
        mat = x - m1
        mat = mat.reshape((len(mat), 1))
        sw1 += np.matmul(mat, mat.T)
    # for class2
    for x in N2:
        mat = x - m2
        mat = mat.reshape((len(mat), 1))
        sw2 += np.matmul(mat, mat.T)
    sw = sw1 + sw2
    assert sw.shape == (2,2)
    print(f"Within-class scatter matrix SW: {sw}")


    # ## 3.  Compute the Between-class scatter matrix SB = (m2-m1) * (m2-m1)T
    matrix = m2 - m1
    matrix = matrix.reshape((len(matrix), 1))
    sb = np.matmul(matrix, matrix.T)
    assert sb.shape == (2,2)
    print(f"Between-class scatter matrix SB: {sb}")


    # ## 4. Compute the Fisher’s linear discriminant, J(w) = (wT*SB*W) / (wT*SW*W) -> w = SW^-1 * (m2-m1)
    w = np.matmul(np.linalg.inv(sw), matrix)
    assert w.shape == (2,1)
    print(f"Fisher’s linear discriminant: {w}")


    # ## 5. Project the test data by linear discriminant to get the class prediction by nearest-neighbor rule and calculate the accuracy score 
    # you can use accuracy_score function from sklearn.metrics.accuracy_score
    # Project by linear discriminant, new x = wT * x
    # test case
    proj_x_test = np.zeros([len(x_test), 2])
    for i in range(len(x_test)):
        proj_x_test[i] = w.T * x_test[i]
    # train case
    proj_x_train = np.zeros([len(x_train), 2])
    for i in range(len(x_train)):
        proj_x_train[i] = w.T * x_train[i]

    # predicte by nearest-neighbor rule
    y_pred = KNN(proj_x_test, proj_x_train, y_train, 6)
    # calculate the accuracy score
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy of test-set {acc}")


    # ## 6. Plot the 1) best projection line on the training data and show the slope and intercept on the title (you can choose any value of intercept for better visualization)
    # 2) colorize the data with each class
    # 3) project all data points on your projection line. Your result should look like the image
    class1_x = []
    class1_y = []
    class2_x = []
    class2_y = []
    proj1_x = []
    proj1_y = []
    proj2_x = []
    proj2_y = []
    slope = float(w[1] / w[0])  # slope = y/x
    intercept = 6

    # Projection line
    x_line = np.linspace(1, 3, 1000)
    y_line = []
    for i in x_line:
        y_line += [slope * i + intercept]

    # Classification ans project point, projection = w * (x-x0, y-y0).w / |w|^2
    for i in range(x_test.shape[0]):
        proj = w[0] * (x_test[i][0] - x_line[0]) + w[1] * (x_test[i][1] - y_line[0])
        proj /= (w[0]**2 + w[1]**2)
        proj = proj * w
        # class1
        if y_test[i] == 0:
            class1_x.append(x_test[i][0])
            class1_y.append(x_test[i][1])
            # project testing data
            proj1_x.append(proj[0] + x_line[0])
            proj1_y.append(proj[1] + y_line[0])
        # class2    
        elif y_test[i] == 1:
            class2_x.append(x_test[i][0])
            class2_y.append(x_test[i][1])
            # project testing data
            proj2_x.append(proj[0] + x_line[0])
            proj2_y.append(proj[1] + y_line[0])

    # Visualization
    plt.figure(figsize=(6, 6))
    plt.title(f"Projection line: w={slope}, b={intercept}")
    plt.plot(x_line, y_line)
    plt.scatter(class1_x, class1_y, color='blue')
    plt.scatter(class2_x, class2_y, color='red')
    plt.scatter(proj1_x, proj1_y, color='green')
    plt.scatter(proj2_x, proj2_y, color='yellow')
    plt.plot([class1_x, proj1_x], [class1_y, proj1_y], color='gray', lw=0.3)
    plt.plot([class2_x, proj2_x], [class2_y, proj2_y], color='gray', lw=0.3)
    plt.show()