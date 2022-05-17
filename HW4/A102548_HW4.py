"""
2022PR homework4 created by Pei Hsuan Tsai.
    Implement the cross-validation and grid search using only NumPy,
    then train the SVM model from scikit-learn on the provided dataset and test the performance with testing data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score


def LoadData():
    'Load data from files.\nOutput : train and test data'
    x_train = np.load("x_train.npy")
    y_train = np.load("y_train.npy")
    x_test = np.load("x_test.npy")
    y_test = np.load("y_test.npy")
    return x_train, y_train, x_test, y_test


# ## Question 1
# K-fold data partition: Implement the K-fold cross-validation function. Your function should take K as an argument and return a list of lists (len(list) should equal to K), which contains K elements.
# Each element is a list contains two parts, the first part contains the index of all training folds. The second part contains the index of validation fold.
def cross_validation(x_train, y_train, k=5):
    'K-fold cross-validation.\nOutput : a list of lists (lenght equal to K)'
    fold_list = []
    N = x_train.shape[0]
    # The number of data in each validation fold equal to training data divieded by K
    validation_size = N // k
    sample1_num = N % k
    # Handle if the sample size is not divisible by K
    if sample1_num != 0:
        sample1_size = validation_size + 1
    # Shuffle data before partition
    shuffled_data = np.arange(N)
    np.random.shuffle(shuffled_data)
    # K elements
    valid_end = 0
    for i in range(k):
        valid_start = valid_end
        if sample1_num != 0:
            valid_end = sample1_size * (i+1)
            sample1_num -= 1
        else:
            valid_end = validation_size * (i+1)
        # print(f"start : {valid_start}, end : {valid_end}")
        # split list to get train and validation lists
        validation_list = shuffled_data[valid_start : valid_end]
        train_list = np.delete(shuffled_data, validation_list)
        # print("Split: %s, Training index: %s, Validation index: %s" % (i+1, train_list, validation_list))
        fold_list.append([train_list, validation_list]) # add one element list
    return fold_list


# ## Question 2
# Using sklearn.svm.SVC to train a classifier on the provided train set and conduct the grid search of “C”, “kernel” and “gamma” to find the best parameters by cross-validation.
def GridCross(x_train, y_train, k=5):
    'Using grid search and cross-validation to find the best hyperparameters.\nOutput : the best parameters [C, gamma]'
    best_score = 0
    C_range = [0.001, 0.01, 1, 10]
    gamma_range = [0.001, 0.01, 0.1, 1]
    kfold = cross_validation(x_train, y_train, k)
    # grid search start   
    for g in gamma_range:
        for c in C_range:
            # for every possible parameter combination, train the model with cross-validation
            acc = []
            clf = SVC(C=c, kernel='rbf', gamma=g)
            for fold in kfold:
                x = []
                y = []
                test_x = []
                test_label = []
                for i in fold[0]:
                    x.append(x_train[i])
                    y.append(y_train[i])
                for j in fold[1]:
                    test_x.append(x_train[j])
                    test_label.append(y_train[j])
                clf.fit(x, y)   # train a model by fold data
                acc.append(clf.score(test_x, test_label))   # accuracy of the model
            avg_score = np.sum(acc) / k
            # best performing parameters
            if avg_score > best_score:
                best_score = avg_score
                best_parameters = [c, g]
    return best_parameters    


# ## Question 3
# Plot the grid search results of your SVM. The x, y represents the hyperparameters of “gamma” and “C”, respectively. And the color represents the average score of validation folds 
# TODO


# ## Question 4
# Train your SVM model by the best parameters you found from question 2 on the whole training set and evaluate the performance on the test set.
# TODO


if __name__ == "__main__":
    # Load data
    x_train, y_train, x_test, y_test = LoadData()
    
    # # 550 data with 300 features
    # print(x_train.shape)
    # # It's a binary classification problem 
    # print(np.unique(y_train))

    # Question 1
    kfold_data = cross_validation(x_train, y_train, k=10)
    assert len(kfold_data) == 10 # should contain 10 fold of data
    assert len(kfold_data[0]) == 2 # each element should contain train fold and validation fold
    assert kfold_data[0][1].shape[0] == 55 # The number of data in each validation fold should equal to training data divieded by K

    # Question 2
    best_parameters = GridCross(x_train, y_train, k=5)
    print(f"The best parameters: C = {best_parameters[0]}, gamma = {best_parameters[1]}")

    # # Question 3
    # # TODO

    # # Question 4
    # y_pred = best_model.predict(x_test)
    # print("Accuracy score: ", accuracy_score(y_pred, y_test))   