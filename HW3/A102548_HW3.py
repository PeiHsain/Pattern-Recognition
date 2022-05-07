"""
2022PR homework3 created by Pei Hsuan Tsai.
    Implement the Decision Tree, AdaBoost and Random Forest algorithm by using only NumPy,
    then train the implemented model by the provided dataset and test the performance with testing data.
"""
#!/usr/bin/env python
# coding: utf-8

# ## HW3: Decision Tree, AdaBoost and Random Forest
# In hw3, you need to implement decision tree, adaboost and random forest by using only numpy, then train your implemented model by the provided dataset and test the performance with testing data
# 
# Please note that only **NUMPY** can be used to implement your model, you will get no points by simply calling sklearn.tree.DecisionTreeClassifier

# ## Load data
# The dataset is the Heart Disease Data Set from UCI Machine Learning Repository. It is a binary classifiation dataset, the label is stored in `target` column. **Please note that there exist categorical features which need to be [one-hot encoding] before fit into your model!**
# See follow links for more information
# https://archive.ics.uci.edu/ml/datasets/heart+Disease


from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
df = pd.read_csv(file_url)

train_idx = np.load('train_idx.npy')
test_idx = np.load('test_idx.npy')

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

y_test = test_df['target']

# one-hot encoding
train_num = len(train_df.index)
test_num = len(test_df.index)
feature_num = train_df.shape[1] - 1 # except the target
# for feature 'thal' -> normal, fixed, reversable
train_tmp = train_df[['target']]
test_tmp = test_df[['target']]
train_encoding = pd.get_dummies(train_df, prefix=['thal'])
test_encoding = pd.get_dummies(test_df, prefix=['thal'])
train_encoding = train_encoding.drop('target', axis=1)
test_encoding = test_encoding.drop('target', axis=1)
train_encoding = pd.concat([train_encoding, train_tmp], axis=1)
test_encoding = pd.concat([test_encoding, test_tmp], axis=1)
train_encoding.index = pd.Series([i for i in range(train_num)])
test_encoding.index = pd.Series([i for i in range(test_num)])

# for feature 'cp' -> value 1-4
for n in range(1, 5):
    tmp1 = pd.DataFrame(data=train_encoding, columns=['cp'])
    tmp2 = pd.DataFrame(data=test_encoding, columns=['cp'])
    for i in range(train_num):
        if tmp1.at[i, 'cp'] == n:
            tmp1.at[i, 'cp'] = 1
        else:
            tmp1.at[i, 'cp'] = 0
    for i in range(test_num):
        if tmp2.at[i, 'cp'] == n:
            tmp2.at[i, 'cp'] = 1
        else:
            tmp2.at[i, 'cp'] = 0
    train_encoding.insert(0, f'cp_{n}', tmp1)
    test_encoding.insert(0, f'cp_{n}', tmp2)
train_encoding = train_encoding.drop('cp', axis=1)
test_encoding = test_encoding.drop('cp', axis=1)

# for feature 'restecg' -> value 0-2
for n in range(3):
    tmp1 = pd.DataFrame(data=train_encoding, columns=['restecg'])
    tmp2 = pd.DataFrame(data=test_encoding, columns=['restecg'])
    for i in range(train_num):
        if tmp1.at[i, 'restecg'] == n:
            tmp1.at[i, 'restecg'] = 1
        else:
            tmp1.at[i, 'restecg'] = 0
    for i in range(test_num):
        if tmp2.at[i, 'restecg'] == n:
            tmp2.at[i, 'restecg'] = 1
        else:
            tmp2.at[i, 'restecg'] = 0
    train_encoding.insert(0, f'restecg_{n}', tmp1)
    test_encoding.insert(0, f'restecg_{n}', tmp2)
train_encoding = train_encoding.drop('restecg', axis=1)
test_encoding = test_encoding.drop('restecg', axis=1)

# for feature 'slop' -> value 1-3
for n in range(1, 4):
    tmp1 = pd.DataFrame(data=train_encoding, columns=['slope'])
    tmp2 = pd.DataFrame(data=test_encoding, columns=['slope'])
    for i in range(train_num):
        if tmp1.at[i, 'slope'] == n:
            tmp1.at[i, 'slope'] = 1
        else:
            tmp1.at[i, 'slope'] = 0
    for i in range(test_num):
        if tmp2.at[i, 'slope'] == n:
            tmp2.at[i, 'slope'] = 1
        else:
            tmp2.at[i, 'slope'] = 0
    train_encoding.insert(0, f'slope_{n}', tmp1)
    test_encoding.insert(0, f'slope_{n}', tmp2)
train_encoding = train_encoding.drop('slope', axis=1)
test_encoding = test_encoding.drop('slope', axis=1)

# for feature 'ca' -> value 0-3
for n in range(4):
    tmp1 = pd.DataFrame(data=train_encoding, columns=['ca'])
    tmp2 = pd.DataFrame(data=test_encoding, columns=['ca'])
    for i in range(train_num):
        if tmp1.at[i, 'ca'] == n:
            tmp1.at[i, 'ca'] = 1
        else:
            tmp1.at[i, 'ca'] = 0
    for i in range(test_num):
        if tmp2.at[i, 'ca'] == n:
            tmp2.at[i, 'ca'] = 1
        else:
            tmp2.at[i, 'ca'] = 0    
    train_encoding.insert(0, f'ca_{n}', tmp1)
    test_encoding.insert(0, f'ca_{n}', tmp2)
train_encoding = train_encoding.drop('ca', axis=1)
test_encoding = test_encoding.drop('ca', axis=1)


def gini(sequence):
    'Compute the Gini-index for two class. The smaller, the purer.\nOutput : Gini-index'
    seq_len = len(sequence)
    if seq_len <= 0:
        return 0
    else:
        p1 = np.sum(sequence == 0) / seq_len
        p2 = np.sum(sequence == 1) / seq_len
        # Gini = 1 - sum_all_class_k(Pk^2)
        return 1 - (float(p1)**2) - (float(p2)**2)


def entropy(sequence):
    'Compute the entropy for two class. The smaller, the purer.\nOutput : entropy'
    seq_len = len(sequence)
    if seq_len <= 0:
        return 0
    else:
        p1 = np.sum(sequence == 0) / seq_len
        p2 = np.sum(sequence == 1) / seq_len
        p1 = float(p1)
        p2 = float(p2)

        if (p1 == 0) | (p2 == 0):  # all class are the same in one node
            return 0
        else:   # Entropy = -sum_all_class_k(Pk*log2(Pk))
            return -(p1*np.log2(p1)) - (p2*np.log2(p2))    


def SplitImpurity(mode, dataset):
    'Calculate the impurity for the split.\nOutput : value of impurity'
    if mode == 'gini':
        value = gini(dataset)
    elif mode == 'entropy':
        value = entropy(dataset)
    return value


def Vote(node):
    'Majority vote for the node.\nOutput : the prediction'
    count1 = 0
    count2 = 0
    for i in range(node.DataSet.shape[0]):
        if node.DataSet.at[node.DataSet.index[i], 'target'] == 0:
            count1 += 1
        else:
            count2 += 1
    # Classifier the data to major class
    if count1 >= count2:
        return 0
    else:
        return 1


def Acuracy(y_label, y_pred):
    'Show the accuracy of prediction.'
    print('Test-set accuarcy score: ', accuracy_score(y_label, y_pred))


class TreeNode():
    def __init__(self, dataSet=None, depth=0):
        self.DataSet = dataSet    # input data
        self.Attribute = None # selected attribute
        self.Threshold = 0  # threshold for node
        self.Left = None    # left node
        self.Right = None   # right node
        self.Depth = depth  # depth of node
        self.Leaf = False   # is leaf node or not
        self.Gain = 0   # information gain

    def Update_A_G_T(self, A, G, T):
        self.Attribute = A
        self.Gain = G
        self.Threshold = T

    def Set_Right(self, R):
        self.Right = R

    def Set_Left(self, L):
        self.Left = L

    def Set_Leaf(self, Le):
        self.Leaf = Le


# Question 2
# Implement the Decision Tree algorithm (CART, Classification and Regression Trees) and trained the model by the given arguments, and print the accuracy score on the test data. You should implement two arguments for the Decision Tree algorithm
# 1. **criterion**: The function to measure the quality of a split. Your model should support `gini` for the Gini impurity and `entropy` for the information gain. 
# 2. **max_depth**: The maximum depth of the tree. If `max_depth=None`, then nodes are expanded until all leaves are pure. `max_depth=1` equals to split data once
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.Criterion = criterion  # The function to measure the quality of a split
        self.Max_depth = max_depth  # The maximum depth of the tree
        self.Tree = None
        self.Import = np.zeros(feature_num)
        return None

    def SplitAttribute(self, node, feature):
        'Select the best attribute to split data.\nOutput : best attribute index'
        max_gain = 0
        bestfeature = feature[0]
        best_threshold = 0
        # loop all values of features
        for f in feature:
            # partition the node and calculate the purity of data
            # pick the col of feature, sort by ascending order
            tmp_data = node.DataSet.sort_values(by=[f], ascending=True)
            N = len(tmp_data)
            # split by threshold of average(i-th, (i+1)-th)
            # Continuous
            if f == 'age' or f == 'trestbps' or f == 'chol' or f == 'thalach' or f == 'oldpeak':
                for k in range(N-1):
                    tmp_left_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[0:k+1], ['target']])
                    tmp_right_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[k+1:N], ['target']])
                    after_gain = ((k+1) * tmp_left_split + (N-k-1) * tmp_right_split) / N
                    information_gain = node.Gain - after_gain
                    # find the value of feature can yield lowest value of gini or entropy
                    if information_gain > max_gain:
                        max_gain = information_gain
                        bestfeature = f
                        best_threshold = (tmp_data.at[tmp_data.index[k], bestfeature] + tmp_data.at[tmp_data.index[k+1], bestfeature]) / 2
            else:
                # Discrete
                k = len(tmp_data[tmp_data[f] < 1])
                tmp_left_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[0:k], ['target']]) # data = 0
                tmp_right_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[k:N], ['target']]) # data = 1
                after_gain = (k * tmp_left_split + (N-k) * tmp_right_split) / N
                information_gain = node.Gain - after_gain
                # find the value of feature can yield lowest value of gini or entropy
                if information_gain > max_gain:
                    max_gain = information_gain
                    bestfeature = f
                    best_threshold = 0.5
        return bestfeature, max_gain, best_threshold

    def SplitNode(self, node):
        'Split the data(node) by the attribute.\nOutput : splited nodes'
        R_index = []
        L_index = []
        # Split data into left and right node by threshold
        for i in range(node.DataSet.shape[0]):
            if node.DataSet.at[node.DataSet.index[i], node.Attribute] <= node.Threshold:
                L_index.append(node.DataSet.index[i])
            else:
                R_index.append(node.DataSet.index[i])
        L = pd.DataFrame(node.DataSet.loc[L_index])
        R = pd.DataFrame(node.DataSet.loc[R_index])
        node_L = TreeNode(L, node.Depth+1)
        node_R = TreeNode(R, node.Depth+1)
        # Check is node empty or not
        if (len(L_index) == 0) or (len(L_index) == node.DataSet.shape[0]):
            node_L.Set_Leaf(True)
        if len(R_index) == 0 or (len(R_index) == node.DataSet.shape[0]):
            node_R.Set_Leaf(True)
        return node_R, node_L

    def GenerateTree(self, node):
        'Generate the decision tree by recursive method.'
        # Initial gain
        node.Gain = SplitImpurity(self.Criterion, node.DataSet['target'])
        # Reach the leaf 
        if node.Gain == 0 or node.Leaf == True:
            node.Set_Leaf(True)
            return node
        # Reach the max depth
        if self.Max_depth != None:
            if node.Depth >= self.Max_depth:
                node.Set_Leaf(True)
                return node
        # If feature don't split the original node, delete the feature and split again
        feature = node.DataSet.columns[1:node.DataSet.shape[1]-1]
        for i in range(3):
            # Pick the best attribute and its threshold as the split
            A, G, T = self.SplitAttribute(node, feature)
            node.Update_A_G_T(A, G, T)
            # Split the node into two daughter nodes
            right_node, left_node = self.SplitNode(node)
            if (right_node.DataSet.shape[0] != node.DataSet.shape[0]) and (left_node.DataSet.shape[0] != node.DataSet.shape[0]):
                break
            else:
                # delete the feature from feature set
                feature = np.setdiff1d(feature, [node.Attribute])
        # Recursive untill get leaf
        if left_node.Leaf == False:
            node.Set_Left(left_node)
            self.GenerateTree(node.Left)
        if right_node.Leaf == False:
            node.Set_Right(right_node)
            self.GenerateTree(node.Right)
        return node

    def Create(self, Data=None):
        'Create a initial decision tree.'
        self.Tree = TreeNode(Data)
        self.GenerateTree(self.Tree)

    def Testing(self, test):
        'Put the testing data into decision tree to predicte.'
        y_pred = np.zeros((test_num))
        # Prediction
        for i in range(test_num):
            tmp_node = self.Tree
            x = test.loc[i]
            # go down the decision tree untill reach the leaf
            while tmp_node.Leaf == False:
                if x[tmp_node.Attribute] <= tmp_node.Threshold:
                    if tmp_node.Left == None:
                        break
                    tmp_node = tmp_node.Left    # go to left node
                else:
                    if tmp_node.Right == None:
                        break
                    tmp_node = tmp_node.Right   # go to right node
            # majority vote
            y_pred[i] = Vote(tmp_node)
        return y_pred

    def Importance(self, node):
        'Get the feature importanceget by counting the feature used for splitting data.'
        if node.Leaf == True:
            return
        # count the feature
        if node.Attribute == 'age':
            self.Import[0] += 1
        elif node.Attribute == 'sex':
            self.Import[1] += 1
        elif node.Attribute == 'cp_1' or node.Attribute == 'cp_2' or node.Attribute == 'cp_3' or node.Attribute == 'cp_4':
            self.Import[2] += 1
        elif node.Attribute == 'trestbps':
            self.Import[3] += 1
        elif node.Attribute == 'chol':
            self.Import[4] += 1
        elif node.Attribute == 'fbs':
            self.Import[5] += 1
        elif node.Attribute == 'restecg_0' or node.Attribute == 'restecg_1' or node.Attribute == 'restecg_2':
            self.Import[6] += 1
        elif node.Attribute == 'thalach':
            self.Import[7] += 1
        elif node.Attribute == 'exang':
            self.Import[8] += 1
        elif node.Attribute == 'oldpeak':
            self.Import[9] += 1
        elif node.Attribute == 'slope_1' or node.Attribute == 'slope_2' or node.Attribute == 'slope_3':
            self.Import[10] += 1
        elif node.Attribute == 'ca_0' or node.Attribute == 'ca_1' or node.Attribute == 'ca_2'or node.Attribute == 'ca_3':
            self.Import[11] += 1
        elif node.Attribute == 'thal_normal' or node.Attribute == 'thal_fixed' or node.Attribute == 'thal_reversable':
            self.Import[12] += 1
        # Go to left and right child
        self.Importance(node.Left)
        self.Importance(node.Right)

    def PlotImportance(self):
        'Plot the feature importance.'
        self.Importance(self.Tree)
        feature_tag = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        plt.title(f"Feature Importance")
        plt.xlabel(f"Count")
        plt.ylabel(f"Feature")
        plt.barh(feature_tag, self.Import)
        plt.show()


# Question 4
# implement the AdaBooest algorithm by using the CART you just implemented from question 2 as base learner. You should implement one arguments for the AdaBooest.
# 1. **n_estimators**: The maximum number of estimators at which boosting is terminated
class AdaBoost():
    def __init__(self, n_estimators):
        self.D = 0
        self.h_f = []   # to keep weak learner feature
        self.h_t = []   # to keep weak learner threshold
        self.a = []   # to keep weak learner weight
        self.limitEstimator = n_estimators  # The number of trees in the forest
        return None

    def Impurity(self, data):
        'Compute the Gini-index for two class. The smaller, the purer.\nOutput : Gini-index'
        p1 = 0
        p2 = 0
        seq_len = len(data)
        if seq_len <= 0:
            return 0
        else:
            for i in range(seq_len):
                if float(data.loc[data.index[i]]) == 0:
                    p1 += self.D[data.index[i]]
                else:
                    p2 += self.D[data.index[i]]
            # Gini = 1 - sum_all_class_k(Pk^2)
            return 1 - (p1**2) - (p2**2)

    def CountError(self, label, pred):
        'Compute the error for weak learner.\nOutput : the error'
        error = 0
        for k in range(len(label)):
            if label[k] != pred[k]:
                error += self.D[k]
        return error
            
    def Classifier(self, data, feature, threshold, mode):
        'Predicte by the threshold for value -1 or 1.\nOutput : the prediction'
        pred = np.ones(data.shape[0])
        for k in range(data.shape[0]):
            if mode == 1:   # <=
                if data.at[k, feature] <= threshold:
                    pred[k] = -1
            else:   # >
                if data.at[k, feature] > threshold:
                    pred[k] = -1
        return pred

    def WeakLearner(self, train_data, label):
        'Generate weak learner use CRAT and choose the min error one.\nOutput : weak learner, min error and its prediction'
        error = float('inf')
        N = train_data.shape[0]
        # Build CRAT(1 depth tree) for every feature to compute error
        for i in range(train_data.shape[1]-1):  # remove the 'target'
            # Initial
            feature = train_data.columns[i]
            tmp_data = train_data.sort_values(by=[feature], ascending=True)
            # Continuous
            if feature == 'age' or feature == 'trestbps' or feature == 'chol' or feature == 'thalach' or feature == 'oldpeak':
                init_gain = self.Impurity(train_data['target'])
                max_gain = 0
                for k in range(N-1):
                    tmp_left_split = self.Impurity(tmp_data.loc[tmp_data.index[0:k+1], ['target']])
                    tmp_right_split = self.Impurity(tmp_data.loc[tmp_data.index[k+1:N], ['target']])
                    after_gain = np.sum(self.D[tmp_data.index[0:k+1]]) * tmp_left_split + np.sum(self.D[tmp_data.index[k+1:N]]) * tmp_right_split
                    information_gain = init_gain - after_gain
                    # find the value of feature can yield lowest value of gini or entropy
                    if information_gain > max_gain:
                        max_gain = information_gain
                        best_threshold = (tmp_data.at[tmp_data.index[k], feature] + tmp_data.at[tmp_data.index[k+1], feature]) / 2
            else:
                # Discrete
                threshold = 0.5
            # Prediction and compute the error for the classifier mode of <= or >
            tmp_pred1 = self.Classifier(train_data, feature, threshold, 1)    # -1 or 1 => (<=)
            tmp_e1 = self.CountError(label, tmp_pred1)
            tmp_pred2 = self.Classifier(train_data, feature, threshold, 2)    # -1 or 1 => (>)
            tmp_e2 = self.CountError(label, tmp_pred2)
            if tmp_e1 <= tmp_e2:
                tmp_e = tmp_e1
                tmp_pred = tmp_pred1
            else:
                tmp_e = tmp_e2
                tmp_pred = tmp_pred2
            # find the weak learner has min error
            if tmp_e < error:
                best_feature = feature
                best_threshold = threshold
                error = tmp_e
                prediction = tmp_pred
        return best_feature, best_threshold, error, prediction

    def Training(self, train_data):
        'Use Adaboost to train the model.'
        # Initial
        N = train_data.shape[0]
        self.D = np.full((N), 1/N)
        train_label = train_data['target']  #value = 0 or 1
        y_label = np.ones(N)    # value = -1 or 1
        for i in range(N):
            if train_label.loc[i] == 0:
                y_label[i] = -1
        # Untill meet the limited estimator
        for k in range(self.limitEstimator):
            # find classifier h and compute error e
            feature, threshold, e, y_pred = self.WeakLearner(train_data, y_label)  # value = -1 or 1
            self.h_f.append(feature)
            self.h_t.append(threshold)
            # compute weight classifier a = (1/2)*ln((1-e)/e)
            if e == 0:
                self.a.append(1)
                break
            else:
                self.a.append(np.log((1-e)/e) / 2)
            # update distribution D = D*exp(-a*y*h(x)) / Z
            self.D = (self.D * np.exp((-self.a[k])*y_label*y_pred))
            self.D /= np.sum(self.D)

    def Prediction(self, test_data):
        'Use model to predicte the testing data.\nOutput : the prediction'
        pred = np.zeros((test_num))
        for k in range(test_num):
            for i in range(len(self.h_f)):
                if test_data.at[k, self.h_f[i]] <= self.h_t[i]:
                    h = -1
                else:
                    h = 1
                pred[k] += self.a[i] * h
            if pred[k] < 0: #negative
                pred[k] = 0
            else:   #positive
                pred[k] = 1
        return pred


# Question 5
# implement the Random Forest algorithm by using the CART you just implemented from question 2. You should implement three arguments for the Random Forest.
# 1. **n_estimators**: The number of trees in the forest. 
# 2. **max_features**: The number of random select features to consider when looking for the best split
# 3. **bootstrap**: Whether bootstrap samples are used when building tree
class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        self.Tree_num = n_estimators   # The number of trees in the forest
        self.Feature_num = int(max_features) # The number of features to consider when looking for the best split
        self.Boostrap = boostrap    # Whether bootstrap samples are used when building trees
        self.Criterion = criterion  # The function to measure the quality of a split
        self.Depth = max_depth  # The maximum depth of the tree
        self.Forest = []    # Keep trees to build the forest
        return None

    def Bagging(self, data):
        'Booststrap aggregating. Re-sample from the data.\nOutput : a data set'
        # Randomly choose sample from data to create a new dataset
        dataset = pd.DataFrame()
        N = data.shape[0]
        for i in range(N):
            index = np.random.randint(0, N)
            dataset = dataset.append(data.loc[index])
        dataset.index = pd.Series([i for i in range(N)])
        return dataset

    def SplitAttribute_Forest(self, node, feature):
        'Pick the best attribute and its threshold.\nOutput : a feature, threshold, gain'
        max_gain = 0
        bestfeature = feature[0]
        best_threshold = 0
        # loop all values of features
        for f in feature:
            # partition the node and calculate the purity of data
            # pick the col of feature, sort by ascending order
            tmp_data = node.DataSet.sort_values(by=[f], ascending=True)
            N = len(tmp_data)
            # split by threshold of average(i-th, (i+1)-th)
            # Continuous
            if f == 'age' or f == 'trestbps' or f == 'chol' or f == 'thalach' or f == 'oldpeak':
                for k in range(N-1):
                    tmp_left_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[0:k+1], ['target']])
                    tmp_right_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[k+1:N], ['target']])
                    after_gain = ((k+1) * tmp_left_split + (N-k-1) * tmp_right_split) / N
                    information_gain = node.Gain - after_gain
                    # find the value of feature can yield lowest value of gini or entropy
                    if information_gain > max_gain:
                        max_gain = information_gain
                        bestfeature = f
                        best_threshold = (tmp_data.at[tmp_data.index[k], bestfeature] + tmp_data.at[tmp_data.index[k+1], bestfeature]) / 2
            else:
                # Discrete
                k = len(tmp_data[tmp_data[f] < 1])
                tmp_left_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[0:k], ['target']]) # data = 0
                tmp_right_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[k:N], ['target']]) # data = 1
                after_gain = (k * tmp_left_split + (N-k) * tmp_right_split) / N
                information_gain = node.Gain - after_gain
                # find the value of feature can yield lowest value of gini or entropy
                if information_gain > max_gain:
                    max_gain = information_gain
                    bestfeature = f
                    best_threshold = 0.5
        return bestfeature, max_gain, best_threshold

    def SplitNode_Forest(self, node):
        'Split the data(node) by the attribute.\nOutput : splited nodes'
        R_index = []
        L_index = []
        # Split data into left and right node by threshold
        for i in range(node.DataSet.shape[0]):
            if node.DataSet.at[node.DataSet.index[i], node.Attribute] <= node.Threshold:
                L_index.append(node.DataSet.index[i])
            else:
                R_index.append(node.DataSet.index[i])
        L = pd.DataFrame(node.DataSet.loc[L_index])
        R = pd.DataFrame(node.DataSet.loc[R_index])
        node_L = TreeNode(L, node.Depth+1)
        node_R = TreeNode(R, node.Depth+1)
        # Check is node empty or not
        if (len(L_index) == 0) or (len(L_index) == node.DataSet.shape[0]):
            node_L.Set_Leaf(True)
        if len(R_index) == 0 or (len(R_index) == node.DataSet.shape[0]):
            node_R.Set_Leaf(True)
        return node_R, node_L

    def GrowTree(self, node, feature):
        'Grow a decision tree by recursively.\nOutput : a tree'
        # Initial gain
        node.Gain = SplitImpurity(self.Criterion, node.DataSet['target'])
        # Reach the leaf 
        if node.Gain == 0 or node.Leaf == True:
            node.Set_Leaf(True)
            return node
        # Reach the max depth
        if self.Depth != None:
            if node.Depth >= self.Depth:
                node.Set_Leaf(True)
                return node
        # If feature don't split the original node, delete the feature and split again
        tmp_feature = feature
        for i in range(3):
            # Pick the best attribute and its threshold as the split
            A, G, T = self.SplitAttribute_Forest(node, tmp_feature)
            node.Update_A_G_T(A, G, T)
            # Split the node into two daughter nodes
            right_node, left_node = self.SplitNode_Forest(node)
            if (right_node.DataSet.shape[0] != node.DataSet.shape[0]) and (left_node.DataSet.shape[0] != node.DataSet.shape[0]):
                break
            else:
                # delete the feature from feature set
                feature = np.setdiff1d(feature, [node.Attribute])
        # Recursive untill get leaf
        if left_node.Leaf == False:
            node.Set_Left(left_node)
            self.GrowTree(node.Left, feature)
        if right_node.Leaf == False:
            node.Set_Right(right_node)
            self.GrowTree(node.Right, feature)
        return node
        
    def Create(self, train_data):
        'Creat a random forest by decision trees.'
        for i in range(self.Tree_num):
            # Draw a bootstrap dataset Db of size N from dataset D
            if self.Boostrap == True:
                Db = self.Bagging(train_data)
            else:
                Db = train_data
            # Randomly select m attributes from the M attributes, expect 'target'
            feature_set = np.random.choice(train_data.columns[0:train_data.shape[1]-1], self.Feature_num, replace=False)
            # Grow a decision tree Tb based on Db and random vector
            Tb = TreeNode(Db)
            # The esemble of tree Tb
            self.Forest.append(self.GrowTree(Tb, feature_set))
        
    def Testing(self, test_data):
        'Use forest to predicte the testing data.\nOutput : the prediction'
        # Feed testing data to all the trees, and use the majority vote as the classification result
        pred = np.zeros((test_num))
        for i in range(test_num):
            y_pred = 0
            for tree in self.Forest:
                tmp_node = tree
                x = test_data.loc[i]
                # go down the decision tree untill reach the leaf
                while tmp_node.Leaf == False:
                    if x[tmp_node.Attribute] <= tmp_node.Threshold:
                        if tmp_node.Left == None:
                            break
                        tmp_node = tmp_node.Left    # go to left node
                    else:
                        if tmp_node.Right == None:
                            break
                        tmp_node = tmp_node.Right   # go to right node
                # Prediction
                y_pred += Vote(tmp_node) # value 0 or 1
            # majority vote
            if y_pred / len(self.Forest) <= 0.5:    # average <= 0.5 -> more than half vote for 0
                pred[i] = 0
            else:   # average > 0.5 -> more than half vote for 1
                pred[i] = 1
        return pred


if __name__ == "__main__":
    # Question 1
    # Gini Index or Entropy is often used for measuring the “best” splitting of the data. Please compute the Entropy and Gini Index of provided data.
    # 1 = class 1, 2 = class 2
    data = np.array([1,2,1,1,1,1,2,2,1,1,2])
    for i in range(len(data)):
        if data[i] == 1:
            data[i] = 0
        elif data[i] == 2:
            data[i] = 1

    print("Gini of data is ", gini(data))
    print("Entropy of data is ", entropy(data))

    # Question 2.1
    # Using `criterion=gini`, showing the accuracy score of test data by `max_depth=3` and `max_depth=10`, respectively.
    print("Q 2.1:")
    clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
    print("Max_depth = 3:")
    clf_depth3.Create(train_encoding)
    pred_depth3 = clf_depth3.Testing(test_encoding)
    Acuracy(y_test, pred_depth3)
    print("--------------------------------------")

    clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
    print("Max_depth = 10:")
    clf_depth10.Create(train_encoding)
    pred_depth10 = clf_depth10.Testing(test_encoding)
    Acuracy(y_test, pred_depth10)
    print("--------------------------------------")

    # Question 2.2
    # Using `max_depth=3`, showing the accuracy score of test data by `criterion=gini` and `criterion=entropy`, respectively.
    print("Q 2.2:")
    clf_gini = DecisionTree(criterion='gini', max_depth=3)
    print("Criterion = gini:")
    clf_gini.Create(train_encoding)
    pred_gini = clf_gini.Testing(test_encoding)
    Acuracy(y_test, pred_gini)
    print("--------------------------------------")

    clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
    print("Criterion = entropy:")
    clf_entropy.Create(train_encoding)
    pred_entropy = clf_entropy.Testing(test_encoding)
    Acuracy(y_test, pred_entropy)
    print("--------------------------------------")

    # Question 3
    # Plot the [feature importance] of your Decision Tree model. You can get the feature importance by counting the feature used for splitting data.
    # You can simply plot the **counts of feature used** for building tree without normalize the importance.
    clf_depth10.PlotImportance()

    # Question 4.1
    # Show the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.
    print("Q 4:")
    clf_10estimator = AdaBoost(n_estimators=10)
    print("N_estimator = 10:")
    clf_10estimator.Training(train_encoding)
    pred_10estimator = clf_10estimator.Prediction(test_encoding)
    Acuracy(y_test, pred_10estimator)
    print("--------------------------------------")

    clf_100estimator = AdaBoost(n_estimators=100)
    print("N_estimator = 100:")
    clf_100estimator.Training(train_encoding)
    pred_100estimator = clf_100estimator.Prediction(test_encoding)
    Acuracy(y_test, pred_100estimator)
    print("--------------------------------------")

    # Question 5.1
    # Using `criterion=gini`, `max_depth=None`, `max_features=sqrt(n_features)`, showing the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.
    print("Q 5.1:")
    clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(train_encoding.shape[1]-1))
    print("N_estimator = 10:")
    clf_10tree.Create(train_encoding)
    pred_10tree = clf_10tree.Testing(test_encoding)
    Acuracy(y_test, pred_10tree)
    print("--------------------------------------")

    clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(train_encoding.shape[1]-1))
    print("N_estimator = 100:")
    clf_100tree.Create(train_encoding)
    pred_100tree = clf_100tree.Testing(test_encoding)
    Acuracy(y_test, pred_100tree)
    print("--------------------------------------")

    # Question 5.2
    # Using `criterion=gini`, `max_depth=None`, `n_estimators=10`, showing the accuracy score of test data by `max_features=sqrt(n_features)` and `max_features=n_features`, respectively.
    print("Q 5.2:")
    clf_random_features = RandomForest(n_estimators=10, max_features=np.sqrt(train_encoding.shape[1]-1))
    print("Random features:")
    clf_random_features.Create(train_encoding)
    pred_random_features = clf_random_features.Testing(test_encoding)
    Acuracy(y_test, pred_random_features)
    print("--------------------------------------")

    clf_all_features = RandomForest(n_estimators=10, max_features=train_encoding.shape[1]-1)
    print("All features:")
    clf_all_features.Create(train_encoding)
    pred_all_features = clf_all_features.Testing(test_encoding)
    Acuracy(y_test, pred_all_features)
    print("--------------------------------------")
    
    # Question 6.
    # Try you best to get highest test accuracy score by 
    # - Feature engineering
    # - Hyperparameter tuning
    # - Implement any other ensemble methods, such as gradient boosting. Please note that you cannot call any package. Also, only ensemble method can be used. Neural network method is not allowed to used.
    print("Q 6:")
    clf_best = RandomForest(n_estimators=20, max_features=7, criterion='entropy', max_depth=10)
    print("Random Forest with N_estimator = 20, Max_features = 7, Criterion ='entropy', Max_depth = 10:")
    clf_best.Create(train_encoding)
    pred_best = clf_best.Testing(test_encoding)
    Acuracy(y_test, pred_best)
    print("--------------------------------------")