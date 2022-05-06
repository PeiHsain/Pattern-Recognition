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

# In[2]:


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

# In[3]:


train_df.head()
print(train_encoding)
print(test_encoding)
test_df.head()


# ## Question 1
# Gini Index or Entropy is often used for measuring the “best” splitting of the data. Please compute the Entropy and Gini Index of provided data. Please use the formula from [page 5 of hw3 slides]

# In[5]:


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


def Acuracy(y_label, y_pred):
    'Show the accuracy of prediction.'
    print('Test-set accuarcy score: ', accuracy_score(y_label, y_pred))


# In[14]:


# 1 = class 1,
# 2 = class 2
data = np.array([1,2,1,1,1,1,2,2,1,1,2])
for i in range(len(data)):
    if data[i] == 1:
        data[i] = 0
    elif data[i] == 2:
        data[i] = 1

# In[15]:


print("Gini of data is ", gini(data))


# In[16]:


print("Entropy of data is ", entropy(data))


# ## Question 2
# Implement the Decision Tree algorithm (CART, Classification and Regression Trees) and trained the model by the given arguments, and print the accuracy score on the test data. You should implement two arguments for the Decision Tree algorithm
# 1. **criterion**: The function to measure the quality of a split. Your model should support `gini` for the Gini impurity and `entropy` for the information gain. 
# 2. **max_depth**: The maximum depth of the tree. If `max_depth=None`, then nodes are expanded until all leaves are pure. `max_depth=1` equals to split data once
# 

# In[7]:


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


class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None):
        self.Criterion = criterion  # The function to measure the quality of a split
        self.Max_depth = max_depth  # The maximum depth of the tree
        self.Tree = None
        self.Import = np.zeros(feature_num)
        return None

    def SplitAttribute(self, node):
        'Select the best attribute to split data.\nOutput : best attribute index'
        max_gain = 0
        bestfeature = None
        best_threshold = 0
        # loop all values of all features
        for i in range(int(node.DataSet.shape[1])-1): # remove the 'target' colcum
            # partition the node and calculate the purity of data
            # pick the col of feature[i], sort by ascending order
            tmp_data = node.DataSet.sort_values(by=[node.DataSet.columns[i]], ascending=True)
            N = len(tmp_data)
            # split by threshold of average(i-th, (i+1)-th)
            # Continuous
            if node.DataSet.columns[i] == 'age' or node.DataSet.columns[i] == 'trestbps' or node.DataSet.columns[i] == 'chol' or node.DataSet.columns[i] == 'thalach' or node.DataSet.columns[i] == 'oldpeak':
                for k in range(N+1):
                    tmp_left_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[0:k], ['target']])
                    tmp_right_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[k:N], ['target']])
                    after_gain = (k * tmp_left_split + (N-k) * tmp_right_split) / N
                    information_gain = node.Gain - after_gain
                    # find the value of feature can yield lowest value of gini or entropy
                    if information_gain > max_gain:
                        max_gain = information_gain
                        bestfeature = node.DataSet.columns[i]
                        if k == 0:  # all data split in right
                            best_threshold = tmp_data.at[tmp_data.index[0], bestfeature]
                        elif k >= N:  # all data split in left
                            best_threshold = tmp_data.at[tmp_data.index[N-1], bestfeature]
                        else:
                            best_threshold = (tmp_data.at[tmp_data.index[k-1], bestfeature] + tmp_data.at[tmp_data.index[k], bestfeature]) / 2
            else:
                # Discrete
                k = len(tmp_data[tmp_data[node.DataSet.columns[i]] < 1])
                tmp_left_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[0:k], ['target']]) # data = 0
                tmp_right_split = SplitImpurity(self.Criterion, tmp_data.loc[tmp_data.index[k:N], ['target']]) # data = 1
                after_gain = (k * tmp_left_split + (N-k) * tmp_right_split) / N
                information_gain = node.Gain - after_gain
                # find the value of feature can yield lowest value of gini or entropy
                if information_gain > max_gain:
                    max_gain = information_gain
                    bestfeature = node.DataSet.columns[i]
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
        if len(L_index) == 0:
            node_L.Set_Leaf(True)
        if len(R_index) == 0:
            node_R.Set_Leaf(True)
        return node_R, node_L

    def GenerateTree(self, node):
        'Generate the decision tree by recursive method.'
        # Initial gain
        node.Gain = SplitImpurity(self.Criterion, node.DataSet['target'])
        # Stopping criteria
        # The data in each leaf-node belongs to the same class
        # Depth of the tree is equal to some pre-specified limit
        if node.Gain == 0 or node.Depth >= self.Max_depth or node.Leaf == True:
            node.Set_Leaf(True)
            return node

        # Until stopped
        # a. Select a node
        # b. loop all values of all features
        A, G, T = self.SplitAttribute(node)
        node.Update_A_G_T(A, G, T)
        # c. Split the node using the feature value found in step b.
        right_node, left_node = self.SplitNode(node)
        # d. Go to next node and repeat step a to c.
        node.Set_Left(left_node)
        self.GenerateTree(node.Left)
        node.Set_Right(right_node)
        self.GenerateTree(node.Right)

    def Create(self, Data=None):
        'Create a initial decision tree.'
        self.Tree = TreeNode(Data)
        self.GenerateTree(self.Tree)

    def Vote(self, node):
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

    def Testing(self, test):
        'Put the testing data into decision tree to predicte.'
        y_pred = np.zeros((test_num))
        # Prediction
        for i in range(test_num):
            tmp_node = self.Tree
            x = test.loc[i]
            # go down the decision tree untill reach the leaf
            while tmp_node.Leaf == False:
                if x[tmp_node.Attribute] < tmp_node.Threshold:
                    tmp_node = tmp_node.Left    # go to left node
                else:
                    tmp_node = tmp_node.Right   # go to right node
            # majority vote
            y_pred[i] = self.Vote(tmp_node)
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
        

# ### Question 2.1
# Using `criterion=gini`, showing the accuracy score of test data by `max_depth=3` and `max_depth=10`, respectively.
# 

# In[8]:


print("Q 2.1:")
clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
print("Max_depth = 3:")
clf_depth3.Create(train_encoding)
pred_depth3 = clf_depth3.Testing(test_encoding)
Acuracy(y_test, pred_depth3)
print("--------------------------------------")
print("Max_depth = 10:")
clf_depth10.Create(train_encoding)
pred_depth10 = clf_depth10.Testing(test_encoding)
Acuracy(y_test, pred_depth10)
print("--------------------------------------")

# ### Question 2.2
# Using `max_depth=3`, showing the accuracy score of test data by `criterion=gini` and `criterion=entropy`, respectively.
# 

# In[9]:


print("Q 2.2:")
clf_gini = DecisionTree(criterion='gini', max_depth=3)
clf_entropy = DecisionTree(criterion='entropy', max_depth=3)
print("Criterion = gini:")
clf_gini.Create(train_encoding)
pred_gini = clf_gini.Testing(test_encoding)
Acuracy(y_test, pred_gini)
print("--------------------------------------")
print("Criterion = entropy:")
clf_entropy.Create(train_encoding)
pred_entropy = clf_entropy.Testing(test_encoding)
Acuracy(y_test, pred_entropy)
print("--------------------------------------")


# - Note: Your decisition tree scores should over **0.7**. It may suffer from overfitting, if so, you can tune the hyperparameter such as `max_depth`
# - Note: You should get the same results when re-building the model with the same arguments,  no need to prune the trees
# - Hint: You can use the recursive method to build the nodes
# 

# In[16]:

# ## Question 3
# Plot the [feature importance] of your Decision Tree model. You can get the feature importance by counting the feature used for splitting data.
# 
# - You can simply plot the **counts of feature used** for building tree without normalize the importance. Take the figure below as example, outlook feature has been used for splitting for almost 50 times. Therefore, it has the largest importance
clf_depth10.PlotImportance()


# ## Question 4
# implement the AdaBooest algorithm by using the CART you just implemented from question 2 as base learner. You should implement one arguments for the AdaBooest.
# 1. **n_estimators**: The maximum number of estimators at which boosting is terminated

# In[343]:


class AdaBoost():
    def __init__(self, n_estimators):
        self.D = 0
        self.h_f = []   # to keep weak learner feature
        self.h_t = np.zeros(n_estimators)   # to keep weak learner threshold
        self.a = np.zeros(n_estimators)   # to keep weak learner weight
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
        for k in range(data.shape[0]):
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
            # x_pred = np.zeros(N)
            feature = train_data.columns[i]
            tmp_data = train_data.sort_values(by=[feature], ascending=True)
            # Continuous
            if feature == 'age' or feature == 'trestbps' or feature == 'chol' or feature == 'thalach' or feature == 'oldpeak':
                # continuous_e = float('inf')
                
                # for k in range(N-1):
                #     tmp_threshold = (train_data.at[tmp_data.index[k], feature] + train_data.at[tmp_data.index[k+1], feature]) / 2
                #     # Prediction and compute the error for weak learner
                #     tmp_pred = self.Classifier(train_data, feature, tmp_threshold, mode)    # -1 or 1
                #     tmp_e = self.CountError(label, tmp_pred)
                #     if tmp_e < continuous_e:
                #         continuous_e = tmp_e
                #         threshold = tmp_threshold
                #         tmp_pred = x_pred
                init_gain = self.Impurity(train_data['target'])
                max_gain = 0
                for k in range(N+1):
                    # tmp_threshold = train_data.at[train_data.index[k], feature]
                    tmp_left_split = self.Impurity(tmp_data.loc[tmp_data.index[0:k], ['target']])
                    tmp_right_split = self.Impurity(tmp_data.loc[tmp_data.index[k:N], ['target']])
                    after_gain = np.sum(self.D[tmp_data.index[0:k]]) * tmp_left_split + np.sum(self.D[tmp_data.index[k:N]]) * tmp_right_split
                    information_gain = init_gain - after_gain
                    # find the value of feature can yield lowest value of gini or entropy
                    if information_gain > max_gain:
                        max_gain = information_gain
                        if k == 0:  # all data split in right
                            threshold = tmp_data.at[tmp_data.index[0], feature]
                        elif k >= N:  # all data split in left
                            threshold = tmp_data.at[tmp_data.index[N-1], feature]
                        else:
                            threshold = (tmp_data.at[tmp_data.index[k-1], feature] + tmp_data.at[tmp_data.index[k], feature]) / 2
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
            feature, self.h_t[k], e, y_pred = self.WeakLearner(train_data, y_label)  # value = -1 or 1
            self.h_f.append(feature)
            print(e)
            # compute weight classifier a = (1/2)*ln((1-e)/e)
            if e == 0:  # e = 0 -> classifier good but not perfect, give it very small error
                e = np.argmin(self.D) / 10
            self.a[k] = np.log((1-e)/e) / 2
            # update distribution D = D*exp(-a*y*h(x)) / Z
            self.D = (self.D * np.exp((-self.a[k])*y_label*y_pred))
            self.D /= np.sum(self.D)

    def Prediction(self, test_data):
        'Use model to predicte the testing data.\nOutput : the prediction'
        pred = np.zeros((test_num))
        for k in range(test_num):
            for i in range(self.limitEstimator):
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
  

# ### Question 4.1
# Show the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.
# 

# In[ ]:
print("Q 4:")
# clf_10estimator = AdaBoost(n_estimators=10)
clf_100estimator = AdaBoost(n_estimators=100)
# print("N_estimator = 10:")
# clf_10estimator.Training(train_encoding)
# pred_10estimator = clf_10estimator.Prediction(test_encoding)
# Acuracy(y_test, pred_10estimator)
# print("--------------------------------------")
print("N_estimator = 100:")
clf_100estimator.Training(train_encoding)
pred_100estimator = clf_100estimator.Prediction(test_encoding)
Acuracy(y_test, pred_100estimator)
print("--------------------------------------")




# ## Question 5
# implement the Random Forest algorithm by using the CART you just implemented from question 2. You should implement three arguments for the Random Forest.
# 
# 1. **n_estimators**: The number of trees in the forest. 
# 2. **max_features**: The number of random select features to consider when looking for the best split
# 3. **bootstrap**: Whether bootstrap samples are used when building tree
# 

# In[11]:


class RandomForest():
    def __init__(self, n_estimators, max_features, boostrap=True, criterion='gini', max_depth=None):
        self.Estimator = n_estimators   # The number of trees in the forest
        self.Feature_num = max_features # The number of features to consider when looking for the best split
        self.Boostrap = boostrap    # Whether bootstrap samples are used when building trees
        self.Criterion = criterion  # The function to measure the quality of a split
        self.Depth = max_depth  # The maximum depth of the tree
        return None

    def Bagging(self):
        'Booststrap aggregating'

    def Create(self, train_data):
        'Creat a random forest.'
    
    def Testing(self, test_data):
        'Use forest to predicte the testing data.\nOutput : the prediction'
        return pred

# ### Question 5.1
# Using `criterion=gini`, `max_depth=None`, `max_features=sqrt(n_features)`, showing the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.
# 

# In[12]:


print("Q 5.1:")
clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(feature_num))
# clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(feature_num))
print("N_estimator = 10:")
clf_10tree.Create(train_encoding)
pred_10tree = clf_10tree.Testing(test_encoding)
Acuracy(y_test, pred_10tree)
print("--------------------------------------")
# print("N_estimator = 100:")
# clf_100tree.Create(train_encoding)
# pred_100tree = clf_100tree.Testing(test_encoding)
# Acuracy(y_test, pred_100tree)
# print("--------------------------------------")

# In[ ]:





# ### Question 5.2
# Using `criterion=gini`, `max_depth=None`, `n_estimators=10`, showing the accuracy score of test data by `max_features=sqrt(n_features)` and `max_features=n_features`, respectively.
# 

# In[13]:


print("Q 5.2:")
# clf_random_features = RandomForest(n_estimators=10, max_features=np.sqrt(feature_num))
clf_all_features = RandomForest(n_estimators=10, max_features=feature_num)
# print("Random features:")
# clf_random_features.Create(train_encoding)
# pred_random_features = clf_random_features.Testing(test_encoding)
# Acuracy(y_test, pred_random_features)
# print("--------------------------------------")
print("All features:")
clf_all_features.Create(train_encoding)
pred_all_features = clf_all_features.Testing(test_encoding)
Acuracy(y_test, pred_all_features)
print("--------------------------------------")

# - Note: Use majority votes to get the final prediction, you may get slightly different results when re-building the random forest model

# In[ ]:





# ### Question 6.
# Try you best to get highest test accuracy score by 
# - Feature engineering
# - Hyperparameter tuning
# - Implement any other ensemble methods, such as gradient boosting. Please note that you cannot call any package. Also, only ensemble method can be used. Neural network method is not allowed to used.



# ## Supplementary
# If you have trouble to implement this homework, TA strongly recommend watching [this video](https://www.youtube.com/watch?v=LDRbO9a6XPU), which explains Decision Tree model clearly. But don't copy code from any resources, try to finish this homework by yourself! 
