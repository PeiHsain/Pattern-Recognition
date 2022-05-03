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


import pandas as pd
import numpy as np
file_url = "http://storage.googleapis.com/download.tensorflow.org/data/heart.csv"
df = pd.read_csv(file_url)

train_idx = np.load('train_idx.npy')
test_idx = np.load('test_idx.npy')

train_df = df.iloc[train_idx]
test_df = df.iloc[test_idx]

# one-hot encoding
train_num = len(train_df.index)
test_num = len(test_df.index)
# for feature 'thal' -> normal, fixed, reversable
train_tmp = train_df[['target']]
test_tmp = test_df[['target']]
train_encoding = pd.get_dummies(train_df, prefix=['thal'])
test_encoding = pd.get_dummies(test_df, prefix=['thal'])
train_encoding = train_encoding.drop('target', axis=1)
test_encoding = test_encoding.drop('target', axis=1)
train_encoding = pd.concat([train_encoding, train_tmp], axis=1)
test_encoding = pd.concat([test_encoding, test_tmp], axis=1)

# for feature 'cp' -> value 1-4
for n in range(1, 5):
    tmp = pd.DataFrame(data=train_encoding, columns=['cp'])
    for i in range(train_num):
        if tmp.at[tmp.index[i], 'cp'] == n:
            tmp.at[tmp.index[i], 'cp'] = 1
        else:
            tmp.at[tmp.index[i], 'cp'] = 0
    train_encoding.insert(0, f'cp_{n}', tmp)
    tmp = pd.DataFrame(data=test_encoding, columns=['cp'])
    for i in range(test_num):
        if tmp.at[tmp.index[i], 'cp'] == n:
            tmp.at[tmp.index[i], 'cp'] = 1
        else:
            tmp.at[tmp.index[i], 'cp'] = 0
    test_encoding.insert(0, f'cp_{n}', tmp)
train_encoding = train_encoding.drop('cp', axis=1)
test_encoding = test_encoding.drop('cp', axis=1)

# for feature 'restecg' -> value 0-2
for n in range(3):
    tmp = pd.DataFrame(data=train_encoding, columns=['restecg'])
    for i in range(train_num):
        if tmp.at[tmp.index[i], 'restecg'] == n:
            tmp.at[tmp.index[i], 'restecg'] = 1
        else:
            tmp.at[tmp.index[i], 'restecg'] = 0
    train_encoding.insert(0, f'restecg_{n}', tmp)
    tmp = pd.DataFrame(data=test_encoding, columns=['restecg'])
    for i in range(test_num):
        if tmp.at[tmp.index[i], 'restecg'] == n:
            tmp.at[tmp.index[i], 'restecg'] = 1
        else:
            tmp.at[tmp.index[i], 'restecg'] = 0
    test_encoding.insert(0, f'restecg_{n}', tmp)
train_encoding = train_encoding.drop('restecg', axis=1)
test_encoding = test_encoding.drop('restecg', axis=1)

# for feature 'slop' -> value 1-3
for n in range(1, 4):
    tmp = pd.DataFrame(data=train_encoding, columns=['slope'])
    for i in range(train_num):
        if tmp.at[tmp.index[i], 'slope'] == n:
            tmp.at[tmp.index[i], 'slope'] = 1
        else:
            tmp.at[tmp.index[i], 'slope'] = 0
    train_encoding.insert(0, f'slope_{n}', tmp)
    tmp = pd.DataFrame(data=test_encoding, columns=['slope'])
    for i in range(test_num):
        if tmp.at[tmp.index[i], 'slope'] == n:
            tmp.at[tmp.index[i], 'slope'] = 1
        else:
            tmp.at[tmp.index[i], 'slope'] = 0
    test_encoding.insert(0, f'slope_{n}', tmp)
train_encoding = train_encoding.drop('slope', axis=1)
test_encoding = test_encoding.drop('slope', axis=1)

# for feature 'ca' -> value 0-3
for n in range(4):
    tmp = pd.DataFrame(data=train_encoding, columns=['ca'])
    for i in range(train_num):
        if tmp.at[tmp.index[i], 'ca'] == n:
            tmp.at[tmp.index[i], 'ca'] = 1
        else:
            tmp.at[tmp.index[i], 'ca'] = 0
    train_encoding.insert(0, f'ca_{n}', tmp)
    tmp = pd.DataFrame(data=test_encoding, columns=['ca'])
    for i in range(test_num):
        if tmp.at[tmp.index[i], 'ca'] == n:
            tmp.at[tmp.index[i], 'ca'] = 1
        else:
            tmp.at[tmp.index[i], 'ca'] = 0
    test_encoding.insert(0, f'ca_{n}', tmp)
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
        self.Criterion = criterion
        self.Max_depth = max_depth
        self.Tree = None
        return None

    def SplitPurity(self, dataset):
        'Calculate the purity for the split.\nOutput : value of purity'
        if self.Criterion == 'gini':
            value = gini(dataset)
        elif self.Criterion == 'entropy':
            value = entropy(dataset)
        return value

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
                for k in range(N-1):
                    tmp_left_split = self.SplitPurity(tmp_data.loc[tmp_data.index[0:k+1], ['target']])
                    tmp_right_split = self.SplitPurity(tmp_data.loc[tmp_data.index[k+1:N], ['target']])
                    after_gain = ((k+1) * float(tmp_left_split) + (N-k-1) * float(tmp_right_split)) / N
                    information_gain = node.Gain - after_gain
                    # find the value of feature can yield lowest value of gini or entropy
                    if information_gain > max_gain:
                        max_gain = information_gain
                        bestfeature = node.DataSet.columns[i]
                        best_threshold = (tmp_data.at[tmp_data.index[k], bestfeature] + tmp_data.at[tmp_data.index[k+1], bestfeature]) / 2
            else:
                # Discrete
                k = len(tmp_data[tmp_data[node.DataSet.columns[i]] < 1])
                tmp_left_split = self.SplitPurity(tmp_data.loc[tmp_data.index[0:k], ['target']]) # data = 0
                tmp_right_split = self.SplitPurity(tmp_data.loc[tmp_data.index[k:N], ['target']]) # data = 1
                after_gain = (k * float(tmp_left_split) + (N-k) * float(tmp_right_split)) / N
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
        node.Gain = self.SplitPurity(node.DataSet['target'])
        # print(node.Gain)
        # print("len ", node.DataSet.shape[0])
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
        # Cannot split by the feature
        # if node.Gain == 0:
        #     node.Set_Leaf(True)
        #     return node 
        # print(node.Attribute)
        # c. Split the node using the feature value found in step b.
        right_node, left_node = self.SplitNode(node)
        # d. Go to next node and repeat step a to c.
        # print("L")
        node.Set_Left(left_node)
        self.GenerateTree(node.Left)
        # print("R")
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

    def Accuracy(self, test, test_predict):
        'Compute and show the accuracy score.'
        accur = 0
        for i in range(test_num):
            if test.at[test.index[i], 'target'] == test_predict[i]:
                accur += 1
        accur /= test_num
        print(f"Accuracy score = {accur}")


    def Testing(self, test):
        'Put the testing data into decision tree to predicte.'
        test_pred = np.zeros((test_num))
        # Prediction
        for i in range(test_num):
            tmp_node = self.Tree
            x = test.loc[test.index[i]]
            # go down the decision tree untill reach the leaf
            while tmp_node.Leaf == False:
                if x[tmp_node.Attribute] < tmp_node.Threshold:
                    tmp_node = tmp_node.Left    # go to left node
                else:
                    tmp_node = tmp_node.Right   # go to right node
            # majority vote
            test_pred[i] = self.Vote(tmp_node)
        # Show the accuracy
        self.Accuracy(test, test_pred)
        

# ### Question 2.1
# Using `criterion=gini`, showing the accuracy score of test data by `max_depth=3` and `max_depth=10`, respectively.
# 

# In[8]:


print("Q 2.1:")
clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth10 = DecisionTree(criterion='gini', max_depth=10)
print("Max_depth = 3:")
clf_depth3.Create(train_encoding)
clf_depth3.Testing(test_encoding)
print("--------------------------------------")
print("Max_depth = 10:")
clf_depth10.Create(train_encoding)
clf_depth10.Testing(test_encoding)
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
clf_gini.Testing(test_encoding)
print("--------------------------------------")
print("Criterion = entropy:")
clf_entropy.Create(train_encoding)
clf_entropy.Testing(test_encoding)
print("--------------------------------------")


# - Note: Your decisition tree scores should over **0.7**. It may suffer from overfitting, if so, you can tune the hyperparameter such as `max_depth`
# - Note: You should get the same results when re-building the model with the same arguments,  no need to prune the trees
# - Hint: You can use the recursive method to build the nodes
# 

# ## Question 3
# Plot the [feature importance] of your Decision Tree model. You can get the feature importance by counting the feature used for splitting data.
# 
# - You can simply plot the **counts of feature used** for building tree without normalize the importance. Take the figure below as example, outlook feature has been used for splitting for almost 50 times. Therefore, it has the largest importance
# 
# ![image]

# ## Question 4
# implement the AdaBooest algorithm by using the CART you just implemented from question 2 as base learner. You should implement one arguments for the AdaBooest.
# 1. **n_estimators**: The maximum number of estimators at which boosting is terminated

# In[343]:


class AdaBoost():
    def __init__(self, n_estimators):
        return None


# In[ ]:





# ### Question 4.1
# Show the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.
# 

# In[ ]:





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
        return None


# ### Question 5.1
# Using `criterion=gini`, `max_depth=None`, `max_features=sqrt(n_features)`, showing the accuracy score of test data by `n_estimators=10` and `n_estimators=100`, respectively.
# 

# In[12]:


clf_10tree = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
clf_100tree = RandomForest(n_estimators=100, max_features=np.sqrt(x_train.shape[1]))


# In[ ]:





# ### Question 5.2
# Using `criterion=gini`, `max_depth=None`, `n_estimators=10`, showing the accuracy score of test data by `max_features=sqrt(n_features)` and `max_features=n_features`, respectively.
# 

# In[13]:


clf_random_features = RandomForest(n_estimators=10, max_features=np.sqrt(x_train.shape[1]))
clf_all_features = RandomForest(n_estimators=10, max_features=x_train.shape[1])


# - Note: Use majority votes to get the final prediction, you may get slightly different results when re-building the random forest model

# In[ ]:





# ### Question 6.
# Try you best to get highest test accuracy score by 
# - Feature engineering
# - Hyperparameter tuning
# - Implement any other ensemble methods, such as gradient boosting. Please note that you cannot call any package. Also, only ensemble method can be used. Neural network method is not allowed to used.

# In[ ]:


from sklearn.metrics import accuracy_score


# In[4]:


y_test = test_df['target']


# In[ ]:


y_pred = your_model.predict(x_test)


# In[ ]:


print('Test-set accuarcy score: ', accuracy_score(y_test, y_pred))


# ## Supplementary
# If you have trouble to implement this homework, TA strongly recommend watching [this video](https://www.youtube.com/watch?v=LDRbO9a6XPU), which explains Decision Tree model clearly. But don't copy code from any resources, try to finish this homework by yourself! 

# In[ ]:
