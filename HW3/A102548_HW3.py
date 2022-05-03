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


from re import L
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
    p1 = np.sum(sequence == 0) / seq_len
    p2 = np.sum(sequence == 1) / seq_len

    # Gini = 1 - sum_all_class_k(Pk^2)
    gini = 1 - (p1**2) - (p2**2)
    return gini


def entropy(sequence):
    'Compute the entropy for two class. The smaller, the purer.\nOutput : entropy'
    seq_len = len(sequence)
    p1 = np.sum(sequence == 0) / seq_len
    print(p1)
    p2 = np.sum(sequence == 1) / seq_len
    print(p2)

    if p1 == 0 or p2 == 0:  # all class are the same in one node
        entropy = 0
    elif p1 == p2:  # class are half-and-half
        entropy = 1
    else:   # Entropy = -sum_all_class_k(Pk*log2(Pk))
        entropy = -(p1*np.log2(p1)) - (p2*np.log2(p2))    
    return entropy


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
    def __init__(self, dataSet=None, depth=0) -> None:
        self.DataSet = dataSet    # input data
        self.Attribute = None # selected attribute
        self.Threshold = 0  # threshold for node
        self.Left = None    # left node
        self.Right = None   # right node
        self.Depth = depth  # depth of node
        self.Leaf = False   # is leaf node or not
        self.Gain = 0   # information gain


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
        bestfeature = 0
        # loop all values of all features
        for i in range(len(node.DataSet.column)-1): # remove the 'target' colcum
            # partition the node and calculate the purity of data
            # pick the col of feature[i], sort by ascending order
            tmp_data = node.DataSet.sort_values([node.DataSet.colcum[i]], ascending=True).to_numpy
            # split by threshold of average(i-th, (i+1)-th)
            N = len(tmp_data)
            for i in range(N-1):
                tmp_left_split = self.SplitPurity(tmp_data[:i+1])
                tmp_right_split = self.SplitPurity(tmp_data[i+1:])
                after_gain = ((i+1) * tmp_left_split + (N-i-1) * tmp_right_split) / N
                information_gain = node.Gain - after_gain
                # find the value of feature can yield lowest value of gini or entropy
                if information_gain > max_gain:
                    max_gain = information_gain
                    bestfeature = node.DataSet.column[i]
                    best_threshold = (tmp_data[i] + tmp_data[i+1]) / 2
        return bestfeature, max_gain, best_threshold

    def SplitNode(self, node):
        'Split the data(node) by the attribute.\nOutput : splited nodes'
        R_index = []
        L_index = []
        # Split data into left and right node by threshold
        for i in range(len(node.DataSet[node.Attribute])):
            if node.DataSet[node.Attribute] <= node.Threshold:
                R_index.append(node.index[i])
            else:
                L_index.append(node.index[i])
        R = pd.DataFrame(node.DataSet.iloc[R_index])
        L = pd.DataFrame(node.DataSet.iloc[L_index])
        # Remove the feature that had pick
        R = R.drop([node.Attribute], axis=1)
        L = L.drop([node.Attribute], axis=1)
        node_R = TreeNode(R, node.Depth+1)
        node_L = TreeNode(L, node.Depth+1)
        return node_R, node_L

    def GenerateTree(self, node):
        'Generate the decision tree by recursive method.'
        node.Gain = self.SplitPurity(node.DataSet['target'])
        # Stopping criteria
        # The data in each leaf-node belongs to the same class
        # Depth of the tree is equal to some pre-specified limit
        if node.Gain == 0 or node.Depth >= self.Max_depth:
            node.Leaf = True
            return node

        # Until stopped
        # a. Select a node
        # b. loop all values of all features
        node.Attribute, node.Gain, node.Threshold = self.SplitAttribute(node)    
        # c. Split the node using the feature value found in step b.
        right_node, left_node = self.SplitNode(node)
        # d. Go to next node and repeat step a to c.
        node.Left = self.GenerateTree(left_node)
        node.Right = self.GenerateTree(right_node)

    def Create(self, Data=None):
        'Create a initial decision tree.'
        self.Tree = TreeNode(Data)
        self.GenerateTree(self.Tree)

        


# ### Question 2.1
# Using `criterion=gini`, showing the accuracy score of test data by `max_depth=3` and `max_depth=10`, respectively.
# 

# In[8]:


clf_depth3 = DecisionTree(criterion='gini', max_depth=3)
clf_depth10 = DecisionTree(criterion='gini', max_depth=10)


# ### Question 2.2
# Using `max_depth=3`, showing the accuracy score of test data by `criterion=gini` and `criterion=entropy`, respectively.
# 

# In[9]:


clf_gini = DecisionTree(criterion='gini', max_depth=3)
clf_entropy = DecisionTree(criterion='entropy', max_depth=3)


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
