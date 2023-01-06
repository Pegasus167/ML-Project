#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor


# In[3]:


fare = pd.read_csv("Taxifare.csv")
fare.head(50)


# In[4]:


fare.info()


# In[5]:


fare2 = fare.drop("unique_id", axis=1)
fare1 = fare2.drop("date_time_of_pickup", axis=1)


# In[6]:


fare1.info()


# In[50]:


y = fare1.drop(['amount'], axis=1)
x = fare1[['amount']]


# In[55]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30,random_state=42)


# In[56]:


reg_model = LinearRegression()
reg_model.fit(x_train, y_train)


# In[57]:


reg_model.score(x_train, y_train)


# In[58]:


reg_model.score(x_test, y_test)


# In[68]:


dTreeR = DecisionTreeRegressor(criterion = 'mse', max_depth = 3, random_state = 0)
dTreeR.fit(x_train, y_train)

# max_depth is stopping the tree at level 3 (you have to check what's the level perfect for your dataset)
# overfitting can be overcomed 

dTreeR.score(x_train, y_train)


# In[69]:


dTreeR.score(x_test, y_test)


# In[73]:


# train_char_label=['No', 'Yes']
# Credit_Tree_FileR = open('credit_treeR.dot','w')
# dot_data = tree.export_graphviz(dTree, out_file = Credit_Tree_FileR, feature_names = list(x_train), class)
# Credit_Tree_FileR.close()

# #works only if dot command works on your machine

# retCode = system("dot - Tpng credit_treeR.dot -o credit_treeR.png")
# if(retCode.0):
#     print("System command returning error: "+str(retCode))
# else:
#     display(Image("credit_treeR.png"))


# In[74]:


from sklearn.ensemble import BaggingRegressor
bgcl = BaggingRegressor(n_estimators = 50, base_estimator=dTreeR, random_state=0)

# n_estimators(can be changed) are the number of model working simultaneously to get a result and base_estimators os the type of process that is gonna take place
# basically data is divided into 50 models and process the data simultaneously and provide the result and then eventually majority of the answer is seleted

bgcl = bgcl.fit(x_train, y_train)
y_predict = bgcl.predict(x_test)
bgcl.score(x_test, y_test)


# In[82]:


from sklearn.ensemble import AdaBoostRegressor
abcl = AdaBoostRegressor(n_estimators=120, random_state=1)

# dataset is converted in a base model and then the predictions are made from the base model. Misclassification and mistakes are resolved by giving it more weightage and then again a model and the cycle continous.
# boosting is a sequencial model and bagging is a parallel method


abcl = abcl.fit(x_train, x_train)
y_predict = abcl.predict(x_test)
abcl.score(x_test, x_test)

