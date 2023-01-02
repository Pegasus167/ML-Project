#!/usr/bin/env python
# coding: utf-8

# # Decision Tree

# # Build Decision Tree Model

# We will build our model using the Decision Tree Classifier function. Using default "gini criteria" to split. Other option include ' entropy' and 'Information Gate'

# In[54]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


flo = pd.read_csv("Automobile.csv")
flo.info()


# In[3]:


flo.head()


# In[4]:


flo.all()


# In[5]:


print(flo.dtypes)


# In[7]:


flo['Age']=flo['Age'].replace(to_replace='NaN', value=np.nan)
flo['Age']=flo['Age'].astype(float)
flo


# In[ ]:


flow=flo.drop('Cabin')


# In[6]:


flo['Age']=flo['Age'].astype(int)


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# # Trying with Gini

# In[ ]:


dtree = DecisionTreeClassifier(criterion = 'gini', random_state=0)
dtree.fit(x_train, y_train)


# ## Scoring our Decision Tree

# In[ ]:


print(dTree.score(x_train, y_train))
print(dTree.score(x_test, y_test))

#(overfitting-- train is good and test is bad)


# # Trying with Entropy

# In[ ]:


dtree = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
dtree.fit(x_train, y_train)


# ## Scoring our Decision Tree

# In[ ]:


print(dTree.score(x_train, y_train))
print(dTree.score(x_test, y_test))

#(overfitting again)5% different is allowed


# problem with decision tree is that it is always overfitting
# 
# ## Reducing over fitting (Regularization)

# In[ ]:


dTreeR = DecisionTreeClassifier(criterion = 'gini', max_depth = 3, random_state = 0)
dTree.fit(x_train, y_train)

# max_depth is stopping the tree at level 3 (you have to check what's the level perfect for your dataset)
# overfitting can be overcomed 
print(dTreeR.score(x_train, y_train)
print(dTreeR.score(x_train, y_train)


# In[ ]:


dTreeR = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, random_state = 0)
dTree.fit(x_train, y_train)

# max_depth is stopping the tree at level 3 (you have to check what's the level perfect for your dataset)
# overfitting can be overcomed 
print(dTreeR.score(x_train, y_train)
print(dTreeR.score(x_train, y_train)


# In[ ]:


train_char_label=['No', 'Yes']
Credit_Tree_FileR = open('credit_treeR.dot','w')
dot_data = tree.export_graphviz(dTree, out_file = Credit_Tree_FileR, feature_names = list(x_train), class)
Credit_Tree_FileR.close()

#works only if dot command works on your machine

retCode = system("dot - Tpng credit_treeR.dot -o credit_treeR.png")
if(retCode.0):
    print("System command returning error: "+str(retCode))
else:
    display(Image("credit_treeR.png"))
    
# to display the tree


# In[ ]:


# importance of feature in the tree building (The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini *incomplete)

print (flo.DataFrame(dTreeR.feature_importances_, columns = ['Imp'], index = x_train.columns))


# # Ensemble learning - Bagging

# In[ ]:


from sklearn.ensemble import BaggingClassifier 
bgcl = BaggingClassifier(n_estimators = 50, base_estimator=dTree, random_state=0)

# n_estimators(can be changed) are the number of model working simultaneously to get a result and base_estimators os the type of process that is gonna take place
# basically data is divided into 50 models and process the data simultaneously and provide the result and then eventually majority of the answer is seleted

bgcl = bgcl.fit(x_train, y_train)
y_predict = bgcl.predict(x_test)
print(bgcl.score(x_test, y_test))


# In[ ]:


cm=metrics.confusion_matrix(y_test, ypredict, labels=[0, 1])

df_cm = pd.DataFrame(cm, index = [i for i in ["No","Yes"]],
                    columns=[i fori in ["No","Yes"]])
plt.figure(figsize = (7, 5))
sns.heatmap(df_cm, annot=True, fmt='g')


# # Ensemble Learning - AdaBoosting model

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(n_estimators=100, random_state=1)

# dataset is converted in a base model and then the predictions are made from the base model. Misclassification and mistakes are resolved by giving it more weightage and then again a model and the cycle continous.
# boosting is a sequencial model and bagging is a parallel method
abcl = abcl.fit(x_train, y_train)
y_predict = abcl.predict(x_test)
print(abcl.score(x_test, y_test))


# In[ ]:


cm metric.confusion_matrix(y_test, y_predict, labels=[0,1])

df_cm = flo.DtaFrame(cm, index = [i for i in ["No","Yes"]]
                    columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(df_cm, annot=True, fmt='g')


# # Ensemble Learning - GradientBoost

# In[ ]:


from sklearn.ensemble import GradientBoostingClassifier
gbcl = GradientBoostingClassifier(n_estimators = 50, random_state=1)

#works the same way

gbcl = gbcl.fit(x_train, y_train)
y_predict = gbcl.predict(x_test)
print(gbcl,score(x_test, y_test))


# In[ ]:


cm=metrics.confusion_matrix(y_test, y_predict, labels=[0, 1])

flo_cm = pd.DataFrame(cm, index = [i for i in ["No", "Yes"]],
                      columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7, 5))
sns.heatmap(flo_cm, annot=True, fmt='g')


# # Ensemble RandomForest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators = 50, random_state=1, max_features=12)
rfcl = rfcl.fit(x_train, y_train)

# basically it processess the data by columns and by rows

y_predict = rfcl.predict(x_test)
print(rfcl.score(x_test, y_test))


# In[ ]:




cm = metrics.confusion_matrix(y_test, y_predict, lablels=[0, 1])

flo_cm = pd.DataFrame(cm, index=[i for i in ["No","Yes"]],
                      columns = [i for i in ["No","Yes"]])
plt.figure(figsize = (7,5))
sns.heatmap(flo_cm, annot=True ,fmt='g')


# In[ ]:





# In[ ]:





# # Data used from linear regression 

# In[27]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# In[28]:


card1 = pd.read_csv("mpg.csv")
card1


# In[29]:


card2 = card1.drop('name',axis=1)
card2.head() 


# In[30]:


corr1 = card2.corr()
corr1


# In[31]:


card2['origin']=card2['origin'].replace({1:'america',2:'europe',3:'asia'})
card2.sample(10)


# In[32]:


card2.describe()


# In[33]:


card2 = pd.get_dummies(card2, columns=['origin'])
card2.sample(10)


# In[34]:


card2['horsepower']=card2['horsepower'].replace(to_replace='?', value=np.nan)
card2['horsepower']=card2['horsepower'].astype(float)
card2


# In[35]:


median1=card2['horsepower'].median()
median1


# In[40]:


card2['horsepower'].replace(np.nan,median1, inplace=True)
card2


# In[41]:


card2.info()


# ### Regression

# In[57]:


# Let build our Linear model
# independent variable
x = card2.drop(['mpg'],axis=1)
# dependent variable
y = card2[['mpg']]


# In[58]:


x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30,random_state=1)


# In[44]:


reg_model = LinearRegression()
reg_model.fit(x_train, y_train)


# In[47]:


reg_model.score(x_train, y_train)


# In[48]:


reg_model.score(x_test, y_test)


# ## DecisionTree Classifier

# In[74]:


from sklearn.tree import DecisionTreeRegressor


dTreeR = DecisionTreeRegressor(criterion='mse', max_depth = 3, random_state = 0)
dTreeR.fit(x_train, y_train)

# max_depth is stopping the tree at level 3 (you have to check what's the level perfect for your dataset)
# overfitting can be overcomed 

dTreeR.score(x_train, y_train)


# In[75]:


dTreeR.score(x_test, y_test)


# ## Bagging

# In[83]:


from sklearn.ensemble import BaggingRegressor
bgcl = BaggingRegressor(n_estimators = 50, base_estimator=dTreeR, random_state=0)

# n_estimators(can be changed) are the number of model working simultaneously to get a result and base_estimators os the type of process that is gonna take place
# basically data is divided into 50 models and process the data simultaneously and provide the result and then eventually majority of the answer is seleted

bgcl = bgcl.fit(x_train, y_train)
y_predict = bgcl.predict(x_test)
bgcl.score(x_test, y_test)


# ## AdaBooting

# In[88]:


from sklearn.ensemble import AdaBoostRegressor
abcl = AdaBoostRegressor(n_estimators=120, random_state=1)

# dataset is converted in a base model and then the predictions are made from the base model. Misclassification and mistakes are resolved by giving it more weightage and then again a model and the cycle continous.
# boosting is a sequencial model and bagging is a parallel method
abcl = abcl.fit(x_train, y_train)
y_predict = abcl.predict(x_test)
print(abcl.score(x_test, y_test))


# ## Gradient 

# In[104]:


from sklearn.ensemble import GradientBoostingRegressor
gbcl = GradientBoostingRegressor(n_estimators = 100, random_state=1)

#works the same way

gbcl = gbcl.fit(x_train, y_train)
y_predict = gbcl.predict(x_test)
print(gbcl.score(x_test, y_test))


# ## Random Forest Classifier

# In[105]:


from sklearn.ensemble import RandomForestRegressor
rfcl = RandomForestRegressor(n_estimators = 100, random_state=1, max_features=5)

rfcl = rfcl.fit(x_train, y_train)
y_predict = rfcl.predict(x_test)
rfcl.score(x_test, y_test)


# In[ ]:




