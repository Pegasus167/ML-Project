#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# card=pd.read_csv("IRIS.csv.xls")
# card

# card.head()

# card.shape

# card.describe(include='all')

# card.describe()

# card.isnull().sum()

# corr1 = card.corr()
# corr1

# sns.heatmap(corr1, annot=True)

# In[5]:


card1 = pd.read_csv("mpg.csv")
card1


# In[40]:


card1.info()


# In[6]:


card2 = card1.drop('name',axis=1)
card2.head() 


# In[7]:


corr1 = card2.corr()
corr1


# In[8]:


card2['origin']=card2['origin'].replace({1:'america',2:'europe',3:'asia'})
card2.sample(10)


# In[9]:


# card2['dummy1']=card2['origin'].replace({'america':1,'europe':0,'asia':0})
# card2.sample(10)


# In[10]:


card2.head(100)


# In[11]:


card2.tail(20)


# In[12]:


card2.describe()


# In[13]:


card2.dtypes


# In[14]:


card2.describe(include='all')


# In[15]:


card2 = pd.get_dummies(card2, columns=['origin'])
card2.sample(10)


# In[39]:


card2['horsepower']=card2['horsepower'].replace(to_replace='?', value=np.nan)
card2['horsepower']=card2['horsepower'].astype(float)
card2


# In[37]:


card2.info()


# In[36]:


card2.isnull().sum()


# In[45]:


median1=card2['horsepower'].median()
median1


# In[43]:


card2['horsepower'].replace(np.nan,median1, inplace=True)
card2


# In[20]:


card2.isnull().sum()


# In[44]:


card2.info()


# In[22]:


median1=card2['horsepower'].median()
median1


# In[23]:


mean1=card2.mean()
mean1


# In[24]:


sns.pairplot(card2)


# In[25]:


sns.countplot(card1['origin'].value_counts())


# # SPLIT DATA

# In[26]:


# Let build our Linear model
# independent variable
x = card2.drop(['mpg'],axis=1)
# dependent variable
y = card2[['mpg']]


# In[27]:


# Split x and y into training and test set in 70:30 ratio(so test size is 0.30)(random state is to freeze the state or else it will run)

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30,random_state=1)


# ## Fit Linear Model

# In[28]:


reg_model = LinearRegression()
reg_model.fit(x_train, y_train)


# In[49]:


reg_model.score(x_train, y_train)


# In[50]:


reg_model.score(x_test, y_test)


# In[55]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=2, interaction_only=True)
x_train2 = poly.fit_transform(x_train)
x_test2 = poly.fit_transform(x_test)

poly_clf = linear_model.LinearRegression()

poly_clf.fit(x_train2, y_train)

# y_pred = poly_clf.predict(x_test2)
# print(y_pred)

# in sample (training) R^2 will always improve with the number of varuables
print(poly_clf.score(x_train2, y_train))


# In[57]:


print(poly_clf.score(x_test2, y_test))


# In[62]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=3, interaction_only=True)
x_train3 = poly.fit_transform(x_train)
x_test3 = poly.fit_transform(x_test)

poly_clf = linear_model.LinearRegression()

poly_clf.fit(x_train3, y_train)

# y_pred = poly_clf.predict(x_test2)
# print(y_pred)

# in sample (training) R^2 will always improve with the number of varuables
print(poly_clf.score(x_train3, y_train))


# In[63]:


print(poly_clf.score(x_test3, y_test))


# In[59]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

poly = PolynomialFeatures(degree=4, interaction_only=True)
x_train2 = poly.fit_transform(x_train)
x_test2 = poly.fit_transform(x_test)

poly_clf = linear_model.LinearRegression()

poly_clf.fit(x_train2, y_train)

# y_pred = poly_clf.predict(x_test2)
# print(y_pred)

# in sample (training) R^2 will always improve with the number of varuables
print(poly_clf.score(x_train2, y_train))


# # overfit --> train = good
# #                test = bad
# # underfit --> train = bad
# #                test = bad
# # Generaliced --> equally good

# Lasso Regression --> (least absolute shrinkage and selection operator) is a regressoun analysis method that performs both variable selection and regularization in order to enhance the prediction accuracy and interpretability of the resulting statistical method
# Ridge Regression --> method of estimating the coeffients of multiple regression models in scenarios where the independent variable are highly correlated. (tunning performance)
# ## for less significant variables it may be reducing the value of the slope but lasso makes the value of the slope as zero.
