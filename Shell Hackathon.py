#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.svm import SVC


# In[2]:


shl = pd.read_csv("Shell Demand.csv")


# In[3]:


shl.head()


# In[4]:


shl2= pd.read_csv("Shell Existing EV infrastructures.csv")


# In[5]:


shl2.head()


# In[6]:


shl.describe()


# In[7]:


shl.info()


# In[8]:


mean1=shl["2010"].mean()
mean1


# In[9]:


mean2=shl["2011"].mean()
mean2


# In[10]:


mean3=shl["2012"].mean()
mean3


# In[11]:


mean4=shl["2013"].mean()
mean4


# In[12]:


mean5=shl["2014"].mean()
mean5


# In[13]:


mean6=shl["2015"].mean()
mean6


# In[14]:


mean7=shl["2016"].mean()
mean7


# In[15]:


mean8=shl["2017"].mean()
mean8


# In[16]:


mean9=shl["2018"].mean()
mean9


# In[17]:


shl['2010'].replace(0,mean1)#inplace=True)
shl['2011'].replace(0,mean2)
shl['2012'].replace(0,mean3)
shl['2013'].replace(0,mean4)
shl['2014'].replace(0,mean5)
shl['2015'].replace(0,mean6)
shl['2016'].replace(0,mean7)
shl['2017'].replace(0,mean8)
shl['2018'].replace(0,mean9)
shl


# In[18]:


shl2.info()


# In[1]:


corr1=shl2.corr()
corr1


# In[20]:


sns.pairplot(shl)


# In[21]:


sns.pairplot(shl2)


# In[22]:


# shl.columns


# In[23]:


'''
x = shl.drop(['demand_point_index'],axis=1)

y = shl[['demand_point_index']]
'''


# In[24]:


# x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.30,random_state=1)


# In[28]:


svm.SVC(kernel='linear')
slf = svm.SVC(kernel='linear', gamma=0.025) #c=3)
slf.fit(x_train, y_train)

y_predict = slf.predict(x_test)


# In[33]:


from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict))


# In[ ]:





# In[34]:


shl2.columns


# In[36]:


a = shl2.drop(['supply_point_index'],axis=1)

b = shl2[['supply_point_index']]


# In[39]:


a_train, a_test, b_train, b_test= train_test_split(a, b, test_size=0.30,random_state=1)


# In[40]:


slf = svm.SVC(kernel='linear', gamma=0.025) #c=3)
slf.fit(a_train, b_train)

b_predict = slf.predict(a_test)


# In[41]:


print(classification_report(b_test, b_predict))


# In[ ]:




