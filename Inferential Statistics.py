#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[4]:


df=pd.read_csv('Automobile.csv')


# In[5]:


df.head()


# In[6]:


df.tail()


# In[8]:


df.isnull().sum()


# In[10]:


df.shape


# In[13]:


df.dtypes


# In[17]:


df1=pd.read_csv('IRIS.csv.xls')


# In[18]:


df1.shape


# In[19]:


df1.dtypes


# In[20]:


df1.isnull().sum()


# In[21]:


df1.head()


# In[22]:


df2 = pd.read_csv('https://raw.githubusercontent.com/Premalatha-success/Statistics/main/CardioGoodFitness-1.csv')


# In[23]:


df2.head()


# In[24]:


df2.columns


# In[25]:


df2.describe()


# In[26]:


df2.describe(include="all")


# In[30]:


df2.hist(figsize=(10,20))


# In[31]:


sns.boxplot(x='Product',y='Age',data=df2)


# In[32]:


pd.crosstab(df2['Product'],df2['Gender'])


# In[36]:


sns.countplot(x='Product',hue='Gender',data=df2)


# In[39]:


pd.pivot_table(df2,'MaritalStatus', index=['Product','Gender'],aggfunc=len)


# In[41]:


pd.pivot_table(df2,'Income',index=['Product','Gender'],columns=['MaritalStatus'])


# In[54]:


pd.pivot_table(df2,'Miles',index=['Product','Gender'],columns=['MaritalStatus'])


# In[43]:


sns.pairplot(df2,size)


# In[44]:


sns.pairplot(df2,kind='kde')


# In[45]:


sns.pairplot(df2,diag_kind='kde')


# In[46]:


corr=df2.corr()
corr


# In[50]:


sns.heatmap(corr,annot=True)


# In[62]:


df2.hist(by='Gender',column='Income',figsize=(10,10))


# In[ ]:




