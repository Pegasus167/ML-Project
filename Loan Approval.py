#!/usr/bin/env python
# coding: utf-8

# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[45]:


ln = pd.read_csv("Loan Approval.csv")
ln.head()


# In[47]:


ln['LoanAmount']=ln['LoanAmount'].replace(to_replace='NaN', value=np.nan)
ln['LoanAmount']=ln['LoanAmount'].astype(float)
ln


# In[48]:


median1=ln['LoanAmount'].median()
median1


# In[49]:


mean1=ln['LoanAmount'].mean()
mean1


# In[50]:


ln['LoanAmount'].replace(np.nan, mean1, inplace=True)
ln


# In[51]:


ln['Credit_History']=ln['Credit_History'].replace(to_replace='NaN', value=np.nan)
ln['Credit_History']=ln['Credit_History'].astype(float)
ln.head()


# In[52]:


mode1=ln['Credit_History'].mode()
mode1


# In[53]:


ln['Credit_History'].replace(np.nan, 1, inplace=True)
ln.head()


# In[55]:


ln['Loan_Amount_Term']=ln['Loan_Amount_Term'].replace(to_replace='NaN', value=np.nan)
ln['Loan_Amount_Term']=ln['Loan_Amount_Term'].astype(float)
ln.head()


# In[56]:


mean2=ln['Loan_Amount_Term'].mean()
mean2


# In[57]:


ln['Loan_Amount_Term'].replace(np.nan, mean2, inplace=True)
ln.head()


# In[58]:


ln['Dependents']=ln['Dependents'].replace(to_replace='NaN', value=np.nan)
ln['Dependents']=ln['Dependents'].astype(float)

ln['Dependents'].replace(np.nan, 0, inplace=True)


# In[59]:


ln['Gender']=ln['Gender'].replace({'Male':1 ,'Female':0})
ln['Married']=ln['Married'].replace({'Yes':1 ,'No':0})
ln['Education']=ln['Education'].replace({'Graduate':1 ,'Not Graduate':0})
ln['Self_Employed']=ln['Self_Employed'].replace({'Yes':1 ,'No':0})
ln['Property_Area']=ln['Property_Area'].replace({'Urban':1,'Semiurban':2, 'Rural':0})
ln['Loan_Status']=ln['Loan_Status'].replace({'Y':1 ,'N':0})


# In[60]:


lnd=ln.drop('Loan_ID', axis=1)
lnd.head()


# In[61]:


lnd.info()


# In[62]:


corr1=lnd.corr()
corr1


# ## Regression

# In[72]:


x = lnd.drop('Loan_Status', axis=1)
y = lnd['Loan_Status']


# In[73]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# In[74]:


lnd_model = LogisticRegression()
lnd_model.fit(x_train, y_train)


# In[75]:


lnd_model.score(x_train, y_train)


# In[76]:


lnd_model.score(x_test, y_test)


# ## Decision Tree Classifier

# In[83]:


dTree = DecisionTreeClassifier(criterion = 'gini', max_depth=3, random_state=0)
dTree.fit(x_train, y_train)


# In[84]:


print(dTree.score(x_train, y_train))
print(dTree.score(x_test, y_test))


# In[85]:


from sklearn.ensemble import BaggingClassifier 
bgcl = BaggingClassifier(n_estimators = 50, base_estimator=dTree, random_state=0)

# n_estimators(can be changed) are the number of model working simultaneously to get a result and base_estimators os the type of process that is gonna take place
# basically data is divided into 50 models and process the data simultaneously and provide the result and then eventually majority of the answer is seleted

bgcl = bgcl.fit(x_train, y_train)
y_predict = bgcl.predict(x_test)
print(bgcl.score(x_test, y_test))


# In[86]:


from sklearn.ensemble import AdaBoostClassifier
abcl = AdaBoostClassifier(n_estimators=100, random_state=1)

# dataset is converted in a base model and then the predictions are made from the base model. Misclassification and mistakes are resolved by giving it more weightage and then again a model and the cycle continous.
# boosting is a sequencial model and bagging is a parallel method
abcl = abcl.fit(x_train, y_train)
y_predict = abcl.predict(x_test)
print(abcl.score(x_test, y_test))


# In[88]:


from sklearn.ensemble import RandomForestClassifier
rfcl = RandomForestClassifier(n_estimators = 50, random_state=1, max_features=7)
rfcl = rfcl.fit(x_train, y_train)

# basically it processess the data by columns and by rows

y_predict = rfcl.predict(x_test)
print(rfcl.score(x_test, y_test))


# In[91]:


from sklearn import svm
from sklearn.svm import SVC


# In[92]:


clf = svm.SVC(gamma=0.25, C=3)


# In[94]:


clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
clf.score(x_train, y_train)
clf.score(x_test, y_test)


# In[95]:


plt.clf


# In[ ]:




