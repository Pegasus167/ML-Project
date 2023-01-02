#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[2]:


flo = pd.read_csv("Automobile.csv")


# In[3]:


flo.info()


# In[4]:


flo.shape


# In[5]:


flo.describe(include='all')


# In[6]:


flo.head(10)


# In[7]:


flo.sample(10)


# sns.countplot(x='sepal_length',data=flo)

# In[8]:


sns.heatmap(flo.isnull(),yticklabels=False,cmap='virdis')


# In[9]:


median=flo[sepal_width].median()
median


# In[10]:


flo2=flo.drop('species', axis=1)
flo2


# mode = flo[''].mode().values[0]
# mode

# flo['']=flo[''].replace(np.nan,mode)
# flo.head()

# In[11]:


flo1 = pd.get_dummies(flo, columns=['species'], drop_first=True)
flo1.head()


# In[ ]:


columns=[]
flo = pd.concat([flo, dummies],axis=1)
flo.drop(columns, axis=1, inplace=True)


# In[ ]:


sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmaps='virdis')


# # split the data

# In[ ]:


# flow=pd.read_csv("Automobile.csv")
# flow


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:


x = flo.drop('Survived', axis=1)
y = flo['Survived']


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# ## Fit the model

# In[ ]:


model=LogisticRegression()
model.fit(x_train, y_train)


# In[ ]:


predictions=model.predict(x_test)


# In[ ]:


from sklearn.metrics import accuracy_score 
accuracy_score(y_test,predictions)


# In[ ]:


from sklearn metrics import metrics 
print(metrics.classification_report(y_test, predictions))


# In[ ]:


cm=metric.confusion_matrix(y_test, predictions, labels=[1,0])

df_cm = pd.DataFrame(cm, index = [i for i in ['1', '0']],
                    columns = [i for i in ['Predict1','Predict0']])
plt.figure(figsixe = (7,5))
sns.heatmap(df_cm, annot=Trie,fmt='g')


# In[ ]:





# In[ ]:


from sklearn.tree import DecisionTreeClassifier


# In[ ]:


dtree = DecisionTreeClassifier(criterion = 'gini', random_state=0)
dtree.fit(x_train, y_train)


# In[ ]:





# In[ ]:





# # Project 2

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# %matplotlib inline
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression

# dia=pd.read_csv("Diabetes.csv")

# dia.head()

# dia.describe(include='all')

# dia.info()
