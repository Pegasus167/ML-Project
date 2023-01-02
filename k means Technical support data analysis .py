#!/usr/bin/env python
# coding: utf-8

# # Techincal support data analysis

# Technical suppoer data can often be a rich source of information on opportunities for improving customer experience. The lesser the trouble customers have with the product the better. Even better when the customers are able to overcome technical challenge quickly wurh minimal effort. Let us analyze the tecg support data and do some basic analysis on problem types, time to resolve th problem and channel of support that is most suitable.

# In[2]:


import pandas as pd 
import numpy as np 
import matplotlib.pylab as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.cluster import KMeans

from scipy.stats import zscore

#import sklearn.metrics


# In[3]:


tech = pd.read_csv("technical_support_data.csv")
tech.head()


# In[4]:


tech.info()


# the dataset contains one record for each unique problem type. It has metrics for each type like count, average calls to resolve, average resolution time etc.

# In[5]:


techsuppattr = tech.iloc[:,1:]
techsuppscaled=techsuppattr.apply(zscore)
sns.pairplot(techsuppscaled, diag_kind='kde')


# # Group Data into similar clusters

# Now, we will use K-means clustering to group data based on their attribute. First we need to determine the optimal number of groups, For that we conduct the knee test to see where th knee happens

# In[12]:


# Finding optimal no of clusters
from scipy.spatial.distance import cdist ### compute distance between paor of the two collections 
clusters = range(1, 10)
meanDistortions=[]

for k in clusters:
    model = KMeans(n_clusters=k)
    model.fit(techsuppscaled)
    predicton=model.predict(techsuppscaled)
    meanDistortions.append(sum(np.min(cdist(techsuppscaled, model.cluster_centers_, 'euclidean'),axis=1))/ techsuppscaled.shape[0])
    
plt.plot(clusters, meanDistortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k  eith the Elbow Method')


# Though the bend is not coming out clearly as there are many bends, let us look at 3 clusters and 5 cluster

# In[13]:


techsuppscaled.head()


# In[14]:


# Let us first start with K = 3
final_model = KMeans(3)
final_model.fit(techsuppscaled)
prediction = final_model.predict(techsuppscaled)

# Append the prediction
tech["GROUP"] = prediction
techsuppscaled["GROUP"] = prediction
print("Group Assigned : \n")
tech.head(23)


# ## analyze the distribution of the data amoung the two groups(K = 3). One of the most informative visual tool is boxplot.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




