#!/usr/bin/env python
# coding: utf-8

# # SUPPORT VECTOR MACHINE

# We are going to build a SVM to identify each of a large number of black-and-white rectangular pixel displays as one of the 26 capital letters in the english alphabet.
# 
# The character images were based on 20 different fonts and eacha letter wirhin these 20 fonts was randomly distorted to produce a file of 20,000 unique stiule.
# 
# Each stimulus was converted into 16 primitive numerical attribute(statistical moments and edge counts) which were then scaled to fit into a range of integr values from 0 through 15.We typically train on thef first 16000 items and then use the resulting model to predict the letter category for the remaining 4000. See the article(P.W Frey and D.J State.'Letter Recongnition Using Holland style Adaptive Classifiers'.(Machine Learning Vol 6 No 2 March 91)) for more details.

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn import svm


# In[ ]:





# # BUILD Model

# In[ ]:


clf = svm.SVC(gamma=0.025, c=3)


# gamma is a measure of influence of a data point. It is inverse of distance of influence. C is penalty of wrong classifications.

# In[ ]:


clf.fit(x_train, y_train)


# In[ ]:


y_pred = clf.predict(x_test)


# In[ ]:


clf.score(x_train, y_train)


# In[ ]:


clf.score(x_test, y_test)


# In[ ]:





# # FIT THE MODEL

# In[ ]:


model= LogisticRegression()

