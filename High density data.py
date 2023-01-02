#!/usr/bin/env python
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[ ]:


# create a user definded function called remove_outlier for getting threshold value from IQR
# def remove_outlier(col):
#     sorted(col)
#     Q1, Q3=col.quantile([0.25, 0.75])
#     IQR = Q3-Q1
#     lower_range= Q1-(1.5*IQR)
#     upper_range= Q3+(1.5*IQR)
#     return lower_range, upper_range

# LINE 28


# # Loading the data set

# In[2]:


tp = pd.read_csv("hotelbooking.csv")
tp.head()


# In[3]:


tp.describe()


# In[4]:


tp.info()


# In[5]:


tp.tail()


# In[6]:


tp.shape


# In[7]:


tp.isnull().sum()


# In[8]:


median1=tp["children"].median()
median1


# In[9]:


mode2=tp["country"].mode().values[0]
mode2
tp["country"]= tp["country"].replace(np.nan,mode2)


# In[10]:


median3=tp["agent"].median()
median3


# In[11]:


median4=tp["company"].median()
median4


# In[12]:


mean1=tp["children"].mean()
mean1


# In[13]:


mean3=tp["agent"].mean()
mean3


# In[14]:


mean4=tp["company"].mean()
mean4


# In[15]:


tp["children"].replace(np.nan, median1, inplace=True)
tp["agent"].replace(np.nan, median3, inplace=True)
tp["company"].replace(np.nan, median4, inplace=True)


# In[16]:


tp.isnull().sum()


# In[17]:


duplicate = tp.duplicated()
print(duplicate.sum())
tp[duplicate]


# In[18]:


tp.drop_duplicates(inplace=True)


# In[19]:


tp1=tp.duplicated()
tp1.sum()


# In[29]:


tp.info()


# In[27]:


tp.boxplot(column=["previous_bookings_not_canceled"])
plt.show


# In[39]:


tp.boxplot(column=["stays_in_weekend_nights"])
plt.show


# In[41]:


tp.boxplot(column=["stays_in_week_nights"])
plt.show


# In[28]:


# create a user definded function called remove_outlier for getting threshold value from IQR
def remove_outlier(col):
    sorted(col)
    Q1, Q3=col.quantile([0.25, 0.75])
    IQR = Q3-Q1
    lower_range= Q1-(1.5*IQR)
    upper_range= Q3+(1.5*IQR)
    return lower_range, upper_range


# In[33]:


# create a user definded function called remove_outlier for getting threshold value from IQR
lowtime, upptime=remove_outlier(tp["lead_time"])
tp['lead_time']=np.where(tp['lead_time']>upptime,upptime,tp['lead_time'])
tp['lead_time']=np.where(tp['lead_time']<lowtime,lowtime,tp['lead_time'])


# In[34]:


tp.boxplot(column=["lead_time"]) 
plt.show


# In[42]:


lowtime, upptime=remove_outlier(tp["stays_in_weekend_nights"])
tp['stays_in_weekend_nights']=np.where(tp['stays_in_weekend_nights']>upptime,upptime,tp['stays_in_weekend_nights'])
tp['stays_in_weekend_nights']=np.where(tp['stays_in_weekend_nights']<lowtime,lowtime,tp['stays_in_weekend_nights'])


# In[43]:


tp.boxplot(column=["stays_in_weekend_nights"])
plt.show


# In[44]:


lowtime, upptime=remove_outlier(tp["stays_in_week_nights"])
tp['stays_in_week_nights']=np.where(tp['stays_in_week_nights']>upptime,upptime,tp['stays_in_week_nights'])
tp['stays_in_week_nights']=np.where(tp['stays_in_week_nights']<lowtime,lowtime,tp['stays_in_week_nights'])


# In[45]:


tp.boxplot(column=["stays_in_week_nights"])
plt.show


# In[46]:


# STANDARD SCALER
# scales the data. Esssentially returns the z-scoresof every attribute
# z score: z score(also called a standord score) gives you an idea of how far from the mean a data point is.
#     but more technically it's a measure of how many standard deviations below or above the population mean a 
#     z = x-pwe(lu)(mean)/sigma
from sklearn.preprocessing import StandardScaler
std_scale = StandardScaler()
std_scale


# In[48]:


tp["lead_time"] = std_scale.fit_transform(tp[["lead_time"]])
tp["stays_in_weekend_nights"] = std_scale.fit_transform(tp[["stays_in_weekend_nights"]])
tp["stays_in_week_nights"] = std_scale.fit_transform(tp[["stays_in_week_nights"]])


# In[49]:


dummies=pd.get_dummies(tp[["hotel","arrival_date_month","meal","country","market_segment","distribution_channel","reserved_room_type","assigned_room_type","deposit_type","customer_type","reservation_status","reservation_status_date"]],
                        columns=["hotel","arrival_date_month","meal","country","market_segment","distribution_channel","reserved_room_type","assigned_room_type","deposit_type","customer_type","reservation_status","reservation_status_date"],drop_first=True)


# In[50]:


dummies.head()


# In[52]:


# concat helps to integrate the columns forms from an object column to numerical column to the pain data frame
columns=["hotel","arrival_date_month","meal","country","market_segment","distribution_channel","reserved_room_type","assigned_room_type","deposit_type","customer_type","reservation_status","reservation_status_date"]
tp = pd.concat([tp, dummies], axis=1)
# drop original columns
tp.drop(columns, axis = 1, inplace=True)


# In[53]:


tp.head()


# In[54]:


tp.tail()


# In[56]:


tp.info()


# # Split the Data

# In[59]:


x = tp.drop("is_canceled", axis=1)
y = tp["is_canceled"]


# In[60]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)


# # Fit the Model

# In[61]:


model=LogisticRegression()
model.fit(x_train, y_train)


# In[ ]:




