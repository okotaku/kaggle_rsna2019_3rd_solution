
# coding: utf-8

# In[1]:


import os
import pandas as pd
dir_csv = '../input'


# In[2]:


train = pd.read_csv(os.path.join(dir_csv, 'stage_2_train.csv'))


# In[3]:


train.head()


# In[4]:


test.head()


# In[5]:


train[['ID', 'Image', 'Diagnosis']] = train['ID'].str.split('_', expand=True)
train = train[['Image', 'Diagnosis', 'Label']]
train.drop_duplicates(inplace=True)
train = train.pivot(index='Image', columns='Diagnosis', values='Label').reset_index()
train['Image'] = 'ID_' + train['Image']
train.head()


# In[6]:


train.to_csv("../input/rsna_train.csv", index=False)

