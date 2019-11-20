## code from https://www.kaggle.com/jhoward/creating-a-metadata-dataframe-fastai
# coding: utf-8

# It's really handy to have all the DICOM info available in a single DataFrame, so let's create that! In this notebook, we'll just create the DICOM DataFrames. To see how to use them to analyze the competition data, see [this followup notebook](https://www.kaggle.com/jhoward/some-dicom-gotchas-to-be-aware-of-fastai).
#
# First, we'll install the latest versions of pytorch and fastai v2 (not officially released yet) so we can use the fastai medical imaging module.


# In[4]:


import gc
from fastai2.basics import *
from fastai2.medical.imaging import *


# Let's take a look at what files we have in the dataset.

# In[5]:


path = Path('../input/')


# Most lists in fastai v2, including that returned by `Path.ls`, are returned as a [fastai.core.L](http://dev.fast.ai/core.html#L), which has lots of handy methods, such as `attrgot` used here to grab file names.


# In[7]:


path_tst = path/'stage_2_test_images'
fns_tst = path_tst.ls()
len(fns_tst)


# We can grab a file and take a look inside using the `dcmread` method that fastai v2 adds.

# In[9]:


fn = fns_tst[0]
dcm = fn.dcmread()


# # DICOM Meta

# To turn the DICOM file metadata into a DataFrame we can use the `from_dicoms` function that fastai v2 adds. By passing `px_summ=True` summary statistics of the image pixels (mean/min/max/std) will be added to the DataFrame as well (although it takes much longer if you include this, since the image data has to be uncompressed).

# In[15]:


df_tst = pd.DataFrame.from_dicoms(fns_tst, px_summ=True)
df_tst.to_feather('../input/df_tst_st2.fth')
df_tst.head()


# In[13]:


del(df_tst)
gc.collect();
