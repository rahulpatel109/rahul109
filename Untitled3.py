#!/usr/bin/env python
# coding: utf-8

# In[17]:


import pandas as pd
import numpy as np


# In[18]:


df = pd.read_csv("universities.csv")
df


# In[19]:


df[(df["GradRate"]>=95)]


# In[26]:


df[(df["GradRate"]>=80) & (df["SFRatio"]<=12)]


# In[ ]:





# In[ ]:




