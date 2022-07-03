#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd
import re
import string
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')
import plotly as py
import cufflinks as cf
from plotly.offline import iplot


# # Data Pre - Processing

# In[5]:


#to display commplete reviews
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)
pd.set_option('display.max_colwidth',None)


# In[7]:


#Importing dataset
data = pd.read_csv('C://Users//swapn//Downloads//archive (1)//ECO.csv')


# In[11]:


data.shape
data.reset_index(drop = True, inplace = True)


# In[12]:


data.head()


# In[13]:


data.info()


# In[15]:


data.isnull().sum()


# In[16]:


data = data.astype({'Reviews':str},errors = "raise")


# In[18]:


data['cleaned'] = data['Reviews'].apply(lambda x:x.lower())


# In[19]:


data.head()


# In[20]:


#Removing numbers or numeric values

data['cleaned'] = data['cleaned'].apply(lambda x: re.sub('\w*\d\w*','',x))


# In[22]:


#removing irrelevant reviews 
data['cleaned'] = data['cleaned'].str.replace('the media could not be loaded.','')


# In[23]:


#removing punctuations
data['cleaned'] = data['cleaned'].apply(lambda x :re.sub('[%s]' % re.escape(string.punctuation),'',x))


# In[24]:


#Removing all non-english text values by encoding it to ASCII

data['cleaned']= data['cleaned'].str.encode('ascii','ignore').str.decode('ascii')


# In[25]:


#removing all \n 
data['cleaned'] = data['cleaned'].replace(r'\n',' ',regex = True)
data


# # EDA

# In[29]:


pip install textblob


# In[30]:


#Calculating the polarity scores using TextBlob
from textblob import TextBlob
data['polarity'] = (round(data['cleaned'].apply(lambda x: TextBlob(x).sentiment.polarity),4))


# In[31]:


data


# In[42]:


pip install chart_studio


# In[43]:


data['polarity'].iplot(kind='hist',
                         xTitle='Polarity',yTitle='Count',title="Sentiment Polarity Distribution")


# In[50]:


pol= []

for i in data.polarity:
    if i < 0:
        pol.append('neagtive')
    elif i > 0:
        pol.append('positive')
    else:
        pol.append('neutral')
    
data['division'] = pol


# In[51]:


data.division.value_counts().plot()


# In[52]:


#Display 10 highly polarized random reviews

print("10 highly polarized random reviews")
for index,review in enumerate(data.iloc[data['polarity'].sort_values(ascending = False)[:10].index]['cleaned']):
    print ('Review {}:\n'.format(index+1),review)


# In[54]:


#Displaying 10 Random Lowest polarized reviews

print("10 Random Reviews with the Lowest Polarity")
for index,review in enumerate(data.iloc[data['polarity'].sort_values(ascending=True)[:10].index]['cleaned']):
    print('Review {}:\n'.format(index+1),review)

