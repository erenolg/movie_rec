#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


# In[2]:


df = pd.read_csv("n_movies.csv")
pd.options.display.max_columns = 5000


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.iloc[df.vote_average.sort_values(ascending=False)]


# In[6]:


for_use = df[["genres", "keywords", "tagline", "cast", "director"]]


# In[7]:


data = for_use
data.head()


# In[8]:


for i in data.columns:
    data[i] = data[i].str.lower()


# In[9]:


data.head()


# In[10]:


data.isna().sum()


# In[11]:


data.fillna("", inplace=True)


# In[12]:


data.isna().sum()


# In[13]:


words = data["genres"] + " " + data["keywords"] + " " + data["tagline"] + " " + data["cast"] + " " + data["director"]


# In[14]:


words


# In[15]:


vector = TfidfVectorizer()


# In[16]:


vectors = vector.fit_transform(words)


# In[17]:


similarity = cosine_similarity(vectors)


# In[18]:


sim = pd.DataFrame(similarity, index = df.title, columns = df.title)


# In[19]:


sim.head()


# In[20]:


def recommend(movie):
    score = sim[movie].sort_values(ascending=False)
    return score


# In[21]:


recommend("Spider-Man 3")


# In[22]:


movies  = ["Spider-Man 3", "The Specials", "Daddy's Home"]
new = pd.DataFrame()
for i in movies:
    new = new.append(recommend(i), ignore_index = True)


# In[23]:


new.sum().sort_values(ascending=False)


# In[ ]:




