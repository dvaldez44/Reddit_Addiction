#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import the dataset from sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# import other required libs
import pandas as pd
import numpy as np

# string manipulation libs
import re
import string
import nltk
from nltk.corpus import stopwords

# viz libs
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df = pd.read_csv(".csv")


# In[4]:


df


# In[5]:


import nltk #data pre-processing
from nltk.corpus import stopwords
# nltk.download('stopwords')

stopwords.words("english")[:500] # <-- import the english stopwords


# In[6]:


new_words = ('yeah', 'okay')

for i in new_words:
    stopwords.words('english').append(i)


# In[7]:


def preprocess_text(text: str, remove_stopwords: bool) -> str:
    """This utility function sanitizes a string by:
    - removing links
    - removing special characters
    - removing numbers
    - removing stopwords
    - transforming in lowercase
    - removing excessive whitespaces
    Args:
        text (str): the input text you want to clean
        remove_stopwords (bool): whether or not to remove stopwords
    Returns:
        str: the cleaned text
    """

    # remove links
    text = re.sub(r"http\S+", "", text)
    # remove special chars and numbers
    text = re.sub("[^A-Za-z]+", " ", text)
    # remove stopwords
    if remove_stopwords:
        # 1. tokenize
        tokens = nltk.word_tokenize(text)
        # 2. check if stopword
        tokens = [w for w in tokens if not w.lower() in stopwords.words("english")]
        # 3. join back together
        text = " ".join(tokens)
    # return text in lower case and stripped of whitespaces
    text = text.lower().strip()
    return text


# In[9]:


for doc in df['text']:
    re.sub("[^a-zA-Z]", " ",str(df['text']))


# In[10]:


df.dropna()


# In[11]:


df['cleaned'] = df['text'].apply(lambda x: preprocess_text(x, remove_stopwords=True))


# In[85]:


df


# In[12]:


# initialize the vectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=5, max_df=0.95)
# fit_transform applies TF-IDF to clean texts - we save the array of vectors in X
X = vectorizer.fit_transform(df['cleaned'])


# In[13]:


X.toarray()


# In[88]:


Sum_of_squared_distances = [] #elbow method for optimal clusters
K = range(1,10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(X)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(range(1, 10), Sum_of_squared_distances)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Sum of Squared Differences')
plt.show()


# In[24]:


from sklearn.cluster import KMeans

# initialize kmeans with 3 centroids
kmeans = KMeans(n_clusters=6, random_state=42)
# fit the model
kmeans.fit(X)
# store cluster labels in a variable
clusters = kmeans.labels_


# In[25]:


[c for c in clusters][:10]


# In[26]:


from sklearn.decomposition import PCA

# initialize PCA with 2 components
pca = PCA(n_components=2, random_state=42)
# pass our X to the pca and store the reduced vectors into pca_vecs
pca_vecs = pca.fit_transform(X.toarray())
# save our two dimensions into x0 and x1
x0 = pca_vecs[:, 0]
x1 = pca_vecs[:, 1]


# In[27]:


x0


# In[28]:


x1


# In[29]:


df['cluster'] = clusters
df['x0'] = x0
df['x1'] = x1


# In[30]:


def get_top_keywords(n_terms):
    df = pd.DataFrame(X.todense()).groupby(clusters).mean() # groups the TF-IDF vector by cluster
    terms = vectorizer.get_feature_names() # access tf-idf terms
    for i,r in df.iterrows():
        print('\nCluster {}'.format(i))
        print(','.join([terms[t] for t in np.argsort(r)[-n_terms:]])) # for each row of the dataframe, find the n terms that have the highest tf idf score
            


# In[31]:


get_top_keywords(100)


# In[32]:



# set image size
plt.figure(figsize=(12, 7))
# set a title
plt.title("2022 addiction reddit with K-Means Clustering", fontdict={"fontsize": 18})
# set axes names
plt.xlabel("X0", fontdict={"fontsize": 16})
plt.ylabel("X1", fontdict={"fontsize": 16})
# create scatter plot with seaborn, where hue is the class used to group the data
sns.scatterplot(data=df, x='x0', y='x1', hue='cluster', palette="Set2")
plt.show()


# In[33]:


cluster_map = {0: "Cluster 0", 1: "Cluster 1", 2: "Cluster 2", 3: "Cluster 3", 4: "Cluster 4", 5: "Cluster 5"}
# apply mapping
df['cluster'] = df['cluster'].map(cluster_map)


# In[34]:


df['cluster'].unique()


# In[35]:


df = pd.get_dummies(df, columns=['cluster'], drop_first=False)


# In[36]:


df


# In[37]:


####################VADER########################


# In[38]:


import nltk
import os
import pandas as pd
import numpy as np
import vaderSentiment
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[39]:


sid = SentimentIntensityAnalyzer()


# In[41]:


df['scores'] = df['text'].apply(lambda review:sid.polarity_scores(review))


# In[42]:


df['compound'] = df['scores'].apply(lambda d:d['compound'])


# In[43]:


df


# In[45]:


df.to_csv("D:.csv")


# In[ ]:


#########################within cluster lda######################

