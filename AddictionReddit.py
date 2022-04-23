#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import praw
import pandas as pd
import datetime as dt


# In[ ]:


reddit = praw.Reddit(
    client_id="",
    client_secret="",
    password="",
    user_agent="",
    username="",
)


# In[ ]:


subreddit1 = reddit.subreddit('') #r/
prochoice_subreddit = subreddit1.controversial(limit=1000) # subreddit1.top(limit=100) for top posts in the channel

subreddit2 = reddit.subreddit('') #r/
abortiondebate_subreddit = subreddit2.controversial(limit=1000)


# In[ ]:


dict =        { "title":[],
                "subreddit":[],
                "score":[], 
                "id":[], 
                "url":[], 
                "comms_num": [], 
                "created": [], 
                "body":[],
                "redditor":[]}


# In[ ]:


dict = { "title":[],
                "subreddit":[],
                "score":[], 
                "id":[], 
                "url":[], 
                "comms_num": [], 
                "created": [], 
                "body":[],
                "redditor":[]}


# In[ ]:


for submission in abortion_subreddit:
    dict["title"].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict["score"].append(submission.score)
    dict["id"].append(submission.id)
    dict["url"].append(submission.url)
    dict["comms_num"].append(submission.num_comments)
    dict["created"].append(submission.created)
    dict["body"].append(submission.selftext)
    dict["redditor"].append(submission.author)


# In[ ]:


for submission in abortiondebate_subreddit:
    dict["title"].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict["score"].append(submission.score)
    dict["id"].append(submission.id)
    dict["url"].append(submission.url)
    dict["comms_num"].append(submission.num_comments)
    dict["created"].append(submission.created)
    dict["body"].append(submission.selftext)
    dict["redditor"].append(submission.author)


# In[ ]:


for submission in bebetter_subreddit:
    dict["title"].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict["score"].append(submission.score)
    dict["id"].append(submission.id)
    dict["url"].append(submission.url)
    dict["comms_num"].append(submission.num_comments)
    dict["created"].append(submission.created)
    dict["body"].append(submission.selftext)
    dict["redditor"].append(submission.author)


# In[ ]:


for submission in selfimprovement_subreddit:
    dict["title"].append(submission.title)
    dict['subreddit'].append(submission.subreddit)
    dict["score"].append(submission.score)
    dict["id"].append(submission.id)
    dict["url"].append(submission.url)
    dict["comms_num"].append(submission.num_comments)
    dict["created"].append(submission.created)
    dict["body"].append(submission.selftext)
    dict["redditor"].append(submission.author)


# In[ ]:


df = pd.DataFrame(dict)


# In[ ]:


df


# In[ ]:


df.drop_duplicates(subset=['id'], inplace=True)


# In[ ]:


###Create Dummy Variables###
df['subreddit'].unique()


# In[ ]:


df['subreddit'] = df['subreddit'].apply(lambda x: x.display_name)


# In[ ]:


df['subreddit'].unique()


# In[ ]:


df = pd.get_dummies(df, columns=['subreddit'], drop_first=False)


# In[ ]:


def get_date(created):
    return dt.datetime.fromtimestamp(created)


# In[ ]:


df["created"] = df['created'].apply(get_date)


# In[ ]:


df


# In[ ]:


df.to_csv("D://abortion_reddit3.csv")


# In[ ]:


df.to_pickle("./reddit.pkl")


# In[ ]:


unpickled_df = pd.read_pickle("./reddit.pkl")

