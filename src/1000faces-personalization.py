#!/usr/bin/env python
# coding: utf-8

# # Personalization
# #### This module will handle collaborative filter based recommendation for individual employees, this will produce two outputs: 1. recommended project 2. similar person to check out

# In[1]:


import pandas as pd
import numpy as np
import time
import turicreate as tc
from sklearn.model_selection import train_test_split
import sys


# In[60]:


employee = pd.read_csv('../data/employee_M23.txt',sep = '|',index_col = 'ID')
project = pd.read_csv('../data/project_M25_matched.txt',sep = '|', index_col = 'pID')
projectCategory = pd.read_csv('../data/Project_category_M23.txt',sep = '|')


# In[62]:


# massage empolyee project data
employee['ID'] = employee.index
projectData = pd.melt(employee[['ID','PastProjectsID']].set_index('ID')['PastProjectsID'].str.split(";", n = -1, expand = True).reset_index(),
              id_vars = ['ID'],
              value_name = 'PastProjectsID')\
        .dropna().drop(['variable'], axis = 1)\
        .groupby(['ID','PastProjectsID']).agg({'PastProjectsID':"count"})\
        .rename(columns={'PastProjectsID':'PastProjectsCount'}).reset_index()
projectData['PastProjectsCount'] = projectData['PastProjectsCount'].astype(np.int64)
projectData.head()


# In[63]:


def split_data(data):
    train, test = train_test_split(data, test_size = 0.2)
    train_data = tc.SFrame(train)
    test_data = tc.SFrame(test)
    return train_data, test_data


# In[64]:


train_data, test_data = split_data(projectData)


# # Define model using Turicreate library
# ## Baseline: most popular items

# In[65]:


user_id = 'ID'
item_id = 'PastProjectsID'
users_to_recommend = list(projectData['ID'])
n_rec = 10
n_display = 30


# In[66]:


# Turicreate is a great library
def model(train_data, name, user_id, item_id, target,
         users_to_recommend, n_rec, n_display):
    if name == 'popularity':
        model = tc.popularity_recommender.create(train_data,
                                                user_id = user_id,
                                                item_id = item_id,
                                                target = target)
    elif name == 'cosine':
        model = tc.item_similarity_recommender.create(train_data,
                                                     user_id = user_id,
                                                     item_id = item_id,
                                                     target = target,
                                                     similarity_type = 'cosine')
    elif name == 'pearson':
        model = tc.item_similarity_recommender.create(train_data,
                                                     user_id = user_id,
                                                     item_id = item_id,
                                                     target = target,
                                                     similarity_type = 'pearson')
#     recom = model.recommend(users = users_to_recommend, k = n_rec)
    return model


# # Popularity model as baseline

# In[67]:


name = 'popularity'
target = 'PastProjectsCount'
popularity_model = model(train_data, name, user_id, item_id, target,
                  users_to_recommend, n_rec, n_display)


# In[68]:


popularity_model.recommend(users_to_recommend, k = n_rec).print_rows(30)


# # use collaborative filter
# 

# In[69]:


name = 'pearson'
target = 'PastProjectsCount'
pear = model(train_data, name, user_id, item_id, target,
           users_to_recommend, n_rec, n_display)


# In[70]:


pear.recommend(users_to_recommend, k = n_rec).print_rows(n_display)


# In[71]:


models_w_dummy = [popularity_model, pear]

names_w_dummy = ['Popularity Model on Purchase Counts', 'Pearson Similarity on Purchase Counts']
eval_counts = tc.recommender.util.compare_models(test_data,
                                                models_w_dummy, model_names=names_w_dummy)


# # final model

# In[72]:


final_model = tc.item_similarity_recommender.create(tc.SFrame(projectData), 
                                            user_id=user_id, 
                                            item_id=item_id, 
                                            target='PastProjectsCount', similarity_type='cosine')
recom = final_model.recommend(users=users_to_recommend, k=n_rec)
recom.print_rows(n_display)
recom.to_dataframe().head()


# In[73]:


def create_output(model, users_to_recommend, n_rec, print_csv=True):
    recomendation = model.recommend(users=users_to_recommend, k=n_rec)
    df_rec = recomendation.to_dataframe()
    df_rec['recommendedProjects'] = df_rec.groupby([user_id])[item_id]         .transform(lambda x: '|'.join(x.astype(str)))
    df_output = df_rec[['ID', 'recommendedProjects']].drop_duplicates()         .sort_values('ID').set_index('ID')
    if print_csv:
        df_output.to_csv('../output/option1_recommendation.csv')
        print("An output file can be found in 'output' folder with name 'option1_recommendation.csv'")
    return df_output


# In[74]:


df_output = create_output(pear, users_to_recommend, n_rec, print_csv=True)
print(df_output.shape)
df_output.head()


# In[ ]:





# In[ ]:





# In[ ]:




