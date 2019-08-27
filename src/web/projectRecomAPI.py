#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
This function returns a list of employe ID (string) with highest similarity to lowest
"""
import pandas as pd
import numpy as np
def findTopKSimilarProject(pId, topK = 5, pSimilarityMatrixFile='project_similarity_matrix.csv'):
    matrix = pd.read_csv(pSimilarityMatrixFile, index_col = 'pID')
    matrix.index = matrix.index.map(str)
    # retrieve ranked employ based on e-e similarity
    sim = matrix.loc[str(pId),:]
    sortedSim = sim.sort_values(ascending=False)
    # return a list of employee id from high to low
    pIdSorted = sortedSim.iloc[1:].index.tolist()
    if(topK == 'all'):
        return pIdSorted
    else:
        return pIdSorted[0:topK]


# In[6]:


# print("The top 5 similar projects for project %d is: %s"%(5, '|'.join(findTopKSimilarProject('2',5))))


# In[18]:


def findTopKProjectsPerUser(eId, topK=5, employeTable = 'employee_M25.txt'):
    matrix = pd.read_csv(employeTable, sep = '|',index_col = 'ID')
    matrix.index = matrix.index.map(str)
    projectList = matrix.loc[str(eId)]['PastProjectsID'].split(';')
    result = []
    for project in projectList:
        result = result + findTopKSimilarProject('2',3)

    if(topK == 'all'):
        return result
    else:
        return result[0:topK]


# In[29]:


#findTopKProjectsPerUser('12126')


# In[21]:


# a = employee.head().to_json()


# In[23]:


# import json


# In[28]:


# json.dumps([{"name":'liang',"lastnaem":'kuang'}, {"name":'oliva', "last":'kkk'}])


# In[ ]:
