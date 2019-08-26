#!/usr/bin/env python
# coding: utf-8

# In[1]:


"""
This function returns a list of employe ID (string) with highest similarity to lowest
"""
import pandas as pd
import numpy as np
def findTopKSimilarProject(eId, topK = 'all', eSimilarityMatrixFile='project_similarity_matrix.csv'):
    matrix = pd.read_csv(eSimilarityMatrixFile, index_col = 'pID')
    matrix.index = matrix.index.map(str)
    # retrieve ranked employ based on e-e similarity
    sim = matrix.loc[str(eId),:]
    sortedSim = sim.sort_values(ascending=False)
    # return a list of employee id from high to low 
    pIdSorted = sortedSim.iloc[1:].index.tolist()
    if(topK == 'all'):
        return pIdSorted
    else:
        return pIdSorted[0:topK]


# In[2]:


print("The top 5 similar projects for project %d is: %s"%(5, '|'.join(findTopKSimilarProject('2',5))))


# In[ ]:




