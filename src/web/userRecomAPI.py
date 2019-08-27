#!/usr/bin/env python
# coding: utf-8

# In[5]:


"""
This function returns a list of employe ID (string) with highest similarity to lowest
"""
import pandas as pd
import numpy as np
def findTopKSimilarEmployee(eId, topK = 5, eSimilarityMatrixFile='employee_similarity_matrix.csv'):
    matrix = pd.read_csv(eSimilarityMatrixFile, index_col = 'ID')
    matrix.index = matrix.index.map(str)
    # retrieve ranked employ based on e-e similarity
    sim = matrix.loc[str(eId),:]
    sortedSim = sim.sort_values(ascending=False)
    # return a list of employee id from high to low
    eIdSorted = sortedSim.iloc[1:].index.tolist()
    if(topK == 'all'):
        return eIdSorted
    else:
        return eIdSorted[0:topK]


# In[6]:


#print("The top 5 similar employee for employee %d is: %s"%(5, '|'.join(findTopKSimilarEmployee('12070',5))))


# In[ ]:
