#!/usr/bin/env python
# coding: utf-8

# # This is the api to take the input from UI and return a list of EmployeeID
import pandas as pd
import numpy as np
import re
import uuid
import nltk
import pickle
import psycopg2
from autocorrect import spell
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
nltk.download('punkt', 'stopwords')

def preprocess(text):
    from nltk.stem import PorterStemmer
    # remove non-alphabetic characters
    textAlphabetic = re.sub('[^A-Za-z]', ' ', text)
    # make all words lower case
    textLower = textAlphabetic.lower()
    # remove stop words
    tokenized_text = word_tokenize(textLower)
    for word in tokenized_text:
        if word in stopwords.words('english'):
            tokenized_text.remove(word)
    # stemming
    stemmer = PorterStemmer()
    for i in range(len(tokenized_text)):
        tokenized_text[i] = stemmer.stem(spell(tokenized_text[i]))

    return tokenized_text

def stem_tokenize(text):
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    return [stemmer.stem(i) for i in word_tokenize(text)]

"""
input: text
Return project ID list given a project ID: string
Return a list.
"""
def predictTopKProject(text, topK = 5, vectorizer = 'countVectorizer.pkl',
            embeddedProject = 'embeddedProject.csv',
            employeeSimMatrix = 'employee_similarity_matrix.csv',
           projectSimMatrix = 'project_similarity_matrix.csv'):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
    # load project embedding
    embeddedProject = pd.read_csv('embeddedProject.csv',index_col = 'pID')
    # preprocess
    text_preprocessed = preprocess(text)
    with open(vectorizer, 'rb') as f1:
        vectorizer = pickle.load(f1)
    
    text_vectorized = vectorizer.transform([text]).toarray()
    score = []
    for i in range(embeddedProject.shape[0]):
        prior_project = embeddedProject.iloc[i,:]
        score.append(np.corrcoef(text_vectorized, prior_project)[0][1])
    mylist = sorted(enumerate(score), key=lambda x: -x[1])
    idx = [l[0] for l in mylist]
    score_sorted = [l[1] for l in mylist] 
    pIds = [embeddedProject.index[ii] for ii in idx]
    if topK == 'all':
        return pIds
    else:
        return pIds[0:topK]

"""
input: a List of pIds
Return employee ID list given a project ID: string
Return a list.
"""
def getEmployeeIDForPid(pId, projectTablePath = r'../data/project_M25_matched.txt'):
    if len(pId) == 0:
        return None
    project = pd.read_csv(projectTablePath,sep = '|', index_col = 'pID')
    project.index = project.index.map(str)
    eIds = []
    for ids in pId:
        eIds = eIds + project.loc[str(ids)]['ProjectTeam'].split(',')
    return list(eIds)


def buildTeam(text):
    # get topK project
    projectIdList = predictTopKProject(text)
    # get topK employee ID list
    if len(projectIdList) > 0: 
        employeeIdList = getEmployeeIDForPid(projectIdList)
        return employeeIdList
    else:
        return []

if __name__ == "__main__":
    import sys
    inputString = 'Tax, Payment and Compliance Solution meeting'
    try:
        inputString = sys.argv[1]
    except:
        inputString = ''
    eIdList = buildTeam(inputString)
#    eIdList = buildTeam()
    print(eIdList)
    print("test working fine")







