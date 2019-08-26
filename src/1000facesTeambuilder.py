#!/usr/bin/env python
# coding: utf-8

# This module will process all the projects data and serve as a hub for team building. 
# Algorithm flow:
# 1. Preprocess all project-related text
# 2. Create an project embedding matrix for all the matrix
# 3. Use cosine similarity score for ranking top related projects
# 4. Pull people worked in the top K projects, return a list of people (eId)
# 5. If there is not enough people, tab into people's similarity matrix for more people. **

# In[409]:


import pandas as pd
import psycopg2
import matplotlib
import matplotlib.pyplot as plt
import re
import numpy as np
import nltk
import pickle
import seaborn as sns
from autocorrect import spell
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import preprocessing
from nltk.stem import SnowballStemmer
from functools import reduce
from sklearn.metrics.pairwise import cosine_similarity
sns.set(style="white")

plt.rcParams["figure.figsize"] = (10,8)

nltk.download('punkt', 'stopwords')


# # 1. Data processing

# ## 1.1 Load the data

# In[620]:


employee = pd.read_csv('../data/employee_M25.txt',sep = '|',index_col = 'ID')
project = pd.read_csv('../data/project_M25_matched.txt',sep = '|', index_col = 'pID')
projectCategory = pd.read_csv('../data/Project_category_M23.txt',sep = '|')


# In[621]:


print("Shape of employee table is (%d,%d)"%employee.shape)
print("Shape of project table is (%d,%d)"%project.shape)
print("Project categoryes is %s"%projectCategory.values)


# In[622]:


# save employee name -> ID dictionary
employee['fullname'] = employee['FirstName'].str.strip() + " " + employee['LastName'].str.strip()
name2IdDict = dict(zip(employee['fullname'].values, employee.index.values))
with open("employeNameIDDict.pkl",'wb') as infile:
    pickle.dump(name2IdDict,infile)


# In[623]:


employee.head()


# In[624]:


employeeBaseInf = employee[['FirstName','LastName','profilePictrueName']]


# In[625]:


project.head()


# ## 1.2 preprocess the data
# There are two data tables to be processed: 1. employee table 2. Project table

# In[626]:


col2use = ['Degree','YeasInCompany','Skills','PastProjectsID','Hobbies','Hub']


# In[627]:


employee.columns


# In[628]:


# catFeatures = ['Degree','YeasInCompany','Hobbies','Hub']
# for feat in catFeatures:
#     employee[feat] = employee[feat].astype('category')
# employee[catFeatures] = employee[catFeatures].apply(lambda x: x.cat.codes)
# employee.head()


# In[629]:


## Convert categorical features to one hot encoding
catFeatures = ['Degree','YeasInCompany','Hobbies','Hub']
emplyeeCatFeaturesWithoutSkillsProjects = pd.get_dummies(employee[catFeatures], prefix_sep = '_', drop_first=True)
emplyeeCatFeaturesWithoutSkillsProjects.head()


# In[630]:


# massage employee skill data
employee['ID'] = employee.index
emplyeeSkillsdata = pd.melt(employee[['ID','Skills']].set_index('ID')['Skills'].str.split(",", n = -1, expand = True).reset_index(),
              id_vars = ['ID'],
              value_name = 'Skills')\
        .dropna().drop(['variable'], axis = 1)\
        .groupby(['ID','Skills']).agg({'Skills':"count"})\
        .rename(columns={'Skills':'SkillsCount'}).reset_index()
emplyeeSkillsdata['SkillsCount'] = emplyeeSkillsdata['SkillsCount'].astype(np.int64)


# In[631]:


emplyeeSkillsdata.head()


# In[632]:


emplyeeSkillsMatrix = pd.pivot_table(emplyeeSkillsdata, values = 'SkillsCount', 
                          index = 'ID', columns = 'Skills').reset_index()
emplyeeSkillsdataNoNan = emplyeeSkillsMatrix.fillna(0)
emplyeeSkillsdataNoNan.head()


# In[633]:


ax1 = plt.axes()
sns.heatmap(emplyeeSkillsMatrix, ax = ax1)
ax1.set_title('Skill set distribution')


# In[634]:


# massage empolyee project data
projectData = pd.melt(employee[['ID','PastProjectsID']].set_index('ID')['PastProjectsID'].str.split(";", n = -1, expand = True).reset_index(),
              id_vars = ['ID'],
              value_name = 'PastProjectsID')\
        .dropna().drop(['variable'], axis = 1)\
        .groupby(['ID','PastProjectsID']).agg({'PastProjectsID':"count"})\
        .rename(columns={'PastProjectsID':'PastProjectsCount'}).reset_index()
projectData['PastProjectsCount'] = projectData['PastProjectsCount'].astype(np.int64)
projectDataMatrix = pd.pivot_table(projectData, values = 'PastProjectsCount', 
                          index = 'ID', columns = 'PastProjectsID').reset_index()
projectDataMatrixNoNan = projectDataMatrix.fillna(0)
projectDataMatrixNoNan.head()


# In[635]:


ax1 = plt.axes()
sns.heatmap(projectDataMatrix.values)
ax1.set_title('Project participation distribution')


# In[636]:


# Join all the information for employe processed table
list_df = [employeeBaseInf,emplyeeCatFeaturesWithoutSkillsProjects,emplyeeSkillsdataNoNan,projectDataMatrixNoNan ]
employeeReady = reduce(lambda left, right: pd.merge(left, right, on = ['ID'], how = 'inner'), list_df)
employeeReady.shape
employeeReady.head()


# In[637]:


employeeNumericOnly = employeeReady.iloc[:,4:] # selection only numeric data
employeeNumericOnly.head()


# In[638]:


ax1 = plt.axes()
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(employeeNumericOnly,cmap = cmap, ax = ax1)
ax1.set_title('Employee process numerical features')


# In[639]:


corr = np.corrcoef(employeeNumericOnly)
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(20, 18))
cmap = sns.color_palette("RdBu_r", 7)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask,cmap = cmap,  vmax=.5, center=0,
            square=True, linewidths=0.01, cbar_kws={"shrink": .5})
ax.set_title('Employee similarity matrix')


# In[640]:


# save matrix
employeeSimilarityDF = pd.DataFrame(corr, index=employee.ID)
employeeSimilarityDF.columns = list(employee.ID)
employeeSimilarityDF.to_csv('employee_similarity_matrix.csv')
employeeSimilarityDF.head()


# In[641]:


temp = pd.read_csv('employee_similarity_matrix.csv',index_col = 'ID')
temp.index = temp.index.map(str)
# score = temp.loc['12057',:]
# score


# # 2. Return top K similar users
# * Input: User ID, User Similarity Matrix
# * Output: Top K user from high to low

# In[642]:


"""
This function returns a list of employe ID (string) with highest similarity to lowest
"""
def findTopKSimilarEmployee(eId, topK = 'all', eSimilarityMatrixFile='employee_similarity_matrix.csv'):
    import pandas as pd
    import numpy as np
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


# In[643]:


# test
print("The top 5 similar employee for employee %d is: %s"%(5, '|'.join(findTopKSimilarEmployee('12250',5))))


# # 3. Traning - team builder
# This will create a project embedding matrix for all existing projects and take an project text input and returns the top K similar projects. <\br>
# 
# *. 1st, create project embedding and save the embedding object for predcit
# 
# *. 2nd, create project similarity matrix for later recommendation

# In[644]:


project = pd.read_csv('../data/Projects_M23.txt',sep = '|')
projectCategory = pd.read_csv('../data/Project_category_M23.txt',sep = '|')
print(project.shape)
project.head()


# In[645]:


# plot project per category distribution
project['ProjectCategoryId'].hist()
plt.title('Project distribution by category')


# In[646]:


def preprocess(text):
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
    return [stemmer(i) for i in word_tokenize(text)]


# In[647]:


def dataPreparation(data,textCol = 'ProjectDescription', vectMethod = 'countVectorizer'):
    if vectMethod == "countVectorizer":
        vectorizer = CountVectorizer(analyzer=preprocess, min_df = 2, max_df= 0.8)
        vectorizerFile = 'countVectorizer.pkl'
    elif vectMethod == "tfidf":
        vectorizer = TfidfVectorizer(ngram_range=(0,2),analyzer='word',
                               lowercase=True, token_pattern='[a-zA-Z0-9]+',
                               strip_accents='unicode',tokenizer=stem_tokenize)
        vectorizerFile = 'tfidfVectorizer.pkl'
    else:
        raise ValueError("Not accepted tokenizer.")
        
    # start to transform data and save vectorizer for predict
    
    X = vectorizer.fit_transform(data[textCol]).toarray()
    with open(vectorizerFile, 'wb') as fin:
        pickle.dump(vectorizer, fin)
    
    labels = list(set(data.iloc[:,-1])) # the target feature is the last column
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    y = le.fit_transform(project.iloc[:, -1])
    return X,y


# In[648]:


def train(X,y, model = 'NaiveBayes'):
    from sklearn.model_selection import train_test_split
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)
    if model == 'NaiveBayes':
        model = OneVsRestClassifier(MultinomialNB(fit_prior = True, class_prior = None))
    elif model == 'SVC':
        model = LinearSVC()
    elif model == 'LogisticRegress':
        model = LogisticRegression(solver='sag')
    for category in categories:
        print(".. Processing category: {}: {} -- ".format(category,projectCatDict[category]))
        # train model using the x_dtm & y
        model.fit(X_train, (y_train == category).astype('int'))
        prediction = model.predict(X_test)
        print("Test accuracy is {}".format(accuracy_score((y_test == category).astype('int'), prediction)))
    return model


# In[649]:


X, y = dataPreparation(project)
categories = set(project['ProjectCategoryId'])


# In[650]:


projectCatDict = dict(zip(projectCategory['id'], projectCategory['ProjectCategory']))


# In[651]:


train(X,y)


# In[652]:


# save embedded project matrix to disk
embeddedProjectDF = pd.DataFrame(X, index = project['pID'])
embeddedProjectDF.to_csv("embeddedProject.csv")


# In[653]:


a = pd.read_csv('embeddedProject.csv',index_col = 'pID')
a.head()


# # 3.2 Project similarity

# In[654]:


# save matrix
corrProject = np.corrcoef(embeddedProjectDF)
projectSimilarityDF = pd.DataFrame(corrProject, index=project.pID)
projectSimilarityDF.columns = list(project.pID)
projectSimilarityDF.to_csv('project_similarity_matrix.csv')
projectSimilarityDF.head()


# In[655]:


sns.heatmap(projectSimilarityDF)


# # 3.3 production
# * 1. Return top K similar project from existing projects table

# In[656]:


"""
This function returns a list of employe ID (string) with highest similarity to lowest
"""
def findTopKSimilarProject(eId, topK = 'all', eSimilarityMatrixFile='project_similarity_matrix.csv'):
    import pandas as pd
    import numpy as np
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


# In[657]:


print("The top 5 similar projects for project %d is: %s"%(5, '|'.join(findTopKSimilarProject('2',5))))


# In[658]:


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

def cosine_similarity_one_vs_array(vector,matrix):
    return ( np.sum(vector*matrix,axis=1) / 
            ( np.sqrt(np.sum(matrix**2,axis=1)) * np.sqrt(np.sum(vector**2)) ) )[::-1]


# In[659]:


a = project['ProjectDescription'][1]
a


# In[660]:


b = predictTopKProject(a)


# In[661]:


b


# In[ ]:





# * 2. Return a list of employee ID given a project description
# 
# Based on top K similar project, get all employee ID worked on these project, return top K employee. For each category, return at lease one employee.

# * 1. grab all employee worked on top K project
# * 2. for each category, find top K employees
# * 3. If not enough, tap into employee similarity matrix and retrieve other similar employees to the top 1. Exclude already picked one.
# 

# In[662]:


"""
input: a List of pIds
Return employee ID list given a project ID: string
Return a list.
"""
def getEmployeeIDForPid(pId, projectTablePath = r'../data/Projects_M23.txt'):
    if len(pId) == 0:
        return None
    project = pd.read_csv(projectTablePath,sep = '|', index_col = 'pID')
    project.index = project.index.map(str)
    eNames = []
    for ids in pId:
        eNames = eNames + project.loc[ids,:]['ProjectTeam'].split(',')
    # load name 2 ID dictionary if it's name, remove this with new data using eId in project data
    with open('employeNameIDDict.pkl','rb') as fin:
        name2idDict = pickle.load(fin)
    print(eNames)
    eIds = [name2idDict[name] for name in eNames]
    return list(eIds)


# In[663]:


eids = getEmployeeIDForPid(['1','2'])
print(eids)


# In[664]:


# get the hub
ehubs = [str(employee.loc[employee['ID'] == ids,:]['Hub'].values) for ids in eids]
ehubs


# * 3. logic to get the top K employee for each hub
# 
#     Hub is also matched to project category. employee similarity table will be used. The logic here is to: scan through all the employees and get the first couple employee at teach hub, if not enough then tap into employee similarity table to get the rest.

# In[ ]:





# In[ ]:




