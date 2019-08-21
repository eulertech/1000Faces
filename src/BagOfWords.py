#!/usr/bin/env python
# coding: utf-8

# # API requirement
# # 1. Training 

# In[223]:


# Import libraries
import pandas as pd
import re
import nltk
import pickle
from autocorrect import spell
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn import preprocessing
from nltk.stem import SnowballStemmer

nltk.download('punkt', 'stopwords')


# In[100]:


dataset = pd.read_csv('../data/spam.csv', encoding="ISO-8859-1");
dataset.columns = ['target','sms','v3','v4','v5']


# In[101]:


dataset.head()


# ## Preprocessing

# In[159]:


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


# In[160]:


a = preprocess("this is 0000 2222333 skdf aa2222 07099833605")
a


# In[167]:


labels = list(set(dataset.iloc[:, 0]))
labels


# In[224]:


# tf-idf vectorizer
def stem_tokenize(text):
    return [stemmer(i) for i in word_tokenize(text)]

tfVectorizer = TfidfVectorizer(ngram_range=(0,2),analyzer='word',
                               lowercase=True, token_pattern='[a-zA-Z0-9]+',
                               strip_accents='unicode',tokenizer=stem_tokenize)
with open('tfidv_vectorizer.pk', 'wb') as fin:
    pickle.dump(tfVectorizer, fin)


# In[204]:


# creating the feature matrix using bag of words
# vectorizer = CountVectorizer(max_features=10000,decode_error="replace", 
#                              stop_words='english', min_df=2, max_df=0.7)
vectorizer = CountVectorizer(analyzer=preprocess)
# vectorizer = TfidfVectorizer()
cvf = vectorizer.fit(dataset.iloc[0:100,]['sms_processed'])
X = cvf.transform(dataset.iloc[0:100,]['sms_processed']).toarray()
print(vectorizer.get_feature_names())
le = preprocessing.LabelEncoder()
le.fit(labels)
y = le.fit_transform(dataset.iloc[0:100, 0])


# In[205]:


# split train and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)


# In[206]:


# Naive Bayes 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predict Class
y_pred = classifier.predict(X_test)

# Accuracy 
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
accuracy


# In[241]:


with open('transformer.pkl','wb') as f1:
    pickle.dump(cvf, f1)

with open("simpleModel.pkl","wb") as f2:
    pickle.dump(classifier, f2)


# In[240]:


with open('tfidf_transformer.pkl','wb') as f1:
    pickle.dump(tfVectorizer, f1)
with open('tfidf_transformer.pkl','rb') as f1:
    xx = pickle.load(f1)


# In[217]:


transformed = cvf.transform(["go jure point crazi avail in sugi n great world"]).toarray()


# In[218]:


testScore = classifier.predict(transformed)
testScore


# # 2. Production

# In[245]:


def predict(text,modelPath = "simpleModel.pkl", vectorizer = 'transformer.pkl'):
    from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
    # preprocess
    text_preprocessed = preprocess(text)
    with open(vectorizer, 'rb') as f1:
        vectorizer = pickle.load(f1)
    with open(modelPath, "rb") as f2:
        clf = pickle.load(f2)
    score = clf.predict(vectorizer.transform([text]).toarray())
    
    return score
    


# In[246]:


predict("go jure point crazi avail in sugi n great world")


# In[ ]:




