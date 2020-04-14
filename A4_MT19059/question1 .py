#!/usr/bin/env python
# coding: utf-8

# In[2]:


#importing all libraries 
import os
from os import listdir
import nltk
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import string
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import math
from collections import Counter
import numpy as np
import statistics
import operator
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
from num2words import num2words


# In[4]:


# defined a function to retrieve names of file from 20_newsgroup
def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


path="20_newsgroups"
folder_path=getListOfFiles(path)
#print((folder_path))
#print("done")


print(len(folder_path))

for i in range (0,5000,1000):
    print(folder_path[i])
print(len(folder_path))


# In[5]:


#defined a function for all the preprocessing
def convert_tolowercase(data):
    return (data.lower())


def regextokenizer_func(data):
    #print(type(data))
    tokenizer=RegexpTokenizer(r'\w+')
    data=tokenizer.tokenize(data)
    return data

def remove_stopwords(data):
    stop_words=set(stopwords.words('english'))
    result=[i for i in data if not i in stop_words]
    return result


def lemmatization_func(data):
    lemmatizer=WordNetLemmatizer()
    result=[]
    for word in data:
        result.append(lemmatizer.lemmatize(word))
    return result

def stemming_func(data):
    stemmer=PorterStemmer()
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(data)
    data_new=""
    for i in tokens:
        data_new+=" "+stemmer.stem(i)
    return data_new
def convert_numbers(k):
    for i in range(len(k)):
        try:
            k[i] = num2words(int(k[i]))
        except:
            pass
    return k


# In[6]:


#function for counting frequency of word in DF dictionary
def doc_freq(word):
    c = 0
    try:
        c = DF[word]
    except:
        pass
    return c

#DOT product calculation
def cosine_dot(a,b):
    if (np.linalg.norm(a)==0 or np.linalg.norm(b)==0):
        return 0;
    else:
        temp=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return temp



# In[7]:


#opened file and applied all the preprocessing steps and stored in dictionary named body_list
total_doc_id=[]
body_list=[]
doc=0
for i in range (0,len(folder_path),1):
    text=open(folder_path[i],encoding='utf-8',errors='ignore').read().strip()
    text=convert_tolowercase(text)
    text=stemming_func(text)
    text=regextokenizer_func(text)
    text=lemmatization_func(text)
    text=remove_stopwords(text)
    body_list.append(text)
    
print(len(body_list))


# In[8]:


#calculated occurence of words in document DF
DF={}
cnt=0
for tokens in body_list:
    for word in (tokens):
        try:
            DF[word].add(cnt)
        except:
            DF[word]={cnt}
    cnt+=1
for i in DF:
    DF[i]=len(DF[i])
print(len(DF))


# In[9]:


#calculated TF_IDF 
N=len(body_list)
tf_idf={}
doc=0
for tokens in body_list:
    counter=Counter(tokens)
    word_len=len(counter)
    for word in np.unique(tokens):
        tf=counter[word]/word_len
        df=doc_freq(word)
        idf=np.log(N+1)/df+1
        tf_idf[doc,word]=tf*idf
    doc+=1
print(len(tf_idf))


# In[10]:


#Created matrix for calculation of cosine Sim
total_vocab_size = len(DF)
total_vocab = [x for x in DF]
D = np.zeros((N, total_vocab_size),dtype='float16')
for i in tf_idf:
    try:
        ind = total_vocab.index(i[1])
        D[i[0]][ind] = tf_idf[i]
    except:
        pass

D.shape


# In[11]:


#Defined a query maatrix 
def query_matrix_func(query):
    mat=np.zeros((len(total_vocab)))
    counter1=Counter(query)
    word_len1=len(counter1)
    query_tf_idf={}
    for word in np.unique(query):
        tf=counter1[word]/word_len1
        df=doc_freq(word)
        idf=np.log(N+1)/(df+1)
        try:
            ind=total_vocab.index(word)
            mat[ind]=tf*idf
        except:
            pass
    return mat


# In[12]:


#function to calculate cosine scores
def cosine_similarity_func(query_mat,k):
    cos_sim=[]
    for d in D:
        cos_sim.append(cosine_dot(query_mat,d))
    
    outpt=np.array(cos_sim).argsort()[-k:][::-1]
    #print(cos_sim)
    return list(outpt)


# In[15]:


#main program
#query="substantiate my statement"
#query="I don't know what kind of machine you want it for, but the program Radiance comes with 'C' source"
#query="I claim that I can substantiate my statement that Perot was investigating him."
query="Pretty good opinions on biochemistry machines"
k=100
#query=input("Enter phrasal query")
#k=input("enter k")
orig_query=query
query=convert_tolowercase(query)
query=regextokenizer_func(query)
query=lemmatization_func(query)
query=remove_stopwords(query)
print((query))
query_len=len(query)
print(query_len)
print(orig_query)
query_mat=query_matrix_func(query)
retrieved_docs=cosine_similarity_func(query_mat,k)
print("Retrieved Docs are :- ",retrieved_docs)


# In[ ]:





# In[ ]:




