#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import pickle
import time
from sklearn.metrics.pairwise import cosine_similarity


# In[2]:


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


# In[3]:


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
unique_labels = ['comp.graphics', 'rec.sport.hockey', 'sci.med', 'sci.space', 'talk.politics.misc']


# In[4]:


total_doc_id=[]
body_list=[]
labels=[]
doc=0
j=0
for i in range (0,len(folder_path),1):
    text=open(folder_path[i],encoding='utf-8',errors='ignore').read().strip()
    text=convert_tolowercase(text)
    text=stemming_func(text)
    text=regextokenizer_func(text)
    text=lemmatization_func(text)
    text=remove_stopwords(text)
    body_list.append(text)
    labels.append(unique_labels[j])
    if((i+1)%1000==0):
        j+=1
    
    
print(len(body_list))
print(len(labels))
docs_pd=pd.DataFrame([body_list,labels]).T
docs_pd.to_pickle("docs_pd")
print("done")


# In[56]:


def accuaracy(actual,predicted):
    correct=0
    for i in range(len(actual)):
        if actual[i]==predicted[i]:
            correct+=1
    acc=float(correct/len(actual))*100.0
    print(acc)
    return acc


# In[53]:


from sklearn.metrics import confusion_matrix
def confusion_matrix_func(actual,predicted):
    print(confusion_matrix(actual,predicted))
    return (confusion_matrix(actual,predicted))


# In[7]:


def TF_frequency(m):
    
    class_frequency = {}
    class_count = {}
    counter = 0
    for i in unique_labels:
        current_count = len(Counter(m[i]))
        class_count[i] = current_count
        counter += current_count
        ll = Counter(m[i])
        for j in ll:
            class_frequency[i, j] = ll[j]
        
    return class_frequency,class_count


# In[72]:


docs_pd=pd.read_pickle("docs_pd")
df=docs_pd.sample(frac=1.0)
train=docs_pd.sample(frac=0.8)
test=df.drop(train.index)
train=train.reset_index(drop=True)
test=test.reset_index(drop=True)
print(len(train))
print(len(test))
train_prior=Counter(train[1])
print(train_prior)
class_dict={}
for i in range(train.shape[0]):
    try:
        class_dict[train[1][i]]+=train[0][i]
    except:
        class_dict[train[1][i]]=train[0][i]
#print(class_dict['sci.space'])      
class_frequecy,class_count=TF_frequency(class_dict)
unique_words = set()
for i in class_dict:
    unique_words = unique_words | set(class_dict[i])
unique_words_count = len(unique_words)
print(len(class_frequecy))
print(class_count)
print(len(class_dict['sci.med']))


# In[9]:


def Naive_Bayes(class_frequecy,class_count):
    actual=[]
    predicted=[]
    for i in range(test.shape[0]):
        classes_word_probablity=[]
        actual.append(test[1][i])
        for labels in unique_labels:
            word_prob=0;
            for word in test[0][i]:
                try:
                    temp1,temp2=class_frequecy[labels,word],class_count[labels]
                except:
                    temp1,temp2=0,class_count[labels]
                temp3=(temp1+1)/(temp2+unique_words_count)
                word_prob+=np.log(temp3)
            classes_word_probablity.append(word_prob)
        predicted.append(unique_labels[np.argmax(classes_word_probablity)])
        
    return actual,predicted       
                
                


# In[10]:


actual,predicted=Naive_Bayes(class_frequecy,class_count)
#print(predicted)
#print(actual)

acc=accuaracy(actual,predicted)
print(acc)


# In[23]:


#TF_IDF FEATURE SELECTION
def FEATURE_SELECTION_TFIDF(train,test,percent):
    corpus=[]
    for i in class_dict:
        corpus+=class_dict[i]
    #print(len(corpus))
    DF={}
    n=0
    for i in class_dict:
        for word in class_dict[i]:
            try:
                DF[word].add(n)
            except:
                DF[word]={n}
        n+=1
    for word in DF:
        DF[word]=len(DF[word])
    #print(len(DF))
    tf_idf={}
    N=train.shape[0]
    counter=Counter(corpus)
    word_count=len(corpus)
    for token in set(corpus):
        tf=counter[token]/word_count
        try:
            df=DF[token]
        except:
            pass
        idf=np.log((N+1)/(df+1))
        tf_idf[token]=idf*tf
    sorted_tf_idf=sorted(tf_idf.items(),key=operator.itemgetter(1),reverse=True)
    #print((sorted_tf_idf))
    sorted_tf_idf=sorted_tf_idf[:int(len(sorted_tf_idf)*(percent/100.0))]
    #print(len(sorted_tf_idf))
    sorted_tf_idf=[i[0] for i in sorted_tf_idf]
    #print(sorted_tf_idf)
    return sorted_tf_idf
    
    


# In[12]:


def storing_terms_TFIDF_func(sorted_tf_idf):
    class_count={}
    class_frequency={}
    #print(unique_labels)
    for i in unique_labels:
        ll=Counter(class_dict[i])
        for word in sorted_tf_idf:
            class_frequency[i,word]=ll[word]
            try:
                class_count[i]+=ll[word]
            except:
                class_count[i]=ll[word]
    return class_frequecy,class_count


# In[13]:


def storing_terms_MI_func(MI_score):
    class_frequecy={}
    class_count={}
    for label in unique_labels:
        ll=Counter(MI_score[label])
        for word in MI_score[label]:
            class_frequecy[label,word]=ll[word]
            try:
                class_count[label]+=ll[word]
            except:
                class_count[label]=ll[word]
    return class_frequecy,class_count


# In[14]:


sorted_feature_list=FEATURE_SELECTION_TFIDF(train,test,10)
class_frequecy1,class_count1=storing_terms_TFIDF_func(sorted_feature_list)
actual1,predicted1=Naive_Bayes(class_frequecy1,class_count1)

acc=accuaracy(actual1,predicted1)
print(acc)


# In[15]:


def calculateMI(N11,N10,N01,N00):
    N=N11+N10+N01+N00
    N1_dot=N11+N10
    N0_dot=N00+N01
    Ndot_1=N01+N11
    Ndot_0=N00+N10
    MI=0
    try:
        MI=(N11/N)*np.log2((N*N11)/(N1_dot*Ndot_1))+(N01/N)*np.log2((N*N01)/(N0_dot*Ndot_1))+(N10/N)*np.log2((N*N10)/(N1_dot*Ndot_0))+(N00/N)*np.log2((N*N00)/(N0_dot*Ndot_0))
    except:
        pass
    return float(MI)


# In[61]:


def FEATURE_SELECTION_MUTUAL_INFORMATION(train,test,percent):
    #print(train.head())
    #print(test.head())
    N=train.shape[0]
    class_count_docs=Counter(train[1])
    #print(class_count_docs)
    class_docs_words={}
    MI_score={}
    for i in unique_labels:
        class_docs_words[i]={}
        MI_score[i]={}
    for i in range(train.shape[0]):
        class_docs_words[train[1][i]][i]=train[0][i]
    DF={}
    
    for labels in class_docs_words:
        cnt=0;
        for docs in class_docs_words[labels]:
            for word in class_docs_words[labels][docs]:
                try:
                    DF[labels,word].add(cnt)
                except:
                    DF[labels,word]={cnt}
            cnt+=1
    
    for labels,word in DF:
        DF[labels,word]=len(DF[labels,word])
   
    
    for labels,word in DF:
        N11=DF[labels,word]
        N01=class_count_docs[labels]-N11
        N10=0
        for not_labels in unique_labels:
            if(not_labels!=labels):
                try:
                    N10+=DF[not_labels,word]
                except:
                    N10+=0
        N00=N-(N11+N10+N01)
        MI_score[labels][word]=calculateMI(N11,N10,N01,N00)
    for label in unique_labels:
        MI_score[label]=sorted(MI_score[label].items(),key=operator.itemgetter(1),reverse=True)
        
        MI_score[label]=[i[0] for i in MI_score[label]]
        MI_score[label]=MI_score[label][0:int(len(MI_score[label])*(percent/100.0))]
    
    #print(MI_score)
    intersection_list=[]
    for label in unique_labels:
        intersection_list+=MI_score[label]
    intersection_list=set(intersection_list)
    return intersection_list
    


# In[17]:


MI_score=FEATURE_SELECTION_MUTUAL_INFORMATION(train,test,10)
class_frequecy2,class_count2=storing_terms_MI_func(MI_score)  
actual2,predicted2=Naive_Bayes(class_frequecy2,class_count2)
acc2=accuaracy(actual2,predicted2)
print(acc2)


# In[18]:


def preprocessing_for_KNN_func(train,sorted_features_list):
    N=train.shape[0]
    M=len(sorted_features_list)
    train_list=np.empty(shape=[N,M])
    #print(train_list.shape)
    
    for i in range(train.shape[0]):
        ll=Counter(train[0][i])
        temp_list=[]
        for word in sorted_feature_list:
            if (ll[word]>0):
                temp_list.append(1)
            else:
                temp_list.append(0)
        #train_list=np.insert(train_list,i,temp_list,axis=0)
        train_list[i]=temp_list
        temp_list.clear()
    #print(N)        
    print((train_list.shape))
    return np.array(train_list)


# In[33]:


import math
def cosine_dot(a,b):
#     if (np.linalg.norm(a)==0 or np.linalg.norm(b)==0):
#         return 0;
#     else:
#         #temp=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
    #temp=np.dot(a,b)
#     temp=0.0
#     for i in range(len(a)):
#         temp+=(a[i] and b[i])
#     return float(temp/(math.sqrt(sum(a))*math.sqrt(sum(b))))
    a=np.array(a)
    b=np.array(b)
    dist = np.linalg.norm(a-b)
    return dist


# In[36]:


def KNN_algorithm_func(train_matrix,test_matrix,K):
    min_dist=0.0
    min_index=0.0
    actual=[]
    predicted=[]
    train_matrix=train_matrix.tolist()
    test_matrix=test_matrix.tolist()
    print(len(train_matrix))
    for i in range(len(test_matrix)):
        actual.append(test[1][i])
        temp_distance={}
        for j in range(len(train_matrix)):
            dist=cosine_dot(test_matrix[i],train_matrix[j])
            temp_distance[j]=dist
        temp_distance=sorted(temp_distance.items(),key=operator.itemgetter(1))
        temp_distance=temp_distance[:K]
        temp_list=[]
        for key,value in temp_distance:
            temp_list.append(train[1][key])
        predicted.append(max(temp_list))
    return actual,predicted
        


# In[59]:


sorted_feature_list=FEATURE_SELECTION_TFIDF(train,test,5)
print(len(sorted_feature_list))
train_matrix=preprocessing_for_KNN_func(train,sorted_feature_list)
test_matrix=preprocessing_for_KNN_func(test,sorted_feature_list)
# actual3,predicted3=KNN_algorithm_func(train_matrix,test_matrix)
# acc3=accuaracy(actual3,predicted3)
# print(acc3)
# print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[73]:


sorted_feature_list=FEATURE_SELECTION_MUTUAL_INFORMATION(train,test,1)
print(len(sorted_feature_list))
train_matrix=preprocessing_for_KNN_func(train,sorted_feature_list)
test_matrix=preprocessing_for_KNN_func(test,sorted_feature_list)


# In[74]:


start_time=time.time()
K=[1,3,5]
for kkk in K:
    actual3,predicted3=KNN_algorithm_func(train_matrix,test_matrix,kkk)
    acc3=accuaracy(actual3,predicted3)
    conf=confusion_matrix_func(actual3,predicted3)

print("--- %s seconds ---" % (time.time() - start_time))


# In[ ]:





# In[ ]:





# In[ ]:




