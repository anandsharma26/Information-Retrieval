#!/usr/bin/env python
# coding: utf-8

# In[20]:


import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import os
import string
import re
from collections import defaultdict
import operator
import numpy as np
import pandas as pd
from collections import Counter
import math


# In[21]:


def convert_tolowercase(data):
    return (data.lower())


# In[22]:


def regextokenizer_func(data):
    #print(type(data))
    tokenizer=RegexpTokenizer(r'\w+')
    data=tokenizer.tokenize(data)
    return data


# In[23]:


def remove_stopwords(data):
    stop_words=set(stopwords.words('english'))
    result=[i for i in data if not i in stop_words]
    return result


# In[24]:


from nltk.stem import WordNetLemmatizer
def lemmatization_func(data):
    lemmatizer=WordNetLemmatizer()
    result=[]
    for word in data:
        result.append(lemmatizer.lemmatize(word))
    return result


# In[25]:


def stemming_func(data):
    stemmer=PorterStemmer()
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(data)
    data_new=""
    for i in tokens:
        data_new+=" "+stemmer.stem(i)
    return data_new


# In[26]:


def jaccard_coefficient(a,b,k,documentname_list):
    scores_docid={}
    for i in range(len(a)):
        count=0
        current_doc=a[i]
        for j in current_doc:
            if j in b:
                count+=1
        scores_docid[i]=(count/(len(current_doc)+len(b)))
    temp=sorted(scores_docid.items(), key=operator.itemgetter(1),reverse=True)
    temp1=(temp[:k])
    #print(temp1)
    retrieved_doc_name=[]
    for i in range(k):
        docid=temp1[i][0]
        retrieved_doc_name.append(documentname_list[docid])
    print("RETRIEVED DOCUMENTS BASAED ON JACCARD COEFFICINT")
    print(retrieved_doc_name)
    


# In[27]:


def calculate_idf1(temp,N):
    return math.log(N/temp)
def calculate_idf2(temp,N):
    return (math.log(N/(1+temp)))
def calculate_idf3(temp,N,maxtemp):
    return (math.log(float(maxtemp)/(1+temp)))
def calculate_tf1(temp):
    return math.log(temp+1)
def calculate_tf2(temp):
    return (1 if temp>0 else 0)
def calculate_tf3(temp,maxtemp):
    return (0.5+0.5*(temp/float(maxtemp)))
def calculate_tf4(temp,sum_total):
    return (temp/sum_total)
def calculate_tf5(temp):
    return temp


# In[28]:


title = "stories"
dataset = []
folders = [x[0] for x in os.walk(str(os.getcwd())+'/'+title+'/')]
folders[0] = folders[0][:len(folders[0])-1]
c = False

for i in folders:
    file = open(i+"/index.html", 'r')
    text = file.read().strip()
    file.close()

    file_name = re.findall('><A HREF="(.*)">', text)
    file_title = re.findall('<BR><TD> (.*)\n', text)

    if c == False:
        file_name = file_name[2:]
        c = True
        
    print(len(file_name), len(file_title))

    for j in range(len(file_name)):
        dataset.append((str(i) +"/"+ str(file_name[j]), file_title[j]))


# In[29]:


N=len(dataset)
title_list=[]
body_list=[]
documentname_list=[]
for i in ((dataset)):
    #print(i)
    current_file=open(i[0],'r',encoding='utf-8',errors='ignore')
    current_text=current_file.read().strip()
    current_title=i[1]
    #print(type(current_text))
    #print((current_title))
    current_text=convert_tolowercase(current_text)
    current_title=convert_tolowercase(current_title)
    current_text=regextokenizer_func(current_text)
    current_title=regextokenizer_func(current_title)
    #current_text=remove_stopwords(current_text)
    current_text=lemmatization_func(current_text)
    #current_text=remove_stopwords(current_text)
    current_title=lemmatization_func(current_title)
    title_list.append(current_title)
    body_list.append(current_text)
    loc=i[0]
    indexof=loc.rfind('/')
    documentname_list.append(i[0][indexof+1:len(loc)])
#print(len(title_list))
print(len(body_list))
#print(documentname_list)


# In[30]:


def tf_idf_retrieval(title_list,body_list,query,k,documentname_list):
    #calculation of DF AND IDF score
    df={}
    N=len(title_list)
    print(N)
    #Calculation of DF score
    for i in range(N):
        tokens1=title_list[i]
        tokens2=body_list[i]
        #print(tokens1)
        #print(tokens2)
#         for j in tokens1:
#             if j  in df.keys():
#                 df[j].add(i)
#                 #df[j].append(i)
#             else :
#                 df[j]={i}
        for j in tokens2:
            if j in df.keys():
                df[j].add(i)
            else:
                df[j]={i}
    #Calculation of idf from df
    idf={}
    for i in df:
        df[i]=len(df[i])
    for i in df:
        #idf[i]=calculate_idf1(df[i],N)
        idf[i]=calculate_idf2(df[i],N)
        #idf[i]=calculate_idf3(df[i],N,max(df))
    #print(len(idf))
    
    #calculation of TF SCORE
    tf_dict=calculate_tf_score(body_list)
    tf_dict1=calculate_tf_score(title_list)
    
    #print(len(tf_dict))

    #calculating tf-idf score
    
    tf_idf_scores={}
    for i in range((N)):
        local_score=0
        for j in query:
            key=(i,j)
            if key in tf_dict:
                local_score+=(idf.get(j,"0")*tf_dict.get(key,"0"))
        tf_idf_scores[i]=(local_score)
        
    tf_idf_scores1={}
    for i in range((N)):
        local_score=0
        for j in query:
            key=(i,j)
            if key in tf_dict1:
                temp10=(0.3*float(idf.get(j,"0"))*float(tf_dict.get(key,"0")))+(0.7*float(idf.get(j,"0"))*float(tf_dict1.get(key,"0")))
                local_score+=temp10
        tf_idf_scores1[i]=(local_score)
    
    print("RETRIEVED DOCUMENTS BASAED ON TF IDF SCORE with TITLE considered same weight")
    temp=sorted(tf_idf_scores.items(), key=operator.itemgetter(1),reverse=True)
    temp1=(temp[:k])
    #print(temp1)
    #print((temp1[0]))
    retrieved_doc_name=[]
    for i in range(k):
        docid=temp1[i][0]
        retrieved_doc_name.append(documentname_list[docid])
    print(retrieved_doc_name)
    print("RETRIEVED DOCUMENTS BASAED ON TF IDF SCORE WITH TITLE given more weightage")
    temp=sorted(tf_idf_scores1.items(), key=operator.itemgetter(1),reverse=True)
    temp1=(temp[:k])
    #print(temp1)
    #print((temp1[0]))
    retrieved_doc_name=[]
    for i in range(k):
        docid=temp1[i][0]
        retrieved_doc_name.append(documentname_list[docid])
    print(retrieved_doc_name)


# In[31]:


def calculate_idf_score(body_list):
    N=len(body_list)
    df={}
    for i in range(N):
        tokens2=body_list[i]
        for j in tokens2:
            if j in df.keys():
                df[j].add(i)
            else:
                df[j]={i}
    #Calculation of idf from df
    idf={}
    for i in df:
        df[i]=len(df[i])
    for i in df:
        idf[i]=calculate_idf1(df[i],N) #inverse document frequency
        #idf[i]=calculate_idf2(df[i],N) #inverse document frequency smooth
        #idf[i]=calculate_idf3(df[i],N,max(df)) #inverse document frequency max
    return idf


# In[32]:


def calculate_tf_score(body_list):
    tf_dict={}
    N=len(body_list)
    for i in range(N):
        tokens2=body_list[i]
        terms_counter=Counter(tokens2)
        tf_count_list=[]
        sum_of_counts=0
        for j in tokens2:
            sum_of_counts+=terms_counter[j]
            tf_count_list.append(terms_counter[j])
        for j in tokens2:
            count=terms_counter[j]
            if j not in tf_dict.keys():
                tf_dict[0]=(i)
                tf_dict[1]=(j)
                tf_dict[i,j]=calculate_tf1(count) #log_normaliztion_TF
                #tf_dict[i,j]=calculate_tf2(count) #binary_TF
                #tf_dict[i,j]=calculate_tf3(count,max(tf_count_list)) #double normalization TF
                #tf_dict[i,j]=calculate_tf4(count,sum_of_counts) #term frequency
                #tf_dict[i,j]=calculate_tf5(count) #RAW_TF
            else :
                tf_dict[i,j].append(calculate_tf1(count))
                #tf_dict[i,j].append(calculate_tf2(count))
                #tf_dict[i,j].append(calculate_tf3(count,max(tf_count_list)))
                #tf_dict[i,j].append(calculate_tf4(count,sum_of_counts))
                #tf_dict[i,j].append(calculate_tf5(count))
    return tf_dict


# In[33]:


def calculate_tf_idf_score(tf_dict,idf):
    tf_idf_scores={}
    for i in range((N)):
        for j in idf.keys():
            key=(i,j)
            if key in tf_dict:
                tf_idf_scores[i,j]=(idf.get(j,"0")*tf_dict.get(key,"0"))
    return tf_idf_scores


# In[50]:


def calculate_tf_idf_score1(tf_dict,tf_dict1,idf):
    tf_idf_scores={}
    for i in range((N)):
        for j in idf.keys():
            key=(i,j)
            if key in tf_dict:
                tf_idf_scores[i,j]=(0.3*float(idf.get(j,"0"))*float(tf_dict.get(key,"0")))+(0.7*float(idf.get(j,"0"))*float(tf_dict1.get(key,"0")))
    return tf_idf_scores


# In[51]:


def calculate_tf_idf_query_score(query,idf_document):
    idf_document_list=[]
    for i in idf_document:
        idf_document_list.append(i)
    #print(idf_document_list)
    term_counter=Counter(query)
    query_len=len(query)
    query_tf_idf_dict=np.zeros((len(idf_document)))
    for i in query:
        tf=term_counter[i]/query_len
        try:
            idf=idf_document[i]
        except:
            pass
        #print(tf*idf)
        try:
            indexxx=idf_document_list.index(i)
            query_tf_idf_dict[indexxx]=tf*idf
        except:
            pass
    #print(query_tf_idf_dict)
    return query_tf_idf_dict


# In[52]:


def cosine_dot(a,b):
    if (np.linalg.norm(a)==0 or np.linalg.norm(b)==0):
        return 0;
    else:
        temp=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return temp

    


# In[56]:


def cosine_similarity_func(title_list,body_list,query,k,documentname_list):
    N=len(body_list)
    idf_document=calculate_idf_score(body_list)
    idf_document_list=[]
    for i in idf_document:
        idf_document_list.append(i)
    #calculation of TF SCORE document
    tf_score_document=calculate_tf_score(body_list)
    #tf_score_document1=calculate_tf_score(title_list)
    tf_idf_score_document=calculate_tf_idf_score(tf_score_document,idf_document)
    #tf_idf_score_document=calculate_tf_idf_score1(tf_score_document,tf_score_document1,idf_document)
    print(len(idf_document))
    #print(len(tf_score_document))
    print(len(tf_idf_score_document))
    #TF IDF SCORE FOR QUERY
    tf_idf_score_query=calculate_tf_idf_query_score(query,idf_document)
    
    #print(tf_idf_score_query)
    #print(len(tf_idf_score_query))
    M=len(idf_document)
    matrix=np.zeros((N,M))
    for i in tf_idf_score_document:
        try:
            idx=idf_document_list.index(i[1])
            matrix[i[0]][idx]=tf_idf_score_document[i]
        except:
            pass
    #print(len(matrix))
    tf_idf_scores={}
    for i in range(N):
        temp_cos=cosine_dot(tf_idf_score_query,matrix[i])
        tf_idf_scores[i]=(temp_cos)
    #print(tf_idf_scores)       
    temp=sorted(tf_idf_scores.items(), key=operator.itemgetter(1),reverse=True)
    temp1=(temp[:k])
    #print(temp1)
    #print((temp1[0]))
    retrieved_doc_name=[]
    for i in range(k):
        docid=temp1[i][0]
        retrieved_doc_name.append(documentname_list[docid])
    print("RETRIEVED DOCUMENTS BASAED ON COSINE SIMILARITY ")
    print(retrieved_doc_name)
        


# In[57]:


def cosine_similarity_func1(title_list,body_list,query,k,documentname_list):
    N=len(body_list)
    idf_document1=calculate_idf_score(body_list)
    idf_document2=calculate_idf_score(title_list)
    idf_document={**idf_document1,**idf_document2}
    print(len(idf_document))
    idf_document_list=[]
    for i in idf_document:
        idf_document_list.append(i)
    #calculation of TF SCORE document
    tf_score_document=calculate_tf_score(body_list)
    tf_score_document1=calculate_tf_score(title_list)
    #tf_idf_score_document=calculate_tf_idf_score(tf_score_document,idf_document)
    tf_idf_score_document=calculate_tf_idf_score1(tf_score_document,tf_score_document1,idf_document)
    #print(len(idf_document))
    #print(len(tf_score_document))
    #print(len(tf_idf_score_document))
    #TF IDF SCORE FOR QUERY
    tf_idf_score_query=calculate_tf_idf_query_score(query,idf_document)
    
    #print(tf_idf_score_query)
    #print(len(tf_idf_score_query))
    M=len(idf_document)
    matrix=np.zeros((N,M))
    for i in tf_idf_score_document:
        try:
            idx=idf_document_list.index(i[1])
            matrix[i[0]][idx]=tf_idf_score_document[i]
        except:
            pass
    #print(len(matrix))
    tf_idf_scores={}
    for i in range(N):
        temp_cos=cosine_dot(tf_idf_score_query,matrix[i])
        tf_idf_scores[i]=(temp_cos)
    #print(tf_idf_scores)       
    temp=sorted(tf_idf_scores.items(), key=operator.itemgetter(1),reverse=True)
    temp1=(temp[:k])
    #print(temp1)
    #print((temp1[0]))
    retrieved_doc_name=[]
    for i in range(k):
        docid=temp1[i][0]
        retrieved_doc_name.append(documentname_list[docid])
    print("RETRIEVED DOCUMENTS BASED ON COSINE SIMILARITY WITH TITLE GIVEN MORE WEIGHT")
    print(retrieved_doc_name)


# In[59]:


query=input("Enter the query to be searched")
k=int(input("Enter the value of no of documents to be retrieved"))
#query="The Adventure of the adventure"
#query="Disco can be fun"
#k=10
#query="Without the drive of Rebeccah's insistence, Kate lost her momentum. She stood next a slatted oak bench, canisters still clutched, surveying"
#query="anand sharma  on the garden"
#query="50000 variety of flowers"
if (len(query)==0):
    print("Empty query so no documents retrieved")
else:
    print(query)
    query=query.lower()
    #print(query)
    query=regextokenizer_func(query)
    #print(query)
    #query=remove_stopwords(query)
    query=lemmatization_func(query)
    print(query)
    jaccard_coefficient(body_list,query,k,documentname_list)
    tf_idf_retrieval(title_list,body_list,query,k,documentname_list)
    cosine_similarity_func(title_list,body_list,query,k,documentname_list)
    cosine_similarity_func1(title_list,body_list,query,k,documentname_list)


# In[ ]:





# In[ ]:




