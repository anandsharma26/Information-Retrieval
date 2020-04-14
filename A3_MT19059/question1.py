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



print(folder_path[0])
print(len(folder_path))
with open('file.txt','r') as f:
    new_file=f.readlines()
new_file=[line.replace('\n','') for line in new_file]
#print(new_file)
reviews_list={}
for i in new_file:
    i=regextokenizer_func(i)
    reviews_list[i[0]]=i[1]
print((reviews_list[str(1)]))

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


def calculate_idf1(temp,N):
    return math.log(N/temp)
def calculate_tf1(temp):
    return math.log(temp+1)
def calculate_tf_idf_score(tf_dict,idf):
    tf_idf_scores={}
    N=len(tf_dict)
    for i in range((N)):
        for j in idf.keys():
            key=(i,j)
            if key in tf_dict:
                tf_idf_scores[i,j]=(idf.get(j,"0")*tf_dict.get(key,"0"))
    return tf_idf_scores


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
            else :
                tf_dict[i,j].append(calculate_tf1(count))
                
    return tf_dict


def cosine_dot(a,b):
    if (np.linalg.norm(a)==0 or np.linalg.norm(b)==0):
        return 0;
    else:
        temp=np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))
        return temp


def cosine_similarity_func(body_list,query,k):
    N=len(body_list)
    idf_document=calculate_idf_score(body_list)
    idf_document_list=[]
    for i in idf_document:
        idf_document_list.append(i)
 
    tf_score_document=calculate_tf_score(body_list)
  
    tf_idf_score_document=calculate_tf_idf_score(tf_score_document,idf_document)
   
    tf_idf_score_query=calculate_tf_idf_query_score(query,idf_document)
    
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
    return tf_idf_scores

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

total_doc_id=[]
tokens={}
for i in range (1,len(folder_path),1):
    text=open(folder_path[i],encoding='utf-8',errors='ignore').read()
    text=convert_tolowercase(text)
    text=regextokenizer_func(text)
    text=lemmatization_func(text)
    text=remove_stopwords(text)
    
#     index=folder_path[i].rfind('/')
#     doc_id=folder_path[i][index+1:len(folder_path[i])]
    doc_id=i;
    total_doc_id.append(doc_id)
    for count,name in enumerate(text):
        if name in tokens:
            if doc_id  not in tokens[name]:
                tokens[name][doc_id]=[None]*2
                tokens[name][doc_id][0]=count/len(text)
                tokens[name][doc_id][1]=reviews_list[str(doc_id)]
            else:
                tokens[name][doc_id][0]=count/len(text)
        else:
            tokens[name]={}
            tokens[name][doc_id]=[None]*2
            tokens[name][doc_id][0]=(count)
            tokens[name][doc_id][1]=(reviews_list[str(doc_id)])
#print((tokens))
# mean_r={}
# for i in tokens.keys():
#     mean_r[i]=(sum(tokens[i])/len(tokens[i]))
# #print(mean_r)
body_list=[]
for i in tokens.keys():
    body_list.append(i)
temp=calculate_idf_score(body_list)

championlist={}
for i in tokens.keys():
    championlist[i]=[None]*2
    championlist[i][0]={}
    championlist[i][1]={}
    for j in tokens[i].keys():
        if (float(tokens[i][j][0])>=mean_r[i]):
            championlist[i][0][j]=reviews_list[str(j)]
        else :
            championlist[i][1][j]=reviews_list[str(j)]
    championlist[i][0]=sorted(championlist[i][0].items(),key=lambda kv:kv[1],reverse=True)
    championlist[i][1]=sorted(championlist[i][1].items(),key=lambda kv:kv[1],reverse=True)
    





query="path cmu harvard is howland"
#query="unfalsifiable theism hypothesis"
k=30;
#query=input("Enter phrasal query")
#k=input("enter k")
query=convert_tolowercase(query)
query=regextokenizer_func(query)
query=lemmatization_func(query)
query=remove_stopwords(query)
print((query))
query_len=len(query)
cosine_dict=cosine_similarity_func(body_list,query,k)
#print(type(cosine_dict))

final_list=[]
for i in query:
    print(i)
    count_k=0
    for j in championlist[i][0]:
        final_list.append(j[0])
        count_k+=1
        if (count_k==k):
            break
    if count_k<k:
        for j in championlist[i][1]:
            final_list.append(j[0])
            count_k+=1
            if(count_k==k):
                break;
print(final_list)

net_score_list={}
for i in final_list:
    temp1=int(reviews_list[str(i)])
    temp2=cosine_dict[i]
    net_score_list[i]=temp1+temp2

net_score_list=sorted(net_score_list.items(),key=lambda kv:kv[1],reverse=True)
final_doc_id=[]
print(type(net_score_list))
for i in net_score_list:
    final_doc_id.append(i[0])
final_doc_id=final_doc_id[0:k]
print(final_doc_id)




