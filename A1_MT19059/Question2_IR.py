#REFERNCES :-Geeks for Geeks and StackOverflow
# coding: utf-8

# In[1]:


import os 
from os import listdir
import nltk
from nltk import RegexpTokenizer
import re
import string


# In[2]:


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


# In[3]:


path1="/home/anand/Desktop/IR_ASSIGNMENT/20_newsgroups/comp.graphics"
path2="/home/anand/Desktop/IR_ASSIGNMENT/20_newsgroups/rec.motorcycles"
tokens={}
folder_path=[]
total_doc_id=[]
folder_path=getListOfFiles(path1)
folder_path+=(getListOfFiles(path2))
#print(len(folder_path))
#print(len(folder_path))
for i in range(len(folder_path)):
    
    text=open(folder_path[i],encoding='utf-8',errors='ignore').read()
    text=text.lower()
    #temp2=re.sub(r'\d+','',text)
    tokenizer=RegexpTokenizer(r'\w+')
    temp1=tokenizer.tokenize(text)
    index=folder_path[i].rfind('/')
    doc_id=folder_path[i][index+1:len(folder_path[i])]
    total_doc_id.append(doc_id)
    #print(len(temp1))
    #print(doc_id)
    #print(temp1)
    for count,name in enumerate(temp1):
        if name in tokens:
            tokens[name][0]=tokens[name][0]+1
            if doc_id in tokens[name][1]:
                tokens[name][1][doc_id].append(count)
            else:
                tokens[name][1][doc_id]=[count] 
        else:
            tokens[name]=[]
            tokens[name].append(1)
            tokens[name].append({})
            tokens[name][1][doc_id]=[count]

            
        
        
#print((temp1))
print(len(tokens))
#print(tokens['the'])
#print(len(temp1))
#print(len(total_doc_id))
#print(tokens)


# In[17]:


#query_str="It is a "
#query_str="Presentations are solicited on all"
query_str=input("Enter the prase query to be retrieved")
query_str=query_str.lower()
query=tokenizer.tokenize(query_str)
query=query_str.split()

print(query)
#print(len(query))


# In[18]:


#print(total_doc_id)
matched_doc_id=[]
for i in (total_doc_id):
    temp=[]
    for j in range(len(query)):
        temp.append(tokens[query[j]][1].get(i))
    #print(len(temp))
    #print(temp[0])
    if(temp[0]!=None):
        for k in range(0,len(temp[0]),1):
            count=0
            val=temp[0][k]
            for j in range(1,len(query),1):
                if((temp[j])!=None):
                    if((val+j) in temp[j]):
                        count+=1
                else :
                    continue
            if(count==len(query)-1):
                matched_doc_id.append(i)
                break
print(matched_doc_id) 
print(len(matched_doc_id))

