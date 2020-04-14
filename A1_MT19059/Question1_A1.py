#REFERNCES :-Geeks for Geeks and StackOverflow
# coding: utf-8

# In[2]:


import os
from os import listdir

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


path="/home/anand/Desktop/IR_ASSIGNMENT/20_newsgroups"
folder_path=getListOfFiles(path)
#print((folder_path))
#print("done")


# In[3]:


print(folder_path[0])
print(len(folder_path))
#print("done")


# In[4]:


i=open(folder_path[0],encoding= 'utf-8',errors='ignore').read()
print(len(i))
#print("done")


# In[5]:


import nltk
from nltk.tokenize import RegexpTokenizer
from collections import defaultdict
import string
import re
tokens={}
frequency={}
total_doc_id=[]
for i in range(len(folder_path)):
    text=open(folder_path[i],encoding= 'utf-8',errors='ignore').read()
    text=text.lower()#converting to lower case
    temp2=re.sub(r'\d+','',text)#removing numbers
    tokenizer=RegexpTokenizer(r'\w+')
    temp1=tokenizer.tokenize(temp2)
    #print(folder_path)
    index=folder_path[i].rfind('/')
    doc_id=folder_path[i][index+1:len(folder_path[i])]
    total_doc_id.append(doc_id)
    for j in temp1:
        count=0;
        if j not in tokens.keys():
            tokens[j]=[]
            tokens[j].append(doc_id)
            frequency[j]=[]
            count+=1
            frequency[j]=count
        else :
            count=frequency.get(j)
            count+=1
            #if doc_id not in tokens[j]:
            tokens[j].append(doc_id)
            frequency[j]=(count)
    #print(tokens)
    #print(len(tokens))
    
    #print(len(frequency))
#print(tokens)
print(len(tokens))
print(len(frequency))
#print("done")


# In[7]:


#query=input("Enter your query")
def and_operation(list1,list2,compar):
    list1.sort()
    list2.sort()
    list3=[]
    i=0
    j=0
    while(i<len(list1) and j<len(list2)):
        if(list1[i]==list2[j]):
            list3.append(list1[i])
            i+=1
            j+=1
        elif (list1[i] < list2[j]):
            i+=1
        else:
            j+=1
        compar+=1
    return list3,compar
    
def or_operation(list1,list2,compar):
    list1.sort()
    list2.sort()
    list3=[]
    i=0
    j=0
    while(i<len(list1) and j<len(list2)):
        if(list1[i]==list2[j]):
            list3.append(list1[i])
            i+=1
            j+=1
        if (list1[i] < list2[j]):
            list3.append(list1[i])
            i+=1
        else:
            list3.append(list2[j])
            j+=1
        compar+=1
        if(i==len(list1)):
            while(j<len(list2)):
                list3.append(list2[j])
                j+=1
        if (j==len(list2)):
            while(i<len(list1)):
                list3.append(list1[i])
    return list3,compar
#print("done")


# In[10]:


#query_str="not srv and srv and not srv and not nikunj or srv"
#query_str="a and b and c and d or e or f and h and k"
#query_str="i or just and recently and that or not see and not am or bisexual or also and returned and not pointed"
query_str=input("enter the boolean query to be retrieved")
query_str=re.sub(r'\d+','',query_str)
query_str=query_str.lower()
#query_str="not zola and esd and sgi or scale and width"
#query_str="the and not has or not aslo and srv and is or  the"
query=query_str.split()
print(len(query))
#NOT OPERATION FUNCTION
i=0
while i<(len(query)-1):
    if (query[i].lower()!='and' and query[i].lower()!='or'):
        if(query[i]=='not'):
            if(not(query[i+1]in tokens)):
                tokens[query[i+1]]=set()
            query[i]=(set(total_doc_id)-set(tokens.get(query[i+1])))
            query.pop(i+1)
        
            #query[i]=(tokens.get(query[i]))
    i+=1
#query[len(query)-1]=tokens.get(query[i])
#print (query)
#print(len(query))
#AND OPTIMIZATION OPERATOR fUNCTION
i=1
count=0
templist=[]
templist_index=[]
while i<(len(query)):
    
    if(query[i]=='and'):
        count+=1
        templist.append(query[i-1])
        templist_index.append(i-1)
    if((query[i]=='or' or i==len(query)-1) and count>=1):
        templist.append(query[i-1])
        templist_index.append(i-1)
        templist.sort(key=lambda j:frequency[j] if type(j) is str else len(j))
        l=0
        for k in templist_index:
            query[k]=templist[l]
            l+=1
        count=0;
        #print(templist)
        templist.clear()
        templist_index.clear()
    
    i+=1
# if(query[len(query)-2]=='and'):
#     templist.append(query[len(query)-1])

#print(query)   
#print(len(query))
#AND OPERATOR FUNCTION
no_of_comparision=0
i=1
while i<(len(query)-1):
    if(query[i]=='and'):
        if (type(query[i-1])==str):
            query[i-1]=tokens.get(query[i-1])
        if (type(query[i+1])==str):
            query[i+1]=tokens.get(query[i+1])
        temp,cont=(and_operation(list(query[i-1]),list(query[i+1]),0))
        no_of_comparision+=cont
        temp=set(temp)
        query[i-1]=temp
        query.pop(i)
        query.pop(i)
    else :
        i+=1
#print(query)
#print(len(query))
#OR OPERATOR FUNCTION
i=1
while i<(len(query)-1):
    if(query[i]=='or'):
        if (type(query[i-1])==str):
            query[i-1]=tokens.get(query[i-1])
        if (type(query[i+1])==str):
            query[i+1]=tokens.get(query[i+1])
        temp,cont=(or_operation(list(query[i-1]),list(query[i+1]),0))
        temp=set(temp)
        no_of_comparision+=cont
        query[i-1]=temp
        query.pop(i)
        query.pop(i)
    else :
        i+=1
#print(query)
#print(len(query))
print("NO of DOCUMENTS RETRIEVED",len(query[0]))
print("DOC_ID OF ALL THE DOCS RETRIEVED",(query[0]))
print("NO OF COMPARISION PERFORMED ",no_of_comparision)

