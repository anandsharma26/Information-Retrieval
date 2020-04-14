#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
import nltk
import matplotlib.pyplot as plt


# In[ ]:





# In[36]:


qid = []
f=open("IR-assignment-3-data.txt",'r')
total = 0
relevance_score={}
ind=0
for i in f.readlines():
    #print(i)
    p = i.split(" ")
    
    if p[1] == 'qid:4':
        relevance_score[ind]=p[0]
        ind+=1
        if int(i.split(" ")[0]) > 0:
            qid.append([1, float(i.split(" ")[76].split(":")[1])])
            total += 1
        else:
            qid.append([0, float(i.split(" ")[76].split(":")[1])])
    else:
        continue
        
relevance_score={key:val for key,val in relevance_score.items() }
relevance_score1=relevance_score
print(ind)
#print(relevance_score)


# In[37]:


relevance_score1=dict(sorted(relevance_score1.items(),key=lambda kv:kv[1],reverse=True))
#print(relevance_score1)


# In[38]:


def DCG(relevance_score):
    maxscore=0
    for i in range (1,len(relevance_score),1):
        maxscore+=(float(relevance_score[i-1])/(math.log((i+1),2)))
    print(maxscore)
    return float(maxscore)


# In[41]:


rel_list=[]
rel_list_sorted=[]
rel_unique=[0]*5

for i in relevance_score.keys():
    rel_list.append(relevance_score[i])
    if(relevance_score[i]=='0'):
        rel_unique[0]+=1
    if(relevance_score[i]=='1'):
        rel_unique[1]+=1
    if(relevance_score[i]=='2'):
        rel_unique[2]+=1
    if(relevance_score[i]=='3'):
        rel_unique[3]+=1
    if (relevance_score[i]=='4'):
        rel_unique[4]+=1
  

for i in relevance_score1.keys():
    rel_list_sorted.append(relevance_score1[i])

dcg=DCG(rel_list)
idcg=DCG(rel_list_sorted)
ndcg=dcg/idcg
print(ndcg)
print(rel_unique)


# In[25]:



f=open("IR-assignment-3-data.txt",'r')
output_file=open("output_file_url.txt","w")
cnt=0
for i in f.readlines():
    #print(i)
    if cnt in relevance_score.keys():
        output_file.writelines(i)
    cnt+=1
    


# In[10]:


qid=sorted(qid,key=lambda x:x[1],reverse=True)
print(qid)
count = 1

precision = []
recall = []

classified = 0

for i in [m[0] for m in qid]:
    precision.append(classified/count)
    recall.append(classified/total)
    if i == 1:
        classified += 1
    count += 1


# In[11]:



plt.xlabel('Precision')
plt.ylabel('Recall')
plt.title('Precision-Recall Curve')
plt.plot(recall, precision)
plt.show()


# In[ ]:




