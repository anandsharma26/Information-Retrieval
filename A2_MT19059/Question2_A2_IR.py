import itertools
import operator
with open('/home/anandsharma/Desktop/IR_ASSIGNMENT/Assignment2_IR/english2.txt') as f:
    lines=f.read().splitlines()
    #print(lines)
    print(len(lines))


#Naive edit_distance using Recursion
def edit_distance1(a,b,m,n):
    if n==0:
        return m*2
    elif m==0:
        return n
    elif (a[m-1] == b[n-1]):
        return edit_distance(a,b,m-1,n-1)
    else :
        return min(1+edit_distance(a,b,m-1,n),2+edit_distance(a,b,m,n-1),3+edit_distance(a,b,m-1,n-1))
    

def edit_distance(a,b,m,n):
    dp= [[0 for x in range(n + 1)] for x in range(m + 1)]
    for i in range(0,m+1,1):
        for j in range(0,n+1,1):
            if(i==0):
                dp[i][j]=j*2
            elif (j==0):
                dp[i][j]=i*1
            elif(a[i-1]==b[j-1]):
                dp[i][j]=dp[i-1][j-1]
            else:
                dp[i][j]=min((1+dp[i-1][j]),(2+dp[i][j-1]),(3+dp[i-1][j-1]))
    return dp[m][n]

query=input("Enter the query ")
query=query.lower()
k=int(input("enter the value of k "))
#query=[]
#query="anand sharma is the best"
query=query.split()
#print(lines)
print(query)

for i in query:
    
    if i not in lines:
        temp={}
        for j in lines:
            temp[j]=(edit_distance(j,i,len(j),len(i)))
        temp=sorted(temp.items(), key=operator.itemgetter(1))
        #outed=dict(itertools.islice(temp.items(),k))
        temp1=(temp[:k])
        templist=[]
        for j in range(0,k,1):
            templist.append(temp1[j][0])
                
        print(templist)
        #print("final k words are :-",str(outed))
    else:
        print(i)




