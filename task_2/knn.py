import heap
import math
 
def distance(x,y,method="E"):
    if method=="E":#Euclidean distance
        dis=0
        for i in range(len(x)):
            dis+=(x[i]-y[i])**2
        return math.sqrt(dis)
    if method=="M":#Manhattan distance
        dis=0
        for i in range(len(x)):
            dis+=abs(x[i]-y[i])
        return dis

def majority(h):
    a=[]
    while not heap.is_empty(h):
        a+=[heap.remove(h)[1]]
    a.sort()
    count=0
    maj=a[0]
    max=1
    for i in range(1,len(a)):
        if a[i]==a[i-1]:
            count+=1
            if count>max:
                max=count
                maj=a[i]
        else:
            count=1
    return maj

def knn(k, points, x):
    h=heap.create()
    f=open(points, "r")
    s=f.readline()
    while s!="":
        s=s.split(",")
        for i in range(len(s)):
            s[i]=float(s[i])
        d=distance(s[:-1],x)
        heap.insert(h,[d,s[-1]])
        if heap.size(h)>k:
            heap.remove(h)
        s=f.readline()
    f.close()
    return majority(h)



            
        
        
