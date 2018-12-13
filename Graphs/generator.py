import numpy as np
from random import randint

def genearetGraph(gradeList,nrNodes):
    adj=np.zeros((nrNodes,nrNodes),np.int32)
    if(len(gradeList) != nrNodes):#populate missing degreees
        for k in range(0,nrNodes-len(gradeList)):
            gradeList.append(randint(0,nrNodes-1))
    gradeList.sort(reverse=True);
    print(gradeList)
    cop=gradeList.copy()

    degreesSum=sum(gradeList)
    if(degreesSum % 2!=0  ):
        print("Wrong sequence")
        return;
    for k in range (1,nrNodes+1):
        sumPeaks=0
        minSum=0
        for j in range(1,k+1):
            sumPeaks+=gradeList[j-1]
        for j in range(k+1,nrNodes+1):
            minSum+=min(gradeList[j-1],k)
        print("sum", sumPeaks, " minsum ", minSum, " k", k)
        if(sumPeaks>((k)*(k-1)+minSum)):
            print("Not good")

            return;
    print("Looks good")
    for i in range(0,nrNodes):
        for j in range(i+1,nrNodes):
            if(gradeList[i]>0 and gradeList[j]>0):
                gradeList[i]-=1
                gradeList[j]-=1
                adj[i][j]=1
                adj[j][i]=1

    print(adj)

    print(degreesSum)


while(True):
    genearetGraph([],10)
