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
    for k in range (0,nrNodes):
        sumPeaks=0
        minSum=0
        for j in range(0,k+1):
            sumPeaks+=gradeList[j]
        for j in range(k,nrNodes):
            minSum+=min(gradeList[j],k+1)
        if(sumPeaks>(k+1)*(k)+minSum):
            print("Not good")
            print("sum",sumPeaks," minsum ",minSum," k",k)
            return;
    print("Looks good")
    while (degreesSum >0):
        x = randint(0, nrNodes - 1)
        y = randint(0, nrNodes - 1)

        for m in range(0,nrNodes):
            if(gradeList[m]!=0):
                x=m
                for n in range(x + 1, nrNodes):
                    if (gradeList[n] != 0 and adj[x][n]):
                        y = n



        gradeList[x]-=1
        gradeList[y]-=1
        degreesSum-=2
        adj[x][y]=1
        adj[y][x]=1
        gradeList.sort(reverse=True);

    print(adj)

    print(degreesSum)


while(True):
    genearetGraph([2],5)
