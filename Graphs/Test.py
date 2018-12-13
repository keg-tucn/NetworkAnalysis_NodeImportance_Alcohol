from MatProc import MatProc
from Reader import Reader
import numpy as np
import os
reader=Reader()
reader.readAll()
#mat=reader.getMatByName("P02_SCAPearson-1-Control-91-ValAtTimeOffset.csv")
#np.savetxt("original.csv",mat,delimiter=' , ')
proc=MatProc()
#mat=proc.binarize(mat)
#np.savetxt("test.csv",mat,delimiter=' , ')
#size=mat.shape;
matsi=reader.getReadingsByConditionAndAnimal(4,"Control")
index=0
#np.savetxt("da" + str(index) + ".txt", matsi, delimiter=' ')
print(matsi)
for matu in matsi:
    #np.savetxt("test.csv",matu,delimiter=' , ')

    aux=proc.binarize(matu)
    np.savetxt("out"+str(index)+".txt",aux,delimiter=' ')
    size=aux.shape
    with open("graf"+str(index)+".txt","a") as file:
        for i in range(0,size[1]):
            for j in range(i+1,size[0]):
                if aux[i][j]==1:
                    file.write(str(i) + " " + str(j) + "\n");
    os.system("node2vec_main.py"+ " --input graf"+str(index)+".txt"+ " --output embedding"+str(index)+".txt")
    index=index+1



