from MatProc import MatProc
from Reader import Reader
import shutil
import numpy as np
import os
import re
#mat=reader.getMatByName("P02_SCAPearson-1-Control-91-ValAtTimeOffset.csv")
#np.savetxt("original.csv",mat,delimiter=' , ')
proc=MatProc()
#mat=proc.binarize(mat)
#np.savetxt("test.csv",mat,delimiter=' , ')
#size=mat.shape;

class Mat2Graph():
    def writeAdjMatrix(self,condition,outFolder):
        index=0
        matsi = reader.getAllByCondition(condition)#getall mats
        print(matsi)
        for [matu,fileName] in matsi:
            print("Processing file "+str(fileName))
            #animalId = int(index / 5) + 1  # each animal has 5 readings for a state
            m=re.search("SCAPearson-(.+?)-",fileName)#get animal id and trial number
            animalId=m.group(1)
            m = re.search(condition+"-(.+?)-", fileName)
            trial = m.group(1)
            aux = proc.binarize(matu)
            np.savetxt(outFolder+trial,aux,fmt="%d",delimiter=',')
            size = aux.shape
            with open(outFolder+"graf" + str(trial) + ".txt", 'w+') as file: #save as adj lisr
                for i in range(0, size[0]):
                    for j in range(i + 1, size[0]):
                        if aux[i][j] == 1:
                             file.write(str(i) + " " + str(j) + "\n");
            os.system("python ./node2vec_main.py" + " --input "+outFolder+"/graf" + str(
                trial) + ".txt" + "  --dimensions 85 --num-walks 40 --output "+outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(    #get the embedding
                trial) + ".txt")
            index = index + 1



outFolder='./training/'
reader=Reader()
reader.readAll2("./sum_weight_high_edge_values/sum_weight_30/training/",["EtOH","Control"])
try:
    shutil.rmtree(outFolder)
except:
    print("TMP folder not found\n");
os.makedirs(os.path.dirname(outFolder))
os.makedirs(os.path.dirname(outFolder + "embeddings/"))
embedder=Mat2Graph()
embedder.writeAdjMatrix("Control",outFolder);

embedder.writeAdjMatrix("EtOH",outFolder);
embedder.writeAdjMatrix("Abstinence",outFolder);






outFolder='./testing/'
reader=Reader()
reader.readAll2("./sum_weight_high_edge_values/sum_weight_30/testing/",["EtOH","Control"])
try:
    shutil.rmtree(outFolder)
except:
    print("TMP folder not found\n");
os.makedirs(os.path.dirname(outFolder))
os.makedirs(os.path.dirname(outFolder + "embeddings/"))
embedder=Mat2Graph()
embedder.writeAdjMatrix("Control",outFolder);

embedder.writeAdjMatrix("EtOH",outFolder);
embedder.writeAdjMatrix("Abstinence",outFolder);





