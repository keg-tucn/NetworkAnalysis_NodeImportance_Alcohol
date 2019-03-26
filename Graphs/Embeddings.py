from MatProc import MatProc
from Reader import Reader
import shutil
import numpy as np
import os
import re
from node2vec_main import *
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
            aux = matu
            np.savetxt(outFolder+trial,aux,fmt="%f",delimiter=',')#just for testing
            size = aux.shape
            with open(outFolder+"graf" + str(trial) + ".txt", 'w+') as file: #save as adj lisr
                for i in range(0, size[0]):
                    for j in range(i + 1, size[1]):
                        if aux[i][j] is not 0:
                             file.write(str(i) + " " + str(j) +" "+str(aux[i][j])+ "\n");
            os.system("python ./node2vec_main.py" + " --input "+outFolder+"/graf" + str(
                trial) + ".txt" + "  --dimensions 85 --num-walks 14 --weighted   --output "+outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(    #get the embedding
                trial) + ".txt")
            index = index + 1
    def writeAdjMatrixForCondition(self,condition,outFolder):
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
            aux = matu
            np.savetxt(outFolder+trial,aux,fmt="%f",delimiter=',')#just for testing
            size = aux.shape
            with open(outFolder+"graf" + str(trial) + ".txt", 'w+') as file: #save as adj lisr
                for i in range(0, size[0]):
                    for j in range(i + 1, size[1]):
                        if aux[i][j] is not 0:
                             file.write(str(i) + " " + str(j) +" "+str(aux[i][j])+ "\n");
            inputName=outFolder+"/graf" + str(trial) + ".txt"
            dimensions=85
            nrWalks=14
            output=outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(trial) + ".txt"
            newMain(inputName,dimensions,output,condition,30,nrWalks,True)
            # os.system("python ./node2vec_main.py" + " --input "+outFolder+"/graf" + str(
            #     trial) + ".txt" + "  --dimensions 85 --num-walks 14 --weighted   --output "+outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(    #get the embedding
            #     trial) + ".txt")
            index = index + 1




outFolder='./training/'
reader=Reader()
#reader.readAll2("./Readings/Readings_Train/",["EtOH","Control"])
reader.readAll("./Readings/Readings_Train/")

try:
    shutil.rmtree(outFolder)
except:
    print("TMP folder not found\n");
os.makedirs(os.path.dirname(outFolder))
os.makedirs(os.path.dirname(outFolder + "embeddings/"))
embedder=Mat2Graph()
# embedder.writeAdjMatrix("Control",outFolder);

embedder.writeAdjMatrixForCondition("Control",outFolder);
embedder.writeAdjMatrixForCondition("EtOH",outFolder);
embedder.writeAdjMatrixForCondition("Abstinence",outFolder);






outFolder='./testing/'
reader=Reader()
reader.readAll("./Readings/Readings_Test/")

# reader.readAll2(".Readings/Readings_Test/",["EtOH","Control"])
try:
    shutil.rmtree(outFolder)
except:
    print("TMP folder not found\n");
os.makedirs(os.path.dirname(outFolder))
os.makedirs(os.path.dirname(outFolder + "embeddings/"))
embedder=Mat2Graph()
embedder.writeAdjMatrixForCondition("Control",outFolder);

embedder.writeAdjMatrixForCondition("EtOH",outFolder);
#embedder.writeAdjMatrix("Abstinence",outFolder);





