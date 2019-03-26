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
reader = Reader()

class Mat2Graph():
    def writeAdjMatrix(self,condition,outFolder):
        index=0
        matsi = reader.getAllByCondition(condition)#getall mats

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
        dimensions = 85
        nrWalks = 30
        matsi = reader.getAllByCondition(condition)#getall mats

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

            output=outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(trial) + ".txt"
            newMain(inputName,dimensions,output,condition,30,nrWalks,True)
            # os.system("python ./node2vec_main.py" + " --input "+outFolder+"/graf" + str(
            #     trial) + ".txt" + "  --dimensions 85 --num-walks 14 --weighted   --output "+outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(    #get the embedding
            #     trial) + ".txt")
            index = index + 1

def createEmbeddings(outFolder,readingsFolder,conditions):

    # reader.readAll2("./Readings/Readings_Train/",["EtOH","Control"])
    reader.readAll(readingsFolder)
    try:
        shutil.rmtree(outFolder)
    except:
        print("The folder "+outFolder+" does not exist\n")
    os.makedirs(os.path.dirname(outFolder))
    os.makedirs(os.path.dirname(outFolder + "embeddings/"))
    embedder = Mat2Graph()
    for condition in conditions:
        embedder.writeAdjMatrixForCondition(condition, outFolder)


createEmbeddings('./training/',"./Readings/Readings_Train/",["Control","EtOH","Abstinence"])
createEmbeddings('./testing/',"./Readings/Readings_Test/",["Control","EtOH"])



exit(1)




