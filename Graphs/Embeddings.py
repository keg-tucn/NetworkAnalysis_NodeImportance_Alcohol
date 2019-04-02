import time

from MatProc import MatProc
from Reader import Reader
import shutil
import numpy as np
import os
import networkx as nx
import re
import pickle
from node2vec_main import *
from gensim.models import Word2Vec
from SVM_Learn import *
import matplotlib.pyplot as plt
#mat=reader.getMatByName("P02_SCAPearson-1-Control-91-ValAtTimeOffset.csv")
#np.savetxt("original.csv",mat,delimiter=' , ')

#mat=proc.binarize(mat)
#np.savetxt("test.csv",mat,delimiter=' , ')
#size=mat.shape;

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
    def writeAdjMatrixForCondition(self,condition,outFolder,walkLength,nrWalks,windowSize):
        index=0
        dimensions = 85

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
            newMain(inputName,dimensions,output,condition,walkLength=walkLength,nrWalks=nrWalks,weighted=True,windowSize=windowSize)
            # os.system("python ./node2vec_main.py" + " --input "+outFolder+"/graf" + str(
            #     trial) + ".txt" + "  --dimensions 85 --num-walks 14 --weighted   --output "+outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(    #get the embedding
            #     trial) + ".txt")
            index = index + 1

    def learn_embeddings(self,walks,dimensions,windowSize,nrWorkers,nrIterations):
        '''
        Learn embeddings by optimizing the Skipgram objective using SGD.
        '''
        global model
        walks=[item for x in walks for item in x ]
        walks = [map(str, walk) for walk in walks]

        if (model is None):
            model = Word2Vec(walks, size=dimensions, window=windowSize, min_count=0, sg=1, workers=nrWorkers,
                             iter=nrIterations)
        else:
            model.train(walks,total_examples=len(walks),epochs=10)

        return
    def writeConditionEmbedding(self,condition,outFolder,walkLength,nrWalks,windowSize):
        index = 0
        global model

        dimensions = 85

        matsi = reader.getAllByCondition(condition)  # getall mats
        allWalks=[]
        output = outFolder + "embeddings/EMBD_" + condition +"_777_777.txt"
        model=None
        for [matu, fileName] in matsi:
            print("Processing file " + str(fileName))
            # animalId = int(index / 5) + 1  # each animal has 5 readings for a state
            m = re.search(condition + "-(.+?)-", fileName)
            trial = m.group(1)
            aux = matu
            np.savetxt(outFolder + trial, aux, fmt="%f", delimiter=',')  # just for testing
            size = aux.shape
            grafFileName=outFolder + "graf" + str(trial) + ".txt"
            with open(grafFileName, 'w+') as file:  # save as adj lisr
                for i in range(0, size[0]):
                    for j in range(i + 1, size[1]):
                        if aux[i][j] is not 0:
                            file.write(str(i) + " " + str(j) + " " + str(aux[i][j]) + "\n");


            allWalks.append(getGraphWalks(grafFileName,dimensions,directed=False,walk_length=walkLength,num_walks=nrWalks,weighted=True))
            #newMain(inputName, dimensions, output, condition, walkLength, nrWalks, True)
            # os.system("python ./node2vec_main.py" + " --input "+outFolder+"/graf" + str(
            #     trial) + ".txt" + "  --dimensions 85 --num-walks 14 --weighted   --output "+outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(    #get the embedding
            #     trial) + ".txt")
            index = index + 1
            self.learn_embeddings(allWalks,dimensions,windowSize=windowSize,nrWorkers=16,nrIterations=1)
        try:
            model.wv.save_word2vec_format(output)
        except:
            print("Something is not good with your trained model or it doesn't exist")


def createEmbeddings(function,outFolder,readingsFolder,conditions,walkLength,nrWalks,windowSize):

    # reader.readAll2("./Readings/Readings_Train/",["EtOH","Control"])
    reader.readAll(readingsFolder)
    try:
        shutil.rmtree(outFolder)
    except:
        print("The folder "+outFolder+" does not exist\n")
    os.makedirs(os.path.dirname(outFolder))
    os.makedirs(os.path.dirname(outFolder + "embeddings/"))

    for condition in conditions:
        function(condition, outFolder,walkLength=walkLength,nrWalks=nrWalks,windowSize=windowSize)
def classifyAndTrain():

    obj.storeEmbedding("Control", "./training/embeddings/")
    obj.storeEmbedding("EtOH", "./training/embeddings/")
    obj.storeEmbedding("Abstinence", "./training/embeddings/")

    #obj.KNN("./testing/embeddings",1)

    plt.ylabel('some numbers')
    plt.show()
    obj.classifyByClosestNeighbor("./testing/embeddings")
    classifiers = obj.train()
    # obj.classify("./testing/embeddings")


proc=MatProc()
reader = Reader()
model=None
embedder = Mat2Graph()

readings=None# 4d mat 1 dim=clasify alg
#                     2 dim-nrwalks
# 3 dim-walkLength
# 4 dim window size
# 0-knn,1 closest

needNewData=False
nrClassifiers=2
walksSet=[20,30]
walkLengthSet=[10,15,20]
windowSizeSet=[5,7,9]
if needNewData:
    for i,nrWalks in enumerate(walksSet):
        for j,walkLength in enumerate(walkLengthSet):
            for k,windowSize in enumerate(windowSizeSet):
                if readings is None:

                    readings=np.zeros((nrClassifiers,len(walksSet),len(walkLengthSet),len(windowSizeSet)))
                print("Starting new execution",i, j, k)
                start = time.time()
                createEmbeddings(embedder.writeConditionEmbedding, './training/', "./Readings/Readings_Train/",
                                 ["Control", "EtOH", "Abstinence"], walkLength=walkLength, nrWalks=nrWalks, windowSize=windowSize)
                createEmbeddings(embedder.writeAdjMatrixForCondition, './testing/', "./Readings/Readings_Test/",
                                 ["Control", "EtOH", "Abstinence"], walkLength=walkLength, nrWalks=nrWalks, windowSize=windowSize)

                print("Finished with ",i, j, k)
                print("Elapsed time ",str(int(time.time()-start)))

                obj = SVMobj()
                obj.storeEmbedding("Control", "./training/embeddings/")
                obj.storeEmbedding("EtOH", "./training/embeddings/")
                obj.storeEmbedding("Abstinence", "./training/embeddings/")

                readings[0,i, j, k]=  obj.KNN("./testing/embeddings",1)
                readings[1, i, j, k]=obj.classifyByClosestNeighbor("./testing/embeddings/")
    filehandler = open(b"results.obj", "wb")
    pickle.dump(readings, filehandler)
    filehandler.close()
else:
    # readings = None  # 4d mat 1 dim=clasify alg
    #                     2 dim-nrwalks
    # 3 dim-walkLength
    # 4 dim window size
    # 0-knn,1 closest
    obj = SVMobj()
    obj.storeEmbedding("Control", "./training/embeddings/")
    obj.storeEmbedding("EtOH", "./training/embeddings/")
    obj.storeEmbedding("Abstinence", "./training/embeddings/")

    obj.KNN("./testing/embeddings", 1)
    obj.classifyByClosestNeighbor("./testing/embeddings/")
    exit(1)
    fig, ax = plt.subplots()
    filehandler = open(b"results.obj", "rb")

    readings=pickle.load(filehandler)
    # ax.plot(walksSet, readings[0,:,2,2], 'k--', label='walksSet length KNN')
    # ax.plot(walksSet, readings[1,:,2,2], 'k', label='walksSet length closest')
    # ax.plot(walkLengthSet, readings[0, 1, :, 2], 'g--', label='walksLengthSet length KNN')
    # ax.plot(walkLengthSet, readings[1, 1, :, 2], 'r--', label='walksLengthSet length length closest')
    ax.plot(windowSizeSet, readings[0, 1, 2, :], 'g--', label='windowsSize KNN')
    ax.plot(windowSizeSet, readings[1, 1, 2, :], 'r--', label='windowSize closest')

    legend = ax.legend(loc='center', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')

    plt.show()


sys.exit()



