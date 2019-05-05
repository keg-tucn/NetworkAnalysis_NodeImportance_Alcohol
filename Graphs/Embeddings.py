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

    def writeGlobalEmbedding(self, condition, outFolder, walkLength, nrWalks, windowSize):
        index = 0
        global model

        dimensions = 85
        output = outFolder + "embeddings/All.txt"


        matsi = reader.getAllByCondition(condition)  # getall mats
        allWalks = []

        for [matu, fileName] in matsi:
            print("Processing file " + str(fileName))
            # animalId = int(index / 5) + 1  # each animal has 5 readings for a state
            m = re.search(condition + "-(.+?)-", fileName)
            trial = m.group(1)
            aux = matu
            np.savetxt(outFolder + trial, aux, fmt="%f", delimiter=',')  # just for testing
            size = aux.shape
            grafFileName = outFolder + "graf" + str(trial) + ".txt"
            with open(grafFileName, 'w+') as file:  # save as adj lisr
                for i in range(0, size[0]):
                    for j in range(i + 1, size[1]):

                        file.write(str(i) + " " + str(j) + " " + str(aux[i][j]) + "\n");

            allWalks.append(
                getGraphWalks(grafFileName, dimensions, directed=False, walk_length=walkLength, num_walks=nrWalks,
                              weighted=True))
            # newMain(inputName, dimensions, output, condition, walkLength, nrWalks, True)
            # os.system("python ./node2vec_main.py" + " --input "+outFolder+"/graf" + str(
            #     trial) + ".txt" + "  --dimensions 85 --num-walks 14 --weighted   --output "+outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(    #get the embedding
            #     trial) + ".txt")
            index = index + 1
            self.learn_embeddings(allWalks, dimensions, windowSize=windowSize, nrWorkers=16, nrIterations=1)
        try:
            model.wv.save_word2vec_format(output)
        except:
            print("Something is not good with your trained model or it doesn't exist")


def createEmbeddings(function,outFolder,readingsFolder,conditions,walkLength,nrWalks,windowSize):


    try:
        shutil.rmtree(outFolder)
    except:
        print("The folder "+outFolder+" does not exist\n")
    os.makedirs(os.path.dirname(outFolder))
    os.makedirs(os.path.dirname(outFolder + "embeddings/"))

    for condition in conditions:
        function(condition, outFolder,walkLength=walkLength,nrWalks=nrWalks,windowSize=windowSize)
def classifyAndTrain(obj):

    obj.storeEmbedding("Control", "./training/embeddings/")
    obj.storeEmbedding("EtOH", "./training/embeddings/")
    obj.storeEmbedding("Abstinence", "./training/embeddings/")

    #obj.KNN("./testing/embeddings",1)

    plt.ylabel('some numbers')
    plt.show()
    obj.classifyByClosestNeighbor("./testing/embeddings")
    classifiers = obj.train()
    # obj.classify("./testing/embeddings")


def read(srcFolder, option):
    global reader
    allConditions = ["Control", "EtOH", "Abstinence"]
    if (option is 1):
        reader.readAll(srcFolder) #my files
    if (option is 2):
        reader.readAll2(srcFolder, allConditions)#separate conditions in folders
    if (option is 3):
        reader.readAll3(srcFolder, allConditions)#all conditions in one folder



def runDataMining(trainSource,testSource,nrClassifiers,walksSet,walkLengthSet,windowSizeSet):
    global model
    readings = None  # 4d mat 1 dim=clasify alg
    #                     2 dim-nrwalks
    # 3 dim-walkLength
    # 4 dim window size
    # 0-knn,1 closest
    for i,nrWalks in enumerate(walksSet):

        for j,walkLength in enumerate(walkLengthSet):
            for k,windowSize in enumerate(windowSizeSet):
                if readings is None:

                    readings=np.zeros((nrClassifiers,len(walksSet),len(walkLengthSet),len(windowSizeSet)))
                print("Starting new execution",i, j, k)

                start = time.time()
                reader.readings=None
                read(trainSource, 2)
                createEmbeddings(embedder.writeConditionEmbedding, './training/', trainSource,
                                 ["Control", "EtOH", "Abstinence"], walkLength=walkLength, nrWalks=nrWalks, windowSize=windowSize)
                reader.readings = None
                read(testSource, 2)
                createEmbeddings(embedder.writeAdjMatrixForCondition, './testing/', testSource,
                                 ["Control", "EtOH", "Abstinence"], walkLength=walkLength, nrWalks=nrWalks, windowSize=windowSize)

                print("Finished with ",i, j, k)
                print("Elapsed time ",str(int(time.time()-start)))

                obj = SVMobj()
                obj.storeEmbedding("Control", "./training/embeddings/")
                obj.storeEmbedding("EtOH", "./training/embeddings/")
                obj.storeEmbedding("Abstinence", "./training/embeddings/")

                readings[0,i, j, k]=  obj.KNN("./testing/embeddings",1)
                readings[1, i, j, k]=obj.classifyByClosestNeighbor("./testing/embeddings/")
                filehandler = open(b"resultsSingleInstances.obj", "wb")

                pickle.dump(readings, filehandler)
                filehandler.close()
    nrClassifiers = 3  # third dim for SVM classifier
    readings=None
    filehandler = open(b"resultsMultipleInstances.obj", "wb")
    for i, nrWalks in enumerate(walksSet):
        for j, walkLength in enumerate(walkLengthSet):
            for k, windowSize in enumerate(windowSizeSet):
                if readings is None:
                    readings = np.zeros((nrClassifiers, len(walksSet), len(walkLengthSet), len(windowSizeSet)))
                print("Starting new execution", i, j, k)
                start = time.time()
                read(trainSource, 2)
                model = None
                createEmbeddings(embedder.writeAdjMatrixForCondition, './training/', trainSource,
                                 ["Control", "EtOH", "Abstinence"], walkLength=walkLength, nrWalks=nrWalks,
                                 windowSize=windowSize)
                reader.readings = None
                read(testSource, 2)
                model = None
                createEmbeddings(embedder.writeAdjMatrixForCondition, './testing/', testSource,
                                 ["Control", "EtOH", "Abstinence"], walkLength=walkLength, nrWalks=nrWalks,
                                 windowSize=windowSize)

                print("Finished with ", i, j, k)
                print("Elapsed time ", str(int(time.time() - start)))

                obj = SVMobj()
                obj.storeEmbedding("Control", "./training/embeddings/")
                obj.storeEmbedding("EtOH", "./training/embeddings/")
                obj.storeEmbedding("Abstinence", "./training/embeddings/")

                readings[0, i, j, k] = obj.KNN("./testing/embeddings", 8)
                readings[1, i, j, k] = obj.classifyByClosestNeighbor("./testing/embeddings/")
                obj.train()
                readings[2, i, j, k] = obj.classify("./testing/embeddings/")

                pickle.dump(readings, filehandler)
                filehandler.close()
def savePlot(X,Y,labels,output):
    fig, ax = plt.subplots()

    colors=['r--','g--','b--']
    for index,(subY,label,color) in enumerate(zip(Y,labels,colors)):
        ax.plot(X, subY, color, label=label)

    # ax.plot(windowSizeSet, readings[0, 1, 1, :], 'g--', label='windowsSize KNN')
    # ax.plot(windowSizeSet, readings[1, 1, 1, :], 'r--', label='windowSize closest')

    legend = ax.legend(loc='best', shadow=True, fontsize='x-large')

    # Put a nicer background color on the legend.
    legend.get_frame().set_facecolor('C0')
    plt.savefig(output+".png")
    plt.show()
    plt.clf()
def readLabels():
    fileName="brainZones.txt"
    readu=[]
    with open( fileName) as file:
        for line in file:
            line=line.split()
            nr=line[0]
            longName=' '.join(line[2:])
            readu.append(   [nr,longName] )

    print(readu)
    return readu;
def test():
    # readLabels()
    wantClassify=True
    wantNewData=False
    if(wantNewData):
        root = "./Dataset_without_time/sum_weight_high_edge_values/sum_weight_70"
        trainSource = os.path.join(root, 'train')
        read(trainSource, 2)
        model = None
        createEmbeddings(embedder.writeGlobalEmbedding, './training/', trainSource,
                         ["Control", "EtOH", "Abstinence"], walkLength=15, nrWalks=30,
                         windowSize=7)


        testSorce = os.path.join(root, 'test')
        read(testSorce, 2)
        model = None
        createEmbeddings(embedder.writeAdjMatrixForCondition, './testing/', testSorce,
                         ["Control", "EtOH", "Abstinence"], walkLength=15, nrWalks=30,
                         windowSize=7)
    if(wantClassify):
        obj = SVMobj()

        obj.storeEmbedding("Control", "./training/embeddings/")
        obj.storeEmbedding("EtOH", "./training/embeddings/")
        obj.storeEmbedding("Abstinence", "./training/embeddings/")
        # obj.train()
        # obj.classify("./testing/embeddings/")
        obj.computeParticulars(readLabels(), "./training/embeddings", "./testing/embeddings/")


proc=MatProc()
reader = Reader()
model=None
embedder = Mat2Graph()
readings=None# 4d mat 1 dim=clasify alg
#                     2 dim-nrwalks
# 3 dim-walkLength
# 4 dim window size
# 0-knn,1 closest

needNewData=True
nrClassifiers=2
walksSet=[20,30,40,50]
walkLengthSet=[10,15,20,25]
windowSizeSet=[5,7]
#
# test()
# sys.exit(2)
#Read pickled data





root="./Dataset_without_time/sum_weight_high_edge_values/"
folders=["sum_weight_10","sum_weight_20","sum_weight_30","sum_weight_40","sum_weight_50","sum_weight_60","sum_weight_70"]

# for folder in folders:
#     source=os.path.join(root,folder)
root=os.path.join(root,folders[6])
trainSource=os.path.join(root,'train')
testSorce = os.path.join(root, 'test')

runDataMining(trainSource,testSorce,nrClassifiers,walksSet,walkLengthSet,windowSizeSet)
sys.exit(10)
lineReadings=readLabels()

plt.close('all')
filehandler = open(b"resultsMultipleInstances.obj", "rb")
readings=pickle.load(filehandler)
savePlot(walkLengthSet,[readings[0, 2, :, 1],readings[1, 2, :, 1],readings[2, 2, :, 1]],['walksLengthSet  KNN','walksLengthSet   closest','walksLengthSet   SVM'],"WalkLength Set Multiple")
savePlot(windowSizeSet,[readings[0, 1, 1, :],readings[1, 1, 1, :],readings[2, 1, 1, :]],['windowsSize KNN','windowSize closest','windowSize SVM'],"Windows size set multi")
savePlot(walksSet,[readings[0, :, 1, 0],readings[1, :, 1, 0],readings[2, :, 1, 0]],['walksSet KNN','walksSet closest','walksSet SVM'],"walksSet multi")


filehandler = open(b"resultsSingleInstances.obj", "rb")
readings=pickle.load(filehandler)# ax.plot(walksSet, readings[0,:,1,1], 'k--', label='walksSet length KNN')
# ax.plot(walksSet, readings[1,:,1,1], 'k', label='walksSet length closest')
savePlot(walkLengthSet,[readings[0, 2, :, 1],readings[1, 2, :, 1]],['walksLengthSet length KNN','walksLengthSet  closest'],"WalkLength Set Single")
savePlot(windowSizeSet,[readings[0, 1, 1, :],readings[1, 1, 1, :]],['windowsSize KNN','windowSize closest'],"Windows size set Single")
savePlot(walksSet,[readings[0, :, 1, 0],readings[1, :, 1, 0]],['walksSet KNN','walksSet closest'],"walksSet single")


