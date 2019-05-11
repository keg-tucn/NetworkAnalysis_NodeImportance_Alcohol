import time

from sklearn.manifold import TSNE

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
import gc
import matplotlib.pyplot as plt
#mat=reader.getMatByName("P02_SCAPearson-1-Control-91-ValAtTimeOffset.csv")
#np.savetxt("original.csv",mat,delimiter=' , ')

#mat=proc.binarize(mat)
#np.savetxt("test.csv",mat,delimiter=' , ')
#size=mat.shape;

dimensions=35
N=85
class Mat2Graph():
    def __init__(self):
        self.models=[None,None,None,None,None]
        self.LabelDict={}
        self.LabelDict["Control"] = 0
        self.LabelDict["EtOH"] = 1
        self.LabelDict["Abstinence"] = 2
        self.LabelDict["Naltrexon"] = 3
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
    def trainModel(self,modelIndex,windowSize,walks):
        currentModel = self.models[modelIndex]
        if (currentModel is None):
            currentModel = Word2Vec(walks, size=dimensions, window=windowSize, min_count=0, sg=1, workers=6,
                                    iter=1)
        else:
            currentModel.train(walks, total_examples=len(walks), epochs=10)
        self.models[modelIndex] = currentModel
    def trainModelsForConditions(self,conditions,outFolder,walkLength,nrWalks,windowSize):
        index = 0
        print("Running trainModelsForConditions")
        gc.collect()
        for condition in conditions:
            matsi = reader.getAllByCondition(condition)  # getall mats
            allWalks = []

            for [matu, fileName] in matsi:
                print("Processing file " + str(fileName))
                m = re.search(condition + "-(.+?)-", fileName)
                trial = m.group(1)
                aux = matu
                np.savetxt(outFolder + trial, aux, fmt="%f", delimiter=',')  # just for testing
                grafFileName = outFolder + "graf" + str(trial) + ".txt"
                allWalks.append(
                    getGraphWalks(grafFileName, dimensions, directed=False, walk_length=walkLength, num_walks=nrWalks,
                                  weighted=True))
                index = index + 1
            walks = [item for x in allWalks for item in x]
            walks = [map(str, walk) for walk in walks]
            self.trainModel(self.LabelDict[condition],windowSize,walks)
            self.trainModel(4, windowSize,walks)#train the last model because he is the bigger one


    def writeAdjMatrixForCondition(self,condition,outFolder,walkLength,nrWalks,windowSize):
        index=0

        gc.collect()
        matsi = reader.getAllByCondition(condition)#getall mats
        allWalks=[]
        for [matu,fileName] in matsi:
            print("Processing file "+str(fileName))
            m=re.search("SCAPearson-(.+?)-",fileName)#get animal id and trial number
            animalId=m.group(1)
            m = re.search(condition+"-(.+?)-", fileName)
            trial = m.group(1)
            aux = matu
            np.savetxt(outFolder+trial,aux,fmt="%f",delimiter=',')#just for testing
            size = aux.shape
            grafFileName = outFolder + "graf" + str(trial) + ".txt"
            with open(grafFileName, 'w+') as file: #save as adj lisr
                for i in range(0, size[0]):
                    for j in range(i + 1, size[1]):
                        if aux[i][j] is not 0:
                             file.write(str(i) + " " + str(j) +" "+str(aux[i][j])+ "\n");

            output=outFolder+"embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(trial) + ".txt"
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
            index = index + 1
            self.learn_embeddings(allWalks,dimensions,windowSize=windowSize,nrWorkers=16,nrIterations=1)
        try:
            model.wv.save_word2vec_format(output)
        except:
            print("Something is not good with your trained model or it doesn't exist")

    def writeGlobalEmbedding(self, condition, outFolder, walkLength, nrWalks, windowSize):
        index = 0
        global model


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
                # print("Starting new execution",i, j, k)
                #
                # start = time.time()
                # reader.readings=None
                # read(trainSource, 2)
                # createEmbeddings(embedder.writeAdjMatrixForCondition, './training/', trainSource,
                #                  ["Control", "EtOH", "Abstinence"], walkLength=walkLength, nrWalks=nrWalks, windowSize=windowSize)
                # reader.readings = None
                # read(testSource, 2)
                # createEmbeddings(embedder.writeAdjMatrixForCondition, './testing/', testSource,
                #                  ["Control", "EtOH", "Abstinence"], walkLength=walkLength, nrWalks=nrWalks, windowSize=windowSize)
                #
                # print("Finished with ",i, j, k)
                # print("Elapsed time ",str(int(time.time()-start)))

                obj = SVMobj(N,dimensions)
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
                filehandler = open(b"resultsMultipleInstances.obj", "wb")

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
            readu.append(   longName)

    print(readu)
    return readu;
def createProbabilityMatrix(model):
    mat = np.zeros(shape=(N,N))
    for node in range(0,N):
        row=model.wv.most_similar(positive=[str(node)],topn=N)
        normalize_constant=sum([elem for index,elem in row])
        for index,val in row:
            mat[node,int(index)]=val/float(normalize_constant)
    return mat


def display_closestwords_tsnescatterplot(model, word):
    arr = np.empty((0, 300), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.similar_by_word(word)

    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)

    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min() + 0.00005, x_coords.max() + 0.00005)
    plt.ylim(y_coords.min() + 0.00005, y_coords.max() + 0.00005)
    plt.show()
def autolabel(ax,rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{}'.format(height), ha=ha[xpos], va='bottom')
def getNodesProbabilities(mat):
    newMat=[sum(mat[:,col]) for col in range(0,mat.shape[1]) ]
    return newMat
def test():
    global model
    root = "./Dataset_without_time/sum_weight_high_edge_values/sum_weight_70_redus"

    # readLabels()
    wantClassify=True
    wantNewData=False
    if(wantNewData):
        trainSource = os.path.join(root, 'train')
        read(trainSource, 2)
        model = None
        createEmbeddings(embedder.writeAdjMatrixForCondition, './training/', trainSource,
                         ["Control", "EtOH", "Abstinence"], walkLength=15, nrWalks=40,
                         windowSize=7)


        testSorce = os.path.join(root, 'test')
        read(testSorce, 2)
        model = None
        createEmbeddings(embedder.writeAdjMatrixForCondition, './testing/', testSorce,
                         ["Control", "EtOH", "Abstinence"], walkLength=15, nrWalks=40,
                         windowSize=7)
    if(wantClassify):
        trainSource = os.path.join(root, 'train')
        read(trainSource, 2)
        embedder.trainModelsForConditions(["Control","EtOH","Abstinence"],"./training/",10,10,5)
        controlMatrix=createProbabilityMatrix(embedder.models[0])
        etohMatrix = createProbabilityMatrix(embedder.models[1])
        controlProbs=getNodesProbabilities(controlMatrix)

        ethohProbs=getNodesProbabilities(etohMatrix)


        testPlot(controlProbs,ethohProbs,"Control-EtOH")
        plotToTalProbabilitiesDiss(controlProbs,ethohProbs,"Control-EtOH_2bars")
        display_closestwords_tsnescatterplot(embedder.models[0],str(1))
        # obj = SVMobj(N,dimensions)
        # controlMatrix=createProbabilityMatrix(embedder.models[0])
        # etohMatrix = createProbabilityMatrix(embedder.models[1])
        # generalMatrix=createProbabilityMatrix(embedder.models[4])
        # diff=controlMatrix-etohMatrix;
        #
        # obj.create_heatmap_cam_2d(controlMatrix,"Control", readLabels())
        # obj.create_heatmap_cam_2d(etohMatrix,"EtOH", readLabels())
        # obj.create_heatmap_cam_2d(diff,"COntrol-EtOH", readLabels())
        #
        # obj.create_heatmap_cam_2d(controlMatrix-generalMatrix, "C0ntrol-General", readLabels())
        # obj.create_heatmap_cam_2d(etohMatrix-generalMatrix, "EtOH general", readLabels())




        #
        # obj.storeEmbedding("Control", "./training/embeddings/")
        # obj.storeEmbedding("EtOH", "./training/embeddings/")
        # obj.storeEmbedding("Abstinence", "./training/embeddings/")
        # # obj.train()
        # # obj.classify("./testing/embeddings/")
        # obj.computeParticulars(readLabels(), "./training/embeddings", "./testing/embeddings/")
def plotToTalProbabilitiesDiss(F1,F2,output):
    objects =readLabels()
    fig, ax = plt.subplots(figsize=(16, 16))
    y_pos = np.arange(len(objects))
    F1=np.array(F1)
    F2=np.array(F2)
    performance = F1-F2

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects,    rotation = 'vertical',fontsize=8)
    plt.ylabel('Usage')
    plt.title('Programming language usage')
    plt.savefig( output, dpi=600)
    plt.show()


def testPlot(F1,F2,output):
    n_groups = len(F1)


    # create plot
    fig, ax = plt.subplots(figsize=(16,16))
    index = np.arange(n_groups)
    bar_width = 0.55
    opacity = 0.8

    rects1 = plt.bar(index, F1, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Frank')

    rects2 = plt.bar(index + bar_width, F2, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Guido')

    plt.xlabel('Node')
    plt.ylabel('Total Probabilities')
    plt.title('Scores of total probabilities')

    plt.xticks(index + bar_width, readLabels(),    rotation = 'vertical')
    plt.legend()

    plt.tight_layout()
    plt.savefig( output, dpi=600)

    plt.show()
proc=MatProc()
reader = Reader(N,dimensions)
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
# testPlot()
test()
sys.exit(2)
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


