from sklearn import svm
import os
import sys
import re
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sns;
import array
class Classifs:
    def __init__(self,label,score):
        self.label=label
        self.score=score
class SVMobj:
    def __init__(self,):
        self.index = 0;
        self.labels = []
        self.LabelDict={}
        self.N=85
        self.LabelDict["Control"] = 0
        self.LabelDict["EtOH"] = 1
        self.LabelDict["Abstinence"] = 2
        self.LabelDict["Naltrexon"] = 3

        self.data = []
        #self.storeEmbedding("Control")
        #self.storeEmbedding("EtOH")

    def storeEmbedding(self,condition,srcDir):#load embeddings in memory for training

        files = os.listdir(srcDir)

        controlFiles = [x for x in files if re.match(r"EMBD_" + condition + "_[0-9]+_[0-9]+.txt", x) is not None]
        nrSamples = len(controlFiles)



        for i in range(0, nrSamples):
            with open(os.path.join(srcDir, controlFiles[i])) as file:
                w, h = [int(x) for x in next(file).split()]
                self.nrRows = w
                self.depth = h

                self.labels.insert(self.index,self.LabelDict[condition])
                r = np.zeros((self.N,self.N))
                for j in range(0, w):
                    s = [float(x) for x in next(file).split()]#take every line and remove the header(node number)
                    m = int(s.pop(0))
                   # r=np.insert(r, m, np.array(s), 0)
                    r[m]=s
                self.data.append(r)
                self.index = self.index + 1;
    def distance(self,sample, train):
        return 0
    def vectorDistance(self,a,b):#euclidean distance
        if(len(a) is not len (b)):
            raise ValueError('The shapes are not equal')
        sum=0
        for i in range(len(a)):
            sum+=pow(a[i]-b[i],2)
        if sum<0:
            raise ValueError("The distance between 2 vectors cannot be negative")
        return sqrt(sum)
    def closestNeighboor(self,sample):
        bestScore=sys.maxint;
        bestLabel=-1;
        for i,ngb in enumerate(self.data):
            tmpDistance=0
            for j in range(0,len(sample)):
                try:
                    sampleRow=sample[j]
                    ngbRow=ngb[j]
                    tmpDistance+=self.vectorDistance(sampleRow,ngbRow)
                except:
                    print("File with index "+str(i)+" is missing vector "+str(j))
            if tmpDistance<bestScore:
                bestScore=tmpDistance
                bestLabel=self.labels[i]
        return bestLabel

    def compareWithTraining(self,sample,nrNeighbors):

        scores = []
        for index in range(0,len(sample)):

            row=sample[index]

            bestLabel=-1
            bestScore = sys.maxint
            for train,label in zip(self.data,self.labels):
                try:
                    trainRow=train[index]
                except:
                    continue
                computedScore=self.vectorDistance(row, trainRow)
                if(computedScore<bestScore):
                    bestScore=computedScore
                    bestLabel=label
            scores.append(Classifs(bestLabel,bestScore));
        scores.sort(key=lambda x:x.score,reverse=True)
        votes=np.zeros(len(self.LabelDict))
        for i in range(nrNeighbors):
            votes[scores[i].label]= votes[scores[i].label]+1
        smallesScore=0
        trueLabel=-1
        for i in range(0,len(self.LabelDict)):
            if(votes[i]>smallesScore):
                smallesScore=votes[i]
                trueLabel=i
        return trueLabel


    def classifyByClosestNeighbor(self,srcDir):
        overallScore = 0
        allFiles = os.listdir(srcDir)
        for filename in allFiles:
            with open(os.path.join(srcDir, filename)) as file:
                m = re.search("EMBD_(.+?)_", filename)
                condition = m.group(1)
                w, h = [int(x) for x in next(file).split()]
                print("CLOSEST:Predicting for file " + filename);
                sample = {}
                for j in range(0, w):  # read embedding
                    s = [float(x) for x in
                         next(file).split()]  # get line by line embedding and use the appropiat classifier
                    m = int(s.pop(0))
                    sample[m] = s
                classifiedLabel = self.closestNeighboor(sample)
                print("The found label is"+str(classifiedLabel))
                isOk = classifiedLabel is self.LabelDict[condition]
                if isOk:
                    overallScore = overallScore + 1
                print("The prediction is " + str(isOk))
        accuracy = overallScore / float(len(allFiles))
        print("The acc is " + str(accuracy))
        return accuracy
    def readEmbedding(self,file_path):#returns condition,weight and mat
        with open(file_path) as file:
            m = re.search("EMBD_(.+?)_", file_path)
            try:
                condition = m.group(1)
            except:
                condition=""
            w, h = [int(x) for x in next(file).split()]
            print("Reading " + file_path);
            sample = {}
            for j in range(0, w):  # read embedding
                s = [float(x) for x in
                     next(file).split()]  # get line by line embedding and use the appropiat classifier
                m = int(s.pop(0))
                sample[m] = s
            return condition,w,sample
    def KNN(self,srcDir,nrNeighbors):
        overallScore=0
        allFiles=os.listdir(srcDir)
        for filename in allFiles :
            with open (os.path.join(srcDir,filename)) as file:
                m = re.search("EMBD_(.+?)_", filename)
                condition = m.group(1)
                w, h = [int(x) for x in next(file).split()]
                print("KNN:Predicting for file " + filename);
                sample={}
                for j in range(0, w):  # loop through all clasifiers
                    s = [float(x) for x in
                         next(file).split()]  # get line by line embedding and use the appropiat classifier
                    m = int(s.pop(0))
                    sample[m]=s
                classifiedLabel=self.compareWithTraining(sample,nrNeighbors)
                print(classifiedLabel)
                isOk=classifiedLabel is self.LabelDict[condition]
                if isOk:
                    overallScore=overallScore+1
                print("The prediction is "+str(isOk))
        accuracy=overallScore/float(len(allFiles))
        print("The acc is "+str(accuracy))
        return accuracy
    def computeParticulars(self,testingdir,srcDir):
        overallScore = 0

        all_embeddings_filename="All.txt"
        all_embeddings_filepath=os.path.join(testingdir,all_embeddings_filename)
        dontCare1,dontCare2,AllEMBD=self.readEmbedding(all_embeddings_filepath)
        allFiles = os.listdir(srcDir)
        AllEMBDlist=[AllEMBD[i] for i,m in enumerate(AllEMBD) ]
        meanSum=np.zeros((4,85,85),dtype=float)
        counts=np.zeros(4)
        for filename in allFiles:

            file_path=os.path.join(srcDir, filename)
            condition,w,sample=self.readEmbedding(file_path)
            sampleList=[sample[i] for i,m in enumerate(sample)]
            difference=np.array(sampleList)-np.array(AllEMBDlist)
            meanSum[self.LabelDict[condition]]+=difference
            counts[self.LabelDict[condition]]+=1
            # print(difference)
        for i,mat in enumerate(meanSum[0:2,:,:]):
            plt.figure()
            aux=mat/counts[i]
            plt.figure(figsize=(12, 12))
            sns.heatmap(aux)
            plt.savefig(str(i)+".png", dpi=600)

            plt.show()
            plt.close()
        control=meanSum[0,:,:]
        control/=counts[0]
        alcohol=meanSum[1,:,:]
        alcohol/=counts[1]
        diff=control-alcohol
        plt.figure()

        plt.figure(figsize=(12, 12))
        sns.heatmap(diff)
        plt.savefig(  "Cont-EtOH.png", dpi=600)

        plt.show()
        plt.close()




    def classify(self,srcDir):



        allfiles=os.listdir(srcDir)

        #allfiles = [x for x in files if re.match(r"EMBD_" + condition + "_[0-9]+_[0-9]+.txt", x) is not None]
        nrOfFiles = len(allfiles)
        meanAcc=0
        scores=np.zeros(len(self.classifiers))
        weight=np.ones(len(self.classifiers))/len(self.classifiers);
        for i in range(0, nrOfFiles):#load testing files
            with open(os.path.join(srcDir, allfiles[i])) as file:
                m=re.search("EMBD_(.+?)_",allfiles[i])
                condition=m.group(1)
                w, h = [int(x) for x in next(file).split()]
                print("Predicting for file "+allfiles[i]);
                self.nrRows = w
                self.depth = h

                good=0
                bad=0
                fileScore=0;
                label = self.LabelDict[condition]

                for j in range(0, w):#loop through all clasifiers
                    s = [float(x) for x in next(file).split()] #get line by line embedding and use the appropiat classifier
                    m = int(s.pop(0))
                    s=[s]
                    prediction=self.classifiers[m].predict(s)[0];
                    fileScore+=weight[m]*prediction;#use weights for classifiers
                    if prediction == label:
                        scores[m]=scores[m]+1
                        good=good+1
                        weight[m]+= len(self.classifiers)#increase classifier weight
                    else:
                        bad=bad+1


                weight/=sum(abs(weight))#normalize weight vector
                if(int(round(fileScore))==label):
                    meanAcc=meanAcc+1

                print "The label is "+str(self.LabelDict[condition])
                print 'I have {}'.format(int(round(fileScore)))
                print 'Raw{:0.16f}'.format(fileScore)
                print "The classifiers acc is"+'{:0.16f}'.format(float(good/(good+bad)))
               # print("The good is "+str(good)+" and bad:"+str(bad));

        print("I had a number of images:"+str(nrOfFiles))
        print("The scores are "+str(np.sort(scores)))
        print("Final acc"+str(meanAcc/float(nrOfFiles)))
        return meanAcc/float(nrOfFiles)


    def train(self):
        nrOfNodes=len(self.data[0]);
        classifiers=[]
        for i in range(0,nrOfNodes):
            clf=svm.SVC(kernel='linear')
            m=self.data[0]
            n=m[0]

            X=[X[i] for X in self.data if np.sum(abs(X[i])) is not 0]#take every i-th line from every embedding and toss out the 0 valued
            #X=[x for y in X for x in y]# reduce an unnecessary dimension

            Y=[self.labels[x] for x in range(0,len(self.labels))  ] #            Y=[self.labels[i] for i in range(0,len(self.labels))  ] #

            xarray=np.array(X)
            yarray=np.array(Y)
            if(len(xarray) != len(yarray)):
                raise ValueError("Input vectors for the SVM  do not have a similar size")
            clf.fit(xarray,yarray);
            classifiers.append(clf) #train
            #single=X[55]
            #single=[single]
            #print(clf.predict(single))
        self.classifiers= classifiers



