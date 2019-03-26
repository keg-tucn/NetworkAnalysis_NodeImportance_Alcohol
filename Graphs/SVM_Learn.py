from sklearn import svm
import os
import sys
import re
import numpy as np
from math import sqrt
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
        return sqrt(sum)


    def compareWithTraining(self,sample):
        nrNeighbors=30
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




    def KNN(self,srcDir):
        overallScore=0
        allFiles=os.listdir(srcDir)
        for filename in allFiles :
            with open (os.path.join(srcDir,filename)) as file:
                m = re.search("EMBD_(.+?)_", filename)
                condition = m.group(1)
                w, h = [int(x) for x in next(file).split()]
                print("Predicting for file " + filename);
                sample={}
                for j in range(0, w):  # loop through all clasifiers
                    s = [float(x) for x in
                         next(file).split()]  # get line by line embedding and use the appropiat classifier
                    m = int(s.pop(0))
                    sample[m]=s
                classifiedLabel=self.compareWithTraining(sample)
                print(classifiedLabel)
                isOk=classifiedLabel is self.LabelDict[condition]
                if isOk:
                    overallScore=overallScore+1
                print("The prediction is "+str(isOk))
        print("The acc is "+str(overallScore/float(len(allFiles))))




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

                r = range(w)
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


                weight/=sum(weight)#normalize weight vector
                if(int(round(fileScore))==label):
                    meanAcc=meanAcc+1

                print "The label is "+str(self.LabelDict[condition])
                print '{}'.format(int(round(fileScore)))
                print 'Raw{:0.16f}'.format(fileScore)
                print "The classifiers acc is"+'{:0.16f}'.format(float(good/(good+bad)))
               # print("The good is "+str(good)+" and bad:"+str(bad));

        print("I had a number of images:"+str(nrOfFiles))
        print("The scores are "+str(np.sort(scores)))
        print("Final acc"+str(meanAcc/float(nrOfFiles)))



    def train(self):
        nrOfNodes=len(self.data[0]);
        classifiers=[]
        for i in range(0,nrOfNodes):
            clf=svm.SVC(kernel='poly',degree=4)
            m=self.data[0]
            n=m[0]

            X=[X[i] for X in self.data if np.sum(X[i]) is not 0]#take every i-th line from every embedding and toss out the 0 valued
            #X=[x for y in X for x in y]# reduce an unnecessary dimension

            Y=[self.labels[x] for x in range(0,len(self.labels))  ] #            Y=[self.labels[i] for i in range(0,len(self.labels))  ] #

            xarray=np.array(X)
            yarray=np.array(Y)
            clf.fit(xarray,yarray);
            classifiers.append(clf) #train
            #single=X[55]
            #single=[single]
            #print(clf.predict(single))
        self.classifiers= classifiers



obj=SVMobj()
obj.storeEmbedding("Control", "./training/embeddings/")
obj.storeEmbedding("EtOH", "./training/embeddings/")
#obj.storeEmbedding("Abstinence", "./training/embeddings/")
obj.KNN("./testing/embeddings")
# classifiers=obj.train()
# obj.classify("./testing/embeddings")
