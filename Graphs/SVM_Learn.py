from sklearn import svm
import os
import re
import numpy as np
import array
class SVMobj:
    def __init__(self,):
        self.index = 0;
        self.labels = []
        self.LabelDict={}

        self.LabelDict["Control"] = 0
        self.LabelDict["EtOH"] = 1

        self.data = []
        self.storeEmbedding("Control")
        self.storeEmbedding("EtOH")

    def storeEmbedding(self,condition):

        files = os.listdir("./embeddings/")

        controlFiles = [x for x in files if re.match(r"EMBD_" + condition + "_[0-9]+_[0-9]+.txt", x) is not None]
        nrSamples = len(controlFiles)



        for i in range(0, nrSamples):
            with open(os.path.join("./embeddings/", controlFiles[i])) as file:
                w, h = [int(x) for x in next(file).split()]
                self.nrRows = w
                self.depth = h

                self.labels.insert(self.index,self.LabelDict[condition])
                r = range(w)
                for j in range(0, w):
                    s = [float(x) for x in next(file).split()]#take every line and remove the header(node number)
                    m = int(s.pop(0))
                    r[m] = s
                self.data.append(r)
                self.index = self.index + 1;

    def classify(self):
        condition="Control"
        label=0;
        limit=43
        EMBD_dir="./embeddings"
        files=os.listdir("./embeddings/")

        allfiles = [x for x in files if re.match(r"EMBD_" + condition + "_[0-9]+_[0-9]+.txt", x) is not None]
        nrOfFiles = len(allfiles)
        meanAcc=0
        for i in range(limit, nrOfFiles):
            with open(os.path.join(EMBD_dir, allfiles[i])) as file:
                w, h = [int(x) for x in next(file).split()]
                print("Predicting for file "+allfiles[i]);
                self.nrRows = w
                self.depth = h

                r = range(w)
                good=1
                bad=1
                for j in range(0, w):
                    s = [float(x) for x in next(file).split()]
                    m = int(s.pop(0))
                    s=[s]
                    if self.classifiers[m].predict(s)==label:
                        good=good+1
                    else:
                        bad=bad+1
                m=good/float((good+bad))
                meanAcc+=m;
                print '{0:.16f}'.format(m)
                print("The result is "+str(good)+" and bad:"+str(bad));
        print("FInal acc"+str(meanAcc/float(nrOfFiles-limit)))



    def train(self):
        nrOfNodes=len(self.data[0]);
        classifiers=[]
        for i in range(0,nrOfNodes):
            clf=svm.SVC(kernel='poly',degree=4)
            m=self.data[0]
            n=m[0]

            X=[[X[i]] for X in self.data]#take every i-th line from every embedding
            X=[x for y in X for x in y]# reduce an unnecessary dimension
            Y=self.labels
            xarray=np.array(X)
            yarray=np.array(Y)
            clf.fit(xarray,yarray);
            classifiers.append(clf) #train
            #single=X[55]
            #single=[single]
            #print(clf.predict(single))
        self.classifiers= classifiers



obj=SVMobj()

classifiers=obj.train()
obj.classify()
