from sklearn import svm
import os
import re
import numpy as np
import array
class SVMobj:
    def __init__(self,animalId):
        dict=[]
        index=0;
        controlLimit=42;
        etohLimit=42;
        85

        files=os.listdir("./tmp/")
        controlFiles=[x for x in files if re.match(r"embedding"+"Control"+"[0-9]+.txt",x) is not None ]
        nrControl=len(controlFiles)

        nr_features = 84

        labels=range(84)
        for i in range(0,controlLimit):
            with open(os.path.join("./tmp/",controlFiles[i])) as file:
                w, h = [int(x) for x in next(file).split()]
                self.nrRows=w
                self.depth=h

                labels[index]=0
                r=range(w)
                for j in range(0,w):
                    s=[float(x) for x in next(file).split()]
                    m=int(s.pop(0))
                    r[m]=s
                dict.append(r)
                index=index+1;
        etohFiles = [x for x in files if re.match(r"embedding"+"EtOH"+"[0-9]+.txt",x) is not None]
        nrEtOH = len(etohFiles)
        p=len(dict)
        for i in range(0, etohLimit):
            with open(os.path.join("./tmp/",etohFiles[i])) as file:
                w, h = [int(x) for x in next(file).split()]
                self.nrRows = w
                self.depth = h

                labels[index]=1
                r = range(w)

                for j in range(0,w):
                    s=[float(x) for x in next(file).split()]
                    m=int(s.pop(0))
                    r[m]=s

                dict.append(r)
                index = index + 1;
        p=len(dict)

        self.data = dict
        self.labes=labels
    def classify(self,classifiers):
        condition="Control"
        label=0;
        limit=43
        files=os.listdir("./tmp/")

        allfiles = [x for x in files if re.match(r"embedding" + condition + "[0-9]+.txt", x) is not None]
        nrOfFiles = len(allfiles)
        meanAcc=0
        for i in range(limit, nrOfFiles):
            with open(os.path.join("./tmp/", allfiles[i])) as file:
                w, h = [int(x) for x in next(file).split()]
                print("Predictinf for file "+allfiles[i]);
                self.nrRows = w
                self.depth = h

                r = range(w)
                good=1
                bad=1
                for j in range(0, w):
                    s = [float(x) for x in next(file).split()]
                    m = int(s.pop(0))
                    s=[s]
                    if classifiers[m].predict(s)==label:
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

            X=[[X[i]] for X in self.data]
            X=[x for y in X for x in y]
            Y=self.labes
            xarray=np.array(X)
            yarray=np.array(Y)
            clf.fit(xarray,yarray);
            classifiers.append(clf)
            #single=X[55]
            #single=[single]
            #print(clf.predict(single))
        return classifiers



obj=SVMobj(4)

classifiers=obj.train()
obj.classify(classifiers)
