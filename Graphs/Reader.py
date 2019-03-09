from os import listdir
from os.path import isfile, join
import csv
import re
import numpy as np
import networkx as nx
class Reader:


    def readAll(self,srcDir):
        self.directory=srcDir
        onlyfiles = [f for f in listdir(self.directory) if isfile(join(self.directory, f)) and f.endswith('.csv')]
        dictionary={}
        for file in onlyfiles:
            mat=np.loadtxt(open(join(self.directory,file), "rb"), delimiter=",", skiprows=0)
            dictionary[file]=mat
        self.readings=dictionary

    def readAll2(self,srcDir,environments):
        self.directory=srcDir
        dictionary={}
        for env in environments:
            currDir=join(srcDir,env)
            onlyfiles = [f for f in listdir(currDir) if isfile(join(currDir, f)) and f.endswith('.xml')]
            for file in onlyfiles:
                finePath=join(currDir,file)
                G = nx.read_graphml(finePath)
                m = re.search( "(.+?)-", file)
                animalId = m.group(1)
                m = re.search(env+"-(.+?).xml", file)
                trial = m.group(1)
                oldName="P02_SCAPearson-"+str(animalId)+"-"+env+"-"+str(trial)+"ValAtTimeOffset.csv"
                mat=nx.to_numpy_matrix(G,nodelist=range(0,85))
                dictionary[file]=mat
                np.savetxt("Testule",mat)
                print("Good")
        self.readings=dictionary

        for file in onlyfiles:
            mat=np.loadtxt(open(join(self.directory,file), "rb"), delimiter=",", skiprows=0)
            dictionary[file]=mat
        self.readings=dictionary
    def getMat(self,method,animalCode,condition,id):
        return self.readings[method+animalCode+condition+id]
    def getMatByName(self,name):
        return self.readings[name]
 #read a specific mat belonging to an animal and environment
    def getAllByCondition(self,condition):
        mats=[[self.readings[x],x] for x in self.readings.keys() if condition in str(x) ]
        return mats
    def getReadingsByConditionAndAnimal(self,animalId,condition):
        mats=self.readings
        readings=[self.readings[x] for x in mats.keys()  if re.match(r"P02_SCAPearson-"+str(animalId)+"-"+condition+"-[0-9]+-ValAtTimeOffset.csv",x) is not None ]
        print(readings)
        return readings;

R=Reader()
R.readAll2("./sum_weight_high_edge_values/sum_weight_40/",["EtOH","Control"])