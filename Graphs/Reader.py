from os import listdir
from os.path import isfile, join
import csv
import re
import numpy as np
class Reader:
    def __init__(self):
        self.directory="Readings/P02_SCAPearson"
    def readAll(self):
        onlyfiles = [f for f in listdir(self.directory) if isfile(join(self.directory, f)) and f.endswith('.csv')]
        dictionary={}
        for file in onlyfiles:
            mat=np.loadtxt(open(join(self.directory,file), "rb"), delimiter=",", skiprows=1)
            dictionary[file]=mat
        self.readings=dictionary
    def getMat(self,method,animalCode,condition,id):
        return self.readings[method+animalCode+condition+id]
    def getMatByName(self,name):
        return self.readings[name]

    def getReadingsByConditionAndAnimal(self,animalId,condition):
        mats=self.readings
        readings=[self.readings[x] for x in mats.keys()  if re.match(r"P02_SCAPearson-"+str(animalId)+"-"+condition+"-[0-9]+-ValAtTimeOffset.csv",x) is not None ]
        print(readings)
        return readings;