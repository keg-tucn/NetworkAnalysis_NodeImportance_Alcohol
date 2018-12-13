
import numpy as np


class MatProc:
    def __init__(self):
        self.varianceMargin=2
    def binarize(self,mat):
        sum=0
        for (x,y),value in np.ndenumerate(mat):
            sum+=value
        mean=sum/(mat.shape[0]*mat.shape[1])
        variance=0
        for (x, y), value in np.ndenumerate(mat):
            variance+=pow(value-mean,2)
        variance/=(mat.shape[0]*mat.shape[1])
        mat[mat<mean-self.varianceMargin*variance]=0
        mat[mat!=0]=1
        print("Mean:",mean," variance:",variance)
        return mat