from MatProc import MatProc
from Reader import Reader
import numpy as np
reader=Reader()
reader.readAll()
mat=reader.getMatByName("P02_SCAPearson-1-Control-91-ValAtTimeOffset.csv")
np.savetxt("original.csv",mat,delimiter=' , ')
proc=MatProc()
mat=proc.binarize(mat)
np.savetxt("test.csv",mat,delimiter=' , ')