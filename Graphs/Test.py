from MatProc import MatProc
from Reader import Reader
import shutil
import numpy as np
import os
reader=Reader()
reader.readAll()
#mat=reader.getMatByName("P02_SCAPearson-1-Control-91-ValAtTimeOffset.csv")
#np.savetxt("original.csv",mat,delimiter=' , ')
proc=MatProc()
#mat=proc.binarize(mat)
#np.savetxt("test.csv",mat,delimiter=' , ')
#size=mat.shape;

class Mat2Graph():
    def writeAdjMatrix(self,nrSamples,condition):
        index=0
        try:
            shutil.rmtree("./tmp")
        except:
            print("TMP folder not found\n");
        os.makedirs(os.path.dirname("./tmp/"))
        ok=True;
        while(ok):
            animalId = int(index / 5) + 1#each animal has 5 readings for a state

            matsi = reader.getReadingsByConditionAndAnimal(animalId, condition)#getall mats

            print(matsi)
            for matu in matsi:

                aux = proc.binarize(matu)
                size = aux.shape
                with open("./tmp/graf" + str(index) + ".txt", 'w+') as file: #save as adj lisr
                    for i in range(0, size[1]):
                        for j in range(i + 1, size[0]):
                            if aux[i][j] == 1:
                                file.write(str(i) + " " + str(j) + "\n");
                os.system("python ./node2vec_main.py" + " --input ./tmp/graf" + str(
                    index) + ".txt" + "  --dimensions 84 --num-walks 40 --output ./embeddings/EMBD_" + condition +"_"+str(animalId)+"_"+ str(    #get the embedding
                    index) + ".txt")
                index = index + 1
                if (index == nrSamples):
                    ok = False
                    break;


'''
while(ok):
    animalId=int(index/5)+1
    condition = "Control"

    matsi=reader.getReadingsByConditionAndAnimal(animalId,condition)

    print(matsi)
    for matu in matsi:

        aux=proc.binarize(matu)
        size=aux.shape
        with open("./tmp/graf"+str(index)+".txt",'w+') as file:
            for i in range(0,size[1]):
                for j in range(i+1,size[0]):
                    if aux[i][j]==1:
                        file.write(str(i) + " " + str(j) + "\n");
        os.system("python ./node2vec_main.py"+" --input ./tmp/graf"+str(index)+".txt"+ "  --dimensions 84 --num-walks 40 --output ./tmp/embedding" +condition+str(index)+".txt")
        index=index+1
        if (index == nrControl ):
            ok=False
            break;
index=0
ok=True
while ok:
    animalId=int(index/5)+1

    condition="EtOH"
    alcohol=reader.getReadingsByConditionAndAnimal(animalId,condition)
    for matu in alcohol:
        #np.savetxt("test.csv",matu,delimiter=' , ')

        aux=proc.binarize(matu)
       # np.savetxt("out"+str(index)+".txt",aux,delimiter=' ')
        size=aux.shape
        with open("./tmp/graf"+str(index)+".txt","w+") as file:
            for i in range(0,size[1]):
                for j in range(i+1,size[0]):
                    if aux[i][j]==1:
                        file.write(str(i) + " " + str(j) + "\n");
        os.system("python ./node2vec_main.py"+"  --input ./tmp/graf"+str(index)+".txt"+ "  --dimensions 84 --num-walks 40  --output ./tmp/embedding"+condition+str(index)+".txt")
        index=index+1
        if (index == nrEtOH):
            ok=False
            break;
'''
embedder=Mat2Graph()
embedder.writeAdjMatrix(50,"Control");

embedder.writeAdjMatrix(50,"EtOH");



