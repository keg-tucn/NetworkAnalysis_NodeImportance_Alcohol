import networkx as nx
import numpy as np

G=nx.read_graphml("./1-EtOH-182.xml")
mat=nx.to_numpy_matrix(G)
T=np.array(mat)

print("Done")