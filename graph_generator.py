import numpy as np
import shutil

import WeightDistributionArrays
import sys
import os
import random
import networkx as nx
import Arrays


nr_graphs = 0
distribution = []
folder = ""

state = ""

nr_nodes = 85


# generates adjacency matrix for a complete graph with the edges distribution
# returns the adjacency matrix as a numpy array
def generate_graph():
    mat = np.zeros(shape=(nr_nodes, nr_nodes))

    nodes1 = Arrays.orderd_control_isolates  # list(range(nr_nodes))
    nodes2 = nodes1[::-1]

    nodes1 = nodes1[0:43]
    nodes2 = nodes2[0:42]

    nodes = [0] * 85
    nodes[::2] = nodes1
    nodes[1::2] = nodes2

    # nodes = randomize_back_to_front(nodes, 10, 30, 50)
    # nodes = randomize_local_shuffle(nodes, 7, 4)

    # random.shuffle(nodes)
    # print(nodes)

    """
    for each node:
        get randomly a "next node"
        generate weight using the aggregate distributions
        continue until no nodes left
    """

    for i in range(len(nodes)):
        this = nodes[i]
        # print("i=" + str(this))
        temp = nodes[i+1:]

        while len(temp) != 0:
            j = random.randint(0, len(temp) - 1)
            # print(j)

            neighb = temp[j]
            del temp[j]

            if mat[this, neighb] != 0:
                continue

            newdist = (distribution[:, this])  # + distribution[:, neighb]) / 2

            gen = np.random.choice(range(0, 10), p=newdist)
            min = gen * 0.2 - 1

            w = random.uniform(min, min + 0.2)

            mat[this, neighb] = w
            mat[neighb, this] = w

    return mat


# a shuffle in which elements don't usually go very far from their original positions
# (len(vec) - window_size) % (window_size - overlap) = 0
# so that you can divide in neat windows
def randomize_local_shuffle(vec, window_size, overlap):
    check = (len(vec) - window_size) % (window_size - overlap)
    if check != 0:
        return None

    step = window_size - overlap

    newvec = list(vec)  # copy

    for i in range(0, len(vec) - window_size + 1, step):
        newvec[i:i + window_size] = random.sample(newvec[i:i + window_size], window_size)

    return newvec


# bring a nr of samples (back_samples) from the back (last back_zone_percentage % nodes)
# to the front (front_zone_percentage)
def randomize_back_to_front(vec, back_samples, front_zone_percentage, back_zone_percentage):
    front_zone_percentage /= 100
    back_zone_percentage /= 100

    back_indices = np.random.choice(list(range(int(back_zone_percentage * len(vec)), len(vec))), back_samples)
    front_indices = np.random.choice(list(range(0, int(front_zone_percentage * len(vec)))), back_samples)

    newvec = list(vec)  # copy

    for i in range(len(back_indices)):
        newvec[front_indices[i]], newvec[back_indices[i]] = newvec[back_indices[i]], newvec[front_indices[i]]

    return newvec


def setupFolder(folder):
    shutil.rmtree(folder, ignore_errors=True)
    try:
        os.mkdir(folder)
    except OSError:
        print ("Creation of the directory %s failed" % folder)
    else:
        print ("Successfully created the directory %s " % folder)
if __name__ == "__main__":

    # check args
    if len(sys.argv) != 4:
        print("Usage: python graph_generator nr_graphs [-c|-a] folder")
        exit(0)

    # parse args
    nr_graphs = int(sys.argv[1])

    if sys.argv[2] == "-c":
        distribution = np.array(WeightDistributionArrays.Control)
        state = "Control"

    else:
        if sys.argv[2] == "-a":
            distribution = np.array(WeightDistributionArrays.EtOH)
            state = "EtOH"

        else:
            print("Wrong option, valid options: -c for Control, -a for EtOH")
            exit(0)

    folder = sys.argv[3]

    # check folder and create if nonexistent
    setupFolder(folder)

    os.chdir(folder)

    # generate graphs
    for i in range(1, nr_graphs + 1):
        g = generate_graph()

        # save graph
        with open(str(i) + "-" + state + ".csv", "wb") as f:
            np.savetxt(f, g, delimiter=',')
            nx.write_graphml(nx.Graph(g), str(i) + "-" + state + "-" +str(i)+".xml")
