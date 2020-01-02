"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Generate simple dataset
def create_dataset():
    Gs = list()
    y = list()

    for n in range(3,103):
      Gs.append(nx.cycle_graph(n))
		  y.append(0)
		  Gs.append(nx.path_graph(n))
		  y.append(1)
    return Gs, y


Gs, y = create_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.1)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):    
    all_paths = dict()
    sp_counts_train = dict()
    
    for i,G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)
                        
    sp_counts_test = dict()

    for i,G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(G_train), len(all_paths)))
    for i in range(len(G_train)):
        for length in sp_counts_train[i]:
            phi_train[i,all_paths[length]] = sp_counts_train[i][length]
    
  
    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i,all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test



# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0,1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0,1)
    graphlets[2].add_edge(1,2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0,1)
    graphlets[3].add_edge(1,2)
    graphlets[3].add_edge(0,2)

    
    phi_train = np.zeros((len(G_train), 4))
    
    for i in range(len(G_train)):
		sampled_nodes = [random.choices(list(G_train[i].nodes()),k=3) for k in range(n_samples)]
		subGraphs = [G_train[i].subgraph(s) for s in sampled_nodes]
		for j in range(4):
			phi_train[i,j] = np.sum([int(nx.is_isomorphic(subG,graphlets[j])) for subG in subGraphs])


    phi_test = np.zeros((len(G_test), 4))
    
    for i in range(len(G_test)):
		sampled_nodes = [random.choices(list(G_test[i].nodes()),k=3) for k in range(n_samples)]
		subGraphs = [G_test[i].subgraph(s) for s in sampled_nodes]
		for j in range(4):
			phi_test[i,j] = np.sum([int(nx.is_isomorphic(subG,graphlets[j])) for subG in subGraphs])

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

K_train_gl, K_test_gl = graphlet_kernel(G_train, G_test)



#initialize SVM and train

clf_sp = SVC(kernel='precomputed')
clf_gl = SVC(kernel='precomputed')

clf_sp.fit(K_train_sp,y_train)
clf_gl.fit(K_train_gl,y_train)

y_sp_pred = clf_sp.predict(K_test_sp)
y_gl_pred = clf_gl.predict(K_test_gl)

print("accuracy score for short path kernel is ",accuracy_score(y_test,y_sp_pred))
print("accuracy score for graphlet kernel is ",accuracy_score(y_test,y_gl_pred))
