"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



G = nx.read_edgelist('../datasets/CA-HepTh.txt',comments='#',delimiter='\t')
nb_nodes = G.number_of_nodes()
nb_edges = G.number_of_edges()

print('number of nodes of our graph is ',nb_nodes)
print('number of edges of our graph is ',nb_edges)





CC = nx.connected_components(G) #get connected components of our graph
nb_CC = nx.number_connected_components(G)

print('number of connected components of our graph is ',nb_CC)

largest_cc = G.subgraph(max(CC, key=len)) #get largest connected component of G
nb_nodes_cc = largest_cc.number_of_nodes()
nb_edges_cc = largest_cc.number_of_edges()

print("nodes fraction of the whole graph: ",nb_nodes_cc/nb_nodes)
print("edges fraction of the whole graph: ",nb_edges_cc/nb_edges)



# Degree
degree_sequence = [G.degree(node) for node in G.nodes()]

print('maximum degree is ', np.max(degree_sequence))
print('minimum degree is ', np.min(degree_sequence))
print('mean degree is ', np.mean(degree_sequence))



#Histogram Plots

y = nx.degree_histogram(G)
plt.plot(y,'b-', marker='o')
plt.title("Degree Histogram - Normal Scale")
plt.ylabel("frequency")
plt.xlabel("degree")
plt.show()

y = nx.degree_histogram(G)
plt.loglog(y,'b-', marker='o')
plt.title("Degree Histogram - Log Scale")
plt.ylabel("frequency")
plt.xlabel("degree")
plt.show()

