"""
Graph Mining - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from random import randint
from sklearn.cluster import KMeans


# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):
    
    L = nx.normalized_laplacian_matrix(G).astype(float) # Normalized Laplacian

    # Calculate k smallest eigenvalues and corresponding eigenvectors from L
    eigval , eigvec = eigs(L,k=k, which='SR')

    eigval = eigval.real # Keep the real part
    eigvec = eigvec.real # Keep the real part
    
    idx = eigval.argsort() # Get indices of sorted eigenvalues
    eigvec = eigvec[:,idx] # Sort eigenvectors according to eigenvalues
    
    # Perform k-means clustering 
    km = KMeans(n_clusters=k)
    km.fit(eigvec)
    labels = km.labels_
    # Create a dictionary "clustering" where keys are nodes and values the clusters to which the nodes belong    
    clustering = dict()
    for i,node in enumerate(G.nodes()):
		clustering[node]= labels[i]
		    
    return clustering




G = nx.read_edgelist("CA-HepTh.txt",comments='#', delimiter='\t')
GCC = G.subgraph(max(nx.connected_components(G), key=len))
k = 50
clustering = spectral_clustering(GCC, k)
# sanity check
assert GCC.number_of_nodes() == len(clustering)




# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    m = G.number_of_edges() #number of edges of the whole graphe
    nc = len(set(clustering.values())) #number of communities
    modularity = 0
    for i in range(nc):
		list_nodes = [node for node,cluster in clustering.items() if cluster == i] #get nodes of cluster i
		subG = G.subgraph(list_nodes) #get the corresponding subgraph
		lc = subG.number_of_edges() #number of edges
		dc = np.sum([subG.degree(node) for node in subG.nodes()]) #sum of degrees
		modularity += (lc/m) -(0.5*dc/m)**2
    
    return modularity



randomClustering = dict()
for node in GCC.nodes():
  randomClustering[node] = randint(0,49)
	
print("modularity of spectral clustering is ",modularity(GCC,clustering))
print("modularity of random clustering is",modularity(GCC,randomClustering))

