"""
Deep Learning on Graphs - ALTEGRAD - Dec 2019
"""

import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk

# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i,0]] = class_labels[i,1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)


nx.draw_networkx(G, node_color=y)
plt.show()


n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G,n_walks,walk_length,n_dim)# your code here

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i,:] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train,:]
X_test = embeddings[idx_test,:]

y_train = y[idx_train]
y_test = y[idx_test]


clf = LogisticRegression()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print('accuracy using deep walk embeddings is ', accuracy_score(y_test,y_pred))


sp = SpectralEmbedding()#(n_components=128)
sp_embeddings = sp.fit_transform(nx.to_numpy_matrix(G))

X_sp_train = sp_embeddings[idx_train,:]
X_sp_test = sp_embeddings[idx_test,:]

clf_sp = LogisticRegression()
clf_sp.fit(X_sp_train,y_train)
y_sp_pred = clf_sp.predict(X_sp_test)
print('accuracy using Spectral embedding is ', accuracy_score(y_test,y_sp_pred))
