import string
from nltk.corpus import stopwords
import igraph
import numpy as np
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
#import os
#os.chdir() # to change working directory to where functions live
# import custom functions
from library import clean_text_simple, terms_to_graph, core_dec

stpwds = stopwords.words('english')
punct = string.punctuation.replace('-', '')

my_doc = 'A method for solution of systems of linear algebraic equations \
with m-dimensional lambda matrices. A system of linear algebraic \
equations with m-dimensional lambda matrices is considered. \
The proposed method of searching for the solution of this system \
lies in reducing it to a numerical system of a special kind.'

my_doc = my_doc.replace('\n', '')

# pre-process document
my_tokens = clean_text_simple(my_doc,my_stopwords=stpwds,punct=punct)

g = terms_to_graph(my_tokens, 4)

# number of edges
print(len(g.es))

# the number of nodes should be equal to the number of unique terms
len(g.vs) == len(set(my_tokens))

edge_weights = []
for edge in g.es:
    source = g.vs[edge.source]['name']
    target = g.vs[edge.target]['name']
    weight = edge['weight']
    edge_weights.append([source, target, weight])

print(edge_weights)
densities= []
windows = np.arange(2,30)
for w in range(2,30):
    g = terms_to_graph(my_tokens, w)
    densities.append(g.density())
    print("For window ", w, " density of the graph is ", g.density())

plt.plot(windows, densities)
plt.xlabel('size of sliding window')
plt.ylabel('Graphs density')
plt.title("Evolution of graph's density w.r.t window size")
plt.show()
#plot for window = 4
test = terms_to_graph(my_tokens, 4)
labels = test.vs["name"]

test.write_svg('test')
igraph.plot(test, vertex_label = labels)
# decompose g
core_numbers = core_dec(g,False)
print("core numbers using implemented function is ",core_numbers)
print("core numbers using igraph method is ",g.coreness())
### fill the gap (compare 'core_numbers' with the output of the .coreness() igraph method) ###

# retain main core as keywords
max_c_n = max(core_numbers.values())
keywords = [kwd for kwd, c_n in core_numbers.items() if c_n == max_c_n]
print(keywords)
