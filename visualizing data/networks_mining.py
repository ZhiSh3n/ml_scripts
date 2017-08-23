import networkx as nx
from matplotlib import pylab
from pylab import *
G = nx.read_gml('lesmiserables.gml', relabel=True)
#nx.draw_networkx(G,node_size=0,edge_color='b',alpha=.2,font_size=7)
#show()

# we can study the degree of nodes
deg = nx.degree(G)
print(deg)
print(list(deg.values()))
print(min(list(deg.values())))
print(percentile(list(deg.values()),25)) #first percentile
print(median(list(deg.values())))
print(percentile(list(deg.values()),75))
print(max(list(deg.values())))

# we can choose to only select the characters that have a degree > 10
# so these are relatively main charcters

Gt = G.copy()
dn = nx.degree(Gt)
for n in Gt.nodes():
 if dn[n] <= 10:
  Gt.remove_node(n)
nx.draw_networkx(Gt,node_size=0,edge_color='b',alpha=.2,font_size=12)
show()

# so here are are observing the most relevant characters and their relationships
# we can also find cliques
from networkx import find_cliques
cliques = list(find_cliques(G))
print(max(cliques, key=lambda l: len(l)))
# print the biggest clique

""" other resources
opencv
pandas
scipy
statsmodels
nltk
ipython
"""
