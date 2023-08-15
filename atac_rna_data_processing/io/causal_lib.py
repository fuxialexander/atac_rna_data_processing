#%%
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx.algorithms.community as nxcom
#%%
def get_top_edge_weight(G, n=1000):
    edge_weight_list = []
    for u, v, d in G.edges(data=True):
        edge_weight_list.append(d['weight'])
    w = np.array(edge_weight_list)
    w = np.absolute(w)
    w = w[np.argsort(w)[-n:][0]]
    return w

def set_node_community(G, communities):
    '''Add community to node attributes'''
    for c, v_c in enumerate(communities):
        for v in v_c:
            # Add 1 to save 0 for external edges
            G.nodes[v]['community'] = c + 1
def set_edge_community(G):
    '''Find internal edges and add their community to their attributes'''
    for v, w, in G.edges:
        if G.nodes[v]['community'] == G.nodes[w]['community']:
            # Internal edge, mark with community
            G.edges[v, w]['community'] = G.nodes[v]['community']
        else:
            # External edge, mark as 0
            G.edges[v, w]['community'] = 0
def get_color(i, r_off=1, g_off=1, b_off=1):
    '''Assign a color to a vertex.'''
    r0, g0, b0 = 0, 0, 0
    n = 16
    low, high = 0.5, 0.9
    span = high - low
    r = low + span * (((i + r_off) * 3) % n) / (n - 1)
    g = low + span * (((i + g_off) * 5) % n) / (n - 1)
    b = low + span * (((i + b_off) * 7) % n) / (n - 1)
    return (r, g, b)
def plot_comm(G, figsize=(10, 10), title='Network structure', savefig=False):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # Set community color for internal edges
    external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = ["#fddbc7" if (G.edges[e]['weight'] > 0) else "#d1e5f0" for e in internal]
    external_color = ["#ef8a62" if (G.edges[e]['weight'] > 0) else "#67a9cf" for e in external]
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]

    weights = [np.absolute(G[u][v]['weight'])*20 for u,v in G.edges]
    # external edges
    nx.draw(
        G,
        pos=nx.nx_agraph.graphviz_layout(G, 'neato'),
        node_size=0,
        edgelist=external,
        width = weights,
        ax=ax,
        edge_color=external_color,
        node_color=node_color,
        with_labels=True)
    # internal edges
    nx.draw(
        G, 
        pos=nx.nx_agraph.graphviz_layout(G, 'neato'),
        node_size=100,
        edgelist=internal,
        ax=ax,
        width = weights,
        edge_color=internal_color,
        node_color=node_color,
        with_labels=True)
    ax.set_title(title)
    plt.tight_layout()
    # add padding in the margins
    plt.margins(0.3, 0.3)
    if savefig:
        plt.savefig(os.path.join(savefig, title + '.png'), dpi=300)
    plt.show()
    plt.close()
def get_parents_subnet(G, n):
    '''Get the subnet of the parents of a node'''
    return G.subgraph(list(G.predecessors(n)) + [n])
def get_children_subnet(G, n):
    '''Get the subnet of the children of a node'''
    return G.subgraph(list(G.successors(n)) + [n])
def get_neighbors_subnet(G, n):
    '''Get the subnet of the neighbors of a node'''
    return G.subgraph(list(nx.all_neighbors(G,n)) + [n])
def get_subnet(G, n, type = 'neighbors'):
    '''Get the subnet of a node'''
    if type == 'parents':
        return get_parents_subnet(G, n)
    elif type == 'children':
        return get_children_subnet(G, n)
    elif type == 'neighbors':
        return get_neighbors_subnet(G, n)
    else:
        raise ValueError('type must be one of parents, children, or neighbors')

def preprocess_net(G, threshold=0.0, remove_nodes=False):
    G.remove_edges_from([(n1, n2) for n1, n2, w in G.edges(data="weight") if (w < threshold and w>-threshold)])
    G.remove_edges_from(nx.selfloop_edges(G))
    if remove_nodes:
        G.remove_nodes_from(list(nx.isolates(G)))
    communities = sorted(nxcom.greedy_modularity_communities(G), key=len, reverse=True)
    # Set node and edge communities
    set_node_community(G, communities)
    set_edge_community(G)
    return G
