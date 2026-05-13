# -*- coding: utf-8 -*-
"""
Created on Tue Dec 21 11:06:48 2021

@author: gafrecon
"""
###
import os
import torch
import numpy as np
import networkx as nx
from torch_scatter import scatter_add
from torch_geometric.data import Data
from sklearn.preprocessing import StandardScaler, RobustScaler


def graph_loop(n_patients, classical, list_centers, radiomics, Y, a, parameters):
    G_list = []
    data_list = []
    # sd1 = 0.9324149289534428
    # sd2 = 2.419077652293398

    #Getting the stds for the edge weights computation:
    dist, dist_feat = [],[] 
    for i in range(n_patients):
        n_ROI = len(list_centers[i])
        for c in range(n_ROI):
            for d in range(n_ROI):
                if(c!=d):
                    dist.append(np.linalg.norm(np.array(list_centers[i][c]) - np.array(list_centers[i][d])))
                    dist_feat.append(np.linalg.norm(np.array([*classical[i][c], *radiomics[i][c]]) - np.array([*classical[i][d], *radiomics[i][d]])))
    sd1 = np.std(np.array(dist)) 
    sd2 = np.std(np.array(dist_feat)) 


    for i in range(n_patients):
        n_ROI = len(classical[i])
        if(parameters.snode):
            G, data = make_graph_supernode(n_ROI, classical[i], list_centers[i], radiomics[i], Y[i], sd1, sd2, a)
        else:
            G, data = make_graph(n_ROI, classical[i], list_centers[i], radiomics[i], Y[i], sd1, sd2, a, parameters.sep_edge_weights)

        G_list.append(G)
        data_list.append(data)
    return G_list, data_list





def homophily(edge_index, edge_attr, y, indices, method: str = 'edge', all_pat=False):
    r"""The homophily of a graph characterizes how likely nodes with the same
    label are near each other in a graph.
    There are many measures of homophily that fits this definition.
    In particular:

    - In the `"Beyond Homophily in Graph Neural Networks: Current Limitations
      and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper, the
      homophily is the fraction of edges in a graph which connects nodes
      that have the same class label:

      .. math::
        \text{homophily} = \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge
        y_v = y_w \} | } {|\mathcal{E}|}

      That measure is called the *edge homophily ratio*.

    - In the `"Geom-GCN: Geometric Graph Convolutional Networks"
      <https://arxiv.org/abs/2002.05287>`_ paper, edge homophily is normalized
      across neighborhoods:

      .. math::
        \text{homophily} = \frac{1}{|\mathcal{V}|} \sum_{v \in \mathcal{V}}
        \frac{ | \{ (w,v) : w \in \mathcal{N}(v) \wedge y_v = y_w \} |  }
        { |\mathcal{N}(v)| }

      That measure is called the *node homophily ratio*.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        y (Tensor): The labels.
        method (str, optional): The method used to calculate the homophily,
            either :obj:`"edge"` (first formula) or :obj:`"node"`
            (second formula). (default: :obj:`"edge"`)
    """
    assert method in ['edge', 'node']
    # y = y.squeeze(-1) if y.dim() > 1 else y

    # if isinstance(edge_index, SparseTensor):
    #     col, row, _ = edge_index.coo()
    # else:
    #     row, col = edge_index
    row, col = edge_index

    if method == 'edge':
        # return int((y[row] == y[col]).sum()) / row.size(0)
        # print((edge_attr*(y[row] == y[col])))
        # print((edge_attr*(y[row] == y[col])).sum())
        # print(edge_attr.sum())
        return (edge_attr*(y[row] == y[col])).sum() / edge_attr.sum()
    else:
        # out = torch.zeros_like(row, dtype=float)
        # out[y[row] == y[col]] = 1.
        # out = scatter_mean(out, col, 0, dim_size=y.size(0))
        # return float(out.mean())
        out = edge_attr
        out[y[row] != y[col]] = 0.
        index = [np.where(np.unique(row)==row[i])[0][0] for i in range(len(row))] if(not all_pat) else row
        out = scatter_add(src=torch.tensor(out), index=torch.tensor(index), dim=0, dim_size=len(np.unique(row))) #scatter add because the sum for each node is always 1, so no need to divide
        return float(out.mean())


# def graph_loopCNN(n_patients, feature_list, Y, a):
#     G_list = []
#     data_list = []
#     sd1 = 0.9324149289534428
#     sd2 = 2.419077652293398
#     for i in range(n_patients):
#         n_ROI = len(feature_list[i])
#         G, data = make_graphCNN(n_ROI, feature_list[i], Y[i], sd1, sd2, a)

#         G_list.append(G)
#         data_list.append(data)
#     return G_list, data_list


def make_graph(n_ROI, classical, liste_center, radiomics, Y, sd1, sd2, a, sep_edge_weights):
    G = nx.Graph()
    node_dicts = {}
    edge_dicts = {}
    den1 = sd1 * sd1 * a
    den2 = sd2 * sd2 * a
    knn = 3
    for c in range(n_ROI):
        G.add_node(c)
        node_dicts[c] = [*classical[c], *radiomics[c]]
        for d in range(n_ROI):
            if (c == d):
                G.add_edge(c, d, weight=1)
                if(not sep_edge_weights): 
                    edge_dicts[c, d] = 1
                else:
                    edge_dicts[c, d] = [1,1]
            else:
                e1 = np.exp(-np.square(np.linalg.norm(np.array(liste_center[c]) - np.array(liste_center[d]))) / den1)
                e2 = np.exp(-np.square(np.linalg.norm(np.array([*classical[c], *radiomics[c]]) - np.array([*classical[d], *radiomics[d]]))) / den2)
                if(not sep_edge_weights): 
                    edge_dicts[c, d] = e1*e2
                else:
                    edge_dicts[c, d] = [e1, e2]

                # print("e1 e2 *")
                # print(e1)
                # print(np.array(liste_center[c]) - np.array(liste_center[d]))
                # print(np.linalg.norm(np.array(liste_center[c]) - np.array(liste_center[d])))
                # print(-np.square(np.linalg.norm(np.array(liste_center[c]) - np.array(liste_center[d]))))
                # print(-np.square(np.linalg.norm(np.array(liste_center[c]) - np.array(liste_center[d]))) / den1)
                # print(np.exp(-np.square(np.linalg.norm(np.array(liste_center[c]) - np.array(liste_center[d]))) / den1))
                # print(e2)
                # print(e1*e2)

                # G.add_edge(c, d, weight=e1 )#* e2)
    a= np.array(list(edge_dicts.values()))
    edge_index = torch.tensor(list(edge_dicts.keys()), dtype=torch.long)
    A = np.reshape(np.random.rand(1,n_ROI*n_ROI), (n_ROI, n_ROI))
    edge_ind = []
    if (n_ROI > knn):
        for i in range(A.shape[0]):
            t = np.sort(A[i])[-(knn + 1)]
            A[i][A[i] < t] = 0
            non_zero = np.nonzero(A[i])[0]
            for j in range(non_zero.shape[0]):
                edge_ind.append([i,non_zero[j]])
        edge_list = A.flatten()
        edge_index = torch.tensor(edge_ind, dtype=torch.long)

    features = torch.tensor(list(node_dicts.values()), dtype=torch.float)
    edge_attr = torch.tensor(list(edge_dicts.values()), dtype=torch.float)
    edge_index = torch.tensor(list(edge_dicts.keys()), dtype=torch.long)
    # edge_weights = edge_weights[:, None]
    y = torch.tensor(Y, dtype=torch.long)
    data = Data(x=features, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)

    nx.set_edge_attributes(G, edge_dicts, "edge_weights")
    nx.set_node_attributes(G, node_dicts, "feature_vector")
    #print(edge_attr)

    return G, data


def make_graph_supernode(n_ROI, classical, liste_center, radiomics, Y, sd1, sd2, a):
    G = nx.Graph()
    node_dicts = {}
    edge_dicts = {}
    den1 = sd1 * sd1 * a
    den2 = sd2 * sd2 * a
    knn = 3
    pos_lesions_centroid = np.mean(liste_center, axis=0)
    # print(pos_lesions_centroid)

    #Super node creation (for the NetworkX graph):
    G.add_node(n_ROI)
    #Node feature definition:
    node_dicts[n_ROI] = np.zeros(len(classical[0])+len(radiomics[0]))

    for c in range(n_ROI):
        #Node creation (for the NetworkX graph):
        G.add_node(c)
        #Node feature definition:
        node_dicts[c] = [*classical[c], *radiomics[c]]
        #Edges definition:
        e1 = np.exp(-np.square(np.linalg.norm(np.array(liste_center[c]) - pos_lesions_centroid)) / den1)
        # e2 = np.exp(-np.square(np.linalg.norm(np.array([*classical[c], *radiomics[c]]) - np.array([*classical[d], *radiomics[d]]))) / den2) #if feature distance in node features
        edge_dicts[c, n_ROI] = e1#*e2
        G.add_edge(c, n_ROI, weight=e1)
            #If self-loops:
        # edge_dicts[c, c] = 1
        # G.add_edge(c, c, weight=1)

    # a= np.array(list(edge_dicts.values()))
    # edge_index = torch.tensor(list(edge_dicts.keys()), dtype=torch.long)
    # A = np.reshape(np.random.rand(1,n_ROI*n_ROI), (n_ROI, n_ROI))
    # edge_ind = []
    # if (n_ROI > knn):
    #     for i in range(A.shape[0]):
    #         t = np.sort(A[i])[-(knn + 1)]
    #         A[i][A[i] < t] = 0
    #         non_zero = np.nonzero(A[i])[0]
    #         for j in range(non_zero.shape[0]):
    #             edge_ind.append([i,non_zero[j]])
    #     edge_list = A.flatten()
    #     edge_index = torch.tensor(edge_ind, dtype=torch.long)

    features = torch.tensor(list(node_dicts.values()), dtype=torch.float)
    edge_attr = torch.tensor(list(edge_dicts.values()), dtype=torch.float)
    edge_index = torch.tensor(list(edge_dicts.keys()), dtype=torch.long)
    # edge_weights = edge_weights[:, None]
    y = torch.tensor(Y, dtype=torch.long)
    data = Data(x=features, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)

    nx.set_edge_attributes(G, edge_dicts, "edge_weights")
    nx.set_node_attributes(G, node_dicts, "feature_vector")
    #print(edge_attr)

    # #To look at the graphs:
    # import matplotlib.pyplot as plt
    # nx.draw(G)
    # plt.show()

    return G, data


# def make_graphCNN(n_ROI, feature_list, Y, sd1, sd2, a):
#     G = nx.Graph()
#     node_dicts = {}
#     edge_dicts = {}
#     den1 =  a
#     den2 = sd2 * sd2 * a
#     knn = 3
#     for c in range(n_ROI):
#         G.add_node(c)
#         node_dicts[c] = feature_list[c]
#         for d in range(n_ROI):

#             if (c == d):
#                 edge_dicts[c, d] = 1
#                 G.add_edge(c, d, weight=1)
#             else:

#                 e1 = np.exp(
#                     -np.square(np.linalg.norm(np.array(feature_list[c]) - np.array(feature_list[d]))) / den1)
#                 #e2 = np.exp(-np.square(np.linalg.norm(
#                 #     np.array([*classical[c], *radiomics[c]]) - np.array([*classical[d], *radiomics[d]]))) / den2)
#                 edge_dicts[c, d] = e1
#                 # G.add_edge(c, d, weight=e1 )#* e2)

#     features = torch.tensor(list(node_dicts.values()), dtype=torch.float)
#     edge_attr = torch.tensor(list(edge_dicts.values()), dtype=torch.float)
#     edge_index = torch.tensor(list(edge_dicts.keys()), dtype=torch.long)
#     # edge_weights = edge_weights[:, None]
#     y = torch.tensor(Y, dtype=torch.long)
#     data = Data(x=features, edge_index=edge_index.t().contiguous(), edge_attr=edge_attr, y=y)

#     nx.set_edge_attributes(G, edge_dicts, "edge_weights")
#     nx.set_node_attributes(G, node_dicts, "feature_vector")

#     return G, data
