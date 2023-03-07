import numpy as np
import networkx as nx
import copy

from fa2 import ForceAtlas2
from shapely.geometry import LineString

"""
Graph layout algorithms

"""

"""
Force-atlas2 algorithm, a force-directed fast algorithm to lay out nodes
From: https://github.com/bhargavchippada/forceatlas2
Be careful! The python version has an influence on whether the edge weight argument for ForceAtlas2 functions

Input
G:              networkx Graph object, a simple networkx graph object
pos:            np.array, starting coordinates of the nodes, default set to none so that starting coordinates are random
edge_weith:     float, 0 if edge weights are of no influence, 1 if edge weights should be factored in
max_iter:       int, the maximum number of iterations the algorithm can run, default set to 2000
return_dict:    bool, whether the positions need to be returned as dict or not

Output
coords:         np.array, finalized coordinates of the nodes
"""


def run_forceatlas2(G, pos = None, edge_weight = False, max_iter = 2000, return_dict = True):
    
    if edge_weight:
        weight_influence = 1.0
    else:
        weight_influence = 0.0
    
    # create a forceatlas 2 object with the necessary parameters (some not implemented)
    forceatlas2 = ForceAtlas2(
                        # Behavior alternatives
                        outboundAttractionDistribution = True,  # Dissuade hubs
                        linLogMode = False,  # NOT IMPLEMENTED
                        adjustSizes = False,  # Prevent overlap (NOT IMPLEMENTED)
                        edgeWeightInfluence = weight_influence,

                        # Performance
                        jitterTolerance = 1.0,  # Tolerance
                        barnesHutOptimize = True,
                        barnesHutTheta = 1.2,
                        multiThreaded = False,  # NOT IMPLEMENTED

                        # Tuning
                        scalingRatio = 2.0,
                        strongGravityMode = False,
                        gravity = 1.0,

                        # Log
                        verbose = False)
                        
    # run the forceatlas 2 algorithm to create a layout
    pos = forceatlas2.forceatlas2_networkx_layout(G, pos = pos, iterations = max_iter)

    if return_dict:
        return pos
    
    # convert the given dictionary to a n by 2 matrix    
    coords = np.zeros(shape = (len(pos.keys()), 2))
    for i in pos:
        coords[i][0] = pos[i][0]
        coords[i][1] = pos[i][1]
        
    return coords
    
    
"""
Stress-Majorization algorithm, an approach of getting the pairwise node distances as close to the graph theoretic distances
Proofchecked using: https://github.com/schochastics/graphlayouts/blob/master/src/stress.cpp

Input
pos:            np.array, starting coordinates of the nodes
gtds:           np.array, a symmetric nxn numpy array containing the graph theoretic distances
stress_alpha:   int, an integer value used to set a weighting factor, default set at 2
max_iter:       int, the maximum number of iterations the algorithm can run, default set to 2000
tolerance:      float, if a new iteration has results smaller than the tolerance then stop the algorithm, default set to 1e-5
edge_weights:   dict, if not None then the weights of the edges are taken into account, the higher the weight the longer the edge, default None
return_dict:    bool, if set to True then a dictionary with the coordinates will be returned, default set to False

Output
X:              np.array, finalized coordinates of the nodes
"""


def stress_majorization(G, pos, stress_alpha = 2, max_iter = 2000, tolerance = 1e-5, edge_weight = False, return_dict = True):

    if edge_weight:
        gtds = nx.floyd_warshall_numpy(G)
    else:
        gtds = nx.floyd_warshall_numpy(G, weight = 'nan')

    n = np.shape(pos)[0]
    
    X = copy.deepcopy(pos)
    
    W = np.array(copy.deepcopy(gtds)).astype(float)**-stress_alpha
    W[W == float('inf')] = 0
    D = np.array(gtds).astype(float)
    
    eucl_dis_og = np.sqrt(np.sum(((np.expand_dims(X, axis = 1) - X)**2), 2))
    
    # divided by 2 because we're taking the whole matrix rather than a triangle of the matrix
    stress_og = np.sum(W * ((eucl_dis_og - D)**2)) / 2
    
    # list to keep track of the progression of the stress minimization
    stress_list = [stress_og]        
    
    # loop over all the maximum iterations
    for itr in range(max_iter):
        X_new = np.zeros(shape = (n, 2))

        for i in range(n):
            for j in range(n):
                if i != j:
                    denom = eucl_dis_og[i][j]
                    if denom > 0.00001:
                        X_new[i][0] += W[i][j] * (X[j][0] + D[i][j] * (X[i][0] - X[j][0]) / denom)
                        X_new[i][1] += W[i][j] * (X[j][1] + D[i][j] * (X[i][1] - X[j][1]) / denom)
                        
            X_new[i][0] = X_new[i][0] / np.sum(W, 1)[i]
            X_new[i][1] = X_new[i][1] / np.sum(W, 1)[i]
            
        eucl_dis_new = np.sqrt(np.sum(((np.expand_dims(X_new, axis = 1) - X_new)**2), 2))
        stress_new = np.sum(W * ((eucl_dis_new - D)**2)) / 2

        diff = (stress_list[-1] - stress_new) / stress_list[-1]
        
        # if the new layout its stress value does not differ more than the tolerance then we are in a minimum, stop the algorithm
        if diff <= tolerance:
            if return_dict:
                X_dict = {}
                for i in range(len(X)):
                    X_dict[i] = [X[i, 0], X[i, 1]]
                
                return X_dict
            return X
            
        # add the new stress to the list and now we take the new layout as the layout to be improved
        stress_list.append(stress_new)
        X = copy.deepcopy(X_new)
        eucl_dis_og = copy.deepcopy(eucl_dis_new)
        
    if return_dict:
        X_dict = {}
        for i in range(len(X)):
            X_dict[i] = [X[i, 0], X[i, 1]]
        
        return X_dict
    
    return X

    
"""
A simple function for scaling the current coordinates to a canvas of set width and height

Input
coords:         np.array, current coordinates of the nodes
default_width:  int, the set width of the canvas, default set to 900
Output
coords:         np.array, scaled coordinates of the nodes
"""


def scale_coords_to_canvas(coords, default_width = 900):
        
    if isinstance(coords, dict):
        new_coords = np.zeros((len(coords), 2))
        keys = list(coords.keys())
        for i in range(len(coords)):
            new_coords[i, 0], new_coords[i, 1] = coords[keys[i]][0], coords[keys[i]][1]
                
        coords = new_coords
    # take the maximum of the the maximum differences (max difference of x coordinates & max difference of y coordinates)
    scale_factor = max(np.max(coords[:, 0]) - np.min(coords[:, 0]), np.max(coords[:, 1]) - np.min(coords[:, 1]))
    
    # take the minimimum x and y values 
    translation_factor_x = np.min(coords[:, 0])
    translation_factor_y = np.min(coords[:, 1])
    
    # translate the layout by subtracting the minimum x and y values from the x and y coordinates, respectively
    coords[:, 0] = coords[:, 0] - translation_factor_x
    coords[:, 1] = coords[:, 1] - translation_factor_y
    
    # scale the layout
    coords = coords * default_width / scale_factor
    
    coords_dict = {}
    for i in range(len(coords)):
        coords_dict[i] = [coords[i, 0], coords[i, 1]]
    
    return coords_dict
    
 
"""
Function for planarity testing of a layout

Input
g_dict: dictionary, graph dictionary with 4 keys
            nodes:          list, of nodes
            nodes_coords:   dictionary, nodes as keys and their x and y coordinates
            edges:          list, of edges
            id_no:          unique identifier of the generated graph
Optional
G:          networkx graph object
spec_node:  id of a node, changes to the coordinates of this node are used to test planarity (instead of testing ALL nodes in the graph)

Output
planar:     boolean, indicates whether the layout is planar

"""


def test_layout_planarity(g_dict, G = None, spec_node = None):

    coords = g_dict['nodes_coords']
    planar = True
    
    if not G:
        G = nx.Graph()
        G.add_nodes_from(g_dict['nodes'])
        G.add_edges_from(g_dict['edges'])
    
    edges = list(G.edges())
    
    if spec_node:
        edges_main = list(G.edges(spec_node))
    else:
        edges_main = edges
        
    for i in range(len(edges_main)):
        edge1 = edges_main[i]
                
        # get the line of the first edge
        c1 = (coords[edge1[0]][0], coords[edge1[0]][1])
        c2 = (coords[edge1[1]][0], coords[edge1[1]][1])
        first_line = LineString([c1, c2])
        
        for j in range(len(edges)):
            edge2 = edges[j]
            
            # only check if edges cross if they do not share a node
            if edge1[0] not in edge2 and edge1[1] not in edge2:
                c3 = (coords[edge2[0]][0], coords[edge2[0]][1])
                c4 = (coords[edge2[1]][0], coords[edge2[1]][1])
                second_line = LineString([c3, c4])
                
                if first_line.intersects(second_line):
                    planar = False
                    break
                    
        if not planar:
            break
            
    return planar
