from metrics import *
from sklearn.ensemble import IsolationForest


"""
Applying weights to outlying edges and drawing the results

Input
G:                  networkx graph object, given graph G
stand_function:     function, can be min, max, or np.mean used to standardize footprints and weight problematic edges
layout_f:           function, from layouts_refactor e.g. stress_majorization
start_pos:          np.array, a 2xn numpy array of x and y coordinates for each node
max_len:            float or int, if set to a float then footprint size will be at float of the max footprint size,
                    if set to int then footprint size will be set to that int, paper describes it as variable 'desired length k'
node_disjoint:      boolean, if set to True then shortest node-disjoint paths will be computed, if False then edge-disjoint paths will be taken
problematic_edgesL  list, a list of all edges that are preclassified as problematic, empty by default

Output
pos:                dict, a dictionary where each node is a key and has a list of length 2 to denote coordinates
classif:            dict, a dictionary where each edge is a key and has a two lettered string as value indicating true/false positive/negative

"""


def weight_create_layout(G, stand_function, layout_f, start_pos, max_len = 0.8, node_disjoint = True, problematic_edges = []):

    G = nx.convert_node_labels_to_integers(G)

    # acquire the footprints from given graph G
    footprints = disjoint_path_lengths(G, node_disjoint)
    
    # if the given maximum length is a fraction then we get a cutoff point, if it is an integer then we use the integer as a cutoff point
    if max_len < 1:
    
        # get the lengths of these vectors
        lens = []
        
        for tup in footprints:
            lens.append(len(footprints[tup]))

        # get a cutoff point, meaning 80% of the vectors have at least length vec_length or smaller
        cutoff = int(len(lens) * max_len)
        max_len = sorted(lens)[cutoff]
    elif type(max_len) != int:
        raise Exception('Either provide a float between 0.01 and 0.99, or provide an integer')
        
    # standardize the footprints
    stand_footprints = standardize_footprints(footprints, stand_function, max_len = max_len)
    
    # do outlier detection on the footprints and weight the graph accordingly
    G_weighted, classif = outlier_detection(G, stand_footprints, problematic_edges, layout_f, stand_function)
    
    # create a layout using SM or FA2
    pos = create_layout(G_weighted, layout_f, start_pos)
    
    return pos, classif


"""
Creating a layout

Input
G:                  networkx graph object, given graph G
layout_f:           function, from layouts e.g. stress_majorization
start_pos:          np.array, a 2xn numpy array of x and y coordinates for each node
edge_weight:        bool, whether edge_weights (included in object G) will be included or not, set to True by default

Output
pos:                dict, a dictionary where each node is a key and has a list of length 2 to denote coordinates
"""


def create_layout(G, layout_f, start_pos, edge_weight = True):

    G = nx.convert_node_labels_to_integers(G)

    # repeat the layout technique 5 times and get the best result according to number of crossings
    coords_list = {}
    ncs_list = {}
    
    for i in range(5):
        coords_list[i] = layout_f(G = G, pos = start_pos, edge_weight = edge_weight)
        ncs_list[i] = crossings_metric({'coords' : scale_coords_to_canvas(coords_list[i]), 'G' : G})
        
    # return the layout with the fewest number of crossings
    pos = coords_list[sorted(ncs_list, key = ncs_list.get)[0]]
    
    return pos


"""
Acquiring node or edge disjoint path lengths

Input
G:                  networkx graph object, given graph G
node_disjoint:      bool, whether the disjoint paths of an edge are computed by node or edge disjoint

Output
footprints:        dict, a dictionary where each edge is a key, a key's value is a list of the lengths of the disjoint shortest paths
"""


def disjoint_path_lengths(G, node_disjoint):

    footprints = {}
    edge_list = copy.deepcopy(list(G.edges()))

    for edge in edge_list:
        G.remove_edge(edge[0], edge[1])
        # https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.connectivity.disjoint_paths.node_disjoint_paths.html
        paths = []
        
        # only get disjoint paths if it is connected
        if nx.is_connected(G):
            if node_disjoint:
                paths = list(nx.node_disjoint_paths(G, edge[0], edge[1]))
            else:
                paths = list(nx.edge_disjoint_paths(G, edge[0], edge[1]))
                
        # get the lengths of the disjoint paths
        path_lengths = []
        
        for p in paths:
            path_lengths.append(len(p) - 1)

        footprints[tuple(edge)] = sorted(path_lengths)
        G.add_edge(edge[0], edge[1])

    return footprints


"""
Standardizing footprints by ondensing or expanding 

Input
footprints:         dict, a dictionary where each edge is a key, a key's value is a list of the lengths of the disjoint shortest paths
stand_function:     function, can be min, max, or np.mean used to standardize footprints
max_len:            int, the maximum length a footprint can be, described as variable k in the paper

Output
stand_footprints:   dict, a dictionary where each edge is a key, a key's value is a standardized list of the lengths of the disjoint shortest paths
"""


def standardize_footprints(footprints, stand_function, max_len):

    stand_footprints = {}
    
    for i in footprints:
        # if the current feature vector is already at the target length then we just copy
        if len(footprints[i]) == max_len:
            stand_footprints[i] = copy.deepcopy(footprints[i])
        # if the current feature vector is larger then we shrink the feature vector using some function (e.g. min, mean, max)
        elif len(footprints[i]) > max_len:
            curr_vector = copy.deepcopy(footprints[i])
            stand_footprints[i] = []

            for j in range(max_len - 1):
                stand_footprints[i] += [curr_vector[j]]

            stand_footprints[i] += [stand_function(curr_vector[(max_len - 1):])]
        # if the current feature vector is smaller then we extend the feature vector
        elif 0 < len(footprints[i]) < max_len:
            curr_vector = copy.deepcopy(footprints[i])
            stand_footprints[i] = curr_vector

            for j in range(max_len - len(curr_vector)):
                stand_footprints[i] += [stand_function(curr_vector)]

    return stand_footprints


"""
Detecting outlying edges

Input
G:                  networkx graph object, given graph G
stand_footprints:   dict, a dictionary where each edge is a key, a key's value is a standardized list of the lengths of 
                    the disjoint shortest paths
problematic_edgesL  list, a list of all edges that are preclassified as problematic, empty by default
layout_f:           function, from layouts e.g. stress_majorization
stand_function:     function, can be min, max, or np.mean used to weight problematic edges

Output
G_weighted:         networkx graph object, input graph G with a weight parameter
classif:            dict, dictionary where each edge is a key, a key's value is it's correct/incorrect classification
"""


def outlier_detection(G, stand_footprints, problematic_edges, layout_f, stand_function):

    G_weighted = copy.deepcopy(G)
    
    values = np.array(list(stand_footprints.values()))
    # use isolation forest for outlier detection
    o_detect = IsolationForest(random_state = 0).fit(values)
    outls = o_detect.predict(values)
    
    classif = {'fp' : [], 'fn' : [], 'tp' : [], 'tn' : []}
    # fp == false positive, test indicates edge is problematic whereas ground truth says it is non-problematic
    # fn == false negative, test indicates edge is non-problematic whereas ground truth says it is problematic
    # tp == true positive, test indicates edge is non-problematic, ground truth says it is non-problematic
    # tn == true negative, test indicates edge is problematic, ground truth says it is problematic
    
    # go over all the path vectors
    cnt = 0
    for i in stand_footprints:
        # edge classified as outlier
        if outls[cnt] == -1:
            weight = stand_function(stand_footprints[i])
            G_weighted[i[0]][i[1]]['weight'] = 1 / weight
            
            # if the layout technique is stress majorization then we want weights > 1
            if layout_f == stress_majorization:
                G_weighted[i[0]][i[1]]['weight'] = 1 / G_weighted[i[0]][i[1]]['weight']

            # if it was a problematic edge (we have ground truth data) then increase counter of correctly classifying, true negative
            if ([i[0], i[1]] in problematic_edges) or ([i[1], i[0]] in problematic_edges):
                classif['tn'].append([i[0], i[1]])
            # if it was a normal edge then we falsely identified it as problematic, false positive
            else:
                classif['fp'].append([i[0], i[1]])
        # edge classified as non-outlier
        else:
            # if it was a problematic edge we didn't correctly identify it as problematic, false negative
            if ([i[0], i[1]] in problematic_edges) or ([i[1], i[0]] in problematic_edges):
                classif['fn'].append([i[0], i[1]])
            else:
                classif['tp'].append([i[0], i[1]])

        cnt += 1

    return G_weighted, classif
