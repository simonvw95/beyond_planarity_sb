from layouts import *
from scipy import linalg

"""
A simple function for counting the number of crossings in the current graph

Input
final_g_dict:   dict, a dictionary containg the coordinates "coords", a graph object G "G", the graph theoretic distances "gtds" and the stress alpha "stress_alpha"
                    

Output
cnt:            the number of crossings in the graph
"""


def crossings_metric(final_g_dict):

    coords = final_g_dict['coords']
    G = final_g_dict['G']
    edges = list(G.edges())
    cnt = 0
    
    # loop over all the edges
    for i in range(len(edges)):
        edge1 = edges[i]
                
        # get the line of the first edge
        c1 = (coords[edge1[0]][0], coords[edge1[0]][1])
        c2 = (coords[edge1[1]][0], coords[edge1[1]][1])
        first_line = LineString([c1, c2])
        
        # loop over the other edges starting from i (duplicate crossings won't be counted then)
        for j in range(i, len(edges)):
            edge2 = edges[j]
            
            # only check if edges cross if they do not share a node
            if edge1[0] not in edge2 and edge1[1] not in edge2:
                c3 = (coords[edge2[0]][0], coords[edge2[0]][1])
                c4 = (coords[edge2[1]][0], coords[edge2[1]][1])
                second_line = LineString([c3, c4])
                
                # if there is an intersection increase the ocunt
                if first_line.intersects(second_line):
                    cnt += 1
                    
    return cnt
    
    
"""
A simple function for computing the overall stress of the current layout

Input
final_g_dict:   dict, a dictionary containg the coordinates "coords", a graph object G "G", the graph theoretic distances "gtds" and the stress alpha "stress_alpha"

Output
stress_tot:     float, the current total number of stress
"""


def stress_metric(final_g_dict):

    coords = final_g_dict['coords']

    gtds = final_g_dict['gtds']
    stress_alpha = final_g_dict['stress_alpha']
        
    # compute the weights and set the numbers that turned to infinity (the 0 on the diagonals) to 0
    weights = np.array(gtds).astype(float)**-stress_alpha
    weights[weights == float('inf')] = 0
    
    # calculate the euclidean distances
    eucl_dis = np.sqrt(np.sum(((np.expand_dims(coords, axis = 1) - coords)**2), 2))
    
    # compute stress by multiplying the distances with the graph theoretic distances, squaring it, multiplying by the weight factor and then taking the sum
    stress_tot = np.sum(weights * ((eucl_dis - gtds)**2))
        
    return stress_tot / 2
        
    
"""
A simple function for computing the angular resolution of the current layout

Input
final_g_dict:   dict, a dictionary containg the coordinates "coords", a graph object G "G", the graph theoretic distances "gtds" and the stress alpha "stress_alpha"

Output
res:            float, the angular resolution
"""


def angular_resolution_metric(final_g_dict):
    
    # description of the algorithm
    # go over all nodes
    # take their neighbors
    # for each pair of neighbors (in clockwise or counterclockwise ordering) get their angle
    # save the smallest angle of pairs
    # when looped over all neighbors and all nodes
    # return the smallest seen angle of pairs
    
    # e.g. a square in the grid would have 4 neighbors to a center node
    # degree = 4
    # max angle possible is 2*pi/degree = 2*pi/4 = pi/2
    coords = final_g_dict['coords']

    if isinstance(coords, dict):
        new_coords = np.zeros((len(coords), 2))
        keys = list(coords.keys())
        for i in range(len(coords)):
            new_coords[i, 0], new_coords[i, 1] = coords[keys[i]][0], coords[keys[i]][1]

        coords = new_coords

    # initialize variables
    G = final_g_dict['G']
    n = G.number_of_nodes()
    nodes = list(G.nodes())
    angle_res = 360
    
    # get the maximum degree
    max_degr = max(list(dict(G.degree).values()))
    
    # loop over all nodes
    for i in range(n):
        # only compute angles if there are at least 2 neighbors
        if G.degree(nodes[i]) > 1:
            curr_neighbs = list(G.neighbors(nodes[i]))
            
            # get the ordering and then get the angles of that specific ordering
            order_neighbs = compute_order(curr_node = nodes[i], neighbors = curr_neighbs, coords = coords)
            norm_sub = np.subtract(coords[order_neighbs, ].copy(), coords[nodes[i], ])
            sub_phi = np.arctan2(norm_sub[:, 1:2], norm_sub[:, :1]) * 180 / np.pi
            
            # now compare each consecutive edge pair to get the smallest seen angle
            while len(sub_phi) >= 2:
                first = sub_phi[0]
                second = sub_phi[1]
                
                # can simply subtract the angles in these cases
                if (first >= 0 and second >= 0) or (first <= 0 and second <= 0) or (first >= 0 and second <= 0):
                    angle = abs(first - second)
                # have to add 360 for this case
                elif (first < 0 and second > 0):
                    angle = 360 + first - second
                    
                if angle < angle_res:
                    angle_res = angle

                sub_phi = np.delete(sub_phi, 0)
    
    res = np.radians(angle_res) / (2 * np.pi / max_degr)

    if not isinstance(res, np.float64):
        res = res[0]
    return res


"""
A simple function for computing the crossing resolution of the current layout, crossing resolution computes the minimum angle formed by any pair of crossing edges in G

Input
final_g_dict:   dict, a dictionary containg the coordinates "coords", a graph object G "G", the graph theoretic distances "gtds" and the stress alpha "stress_alpha"

Output
res:            float, the crossing resolution
"""


def crossing_resolution_metric(final_g_dict):
    # description of the algorithm
    # go over all edges pairs
    # if there is an intersection, find the angle of intersection
    # return the smallest seen angle of all intersection, or return all angles to compare distributions

    coords = final_g_dict['coords']

    if isinstance(coords, dict):
        new_coords = np.zeros((len(coords), 2))
        keys = list(coords.keys())
        for i in range(len(coords)):
            new_coords[i, 0], new_coords[i, 1] = coords[keys[i]][0], coords[keys[i]][1]

        coords = new_coords
    # initialize variables
    G = final_g_dict['G']
    m = G.number_of_edges()
    edges = list(G.edges())
    all_angles = []
    curr_min_angle = 360

    # loop over all edges
    for i in range(m):
        edge1 = edges[i]

        # get the line of the first edge
        c1 = [(coords[edge1[0]][0], coords[edge1[0]][1]), (coords[edge1[1]][0], coords[edge1[1]][1])]
        slope1 = (c1[1][1] - c1[0][1]) / (c1[1][0] - c1[0][0])
        first_line = LineString(c1)

        for j in range(len(edges)):
            edge2 = edges[j]

            # only check if edges cross if they do not share a node
            if edge1[0] not in edge2 and edge1[1] not in edge2:
                c2 = [(coords[edge2[0]][0], coords[edge2[0]][1]), (coords[edge2[1]][0], coords[edge2[1]][1])]
                second_line = LineString(c2)

                if first_line.intersects(second_line):
                    slope2 = (c2[1][1] - c2[0][1]) / (c2[1][0] - c2[0][0])
                    angle = abs((slope2 - slope1) / (1 + slope1 * slope2))
                    deg_angle = np.arctan(angle) * 180 / np.pi
                    all_angles.append(deg_angle)

                    if deg_angle < curr_min_angle:
                        curr_min_angle = deg_angle

    if curr_min_angle == 360:
        curr_min_angle = []
    else:
        curr_min_angle = [curr_min_angle]

    return curr_min_angle, all_angles
    

"""
Function that computes the order of neighbors around a node in clockwise-order starting at 12 o'clock
Input
curr_node:      int, the integer id of the node for which we want to know the order
neighbors:      list, a list of integer ids of the neighbors of the current node
coords:         np.array or tensor, a 2xn array or tensor of x,y node coordinates

Output
neighbors:      list, the ordered list of neighbors
"""


def compute_order(curr_node, neighbors, coords):
    
    # get the center x and y coordinate
    center_x = coords[curr_node][0]
    center_y = coords[curr_node][1]
    
    # loop over all the neighbors except the last one
    for i in range(len(neighbors) - 1):
        curr_min_idx = i
        
        # loop over the other neighbors
        for j in range(i + 1, len(neighbors)):
        
            a = coords[neighbors[j]]
            b = coords[neighbors[curr_min_idx]]
            
            # compare the points to see which node comes first in the ordering
            if compare_points(a[0], a[1], b[0], b[1], center_x, center_y):
                curr_min_idx = j
                
        if curr_min_idx != i:
            neighbors[i], neighbors[curr_min_idx] = neighbors[curr_min_idx], neighbors[i]
    
    return neighbors
    

"""
Function that compares two points (nodes) to each other to determine which one comes first w.r.t. a center
Original solution from https://stackoverflow.com/questions/6989100/sort-points-in-clockwise-order

Input
a_x:            float, the x coordinate of the first node
a_y:            float, the y coordinate of the first node
b_x:            float, the x coordinate of the second node
b_y:            float, the y coordinate of the second node
center_x:       float, the x coordinate of the center node (curr_node from compute_order function)
center_y:       float, the y coordinate of the center node (curr_node from compute_order function)

Output
res:            boolean, if True then a comes before b
"""


def compare_points(a_x, a_y, b_x, b_y, center_x, center_y):
        
    if ((a_x - center_x) >= 0 and (b_x - center_x) < 0):
        return True
        
    if ((a_x - center_x) < 0 and (b_x - center_x) >= 0):
        return False
        
    if ((a_x - center_x) == 0 and (b_x - center_x) == 0):
        if ((a_y - center_y) >= 0 or (b_y - center_y) >= 0):
            return a_y > b_y
        return b_y > a_y

    # compute the cross product of vectors (center -> a) x (center -> b)
    det = (a_x - center_x) * (b_y - center_y) - (b_x - center_x) * (a_y - center_y)
    if (det < 0):
        return True
    if (det > 0):
        return False

    # points a and b are on the same line from the center
    # check which point is closer to the center
    d1 = (a_x - center_x) * (a_x - center_x) + (a_y - center_y) * (a_y - center_y)
    d2 = (b_x - center_x) * (b_x - center_x) + (b_y - center_y) * (b_y - center_y)
    
    res = d1 > d2
    
    return res

    
def manual_procrustes(data1, data2, return_mat = False):

    if isinstance(data1, dict):
        mtx1 = np.zeros((len(data1), 2), dtype = np.double)
        mtx2 = np.zeros((len(data2), 2), dtype = np.double)
        keys = list(data1.keys())
        
        for i in range(len(keys)):
            mtx1[i, 0], mtx1[i, 1] = data1[keys[i]][0], data1[keys[i]][1]
            mtx2[i, 0], mtx2[i, 1] = data2[keys[i]][0], data2[keys[i]][1]
    else:
        mtx1 = np.array(data1, dtype = np.double, copy = True)
        mtx2 = np.array(data2, dtype = np.double, copy = True)

    if mtx1.ndim != 2 or mtx2.ndim != 2:
        raise ValueError("Input matrices must be two-dimensional")
    if mtx1.shape != mtx2.shape:
        raise ValueError("Input matrices must be of same shape")
    if mtx1.size == 0:
        raise ValueError("Input matrices must be >0 rows and >0 cols")

    # translate all the data to the origin
    mtx1 -= np.mean(mtx1, 0)
    mtx2 -= np.mean(mtx2, 0)
    norm1 = np.linalg.norm(mtx1)
    norm2 = np.linalg.norm(mtx2)

    if norm1 == 0 or norm2 == 0:
        raise ValueError("Input matrices must contain >1 unique points")

    # change scaling of data (in rows) such that trace(mtx*mtx') = 1
    mtx1 /= norm1
    mtx2 /= norm2

    # transform mtx2 to minimize disparity
    R, s = orthogonal_procrustes(mtx1, mtx2)
    mtx2 = np.dot(mtx2, R.T) * s
    
    # measure the dissimilarity between the two datasets
    
    if return_mat:
        return mtx1, mtx2
    else:
        disparity = np.sum(np.square(mtx1 - mtx2))
        return disparity


# from deepdrawing
def orthogonal_procrustes(A, B, check_finite = True):

    """
    Compute the matrix solution of the orthogonal Procrustes problem.
    Given matrices A and B of equal shape, find an orthogonal matrix R
    that most closely maps A to B [1]_.
    Note that unlike higher level Procrustes analyses of spatial data,
    this function only uses orthogonal transformations like rotations
    and reflections, and it does not use scaling or translation.
    Parameters
    """
    
    if check_finite:
        A = np.asarray_chkfinite(A)
        B = np.asarray_chkfinite(B)
    else:
        A = np.asanyarray(A)
        B = np.asanyarray(B)
        
    if A.ndim != 2:
        raise ValueError('expected ndim to be 2, but observed %s' % A.ndim)
    if A.shape != B.shape:
        raise ValueError('the shapes of A and B differ (%s vs %s)' % (
            A.shape, B.shape))
            
    # Be clever with transposes, with the intention to save memory.
    inputs = B.T.dot(A).T
    u, w, vt = linalg.svd(inputs)
    R = u.dot(vt)
    scale = w.sum()
    
    return R, scale
