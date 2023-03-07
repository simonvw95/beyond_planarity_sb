import matplotlib as plt
import networkx as nx

# global values
width = 1000
height = 1000
dpi = 96


"""
Plotting and saving a layout of a given graph

Input:
G:                      networkx graph object, given graph
coords:                 np.array or dict, input coordinates in a n*2 np.array, or dictionary with nodes as keys and x y coordinates as values
foldername:             string, name of the foldername in which to save the produced images
filename:               string, name of the png image
edge_difference:        list, nested list of edges that have been added to the new graph, can be empty
title:                  string, title to include in the figure
"""


def save_fig(G, coords, foldername, filename, edge_difference, title):

    # check if coordinates are in dict or not
    if isinstance(coords, dict) is False:
        pos = {}
        for i in list(G.nodes()):
            pos[i] = [coords[i][0], coords[i][1]]
    else:
        pos = coords

    edges = list(G.edges())
    edge_colors = ['silver'] * len(edges)

    for e in range(len(edges)):
        curr_e = [edges[e][0], edges[e][1]]
        curr_e_r = [edges[e][1], edges[e][0]]

        # color the problematic edges with a different color
        for f in edge_difference:
            if (curr_e == f or curr_e_r == f):
                edge_colors[e] = 'midnightblue'

    plt.figure(figsize = (width / dpi, height / dpi), dpi = 350)
    ax = plt.gca()

    ax.set_title(title)

    # one can add arguments to nx.draw to include edge widths, labels etc.
    nx.draw(G, pos = pos, node_size = [10] * G.number_of_nodes(), edge_color = edge_colors)

    plt.savefig(foldername + filename + '.png')
    plt.clf()
    plt.close('all')
