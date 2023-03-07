# Beyond Planarity - A Spring-Based Approach
This repository contains a spring-based heuristic for drawing nearly-planar graphs. 

### Requirements
The requirements can be found in requirements.txt

### Motivation
Given a graph layout (Figure 1) and a set of problematic edges that are put on top (Figure 2), common layout algorithms, such as ForceAtlas2 and Stress Majorization, will tend to fold the layout as a result of the problematic edges influencing the global structure. Their effects result in a layout seen in Figure 3.

| ![grid_start.jpg](https://github.com/simonvw95/beyond_planarity_sb/blob/main/figures/grid_19.pickleFD_begin.png) | ![grid_ontop.jpg](https://github.com/simonvw95/beyond_planarity_sb/blob/main/figures/grid_19.pickleFD_end_ontop.png) | ![grid_end.jpg](https://github.com/simonvw95/beyond_planarity_sb/blob/main/figures/grid_19.pickleFD_end_rerun.png) |
|:--:|:--:|:--:|
| *Figure 1: Grid graph, laid out by ForceAtlas2* | *Figure 2: Grid graph with problematic edges ontop, laid out by ForceAtlas2* | *Figure 3: Grid graph with problematic edges ontop rerun, laid out by ForceAtlas2* |

What if we could identify these problematic edges with great accuracy and subsequently decrease their importance in the resulting layout. Figures 4 and 5 depict a simple strategy, where the (known) problematic edges are given a low weight (0.01). 

| ![grid_regular.jpg](https://github.com/simonvw95/beyond_planarity_sb/blob/main/figures/grid_45_regular.png) | ![grid_art.jpg](https://github.com/simonvw95/beyond_planarity_sb/blob/main/figures/grid_45_weight.png) |
|:--:|:--:|
| *Figure 4: Grid graph with problematic edges ontop rerun, laid out by ForceAtlas2* | *Figure 5: Grid graph with problematic edges ontop weighted, laid out by ForceAtlas2* |

As a result of the simple weighting strategy, the planar substruct reveals itself and the grid reappears. Most of the clutter is removed and the layout appears to have folded open.

Therefore, a simple heuristic is proposed to classify edges and weight these edges differently if they are considered outlying (problematic) edges. It has the following steps:

#####1. Compute footprints
A footprint for each edge is computed by finding all the shortest node/edge-disjoint paths after removal of the edge. These paths will have varying lengths, with the intuition that more problematic edges tend to have many paths with longer path lengths. These paths can be computed using the Max-Flow Edmonds-Karp algorithm as described in the [Algorithm Design](https://ict.iitk.ac.in/wp-content/uploads/CS345-Algorithms-II-Algorithm-Design-by-Jon-Kleinberg-Eva-Tardos.pdf) book.

#####2. Standardize footprints
After these footprints are found for each edge we need to standardize these footprints to have the same length. We expand or contract the footprints, depending on the user-specified number of dimensions $k$ and function $\mathcal M$, which can be either the minimum, maximum, or mean function. Equation \ref{eq:fv} portrays how the footprints are expanded or contracted, given a footprint $f(e)$ of initial length $l$ and a desired length $k$. \vspace{-0.3cm}

\begin{equation}
\vspace{-0.2cm}
\label{eq:fv}
f'(e) = \begin{cases}
f(e) \oplus [{\mathcal M}(f(e))]_{k-l} & l < k \\
f(e) & l = k \\
f(e)[0:k-1] \oplus {\mathcal M}(f(e)[k:l]) & l > k \\ 
\end{cases}
\end{equation}
