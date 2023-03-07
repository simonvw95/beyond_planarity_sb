### Beyond Planarity - A Spring-Based Approach
This repository contains a spring-based heuristic for drawing nearly-planar graphs. 

# Requirements
The requirements can be found in requirements.txt

# Motivation
Given a graph layout (Figure 1) and a set of problematic edges that are put on top (Figure 2), common layout algorithms, such as ForceAtlas2 and Stress Majorization, will tend to fold the layout as a result of the problematic edges influencing the global structure. Their effects result in a layout seen in Figure 3.

| ![grid_start.jpg](https://github.com/simonvw95/beyond_planarity_sb/blob/main/figures/grid_19.pickleFD_begin.png) | ![grid_ontop.jpg](https://github.com/simonvw95/beyond_planarity_sb/blob/main/figures/grid_19.pickleFD_end_ontop.png) |
|:--:|:--:| 
| *Figure 1: Grid graph laid out by ForceAtlas2* | *Figure 2: Grid graph with problematic edges ontop laid out by ForceAtlas2* |

What if we could identify these problematic edges with great accuracy and subsequently decrease their importance in the resulting layout. Figures 4 and 5 depict a simple strategy, where the (known) problematic edges are given a low weight (0.01). 
