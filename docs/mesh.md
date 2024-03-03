# Mesh.jl
## Overview

This Julia file provides functions to generate a collocated, staggered mesh, to create geometric moments matrices commonly used in numerical computations, particularly in the cut-cell method.

## Functions

`generate_mesh` : This function generate 2 meshes, one collocated and one staggered

`calculate_first_order_moments` : This function perform geometric computations on meshes. It can calculate volumes, surfaces centroids for the first-kind moments.

`calculate_second_order_moments` : This function perform geometric computations on meshes. It can calculate volumes, surfaces, for second-kind moments.