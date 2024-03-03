# Operators.jl

## Overview
This Julia file contains functions to create various operators including the gradient, divergence, and matrix calculations.

## Functions
`build_matrix_G`: This function constructs the matrix G, which is used to compute the gradient of a scalar field in a fluid domain. It takes as input the dimensions of the mesh (nx and ny), along with certain difference operators (Dx_minus, Dy_minus, Bx, By), and returns the constructed matrix G.

`build_matrix_G_T`: Similar to the previous function, this one constructs the transpose of the matrix G, denoted as G^T, used in certain computations involving gradients.

`build_matrix_H`: Constructs the matrix H, which plays a role in computing the gradient of a scalar field. It also takes difference operators (Dx_minus, Dy_minus, Ax, Ay, Bx, By) as inputs.

`build_matrix_H_T`: Constructs the transpose of the matrix H, denoted as H^T, used in certain computations involving gradients.

`build_matrix_minus_GTHT`: Constructs the matrix -(G^T + H^T), used in the computation of the divergence operator. It takes difference operators (Dx_plus, Dy_plus, Ax, Ay) as inputs.

`compute_grad_operator`: Computes the gradient of a scalar field using the matrices G, H, and their transpose counterparts. It takes the scalar field values (p_omega and p_gamma) and the transpose of the weight matrix (Wdagger) as inputs.

`compute_divergence`: Computes the divergence of a vector field using the matrices G^T and H^T. It takes the vector field components (q_omega and q_gamma) as inputs. Additionally, it has a boolean parameter gradient which, if set to true, computes only the gradient without involving H^T.

`sparse_inverse` : Computes the pseudo-inverse of W matrix.