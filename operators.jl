# Function to build the matrix G
function build_matrix_G(nx::Int, ny::Int, Bx, By)
    # Create backward differentiation matrices
    Dx_minus = backward_differentiation_matrix_sparse(nx)
    Dy_minus = backward_differentiation_matrix_sparse(ny)
    
    # Compute matrix products
    Dx_minus_Bx = Dx_minus * Bx
    Dy_minus_By = Dy_minus * By
    
    # Concatenate vertically to obtain G
    G = vcat(Dx_minus_Bx, Dy_minus_By)
    
    return G
end

# Function to create the matrix H
function build_matrix_H(nx::Int, ny::Int, Ax, Ay, Bx, By)
    # Create backward differentiation matrices
    Dx_minus = backward_differentiation_matrix_sparse(nx)
    Dy_minus = backward_differentiation_matrix_sparse(ny)
    
    # Compute matrix products
    Ax_Dx_minus = Ax * Dx_minus
    Ay_Dy_minus = Ay * Dy_minus
    Dx_minus_Bx = Dx_minus * Bx
    Dy_minus_By = Dy_minus * By
    
    # Compute blocks of H
    block1 = Ax_Dx_minus - Dx_minus_Bx
    block2 = Ay_Dy_minus - Dy_minus_By
    
    # Concatenate vertically to obtain H
    H = vcat(block1, block2)
    
    return H
end

function compute_grad_operator(pw, pg, Wt, G, H)
    # Compute G * pw
    G_pw = G * pw
    
    # Compute H * pg
    H_pgamma = H * pg
    
    # Compute grad = Wt * (G * pw + H * pgamma)
    grad = Wt * (G_pw + H_pgamma)
    
    return grad
end

function compute_divergence(qw, qg, G, H)
    # Compute -(G^T + H^T) * qw
    div_qw = -(transpose(G) + transpose(H)) * qw
    
    # Compute H^T * qg
    div_qg = transpose(H) * qg
    
    # Compute divergence
    divergence = div_qw + div_qg
    
    return divergence
end

