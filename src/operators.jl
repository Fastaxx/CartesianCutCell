include("matrix.jl")
using Test

# Function to build the matrix G
function build_matrix_G(nx::Int, ny::Int, Dx_minus, Dy_minus, Bx, By)    
    # Compute matrix products
    Dx_minus_Bx = Dx_minus * Bx
    Dy_minus_By = Dy_minus * By
    
    # Concatenate vertically to obtain G
    G = vcat(Dx_minus_Bx, Dy_minus_By)
    
    return G
end

# Function to build the matrix Gt
function build_matrix_G_T(nx::Int, ny::Int, Dx_plus, Dy_plus, Bx, By)    
    # Compute matrix products
    Dx_plus_Bx = Bx * Dx_plus
    Dy_plus_By = By * Dy_plus
    
    # Concatenate vertically to obtain G
    G = - hcat(Dx_plus_Bx, Dy_plus_By)
    
    return G
end

# Function to create the matrix H
function build_matrix_H(nx::Int, ny::Int, Dx_minus, Dy_minus, Ax, Ay, Bx, By)
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

# Function to create the matrix Ht
function build_matrix_H_T(nx::Int, ny::Int, Dx_plus, Dy_plus, Ax, Ay, Bx, By)
    # Compute matrix products
    Dx_plus_Ax = Dx_plus * Ax
    Dy_plus_Ay = Dy_plus * Ay 
    Bx_Dx_plus = Bx * Dx_plus
    By_Dy_plus = By * Dy_plus
    
    # Compute blocks of H^T
    block1 = Bx_Dx_plus - Dx_plus_Ax
    block2 = By_Dy_plus - Dy_plus_Ay
    
    # Concatenate vertically to obtain H
    Ht = hcat(block1, block2)
    
    return Ht
end

# Function to create the matrix -(G^T+H^T)
function build_matrix_GTHT(nx::Int, ny::Int, Dx_plus, Dy_plus, Ax, Ay)
    # Compute matrix products
    Dx_plus_Ax = Dx_plus * Ax
    Dy_plus_Ay = Dy_plus * Ay
    
    # Compute blocks of -(G^T+H^T)
    block1 = Dx_plus_Ax
    block2 = Dy_plus_Ay
    
    # Concatenate vertically to obtain H
    minus_GTHT = hcat(block1, block2)
    
    return minus_GTHT
end

# Function to compute the grad operator
function compute_grad_operator(p_omega, p_gamma, Wdagger, G, H)
    # Compute G * pw
    G_pw = G * p_omega
    
    # Compute H * pg
    H_pgamma = H * p_gamma
    
    # Compute grad = Wdagger * (G * p_omega + H * p_gamma)
    grad = Wdagger * (G_pw + H_pgamma)
    
    return grad
end

# Function to compute the div operator
function compute_divergence(q_omega, q_gamma, GT, minus_GTHT, HT, gradient::Bool=false)
    if gradient
        divergence = - GT * q_omega
    else
        # Compute -(G^T + H^T) * q_omega
        div_qw = minus_GTHT * q_omega
    
        # Compute H^T * q_gamma
        div_qg = HT * q_gamma
    
        # Compute divergence
        divergence = div_qw + div_qg
    end
    return divergence
end

