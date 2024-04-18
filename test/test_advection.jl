using CartesianLevelSet
using CartesianGeometry
using Plots
Plots.default(show = true)
using StaticArrays
using SparseArrays
using CartesianArrays
using LinearAlgebra
using IterativeSolvers
using Arpack
include("../src/matrix.jl")
include("../src/operators.jl")
include("../src/solve.jl")
include("../src/boundary.jl")
include("../src/utils.jl")
include("../src/mesh.jl")
include("../src/plot.jl")

## Advection 2D
# Grille 2D
grid = CartesianGrid((80, 80) , (2., 2.))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage 
x, y = mesh
nx = length(x)
ny = length(y)
dx, dy = 1/nx, 1/ny
barycentres = vec([(xi + dx/2, yi + dy/2) for xi in x[1:end], yi in y[1:end]])

# Définir le domaine
domain = ((minimum(x), minimum(y)), (maximum(x), maximum(y)))

# Définir le cercle
circle = SignedDistanceFunction((x, y, _=0) -> sqrt((x-1)^2+(y-1)^2) - 0.5 , domain)

values = evaluate_levelset(circle.sdf_function, mesh)
cut_cells = CartesianLevelSet.get_cut_cells(values)
intersection_points = get_intersection_points(values, cut_cells)
midpoints = get_segment_midpoints(values, cut_cells, intersection_points)

# Operators
Delta_x_minus, Delta_y_minus = backward_difference_matrix_sparse_2D_x(nx, ny), backward_difference_matrix_sparse_2D_y(nx, ny)
Delta_x_plus, Delta_y_plus = forward_difference_matrix_sparse_2D_x(nx, ny), forward_difference_matrix_sparse_2D_y(nx, ny)

# Calculer les moments d'ordre 1 et 2
V, v_diag, bary, ax_diag, ay_diag = calculate_first_order_moments(circle.sdf_function, mesh)
w_diag, bx_diag, by_diag, border_cells_wx, border_cells_wy = calculate_second_order_moments(circle.sdf_function, mesh, bary)

# Définir la fonction de vitesse
function u_x(x, y, w0)
    return -0.5*w0*(y-1.0)
end

function u_y(x, y, w0)
    return 0.5*w0*(x-1.0)
end

# Définir la fonction de vitesse
u = vcat([u_x(x, y, 1.0) for (x, y) in barycentres], [u_y(x, y, 1.0) for (x, y) in barycentres])

# Calculer le masque du cercle
circle_mask = [circle.sdf_function(x, y) <= 0 ? 1 : NaN for (x, y) in barycentres]

# Appliquer le masque au vecteur de vitesse
u_x_masked = u[1:nx*ny] .* circle_mask
u_y_masked = u[nx*ny+1:end] .* circle_mask

# Heatmap
heatmap(x, y, reshape(u_x_masked, nx, ny)', c=:viridis, xlabel="x", ylabel="y", title="u_x", aspect_ratio=:equal)
readline()

heatmap(x, y, reshape(u_y_masked, nx, ny)', c=:viridis, xlabel="x", ylabel="y", title="u_y", aspect_ratio=:equal)
readline()