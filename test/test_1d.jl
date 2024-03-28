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

# Grille 1D
grid = CartesianGrid((80,) , (1.0,))
mesh = CartesianLevelSet.generate_mesh(grid, false) # Génère un maillage
x = mesh[1]
nx = length(x)
dx = 1/nx
@show nx, dx

front_position = 0.5
domain = ((minimum(x),), (maximum(x),))
front = SignedDistanceFunction((x, _=0) -> abs(x-front_position) , domain)

# Définir la plage de x
x_range = 0:0.01:1

# Calculer la valeur de la fonction de distance signée pour chaque x
sdf_values = [front.sdf_function(x) for x in x_range]

# Créer le tracé
plot(x_range, sdf_values, label="SDF", xlabel="x", ylabel="SDF Value", title="Signed Distance Function")
readline()