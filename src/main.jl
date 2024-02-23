# Function to generate Cartesian mesh with border cells marked as fluid and circular solid region
function generate_cartesian_mesh_with_circle(nx::Int, ny::Int, radius::Float64, center::Tuple{Float64, Float64})
    Δx = 1.0 / (nx - 1)
    Δy = 1.0 / (ny - 1)
    x = range(0, stop=1, length=nx)
    y = range(0, stop=1, length=ny)
    
    mesh = fill(1.0, nx, ny)  # Initialize mesh with fluid (1)
    
    # Mark border cells as fluid (1)
    mesh[1, :] .= 1  # Top border
    mesh[end, :] .= 1  # Bottom border
    mesh[:, 1] .= 1  # Left border
    mesh[:, end] .= 1  # Right border
    
    # Mark cells inside circle as solid (0)
    for i in 1:nx, j in 1:ny
        if (x[i] - center[1])^2 + (y[j] - center[2])^2 <= radius^2
            mesh[i, j] = 0
        end
    end
    
    return mesh, Δx, Δy
end

# Example usage
nx = 10  # Number of points in x direction
ny = 10  # Number of points in y direction
radius = 0.3  # Radius of the circular solid region
center = (0.5, 0.5)  # Center of the circular solid region
mesh, Δx, Δy = generate_cartesian_mesh_with_circle(nx, ny, radius, center)

# Print the mesh and cell sizes
println("Cartesian Mesh :")
for j in 1:ny
    println("Row $j: ", mesh[:, j])
end
println("Δx = $Δx")
println("Δy = $Δy")

