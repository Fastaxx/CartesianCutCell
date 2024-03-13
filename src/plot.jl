function plot_levelset(outer, levelset)
    # Générer une grille de points
    x = range(outer[1][1], stop=outer[1][end], length=100)
    y = range(outer[2][1], stop=outer[2][end], length=100)

    # Évaluer la fonction de levelset à chaque point sur la grille
    z = [levelset(xi, yi) for xi in x, yi in y]

    # Tracer la fonction de levelset
    contour(x, y, z)
    
    println(z)
end

function plot_meshes(outer, inner)
    # Créer un nouveau graphique
    p = plot()

    # Ajouter des points pour le maillage extérieur
    for x in outer[1], y in outer[2]
        scatter!(p, [x], [y], color = :blue, label = false)
    end

    # Ajouter des points pour le maillage intérieur
    for x in inner[1], y in inner[2]
        scatter!(p, [x], [y], color = :red, label = false)
    end

    # Afficher le graphique
    return p
end

function plot_cut_cells(cut_cells, xyz)
    plot() # Create a new plot

    for (i, j) in cut_cells
        # Get the coordinates of the corners of the cell
        x_coords = [xyz[1][i], xyz[1][i+1], xyz[1][i+1], xyz[1][i], xyz[1][i]]
        y_coords = [xyz[2][j], xyz[2][j], xyz[2][j+1], xyz[2][j+1], xyz[2][j]]

        # Plot the cell in red
        plot!(x_coords, y_coords, fill = true, color = :red)
        
    end
    readline()
    display(plot) # Display the plot
end

function plot_border_cells(border_cells, xyz)
    plot() # Create a new plot

    for cell in border_cells
        i, j = cell.I # Get the indices from the CartesianIndex
        
        # Get the coordinates of the corners of the cell
        x_coords = [xyz[1][i], xyz[1][i+1], xyz[1][i+1], xyz[1][i], xyz[1][i]]
        y_coords = [xyz[2][j], xyz[2][j], xyz[2][j+1], xyz[2][j+1], xyz[2][j]]

        # Plot the cell in blue
        plot!(x_coords, y_coords, fill = true, color = :blue)
    end

    display(plot) # Display the plot
    readline()
end

function plot_cells(cut_cells, border_cells, xyz)
    plot() # Create a new plot

    for (i, j) in cut_cells
        # Get the coordinates of the corners of the cell
        x_coords = [xyz[1][i], xyz[1][i+1], xyz[1][i+1], xyz[1][i], xyz[1][i]]
        y_coords = [xyz[2][j], xyz[2][j], xyz[2][j+1], xyz[2][j+1], xyz[2][j]]

        # Plot the cut cell in red
        plot!(x_coords, y_coords, fill = true, color = :red)
    end

    for (i, j) in border_cells
        # Get the coordinates of the corners of the cell
        x_coords = [xyz[1][i], xyz[1][i+1], xyz[1][i+1], xyz[1][i], xyz[1][i]]
        y_coords = [xyz[2][j], xyz[2][j], xyz[2][j+1], xyz[2][j+1], xyz[2][j]]

        # Plot the border cell in blue
        plot!(x_coords, y_coords, fill = true, color = :blue, legend = false)
    end

    display(plot) # Display the plot
end