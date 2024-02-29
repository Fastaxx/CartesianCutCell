using Plots
Plots.default(show = true)

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