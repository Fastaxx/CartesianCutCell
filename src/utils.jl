function volume_integrated_p_norm(e1, e2, V, p)
    total_volume = sum(V)
    integral = sum((abs.(e1[i] - e2[i])^p * V[i] for i in 1:length(V)))
    return (integral / total_volume)^(1/p)
end