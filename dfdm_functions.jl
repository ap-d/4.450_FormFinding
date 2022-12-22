"""
Define a branch node connectivity matrix
"""
function BranchNodeMatrix(edges,nn)
    C = spzeros(length(edges),nn)
    for (i,edge) in enumerate(edges)
        C[i, edge[1]] = -1
        C[i, edge[2]] = 1
    end
    return C
end

"""
Solve for the free node positions 
"""
function fdm_solve(Cn, Cf, q, Pi, Nf)
    Q = sparse(diagm(q))
    new_xyz = (Cn' * Q * Cn) \ (Pi - Cn' * Q * Cf * Nf)
    return new_xyz
end

"""
Return a vector of lengths that matches the order of the edges in the connectivity matrix C
"""
function get_edge_lengths(new_xyz, Nf, free, fixed, edges)
    xyz = [new_xyz;Nf]
    indices = [free;fixed]
    p = sortperm(indices)
    sorted_pts = xyz[p,:]
    lengths = []
    for edge in edges
        length = norm(sorted_pts[edge[1],:]-sorted_pts[edge[2],:])
        lengths = [lengths; length]
    end
    return lengths
end

"""
Take in the list of objectives, weights, and target constraints
Return a single scalar value which represents the composite loss value
"""
function composite_objective(objectives, new_xyz, init_xyz, fixed, free, edges, Nf, q)
    weights = Float64[]
    individual_loss = Float64[]
    for objective in objectives
        weights = [weights; objective["weight"]]
        if objective["objective"] == "match_target_shape_euclidian_dist"
            loss = match_target_shape_euclidian_dist(init_xyz, new_xyz)
            individual_loss = [individual_loss;loss]
        elseif objective["objective"] == "match_target_shape_matrix_norm"
            loss = match_target_shape_matrix_norm(init_xyz, new_xyz)
            individual_loss = [individual_loss;loss]
        elseif objective["objective"] == "match_plan_location"
            loss = match_plan_location(init_xyz, new_xyz)
            individual_loss = [individual_loss;loss]
        elseif objective["objective"] == "minimum_edge_length"
            loss = minimum_edge_length(new_xyz, Nf, free, fixed, edges, objective["min_length"])
            individual_loss = [individual_loss;loss]
        elseif objective["objective"] == "min_sum_lengths"
            loss = min_sum_lengths(new_xyz, Nf, free, fixed, edges)
            individual_loss = [individual_loss;loss]
        elseif objective["objective"] == "maximum_edge_length"
            loss = minimum_edge_length(new_xyz, Nf, free, fixed, edges, objective["max_length"])
            individual_loss = [individual_loss;loss]
        elseif objective["objective"] == "structural_performance"
            loss = structural_performance(new_xyz, Nf, free, fixed, edges, q)
            individual_loss = [individual_loss;loss]
        elseif objective["objective"] == "min_force"
            loss = min_force(new_xyz, Nf, free, fixed, edges, q)
            individual_loss = [individual_loss;loss]
        elseif objective["objective"] == "min_variation"
            loss = min_variation(new_xyz, Nf, free, fixed, edges, q)
            individual_loss = [individual_loss;loss]
        else
            individual_loss = [individual_loss;0.0]
        end
    end
    weighted_loss = weights' * individual_loss
    return weighted_loss
end

"""
Return a unitless score of structural performance which is equivalent to ∑FL
"""
function structural_performance(new_xyz, Nf, free, fixed, edges, q)
    lengths = get_edge_lengths(new_xyz, Nf, free, fixed, edges)
    ∑FL = q' * (lengths.^2)
    return ∑FL
end

"""
Return the matrix norm of the component-wise difference between the 
initial xyz configuration and the new configuration
"""
function match_target_shape_matrix_norm(new_xyz::Matrix{Float64}, init_xyz::Matrix{Float64})
    return norm(new_xyz - init_xyz)
end

"""
Return the sum of squared distances between initial configuration and new configuration
"""
function match_target_shape_euclidian_dist(init, new)
    dist = new - init
    square = dist.*dist
    sum_rows = square*[1;1;1]
    sum_col = sum(sum_rows)
    return sum_col
end

"""
Returns a score where edges below the max length contribute 0 
and edges that are above the max length contribute 2 * length overshoot
"""
function maximum_edge_length(new_xyz, Nf, free, fixed, edges, max_length)
    lengths = get_edge_lengths(new_xyz, Nf, free, fixed, edges)
    threshold = -(max_length*ones(length(lengths))-lengths)
    zero_lengths_above_max_length = threshold + abs.(threshold)
    return sum(zero_lengths_above_max_length)
end

"""
Returns a score where edges above the minimum length contribute 0 
and edges that are below the minimum length contribute 2 * length undershoot
"""
function minimum_edge_length(new_xyz, Nf, free, fixed, edges, min_length)
    lengths = get_edge_lengths(new_xyz, Nf, free, fixed, edges)
    threshold = min_length*ones(length(lengths))-lengths
    zero_lengths_above_min_length = threshold + abs.(threshold)
    return sum(zero_lengths_above_min_length)
end

"""
Returns the sum of all forces in all members
"""

function min_force(new_xyz, Nf, free, fixed, edges, q)
    lengths = get_edge_lengths(new_xyz, Nf, free, fixed, edges)                         
    force = q.*lengths
    sum_force = sum(force)
    return sum_force
end

"""
Returns the difference between the highest force member and lowest force member
"""
function min_variation(new_xyz, Nf, free, fixed, edges, q)
    lengths = get_edge_lengths(new_xyz, Nf, free, fixed, edges)                         
    force = q.*lengths
    sorted = sort(force)
    difference = sorted[end]-sorted[1]
    return difference
end

"""
Return the sum of squared distances between initial configuration and new configuration
"""
function match_plan_location(init, new)
    dist = new[:,1:2] - init[:,1:2]
    square = dist.*dist
    sum_rows = square*[1;1]
    sum_col = sum(sum_rows)
    return sum_col
end

"""
Returns the sum of all way lengths in the network
"""
function min_sum_lengths(new_xyz, Nf, free, fixed, edges)
    lengths = get_edge_lengths(new_xyz, Nf, free, fixed, edges)                  
    sum_len = sum(lengths)
    println(sum_len)
    return sum_len
end
