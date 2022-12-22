
begin
    using HTTP.WebSockets
    using JSON
    using Zygote
    using Optimization
    using OptimizationNLopt
    using LinearAlgebra
    using SparseArrays

    include("dfdm_functions.jl")
    println("init complete")
end

server = WebSockets.listen!("127.0.0.1",2000) do ws
    for msg in ws
        println("new msg")
        msg = JSON.parse(msg)
        
        if msg == "close"
            println("Served closed by client message.")      
            close(server)
            
        else
            #format msg
            @time begin
                #Pull out edge keys and values
                edge_IDs = keys(msg["edges"])
                edges = values(msg["edges"])

                #Build array of node initial coordinates
                nodes_xyz = reduce(vcat, msg["nodes"]')
                nodes_xyz = convert(Array{Float64}, nodes_xyz)

                #Build lists of fixed and free node indices
                fixed = Int64.(msg["fixed"])
                #println(fixed)
                free = Int64.(msg["free"])
                #println(free)

                #Pull out force densities
                q = Float64.(msg["q"])

                #Pull out array of load vectors
                p = reduce(vcat, msg["p"]')
                p = convert(Array{Float64}, p)
                
                #Number of Nodes
                num_nodes = size(nodes_xyz, 1)

                #Construct branch-node matrix
                C = BranchNodeMatrix(edges, num_nodes)

                #Separate C into fixed and free
                Cn = C[:, free]
                Cf = C[:, fixed]

                #Separate nodes into fixed and free
                Nn = nodes_xyz[free, :]
                Nf = nodes_xyz[fixed, :]
                init_xyz = convert(Matrix{Float64},Nn)

                #Create array of loads on free nodes
                Pi = p[free, :]
                Pf = p[fixed, :]
            end

            #begin solving
            begin
                gh_settings = msg["settings"]
                if gh_settings == "no settings"
                    new_xyz = fdm_solve(Cn, Cf, q, Pi, Nf)
                    new_pos = [new_xyz;Nf]
                    node_indices = [free;fixed]
                    new_len = get_edge_lengths(new_xyz, Nf, free, fixed, edges)
                    return_data = Dict("new_pos" => new_pos, "node_indices" => node_indices, "q" => q, "edges" => [edge_IDs,edges], "lengths"=> new_len)
                else
                    begin
                        begin
                            lb = Float64.(gh_settings["lb"])*ones(length(q))
                            ub = Float64.(gh_settings["ub"])*ones(length(q))
                            abstol_gh = Float64.(gh_settings["abstol"])
                            reltol_gh = Float64.(gh_settings["reltol"])
                            maxiters_gh = Int64.(gh_settings["maxiters"])
                            freq = Int64.(gh_settings["freq"])
                            show = Bool.(gh_settings["show"])
                            objectives = gh_settings["objectives"]
                        end                        
                        
                        #main objective function
                        function obj(q::Vector{Float64}, p);
                            iter_count = iter_count +1
                            new_xyz = fdm_solve(Cn, Cf, q, Pi, Nf)
                            lengths = get_edge_lengths(new_xyz, Nf, free, fixed, edges)
                            loss = composite_objective(objectives, new_xyz, init_xyz, fixed, free, edges, Nf, q)
                            return loss, new_xyz, iter_count, lengths
                        end

                        #get_gradients objective function
                        function obj_grad(q::Vector{Float64}, p);
                            new_xyz = fdm_solve(Cn, Cf, q, Pi, Nf)
                            loss = composite_objective(objectives, new_xyz, init_xyz, fixed, free, edges, Nf, q)
                            return loss
                        end

                        #trace storage
                        x_storage = Vector{Vector{Float64}}();       
                        obj_storage = Vector{Float64}();
                        iter_count = 0;
                        opt_xyz = Vector{Matrix{Float64}}();
                        grad_storage = Vector{Vector{Float64}}();
                        len_storage = Vector{Vector{Float64}}()

                        #callback function
                        function cb(x_current, loss, new_xyz, iter_count, lengths)
                            push!(opt_xyz, [new_xyz;Nf])
                            #push!(x_storage, deepcopy(x_current))
                            push!(obj_storage, loss)
                            #push!(len_storage, lengths)
                            if show && mod(iter_count, freq) == 0
                                new_pos = [new_xyz;Nf]
                                node_indices = [free;fixed]
                                return_data = Dict("new_pos" => new_pos, "node_indices" => node_indices, "q" => x_current, "loss" => obj_storage, "edges" => [edge_IDs,edges], "lengths" => lengths)
                                send_back = json(return_data)
                                send(ws, send_back)
                            end
                            false
                        end    
                                                
                        
                        #grad = gradient(x -> obj(x, p), 10.0*ones(length(q)))
                        #println(grad)
                        #println("gradient")

                        #Define the optimization function and autodiff with zygote
                        optf = OptimizationFunction(obj, Optimization.AutoZygote());

                        #Define the optimization problem, initial configuration, lower and upper bounds
                        optp = OptimizationProblem(optf, q, p = SciMLBase.NullParameters(), lb = lb, ub = ub);                        

                        #Solve the optimization problem using LD_LBFGS() with gradients
                        @time sol = solve(optp, NLopt.LD_LBFGS(), abstol = abstol_gh, maxiters = maxiters_gh, callback = cb);

                        #println(sol)

                        new_xyz = fdm_solve(Cn, Cf, sol.u, Pi, Nf)
                        try
                            new_pos = last(opt_xyz)
                        catch
                            new_pos = [Nn;Nf]
                        end
                        node_indices = [free;fixed]
                        new_len = get_edge_lengths(new_xyz, Nf, free, fixed, edges)
                        

                        """
                        #Optionally send back gradients for every step of the optimization, uncommenting will slow down form finding
                        for force_densities in x_storage
                            grad_step = -gradient(x -> obj_grad(x, 1), force_densities)[1];
                            push!(grad_storage, grad_step)
                        end
                        """
                        


                        return_data = Dict("new_pos" => new_pos, "node_indices" => node_indices, "q" => sol.u,"loss" => obj_storage,
                                            "edges" => [edge_IDs,edges], "lengths" => new_len, 
                                            "xyz_all" => opt_xyz, "q_all" => x_storage, "iterations_to_converge" => iter_count, 
                                            "len_storage" => len_storage)
                    end
                end
            end
        end
        send_back = json(return_data)
        #println(send_back)
        send(ws, send_back)
        #println("network sent back to grasshopper")
    end
end
close(server)