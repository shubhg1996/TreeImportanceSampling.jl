POMDPs.solve(solver::ISDPWSolver, mdp::Union{POMDP,MDP}) = ISDPWPlanner(solver, mdp)

"""
Delete existing decision tree.
"""
function clear_tree!(p::ISDPWPlanner)
    p.tree = nothing
end

"""
Utility function for numerically stable softmax
Adapted from: https://nextjournal.com/jbowles/the-mathematical-ideal-and-softmax-in-julia
"""
_exp(x) = exp.(x .- maximum(x))
_exp(x, θ::AbstractFloat) = exp.((x .- maximum(x)) * θ)
_sftmax(e, d::Integer) = (e ./ sum(e, dims = d))

function softmax(X, dim::Integer)
    _sftmax(_exp(X), dim)
end

function softmax(X, dim::Integer, θ::Float64)
    _sftmax(_exp(X, θ), dim)
end

softmax(X) = softmax(X, 1)

"""
Calculate next action
"""
function select_action(nodes, values, prob_α, prob_p, n, α, β, γ)
    prob = adaptive_probs(values, prob_α, prob_p, n, α, β, γ)
    # prob = naive_probs(values)
    # prob = topk_probs(values, prob_p; k=2)
    # prob = expec_probs(values, prob_p)
    # prob = cdf_probs(values, prob_α, prob_p)
    # prob = cvar_probs(values, prob_α, prob_p)
    sanode_idx = sample(1:length(nodes), Weights(prob))
    sanode = nodes[sanode_idx]
    q_logprob = log(prob[sanode_idx])
    # @show prob, prob_p
    @show sanode_idx, prob, prob_p
    return sanode, q_logprob
end

function nominal_probs(values, prob_p)
    return prob_p
end

function naive_probs(values)
    prob = ones(length(values))
    prob /= sum(prob)
    return prob
end

function expec_probs(values, prob_p)
    prob = [values[i]*prob_p[i] for i=1:length(values)] .+ max(values...)/10 .+ 1e-7
    prob /= sum(prob)
    return prob
end

function cdf_probs(values, prob_α, prob_p)
    prob = [((1 - prob_α[i])*prob_p[i]) for i=1:length(values)] .+ (max(prob_α...))*max(prob_p...)/10 .+ 1e-7
    prob /= sum(prob)
    return prob
end

function cvar_probs(values, prob_α, prob_p)
    prob = [((1 - prob_α[i])*values[i]*prob_p[i]) for i=1:length(values)] .+ (max(prob_α...))*max(values...)/10 .+ 1e-7
    prob /= sum(prob)
    return prob
end

function topk_probs(values, prob_p; k=2)
    topk_idx = partialsortperm(values, 1:k, rev=true)
    prob = [(i in topk_idx ? values[i]*prob_p[i] : 0) for i=1:length(values)] .+ max(values...)/10 .+ 1e-7
    prob /= sum(prob)
    return prob
end

function adaptive_probs(values, prob_α, prob_p, n, α, β, γ)
    cvar_strategy = [(prob_α[i]*values[i]*prob_p[i]) for i=1:length(values)] .+ (max(prob_α...))*max(values...)/20 .+ 1e-5
    cdf_strategy = [(prob_α[i]*prob_p[i]) for i=1:length(values)] .+ (max(prob_α...))*max(prob_p...)/20 .+ 1e-5
    # mean_strategy = [values[i]*prob_p[i] for i=1:length(values)] .+ max(values...)/20 .+ 1e-7

    # @show prob_p, prob_α
    # Normalize to unity
    cvar_strategy /= sum(cvar_strategy)
    cdf_strategy /= sum(cdf_strategy)
    # mean_strategy /= sum(mean_strategy)

    # Mixture weighting
    prob = β*prob_p .+ γ*cdf_strategy .+ (1 - β - γ)*cvar_strategy

    return prob
end

"""
Calculate IS weights
"""
function compute_IS_weight(q_logprob, a, distribution)
    if distribution === nothing
        w = -q_logprob
    else
        w = logpdf(distribution, a) - q_logprob
    end
    # @show a, q_logprob, logpdf(distribution, a), w
    return w
end

"""
Construct an ISDPW tree and choose an action.
"""
POMDPs.action(p::ISDPWPlanner, s) = first(action_info(p, s))

MCTS.estimate_value(f::Function, mdp::Union{POMDP,MDP}, state, w::Float64, depth::Int) = f(mdp, state, w, depth)

"""
Construct an ISDPW tree and choose the best action. Also output some information.
"""
function POMDPModelTools.action_info(p::ISDPWPlanner, s; tree_in_info=false, w=0.0, β=0.0, γ=1.0, schedule=0.1)
    println("Mixed-tail strategy w/ mean")
    local a::actiontype(p.mdp)
    info = Dict{Symbol, Any}()
    try
        if isterminal(p.mdp, s)
            error("""
                  MCTS cannot handle terminal states. action was called with
                  s = $s
                  """)
        end

        S = statetype(p.mdp)
        A = actiontype(p.mdp)
        if p.solver.keep_tree
            if p.tree === nothing
                tree = ISDPWTree{S,A}(p.solver.n_iterations)
                p.tree = tree
            else
                tree = p.tree
            end
            if haskey(tree.s_lookup, s)
                snode = tree.s_lookup[s]
            else
                snode = MCTS.insert_state_node!(tree, s, true)
            end
        else
            tree = ISDPWTree{S,A}(p.solver.n_iterations)
            p.tree = tree
            snode = MCTS.insert_state_node!(tree, s, p.solver.check_repeat_state)
        end

        p.solver.show_progress ? progress = Progress(p.solver.n_iterations) : nothing
        nquery = 0
        start_us = MCTS.CPUtime_us()
        for i = 1:p.solver.n_iterations
            nquery += 1
            simulate(p, snode, w, p.solver.depth, β, γ, schedule) # (not 100% sure we need to make a copy of the state here)
            p.solver.show_progress ? next!(progress) : nothing
            if MCTS.CPUtime_us() - start_us >= p.solver.max_time * 1e6
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end
        end
        p.reset_callback(p.mdp, s) # Optional: leave the MDP in the current state.
        info[:search_time_us] = MCTS.CPUtime_us() - start_us
        info[:tree_queries] = nquery
        if p.solver.tree_in_info || tree_in_info
            info[:tree] = tree
        end

        sanode = best_sanode(tree, snode)
        a = tree.a_labels[sanode] # choose action randomly based on approximate value
    catch ex
        a = convert(actiontype(p.mdp), default_action(p.solver.default_action, p.mdp, s, ex))
        info[:exception] = ex
    end

    return a, info
end


"""
Return the reward for one iteration of ISDPW.
"""
function simulate(dpw::ISDPWPlanner, snode::Int, w::Float64, d::Int, β, γ, schedule)
    S = statetype(dpw.mdp)
    A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    s = tree.s_labels[snode]
    # s = TreeState(s, w)
    # dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.

    for i=tree.cdf_est.last_i+1:length(dpw.mdp.costs)
        ImportanceWeightedRiskMetrics.update!(tree.cdf_est, dpw.mdp.costs[i], exp(dpw.mdp.IS_weights[i]))
    end

    if isterminal(dpw.mdp, s)
        return 0.0, w
    elseif d == 0
        q_samp, w_samp = estimate_value(dpw.solved_estimate, dpw.mdp, s, w, d) # returns (r, w)
        return q_samp, w_samp
    end

    # action progressive widening
    if dpw.solver.enable_action_pw
        if length(tree.children[snode]) <= sol.k_action*tree.total_n[snode]^sol.alpha_action # criterion for new action generation
            a = next_action(dpw.next_action, dpw.mdp, s, MCTS.DPWStateNode(tree, snode)) # action generation step
            if !sol.check_repeat_action || !haskey(tree.a_lookup, (snode, a))
                n0 = init_N(sol.init_N, dpw.mdp, s, a)
                MCTS.insert_action_node!(tree, snode, a, n0,
                                    init_Q(sol.init_Q, dpw.mdp, s, a),
                                    sol.check_repeat_action
                                   )
                tree.total_n[snode] += n0
            end
        end
    elseif isempty(tree.children[snode])
        for a in support(actions(dpw.mdp, s))
            n0 = init_N(sol.init_N, dpw.mdp, s, a)
            MCTS.insert_action_node!(tree, snode, a, n0,
                                init_Q(sol.init_Q, dpw.mdp, s, a),
                                false)
            tree.total_n[snode] += n0
        end
    end

    all_UCB = []
    all_α = []
    prob_α   = []
    ltn = log(tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        c = sol.exploration_constant # for clarity
        if (ltn <= 0 && n == 0) || c == 0.0
            UCB = q
        else
            UCB = q + c*sqrt(ltn/n)
        end
        @assert !isnan(UCB) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(UCB, -Inf)

        push!(all_UCB, UCB)

        w_annealed = 1.0/(1.0+schedule*n)
        α = w_annealed + (1-w_annealed)*sol.α
        estimated_quantile = ImportanceWeightedRiskMetrics.quantile(tree.cdf_est, α)
        c_cdf = ImportanceWeightedRiskMetrics.cdf(tree.conditional_cdf_est[child], estimated_quantile)
        c_cdf = isnan(c_cdf) ? 1.0 : c_cdf
        push!(prob_α, 1.0 - c_cdf)
        push!(all_α, α)
    end
    # α = sol.α



    # prob_α = [1. - ImportanceWeightedRiskMetrics.cdf(tree.conditional_cdf_est[child], estimated_quantile) for child in tree.children[snode]]
    # while sum(prob_α) < α/100
    #     α = sqrt(α)
    #     estimated_quantile = ImportanceWeightedRiskMetrics.quantile(tree.cdf_est, α)
    #     prob_α = [1. - ImportanceWeightedRiskMetrics.cdf(tree.conditional_cdf_est[child], estimated_quantile) for child in tree.children[snode]]
    # end

    prob_p = [pdf(actions(dpw.mdp, s), tree.a_labels[child]) for child in tree.children[snode]]

    # @show tree.cdf_est, tree.conditional_cdf_est
    sanode, q_logprob = select_action(tree.children[snode], all_UCB, prob_α, prob_p, tree.cdf_est.last_i, all_α, β, γ)
    a = tree.a_labels[sanode] # choose action randomly based on approximate value
    w_node = compute_IS_weight(q_logprob, a, actions(dpw.mdp, s))
    # @show q_logprob, actions(dpw.mdp, s)
    w = w + w_node

    # transition
    new_node = false
    sp, r = @gen(:sp, :r)(dpw.mdp, s, [a, w], dpw.rng)
    # sp = TreeState(sp)  # remove weight to check repetition
    # @show sp.values, sp.costs, sp.mdp_state, sp.done, sp.w

    if sol.check_repeat_state && tree.n_a_children[sanode] > 0
        spnode, r_stored = rand(dpw.rng, tree.transitions[sanode])
        tree.s_labels[spnode] = sp
        tree.s_lookup[sp] = spnode
    else
        spnode = MCTS.insert_state_node!(tree, sp, sol.keep_tree || sol.check_repeat_state)
        new_node = true
    end
    # @show tree.s_labels[spnode].mdp_state === sp.mdp_state

    push!(tree.transitions[sanode], (spnode, r))

    if !sol.check_repeat_state
        tree.n_a_children[sanode] += 1
    elseif !((sanode,spnode) in tree.unique_transitions)
        push!(tree.unique_transitions, (sanode,spnode))
        tree.n_a_children[sanode] += 1
    end

    if new_node
        q_samp, w_samp = estimate_value(dpw.solved_estimate, dpw.mdp, sp, w, d-1)
        # @show "Estimated", q_samp, w_samp
        q = r + discount(dpw.mdp)*q_samp
    else
        q_samp, w_samp = simulate(dpw, spnode, w, d-1, β, γ, schedule)
        q = r + discount(dpw.mdp)*q_samp
    end

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    ImportanceWeightedRiskMetrics.update!(tree.conditional_cdf_est[sanode], q, exp(w_samp)) # check how to weight samples effectively

    return q, w_samp
end

"""
Return the best action.
Some publications say to choose action that has been visited the most
e.g., Continuous Upper Confidence Trees by Couëtoux et al.
"""
function best_sanode(tree::ISDPWTree, snode::Int)
    best_Q = -Inf
    sanode = 0
    for child in tree.children[snode]
        if tree.q[child] > best_Q
            best_Q = tree.q[child]
            sanode = child
        end
    end
    return sanode
end
