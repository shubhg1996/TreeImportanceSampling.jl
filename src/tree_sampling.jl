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
# strategy_text = "Naive"
# strategy_text = "Adaptive"
# strategy_text = "Mean"
# strategy_text = "CVaR"
strategy_text = "VaR"

function select_action(nodes, prob)
    sanode_idx = sample(1:length(nodes), Weights(prob))
    sanode = nodes[sanode_idx]
    q_logprob = log(prob[sanode_idx])
    return sanode, q_logprob
end

function compute_action_probs(nodes, values, prob_α, cost_α, prob_p; kwargs...)
#     prob = adaptive_probs(values, prob_α, cost_α, prob_p; kwargs...)
#     prob = naive_probs(values)
#     prob = nominal_probs(prob_p)
    # prob = topk_probs(values, prob_p; k=2)
#     prob = expec_probs(values, prob_p)
    prob = cdf_probs(values, prob_α, prob_p; kwargs...)
#     prob = cvar_probs(cost_α, prob_p; kwargs...)
    return prob
end

function nominal_probs(prob_p)
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

function cdf_probs(values, prob_α, prob_p; n=1, uniform_floor=0.01, exploration_bonus=0.0, kwargs...)
    prob = [(prob_α[i]*prob_p[i]) for i=1:length(prob_α)] .+ 1e-7
    prob /= sum(prob)
    w_nominal = 0.0 + cooling_scheme_logarithmic(n; w0=1.0)
#     @show w_nominal, n
    
#     prob_defensive = ones(length(prob))
#     prob_defensive /= sum(prob_defensive)
    
    prob_defensive = prob_p
    
    prob_out = w_nominal*prob_defensive .+ (1-w_nominal)*prob
    prob_out = prob_out .+ exploration_bonus
    prob_out /= sum(prob_out)
    return prob_out
end

function cvar_probs(cost_α, prob_p; uniform_floor=0.01, kwargs...)
    prob = [(cost_α[i]*prob_p[i]) for i=1:length(cost_α)] .+ (max(cost_α..., 1e-5)*uniform_floor)
    prob /= sum(prob)
    return prob
end

function topk_probs(values, prob_p; k=2)
    topk_idx = partialsortperm(values, 1:k, rev=true)
    prob = [(i in topk_idx ? values[i]*prob_p[i] : 0) for i=1:length(values)] .+ max(values...)/10 .+ 1e-7
    prob /= sum(prob)
    return prob
end

function adaptive_probs(values, prob_α, cost_α, prob_p; β=0.0, γ=0.0, uniform_floor=0.01, kwargs...)
    cvar_strategy = [(cost_α[i]*prob_p[i]) for i=1:length(values)] .+ (max(cost_α..., 1e-5)*uniform_floor)
    cdf_strategy = [(prob_α[i]*prob_p[i]) for i=1:length(values)] .+ (uniform_floor)
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
Calculate α via schedule
"""
function schedule_α(α, n; schedule=Inf)
    w_annealed = cooling_scheme_exponential(n; β=schedule)
    return w_annealed + (1-w_annealed)*α
end

cooling_scheme_geometric(n; β=0.5, w0=0.99) = (β^n)*w0;
cooling_scheme_logarithmic(n; w0=0.99) = w0*log(2)/log(n+2);
cooling_scheme_hybrid(n; thresh=20, β=0.5, w0=0.99) = n > thresh ? cooling_scheme_geometric(n; β, w0) : w0/(n+1);
cooling_scheme_exponential(n; β=0.5, w0=0.99) = w0/(1 + n*β*w0);

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
function POMDPModelTools.action_info(p::ISDPWPlanner, s; tree_in_info=false, w=0.0, save_frequency=10, save_callback=nothing, kwargs...)
    @show "$(strategy_text) strategy"
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
            q_samp, w_samp = simulate(p, snode, w, p.solver.depth; kwargs...) # (not 100% sure we need to make a copy of the state here)
            ImportanceWeightedRiskMetrics.update!(p.tree.cdf_est, q_samp, exp(w_samp))
            p.solver.show_progress ? next!(progress) : nothing
            if !isnothing(save_callback) && (i % save_frequency == 0)
                save_callback(p)
            end
            if MCTS.CPUtime_us() - start_us >= p.solver.max_time * 1e6
                p.solver.show_progress ? finish!(progress) : nothing
                break
            end
        end
        # p.reset_callback(p.mdp, s) # Optional: leave the MDP in the current state.
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

function store_actions!(dpw::ISDPWPlanner, tree, sol, s, snode)
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
end

function store_state!(dpw::ISDPWPlanner, tree, sol, sp, r, sanode)
    new_node = false
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
    return spnode, new_node
end

function compute_q_w(dpw::ISDPWPlanner, sp, spnode, r, w, d, new_node; kwargs...)
    if new_node
        q_samp, w_samp = estimate_value(dpw.solved_estimate, dpw.mdp, sp, w, d-1)
        # @show "Estimated", q_samp, w_samp
        q = r + discount(dpw.mdp)*q_samp
    else
        q_samp, w_samp = simulate(dpw, spnode, w, d-1; kwargs...)
        q = r + discount(dpw.mdp)*q_samp
    end
    return q, w_samp
end

function get_nominal_prob(dpw::ISDPWPlanner, tree, snode, s)
    [pdf(actions(dpw.mdp, s), tree.a_labels[child]) for child in tree.children[snode]]
end

function compute_costs(dpw::ISDPWPlanner, tree, sol, snode; schedule=Inf)
    all_UCB = []
    all_α = []
    cost_α   = []
    prob_α   = []
    exploration_bonus = []
    ltn = log(1 + tree.total_n[snode])
    for child in tree.children[snode]
        n = tree.n[child]
        q = tree.q[child]
        c = sol.exploration_constant # for clarity
        if (ltn <= 0 && n == 0) || c == 0.0
            _exp_bonus = 0.0
        else
            _exp_bonus = c*sqrt(ltn/(1 + n))
        end
        @assert !isnan(_exp_bonus) "UCB was NaN (q=$q, c=$c, ltn=$ltn, n=$n)"
        @assert !isequal(_exp_bonus, -Inf)

        push!(all_UCB, q)

        α = schedule_α(sol.α, n; schedule=schedule)
        estimated_quantile = ImportanceWeightedRiskMetrics.quantile(tree.cdf_est, α)
        c_tail = ImportanceWeightedRiskMetrics.tail_cost(tree.conditional_cdf_est[child], estimated_quantile)
        c_cdf = ImportanceWeightedRiskMetrics.cdf(tree.conditional_cdf_est[child], estimated_quantile)
        push!(cost_α, c_tail)
        push!(prob_α, 1.0 - c_cdf)
        push!(exploration_bonus, _exp_bonus)
        push!(all_α, α)
    end
    # α = sol.α
    return all_UCB, all_α, cost_α, prob_α, exploration_bonus
end

"""
Return the reward for one iteration of ISDPW.
"""
function simulate(dpw::ISDPWPlanner, snode::Int, w::Float64, d::Int; schedule=Inf, kwargs...)
    S = statetype(dpw.mdp)
    A = actiontype(dpw.mdp)
    sol = dpw.solver
    tree = dpw.tree
    s = tree.s_labels[snode]
    # dpw.reset_callback(dpw.mdp, s) # Optional: used to reset/reinitialize MDP to a given state.

    if isterminal(dpw.mdp, s)
        return 0.0, w
    elseif d == 0
        q_samp, w_samp = estimate_value(dpw.solved_estimate, dpw.mdp, s, w, d) # returns (r, w)
        return q_samp, w_samp
    end

    store_actions!(dpw, tree, sol, s, snode)

    all_UCB, all_α, cost_α, prob_α, exploration_bonus = compute_costs(dpw, tree, sol, snode; schedule=schedule)

    prob_p = get_nominal_prob(dpw, tree, snode, s)

    q_prob = compute_action_probs(tree.children[snode], all_UCB, prob_α, cost_α, prob_p; n=tree.total_n[snode], α=all_α, exploration_bonus=exploration_bonus, kwargs...)
    
    sanode, q_logprob = select_action(tree.children[snode], q_prob)
    a = tree.a_labels[sanode]
    w_node = compute_IS_weight(q_logprob, a, actions(dpw.mdp, s))
    w = w + w_node

    # transition
    sp, r = @gen(:sp, :r)(dpw.mdp, s, [a, w], dpw.rng)
    
    spnode, new_node = store_state!(dpw::ISDPWPlanner, tree, sol, sp, r, sanode)

    q, w_samp = compute_q_w(dpw, sp, spnode, r, w, d, new_node; kwargs...)

    tree.n[sanode] += 1
    tree.total_n[snode] += 1

    tree.q[sanode] += (q - tree.q[sanode])/tree.n[sanode]

    ImportanceWeightedRiskMetrics.update!(tree.conditional_cdf_est[sanode], q, exp(w_samp)) # check how to weight samples effectively
    
    return q, w_samp
end

"""
Create a lookup indexable by mdp state
"""
function mdp_state_lookup(dpw::ISDPWPlanner)
    tree = dpw.tree
    ret = Dict(key.mdp_state => value for (key, value) in tree.s_lookup)
    return ret
end

"""
Return action for a base MDP state
"""
function tree_policy(dpw::ISDPWPlanner, mdp_state, lookup; schedule=Inf, kwargs...)
    sol = dpw.solver
    tree = dpw.tree
    s = TreeState([], [0.0], mdp_state, isterminal(dpw.mdp.rmdp, mdp_state), 0.0)
   
    if mdp_state in keys(lookup)
        snode = lookup[mdp_state]
    else
        snode = nothing
    end
    
    if snode===nothing || isempty(tree.children[snode])
        return rand(actions(dpw.mdp, s))
    end
        
        
    all_UCB, all_α, cost_α, prob_α, exploration_bonus = compute_costs(dpw, tree, sol, snode; schedule=schedule)
    
    prob_p = get_nominal_prob(dpw, tree, snode, s)
    q_prob = compute_action_probs(tree.children[snode], all_UCB, prob_α, cost_α, prob_p; n=tree.total_n[snode], α=all_α, exploration_bonus=exploration_bonus, kwargs...)
    
    sanode, q_logprob = select_action(tree.children[snode], q_prob)
    a = tree.a_labels[sanode]
    return a
end


"""
Generate a sample trajectory from a built tree
"""
function sample_trajectory(dpw::ISDPWPlanner, s0; kwargs...)
    info = Dict(:states => [], :actions => [], :rewards => [])
    sol = dpw.solver
    tree = dpw.tree

    snode - tree.s_lookup[s0]
    s = tree.s_labels[snode]

    w = 0.0
    intree = true

    while !isterminal(dpw.mdp, s)
        if intree
            all_UCB = []
            all_α = []
            cost_α   = []
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

                α = sol.α
                estimated_quantile = ImportanceWeightedRiskMetrics.quantile(tree.cdf_est, α)
                c_tail = ImportanceWeightedRiskMetrics.tail_cost(tree.conditional_cdf_est[child], estimated_quantile)
                c_cdf = ImportanceWeightedRiskMetrics.cdf(tree.conditional_cdf_est[child], estimated_quantile)
                push!(cost_α, c_tail)
                push!(prob_α, 1.0 - c_cdf)
                push!(all_α, α)
            end

            prob_p = [pdf(actions(dpw.mdp, s), tree.a_labels[child]) for child in tree.children[snode]]

            sanode, q_logprob = select_action(tree.children[snode], all_UCB, prob_α, cost_α, prob_p; n=tree.cdf_est.last_i, α=all_α, kwargs...)
            a = tree.a_labels[sanode] # choose action randomly based on approximate value
            w_node = compute_IS_weight(q_logprob, a, actions(dpw.mdp, s))
            w = w + w_node

            if tree.n_a_children[sanode] > 0
                spnode, r_stored = rand(dpw.rng, tree.transitions[sanode])
            else
                intree = false
            end
        else
            a = rand(actions(dpw.mdp, s))
        end

        sp, r = @gen(:sp, :r)(dpw.mdp, s, [a, w], dpw.rng)

        push!(info[:states], sp)
        push!(info[:actions], a)
        push!(info[:rewards], r)

        s = sp
        snode = spnode
    end

    return info
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
