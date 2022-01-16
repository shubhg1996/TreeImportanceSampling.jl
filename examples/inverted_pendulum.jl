using Revise
using POMDPs, POMDPGym, POMDPSimulators, POMDPPolicies, Distributions, Plots
using Crux, Flux
using MCTS
using FileIO

# Basic MDP
mdp = InvertedPendulumMDP(λcost=0, include_time_in_state=true)

# Learn a policy that solves it
policy = ActorCritic(GaussianPolicy(ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))), [0f0]),
                     ContinuousNetwork(Chain(Dense(3, 32, relu), Dense(32, 1))))
policy = solve(PPO(π=policy, S=state_space(mdp), N=20000, ΔN=400), mdp)

# Define the disturbance distribution based on a normal distribution
xnom = Normal(0f0, 0.5f0)
xs = Float32[-2., -0.5, 0, 0.5, 2.]
ps = exp.([logpdf(xnom, x) for x in xs])
ps ./= sum(ps)
px = DiscreteNonParametric(xs, ps)

# Redefine disturbance to find action space
POMDPGym.disturbances(mdp::AdditiveAdversarialMDP) = support(mdp.x_distribution)
POMDPGym.disturbanceindex(mdp::AdditiveAdversarialMDP, x) = findfirst(support(mdp.x_distribution) .== x)

prior(mdp::AdditiveAdversarialMDP) = probs(mdp.x_distribution)

# Construct the adversarial MDP to get access to a transition function like gen(mdp, s, a, x)
amdp = AdditiveAdversarialMDP(mdp, px)

function eval_cost(x)
    x_inv = 1 / (abs(x - mdp.failure_thresh) + 1e-3)
    return x_inv
end

# Construct the risk estimation mdp where actions are disturbances
rmdp = RMDP(amdp, policy, (m, s) -> eval_cost(s[1]))

N = 10000
c = 0.3
softmax_temp = 5.0

fixed_s = rand(initialstate(amdp))

# BASELINE

samps = [maximum(collect(simulate(HistoryRecorder(), rmdp, FunctionPolicy((s) -> rand(px)), fixed_s)[:r])) for _ in 1:N]

save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/inverted_pendulum_baseline_$(N).jld2", Dict("risks:" => samps, "states:" => [], "IS_weights:" => []))

# MCTS

using TreeImportanceSampling

tree_mdp = TreeImportanceSampling.construct_tree_rmdp(rmdp, px; reduction="max")

planner = TreeImportanceSampling.mcts_isdpw(tree_mdp; N, c)

a, info = action_info(planner, TreeImportanceSampling.TreeState(fixed_s); tree_in_info=true, softmax_temp=softmax_temp)

N_dist = length(xs)

save("/home/users/shubhgup/Codes/AutonomousRiskFramework.jl/data/inverted_pendulum_mcts_IS_$(N).jld2", Dict("risks:" => planner.mdp.costs, "states:" => [], "IS_weights:" => planner.mdp.IS_weights, "tree:" => info[:tree]))
