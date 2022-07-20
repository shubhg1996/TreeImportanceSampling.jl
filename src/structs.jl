struct TreeISParams
    c::Float64
    α::Float64

    β::Float64
    γ::Float64

    schedule::Int64 # set to Inf to switch off

    uniform_floor::Float64 # set to 0.0 to switch off
end

TreeISParams() = TreeISParams(0.0, 1e-3, 0.0, 0.0, 1.0, 1e-6)