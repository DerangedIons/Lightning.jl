# Aliev–Panfilov (1996) reduced excitable-medium model, as a CytoZoo cell model.
#
#   dV/dt = −kV(V − a)(V − 1) − VW
#   dW/dt = (ε₀ + μ₁W/(μ₂ + V))(−W − kV(V − a − 1))
#
# Two states, dimensionless time and space, and a genuinely restitution-capable
# recovery variable — the cheapest kinetics that still supports a stable reentrant
# spiral, which is why it is the standard model for spiral-wave demonstrations.
#
# Lives in `examples/` rather than in CytoZoo because it is here to drive a figure.
# Promoting it to the zoo is a separate conversation.

using CytoZoo: CytoZoo, SpatialContext, Stimulus

"""
    AlievPanfilov(; k = 8.0, a = 0.15, ε₀ = 0.002, μ₁ = 0.2, μ₂ = 0.3, stim = Stimulus(; amplitude = 0))

Aliev–Panfilov kinetics. `V` runs over roughly `[0, 1]` and time is dimensionless.

Following the same convention as `CytoZoo.FHNModel`: the parameters are plain
isbits fields, and `stim` defaults to zero amplitude because in tissue the stimulus belongs
to the framework's diffusion half. The sign convention is CytoZoo's — `Istim` is subtracted
from `dV/dt`, so a *negative* amplitude depolarizes.
"""
struct AlievPanfilov{T<:Number,S} <: CytoZoo.AbstractCardiacCellModel
    k::T
    a::T
    ε₀::T
    μ₁::T
    μ₂::T
    stim::S
end

function AlievPanfilov(;
    k::Real=8.0,
    a::Real=0.15,
    ε₀::Real=0.002,
    μ₁::Real=0.2,
    μ₂::Real=0.3,
    stim=Stimulus(; amplitude=0.0),
)
    kk, aa, e, m1, m2 = promote(k, a, ε₀, μ₁, μ₂)
    return AlievPanfilov(kk, aa, e, m1, m2, stim)
end

CytoZoo.num_states(::AlievPanfilov) = 2
CytoZoo.num_parameters(::AlievPanfilov) = 5
CytoZoo.transmembrane_potential_index(::AlievPanfilov) = 1
CytoZoo.default_initial_state(::AlievPanfilov{T}) where {T} = T[0, 0]
CytoZoo.state_names(::AlievPanfilov) = (:V, :W)
CytoZoo.parameter_names(::AlievPanfilov) = (:k, :a, :ε₀, :μ₁, :μ₂)
CytoZoo.state_index(::AlievPanfilov, n::Symbol) = findfirst(==(n), (:V, :W))
function CytoZoo.parameter_index(::AlievPanfilov, n::Symbol)
    return findfirst(==(n), (:k, :a, :ε₀, :μ₁, :μ₂))
end

function _aliev_panfilov_rhs!(du, u, m::AlievPanfilov, x, t)
    V, W = u[1], u[2]
    Istim = m.stim(x, t)
    du[1] = -m.k * V * (V - m.a) * (V - 1) - V * W - Istim
    du[2] = (m.ε₀ + m.μ₁ * W / (m.μ₂ + V)) * (-W - m.k * V * (V - m.a - 1))
    return nothing
end

function (m::AlievPanfilov)(du, u, ::Nothing, t)
    _aliev_panfilov_rhs!(du, u, m, nothing, t)
    return nothing
end

function (m::AlievPanfilov)(du, u, p::SpatialContext, t)
    _aliev_panfilov_rhs!(du, u, m, p.x, t)
    return nothing
end
