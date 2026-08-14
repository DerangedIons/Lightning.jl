"""
    AbstractStimulationProtocol

Supertype for externally applied stimulation currents.

A protocol is a callable `(x, t) -> Iₐₚₚ` returning the applied current density at
physical position `x` (an `SVector` of cell-center coordinates) and time `t`.

# Sign convention

`Iₐₚₚ` is in the **PDE-form convention: positive is depolarizing.** It enters the
monodomain equation as the `χIₛₜᵢₘ` term of

```
χCₘ∂ₜφₘ = ∇⋅κ∇φₘ + χ(Iᵢₒₙ + Iₛₜᵢₘ)
```

so a positive value drives `φₘ` up.

!!! warning "This is the opposite of the cell-model convention"
    Inside a cell model the stimulus is a *membrane current* and follows the
    negative-inward convention — `CytoZoo`'s default `Stimulus` amplitude is
    `-53 µA/µF`, and its right-hand side *subtracts* `Iₛₜᵢₘ` from `dv/dt`. Here a
    stimulus is a source term on the tissue-scale PDE and is *added*. A protocol
    ported from a cell model needs its sign flipped.

Stimulation belongs to the diffusion half of the reaction-diffusion split, uniformly in
every dimension. The cell model's own stimulus must therefore be off;
[`MonodomainModel`](@ref) warns when it detects one that is not.

See also: [`NoStimulationProtocol`](@ref), [`TransmembraneStimulationProtocol`](@ref).
"""
abstract type AbstractStimulationProtocol end

"""
    NoStimulationProtocol()

The absence of an applied stimulus. Dispatching on this type removes the stimulus
evaluation from the diffusion right-hand side entirely rather than multiplying by zero.
"""
struct NoStimulationProtocol <: AbstractStimulationProtocol end

@inline (::NoStimulationProtocol)(x, t) = false

"""
    TransmembraneStimulationProtocol(f; nonzero_intervals = nothing)

Applied stimulus current `f(x, t) -> Iₐₚₚ`, in the positive-is-depolarizing convention
documented on [`AbstractStimulationProtocol`](@ref).

`nonzero_intervals` is an optional collection of `(t₀, t₁)` windows outside which `f` is
known to vanish. When supplied, the diffusion right-hand side skips the whole stimulus
evaluation for times outside every window — worth having when a single 2 ms pulse rides on
a 500 ms simulation. It is stored as a `Tuple`, not a `Vector`, so the protocol stays
isbits and can be captured by value in a GPU broadcast.

The protocol is isbits (and therefore GPU-safe) exactly when `f` is: a plain function or a
small immutable functor is fine, a closure over a boxed variable is not.

### Examples

```julia
# 2 ms depolarizing pulse over the left 1 mm of a cable
stim = TransmembraneStimulationProtocol(
    (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 50.0 : 0.0;
    nonzero_intervals = ((0.0, 2.0),),
)
```
"""
struct TransmembraneStimulationProtocol{F,I} <: AbstractStimulationProtocol
    f::F
    nonzero_intervals::I
end

function TransmembraneStimulationProtocol(f; nonzero_intervals=nothing)
    intervals = _normalize_intervals(nonzero_intervals)
    return TransmembraneStimulationProtocol(f, intervals)
end

_normalize_intervals(::Nothing) = nothing

function _normalize_intervals(intervals)
    normalized = Tuple(
        map(intervals) do iv
            length(iv) == 2 || throw(
                ArgumentError("each nonzero interval must be a (t₀, t₁) pair, got $iv")
            )
            t₀, t₁ = iv
            t₀ <= t₁ || throw(
                ArgumentError("nonzero interval $iv is empty: t₀ must not exceed t₁")
            )
            return (t₀, t₁)
        end,
    )
    isempty(normalized) && throw(
        ArgumentError(
            "nonzero_intervals is empty, which would silence the protocol at every time; " *
            "pass `nothing` to evaluate it unconditionally",
        ),
    )
    return normalized
end

@inline (p::TransmembraneStimulationProtocol)(x, t) = p.f(x, t)

"""
    is_active(protocol, t) -> Bool

Whether `protocol` can be nonzero at time `t`. Conservative: a protocol without declared
`nonzero_intervals` is always reported active. Used by the diffusion right-hand side to
skip stimulus evaluation, never to decide the value.
"""
is_active(::NoStimulationProtocol, t) = false
is_active(::TransmembraneStimulationProtocol{F,Nothing}, t) where {F} = true

function is_active(p::TransmembraneStimulationProtocol, t)
    for (t₀, t₁) in p.nonzero_intervals
        t₀ <= t <= t₁ && return true
    end
    return false
end

function Adapt.adapt_structure(to, p::TransmembraneStimulationProtocol)
    return TransmembraneStimulationProtocol(Adapt.adapt(to, p.f), p.nonzero_intervals)
end
