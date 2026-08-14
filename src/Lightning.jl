"""
    Lightning

Cardiac tissue electrophysiology on orthogonal, structured grids.

The pipeline mirrors Thunderbolt's by convention — same names, same shapes, no dependency:

```julia
model = MonodomainModel(; κ, ion)                       # continuous problem
split = ReactionDiffusionSplit(model)                   # how to split it
f     = semidiscretize(split, FiniteDifferenceDiscretization(), grid)
prob  = OperatorSplittingProblem(f, create_initial_condition(f), tspan)
integrator = init(prob, LieTrotterGodunov((Euler(), Euler())); dt)
```

Spatial operators come from MatrixFreeOperators.jl and cell kinetics from CytoZoo.jl.
Restricting the domain to orthogonal structured grids is what buys the matrix-free
operators; unstructured meshes are Thunderbolt's job, not this package's.

See also: [`MonodomainModel`](@ref), [`semidiscretize`](@ref),
[`create_initial_condition`](@ref).
"""
module Lightning

using Adapt: Adapt
using LinearAlgebra: mul!
using StaticArrays: SVector

using KernelAbstractions: KernelAbstractions
using KernelAbstractions: @index, @kernel

using CytoZoo: CytoZoo

# Explicit imports throughout: MatrixFreeOperators exports `solve` and `gradient`, which
# would clash with the SciML stack and with Base.
using MatrixFreeOperators:
    CartesianGrid,
    Dirichlet,
    Neumann,
    Periodic,
    boundary_conditions,
    cell_center,
    derivative,
    dimension,
    interior,
    laplacian,
    local_size,
    prepare,
    scalar_field,
    spacing

using OrdinaryDiffEqOperatorSplitting:
    GenericSplitFunction, LieTrotterGodunov, OperatorSplittingProblem, StrangMarchuk

using SciMLBase: ODEFunction, init, solve!, step!

include("stimulus.jl")
include("models.jl")
include("discretization.jl")

# Lightning's own vocabulary
export AbstractEPModel, MonodomainModel, ReactionDiffusionSplit
export AbstractStimulationProtocol, NoStimulationProtocol, TransmembraneStimulationProtocol
export FiniteDifferenceDiscretization

# Re-exported mesh vocabulary, so `using Lightning` is enough to build a problem.
export CartesianGrid, Dirichlet, Neumann, Periodic
export boundary_conditions, cell_center, dimension, interior, local_size, spacing

end # module
