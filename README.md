# Lightning.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DerangedIons.github.io/Lightning.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DerangedIons.github.io/Lightning.jl/dev/)
[![Build Status](https://github.com/DerangedIons/Lightning.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/DerangedIons/Lightning.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/DerangedIons/Lightning.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DerangedIons/Lightning.jl)

Cardiac tissue electrophysiology on orthogonal, structured grids.

Lightning solves the monodomain equation

```
χCₘ∂ₜφₘ = ∇⋅κ∇φₘ + χ(Iᵢₒₙ(φₘ, s, t) + Iₛₜᵢₘ(x, t))
                 ∂ₜs = f(φₘ, s, t)
```

by reaction-diffusion operator splitting: [MatrixFreeOperators.jl](https://github.com/DerangedIons/MatrixFreeOperators.jl)
supplies the spatial operators, [CytoZoo.jl](https://github.com/DerangedIons/CytoZoo.jl) the cell
kinetics, and [OrdinaryDiffEqOperatorSplitting.jl](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl)
the splitting itself.

## What it does, and what it deliberately does not

Restricting the domain to **orthogonal structured grids** is the whole design. That
restriction is what makes the spatial operators matrix-free — no assembly, no sparse
factorization, no mesh data structure — and it is why the same source runs on a GPU without a
second implementation. Unstructured meshes, curved geometry, and finite elements are
[Thunderbolt.jl](https://github.com/JuliaHealth/Thunderbolt.jl)'s job, not this package's.

The pipeline is nevertheless *congruent* with Thunderbolt's by convention — same names, same
shapes, so a model description reads the same in either — while depending on none of it.

**Currently in:** monodomain, N-dimensional, isotropic or axis-aligned anisotropic
conductivity, zero-flux boundaries, arbitrary CytoZoo cell models, spatial parameter
heterogeneity, CPU and CUDA.

**Currently out:** unstructured meshes, bidomain, mechanics, spatially varying conductivity
fields, non-zero-flux boundaries, Rush–Larsen inner solvers, adaptive mesh refinement,
conduction-velocity and pseudo-ECG post-processing.

## The pipeline

```julia
using Lightning
using CytoZoo: ParametrizedFHNModel
using OrdinaryDiffEqLowOrderRK: Euler

# 1. A grid. Cell-centered, uniform, zero flux at both ends.
grid = CartesianGrid(((0.0, 20.0),), (400,); bc = ((Neumann(), Neumann()),))

# 2. A stimulus. Positive is depolarizing — the PDE-form convention, which is the
#    opposite of the cell-model-internal one. `nonzero_intervals` lets the right-hand
#    side skip the evaluation once the pulse is over.
stim = TransmembraneStimulationProtocol(
    (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 0.5 : 0.0;
    nonzero_intervals = ((0.0, 2.0),),
)

# 3. The continuous problem, then how to split it, then the semidiscretization.
model = MonodomainModel(; κ = 0.1, ion = ParametrizedFHNModel(), stim)
f = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)

# 4. Solve. Diffusion and reaction each get their own inner algorithm.
prob = OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 400.0))
integrator = init(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005)

while integrator.t < 400.0
    step!(integrator, 0.005, true)
    φ = getvariable(integrator.u, f, :φₘ)     # a view, not a copy
end
```

`examples/cable_1d.jl` is this with reporting and a figure attached.

## The solution vector

State-blocked (structure of arrays), matching Thunderbolt's `StateBlockedLayout`:

```
u = [φₘ(1:N); s₁(1:N); s₂(1:N); … ; s_{M-1}(1:N)]
```

Every state is contiguous, so the diffusion half acts on the leading block `1:N` with no
gather and no synchronizer object, and neighbouring GPU threads read neighbouring nodes.
Reach into it by name rather than by index:

```julia
create_initial_condition(f)             # every node at the cell model's default state
getvariable(u, f, :φₘ)                  # view of the voltage block
getvariable(u, f, :s)                   # view of every non-voltage state
setvariable!(u, f, :φₘ) do x            # impose a profile
    x[1] < 1.0 ? 1.0 : 0.0
end
```

## GPU

Put the backend on the grid and the whole pipeline follows — field allocation, the prepared
operator's scratch, node coordinates, and the initial condition:

```julia
using CUDA
grid = CartesianGrid(((0.0, 20.0),), (400,);
                     bc = ((Neumann(), Neumann()),), device = CUDABackend())
```

The reaction half then runs as one KernelAbstractions kernel over the nodes. The cell model
has to be isbits to ride into it — `CytoZoo.ParametrizedFHNModel` is; `ToRORd`, whose
parameters live in a heap `Vector`, is not yet.

## Precision

Use `Float64`. The types are generic and `Float32` runs, but the ArmyHeart reference measured
non-finite results across 58.6% of the ToRORd parameter space in single precision.

## Status

Pre-release, unregistered, and the API is expected to move. Install from source alongside its
unregistered dependencies:

```julia
using Pkg
Pkg.develop(url = "https://github.com/DerangedIons/MatrixFreeOperators.jl")
Pkg.develop(url = "https://github.com/DerangedIons/CytoZoo.jl")
Pkg.develop(url = "https://github.com/DerangedIons/Lightning.jl")
```
