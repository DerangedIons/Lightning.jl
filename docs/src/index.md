```@meta
CurrentModule = Lightning
```

# Lightning.jl

Cardiac tissue electrophysiology on orthogonal, structured grids.

Lightning solves the monodomain equation

```math
\begin{aligned}
\chi C_\mathrm{m} \partial_t \varphi_\mathrm{m}
    &= \nabla\cdot\kappa\nabla\varphi_\mathrm{m}
     + \chi\left(I_\mathrm{ion}(\varphi_\mathrm{m}, s, t) + I_\mathrm{stim}(x, t)\right) \\
\partial_t s &= f(\varphi_\mathrm{m}, s, t)
\end{aligned}
```

by reaction-diffusion operator splitting. [MatrixFreeOperators.jl](https://github.com/DerangedIons/MatrixFreeOperators.jl)
supplies the spatial operators, [CytoZoo.jl](https://github.com/DerangedIons/CytoZoo.jl) the
cell kinetics, and [OrdinaryDiffEqOperatorSplitting.jl](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl)
the splitting.

Restricting the domain to orthogonal structured grids is the design, not a limitation waiting
to be lifted: it is what makes the spatial operators matrix-free and what lets the same source
run on a GPU. Unstructured meshes are [Thunderbolt.jl](https://github.com/JuliaHealth/Thunderbolt.jl)'s
job. Lightning's pipeline is nevertheless congruent with Thunderbolt's by convention — same
names, same shapes — while depending on none of it.

## The pipeline

Four objects, in order.

| Step | Type | What it says |
|---|---|---|
| 1 | [`MonodomainModel`](@ref) | the continuous problem: `κ`, `χ`, `Cₘ`, the cell model, the stimulus |
| 2 | [`ReactionDiffusionSplit`](@ref) | how to split it, and any spatial parameter heterogeneity |
| 3 | [`FiniteDifferenceDiscretization`](@ref) + a `CartesianGrid` | where to solve it |
| 4 | [`semidiscretize`](@ref) | assembles a `GenericSplitFunction` ready for `OperatorSplittingProblem` |

```julia
using Lightning
using CytoZoo: ParametrizedFHNModel
using OrdinaryDiffEqLowOrderRK: Euler

grid = CartesianGrid(((0.0, 20.0),), (400,); bc = ((Neumann(), Neumann()),))

stim = TransmembraneStimulationProtocol(
    (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 0.5 : 0.0;
    nonzero_intervals = ((0.0, 2.0),),
)

model = MonodomainModel(; κ = 0.1, ion = ParametrizedFHNModel(), stim)
f = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)

prob = OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 400.0))
integrator = init(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005)

while integrator.t < 400.0
    step!(integrator, 0.005, true)
    φ = getvariable(integrator.u, f, :φₘ)
end
```

`examples/cable_1d.jl` is this with reporting and a figure attached.

## The solution vector

State-blocked (structure of arrays), matching Thunderbolt's `StateBlockedLayout`:

```
u = [φₘ(1:N); s₁(1:N); s₂(1:N); … ; s_{M-1}(1:N)]
```

Every state is contiguous, so the diffusion half acts on the leading block `1:N` — no gather,
no synchronizer object, and coalesced access on a GPU. Reach into it by name with
[`getvariable`](@ref), [`setvariable!`](@ref), and [`variable_range`](@ref) rather than by
index.

## Things that will bite you

- **The stimulus sign is inverted relative to a cell model.** A protocol returns `Iₐₚₚ` in
  PDE form, where positive depolarizes; a CytoZoo cell model's internal stimulus is a
  membrane current, where negative depolarizes. See [`AbstractStimulationProtocol`](@ref).
  Stimulation belongs to the diffusion half — the cell model's own stimulus must be off, and
  [`MonodomainModel`](@ref) warns when it can tell that it is not.
- **The grid is cell-centered.** `n` cells across a length `L` put the first node at `Δx/2`,
  not at `0`. Comparisons against a vertex-centered reference differ at `O(Δx)`.
- **Only zero-flux boundaries.** Anything else raises an `ArgumentError` naming the face.
- **One semidiscretization per concurrent solve.** The prepared operator inside the diffusion
  half owns scratch buffers and is stateful and single-threaded.
- **The splitting integrator has no SciML solution interface.** There is no `sol` to hand
  back; step it, or sample it with `SciMLIterators.TimeChoiceIterator`.
- **Use `Float64`.** `Float32` runs, but the ArmyHeart reference measured non-finite results
  across 58.6% of the ToRORd parameter space in single precision.

## GPU

Put the backend on the grid — `CartesianGrid(…; device = CUDABackend())` — and field
allocation, the prepared operator's scratch, [`node_coordinates`](@ref), and
[`create_initial_condition`](@ref) all follow. The reaction half then runs as one
KernelAbstractions kernel over the nodes, which requires the cell model to be isbits.

## API

```@index
```

```@autodocs
Modules = [Lightning]
```
