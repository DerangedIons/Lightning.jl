---
slug: lightning-monodomain-v0
created: 2026-08-13-2019
status: open
---

# Handoff: Implement the monodomain model in Lightning.jl (first physics, v0 core)

## Goal / why this matters

Lightning.jl is the ArmyHeart spin-out's tissue framework (see `~/dev/ArmyHeart/SPINOUT_PLAN.md`, scope sheet 4): a Thunderbolt-like cardiac multiphysics framework deliberately restricted to orthogonal/structured domains so that MatrixFreeOperators.jl supplies the spatial operators. This handoff implements its first physics — the monodomain equation — with the types and abstractions the rest of the framework will grow on. The design below was explored (4 parallel codebase surveys) and approved by Kyle on 2026-08-13; do not relitigate the settled decisions.

## Settled decisions (approved by Kyle — do not reopen)

1. **Thunderbolt-congruent pipeline API.** Lightning owns its own types with Thunderbolt's names and shapes (`MonodomainModel` → `ReactionDiffusionSplit` → `semidiscretize(split, disc, grid)` → `GenericSplitFunction` → `OperatorSplittingProblem`). Zero dependency on Thunderbolt — congruency is by convention, never by import (SPINOUT_PLAN forbids the dep).
2. **State-blocked (SoA) layout**: `u = [V(1:N); s₁(1:N); …; s_{M-1}(1:N)]`, matching Thunderbolt main's `StateBlockedLayout`. Diffusion dofrange is contiguous `1:N`; no synchronizer objects needed; GPU-coalesced.
3. **Dimension-generic types, 1D validated.** Grid/operators are N-dimensional via MatrixFreeOperators; this milestone's tests focus on 1D cable (+ 2D analytic diffusion). No 3D oracle yet.
4. **Single problem only.** No LockstepODE dep, no multi-cable batching in this milestone — but the types must let a batched layer compose on top later without breaking changes.
5. **Core only.** No `prepace`, `conduction_velocity`, or `pseudo_ecg` in this milestone; they are separate follow-ups.
6. **CUDA support in this milestone**: array-generic code + `Adapt` throughout, KernelAbstractions reaction kernel, GPU tests gated on `CUDA.functional()`. No self-hosted GPU CI yet.
7. **The lightweight FHN test model goes into CytoZoo via a small PR** (the cell-model zoo is its home; Thunderbolt ships `FHNModel` too). Lightning's tests/examples then consume it as an ordinary CytoZoo model.
8. Stimulus enters the **diffusion half as a source term, uniformly in every dimension** — deliberately killing ArmyHeart's wart where 1D stimulates via a cell-model parameter but 2D stimulates inside the diffusion kernel.

## Background & current state

- **Lightning.jl** (`~/dev/Lightning`, = the iCloud `pro/dev/Lightning`, this repo) is untouched PkgTemplates output: empty `src/Lightning.jl`, zero `[deps]`, `julia = "1.10"` compat, test suite = plain `Test` + a JET lint testset, Documenter docs (with a `doctest` CI step), blue JuliaFormatter style, CI matrix Julia 1.10/1.11/1.12 on ubuntu x64. Remote `git@github.com:DerangedIons/Lightning.jl.git`.
- **MatrixFreeOperators.jl** (`~/dev/MatrixFreeOperators`, unregistered v0.1.0, API explicitly unstable) is the spatial workhorse and *already runs monodomain*: `examples/niederer_benchmark.jl` (3D Niederer benchmark, ten Tusscher kinetics in an `SVector{18}` field, anisotropic axis-aligned D, zero-flux) and `examples/monodomain_amr.jl` (2D Aliev–Panfilov spiral on AMR). Read both before writing code — they are the house style for using MFO.
- **CytoZoo.jl** (`~/dev/CytoZoo`, unregistered 0.0.1, zero runtime deps) supplies the cell-model interface. Ships `ToRORd` (65 states). Strictly pointwise — the N-node story is Lightning's.
- **OrdinaryDiffEqOperatorSplitting** supplies the splitting layer. Depend on **registered v0.4.0** (SPINOUT 0.5e); the local checkout is a dirty 0.3.2 — do not `[sources]` it.
- **ArmyHeart** (`~/dev/ArmyHeart`) is the frozen reference. Its warts, all deliberately fixed here: five overlapping solve entry points (`src/solve.jl` even calls a deleted function), two coexisting grid types (`LazyGrid` + `StructuredGrid`), the 1D/2D stimulus asymmetry, scalar-only κ, and an AoS layout that forced a custom `VoltageExtractSync`.

### Key API facts from exploration (verified 2026-08-13, with file refs)

**MatrixFreeOperators** (`~/dev/MatrixFreeOperators/src/`):
- `CartesianGrid(extent::NTuple{N,Tuple{T,T}}, ncells::NTuple{N,Int}; bc, halo, device)` — `Grids.jl:42,53`. **Cell-centered**, uniform; spacing derived; carries BCs and device. Zero-flux = `bc = ntuple(_ -> (Neumann(), Neumann()), N)`, and then `boundary_rhs` is identically zero (nothing to fold into the RHS).
- Operators: `laplacian(g)` (`operators/laplacian.jl:71`), `derivative(g, dim; order=2)` (`operators/derivative.jl:53`), `scaling(κ)`, full operator algebra (`+`, `*`, scalar `*`). Anisotropic axis-aligned diffusion is the Niederer-example idiom: `D_T*laplacian(g) + (D_L - D_T)*derivative(g, 1; order=2)` (`examples/niederer_benchmark.jl:138-147`).
- ODE seam: `P = prepare(L, scalar_field(g))` (`linalg.jl:91`) then `mul!(du, P, u)` on **flat interior vectors** (no ghost storage in `u`; ghosts live in P's scratch fields). Zero steady-state allocations. **`P` is stateful and single-threaded — one per concurrent solve.** `mul!` on an unprepared operator deliberately throws.
- Fields/coords: `scalar_field(g)`, `set!(f, fun)` (fun of `SVector` cell center), `flatten`, `flat_to_interior!`/`interior_to_flat!`, `cell_center(g, I)`, `interior(g)`, `spacing(g)`, `local_size(g)` (`Fields.jl`, `Grids.jl:89-206`).
- Device: array-generic broadcasts + `Adapt.adapt_structure` through the whole operator tree; backend inferred per array via `KernelAbstractions.get_backend`. CUDA needs no MFO extension for normal execution.
- Name clashes: MFO exports `solve` (multigrid) and `gradient` — use **explicit imports** in Lightning, never bare `using MatrixFreeOperators`.

**CytoZoo** (`~/dev/CytoZoo/src/`):
- The model **is** the SciML RHS: `(model)(du, u, p, t)` with `p::Nothing` (non-spatial, spatial branches compile away) or `p::SpatialContext` (`models/torord/ToRORd.jl:68,73`). `SpatialContext{X,SF}(x, overrides)` (`interface.jl:197`) — `overrides` is a NamedTuple keyed by parameter name; values are scalars, callables `(x,t)->v`, or isbits `SpatialFunction` functors (`Constant`, `SpatialStep`, `SpatialGradient` in `spatial.jl`).
- Interface: `num_states`, `default_initial_state`, `state_names`, `transmembrane_potential_index`, `parameter_names`/`parameter_index`/`state_index`, optional `has_rush_larsen`/`rush_larsen_step!(u_new, u, p, t, dt, model)` and monitors (`interface.jl`). Hard invariant: **the RHS writes every `du` slot, never `+=` into an unassigned one**.
- Stimulus is a **model field**, callable `(x, t)` returning the full `Istim` (`stimulus.jl:19,54`; enters ToRORd at `rhs.jl:477`). ToRORd only resolves these overrides: `:IKr_Multiplier`, `:T`, `:isHypoxic`, `:celltype`, `:pH`.
- ToRORd gotchas: Rosenbrock solvers (`Rodas5P` etc.) MethodError on it (use non-Rosenbrock or Rush-Larsen); `state_index`/`parameter_index` throw `KeyError` instead of returning `nothing` (known bug); `default_initial_state` is not pre-paced (dV/dt ≈ +240 mV/ms at t=0); parameters are a heap `Vector` (not isbits — the GPU risk below).

**OrdinaryDiffEqOperatorSplitting** (registered 0.4.0; local `~/dev/OrdinaryDiffEqOperatorSplitting` for reference):
- `GenericSplitFunction(fs::Tuple, dofranges::Tuple)` (`src/function.jl:7,43`) — the 2-arg form fills `NoExternalSynchronization()` per operator. `OperatorSplittingProblem(f, u0, tspan)` (`src/problem.jl:5`). Algorithms `LieTrotterGodunov(inner_algs::Tuple)` and `StrangMarchuk(inner_algs::Tuple)` (`src/solver.jl:14,87`); the alg tuple mirrors the function tuple.
- Each substep: forward-sync → advance → backward-sync over `@view u[dofrange]`; **the copy is skipped when the child view aliases the parent** (`src/utils.jl:35-50`) — which our contiguous SoA views do.
- The integrator does **not** implement the SciML solution interface: drive output with `for (u, t) in TimeChoiceIterator(integrator, ts)`. Inner algs are ordinary OrdinaryDiffEq algorithms (`Euler()` from OrdinaryDiffEqLowOrderRK).

**Thunderbolt congruency targets** (github.com/JuliaHealth/Thunderbolt.jl; no local checkout — read via GitHub if needed):
- `MonodomainModel(χ, Cₘ, κ, stim, ion, φsym, ssym)` for `χCₘ∂ₜφₘ = ∇⋅κ∇φₘ + χ(Iᵢₒₙ + Iₛₜᵢₘ)`; `ReactionDiffusionSplit(model[, cs])` is a pure annotation; `semidiscretize(split, discretization, mesh)` returns a `GenericSplitFunction`; inside, κ is folded to `κ/(Cₘχ)` via `ConductivityToDiffusivityCoefficient`.
- Thunderbolt `main`'s named solution API is the congruency target for our helpers: `create_initial_condition`, `setvariable!`, `getvariable`, `solution_size`, and explicit `StateBlockedLayout`.
- Their tutorial calls `MonodomainModel(Cₘ, χ, …)` while the fields are `(χ, Cₘ, …)` — the positional-order trap is why Lightning's constructor is keyword-based.

## Design spec (approved)

### Files

```
src/Lightning.jl        module: explicit imports, includes, exports; re-export the mesh
                        vocabulary (CartesianGrid, Neumann, Dirichlet, Periodic, cell_center,
                        spacing, …) so `using Lightning` is one-stop for users
src/models.jl           AbstractEPModel, MonodomainModel, ReactionDiffusionSplit
src/stimulus.jl         AbstractStimulationProtocol, NoStimulationProtocol,
                        TransmembraneStimulationProtocol
src/discretization.jl   FiniteDifferenceDiscretization
src/reaction.jl         PointwiseODEFunction-style reaction functor + KA kernel
src/semidiscretize.jl   semidiscretize methods + diffusion RHS builder
src/solution.jl         solution_size, create_initial_condition, setvariable!, getvariable
```

### Types

- `MonodomainModel` — fields `χ`, `Cₘ`, `κ`, `stim`, `ion` (+ `φ_symbol::Symbol = :φₘ`, `state_symbol::Symbol = :s`), all type-parameterized (per julia-coding rules: parameterize every field, no `Any`). Keyword constructor with sensible defaults (`χ = 1`, `Cₘ = 1`, `stim = NoStimulationProtocol()`). `ion` is any `CytoZoo.AbstractCellModel`. `κ` accepts a `Number` (isotropic) or an `NTuple{N}`/`SVector{N}` (axis-aligned diagonal anisotropy).
- `ReactionDiffusionSplit{M, O}` — `ReactionDiffusionSplit(model)` or `ReactionDiffusionSplit(model, overrides)` where `overrides` is the CytoZoo `SpatialContext` NamedTuple applied per node (this is how spatial heterogeneity — celltype, pH, hypoxia — enters; congruent with Thunderbolt's `ReactionDiffusionSplit(model, cs)` slot).
- `FiniteDifferenceDiscretization(; order = 2)` — minimal descriptor; validates `order == 2` (only supported); exists to keep the congruent 3-arg `semidiscretize` signature and leave room for 4th order. Spacing/BCs live on the grid — do not duplicate them here (that duplication was an ArmyHeart wart: its `order`/`boundary` fields were validated but never read).
- Stimulation: `abstract type AbstractStimulationProtocol`, `NoStimulationProtocol`, and `TransmembraneStimulationProtocol{F, I}(f; nonzero_intervals = nothing)` where `f(x, t) -> I_app`. Callable `(p::…)(x, t)`. `nonzero_intervals` (vector of `(t₀, t₁)` tuples) lets the diffusion RHS skip evaluation when t is outside all windows.

### `semidiscretize(split::ReactionDiffusionSplit{<:MonodomainModel}, disc::FiniteDifferenceDiscretization, grid::CartesianGrid{N})`

1. Validate zero-flux: every grid BC is `Neumann` (v0 supports only zero-flux; error otherwise with a clear message).
2. Build the diffusion operator: `κ_eff = κ ./ (χ * Cₘ)`; scalar κ → `κ_eff * laplacian(grid)`; per-axis κ → `κ_min*laplacian(grid) + Σ_d (κ_d - κ_min)*derivative(grid, d; order=2)` (collapses to the Niederer idiom). `P = prepare(op, scalar_field(grid, T))`.
3. Precompute node coordinates once: `xs = [cell_center(grid, I) for I in interior(grid)]` flattened in the same order `flatten` uses (verify ordering against `interior_to_flat!`).
4. Diffusion RHS functor (fields: `P`, `stim`, `Cₘ`, `xs`): `(du, u, p, t) -> begin mul!(du, P, u); apply_stimulus!(du, stim, xs, Cₘ, t); end`. Stimulus applied as `du .+= stim.(xs, t) ./ Cₘ` (skip entirely when `nonzero_intervals` says quiet, or when `stim isa NoStimulationProtocol` — dispatch, don't branch).
5. Reaction functor (fields: `ion`, `xs`, `overrides`, `nnodes`, `nstates`): per node `i`, `u_i = @view u[i:N:N*M]`, `p_i = SpatialContext(xs[i], overrides)`, call `ion(du_i, u_i, p_i, t)` (pass `p_i = nothing` when there are no overrides AND the stimulus needs no position — keep the fast non-spatial dispatch available). CPU: threaded loop over nodes (`Threads.@threads` or KA CPU backend — implementer's choice, but dispatch on `KernelAbstractions.get_backend(u)`); GPU: one KA kernel, `ndrange = nnodes`.
6. Return `GenericSplitFunction((ODEFunction(diffusion_f), ODEFunction(reaction_f)), (1:nnodes, 1:nnodes*nstates))` — 2-arg form, no synchronizers (SoA overlap makes them unnecessary; the `1:nnodes` SubArray aliases the parent so OS skips the copy).

### Solution helpers (`src/solution.jl`)

- `solution_size(f)` — `nnodes * nstates`.
- `create_initial_condition(f, ::Type{T} = Float64)` — allocate `Vector{T}(undef, size)`; for each state k, fill block `((k-1)nnodes+1):(k*nnodes)` with `default_initial_state(ion)[k]`.
- `setvariable!(u, f, name::Symbol) do x … end` — resolve `name` against `state_names(ion)` (with `:φₘ`/`model.φ_symbol` aliasing the voltage index), evaluate the function at each node's `cell_center`, write the block.
- `getvariable(u, f, name)` — the corresponding contiguous `@view`.

### Sign convention (define once, pin with a test)

The protocol returns `I_app` in the **PDE-form convention: positive = depolarizing** (matches Thunderbolt's `χCₘ∂ₜφₘ = … + χIₛₜᵢₘ` and ArmyHeart's 2D kernel). This is the opposite sign from the cell-model-internal `i_Stim_Amplitude = -53 µA/µF` convention (negative-inward) used in ArmyHeart's 1D path — document that difference loudly in the docstring. The cell model's own stimulus must be off: document it, and add a best-effort check (if `hasproperty(ion, :stim)` and it is a nonzero `CytoZoo.Stimulus`, warn).

### CUDA / device story

- Everything array-generic; `Adapt.adapt_structure` for every Lightning struct that holds arrays (reaction functor's `xs`, diffusion functor via MFO's own adapt rules). Workflow: build on CPU, `adapt(CuArray, …)` the split function and `u₀`, or construct the grid with `device = CUDABackend()`.
- **Known risk**: CytoZoo models hold `Vector` parameters (not isbits) and CytoZoo has zero deps (no Adapt). If ToRORd won't ride into a kernel, (a) validate the GPU path with the isbits FHN model, and (b) file/land a small `CytoZooAdaptExt` PR (Adapt as weakdep) adapting model parameter vectors. Do not block the CPU milestone on this.
- Verify the strided `@view u[i:N:N*M]` compiles inside a CUDA KA kernel; if it fights, fall back to manual index arithmetic in the kernel body.
- Float64 is mandatory for production use (ArmyHeart `PRECISION.md`: Float32 → non-finite results in 58.6% of parameter space). Keep types generic, default `T = Float64`, and state the constraint in docs.

### Dependencies

- `[deps]` via `Pkg.add`/`Pkg.develop` (never hand-edit): MatrixFreeOperators (`Pkg.develop(path = joinpath(homedir(), "dev", "MatrixFreeOperators"))`), CytoZoo (develop likewise), OrdinaryDiffEqOperatorSplitting (`= "0.4"`), SciMLBase, KernelAbstractions, Adapt, StaticArrays (SVector coords).
- Test-only (`[extras]`/`[targets]`, matching the template's existing convention): OrdinaryDiffEqLowOrderRK, CUDA, JET, Test.
- Compat entries for every dep + `julia = "1.10"`. `[sources]` stays empty for now (dev deps live in the manifest); before anything is public, MFO/CytoZoo become tag-pinned `[sources]` entries per SPINOUT 0.5g/0.5c.
- Never: Thunderbolt, LockstepODE (yet), RushLarsenSolvers (yet). The Rush-Larsen seam is already right: RushLarsenSolvers implements DiffEqBase algorithms, so it later slots into the `LieTrotterGodunov((diff_alg, cell_alg))` tuple without type changes.

## What's left / next steps (ordered)

1. **CytoZoo PR: add an FHN model.** `ParametrizedFHNModel{T}`-style (mirror Thunderbolt's `src/modeling/cells/fhn.jl` equations; 2 states, isbits params, `stim` field per CytoZoo convention), implementing the full Tier-1/2 interface with both `p::Nothing` and `p::SpatialContext` functor methods. Include tests mirroring CytoZoo's existing model-test shape (incl. the zero-allocation assertion). Small, reviewable PR to `DerangedIons/CytoZoo` main; Lightning devs against the branch/main.
2. **Lightning dependency setup** as above (remember `ENV["JULIA_PKG_PRECOMPILE_AUTO"] = "0"` before Pkg ops).
3. **Types**: `models.jl`, `stimulus.jl`, `discretization.jl`.
4. **Reaction functor + kernel** (`reaction.jl`).
5. **`semidiscretize` + diffusion RHS** (`semidiscretize.jl`).
6. **Solution helpers** (`solution.jl`).
7. **Adapt rules** for Lightning structs.
8. **Tests** (write, do NOT run — Kyle runs tests himself; make output verbose enough to debug from):
   - `test/test_diffusion.jl` — analytic oracle through the *full pipeline*: a test-local `NullIonicModel` (1 state, `du[1] = 0`) collapses monodomain to pure diffusion; IC `cos(πx/L)` → `cos(πx/L)·exp(-κ_eff(π/L)²t)` under zero-flux Neumann, 1D and 2D, tolerance ~1e-4 (ArmyHeart's `test/monodomain_fdm.jl` is the template — but note the cell-centered coordinate difference below).
   - `test/test_pipeline.jl` — SoA layout/dofranges, `create_initial_condition`/`setvariable!`/`getvariable`, stimulus sign convention (stimulated node depolarizes), `NoStimulationProtocol` fast path, anisotropic κ operator assembly, error on non-Neumann BCs, `@inferred` on both RHS halves.
   - `test/test_cable.jl` — FHN 1D cable: stimulus captures, wave propagates left→right, activation times monotone in x, mirror-symmetric under mirrored stimulus.
   - `test/test_gpu.jl` — gated on `CUDA.functional()`: adapt the problem, one step, results ≈ CPU.
   - Keep the existing JET testset green.
9. **Example + docs**: `examples/cable_1d.jl` (the approved-pipeline snippet, FHN); rewrite the stock README (template boilerplate is explicitly called out in SPINOUT_PLAN as the thing not to repeat); minimal Documenter index listing the pipeline. Docs CI runs `doctest(Lightning)` — keep docstring examples doctest-valid or unfenced.
10. Commit in atomic conventional commits (`feat(models): …`, `feat(semidiscretize): …`, `test(cable): …`). Do not push/PR without Kyle's go-ahead.

## Gotchas / constraints

- **MFO is cell-centered; ArmyHeart was vertex-centered.** N cells over length L → spacing L/N, first node at Δx/2. Activation/CV comparisons against ArmyHeart oracles will differ at O(Δx); per the SPINOUT validation contract this is a named, documented numerical difference, not a bug. The analytic-decay test is exact for cell-centered Neumann as long as you evaluate the IC at `cell_center` coordinates.
- **`prepare` objects are stateful and single-threaded** — one per solve; never share across concurrent solves; `mul!` on an unprepared operator throws by design.
- **Do not `using MatrixFreeOperators` wholesale** — its `solve` and `gradient` exports clash; import explicitly (house rule anyway: explicit imports for consumed names).
- **OS integrator has no SciML solution interface** — output via `TimeChoiceIterator`; don't pass `saveat` expecting a `sol` object.
- **CytoZoo RHS invariant**: every `du` slot written, never `+=` — matters if anyone is tempted to accumulate stimulus into the reaction half (another reason stimulus lives in the diffusion half).
- **ToRORd + Rosenbrock = MethodError** (element-type constraint); inner algs should be `Euler()`/explicit or (later) Rush-Larsen.
- The registered OrdinaryDiffEqOperatorSplitting 0.4.0 API was cross-checked against the local 0.3.2 checkout (`GenericSplitFunction`, `OperatorSplittingProblem`, `LieTrotterGodunov`, `StrangMarchuk` all present) — but verify the 0.4.0 release notes for signature drift before coding against it.
- **JuliaFormatter blue style**; JET lint testset must stay green; template CI is Linux/x64/CPU-only (GPU tests run locally only for now).
- Kyle's standing rules apply: tests written for every new function but **never run unless he says so**; struct fields always type-parameterized; no `::Any`; integer literals where values are integers; `T(...)`-wrap float literals in generic code.
- ArmyHeart reference files, should the implementer need them: `src/electrophysiology/fdm/monodomain.jl` (the semidiscretize precedent, incl. the SoA-vs-AoS contrast), `fdm/kernels.jl` (boundary stencils), `test/monodomain_fdm.jl` (analytic test), `examples/fdm_1d.jl` (OS-path example). MFO reference: `examples/niederer_benchmark.jl`, `examples/monodomain_amr.jl`, `test/ode_rhs.jl`, `DESIGN.md` §4a (custom fused operators — the sanctioned route if a fused monodomain kernel is ever needed for perf).
- Deferred by decision (do not scope-creep into this milestone): Rush-Larsen path, LockstepODE batching, prepace/CV/pECG, spatially-varying κ fields (`divergence∘scaling∘gradient` composition exists in MFO when needed), Strang-specific validation, AMR, multi-GPU, registration/tagging.
