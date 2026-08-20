# Lightning.jl — Design Specification

**Status:** draft for review · **Date:** 2026-08-19

This document specifies the *target* design of Lightning at the level of abstractions and type hierarchy. It deliberately stops above implementation detail — keyword surfaces, tolerance defaults, migration sequencing — which get filled in once this shape is settled; the one field-level artifact it carries is §4's mocked-up definitions, indicative sketches of what each concrete type owns, explicitly non-final, each placed with the abstraction it realizes. The v0 design was approved 2026-08-13 and is implemented; its decisions (`handoffs/2026-08-13-2019-lightning-monodomain-v0.md`) are cited here as settled, never reopened. The live design material is the v0.x growth path named by the 2026-08-18 architecture review (`handoffs/2026-08-18-1143-julia-tuneup-architecture-review.md`, items A1–A4), which this doc resolves; §9 traces each spec element to the findings it answers. Studied inspirations: **Thunderbolt.jl** (the congruency target — pipeline names, `semidiscretize`, and literally `StateBlockedLayout`), **Capillarium.jl's DESIGN.md** (the in-org sibling spec, whose ecosystem-alignment section explicitly requests the two abstractions this doc adds), and **Oceananigans.jl** (studied for its `Simulation`/callback school — adopted only as the "the loop needs an owner" lesson, deliberately not as a framework).

## 1. What Lightning is

Lightning is the tissue layer of a three-package stack: it solves cardiac electrophysiology PDEs on orthogonal structured grids, with MatrixFreeOperators.jl supplying spatial operators, CytoZoo.jl supplying pointwise cell kinetics, and OrdinaryDiffEqOperatorSplitting supplying time splitting. Its entire value is (1) the structured-grid restriction, which buys matrix-free operators and single-source GPU execution, and (2) Thunderbolt-congruent vocabulary with zero Thunderbolt dependency, so model descriptions read the same in either package. The central claim of this doc: the package is deliberately ~800 lines of glue, naming, and layout — and the two abstractions it still owes itself are a first-class owner for that layout and an owner for the time-stepping loop. Everything else stays thin.

## 2. Design principles

1. **Congruent by convention, never by import.** Lightning owns types with Thunderbolt's names and shapes; congruency survives Thunderbolt's 0.0.x churn precisely because there is no dependency.
2. **Declare, then semidiscretize.** A model is an inert declaration of the continuous problem; geometry and numerics arrive together at one compiler-like verb, so the same model runs on any grid.
3. **Hard delegation boundaries.** Cell kinetics belong to CytoZoo, spatial stencils to MatrixFreeOperators, time splitting to OrdinaryDiffEqOperatorSplitting; Lightning writes none of them, and a feature that would require it to is a feature for the neighbor package.
4. **The layout has exactly one owner.** The state-blocked SoA layout is the package's central invariant; every piece of index arithmetic lives in one type, so a future batched layout is a new type, not a hunt through five files.
5. **Physics choices are typed objects in named slots, validated at construction.** Stimulus protocols, conductivity forms, and spatial overrides dispatch on types — never boolean flags — and validation lives in inner constructors so no path can bypass it.
6. **Device is a property of the data.** Everything is array-generic with `Adapt` rules; build on CPU, `adapt` to the device, and no Lightning type mentions CUDA.
7. **The numbers are the contract.** Analytic decay oracles, the frozen cable conduction-velocity golden master (captured from unmodified code before any change), 0 B/call RHS kernels, and full inference are acceptance criteria for every step toward this design.
8. **Public names never depend on upstream private shape.** Exports run the model; internals are namespaced; and any reach into another package's field layout happens in exactly one sanctioned accessor.

## 3. The grammar

```
MonodomainModel  →  ReactionDiffusionSplit  →  semidiscretize(split, disc, grid)  →  OperatorSplittingProblem  →  init / step!  →  foreach_step + recorders
  (continuous       (splitting annotation        (GenericSplitFunction:                (OrdinaryDiffEq-               (SciML)         (observation layer,
   declaration)      + spatial overrides)         two RHS halves + layout)              OperatorSplitting)                             v0.x — owns the loop)
```

The canonical programs — these are the docs quickstarts and the acceptance examples. First, the 1D FHN cable, the program the observation layer improves most:

```julia
grid  = CartesianGrid(((0.0, 20.0),), (200,); bc = ((Neumann(), Neumann()),))
model = MonodomainModel(; κ = 1.0e-3, ion = FHNModel(),
    stim = TransmembraneStimulationProtocol((x, t) -> (t ≤ 2.0 && x[1] ≤ 1.0) ? 50.0 : 0.0;
                                            nonzero_intervals = ((0.0, 2.0),)))
f  = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)
u₀ = create_initial_condition(f)
integrator = init(OperatorSplittingProblem(f, u₀, (0.0, 100.0)),
                  LieTrotterGodunov((Euler(), Euler())); dt = 0.01)

rec = ActivationRecorder(f; threshold = 0.5)          # v0.x: replaces three hand-rolled copies
foreach_step(integrator, 100.0) do u, t               # v0.x: owns the `t < tend - dt/2` loop
    record!(rec, u, t)
end
activation_times(rec)                                 # per-node first-crossing times; NaN = never
```

The 3D Niederer benchmark — anisotropic conductivity, a many-state ionic model (from CytoZoo, post-promotion), the community validation case:

```julia
grid  = CartesianGrid(((0.0, 20.0), (0.0, 7.0), (0.0, 3.0)), (100, 35, 15);
                      bc = ntuple(_ -> (Neumann(), Neumann()), 3))
model = MonodomainModel(; κ = (0.133, 0.0176, 0.0176), χ = 140.0, Cₘ = 0.01,
                          ion = TenTusscher2006(), stim = corner_stimulus)
```

The 2D spiral — S1–S2 reentry induction is nothing but a stimulus protocol with two windows:

```julia
stim = TransmembraneStimulationProtocol(s1s2; nonzero_intervals = ((0.0, 2.0), (335.0, 337.0)))
model = MonodomainModel(; κ = 1.0e-3, ion = AlievPanfilov(), stim)
```

And the single-source GPU claim as a program — the pipeline above, moved, with nothing rewritten:

```julia
f_gpu  = adapt(CuArray, f)
u₀_gpu = create_initial_condition(f_gpu)              # allocated on the device the operator lives on
```

## 4. The type system

```
AbstractEPModel                                # inert declaration of the continuous problem
└── MonodomainModel{Tχ,TC,Tκ,TS,TI}            #   χ, Cₘ, κ (scalar | per-axis), stim, ion + naming symbols

AbstractStimulationProtocol                    # callable (x, t) → Iₐₚₚ; positive = depolarizing (PDE convention)
├── NoStimulationProtocol                      #   dispatch deletes the evaluation — a strong zero, not ×0
└── TransmembraneStimulationProtocol{F,I}      #   f + nonzero-interval windows; isbits exactly when f is

ReactionDiffusionSplit{M,O}                    # split annotation; O = spatial-override NamedTuple (heterogeneity)
FiniteDifferenceDiscretization                 # inert numerics descriptor; order == 2 in v0

StateBlockedLayout                             # NEW (A1) — isbits owner of N, M, the symbols, all block/stride math
DiffusionFunction{…}                           # internal RHS half: prepared MFO operator + stimulus, on block 1:N
PointwiseODEFunction{…}                        # internal RHS half: cell model per node; carries the layout as one field

foreach_step(g, integrator, tstop; …)          # NEW (A2) — the loop owner; a function, deliberately not a type
ActivationRecorder{T}                          # NEW (A2) — first threshold-crossing times, indexed via the layout
```

The mockups below each why-paragraph are indicative sketches, not frozen field inventories: each fixes what the type *owns*; constructors, validation, and defaults stay in the implementation (existing types show their live v0 fields, the two NEW types the shape the prose commits to).

**`MonodomainModel`** — a small closed component-slot struct: every physics choice is a typed field, every field type-parameterized, construction keyword-only (Thunderbolt's own tutorial has the positional χ/Cₘ swap trap this kills). It declares the continuous problem and nothing else; the settled v0 shape, kept. The one v0.x change is F2: the positivity/symbol validation moves into inner constructors so the auto-generated positional constructors cannot bypass it.

```julia
struct MonodomainModel{Tχ,TC,Tκ,TS<:AbstractStimulationProtocol,TI} <: AbstractEPModel
    χ::Tχ                        # surface-to-volume ratio
    Cₘ::TC                       # membrane capacitance per unit area
    κ::Tκ                        # Number = isotropic | NTuple/SVector = axis-aligned diagonal
    stim::TS
    ion::TI                      # any CytoZoo.AbstractCellModel — it *is* the reaction half
    φ_symbol::Symbol             # solution-helper name for the voltage block
    state_symbol::Symbol         # …and for the whole non-voltage block (plural on purpose)
end
```

**`ReactionDiffusionSplit`** — the pure annotation that says *how* to split, and the slot where spatial heterogeneity (`overrides`) enters, congruent with Thunderbolt's slot. Noted seam, not acted on: heterogeneity is conceptually a property of the tissue, not of the splitting annotation, so if an unsplit/IMEX path ever appears the overrides move onto the model; `semidiscretize`'s internals stay ignorant of where overrides came from to keep that move cheap.

```julia
struct ReactionDiffusionSplit{M<:AbstractEPModel,O}
    model::M
    overrides::O                 # NamedTuple of spatial overrides | nothing = fast non-spatial dispatch
end
```

**`StateBlockedLayout`** — the new v0.x centerpiece. Today the layout `u = [φₘ(1:N); s₁(1:N); …]` exists only as index arithmetic scattered across five sites, four metadata fields smuggled onto the reaction functor, and public helpers that reach through `f.functions[2].f` — upstream's *private* field layout, so an OS field rename would be a Lightning breaking change (Hyrum's law pointed inward). The move: an isbits value type owning N, M, the two naming symbols, and every block/stride computation (`variable_range`, state blocks, node slices); `layout(f)` extracts it as the single sanctioned reach-through; every solution helper and both kernels dispatch on it. This is *more* Thunderbolt-congruent (they name this exact type), it gives the F1 voltage-index contract a home (the guard that voltage is state 1 lives where the layout is born, and lifting the restriction later is a layout change, not a hunt), and it is precisely the seam the settled batching decision needs — a batched layer becomes a second layout type honoring the same accessor contract, changing nothing else.

```julia
struct StateBlockedLayout        # isbits; construction guards transmembrane_potential_index(ion) == 1 (F1)
    nnodes::Int                  # N
    nstates::Int                 # M
    φ_symbol::Symbol
    state_symbol::Symbol
end
# owns every block/stride computation as methods (variable ranges, state blocks, node slices);
# layout(f) is the single sanctioned reach into the GenericSplitFunction internals
```

**`foreach_step` + `ActivationRecorder`** — the observation layer, kept deliberately dumb. The `while integrator.t < tend - dt/2; step!(…)` loop is currently hand-rolled in six places and first-crossing activation recording reimplemented in three; the most-duplicated code in the repo is the package's very first README snippet. `foreach_step` owns the loop and its floating-point half-step guard and yields `(u, t)` after each step; `ActivationRecorder` is a typed recorder that watches the voltage block through the layout and stores first upward threshold crossings. No abstract observer hierarchy, no scheduling framework, no SciML solution interface pretensions (the no-`sol` decision stands) — the deferred conduction-velocity and pseudo-ECG readouts become ordinary consumers of the recorded times when they arrive.

```julia
foreach_step(g, integrator, tstop)   # a function, deliberately not a type: owns the `t < tend − dt/2` loop, yields (u, t)

struct ActivationRecorder{T,A<:AbstractVector{T}}   # immutable-with-buffer shape per §8 Q3 recommendation
    layout::StateBlockedLayout
    threshold::T
    times::A                     # per-node first upward crossing; NaN = never (record! mutates only this)
end
```

**Stimulus protocols, `FiniteDifferenceDiscretization`** — settled v0 shapes, kept as-is: the protocol family is the worked example of principle 5 (dispatch deletes the no-stimulus branch; interval windows skip evaluation on quiet stretches; isbits by construction for GPU capture), and the discretization descriptor stays honest — inert, validating only what the grid genuinely does not know, existing to keep the congruent three-argument `semidiscretize` open for a fourth-order stencil.

```julia
struct NoStimulationProtocol <: AbstractStimulationProtocol end   # fieldless: dispatch deletes the evaluation

struct TransmembraneStimulationProtocol{F,I} <: AbstractStimulationProtocol
    f::F                         # callable (x, t) → Iₐₚₚ; isbits exactly when f is
    nonzero_intervals::I         # Tuple of (t₀, t₁) windows | nothing; Tuple keeps it isbits
end

struct FiniteDifferenceDiscretization
    order::Int                   # == 2 in v0; exists to keep the 3-arg semidiscretize open
end
```

**`DiffusionFunction` / `PointwiseODEFunction`** — internal RHS halves, unexported, reached via public `diffusion_function(f)` / `reaction_function(f)` accessors (promoted from the underscore names 16 test sites already use). The reaction functor sheds its four smuggled metadata fields for one `layout` field. Its *type* is the OS-facing adapter and stays regardless of §5's execution-engine decision — the LockstepODE delegation swaps what runs inside it, not the shape the split function sees.

```julia
struct DiffusionFunction{P,O,G,S<:AbstractStimulationProtocol,T,X<:AbstractVector}
    prepared::P                  # the MFO PreparedOperator mul! applies; stateful — one per solve
    op::O                        # unprepared operator: Adapt re-`prepare`s from it on the target device
    grid::G
    stim::S
    Cₘ::T                        # stimulus enters as Iₛₜᵢₘ/Cₘ
    xs::X                        # node coordinates the stimulus is evaluated at
end

struct PointwiseODEFunction{I,X<:AbstractVector,O}
    ion::I
    xs::X
    overrides::O
    layout::StateBlockedLayout   # replaces the four smuggled metadata fields (nnodes, nstates, 2 symbols)
end
```

## 5. Alternatives considered

**Owning the time-splitting driver vs staying on OrdinaryDiffEqOperatorSplitting (A4).** The measured 3× RHS tax (P1) plus OS's warts (no solution interface, fixed-step usage everywhere) make a self-owned stepper genuinely attractive — for what Lightning uses, it is ~100 lines, and MFO's own examples hand-roll exactly it. It loses because the OS alg-tuple is the settled Rush–Larsen seam: RushLarsenSolvers drops into `LieTrotterGodunov((diff_alg, cell_alg))` with no type changes, and owning a stepper reopens that. Decision: P1 is treated as an upstream bug first (reproducer filed from the review's counting-functor harness); if upstream stalls, the fallback is a minimal non-FSAL forward-Euler substepper that stays *inside* the OS framework; full driver ownership is recorded as last resort so it is never rediscovered from scratch.

**Reaction-half execution: hand-rolled kernels vs delegating to LockstepODE.jl.** The reaction half — N independent per-node cell ODEs advanced in lockstep — is exactly LockstepODE's Batched mode, and the state-blocked layout is byte-identical to its `PerIndex` ordering with ode = node and ode_size = M, so the data seam already matches: no copies, no permutation. The case for delegating: principle 3 says Lightning writes no neighbor code, yet the reaction functor's threaded CPU loop and KA kernel are precisely LockstepODE's job description — delegation turns the review's serial-fallback and GPU-launch-overhead findings (P2, P3) into already-solved problems in the package whose whole job is that loop, and buys the AMDGPU/Metal/oneAPI backends for free. The case against going all the way: the OS split consumes an RHS *function* (the alg tuple's `Euler()` owns the substepping) while LockstepODE's public surface is problem/integrator-level and owns its own timestepping, so nesting its integrator inside a substep reopens A4 and threatens the Rush–Larsen alg-tuple seam; and LockstepODE hard-depends on full OrdinaryDiffEq where Lightning deliberately carries only the lean splitting package. Decision: the middle path — `PointwiseODEFunction` stays as the thin OS-facing adapter the `GenericSplitFunction` contract requires, and its execution engine (the node loop and kernels) delegates to LockstepODE's batched machinery, contingent on (1) the delegated hot path holding the 0 B/call + full-inference contract under the golden masters and (2) LockstepODE exposing its RHS-level engine without the full OrdinaryDiffEq dependency riding along; until both hold, the hand-rolled kernels stay.

**A framework observation layer (Oceananigans-style `Simulation` + callbacks + schedules) vs the dumb primitive.** The framework earns its keep when run control has many axes (output writers, wizards, schedules); Lightning has one loop shape and one recorder today, and the OS integrator does not implement the SciML solution interface a callback framework would want to stand on. A function plus one concrete recorder deletes the duplication now and leaves every richer design open; an interface gets designed when a second recorder exists (rule: no interface before a second implementer).

**Layout as functions/traits vs a value type.** Free functions are what exists today, and the review documents the cost: five scattered arithmetic sites and helpers coupled to upstream field layout. A trait cannot carry N and M. The isbits value type costs one field on the reaction functor and buys a single owner, kernel-passability, and the batching seam.

**Returning a Lightning-owned wrapper from `semidiscretize` vs the bare `GenericSplitFunction`.** The wrapper would make `layout(f)` a plain field read and hide upstream entirely; it loses (as a recommendation — §8 Q1) because the bare return is the congruency contract itself — the result drops into `OperatorSplittingProblem` because it *is* the OS type — and the layout accessor plus public half-accessors already reduce the upstream coupling to one sanctioned site.

**AoS layout, synchronizer objects, per-cell stimulus** — settled against in v0 (approved 2026-08-13) for SoA's contiguous diffusion view, GPU coalescing, and the uniform-stimulus convention; cited here only so this doc is self-contained.

## 6. Cross-cutting rules

**Layout contract.** Voltage is state 1: `semidiscretize` guards `transmembrane_potential_index(ion) == 1` at setup and throws otherwise (F1 — today a `CoupledModel` with `vm_index ≠ 1` silently diffuses a gating variable). The guard lives with `StateBlockedLayout` construction; supporting other indices later is a layout generalization, not an API change.

**Validation.** All construction-time checks live in inner constructors (F2); keyword outers are thin forwards. `Adapt` rules re-`prepare` stateful operators on the target device rather than copying them.

**Exports.** Roughly the current ~33 names plus the v0.x additions (`StateBlockedLayout` (Q2), `layout`, `foreach_step`, `ActivationRecorder`/`record!`/`activation_times`, `diffusion_function`, `reaction_function`): Lightning vocabulary, the re-exported grid vocabulary, and the re-exported splitting vocabulary, so `using Lightning` is one-stop. `solve` stays deliberately unexported. Internals stay namespaced.

**Cell models live in CytoZoo (A3).** `AlievPanfilov` and `TenTusscher2006` graduate from `examples/` to CytoZoo via PRs (the same settled logic that sent FHN there; TT06's self-acceptance test becomes a real CytoZoo test). Lightning's examples are drivers and figures only, and its test suite gains a many-state model for free.

**Invariants.** The frozen cable CV + last-node activation time (measured from unmodified code, 2–5% band) is the coupled-path golden master; the analytic decay oracles are the diffusion golden master; both RHS halves stay 0 B/call (beyond the documented thread-task overhead) and fully inferred; species of change that would move any golden number are rejected, never re-baselined. The DiffEq `p` argument is deliberately ignored by both halves — parameters live in functor fields — and is documented as such.

## 7. Non-goals

**Multi-physics design (bidomain, mechanics).** `AbstractEPModel` is the seam, but designing the second physics before it exists violates the second-implementer rule; deferred to its own doc when scheduled. **Batched multi-simulation runs** — LockstepODE's *other* role here, distinct from §5's reaction-engine delegation: running many tissue solves in lockstep composes later as a second layout type over the A1 seam; nothing to design until then. **CV / activation-map / pseudo-ECG readouts** — future consumers of the recorder, deliberately not designed in §4. **Rush–Larsen** — already slots into the OS alg tuple with no type changes; nothing to design. **A SciML solution object** — settled no; the integrator is driven, not collected. **Spatially varying κ fields** — MFO's `divergence∘scaling∘gradient` composition exists when needed. **AMR, multi-GPU, registration, migration sequencing** — out of scope here; sequencing lives in the review's ordered fix list.

## 8. Open questions

1. **`semidiscretize` return type.** (a) Keep the bare `GenericSplitFunction` + public `layout`/half-accessors; (b) a thin Lightning wrapper forwarding the OS interface. **Recommend (a)** — the bare return is the congruency contract, and A1 already collapses the coupling to one accessor (§5 gives the full trade).
2. **`StateBlockedLayout` export status.** (a) Exported, like Thunderbolt's; (b) namespaced, reached only through `layout(f)`. **Recommend (a)**: the batching layer and any external recorder will dispatch on it, and the printed type is useful configuration reporting.
3. **Recorder mutability shape.** (a) `ActivationRecorder` is a mutable struct mutated by `record!`; (b) immutable struct holding a times buffer. **Recommend (b)** — immutable-with-buffer matches the package's style (functors with array fields), adapts cleanly, and `record!` mutates only the buffer.
4. **`foreach_step` sampling.** (a) Yield after every step; (b) accept a `stride`/`ts` so movie-writing examples don't record every dt. **Recommend (a) now** — the recorder decides what to keep, and a stride keyword is an additive change later if profiling demands it.

## 9. Traceability

| Spec element | Resolves |
|---|---|
| `StateBlockedLayout` + `layout(f)` single reach-through | A1; the Hyrum reach-through in solution.jl; smuggled metadata fields |
| Voltage-index guard at layout construction | F1 |
| Inner-constructor validation rule | F2 |
| `foreach_step` + `ActivationRecorder` | A2; loop ×6 and activation-recording ×3 duplication |
| Cell models graduate to CytoZoo | A3; partially T3 (many-state coverage) |
| Splitting stance: upstream-first, in-framework fallback, ownership last resort | A4, P1 |
| Public `diffusion_function`/`reaction_function` | 16 test-site reaches into underscore internals |
| Golden-master invariants (§6) | T5; review gotcha "the numbers never move" |
| Reaction-engine stance: adapter kept, LockstepODE delegation contingent (§5) | P2, P3 (outsourced when the delegation lands) |
