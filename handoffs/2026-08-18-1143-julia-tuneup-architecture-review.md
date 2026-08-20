---
slug: julia-tuneup-architecture-review
created: 2026-08-18-1143
status: open
---

# Handoff: Lightning.jl full review — julia-tuneup audit + high-level architecture pass (findings parked, nothing applied)

## Goal / why this matters

A fresh agent (or Kyle) should be able to act on this review without re-deriving it. It is the complete output of a read-only `/julia-tuneup` audit (7 parallel dimension agents: structure, antipatterns, kernel, typestab, idiom, tests, docs — the two Julia-running agents measured, everything else is static evidence) plus a step-back architecture review of the abstractions, done 2026-08-18 against `main` at `dc65f77`. **No code was changed and no branch was created.** Every finding below carries file:line evidence; the measured numbers were taken on Julia 1.12.6, Apple M5 Pro, 13 threads, warm environment.

The one-sentence verdict up front: **Lightning v0 is a genuinely well-built package — the layering is right, the types are clean, the tests are unusually good — and the review found no reason for wholesale reorganization.** The real findings are: one silent-wrong-physics contract gap (voltage-index), one measured 3× per-step performance tax that lives upstream, a handful of robustness/test gaps, and three targeted architecture moves for v0.x (a first-class layout type, an observation/recording layer, promoting the example cell models to CytoZoo).

## Background & current state

Lightning solves the monodomain equation on orthogonal structured grids: MatrixFreeOperators.jl (MFO) supplies matrix-free spatial operators, CytoZoo.jl supplies pointwise cell kinetics, OrdinaryDiffEqOperatorSplitting (OS) supplies Lie–Trotter/Strang splitting. The pipeline is Thunderbolt-congruent by convention (`MonodomainModel → ReactionDiffusionSplit → semidiscretize → GenericSplitFunction → OperatorSplittingProblem`), with zero Thunderbolt dependency. The approved v0 design is `handoffs/2026-08-13-2019-lightning-monodomain-v0.md` (still marked `status: open`; the implementation it specifies is now complete — consider flipping it to done). ~750 src lines across 7 files, ~800 test lines across 6 files, 6 examples.

Phase-0 facts: archetype A (module owns all includes/exports, zero orphans); package loads clean; 33 exports; 9 types with every field concrete or a type parameter; 0 untyped globals; CI (1.10/1.11/1.12, ubuntu x64) + Dependabot + TagBot present; all deps have compat bounds.

**Verified clean, with measurements (do not re-audit these):** both RHS halves fully infer (`Core.Compiler.return_type == Nothing`, one level down included, with and without stimulus/overrides); diffusion RHS is 0 bytes/call warm in all three stimulus regimes; the tuple-κ operator-tree setup instability does not leak past the `prepare` barrier (constructed `DiffusionFunction` is concrete, 0 B/call); all solution helpers infer concretely; `node_coordinates` CPU branch folds to `Vector{SVector{N,T}}`; per-call `get_backend(u)` is free on CPU; docstring coverage of the 16 own-vocabulary exports is complete; README/docs snippets match the real signatures; per-site discontinuity review of all 9 `init` call sites found every stimulus window edge is an exact multiple of that site's fixed dt, so no `tstops` mechanism is needed anywhere today.

## Key files / locations

- `src/Lightning.jl` — module, explicit imports, exports (re-exports MFO grid vocabulary + OS splitting vocabulary)
- `src/models.jl` — `MonodomainModel`, `ReactionDiffusionSplit`; `src/stimulus.jl` — protocols; `src/discretization.jl` — `FiniteDifferenceDiscretization`
- `src/semidiscretize.jl` — `DiffusionFunction`, `semidiscretize`, `diffusion_operator`, validation; `src/reaction.jl` — `PointwiseODEFunction`, CPU threaded loop + KA kernel; `src/solution.jl` — layout helpers (`getvariable`/`setvariable!`/`variable_range`/`create_initial_condition`)
- `test/` — analytic heat-equation oracles (`test_diffusion.jl`), FHN cable signatures (`test_cable.jl`), layout/sign-convention/inference (`test_pipeline.jl`), GPU parity (`test_gpu.jl`), `testutils.jl` (`NullIonicModel`, discrete/continuum decay rates)
- `examples/` — `cable_1d.jl`, `spiral_2d.jl`, `niederer_benchmark.jl` (+ untracked `niederer_animation.jl`), and two CytoZoo-interface cell models that live here: `aliev_panfilov.jl` (73 lines), `ten_tusscher_2006.jl` (504 lines, self-acceptance-testing)
- `.github/workflows/CI.yml` — installs the two unregistered deps from un-pinned `main` via `Pkg.add(url=…)` (lines 42–49, 73–93)

## Findings — correctness hazards (report-only; fixes move behavior for currently-invalid inputs, not numbers for valid ones)

### F1. The state-blocked layout hard-codes voltage as cell-model state 1, and nothing validates it *(high; found independently by 3 agents)*

`semidiscretize` assigns the diffusion dof range as `1:nnodes` unconditionally (src/semidiscretize.jl:116) and `variable_range(f, :states)` returns `(N+1):(N*M)` (src/solution.jl:146). But CytoZoo's interface permits any `transmembrane_potential_index`, and `CoupledModel` (a legal `AbstractCardiacCellModel`) reports `cm.layout.vm_index`, which is not structurally 1. With such a model, the diffusion half silently diffuses a gating variable while `getvariable(u, f, :φₘ)` (which *does* honor the index, src/solution.jl:153) reads a different block — wrong physics, no error, and the README/docs promise "the diffusion half acts on the leading block 1:N". **Fix:** one setup-time guard in `semidiscretize` — `CytoZoo.transmembrane_potential_index(model.ion) == 1 || throw(ArgumentError(...))` — plus a test-local model with voltage at index 2 asserting the throw. All currently shipped simple models return 1, so nothing breaks.

### F2. Keyword-constructor validation is bypassable via the auto-generated positional constructors *(medium)*

`MonodomainModel`'s positivity/symbol checks and `TransmembraneStimulationProtocol`'s `_normalize_intervals` run only in the keyword outer constructors (src/models.jl:82-98, src/stimulus.jl:75-78); the default positional constructors skip them, so a positional `TransmembraneStimulationProtocol(f, [(0.0, 2.0)])` stores a `Vector` and silently loses the isbits/GPU property. **Fix:** move validation into inner constructors; keyword outers become thin forwards. Note `Adapt.adapt_structure` (src/stimulus.jl:123-125) calls the positional form with already-normalized intervals — normalization is idempotent on tuples, so this is safe.

## Findings — performance (all measured)

### P1. Every solve pays 3 RHS evaluations per operator per step where forward Euler needs 1 *(high; upstream)*

Counting functors wrapped into the split function: 10 `step!` calls → exactly +30 diffusion and +30 reaction evaluations. Cross-checked twice: `@allocated step!` = 20,448 B = exactly 3 × the 6,816 B of one threaded reaction call, and 100k-node step time 482.6 µs ≈ 3 × 157.4 µs. Cause: OS 0.4.1's substep resync marks `u` modified, forcing an `fsalfirst` recompute on top of Euler's trailing (discarded) `fsallast` evaluation. On the Niederer benchmark (420k nodes × 19 states × 6,000 steps, reaction-dominated) this is ≈3× total solve time. **Fix options:** (a) confirm and fix/file upstream in OrdinaryDiffEqOperatorSplitting; (b) ship a minimal non-FSAL forward-Euler substepper for the split. See also the architecture note A4 below on owning the splitting driver outright.

### P2. `Threads.@threads` in the CPU reaction kernel has no serial fallback *(medium)*

Fixed ~6.8 KiB + task-spawn latency per RHS call regardless of N (src/reaction.jl:55); it is the *only* allocation source in either RHS. At 256 nodes the threaded loop is 7.6× slower than an equivalent serial loop (19.5 µs vs 2.6 µs); at 100k nodes it is 6.5× faster (157 µs vs 1018 µs). **Fix:** size dispatch in the CPU `_react!` — serial below `N < c·nthreads()` (crossover is between 256 and 100k; measure to pick c). Scheduling cannot change any computed value (disjoint strided writes, no reduction).

### P3. GPU path routes every kernel launch and an unconditional `synchronize` through `Base.invokelatest` per RHS call *(medium; unverified on device — no GPU on this machine)*

src/reaction.jl:74,77. The world-age rationale in the comment does not hold: a `CUDABackend` value cannot exist before CUDA.jl is loaded, so plain dynamic dispatch suffices. The bigger cost is the host-blocking `synchronize` on every RHS evaluation (×3 per step given P1) when the integrator's device-side consumers are already queue-ordered. **Fix:** direct calls; audit whether the synchronize can move to where the host actually reads state; verify parity via test/test_gpu.jl on a GPU machine. Run JET once after removing `invokelatest` to confirm nothing surfaces under `target_modules=(Lightning,)`.

### P4. Small polish *(low)*

- Pin `ODEFunction{true}` at src/semidiscretize.jl:115-117 — the iip parameter is the only non-concrete part of `semidiscretize`'s return type (setup-time only).
- On a Float32 grid, `κ_eff` stays Float64 inside the operator tree while `Cₘ` is converted (src/semidiscretize.jl:110 vs 138); fold κ_eff to the grid eltype in `diffusion_operator`. Behavior-changing at O(eps(Float32)) on the discouraged Float32 path only.

## Findings — tests (the suite is strong; these are the holes it acknowledges or can't see)

- **T1 *(medium)*** — The zero-allocation guard on the diffusion RHS is permanently skipped in CI: `Pkg.test` forces `--check-bounds=yes`, the test gates on `check_bounds == 0` (test/test_pipeline.jl:138), and no CI job passes `check_bounds: 'auto'`. Allocation regressions land silently. Fix: one matrix entry or job with `julia-actions/julia-runtest@v1`'s `check_bounds: 'auto'`.
- **T2 *(medium)*** — `StrangMarchuk` is exported but exercised by nothing (grep: src/Lightning.jl only). Strang applies the first operator twice per step with half-steps — a re-entrancy pattern on the stateful prepared operator that Lie–Trotter never triggers. The existing analytic decay test verifies it nearly for free (splitting error is zero when reaction is identity).
- **T3 *(medium)*** — No cell model with more than 2 states anywhere in tests; test_pipeline.jl:61-63 itself admits `:states` vs a named state "coincide here only because FHN has exactly one non-voltage state". Add a test-local 3-state model; assert multi-block `variable_range`, per-block fill, middle-state `setvariable!`, one kernel application with distinct per-state derivatives.
- **T4 *(medium)*** — No 3D grid and no anisotropy with ≥2 correction axes; the `_diffusion_operator` loop's deeper `Added` tree is never built. The `discrete_decay_rate` oracle already generalizes; a 3D κ=(a,b,c) single-application test to ~1e-11 is nearly free.
- **T5 *(medium)*** — The coupled path has no quantitative oracle: conduction velocity is only asserted positive with a 25% self-consistency band (test/test_cable.jl:104-105). A χ·Cₘ-scaling bug slowing CV 20% would pass everything. Fix: freeze a measured CV + last-node activation time with a 2–5% band. **Measure the frozen value BEFORE any code change** — this doubles as the golden master any future tuneup needs.
- **T6 *(medium)*** — No Aqua.jl testset (ambiguities, unbound type params, piracy — the piracy check would mechanically enforce what src/solution.jl:24-25 does by discipline).
- **Low:** Float32 coverage stops at construction (never steps; models.jl:50 claims "Float32 runs" untested); one shared test namespace with leaking consts/usings (bare `module` wrap or SafeTestsets); GPU suite never calls `setvariable!` on a device array (the one solution-API path with a real GPU compilation hazard, test/test_gpu.jl:95-96); cable_1d.jl and test_cable.jl lack the explicit-Euler CFL guard every other driver carries (currently 2.5–5× margin); examples never smoke-tested in CI (they are the only place multi-state models run at all); the CI doctest step is vacuous (zero `jldoctest` blocks in the repo).

## Findings — structure / infra

- **S1 *(medium)*** — Nothing pins the two unregistered deps anywhere: root Manifest is NOT tracked (`.gitignore:4 /Manifest*.toml` — note the v0 design doc believed it would be), root Project.toml has no `[sources]`, and CI installs both from floating `main` (CI.yml:46-48, three separate resolve steps). A breaking upstream push fails or silently changes Lightning CI with no change in Lightning. Pick one: `rev=`-pinned PackageSpecs in CI bumped deliberately, a tracked Manifest, or root `[sources]` once julia compat is ≥1.11. Related: `examples/Project.toml` already uses `[sources]` with `rev = "main"` — which requires Julia ≥ 1.11 while the package claims `julia = "1.10"`, so the documented example invocation cannot resolve on the LTS (README needs a one-line caveat); and the README install block says `Pkg.develop(url=…)` where CI's tested path is `Pkg.add(url=…)`.
- **Low:** `examples/niederer_animation.jl` is the only untracked source file while its `.gitignore` plumbing and tracked output figure are committed — add it or discard it explicitly; `examples/ten_tusscher_2006.jl:25-26` runs `Pkg.activate(@__DIR__)` at include time (guard with `abspath(PROGRAM_FILE) == @__FILE__` like its own self-test at line 424; sibling `aliev_panfilov.jl` has no activate); six generated figures/CSV (~1 MiB) are tracked but referenced by nothing (deliberate per the `.gitignore:8-9` comment — either embed them in README/docs so they earn their place, or untrack); dead `stable` docs badge until the first tag; unfiltered `@autodocs` renders `DiffusionFunction`/`PointwiseODEFunction`/internals alongside the public API (split Public/Private blocks); `[extras]`/`[targets]` vs `test/Project.toml` is defensible as-is per JuliaPackageTemplate; CI matrix is ubuntu/x64 only (one macos-latest entry is the cheap addition); document on both RHS docstrings that the DiffEq `p` argument is deliberately ignored (params live in functor fields — a future `remake`-style sweep would silently do nothing).

## Architecture pass — the step-back review

**What the package is, correctly:** the tissue layer of a three-package stack, deliberately thin (~750 lines of glue + naming + layout), whose entire value proposition is (1) the structured-grid restriction that buys matrix-free operators and single-source GPU, and (2) Thunderbolt-congruent vocabulary so model descriptions read the same in either. The delegation boundaries are exactly right: pointwise kinetics belong to CytoZoo, spatial operators to MFO, time-splitting to OS. **No wholesale reorganization is warranted; suggesting one would be change for its own sake.** The four moves below are targeted, and none reopens a settled v0 decision — they are the v0.x growth path.

### A1. Make the layout a first-class type: `StateBlockedLayout`

The state-blocked layout is the package's central invariant, but it exists only as scattered index arithmetic: `_state_block` (src/solution.jl:150), `_node_slice` (src/reaction.jl:47), `variable_range` (src/solution.jl:143), the `1:nnodes` dof range (src/semidiscretize.jl:116), and `nnodes`/`nstates`/`φ_symbol`/`state_symbol` fields smuggled onto the reaction functor as "metadata for the solution helpers" (its own docstring says so, src/reaction.jl:18-20). Meanwhile every solution helper reaches the metadata via `f.functions[2].f` — a path through the internal field layout of `ODEFunction` and of OS v0.4, a package whose API is explicitly unstable (Hyrum's law, but pointed inward: Lightning's *public* helpers depend on upstream's *private* shape, so an upstream field rename is a Lightning breaking change).

The move: an isbits `StateBlockedLayout` value type owning N, M, the two symbols, and all block/stride arithmetic; `layout(f::GenericSplitFunction)` extracts it once (the single sanctioned reach-through), and every helper dispatches on the layout. The reaction functor carries it as one field instead of four. This is *more* Thunderbolt-congruent, not less — Thunderbolt main literally names `StateBlockedLayout` — and it is precisely the seam the settled "batching composes on top later" decision needs: a batched layer becomes a new layout type with the same accessor contract, no type changes elsewhere. Whether `semidiscretize` should *also* return a thin Lightning-owned wrapper instead of the bare `GenericSplitFunction` is a v0.2 conversation, not a v0 relitigation — the observed cost of the bare return today is 16 test call sites reaching into `Lightning._diffusion_function`/`_reaction_function` (promote those two accessors to public exported names either way; they already exist and are the right shape).

### A2. An observation/recording layer is the missing abstraction the repo is already asking for

The `while integrator.t < TEND - DT/2; step!(integrator, DT, true); …` loop is hand-rolled in six places (README, docs index, test_cable, cable_1d, spiral_2d, niederer_benchmark + animation), and first-threshold-crossing activation recording is independently reimplemented in at least three of them (test/test_cable.jl:47-56, examples/cable_1d.jl:78-88, examples/niederer_benchmark.jl:177-195). CV/pECG *post-processing* was correctly deferred, but this is the layer beneath it: a tiny `record!`-style observer API (or even just an exported, documented `ActivationRecorder` + a `foreach_step` helper wrapping the loop) would delete the most-duplicated code in the repo, give the deferred conduction-velocity/pECG follow-ups their natural home, and fix the fact that the README's very first user-facing snippet is a manual stepping loop. Keep it deliberately dumb: fixed sample stride, no SciML solution interface pretensions (the no-`sol` decision stands).

### A3. Promote the example cell models to CytoZoo

`aliev_panfilov.jl` and `ten_tusscher_2006.jl` (504 lines, with its own single-cell acceptance targets) implement the full CytoZoo interface but live untested in `examples/` — the aliev_panfilov file itself says "Promoting it to the zoo is a separate conversation." Have that conversation: the same logic that sent FHN to CytoZoo (settled decision 7 — the zoo is the home for cell models; Thunderbolt ships both of these too) applies verbatim, TT06's acceptance test becomes a real CytoZoo test, and Lightning's multi-state coverage gap (T3) partially closes for free because CI would then have a many-state model available. Examples shrink to what they should be: drivers and figures.

### A4. The splitting dependency is the one boundary worth re-examining — but fix upstream first

P1 (3× RHS evaluations) plus the already-documented warts (no SciML solution interface, `solve` deliberately unexported, fixed-step-only usage everywhere) raise a fair strategic question: for what Lightning actually uses — fixed-step Lie–Trotter/Strang over two operators with contiguous aliasing dof ranges — a self-owned stepper is ~100 lines, and MFO's own examples already hand-roll exactly it. The reason NOT to: the OS alg-tuple seam is the settled Rush–Larsen slot-in path (RushLarsenSolvers implements DiffEqBase algorithms and drops into `LieTrotterGodunov((diff_alg, cell_alg))` with no type changes), and owning a stepper reopens that. Recommendation: treat P1 as an upstream bug first (file/fix in OrdinaryDiffEqOperatorSplitting with the counting-functor harness as the reproducer); only if upstream stalls does the in-house substepper (option b in P1 — a non-FSAL Euler that stays *inside* the OS framework) become attractive. Full driver ownership is the last resort, recorded here so the option isn't rediscovered from scratch.

### Smaller architecture notes

- `overrides` (spatial heterogeneity) lives on `ReactionDiffusionSplit`, congruent with Thunderbolt's slot — fine, but note the conceptual seam: heterogeneity is a property of the continuous tissue, not of the numerical splitting annotation. If an unsplit/IMEX path ever appears, overrides must move to the model; keeping `semidiscretize`'s internals ignorant of *where* overrides came from keeps that move cheap.
- `FiniteDifferenceDiscretization` as a validated-but-inert descriptor is honest and correctly documented; keep.
- The re-export surface, the `false`-as-strong-zero stimulus, the two-stage conductivity validation, and symbol-based (not `Val`-based) state naming were all explicitly reviewed and judged correct as-is — do not churn them.

## Decisions & conclusions

1. **Review-only run, by instruction** — findings parked here; no branch, no commits, no fixes applied, tests never run (standing rule).
2. The v0 architecture is sound; the growth path is A1–A4 above, not reorganization.
3. F1 (voltage-index guard) is the highest-value trivial fix in the package; three agents found it independently.
4. P1 is real, measured, and upstream; Lightning's own kernels are clean (0 B/call diffusion, inference-clean everywhere).
5. Before ANY future fixing session: capture the golden master first (T5's frozen CV constant is the natural coupled-path oracle and must be measured from unmodified code), and never regenerate a baseline in the same commit as a code change.

## What's left / next steps (ordered for a future fixing session)

1. Capture baselines: frozen CV + activation time from the unmodified cable (T5), and re-run the kernel agent's counting-functor + `@allocated` harness numbers as the perf baseline.
2. F1: voltage-index guard in `semidiscretize` + rejection test. F2: inner-constructor validation.
3. S1: pick and implement the dependency-pinning story (affects CI reproducibility for everything after).
4. T1 (CI `check_bounds: 'auto'` job), T2 (Strang decay test), T3 (3-state test model), T4 (3D anisotropy test), T6 (Aqua).
5. P2 (serial fallback below threshold), P4 (`ODEFunction{true}`, κ_eff eltype).
6. P1: minimal reproducer → upstream issue/PR on OrdinaryDiffEqOperatorSplitting; decide (b) only if upstream stalls.
7. A1 (`StateBlockedLayout` + public half-accessors), then A2 (observer/recorder), then A3 (CytoZoo PRs for AlievPanfilov + TT06).
8. P3 on a GPU machine (drop `invokelatest`, audit the per-call synchronize, verify parity).
9. Low-severity sweep: untracked animation script, `Pkg.activate` guard, README install block, badge, `@autodocs` split, CFL guards, Float32 step test, doctest policy.

## Gotchas / constraints

- **The numbers never move.** Any fixing session must follow the julia-tuneup oracle discipline: baseline before touching anything, oracle after every unit, red oracle ⇒ revert (never re-baseline). Item 1 above exists because this review deliberately created no baseline.
- **Do not run tests without Kyle's explicit go-ahead** (standing rule; a tuneup session's Phase-1 gate is the sanctioned way to get it).
- The `Threads.@threads` allocation (6.8 KiB/call) is task machinery, not boxing — do not "fix" it with closure gymnastics; the fix is the size dispatch (P2).
- The scan script's "script smells" in src/ are all inside docstring example fences — false positives, already verified.
- No tracked mp4s exist (phase-0 scan claimed otherwise; `.gitignore:11` covers them).
- `examples/Project.toml` `[sources]` needs Julia ≥ 1.11; the root package claims 1.10 compat — don't copy the examples mechanism to the root until compat moves.
- GPU findings (P3, T-low setvariable!) are unverifiable on this machine (no CUDA device); anything touching them needs the GPU box.
- Previous-run suppression: this is the first review; there is no prior "Not applied" table to honor.
