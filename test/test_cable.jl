# The 1D FHN cable — the first test in which the reaction and diffusion halves have to
# cooperate. Nothing here has a closed form; what is checked is the qualitative signature of
# a propagating action potential, which a broken coupling cannot fake: a wave that starts
# where it was stimulated, arrives everywhere later, arrives monotonically later with
# distance, and reverses when the stimulus does.

const CABLE_LENGTH = 20.0
const CABLE_NODES = 200
const CABLE_κ = 0.1
const CABLE_DT = 0.01
const CABLE_TEND = 300.0
const CABLE_THRESHOLD = 0.5      # FHN φₘ runs over roughly [0, 1]; halfway is unambiguous

function cable_grid()
    return CartesianGrid(
        ((0.0, CABLE_LENGTH),), (CABLE_NODES,); bc=((Neumann(), Neumann()),)
    )
end

"A 2 ms, 1 mm-wide depolarizing pulse at the `left` or `right` end of the cable."
function cable_stimulus(side::Symbol)
    inside = side === :left ? (x -> x[1] <= 1.0) : (x -> x[1] >= CABLE_LENGTH - 1.0)
    return TransmembraneStimulationProtocol(
        (x, t) -> (t <= 2.0 && inside(x)) ? 0.5 : 0.0; nonzero_intervals=((0.0, 2.0),)
    )
end

"""
Run the cable and return the first time each node crosses `CABLE_THRESHOLD` (`NaN` if it
never does), alongside the final voltage profile.
"""
function run_cable(side::Symbol)
    grid = cable_grid()
    model = MonodomainModel(;
        κ=CABLE_κ, ion=FHNModel(), stim=cable_stimulus(side)
    )
    f = semidiscretize(
        ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid
    )

    integrator = init(
        OperatorSplittingProblem(f, create_initial_condition(f), (0.0, CABLE_TEND)),
        LieTrotterGodunov((Euler(), Euler()));
        dt=CABLE_DT,
    )

    activation = fill(NaN, CABLE_NODES)
    peak = fill(-Inf, CABLE_NODES)
    while integrator.t < CABLE_TEND - CABLE_DT / 2
        step!(integrator, CABLE_DT, true)
        φ = getvariable(integrator.u, f, :φₘ)
        for i in eachindex(activation)
            peak[i] = max(peak[i], φ[i])
            isnan(activation[i]) && φ[i] > CABLE_THRESHOLD && (activation[i] = integrator.t)
        end
    end
    return (;
        activation,
        peak,
        f,
        φ_final=copy(getvariable(integrator.u, f, :φₘ)),
        xs=node_coordinates(f),
    )
end

@testset "Cable — a stimulus captures and the wave propagates away from it" begin
    left = run_cable(:left)

    @info "left-stimulated cable" activated = count(!isnan, left.activation) t_first = minimum(
        left.activation
    ) t_last = maximum(left.activation) peak = maximum(left.peak) φ_final = extrema(
        left.φ_final
    )

    # Capture: every node fires. A stimulus that failed to reach threshold, or a diffusion
    # coefficient that failed to carry it, leaves NaNs here.
    @test all(!isnan, left.activation)
    @test maximum(left.peak) > 0.9                      # a real upstroke, not a smear

    # Direction: the stimulated end goes first, the far end last.
    @test argmin(left.activation) == 1
    @test argmax(left.activation) == CABLE_NODES
    @test left.activation[1] <= 2.0                     # within the stimulus window itself

    # Monotone in x — the property that distinguishes a propagating wave from a domain that
    # simply depolarized everywhere at once.
    @test all(>=(0), diff(left.activation))

    # A plausible conduction velocity, and one that is nearly uniform along the cable: the
    # front should be travelling at a steady speed by the time it leaves the stimulus site.
    span = left.xs[end][1] - left.xs[1][1]
    cv = span / (left.activation[end] - left.activation[1])
    half = CABLE_NODES ÷ 2
    cv_first =
        (left.xs[half][1] - left.xs[1][1]) / (left.activation[half] - left.activation[1])
    cv_second =
        (left.xs[end][1] - left.xs[half][1]) /
        (left.activation[end] - left.activation[half])
    @info "conduction velocity" cv cv_first cv_second
    @test cv > 0
    @test isapprox(cv_first, cv_second; rtol=0.25)

    # And the tissue recovers: FHN's slow variable pulls it back to rest well inside 300
    # time units.
    @test maximum(left.φ_final) < 0.1
end

@testset "Cable — mirrored stimulus gives a mirrored wave" begin
    # The grid, operator, and kinetics are all left-right symmetric, so stimulating the
    # other end must reproduce the same activation sequence reversed. This catches an
    # asymmetric boundary stencil or an off-by-one in the node ordering, neither of which
    # the monotonicity test above would notice.
    left = run_cable(:left)
    right = run_cable(:right)

    @test all(!isnan, right.activation)
    @test argmin(right.activation) == CABLE_NODES
    @test argmax(right.activation) == 1
    @test all(<=(0), diff(right.activation))

    mirrored = reverse(right.activation)
    @info "mirror symmetry" max_deviation = maximum(abs, left.activation .- mirrored)
    @test left.activation ≈ mirrored
end

@testset "Cable — a subthreshold stimulus does not propagate" begin
    # The complement of the capture test: below FHN's excitation threshold the perturbation
    # must diffuse away rather than launch a wave. Without this, "everything fires" would
    # pass even if the model were unconditionally excitable.
    grid = cable_grid()
    weak = TransmembraneStimulationProtocol(
        (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 0.02 : 0.0; nonzero_intervals=((0.0, 2.0),)
    )
    model = MonodomainModel(; κ=CABLE_κ, ion=FHNModel(), stim=weak)
    f = semidiscretize(
        ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid
    )

    integrator = init(
        OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 100.0)),
        LieTrotterGodunov((Euler(), Euler()));
        dt=CABLE_DT,
    )
    solve!(integrator)
    φ = getvariable(integrator.u, f, :φₘ)

    @info "subthreshold cable" φ_extrema = extrema(φ)
    @test maximum(φ) < CABLE_THRESHOLD
    @test all(isfinite, φ)
end
