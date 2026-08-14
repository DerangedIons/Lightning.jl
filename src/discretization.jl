"""
    FiniteDifferenceDiscretization(; order = 2)

Finite-difference spatial discretization descriptor.

Deliberately minimal: spacing, boundary conditions, and the device all live on the
`CartesianGrid`, and duplicating them here is exactly the wart this replaces — ArmyHeart's
discretization object validated an `order` and a `boundary` it then never read. The only
thing this type carries is what the grid genuinely does not know, and the only reason it
exists at all is to keep the three-argument `semidiscretize(split, disc, grid)` signature
open for a fourth-order stencil later.

`order` must be `2`; MatrixFreeOperators' second-derivative stencils are second order.

See also: [`semidiscretize`](@ref).
"""
struct FiniteDifferenceDiscretization
    order::Int

    function FiniteDifferenceDiscretization(order::Integer)
        order == 2 || throw(
            ArgumentError(
                "only second-order finite differences are supported, got order = $order"
            ),
        )
        return new(Int(order))
    end
end

FiniteDifferenceDiscretization(; order::Integer=2) = FiniteDifferenceDiscretization(order)
