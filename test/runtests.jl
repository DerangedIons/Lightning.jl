using Lightning
using Test
using JET

@testset "Lightning.jl" begin
    @testset "Code linting (JET.jl)" begin
        JET.test_package(Lightning; target_defined_modules = true)
    end
    # Write your tests here.
end
