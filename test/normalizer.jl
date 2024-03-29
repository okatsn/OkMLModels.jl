using DataFrames, MLJ
@testset "normalizer.jl" begin
    # test OkMLModels.normalize
    for i = 1:10
        let
            xmin, xmax = extrema(5 * randn(2) .+ 10)
            ratio = rand(0.1:0.1:0.9, 1) |> only
            v = xmin + ratio * (xmax - xmin)
            v0 = OkMLModels.normalize(v, xmin, xmax)
            @test isapprox(v0 - 0.0, ratio)
        end
    end

    x1 = randn(1000)
    x2 = 3 * randn(1000)
    x3 = -2 * randn(1000) .- 5
    df = DataFrame(
        :x1 => x1,
        :x2 => x2,
        :x3 => x3
    )

    mach = machine(Normalizer()) |> fit!
    dfn = MLJ.transform(mach, df)

    # test if minimum and maximum is 0 and 1
    @test all([only(col) for col in eachcol(combine(dfn, All() .=> minimum))] .== 0.0)
    @test all([only(col) for col in eachcol(combine(dfn, All() .=> maximum))] .== 1.0)
    xmins = Dict([:x1, :x2, :x3] .=> minimum.([x1, x2, x3]))
    xmaxs = Dict([:x1, :x2, :x3] .=> maximum.([x1, x2, x3]))


    unnormalize(xn, xmin, xmax) = xn .* (xmax - xmin) .+ xmin

    for (k, v) in pairs(eachcol(dfn))
        v0 = unnormalize(v, xmins[k], xmaxs[k])
        @test all(isapprox.(v0, df[:, k]))
    end

end
