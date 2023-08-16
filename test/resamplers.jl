@testset "MovingWindowCV" begin
    MWCV = MovingWindowCV(10, 5, 5)
    @test OkMLModels.train_test_pairs(MWCV, 1:20) == [
        (  collect(1:10)  ,  collect(11:15)  ),
        (  collect(6:15)  ,  collect(16:20)  ),
    ]

    train_length = 17
    test_length = 6
    step_size = 5

    MWCV = MovingWindowCV(train_length, test_length, step_size)

    trtes = OkMLModels.train_test_pairs(MWCV, 1:273)
    trs = first.(trtes) # train phases
    tes = last.(trtes)  #  test phases

    ## Check length of each partition
    @test all(length.(trs)[2:end] .== train_length)
    @test all(length.(tes) .== test_length)



    @test all(last.(trs) .+ 1 .== first.(tes) ) || "Test if train-test phases are tightly adjacent failed"
    @test isequal(sort(last.(trs)), last.(trs)) || "Test equality in order failed"

    ## Test step size
    @test all(diff(last.(trs)) .== step_size)
    @test all(diff(last.(tes)) .== step_size)
    @test all(diff(first.(tes)) .== step_size)

end
