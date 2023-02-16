@testset "MovingWindowCV" begin
    MWCV = MovingWindowCV(10, 5, 5)
    @test OkMLModels.train_test_pairs(MWCV, 1:20) == [
        (  collect(1:10)  ,  collect(11:15)  ),
        (  collect(6:15)  ,  collect(16:20)  ),
    ]

    MWCV = MovingWindowCV(17, 6, 5)
    trtes = OkMLModels.train_test_pairs(MWCV, 1:273)
    @test all(length.(first.(trtes))[2:end] .== 17)
    @test all(length.(last.(trtes)) .== 6)

    trs = first.(trtes) # train phases
    tes = last.(trtes)  #  test phases
    @test all(last.(trs) .+ 1 .== first.(tes) ) || "Test if train-test phases are tightly adjacent failed"
    @test isequal(sort(last.(trs)), last.(trs)) || "Test equality in order failed"


end
