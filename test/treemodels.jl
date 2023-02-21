@testset "Can model be loaded the first time?" begin
    using OkMLModels
    # Test if the followings occurs:
    # MethodError: no method matching MLJDecisionTreeInterface.DecisionTreeRegressor()
    # The applicable method may be too new: running in world age 29774, while current world is 29778.
    model = manytrees()
    model = fstree()
    @test true
end
