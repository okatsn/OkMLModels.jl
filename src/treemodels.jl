# MLJ use @load to load model. If you use it inside a function, e.g., fstree(), it will fail the first time with `MethodError: no method matching MLJDecisionTreeInterface.DecisionTreeRegressor(). The applicable method may be too new: running in world age 29774, while current world is 29778.`. To solve this, the @load has to be executed at the compile time.

regressortree = (@load "DecisionTreeRegressor" pkg = "DecisionTree" verbosity = 0)

"""
Return a composite tree model with `FeatureSelector`.
"""
function fstree()
    selector = FeatureSelector()

    treemodel = regressortree() # regressor type

    mypipe = Pipeline(
        selector = selector,
        tree = treemodel
    )

    return mypipe
end


"""
Return a composite tree model with TWO `FeatureSelector`: `selector_1` and `selector_2`.
"""
function twofstree()
    selector1 = FeatureSelector()
    selector2 = FeatureSelector()

    treemodel = regressortree() # regressor type

    mypipe = Pipeline(
        selector_1 = selector1,
        selector_2 = selector2,
        tree = treemodel
    )

    return mypipe
end

"""
A random forest using `EnsembleModel`
"""
function manytrees(;bagging_fraction=0.8, n=100)
    tree = regressortree()
    model = EnsembleModel(; model=tree, acceleration= MLJ.CPUThreads(), bagging_fraction=bagging_fraction, n=n)
end
