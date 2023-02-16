"""
Return a composite tree model with `FeatureSelector`.
"""
function fstree()
    selector = FeatureSelector()

    treemodel = (@load "DecisionTreeRegressor" pkg = "DecisionTree" verbosity = 0)() # regressor type

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

    treemodel = (@load "DecisionTreeRegressor" pkg = "DecisionTree" verbosity = 0)() # regressor type

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
    tree = (@load "DecisionTreeRegressor" pkg = "DecisionTree" verbosity = 0)()
    model = EnsembleModel(; model=tree, acceleration= MLJ.CPUThreads(), bagging_fraction=bagging_fraction, n=n)
end
