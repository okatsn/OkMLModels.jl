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
