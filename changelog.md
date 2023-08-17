# Changelog

## v0.1.0

  * Initiating the project.

## v0.2.10

Decision tree model with built-in feature selector.

## v0.2.11

Add a random forest model `manytrees`.

## v0.2.13
Fix the bug of new/old world. Previously calling functions that returns a model will fail at the first time.

## v0.3.0
- Remove all tree models; they now lives in [SWCForecastBase.jl](https://github.com/okatsn/SWCForecastBase.jl).
- New resampler `MovingWindowCV` and corresponding `train_test_pairs` methods for MLJ.

## v0.4.1
- New static transformer `Normalizer` and its underlying functions `normalize`.