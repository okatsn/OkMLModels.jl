whatndo = "rescales `x` onto the scale of 0 to 1"

"""
A static transformer that $whatndo.

# Example

```julia
stand1 = Normalizer() # rescale to 0-1
mach = machine(stand1)
fit!(mach)

MLJ.transform(mach, df::DataFrame)
```


See also: `normalize`
"""
mutable struct Normalizer <: Static
end

function normalize(v, xmin, xmax)
    return (v - xmin) / (xmax - xmin)
    # see https://www.codingninjas.com/studio/library/normalisation-vs-standardisation
end


"""
`normalize(x)` $whatndo.
"""
function normalize(x::Vector)
    xmin = minimum(x)
    xmax = maximum(x)
    return normalize.(x, xmin, xmax)
end

# https://alan-turing-institute.github.io/MLJ.jl/stable/transformers/#Static-transformers
# https://alan-turing-institute.github.io/MLJ.jl/dev/adding_models_for_general_use/#Unsupervised-models
function MLJ.transform(::Normalizer, _, X::DataFrame)
    X = deepcopy(X)
    for (k, v) in pairs(eachcol(X))
        transform!(X, k => normalize; renamecols=false)
    end
    return X
end




# mutable struct Unnormalizer <: Static
#     names
#     xmins
#     xmaxs
#     function Unnormalizer(X)

#         for (k, v) in pairs(eachcol(X))

#         end
#         new(names, xmins, xmaxs)
#     end
# end


# function MLJ.transform(UN::Unnormalizer, _, X::DataFrame)
#     X = deepcopy(X)
#     for (k, v) in pairs(eachcol(X))
#         transform!(X, k => (x -> unnormalize(x, UN.xmin, UN.xmax)); renamecols=false)
#     end
#     return X
# end
