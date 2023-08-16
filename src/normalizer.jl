
mutable struct Normalizer <: Static
end

function normalize(x)
    xmin = minimum(x)
    xmax = maximum(x)
    return (x .- xmin) ./ (xmax - xmin)
    # see https://www.codingninjas.com/studio/library/normalisation-vs-standardisation
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
