struct MovingWindowCV <: ResamplingStrategy
    train_length
    test_length
    step_size
end


function OkMLModels.train_test_pairs(cv::MovingWindowCV, rows)
    if rows != sort(rows)
        @warn "MovingWindowCV is being applied to `rows` not in sequence. "
    end

    n_obs = length(rows)
    if cv.train_length ≥ n_obs
        throw(ArgumentError(
            "Inusufficient data for training with `train_length`. \n" *
            "Try reducing `train_length`. "
        ))
    end


# extended training phase; i.e, remainder r
# ***                   id1[1]
# vvv                   v      id1[2]
# |-------train----|test|      v                   < first fold
# |  |-step-|----train----|test|                   < 2nd fold
# |  |-step-|-step-|----train----|test|            < 3rd fold
# |  |-step-|-step-|-step-|----train----|test|     < last fold
# |         ^
# |         id0[2]
# |
# |-step-|-step-|-step-***|----train----|test|
# ^                       |                  ^end of data (n_obs)
# |                       ^lastwin0 = n_obs - train_length - test_length + 1
# |                      ^firstwin1 = lastwin0 - 1
# id0[1]

    lastwin0 = n_obs - cv.train_length - cv.test_length + 1
    firstwin1 = lastwin0 - 1

    if  1 > lastwin0 # is 0 or negative
        @warn "Inusufficient data for testing. Actual length for testing data is truncated (< `test_length`) and only 1 fold return."
        return [( rows[1:cv.train_length],
                  rows[(cv.train_length + 1):n_obs])]
        # TODO: test this case
    end
    (m, r) = divrem(firstwin1, cv.step_size)
    nfold = m + 1

    # (m, r, nfold, firstwin1) = OkMLModels.train_test_pairs(MWCV, 1:20) # for debug

    M = reshape((r+1):firstwin1, :, m)
    id0 = M[1, :] # indices for the start of the training phase
    push!(id0, lastwin0)
    id1 = id0 .- 1 .+ cv.train_length # the end of the training phase
    te0 = id1 .+ 1 # the start of the testing phase
    te1 = id1 .+ cv.test_length # the end of the testing phase
    id0[1] = 1 # redefine the beginning of the first training phase

    (id0, id1, te0, te1) # for debug
    return map((a0, a1, b0, b1) -> (rows[a0:a1], rows[b0:b1]), id0, id1, te0, te1)
end
