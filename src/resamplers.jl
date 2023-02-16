struct MovingWindowCV <: ResamplingStrategy
    train_length
    test_length
    step_size
end


function OkMLModels.train_test_pairs(cv::MovingWindowCV, rows)
    if rows != sort(rows)
        @warn "MovingWindowCV is being applied to `rows` not in sequence. "
    end
end
