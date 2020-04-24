module DynamicSupport

using Gen

@gen function foo()
    res = Array{Float64, 1}([])
    counter = 0
    # :x => counter is addr
    while @trace(normal(0.0, 1.0),
                 :x => counter) < 2.0
        push!(res, 5)
        counter += 1
    end
    return res
end

@gen function proposal_foo()
    counter = 0
    # :x => counter is addr
    while @trace(normal(0.0, 1.0),
                 :x => counter) < 2.0 && counter < 3
        counter += 1
    end
end

tr, _ = generate(foo, ())
println("Sample from original.")
display(get_choices(tr))

# Say, we observe (:x => 4) = 3.0
obs = choicemap((:x => 4, 3.0))

tr, _ = generate(proposal_foo, (), obs)
println("Sample from proposal.")
display(get_choices(tr))

# Importance sampling
"""
(traces, log_norm_weights, lml_est) = importance_sampling(model::GenerativeFunction,
    model_args::Tuple, observations::ChoiceMap,
    proposal::GenerativeFunction, proposal_args::Tuple,
    num_samples::Int, verbose=false)
"""

trs, _, _ = importance_sampling(foo, (), obs, 5)
println("Traces from importance sampling from prior.")
map(x -> display(get_choices(x)), trs)


trs, _, _ = importance_sampling(foo, (), obs, proposal_foo, (), 5)
println("Traces from importance sampling from proposal_foo.")
map(x -> display(get_choices(x)), trs)

# --------------------------- #

@gen function bar()
    arr = Array{Float64, 1}([])
    x = @trace(normal(0.0, 1.0), :x)
    push!(arr, x)
    if x < 1.0
        y = @trace(normal(5.0, 10.0), :y)
        push!(arr, y)
    end
    counter = 0
    while @trace(geometric(0.5), :geo => counter) < 10
        push!(arr, @trace(normal(0.0, 10.0), :while => counter))
        counter += 1
    end
    return arr
end

#tr, _ = generate(bar, ())
#display(get_choices(tr))
#
end
