using InverseModeling, View5D, Noise
using BenchmarkTools

function test_gaussfit()
    # true_val = (i0 = 10.0, σ=Fixed([5.0, 6.0]), μ=[1.2, 3.1], offset=1.0) # Fixed()
    true_val = (i0 = 10.0, σ=[5.0, 6.0], μ=[1.2, 3.1], offset=1.0) # Fixed()
    sz = (64, 64)
    forward, fit_parameters, get_fit_results, fit_fct = gauss_model(sz, true_val)
    @time perfect = forward(true_val); # 76 alloc 20 kB
    @btime perfect = forward($true_val); # 9 µs
    @time meas = Float32.(poisson(Float64.(perfect)));
    # meas = forward(true_val)
    # start_val = (i0 = 10, σ = [2.2180254609037426, 3.4734027671622583], μ = [0.5654755165137773, 1.4606503074960915], offset = 1.0)
    @btime res, res_img, resv = gauss_fit($meas, verbose=false); # 35 ms, old way: 10s, 
    # start_val = (i0 = 10, σ=[2.0, 2.0], μ=[1.8, 2.5], offset=0.0 ) # Fixed()
    # res, res_img, resv = gauss_fit(meas, start_val, verbose=true);
    @time res, res_img, resv = gauss_fit(meas, verbose=false); # 166k alloc 27 Mb

    @ve meas, res_img, (meas .- res_img)
end

function code_stability_test()
    true_val = (i0 = 10f0, σ=[5f0, 6f0], μ=[1.2f0, 3.1f0], offset=1f0) # Fixed()
    sz = (32, 32)
    forward, fit_parameters, get_fit_results, fit_fct = gauss_model(sz, true_val)
    forward(true_val)
    @code_warntype forward(true_val)
    fwd = fit_fct
    fit_params, fixed_params, get_fit_results, stripped_params = InverseModeling.prepare_fit(true_val, Float32) #

    function forward2(fit_params)::Array{Float32,2}
        function g(id) # an accessor function for the parameters
            InverseModeling.get_fwd_val(stripped_params[id], id, fit_params, fixed_params)
        end
        return fwd(g) # calls fwd giving it the parameter-access function g
    end

    forward2(true_val)
    @code_warntype forward2(true_val)
end
