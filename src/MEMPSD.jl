module MEMPSD
    using LinearAlgebra
    using Statistics
    using FFTW 
    using Distributions
    using Random

    export burg, compute_optimized_order, compute_prediction_effor_coefficients, PSD, forecast
    
    poplast!(x) = pop!(x) # just for consistency


    """
        burg(x::AbstractArray{<:Real})

    Compute the reflection coefficients of the time series `x`.

    Uses the Burg Algorithm to compute the reflection coefficients used to calculate the 
    *prediction error vector* of the power spectral desntiy of the time series `x`. 
    The order of the method (i.e. the number of coefficients) is given by 2N/ln(2N) 
    where N is the number of observed points. In future iterations, the order will be 
    selected by the minimization of a chosen loss method.

    Return an array of the reflection coefficents. 
    """
    function burg(x::AbstractArray{<:Real}) 
        # The reflection coefficients 
        # (see eqn 16 and following paragraph in the paper)
        # The reflection coefficients indicate the time dependence between y(n) and y(n – k) 
        # after subtracting the prediction based on the intervening k – 1 time steps.
        N = length(x) 

        # Maximum number of recursions for the computation of the  power spectral density.
        # i.e. choice of the autoregressive model 
        # Maximum autoregressive order by numerical experiments m = 2N / log(2N) 
        M = Int(round(2 * N / log(2 * N))) # MAX ORDER (not to be confused with the max AR order?)

        # initialize with forward and backwards errors  and reflection coefficients
        f = copy(x) 
        b = copy(x)
        k = zeros(Float64, M)

        # initialize the prediction error coefficients 
        # i.e. the AR coefficients 
        a = zeros(Float64, M)
        a[1] = 1

        for i = 1:M
            # remove the first and last element of b
            popfirst!(f)
            poplast!(b)

            # calculate the reflection coefficient
            num = -2 * dot(f, b)
            den = dot(f, f) + dot(b, b)
            k[i] = num / den
        
            # update the filter errors 
            # f = f + k*b 
            # b = b + k*f <== does not work since f gets updated in the line before. 
            for idx in eachindex(f,b)
                fx = f[idx]
                bx = b[idx]
                f[idx] = fx + k[i]*bx
                b[idx] = bx + k[i]*fx
            end
        end
        return k
    end
   

    """
        compute_optimized_order() 

    The burg algorithm computes reflection coefficients with a default order (i.e. length) 
    of 2N/ln(2N). This is just a plausible upper limit on the order of the AR process.
    An optimal algorithm could employ fewer points. This function uses the following loss functions to determine an optimal model order.

    - Akaike Final Prediction Error. The Final Prediction Error is given by FPE(m) = E[((orig(xt) - predicted(xt))^2]. 
    Minimizing the FPE is equivalent to minimizing the quantity 
    L(m) = Pm (N + M + 1)/(N - M - 1)

    Inputs require data vector `x::Vector` and vector of coefficients `k::Vector` 
    computed from the Burg algorithm
    """
    function compute_optimized_order(x, reflection_coeff)
        N = length(x) 
        k = reflection_coeff 
        M = length(k) # max recursion order from Burg algorithm, will be 2N / log(2N) 
        
        P = zeros(length(k)+1)
        P[1] = var(x, corrected=false)
        aic = zeros(Float64, M)

        for i in 1:M 
            wn_var = P[i] * (1 - (k[i] * conj(k[i]))) 
            P[i+1] = wn_var
            aic[i] = wn_var * (N + i + 1) / (N - i - 1) # computes the AIC IF the max order was i 
        end
        optimal_order = argmin(aic)
        return optimal_order, P[optimal_order], P, aic
    end
    export compute_optimized_order

    """
        compute_prediction_effor_coefficients() 

    Computes the AR coefficients using the Akaike Final Prediction Error criterion. 
    
    The Final Prediction Error is given by FPE(m) = E[((orig(xt) - predicted(xt))^2]. 
    Minimizing the FPE is equivalent to minimizing the quantity L(m) = Pm (N + M + 1)/(N - M - 1)

    Inputs require data vector `x::Vector` and vector of coefficients `k::Vector` 
    computed from the Burg algorithm
    """
    function compute_prediction_effor_coefficients(x, optimized=false) 
        # The reflection coefficients from the Burg Algorithm 
        # are used to compute the vector of prediction errors 
        # The vector of prediction errors are the partial autocorrelation coefficients scaled by –1. 
        
        reflection_coeff = burg(x)
        optim_order = length(reflection_coeff)  # will be 2N / log(2N) 
        P = var(x, corrected=false) 
        if optimized
            optim_order, P, _ = compute_optimized_order(x, reflection_coeff)
        end  
        pred_filter = zeros(Float64, optim_order) # +1 since 1.0 is the default value.
        pred_filter[1] = 1.0 
        for i = 1:(optim_order-1)
            _a = pred_filter[1:i]
            push!(_a, 0) # append a 0 for Levinson Recursion

            # another way to do this is set pred_filter[i+1] = 0; _a = @view pred_filter[1:(i+1)]
            # but only reduces allocations marginally 
            #pred_filter[i+1] = 0; 
            #_a = @view pred_filter[1:(i+1)]

            _t = _a .+ (reflection_coeff[i] * reverse(_a)) # levinson recursion formula
            pred_filter[1:(i+1)] .= _t # update the whole prediction vector for the next order i 
        end

        # return the AR coefficients and the variance of the white noise
        return pred_filter, P, optim_order
    end

    function get_ar_coefficients(pred_filter) 
        # the ar coefficients are the reverse negative of the prediction error coefficients 
        # from https://arxiv.org/abs/2106.09499
        ar_coeff = -reverse(pred_filter[2:end]) 
    end


    """
        forecast(data, forecast_length, [num_of_simulations = 1000, add_noise = true])

    Uses an AR(p) model to forecast an observed time-series for a total number of points 
    given by `forecast_length`. Requires input data `data`.

    Returns a matrix array containing the forecasted points over `num_of_simulations` realizations
    """ 
    function forecast(data, forecast_length, num_of_simulations = 1000, add_noise = true)
        pred_filter, P, optim_order = compute_prediction_effor_coefficients(data, true)
        coeff = get_ar_coefficients(pred_filter)
        
        M = length(coeff)
        @assert M > 0; @assert length(data) > M 
    
        prediction_matrix = zeros(Float64, M + forecast_length, num_of_simulations)
    
        #initialize the prediction vector with p data points (since we need atleast p historic points for AR model)
        for i = 1:num_of_simulations
            prediction_matrix[1:M, i] .= data[(end - M + 1):end]
        end

        display("x pred size: $(size(prediction_matrix)), P $P")
        # for the random noise with P as the variance of the AR model
        Random.seed!(123)
        ND = Normal(0, sqrt(P))

        for sim = 1:num_of_simulations
            prediction_vector = prediction_matrix[:, sim]
            for i = 1:forecast_length # points to forecast
                dvalue = dot(prediction_vector[i:(M + i - 1)], coeff)
                if add_noise 
                    dvalue = dvalue + rand(ND)
                end
                prediction_vector[M + i] = dvalue
            end
            prediction_matrix[:, sim] .= prediction_vector
        end        

        # the prediction matrix has some of the original data used 
        # remove this from the output to have a clean prediction_matrix
        return prediction_matrix[(M+1):end, :]
    end

    """
        PSD(x::AbstractVector)

    Compute the Power Spectral Density of the time series `x`.

    Computes the power spectral density of the model. Default returns power 
    spectral density and frequency array automatically computed by sampling theory. 
    Method requires inputs `P` (the variance of the white noise),  aₖ (the autoregressive coefficients 
    found using Burg's Algorithm), and `dt` (the sampling rate for the time series). Keyword arguments 
    include `N` (the length of the frequency grid) 

    Returns `(; spec, freq)` where `spec` is the PSD of the model, including both positive 
    and negative frequencies and `freq`, the frequencies at which spectrum is evaluated 
    (as provided by `fftfreq`) 
    """
    function PSD(x::AbstractArray{<:Real})  
        dt = 0.1
        pred_filter, P, optim_order = compute_prediction_effor_coefficients(x)
        N = length(x)
        coeff = vcat(pred_filter, zeros(N-length(pred_filter))) # pad the prediction error coefficients
        den = fft(coeff)
        spec = @. (P * dt) / abs(den)^2
        freq = fftfreq(N, dt)
        return (;spec, freq)
    end
    
    """
        autocovariance(spec, freq[, normalize=false])

    Compute the autocovariance of the data based on the autoregressive coefficients.
    The autocovariance is defined as: C(tau) = E_t[(x_t -mu)(x_t+tau -mu)]
    """
    function autocovariance(spec, freq, normalize=false) 
        error("not implemented yet")
    end

    function generate_data(N = 1000, dt = 0.01, freq = 2, wn = false) 
        # lets generate data 
        Random.seed!(5742);
        time = range(start=0, length=N, step=dt)
        white_noise = zeros(Float64, N)
        if wn 
            white_noise = rand(Normal(0.4, 1), N)
        end
        data = @. sin(2 * π * freq * time) + white_noise
        return data
    end
end
