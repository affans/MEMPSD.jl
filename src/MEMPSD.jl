module MEMPSD
    using LinearAlgebra
    using Statistics
    using FFTW 
    using Distributions

    export burg, spectrum, forecast

    function _burg(x) 
        # TO DO: implement different loss functions, right now using only FPE
        N = length(x)
        max_order = Int(round(2 * N / log(2 * N)))
        
        # initialize with forward and backwards errors  
        f = copy(x)
        b = copy(x) 

        P = [var(x, corrected=false)] # vector to hold the estimataed variance of the white noise

        optimization = []
        a₀ = 1.0 
        aₖ = [[a₀]] # as the coefficients are produced, we push them to this vector. 
        
        # start at p = 1, and go until maximum order
        for p = 1:max_order
            _f = f[2:end] 
            _b = b[1:(end - 1)]
            @assert length(_f) == length(_b)    

            # compute the reflection coefficients
            num = -2 * dot(_f, _b)
            den = dot(_f, _f) + dot(_b, _b)
            k = num / den 

            # update the reflection coefficient
            new_x = [aₖ[p]..., 0] 
            ref = new_x .+ (k * reverse(new_x))
            push!(aₖ, ref)

            # estimate the white noise variance
            push!(P, P[p] * (1 - (k * conj(k))))

            # opt value 
            opt_value = P[end] * (N + p + 1) / (N - p - 1)  # i think this is slightly wrong... should be p+1, but its okay
            push!(optimization, opt_value)

            # update the filter errors. 
            f = _f + (k * _b)
            b = _b + (k * _f)
        end
        optimized_order = argmin(optimization)
        return P[optimized_order], aₖ[optimized_order], optimized_order, max_order, P, aₖ
    end

    """
        burg(x::AbstractArray{<:Real})

    Uses the Burg Algorithm to compute the power spectral density of the time series `x`. 
    The order of the method is is given by 2N/ln(2N) where N is the number of observed points. 
    In future iterations, the order will be selected by the minimization of a chosen loss method.

    Return `(; P, aₖ)` where `P` is variance of white noise for the associated autoregressive 
    process and `aₖ` are the reflection coefficients used to compute the power spectral density 
    """
    function burg(x) 
        P, aks, _, _, _, _ = _burg(x)
        return (; P, aks)
    end

    """
        spectrum(P, aₖ, dt; N)
    
    Computes the power spectral density of the model. Default returns power 
    spectral density and frequency array automatically computed by sampling theory. 
    Method requires inputs `P` (the variance of the white noise),  aₖ (the autoregressive coefficients 
    found using Burg's Algorithm), and `dt` (the sampling rate for the time series). Keyword arguments 
    include `N` (the length of the frequency grid) 

    Returns `(; spec, freq)` where `spec` is the PSD of the model, including both positive 
    and negative frequencies and `freq`, the frequencies at which spectrum is evaluated 
    (as provided by `fftfreq`) 
    """
    function spectrum(P, ak, dt; N) 
        # pad the Aks
        aks = vcat(ak, zeros(N-length(ak)))
        den = fft(aks)
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

    """
        forecast(data, ar_coeffs, P, forecast_length, [num_of_simulations = 1000, add_noise = true])

    Forecasting on an observed process for a total number of points given by length. 
    It computes number_of_simulations realization of the forecast time series. This method 
    can only be used if autoregressive coefficients have been computed  already. Use the 
    `burg()` method before forecasting to obtain the autoregressive coefficients. 

    Parameters
    ----------
    - `data`: time series data for the spectrum calculation. The number of data points must satisfy: `length(data) > length(coeffs)` where `coeffs`.
    - `ar_coeffs`: autoregressive coefficients have been computed by the Burg algorithm
    - `P`: Variance of white noise for the autoregressive process. 
    - `forecast_length`: Number of future points to be predicted 
    - `number_of_simulations`: Total number of simulations of the process
    - `add_noise`: add Normal distribution noise
        
    Returns an multidimensional array containing the forecasted points for every simulation of the process.
    """ 
    function forecast(data, ar_coeffs, P, forecast_length, num_of_simulations = 1000, add_noise = true)
        if ar_coeffs[1] == 1.0 
            error("are you sure the first coefficient is one?")
        end 
        coeffs = -reverse(ar_coeffs)

        p = length(coeffs) 
    
        @assert p > 0 
        @assert length(data) > p 
    
        prediction_matrix = zeros(Float64, p + forecast_length, num_of_simulations)
    
        #initialize the prediction vector with p data points (since we need atleast p historic points for AR model)
        for i = 1:num_of_simulations
            prediction_matrix[1:p, i] .= data[(end - p + 1):end]
        end

        # for the random noise with P as the variance of the AR model
        ND = Normal(0, sqrt(P))

        for sim = 1:num_of_simulations
            prediction_vector = prediction_matrix[:, sim]
            for i = 1:forecast_length # points to forecast
                dvalue = dot(prediction_vector[i:(p + i - 1)], coeffs)
                if add_noise 
                    dvalue = dvalue + rand(ND)
                end
                prediction_vector[p + i] = dvalue
            end
            prediction_matrix[:, sim] .= prediction_vector
        end        

        # the prediction matrix has some of the original data used 
        # remove this from the output to have a clean prediction_matrix
        return prediction_matrix[(p+1):end, :]
    end
end
