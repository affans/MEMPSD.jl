module MEMPSD
    using LinearAlgebra
    using Statistics
    using FFTW 
    using Distributions
    using Random
    using DelimitedFiles 

    export burg, PSD, forecast, spectrum, compute_PSD_coefficients
    
    poplast!(x) = pop!(x) # just for consistency with popfirst!

    """
        TimeSeriesData(x::Vector{<:Real})

    An object representing the input time series for which the Power Spectral Density is to be computed. 
    The object contains the input time series `x`, the length of the time series `N`, and the maximum
    order (upper limit) of the PSD given by 2N/ln(2N), which is guided from numerical experiments.
    """
    struct TimeSeriesData 
        x::Vector{<:Real}
        N::Int64 
        M::Int64
    end
    function TimeSeriesData(x::Vector{<:Real}) 
        N = length(x) 
        M = Int(round(2 * N / log(2 * N))) # The upper limit of the number of prediction error coefficients (and therefore the AR coefficients) 
        return TimeSeriesData(x, N, M)
    end
    export TimeSeriesData

    """
        PSD 

    References: 
    [1] Martini, Alessandro, Stefano Schmidt, and Walter Del Pozzo. “Maximum Entropy Spectral Analysis: A Case Study.” arXiv, June 17, 2021. http://arxiv.org/abs/2106.09499.
    [2] Vos, Koen. “A Fast Implementation of Burg’s Method,” 2013. https://svn.xiph.org/websites/opus-codec.org/docs/vos_fastburg.pdf.
    [3] Woodcock, Rebecca, Hussam Muhamedsalih, Haydn Martin, and Xiangqian Jiang. “Burg Algorithm for Enhancing Measurement Performance in Wavelength Scanning Interferometry.” Surface Topography: Metrology and Properties 4, no. 2 (February 19, 2016): 024003. https://doi.org/10.1088/2051-672X/4/2/024003.
    """
    struct PSD 
        prediction_error_coefficients::Vector{<:Real}    
        ar_coefficients::Vector{<:Real}
        scale_factor::Float64
        optimal_order::Int64
        function PSD(p, a, s, o) 
            new(p, a, s, o)
        end 
    end 
    export PSD
    
    """
        burg(::TimeSeriesData)

    Compute the reflection coefficients of an input time series wrapped in a `TimeSeriesData` object.

    Uses the Burg Algorithm to compute the reflection coefficients that are used to calculate the 
    prediction error vector and the scale multiplicative factor. The prediction error vector 
    and the scale multiplicative factor are used to determine the Power Spectral Densitry of the time series.
    
    The number of coefficients (i.e. order of the method) is determined by `TimeSeriesData.M`

    Returns an array of the reflection coefficents. 
    """
    function burg(tsd::TimeSeriesData) 
        # The reflection coefficients 
        # (see eqn 16 and following paragraph in the paper)
        # The reflection coefficients indicate the time dependence between y(n) and y(n – k) 
        # after subtracting the prediction based on the intervening k – 1 time steps.
        x = tsd.x
        M = tsd.M

        # initialize with forward and backwards errors  and reflection coefficients
        f = copy(x) 
        b = copy(x)
        k = zeros(Float64, M)  # there will be M total reflection coefficients. 
    
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
        compute_PSD(::TimeSeriesData) 

    This function computes the power spectral density of the time series data (wrapped in the `TimeSeriesData` object). 
    It returns a `PSD` object that contains the prediction error coefficients, the AR coefficients, the scale factor, and the 
    optimal order of the AR model. 

    The prediction error coefficients are computed by applying the Levinson Recursion on the reflection coefficients determined 
    by the Burg Algorithm (see `burg()`). The burg algorithm computes reflection coefficients with a default order (i.e. length) 
    of 2N/ln(2N). This is just a plausible upper limit on the order of the AR process.
    
    The optimal order is  determined by minimizing the Akaike Final Prediction Error criterion. Other loss functions exist as well 
    (see reference). Once the optimal order is determined, the correct length vector of the prediction error coefficients and the AR coefficients 
    are returned. 

    The returned object `PSD` can be used as an input to the `forecast()` function to forecast the time series data.
    """
    function compute_PSD_coefficients(tsd::TimeSeriesData) 
        # The reflection coefficients from the Burg Algorithm 
        # are used to compute the vector of prediction errors 
        # The vector of prediction errors are the partial autocorrelation coefficients scaled by –1 ??    
        rc = burg(tsd) 

        (; M, x, N) = tsd;
        
        # initialize the prediction error filters [1, a1, a2, ... a_M] (so M+1 elements in total)
        # at order 1: we calculate [1, a1]
        # at order 2: we calculate [1, a1, a2] BUT THE a1 gets updated as well !! 
        # at order 3: we calculate [1, a1, a2, a3] BUT a1 and a2 get updated as well !!
        # we do this way because we don't what the optimized order would be, certainly not M)
        # and so we want to pick the right prediction error filter for the optimized order (we just select that particular array)
        # See paper by Martini, Alessandro for how the prediction error filter is calculated using the reflection coefficeints
        ak = [ones(Float64, i) for i = 1:(M+1)]  # M+1 because the prediction error filter has form [1, a1, a2, ... a_M]

        # the multiplicative scale factors 
        # again, we calculate it at every order (but then pick the right one for the optimized order)
        scale_factors = zeros(Float64, M + 1) # M+1 because its [1, a1, a2, ... a_M] 
        scale_factors[1] = var(x, corrected=false) 
        errors = zeros(Float64, M)

        for i = 1:M
            # apply Levinson Recursion 
            p = ak[i] # get the previous prediction error filter
            b₊ = [p..., 0] # append a zero to it
            b₋ = reverse(b₊) # reverse it 
            _v = (b₊ .+ (rc[i] * b₋)) # update all the coefficients (this returns in a vector with one extra element +  has previous coefficeints updated)
            ak[i + 1] .= _v # update the prediction error filter with the new coefficient

            # at the same time, we can also calculate the multiplicative scale factor 
            _p = scale_factors[i] * (1 - (rc[i] * conj(rc[i]))) 
            scale_factors[i+1] = _p 
            errors[i] = _p * (N + i + 1) / (N - i - 1) # computes the AIC IF the max order was i 
        end
        optimal_order = argmin(errors)

        ak_optimal = ak[optimal_order]
        sf_optimal = scale_factors[optimal_order]
        ar_coefficients = -reverse(ak_optimal[2:end])
        PSD(ak_optimal, ar_coefficients, sf_optimal, optimal_order)
        #return ak, scale_factors, errors, optimal_order
    end

    """
        forecast(tsd::TimeSeriesData, psd::PSD, forecast_length, [num_of_simulations = 1000, add_noise = true])

    Uses an autoregressive model to forecast an observed time-series for a total number of points 
    given by `forecast_length`. Requires input data wrapped in a `TimeSeriesData` object and the
    power spectral density `PSD` object (returned by `compute_PSD()`). 

    Returns a matrix array containing the forecasted points over `num_of_simulations` realizations
    """ 
    function forecast(tsd::TimeSeriesData, psd::PSD, forecast_length, num_of_simulations = 1000, add_noise = true)
        (;prediction_error_coefficients, ar_coefficients, scale_factor, optimal_order) = psd
        
        M = optimal_order - 1 # since ar coefficients don't include the leading 1 from the prediction error coefficients
        data = tsd.x
        coeff = ar_coefficients
        @assert M > 0; @assert length(data) > M 
    
        prediction_matrix = zeros(Float64, M + forecast_length, num_of_simulations)
       
        #initialize the prediction vector with p data points (since we need atleast p historic points for AR model)
        for i = 1:num_of_simulations
            prediction_matrix[1:M, i] .= data[(end - M + 1):end]
        end

        #display(" pred size: $(size(prediction_matrix)), P $P")
        # for the random noise with P as the variance of the AR model
        Random.seed!(123)
        ND = Normal(0, sqrt(scale_factor))

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
        spectrum(tsd::TimeSeriesData, psd::PSD)     

    Compute the Power Spectral Density of the time series `x`.

    Computes the power spectral density of the PSD model. Default returns power 
    spectral density and frequency array automatically computed by sampling theory. 
    
    Returns `(; spec, freq)` where `spec` is the PSD of the model, including both positive 
    and negative frequencies and `freq`, the frequencies at which spectrum is evaluated 
    (as provided by `fftfreq`) 
    """
    function spectrum(tsd::TimeSeriesData, psd::PSD, dt=1, onesided=true) 
        #pred_filter, P, optim_order = compute_prediction_effor_coefficients(x)
        (;prediction_error_coefficients, ar_coefficients, scale_factor, optimal_order) = psd
        N = tsd.N
        coeff = vcat(prediction_error_coefficients, zeros(N-length(prediction_error_coefficients))) # pad the prediction error coefficients
        den = fft(coeff)
        spec = @. (scale_factor * dt) / abs(den)^2
        freq = fftfreq(N, 1/dt) # second arg here is the sampling rate (1/dt)
        
        println("""
        DEBUG SPECTRUM
            sum ak: $(sum(prediction_error_coefficients))
            sum coeff: $(sum(coeff))
            sum den: $(sum(den))
            length den: $(length(den))
            sum spec: $(sum(spec))
        """)

        if onesided 
            spec = spec[1:(N ÷ 2)] .* sqrt(2)
            freq = freq[1:(N ÷ 2)]
        end
        return (;freq, spec)
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
