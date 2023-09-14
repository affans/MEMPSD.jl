using MEMPSD 
using Test
using Random
using Distributions
using DelimitedFiles

# copy paste this into the main module for testing interactively when developing
function compare_with_memspectrum() 
    data = readdlm("./src/data_gen_memspec.csv")[:, 1] # since file is read as a matrix 
    tsd = TimeSeriesData(data)
    reflection_coefficients = burg(tsd)
    PSD = compute_PSD(tsd)
    (;prediction_error_coefficients, ar_coefficients, scale_factor, optimal_order) = PSD
    println("sum of reflection coefficients: $(sum(reflection_coefficients))")
    println("first three aks: $(prediction_error_coefficients[1:3])")
    println("length ar: $(length(ar_coefficients))")
    println("optimal oprder: $(optimal_order)")

    ff = forecast(tsd, PSD, 10, 5, false)
    display(ff) 

    println("compute spectrum") 
    freq, spec = spectrum(tsd, PSD)
    #println("scale factors: sum: $(sum(scale_factors)) data: $(scale_factors[1:4])")
    #println("errors: sum: $(sum(errors)) data: $(errors[1:4])")
    #println("optimal order: $(optimal_order),  error at optimized order: $(errors[optimal_order])")
end

println("$(pwd())")


# TO BE DONE 

# OLD:
# # lets generate data 
# Random.seed!(5742);
# N, dt = 1000, 0.01 
# time = range(start=0, length=N, step=dt)
# frequency = 2 
# white_noise = rand(Normal(0.4, 1), N)
# data = @. sin(2 * π * frequency * time) + white_noise

# # save data to file to compare with the Python package later

# # run the algorithm 
# P, aₖ, optimized_order, MAX_AR_ORDER, fullP, full_aks  = MEMPSD._burg(data);
# spec, freq = spectrum(P, aₖ, 1, N=1000)

# @testset "Burg Algorithm" begin
#     @test P ≈ 0.8941840658946688
#     @test sum(aₖ) ≈ 0.09609368952560363
#     @test MAX_AR_ORDER == Int(round(2 * N / log(2 * N)))
#     @test MAX_AR_ORDER == 263
#     @test length(full_aks) == MAX_AR_ORDER + 1
#     @test optimized_order == 56

#     println("""

#     Debug info
#         P (variance of white noise): $P
#         sum of coefficients: $(sum(aₖ))
#         optimized AR order: $(optimized_order) 
#         maximum AR order: $(MAX_AR_ORDER) 
#     """)
# end

# @testset "PSD Calculation" begin
#     @test sum(spec) ≈ 1611.5080942985446
#     @test sum(freq) ≈ -0.5
# end 