using MEMPSD 
using Test
using Random
using Distributions
using DelimitedFiles

# define the data 
println("pwd: $(pwd())")
data = readdlm("../test_data/data_gen_memspec.csv")[:, 1] # since file is read as a matrix 
tsd = TimeSeriesData(data)

reflection_coefficients = burg(tsd)
PSD = compute_PSD_coefficients(tsd)
(;prediction_error_coefficients, ar_coefficients, scale_factor, optimal_order) = PSD

@testset "PSD Coefficients" begin
    @test sum(reflection_coefficients) ≈ -3.792924175071  rtol=1e-8
    @test sum(prediction_error_coefficients) ≈ 0.06860025212718604 rtol=1e-8
    @test optimal_order == 58 
    @test scale_factor ≈ 0.9011564144027354 rtol=1e-8
end

@testset "Forecast" begin
    println("to do")
end

@testset "Spectrum" begin
    println("to do")
end



