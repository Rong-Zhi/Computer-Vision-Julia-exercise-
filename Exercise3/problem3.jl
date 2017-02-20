using Images
using PyPlot
using JLD

include("Common.jl")


# Compute fundamental matrix from homogenous coordinates
function eightpoint(x1::Array{Float64,2},x2::Array{Float64,2})

  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end

# Apply conditioning.
# Return the conditioned points and the condition matrix.
function condition(points::Array{Float64,2})

  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end

# Compute the fundamental matrix for given conditioned points
function computefundamental(p1::Array{Float64,2},p2::Array{Float64,2})

  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end

# Enforce that the given matrix has rank 2
function enforcerank2(Ft::Array{Float64,2})

  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end

# Draw epipolar lines through given points on top of an image
function showepipolar(F::Array{Float64,2},points::Array{Float64,2},im::Array{Float32,2})

  return nothing::Void
end

# Compute the residuals of the fundamental matrix F
function computeresidual(p1::Array{Float64,2},p2::Array{Float64,2},F::Array{Float64,2})

  return residual::Array{Float64,2}
end


# Problem 3: Fundamental Matrix

function problem3()
  # load images and data
  im1 = PyPlot.imread("../data-julia/a3p2a.png")
  im2 = PyPlot.imread("../data-julia/a3p2b.png")
  points1 = load("../data-julia/points.jld", "points1")
  points2 = load("../data-julia/points.jld", "points2")

  # display images and correspondences
  figure()
  subplot(121)
  imshow(im1,"gray",interpolation="none")
  axis("off")
  scatter(points1[:,1],points1[:,2])
  subplot(122)
  imshow(im2,"gray",interpolation="none")
  axis("off")
  scatter(points2[:,1],points2[:,2])
  gcf()

  # compute fundamental matrix with homogenous coordinates
  x1 = Common.cart2hom(points1')
  x2 = Common.cart2hom(points2')
  F = eightpoint(x1,x2)

  # draw epipolar lines
  figure()
  subplot(121)
  showepipolar(F',points2,im1)
  scatter(points1[:,1],points1[:,2])
  subplot(122)
  showepipolar(F,points1,im2)
  scatter(points2[:,1],points2[:,2])
  gcf()

  # check epipolar constraint by computing the remaining residuals
  residual = computeresidual(x1, x2, F)
  println("Residuals:")
  println(residual)

  # compute epipoles
  U,_,V = svd(F)
  e1 = V[1:2,3]./V[3,3]
  println("Epipole 1: $(e1)")
  e2 = U[1:2,3]./U[3,3]
  println("Epipole 2: $(e2)")

  return
end
