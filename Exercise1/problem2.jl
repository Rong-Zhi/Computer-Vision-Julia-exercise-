using Images
using PyPlot
using JLD
using Base.Test

# Transfrom from Cartesian to homogeneous coordinates
function cart2hom(points::Array{Float64,2})
  L=size(points,2)
  points_hom=[points;ones(1,L)]
  return points_hom::Array{Float64,2}
end

# Transfrom from homogeneous to Cartesian coordinates
function hom2cart(points::Array{Float64,2})
  N=size(points,1)
  points_cart=Array(Float64,N-1,size(points,2))
  for m in 1:(N-1)
    points_cart[m,:]=points[m,:]./points[N,:]
  end
  return points_cart::Array{Float64,2}
end

# Translation by v
function gettranslation(v::Array{Float64,1})
  T=eye(4)
  T[1:3,4]=v
  return T::Array{Float64,2}
end

# Rotation of d degrees around x axis
function getxrotation(d::Int)
  r=deg2rad(d)
  Rx=[1 0 0 0;0 cos(r) -sin(r) 0;0 sin(r) cos(r) 0;0 0 0 1]
  return Rx::Array{Float64,2}
end

# Rotation of d degrees around y axis
function getyrotation(d::Int)
  r=deg2rad(d)
  Ry=[cos(r) 0 sin(r) 0;0 1 0 0;-sin(r) 0 cos(r) 0;0 0 0 1]
  return Ry::Array{Float64,2}
end

# Rotation of d degrees around z axis
function getzrotation(d::Int)
  r=deg2rad(d)
  Rz=[cos(r) -sin(r) 0 0;sin(r) cos(r) 0 0;0 0 1 0;0 0 0 1]
  return Rz::Array{Float64,2}
end

# Central projection matrix
function getprojection(principal::Array{Int,1}, focal::Int)
  P=[focal 0 principal[1] 0;0 focal principal[2] 0;0 0 1 0]
  P=convert(Array{Float64}, P)
  return P::Array{Float64,2}
end

# Return full projection matrix C and full model transformation matrix M
function getfull(T::Array{Float64,2},Rx::Array{Float64,2},Ry::Array{Float64,2},Rz::Array{Float64,2},V::Array{Float64,2})
  M=Rx*Ry*Rz*T
  C=V*M
  return C::Array{Float64,2},M::Array{Float64,2}
end

# Load 2D points
function loadpoints()
  points=load("../data-julia/obj_2d.jld","x")
  return points::Array{Float64,2}
end

# Load z-coordintes
function loadz()
  z=load("../data-julia/zs.jld","Z")
  return z::Array{Float64,2}
end

# Invert just the central projection P of 2d points *P2d* with z-coordinates *z*
function invertprojection(P::Array{Float64,2}, P2d::Array{Float64,2}, z::Array{Float64,2})
  P3d=P[:,1:3]\(cart2hom(P2d).*z)
  return P3d::Array{Float64,2}
end

# Invert just the model transformation of the 3D points *P3d*
function inverttransformation(A::Array{Float64,2}, P3d::Array{Float64,2})
  X=A\cart2hom(P3d)
  return X::Array{Float64,2}
end

# Plot 2D points
function displaypoints2d(points::Array{Float64,2})
  figure()
  plot(points[1,:],points[2,:],".b")
  title("2D Display")
  xlabel("X")
  ylabel("Y")
  return gcf()::Figure
end

# Plot 3D points
function displaypoints3d(points::Array{Float64,2})
  figure()
  scatter3D(points[1,:],points[2,:],points[3,:],".r")
  title("3D Display")
  xlabel("X")
  ylabel("Y")
  zlabel("Z")
  return gcf()::Figure
end

# Apply full projection matrix *C* to 3D points *X*
function projectpoints(C::Array{Float64,2}, X::Array{Float64,2})
  X_p=C*cart2hom(X)
  X_c=hom2cart(X_p)
  displaypoints2d(X_c)
  return gcf()::Figure
end


#= Problem 2
Projective Transformation =#

function problem2()
  # parameters
  t               = [-27.1; -2.9; -3.2]
  principal_point = [8; -10]
  focal_length    = 8

  # model transformations
  T = gettranslation(t)
  Ry = getyrotation(135)
  Rx = getxrotation(-30)
  Rz = getzrotation(90)

  # central projection
  P = getprojection(principal_point,focal_length)

  # full projection and model matrix
  C,M = getfull(T,Rx,Ry,Rz,P)

  # load data and plot it
  points = loadpoints()
  displaypoints2d(points)

  # reconstruct 3d scene
  z = loadz()
  Xt = invertprojection(P,points,z)
  Xh = inverttransformation(M,Xt)
  worldpoints = hom2cart(Xh)
  displaypoints3d(worldpoints)

  # reproject points
  points2 = projectpoints(C,worldpoints)
  #displaypoints2d(points2)

  #@test_approx_eq points points2
  return
end
problem2()
