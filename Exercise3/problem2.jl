using Images
using PyPlot
using Grid

include("Common.jl")

# Load Harris interest points of both images
function loadkeypoints(path::ASCIIString)

  @assert size(keypoints1,2) == 2 # Nx2
  @assert size(keypoints2,2) == 2 # Kx2
  return keypoints1::Array{Int64,2}, keypoints2::Array{Int64,2}
end

# Compute pairwise squared euclidean distances for given features
function euclideansquaredist(f1::Array{Float64,2},f2::Array{Float64,2})

  @assert size(D) == (size(f1,2),size(f2,2))
  return D::Array{Float64,2}
end

# Find pairs of corresponding interest points based on the distance matrix D.
# p1 is a Nx2 and p2 a Mx2 vector describing the coordinates of the interest
# points in the two images.
# The output should be a min(N,M)x4 vector such that each row holds the coordinates of an
# interest point in p1 and p2.
function findmatches(p1::Array{Int,2},p2::Array{Int,2},D::Array{Float64,2})

  @assert size(pairs) == (min(size(p1,1),size(p2,1)),4)
  return pairs::Array{Int,2}
end

# Show given matches on top of the images in a single figure, in a single plot.
# Concatenate the images into a single array.
function showmatches(im1::Array{Float32,2},im2::Array{Float32,2},pairs::Array{Int,2})

  return nothing::Void
end

# Compute the estimated number of iterations for RANSAC
function computeransaciterations(p::Float64,k::Int,z::Float64)

  return n::Int
end

# Randomly select k corresponding point pairs
function picksamples(points1::Array{Int,2},points2::Array{Int,2},k::Int)

  @assert size(sample1) == (k,2)
  @assert size(sample2) == (k,2)
  return sample1::Array{Int,2},sample2::Array{Int,2}
end

# Apply conditioning.
# Return the conditioned points and the condition matrix.
function condition(points::Array{Float64,2})

  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end

# Estimate the homography from the given correspondences
function computehomography(points1::Array{Int,2}, points2::Array{Int,2})

  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end

# Compute distances for keypoints after transformation with the given homography
function computehomographydistance(H::Array{Float64,2},points1::Array{Int,2},points2::Array{Int,2})

  @assert length(d2) == size(points1,1)
  return d2::Array{Float64,2}
end

# Compute the inliers for a given homography distance and threshold
function findinliers(distance::Array{Float64,2},thresh::Float64)

  return n::Int,indices::Array{Int,1}
end

# RANSAC algorithm
function ransac(pairs::Array{Int,2},thresh::Float64,n::Int)

  @assert size(bestinliers,2) == 1
  @assert size(bestpairs) == (4,4)
  @assert size(bestH) == (3,3)
  return bestinliers::Array{Int,1},bestpairs::Array{Int,2},bestH::Array{Float64,2}
end

# Recompute the homography based on all inliers
function refithomography(pairs::Array{Int64,2}, inliers::Array{Int64,1})

  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end

# Show panorama stitch of both images using the given homography.
function showstitch(im1::Array{Float32,2},im2::Array{Float32,2},H::Array{Float64,2})

  return nothing::Void
end

# Problem 2: Image Stitching

function problem2()
  #SIFT Parameters
  sigma = 1.4              # standard deviation
  # RANSAC Parameters
  ransac_threshold = 50.0   # inlier threshold
  p = 0.5                 # probability that any given correspondence is valid
  k = 4                   # number of samples drawn per iteration
  z = 0.99                # total probability of success after all iterations

  # load images
  im1 = PyPlot.imread("../data-julia/a3p1a.png") # left image
  im2 = PyPlot.imread("../data-julia/a3p1b.png") # right image

  # load keypoints
  keypoints1, keypoints2 = loadkeypoints("../data-julia/keypoints.jld")

  # extract SIFT features for the keypoints
  features1 = Common.sift(keypoints1,im1,sigma)
  features2 = Common.sift(keypoints2,im2,sigma)

  # compute squared euclidean distance matirx
  D = euclideansquaredist(features1,features2)

  # find matching pairs
  pairs = findmatches(keypoints1,keypoints2,D)

  # show matches
  showmatches(im1,im2,pairs)
  title("Putative Matching Pairs")

  # compute number of iterations for the RANSAC algorithm
  niterations = computeransaciterations(p,k,z)

  # apply RANSAC
  bestinliers,bestpairs,bestH = ransac(pairs,ransac_threshold,niterations)

  # show best matches
  showmatches(im1,im2,bestpairs)
  title("Best 4 Matches")

  # show all inliers
  showmatches(im1,im2,pairs[bestinliers,:])
  title("All Inliers")

  # stitch images and show the result
  showstitch(im1,im2,bestH)

  # recompute homography with all inliers
  H = refithomography(pairs,bestinliers)
  showstitch(im1,im2,H)

  return
end
