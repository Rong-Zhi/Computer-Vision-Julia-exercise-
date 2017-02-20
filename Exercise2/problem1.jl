using Images
using PyPlot


# Create a gaussian filter
function makegaussianfilter(size::Array{Int,2},sigma::Float64)
  x  =  (-(size[1]-1)/2:(size[1]-1)/2)'
  y  =  -(size[2]-1)/2:(size[2]-1)/2
  X  = repmat(x,size[1],1)
  Y  = repmat(y,1,size[2])
  f  = exp(-(X.^2+Y.^2)/(2*sigma.^2))
  f  = f./sum(f[:])
  return f::Array{Float64,2}
end

# Create a binomial filter
function makebinomialfilter(size::Array{Int,2})
  x=Array(Float64,size[1],1)
  y=Array(Float64,1,size[2])
  for k = 1:size[1]
    x[k] = binomial(size[1]-1,k-1)
  end
  for k = 1:size[2]
    y[k] = binomial(size[2]-1,k-1)
  end
  f= x*y
  f=f./sum(f[:])
  return f::Array{Float64,2}
end

# Downsample an image by a factor of 2
function downsample2(A::Array{Float64,2})
  D=A[1:2:end,1:2:end]
  return D::Array{Float64,2}
end

# Upsample an image by a factor of 2
function upsample2(A::Array{Float64,2},fsize::Array{Int,2})
  D = zeros(size(A,1)*2,size(A,2)*2)
  D[1:2:end,1:2:end] = A
  filter = makebinomialfilter(fsize)
  U = 4 * imfilter(D,filter,"symmetric")
  return U::Array{Float64,2}
end

# Build a gaussian pyramid from an image.
# The output array should contain the pyramid levels in decreasing sizes.
function makegaussianpyramid(im::Array{Float32,2},nlevels::Int,fsize::Array{Int,2},sigma::Float64)
  A = makegaussianfilter(fsize,sigma)
  G=Array(Array{Float64,2},nlevels)
  G[1]=im
  for i=2:nlevels
    F = imfilter(G[i-1],A,"symmetric")
    G[i] =downsample2(F)
  end
  return G::Array{Array{Float64,2},1}
end

# Display a given image pyramid (laplacian or gaussian)
function displaypyramid(P::Array{Array{Float64,2},1})
  im = (P[1]- minimum(P[1]))/(maximum(P[1])-minimum(P[1]))
  for m = 2:length(P)
    N=(P[m].- minimum(P[m]))./(maximum(P[m]).-minimum(P[m]))
    im =[im [N; zeros(size(P[1],1)-size(P[m],1),size(P[m],2))]]
  end
  figure()
  imshow(im,"gray",interpolation="none")
  axis("off")
  return nothing::Void
end

# Build a laplacian pyramid from a gaussian pyramid.
# The output array should contain the pyramid levels in decreasing sizes.
function makelaplacianpyramid(G::Array{Array{Float64,2},1},nlevels::Int,fsize::Array{Int,2})

  L=Array(Array{Float64,2},nlevels)

  for i=1:1:(nlevels-1)
     T = upsample2(G[i+1],fsize)
     L[i] = G[i]-T
  end
  L[nlevels]=G[nlevels]
  return L::Array{Array{Float64,2},1}
end

# Amplify frequencies of the first two layers of the laplacian pyramid
function amplifyhighfreq2(L::Array{Array{Float64,2},1})
  A = L
  A[1] = L[1] * 1.5
  A[2] = L[2] * 1.5
  return A::Array{Array{Float64,2},1}
end

# Reconstruct an image from the laplacian pyramid
function reconstructlaplacianpyramid(L::Array{Array{Float64,2},1},fsize::Array{Int,2})
  im = L[end]
  for i=(size(L,1)-1):-1:1
     im = upsample2(L[i],fsize) + im
  end
  return im::Array{Float64,2}
end


# Problem 1: Image Pyramids and Image Sharpening

function problem1()
  # parameters
  fsize = [5 5]
  sigma = 1.5
  nlevels = 6

  # load image
  im = PyPlot.imread("../data-julia/a2p1.png")

  # create gaussian pyramid
  G = makegaussianpyramid(im,nlevels,fsize,sigma)

  # display gaussianpyramid
  displaypyramid(G)
  title("Gaussian Pyramid")

  # create laplacian pyramid
  L = makelaplacianpyramid(G,nlevels,fsize)

  # dispaly laplacian pyramid
  displaypyramid(L)
  title("Laplacian Pyramid")

  # amplify finest 2 subands
  L_amp = amplifyhighfreq2(L)

  # reconstruct image from laplacian pyramid
  im_rec = reconstructlaplacianpyramid(L_amp,fsize)

  # display original and reconstructed image
  figure()
  subplot(131)
  imshow(im,"gray",interpolation="none")
  axis("off")
  title("Original Image")
  subplot(132)
  imshow(im_rec,"gray",interpolation="none")
  axis("off")
  title("Reconstructed Image")
  subplot(133)
  imshow(im-im_rec,"gray",interpolation="none")
  axis("off")
  title("Difference")
  gcf()

  return
end
fsize = [5 5]
sigma = 1.5
nlevels = 6

# load image
im = PyPlot.imread("../data-julia/a2p1.png")

# create gaussian pyramid
G = makegaussianpyramid(im,nlevels,fsize,sigma)

# display gaussianpyramid
displaypyramid(G)
title("Gaussian Pyramid")

# create laplacian pyramid
L = makelaplacianpyramid(G,nlevels,fsize)

# dispaly laplacian pyramid
displaypyramid(L)
title("Laplacian Pyramid")

# amplify finest 2 subands
L_amp = amplifyhighfreq2(L)

# reconstruct image from laplacian pyramid
im_rec = reconstructlaplacianpyramid(L_amp,fsize)

# display original and reconstructed image
figure()
subplot(131)
imshow(im,"gray",interpolation="none")
axis("off")
title("Original Image")
subplot(132)
imshow(im_rec,"gray",interpolation="none")
axis("off")
title("Reconstructed Image")
subplot(133)
imshow(im-im_rec,"gray",interpolation="none")
axis("off")
title("Difference")
