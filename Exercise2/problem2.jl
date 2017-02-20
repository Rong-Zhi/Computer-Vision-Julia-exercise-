using PyPlot
using Images

# Load images from the yale_faces directory and return a MxN data matrix,
# where M is the number of pixels per face image and N is the number of images.
# Also return the dimensions of a single face image and the number of all face images
function loadfaces()
  facedim=[96 84]
  n=38*20
  i=1
  data=zeros(n,facedim[1]*facedim[2])
  for p=1:38
    for q=1:20
        im=PyPlot.imread(@sprintf("../data-julia/yale_faces/yaleBs%02i/%02i.pgm",p,q))
        data[i,:] = im[:]
        i=i+1
    end
  end
  data=convert(Array{Float64,2},data)
  return data::Array{Float64,2},facedim::Array{Int},n::Int
end

# Apply principal component analysis on the data matrix.
# Return the eigenvectors of covariance matrix of the data, the corresponding eigenvalues,
# the one-dimensional mean data matrix and a cumulated variance vector in increasing order.
function computepca(data::Array{Float64,2})
  data = data'
  mu = mean(data,2)
  data_hat = data.-mu
  U,s,_= svd(data_hat)
  L=length(s)
  lambda = s.^2 / size(data_hat,2)
  S = sum(lambda)
  cumvar = cumsum(lambda)./S

  return U::Array{Float64,2},lambda::Array{Float64,1},mu::Array{Float64,2},cumvar::Array{Float64,1}
end

# Compute required number of components to account for (at least) 80/95 % of the variance
function computencomponents(cumvar::Array{Float64,1})
  N80=find(cumvar.>0.8)
  n80=findmin(N80)[1]
  N95=find(cumvar.>0.95)
  n95=findmin(N95)[1]
  return n80::Int,n95::Int
end

# Display the mean face and the first 10 Eigenfaces in a single figure
function showfaces(U::Array{Float64,2},mu::Array{Float64,2},facedim::Array{Int})
  im=reshape(mu,facedim[1],facedim[2])
  figure()
  imshow(im,"gray")
  axis("off")
  figure()
  for i=1:10
    subplot(2,5,i)
    im=U[:,i]
    Im=reshape(im,facedim[1],facedim[2])
    imshow(Im,"gray")
    axis("off")
  end
  return nothing::Void
end

# Fetch a single face with given index out of the data matrix. Returns the actual face image.
function takeface(data::Array{Float64,2},facedim::Array{Int},n::Int)
  data=data'
  face=reshape(data[:,n],facedim[1],facedim[2])
  return face::Array{Float64,2}
end

# Project a given face into the low-dimensional space with a given number of principal
# components and reconstruct it afterwards
function computereconstruction(faceim::Array{Float64,2},U::Array{Float64,2},mu::Array{Float64,2},n::Int)
  face=reshape(faceim,length(faceim))
  comp = U[:,1:n]'*(faceim[:]-mu)
  recon = U[:,1:n]*comp+mu
  recon = reshape(recon,size(faceim))
  return recon::Array{Float64,2}
end



# Problem 2: Eigenfaces

function problem2()
  # load data
  data,facedim,N = loadfaces()

  # compute PCA
  U,lambda,mu,cumvar = computepca(data)

  # plot cumulative variance
  figure()
  plot(cumvar)
  grid("on")
  title("Cumulative Variance")
  gcf()

  # compute necessary components for 80% / 95% variance coverage
  n80,n95 = computencomponents(cumvar)

  # plot mean face and first 10 eigenfaces
  showfaces(U,mu,facedim)

  # get a random face
  faceim = takeface(data,facedim,rand(1:N))

  # reconstruct the face with 5,15,50,150 principal components
  f5 = computereconstruction(faceim,U,mu,5)
  f15 = computereconstruction(faceim,U,mu,15)
  f50 = computereconstruction(faceim,U,mu,50)
  f150 = computereconstruction(faceim,U,mu,150)

  # display the reconstructed faces
  figure()
  subplot(221)
  imshow(f5,"gray",interpolation="none")
  axis("off")
  title("5 Principal Components")
  subplot(222)
  imshow(f15,"gray",interpolation="none")
  axis("off")
  title("15 Principal Components")
  subplot(223)
  imshow(f50,"gray",interpolation="none")
  axis("off")
  title("50 Principal Components")
  subplot(224)
  imshow(f150,"gray",interpolation="none")
  axis("off")
  title("150 Principal Components")
  gcf()

  return
end
problem2()
