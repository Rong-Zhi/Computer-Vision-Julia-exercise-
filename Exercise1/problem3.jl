using Images
using PyPlot

# Create 3x3 derivative filters in x and y direction
function createfilters()
  x=[-1 0 1]/2
  y=gaussian2d(1,(3,1))
  fx=y*x
  fy=fx'
  return fx::Array{Float64,2},fy::Array{Float64,2}
end

# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})
  Ix=imfilter(I,fx,"replicate")
  Iy=imfilter(I,fy,"replicate")
  return Ix::Array{Float64,2},Iy::Array{Float64,2}
end

# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2}, Iy::Array{Float64,2}, thr::Float64)
  M=Array(Float64,size(Ix))
  M=sqrt(Ix.^2+Iy.^2)
  for n in 1:length(M)
    if M[n]<thr
      M[n]=0
    end
  end
  edges=M
  return edges::Array{Float64,2}
end

# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Íx::Array{Float64,2},Iy::Array{Float64,2})
  orient=ataan(Iy./Ix)
  ori=padarray(orient,(1,1),(1,1),"reflect")
  L=size(orient)
  for m in 2:L[1]-1
    for n in 2:L[2]-1
      if (ori[m,n]<=pi/8)&&(ori[m,n]>-pi/8)&&(edges[m,n]<max(edges[m,n-1],edges[m,n+1]))
        edges[m,n]=0
      elseif ((ori[m,n]>3*pi/8)||(ori[m,n]<=-3*pi/8))&&(edges[m,n]<max(edges[m-1,n],edges[m+1,n]))
        edges[m,n]=0
      elseif (ori[m,n]<=3*pi/8)&&(ori[m,n]>pi/8)&&(edges[m,n]<max(edges[m-1,n-1],edges[m+1,n+1]))
        edges[m,n]=0
      elseif (ori[m,n]>-3*pi/8)&&(ori[m,n]<=-pi/8)&&(edges[m,n]<max(edges[m+1,n-1],edges[m-1,n+1]))
        edges[m,n]=0
      end
    end
  end

  return edges::Array{Float64,2}
end


#= Problem 3
Image Filtering and Edge Detection =#

function problem3()
  # load image
  img = PyPlot.imread("../data-julia/a1p3.png")

  # create filters
  fx, fy = createfilters()

  # filter image
  Ix, Iy = filterimage(img, fx, fy)

  # show filter results
  figure()
  subplot(121)
  imshow(Ix, "gray", interpolation="none")
  title("x derivative")
  axis("off")
  subplot(122)
  imshow(Iy, "gray", interpolation="none")
  title("y derivative")
  axis("off")
  gcf()

  # show gradient magnitude
  figure()
  imshow(sqrt(Ix.^2 + Iy.^2),"gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()

  # threshold derivative
  threshold = 0.1
  edges = detectedges(Ix,Iy,threshold)
  figure()
  imshow(edges.>0, "gray", interpolation="none")
  axis("off")
  title("Binary edges")
  gcf()

  # non maximum suppression
  edges2 = nonmaxsupp(edges,Ix,Iy)
  figure()
  imshow(edges2.>0,"gray", interpolation="none")
  axis("off")
  title("Non-maximum suppression")
  gcf()
  return
end
problem3()
