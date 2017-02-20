using Images
using PyPlot
using JLD

#using ImageView



# load and return the provided image
function loadImage()

    img=imread("a0p2.png")

  return img::Array{Float32, 3}
end

# save the image as a .jld file
function saveFile(img::Array{Float32, 3})

    save("array.jld","img",img)
end

# load the .jld file and return the image
function loadFile()
  img=load("array.jld","img")

  return img::Array{Float32, 3}
end

# create and return a horizontally mirrored image
function mirrorHorizontal(img::Array{Float32, 3})
  L=size(img)
  imgMirrored= Array(Float32,L)
  for i in 1:L[3]
    for m in 1:L[1]
      for n in 1:L[2]
        imgMirrored[m,n,i]=img[(L[1]-m+1),(L[2]-n+1),i]
      end
    end
  end
  return imgMirrored::Array{Float32, 3}
end

# display the original and the mirrored image in one plot
function displayImages(img1::Array{Float32, 3}, img2::Array{Float32, 3})
   figure()
   subplot(1,2,1)
   imshow(img)
   axis("off")
   subplot(1,2,2)
   imshow(img1)
   axis(:off")
end

#= Problem 2
Load and Display Image =#

function problem2()

  img1 = loadImage()

  saveFile(img1)

  img2 = loadFile()

  img2 = mirrorHorizontal(img2)

  displayImages(img1, img2)
end
problem2()
