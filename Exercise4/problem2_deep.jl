# Bonus Task: What happens if you want to train a deep netowrk for the previous classification task?
# What's the issue? what can you do about that?

#My answer: MLP with hidden layers have a non-convex loss function where there exists more than one local minimum.
#     Therefore different random weight initializations can lead to different validation accuracy. It can also get stuck in poor local optima.
#     From the backpropagation algorithm we can see the huge amount of calculation defined by this algorithm leads to a slow convergence speed,
#     especialy when the we want to train a deep network.
#     Network paralysis with sigmoid activation function. The input of the neurons may fall into a saturation region
#     when some of its incoming weights become very large during learning.
#     Consequently, weights stop to change no matter how hard you try.

# For a typical MLP network, we can simply define a network with 3 layers( with proper number of nodes in each layer, #of hidden units is half of input units).
# For a deeper network, we should set fewer nodes in each layer, so that it can be trained faster. At the same time, we should set learning rate to a small value,
# to keep the difference smaller than 1e-4 so that the neurons would not fall into a saturation region. Thus, the algorithm won't get stuck in poor local optima.

using Images
using PyPlot
using JLD
using Optim

include("Common.jl")

# Load features and labels from file
function loaddata(path::ASCIIString)
  features = load(path,"features")
  labels = load(path,"labels")
  @assert length(labels) == size(features,1)
  return features::Array{Float64,2}, labels::Array{Float64,1}
end

# Show a 2-dimensional plot for the given features with different colors according
# to the labels.
function showbefore(features::Array{Float64,2},labels::Array{Float64,1})
  figure()
  label0=(labels.==0)
  scatter(features[label0,1],features[label0,2],color="blue")
  label1=(labels.==1)
  scatter(features[label1,1],features[label1,2],color="red")
  legend(["label0","label1"])
  return nothing::Void
end

# Show a 2-dimensional plot for the given features along with the decision boundary
function showafter(features::Array{Float64,2},labels::Array{Float64,1},Ws::Vector{Any}, bs::Vector{Any},netdefinition::Array{Int,1})
  xmin  = minimum(features[1,:])
  xmax  = maximum(features[1,:])
  z = zeros(100,100)
  showbefore(features,labels)
  max1,max2 = maximum(features,1)
  min1,min2 = minimum(features,1)
  x1  = linspace(min1,max1,100)
  x2  = linspace(min2,max2,100)
  gx,gy = Common.meshgrid(x1,x2)
  for i in 1:100
    f1 = gx[:,i]
    f2 = gy[:,i]
    f = [f1 f2]
    pred,clas = predict(f,Ws,bs,netdefinition)
    z[:,i] = clas
  end
  contourf(gx, gy, z, alpha=0.5)
  p,c  = predict(features,Ws,bs,netdefinition)
  label0=(c.==0)
  label1=(c.==1)
  scatter(features[label0,1],features[label0,2],color="blue")
  scatter(features[label1,1],features[label1,2],color="red")
  legend(["label0","label1"])
  return nothing::Void
end

# Implement the sigmoid function
function sigmoid(z)
  s = zeros(length(z))
  for i in 1:length(z)
    s[i] = 1/(1+exp(-z[i]))
  end
  return s
end

# Implement the derivative of the sigmoid function
function dsigmoid_dz(z)
  temp1 = sigmoid(z)
  temp2 = ones(size(sigmoid(z)))-sigmoid(z)
  dz    = zeros(size(sigmoid(z)))
  for i in 1:length(temp1)
    dz[i] = temp1[i] * temp2[i]
  end
  return dz
end

# Evaluates the loss function of the MLP
function nnloss(theta::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, netdefinition::Array{Int, 1})
  W,b = thetaToWeights(theta,netdefinition)
  N = size(X,1)
  L = zeros(N)
  net = netdefinition
  netN = length(netdefinition)
  for i in 1:size(X,1)
    p = X[i,:]
    iw = 1
    ib = 1
    for a in 1:(netN-1)
      Ws = zeros(net[a+1],net[a])
      bs = zeros(net[a+1])
      updatew = net[a] * net[a+1]
      updateb = net[a+1]
      Ws[:,:] = reshape(W[iw:(iw + updatew-1)],net[a+1],net[a])
      bs[:] = reshape(b[ib:(ib + updateb-1)],1,net[a+1])
      p = sigmoid(Ws * p + bs)
      iw = iw + updatew
      ib = ib + updateb
    end
    L[i] = y[i] * log(p[1]) + (1 - y[i]) * log(1 - p[1])
  end
  loss = -sum(L)/N
  # print("Loss:",loss)
  return loss::Float64
end

# Evaluate the gradient of the MLP loss w.r.t. Ws and Bs
# The gradient should be stored in the vector 'storage'
function nnlossgrad(theta::Array{Float64,1}, storage::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, netdefinition::Array{Int, 1})
  Lr = 1e-7 #Lr is learning-rate
  net = netdefinition
  W,b = thetaToWeights(theta,net)
  W = convert(Array{Float64,1},W)
  b = convert(Array{Float64,1},b)
  netsize = length(net)
  a = Array(Vector{Float64},netsize)
  z = Array(Vector{Float64},netsize)
  theta = Array(Array{Float64,2},netsize)
  delta = Array(Matrix{Float64},netsize)
  Delta = Array(Matrix{Float64},netsize)
  N = size(X,1)
  delta_last = 0
  for p in 1:(netsize-1)
    Delta[p] = zeros(net[p+1],net[p]+1)
  end
  for i in 1:N
    a[1] = convert(Vector{Float64},[X[i,:];1])
    iw = 1
    ib = 1
    for m in 2:netsize
      Ws = reshape(W[iw:(iw+net[m]*net[m-1]-1)],net[m],net[m-1])
      bs = reshape(b[ib:(ib+net[m]-1)],net[m],1)
      theta[m-1] = [Ws bs]
      z[m] = theta[m-1]*a[m-1]
      tmp = sigmoid(z[m])
      a[m] = [tmp;1]
      iw = iw + net[m]*net[m-1]
      ib = ib + net[m]
    end
    delta_last = a[netsize][1] - y[i]
    for q in 1:(netsize-2)
      n = netsize-q
      if n == netsize - 1
        delta[n] = theta[n][:,1:(end-1)]' * delta_last .* dsigmoid_dz(z[n])
      else
        delta[n] = theta[n][:,1:(end-1)]' * delta[n+1] .* dsigmoid_dz(z[n])
      end
    end
    for q in 1:(netsize-1)
      n = netsize - q
      if n == netsize -1
        Delta[n] = Delta[n] + delta_last * a[n]'
        Delta[n][:,1:(end-1)] = Delta[n][:,1:(end-1)] + Lr * theta[n][:,1:(end-1)] #Lr is the learning-rate
      else
        Delta[n] = Delta[n] + delta[n+1] * a[n]'
        Delta[n][:,1:(end-1)] = Delta[n][:,1:(end-1)] + Lr * theta[n][:,1:(end-1)]
      end
    end
  end
  iw = 1
  ib = 1
  for q in 1:(netsize-1)
    Delta[q] = 1/N * Delta[q]
    tmp1 = reshape(Delta[q][:,1:(end-1)],1,net[q]*net[q+1])
    W[iw:(iw+net[q]*net[q+1]-1)] = tmp1[:]
    tmp2 = reshape(Delta[q][:,end],1,net[q+1])
    b[ib:(ib+net[q+1]-1)] = tmp2[:]
    iw = iw + net[q] * net[q+1]
    ib = ib + net[q+1]
  end
  # print("Diff:",mean(b))
  storage[1:end] =[W;b]
  return storage::Array{Float64,1}
end

# Use LBFGS to optmize w and b of the MLP loss
function train(trainfeatures::Array{Float64,2}, trainlabels::Array{Float64,1}, netdefinition::Array{Int, 1})
  Ws,bs = initWeights(netdefinition,0.5,0.5)
  theta = weightsToTheta(Ws,bs)
  loss(theta) = nnloss(theta, trainfeatures, trainlabels, netdefinition)
  grad(theta,storage) = nnlossgrad(theta, storage, trainfeatures, trainlabels, netdefinition)
  res = optimize(loss,grad,theta,LBFGS())
  theta = Optim.minimizer(res)
  Ws,bs = thetaToWeights(theta,netdefinition)
  return Ws::Vector{Any},bs::Vector{Any}
end

# Predict the classes of the given data points using Ws and Bs.
# p, N x 1 array of Array{Float,2}, contains the output class scores (continuous value) for each input feature.
# c, N x 1 array of Array{Float,2}, contains the output class label (either 0 or 1) for each input feature.
function predict(X::Array{Float64,2}, Ws::Vector{Any}, bs::Vector{Any},netdefinition::Array{Int,1})
  Ws = convert(Array{Float64,1},Ws)
  bs = convert(Array{Float64,1},bs)
  net = netdefinition
  N = length(net)
  a = Array(Vector{Float64},N)
  z = Array(Vector{Float64},N)
  theta = Array(Array{Float64,2},N)
  p = zeros(size(X,1))
  for i in 1:size(X,1)
    a[1] = convert(Vector{Any},[X[i,:];1])
    iw = 1
    ib = 1
    for m in 2:N
      W = reshape(Ws[iw:(iw+net[m]*net[m-1]-1)],net[m],net[m-1])
      b = reshape(bs[ib:(ib+net[m]-1)],net[m],1)
      theta[m-1] = [W b]
      z[m] = theta[m-1]*a[m-1]
      tmp = sigmoid(z[m])
      a[m] = [tmp;1]
      iw = iw + net[m]*net[m-1]
      ib = ib + net[m]
    end
    p[i] = a[N][1]
  end
  c = zeros(size(p))
  c = p.>0.5
  c = convert(Array{Float64,1},c)
  return p::Array{Float64,1}, c::Array{Float64,1}
end

# A helper function which concatenates weights and biases into a variable theta
function weightsToTheta(Ws::Vector{Any}, bs::Vector{Any})
  theta = vcat(Ws,bs)
  theta = convert(Array{Float64,1},theta)
  return theta::Array{Float64,1}
end

# A helper function which decomposes and reshapes weights and biases from the variable theta
function thetaToWeights(theta::Array{Float64,1}, netdefinition::Array{Int,1})
  m = 0
  for i = 1:(length(netdefinition)-1)
    m = m + netdefinition[i]*netdefinition[i+1]
  end
  Ws = theta[1:m]
  bs = theta[(m+1):end]
  Ws = convert(Vector{Any},Ws)
  bs = convert(Vector{Any},bs)
  return Ws::Vector{Any}, bs::Vector{Any}
end

# Initialize weights and biases from Gaussian distributions
function initWeights(netdefinition::Array{Int,1}, sigmaW::Float64, sigmaB::Float64)
  m = 0
  n = 0
  for i = 1:(length(netdefinition)-1)
    m = m + netdefinition[i]*netdefinition[i+1]
    n = n + netdefinition[i+1]
  end
  Ws = randn(1,m) * sigmaW
  bs = randn(1,n) * sigmaB
  Ws = squeeze(Ws,1)
  bs = squeeze(bs,1)
  Ws = convert(Vector{Any},Ws)
  bs = convert(Vector{Any},bs)
  return Ws::Vector{Any}, bs::Vector{Any}
end

function problem2_deep()
  # # LINEAR SEPARABLE DATA
  # # load data
  # features,labels = loaddata("../data-julia/separable.jld")
  #
  # # show data points
  # showbefore(features,labels)
  # title("Data for Separable Case")
  #
  # # train MLP
  # Ws,bs = train(features,labels, [2,4,1])
  #
  # # show optimum and plot decision boundary
  # showafter(features,labels,Ws,bs,[2,4,1])
  # title("Learned Decision Boundary for Separable Case")
  #
  #
  # ## LINEAR NON-SEPARABLE DATA
  # # load data
  # features,labels = loaddata("../data-julia/nonseparable.jld")
  #
  # # show data points
  # showbefore(features,labels)
  # title("Data for Non-Separable Case")
  #
  # # train MLP
  # Ws,bs = train(features,labels, [2,4,1])
  #
  # # show optimum and plot decision boundary
  # showafter(features,labels,Ws, bs,[2,4,1])
  # title("Learned Decision Boundary for Non-Separable Case")

  # PLANE-BIKE-CLASSIFICATION FROM PROBLEM 1
  # load data
  trainfeatures,trainlabels = loaddata("../data-julia/imgstrain.jld")
  testfeatures,testlabels = loaddata("../data-julia/imgstest.jld")

  net = [50,20,10,5,3,1]    #  net size [50,20,10,5,3,1] -- training error: 0.0%
                                                          # testing error: 1.67%
  # net = [50,20,1] # the net with size of [50,20,1] is the optima solution of task 2
  #                 # training error: 0.0%
  #                 # testing error:  0.83%

  # train SVM and predict classes
  Ws,bs = train(trainfeatures,trainlabels, net)
  _,trainpredictions = predict(trainfeatures, Ws, bs,net)
  _,testpredictions = predict(testfeatures, Ws, bs,net)

  # show error
  trainerror = sum(trainpredictions.!=trainlabels)/length(trainlabels)
  testerror = sum(testpredictions.!=testlabels)/length(testlabels)
  println("Training Error Rate: $(round(100*trainerror,2))%")
  println("Testing Error Rate: $(round(100*testerror,2))%")
  return
end
