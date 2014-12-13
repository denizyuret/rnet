rnet
====

Matlab code for feed forward neural networks with RELU hidden units and Softmax cost function.


Dependencies:

    ____________
   |            V
 a[l] w[l]--->a[l+1]
 |  \/ 	       	|
 |  /\		|
 V V  V		V
∇z[l] ∇w[l]<--∇z[l+1]
   ∧            |
    ------------


Notation - vectorized:

l = 1...L		# layers, layer 1 is input, layer L is output
M			# number of examples in input batch
N[l]			# size of layer l, N[1] is dimensionality of data, N[L] is the number of outputs
f(z), ∂f(z)		# activation function and its derivative
z[l] : N[l],M		# linear output of layer l=2..L: z[l+1]=w[l]*a[l]+b[l]
a[l] : N[l],M		# output of layer l=1..L: a=f(z); a[1] is input, a[L] is final output
w[l] : N[l+1],N[l]	# weight matrix connecting layer l=1..(L-1) to layer l+1
b[l] : N[l+1],1		# bias vector connecting layer l=1..(L-1) to layer l+1
J(a[L],y)		# cost, where y is the desired output
∇z[l] : N[l],M		# gradient of cost wrt z[l], same size as z[l]


Notation - scalar:

z[l,i,m]		# linear output at layer l, unit i, instance m
a[l,i,m]		# output of layer l, unit i, instance m
w[l,j,i]		# weight from layer l unit i to layer l+1 unit j
b[l,j]			# bias for layer l+1 unit j
∂J / ∂z[l,i,m]		# partial derivative of cost wrt z[l,i,m]


Forward - vectorized form: for l=1..(L-1); a[1]=input.

z[l+1] = w[l] * a[l] + b[l]		# note that + needs singleton expansion, or b[l] needs repmat
a[l+1] = f(z[l+1])			# final a[L] may not be necessary


Forward - scalar form:

a[l,i,m] = f(z[l,i,m])
z[l+1,j,m] = b[l,j] + Σ_i w[l,j,i] * a[l,i,m]


Backward ∇z - vectorized form: for l=(L-1)..2; ∇z[L] manually computed, ∇z[1] not needed.

∇z[l] = (∂a[l] / ∂z[l]) .* [(∂z[l+1] / ∂a[l])' * (∂J / ∂z[l+1])]
      = ∂f(z[l]) .* (w[l]' * ∇z[l+1])


Backward ∇z - scalar form:

∂J / ∂z[l,i,m] = (∂a[l,i,m] / ∂z[l,i,m]) * Σ_j [ (∂z[l+1,j,m] / ∂a[l,i,m]) * (∂J / ∂z[l+1,j,m]) ]
               = ∂f(z[l,i,m]) * Σ_j w[l,j,i] * ∇z[l+1,j,m]


Backward ∇w - vectorized form: for l=(L-1)..1

∇w[l] = (∂J / ∂z[l+1]) * (∂z[l+1] / ∂w[l])'
      = ∇z[l+1] * a[l]'


Backward ∇w - scalar form:

∂J / ∂w[l,j,i] = Σ_m (∂J / ∂z[l+1,j,m]) * (∂z[l+1,j,m] / ∂w[l,j,i])
               = Σ_m (∂J / ∂z[l+1,j,m]) * a[l,i,m]


Backward ∇b - vectorized form: for l=(L-1)..1
(Note b[l] can be considered a matrix N[l+1],M created with repmat, thus Σ_m.)

∇b[l] = (∂J / ∂z[l+1]) * (∂z[l+1] / ∂b[l])'
      = Σ_m ∇z[l+1]


Backward ∇b - scalar form:

∂J / ∂b[l,j] = Σ_m (∂J / ∂z[l+1,j,m]) * (∂z[l+1,j,m] / ∂b[l,j])
             = Σ_m (∂J / ∂z[l+1,j,m])


Softmax ∇z[L] - scalar form: assuming y[m] is the correct class for instance m

J = (1/M) * Σ_m log(Σ_j(exp(z[L,j,m]))) - z[L,y[m],m]

∂J / ∂z[L,i,m] = (1/M) * ((1/(Σ_j(exp(z[L,j,m])))) * exp(z[L,i,m]) - [y[m]==i])
               = (1/M) * (p(y[m]==i) - [y[m]==i])


Softmax ∇z[L] - vectorized form: assuming y[i,m]=1 if instance m is in class i

a[L] = exp(z[L]) ./ log(Σ_i(exp(z[L])))  # with repmat or singleton expansion
∇z[L] = (1/M) * (a[L] - y)


General ∇z[l] = ∂f(z[l]) .* (w[l]' * ∇z[l+1])
RELU    ∇z[l] = [a[l]>0] .* (w[l]' * ∇z[l+1])
Sigmoid ∇z[l] = a[l] .* (1 - a[l]) .* (w[l]' * ∇z[l+1])
