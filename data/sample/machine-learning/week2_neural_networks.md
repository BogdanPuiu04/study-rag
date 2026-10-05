# Week 2: Neural Networks

## Gradient descent

Gradient descent minimises the loss by repeatedly stepping the weights in the
direction of the negative gradient. The learning rate sets the step size: too
large and training diverges, too small and it converges slowly. Stochastic
gradient descent (SGD) estimates the gradient from a mini-batch instead of the
full dataset, trading noise for much cheaper updates.

## Backpropagation

Backpropagation computes the gradient of the loss with respect to every weight by
applying the chain rule layer by layer, from the output back to the input. It
reuses intermediate results from the forward pass, so the cost of the backward
pass is roughly the same as the forward pass.

## Activation functions

Non-linear activation functions let networks represent non-linear functions.
ReLU outputs max(0, x); it is cheap and reduces the vanishing gradient problem
that affects sigmoid and tanh in deep networks. Softmax turns the final layer's
scores into a probability distribution for multi-class classification.

## Dropout

Dropout randomly sets a fraction of activations to zero during training, which
prevents units from co-adapting and acts as a regulariser. At inference time all
units are kept and activations are scaled accordingly.
