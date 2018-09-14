import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  #"""
  #Softmax loss function, naive implementation (with loops)

  #Inputs have dimension D, there are C classes, and we operate on minibatches
  #of N examples.

  #Inputs:
  #- W: A numpy array of shape (D, C) containing weights.
  #- X: A numpy array of shape (N, D) containing a minibatch of data.
  #- y: A numpy array of shape (N,) containing training labels; y[i] = c means
  #  that X[i] has label c, where 0 <= c < C.
  #- reg: (float) regularization strength

  #Returns a tuple of:
  #- loss as single float
  #- gradient with respect to weights W; an array of same shape as W
  #"""
  # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        hyp=X[i].dot(W)
        hyp=np.exp(hyp)/np.sum(np.exp(hyp))
        loss += -np.log(hyp[y[i]])
        for j in range(num_classes):
            dW[:,j] += hyp[j]*X[i]

        dW[:,y[i]] -= X[i]
            
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += reg * W

 
 
 
        
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  #"""
  #Softmax loss function, vectorized version.

  #Inputs and outputs are the same as softmax_loss_naive.
  #"""
  # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    
    hyp = np.exp(X.dot(W))
    
    hyp /= np.reshape(np.repeat(np.sum(hyp,axis=1),num_classes),hyp.shape)
    loss += np.sum(-np.log(hyp[range(num_train),y]))

    hyp[range(num_train),y] += -1
    dW = X.T.dot(hyp)
    
    loss /= num_train
    loss += reg*np.sum(W*W)
    dW /= num_train
    dW += reg * W

    
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
    pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

    return loss, dW

