from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt



class TwoLayerNet(object):
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):

        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
   
    # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

    # Compute the forward pass
        scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
        output = np.maximum(0, X.dot(W1) + b1) 
        scores = output.dot(W2)+b2
        
        pass
    
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################
    
    # If the targets are not given then jump out, we're done
        if y is None:
            return scores

    # Compute the loss
        loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss.                                                          #
    #############################################################################
        shift_scores = scores - np.max(scores, axis = 1).reshape(-1,1)
        softmax_output = np.exp(shift_scores)/np.sum(np.exp(shift_scores), axis = 1).reshape(-1,1)

        loss = -np.sum(-np.log(softmax_output[range(N),y]))
        loss /= N
        loss += reg*(np.sum(W1*W1)+np.sum(W2*W2))
        pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

    # Backward pass: compute gradients
        grads = {}
        dscores = np.copy(softmax_output)
        dscores[range(N),y] -= 1
 
        grads['W2'] = output.T.dot(dscores) + reg * W2
        grads['b2'] = np.sum(dscores, axis = 0)
        
        doutput = dscores.dot(W2.T)
        doutput_ReLU = (output > 0 ) * doutput
        grads['W1'] = X.T.dot(doutput_ReLU) + reg * W1
        grads['b1'] = np.sum(doutput_ReLU,axis = 0) 
        
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
        pass
    #############################################################################
    #                              END OF YOUR CODE                             #
    #############################################################################

        return loss, grads

    def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=5e-6, num_iters=100,
            batch_size=200, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
            idx = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X[idx]
            y_batch = y[idx]
            pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

      # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
            self.params['W2'] += -learning_rate * grads['W2']
            self.params['b2'] += -learning_rate * grads['b2']
            self.params['W1'] += -learning_rate * grads['W1']
            self.params['b1'] += -learning_rate * grads['b1']
            
            pass
      #########################################################################
      #                             END OF YOUR CODE                          #
      #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

      # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
        # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

        # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

    def predict(self, X):

        y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
        scores = np.maximum(0,X.dot(self.params['W1']) + self.params['b1']).dot(
            self.params['W2'])+self.params['b2']
        y_pred = np.argmax(scores,axis=1)
        
        pass
    ###########################################################################
    #                              END OF YOUR CODE                           #
    ###########################################################################

        return y_pred


