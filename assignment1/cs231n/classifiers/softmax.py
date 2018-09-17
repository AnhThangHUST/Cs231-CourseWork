import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
	"""
	Softmax loss function, naive implementation (with loops)

	Inputs have dimension D, there are C classes, and we operate on minibatches
	of N examples.

	Inputs:
	- W: A numpy array of shape (D, C) containing weights.
	- X: A numpy array of shape (N, D) containing a minibatch of data.
	- y: A numpy array of shape (N,) containing training labels; y[i] = c means
		that X[i] has label c, where 0 <= c < C.
	- reg: (float) regularization strength

	Returns a tuple of:
	- loss as single float
	- gradient with respect to weights W; an array of same shape as W
	"""
	# Initialize the loss and gradient to zero.
	num_classes = W.shape[1]
	num_train = X.shape[0]
	loss = 0.0
	dW = np.zeros_like(W)

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.	    #
	# Store the loss in loss and the gradient in dW. If you are not careful		#
	# here, it is easy to run into numeric instability. Don't forget the		#
	# regularization!														    #
	#############################################################################
	#pass
	for i in xrange(num_train):
		scores = X[i].dot(W)
		exp_scores = np.exp(scores)
		loss += -scores[y[i]] + np.log(np.sum(exp_scores))
		for j in xrange(num_classes):
			if j == y[i]:
				dW[:,j] += X[i] * (-1 + 1/np.sum(exp_scores) * exp_scores[j])
				continue
			dW[:,j] += 1/np.sum(exp_scores) * exp_scores[j] *  X[i]
	loss /= num_train
	loss += reg * np.sum(W * W)
	dW /= num_train
	dW += 2*reg*W
	#############################################################################
	#							END OF YOUR CODE								#
	#############################################################################

	return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
	"""
	Softmax loss function, vectorized version.

	Inputs and outputs are the same as softmax_loss_naive.
	"""
	# Initialize the loss and gradient to zero.
	loss = 0.0
	dW = np.zeros_like(W)
	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.	#
	# Store the loss in loss and the gradient in dW. If you are not careful		#
	# here, it is easy to run into numeric instability. Don't forget the		#
	# regularization!															#
	#############################################################################
	#pass
	# save loss
	num_train = X.shape[0]
	scores = X.dot(W)
	stable_scores = scores - np.max(scores,axis=1).reshape(num_train,1)     
	exp_scores = np.exp(stable_scores)
	sum_exp_scores = np.sum(exp_scores, axis = 1)
	log_exp_scores = np.log(sum_exp_scores)
	index = np.arange(num_train)
	correct_classes = stable_scores[index, y]
	loss = np.sum(log_exp_scores) - np.sum(correct_classes)
	loss /= num_train
	loss += reg * np.sum(W * W)
	# save dW  
	mask = np.zeros(scores.shape)
	mask[index,y] = -1
	mask += exp_scores * 1/sum_exp_scores.reshape(num_train,1)
	dW = np.transpose(X).dot(mask)
	dW /= num_train
	dW += 2*reg*W
	#############################################################################
	#								END OF YOUR CODE							#
	#############################################################################

	return loss, dW

