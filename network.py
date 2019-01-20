import numpy as np
import numpy.linalg as la

import tensorflow as tf
import tools.shrinkage as shrinkage

def build_LISTA(prob,T,initial_lambda=.1,untied=False):
	
    #Builds a LISTA network to infer x from prob.y_ = matmul(prob.A,x) + AWGN
    #return a list of layer info (name,xhat_,newvars)
    #name : description, e.g. 'LISTA T=1'
    #xhat_ : that which approximates x_ at some point in the algorithm
    #newvars : a tuple of layer-specific trainable variables
    
    assert not untied,'TODO: untied'
    eta = shrinkage.simple_soft_threshold #shrinkage function
    layers = []
    A = prob.A #dictionary matrix
    M,N = A.shape
    B = A.T / (1.01 * la.norm(A,2)**2)
    B_ =  tf.Variable(B,dtype=tf.float32,name='B_0') #creating a variable for the dictionary matrix
    S_ = tf.Variable( np.identity(N) - np.matmul(B,A),dtype=tf.float32,name='S_0') #mutual inhibition matrix
    By_ = tf.matmul( B_ , prob.y_ ) #By_ is the signal after going through B_
    layers.append( ('Linear',By_,None) )

    initial_lambda = np.array(initial_lambda).astype(np.float32)
    if getattr(prob,'iid',True) == False:
        # create a parameter for each coordinate in x
        initial_lambda = initial_lambda*np.ones( (N,1),dtype=np.float32 )
    lam0_ = tf.Variable( initial_lambda,name='lam_0')
    xhat_ = eta( By_, lam0_) # the signal after going through the first shrinkage function
    layers.append( ('LISTA T=1',xhat_, (lam0_,) ) )
    for t in range(1,T): #construct LISTA iterative structure using for loop
        lam_ = tf.Variable( initial_lambda,name='lam_{0}'.format(t) )
        xhat_ = eta( tf.matmul(S_,xhat_) + By_, lam_ ) #xhat_ = eta(xhat_*S_ + By_)
        layers.append( ('LISTA T='+str(t+1),xhat_,(lam_,)) )
    return layers