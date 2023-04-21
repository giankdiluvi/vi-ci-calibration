"""
Functions to do 1d Gaussian variational inference
"""

import numpy as np

"""
##########################
##########################
#  optimization wrapper  #
##########################
##########################
"""

def gaussianVI(lp,randp,divergence='rev-kl',B=1000,max_iters=1001,lr_mu=1e-2,lr_lsigma=1e-2):
    """
    This function finds an approximation to p
    by minimizing the divergence from the family of Gaussians
    using stochastic gradient descent (sgd)
    If you want to minimize the reverse KL, randp can be None - it is only used if divergence=='fwd-kl'

    Inputs:
        lp         : function, target log pdf
        randp      : function, target random number generator
        divergence : string, one of 'rev-kl' or 'fwd-kl' indicating divergence to minimize
        B          : int, Monte Carlo sample size for gradient estimation
        max_iters  : int, max number of sgd iterations
        lr_mu      : float, sgd learning rate for the mean
        lr_lsigma  : float, sgd learning rate for the log standard deviation

    Outputs:
        mus     : (max_iters,) array, all values of the mean through the optimization
        lsigmas : (max_iters,) array, all values of the log sd through the optimization
    """

    # initial values
    mus=np.zeros(max_iters+1)
    mus[0]=np.random.randn() # initialize mu at a draw from std normal (cause 0 is probably the opt, let's make it harder)
    lsigmas=np.zeros(max_iters+1)
    lsigmas[0]=np.random.randn() # ditto

    # optimize
    print('Initial μ: '+str(mus[0]))
    print('Initial σ: '+str(np.exp(lsigmas[0])))
    print()
    print('Iter  |          μ          |          σ          |      μ gradient     |   logσ gradient')
    for t in range(max_iters):
        # calculate gradients
        grad_mu=mu_gradient(B,mus[t],lsigmas[t],lp,randp,divergence)
        grad_lsigma=ls_gradient(B,mus[t],lsigmas[t],lp,randp,divergence)

        # take step
        mus[t+1]=mus[t]-grad_mu*lr_mu/np.sqrt(t+1)
        lsigmas[t+1]=lsigmas[t]-grad_lsigma*lr_lsigma/np.sqrt(t+1)

        # do printout
        if t%(max_iters//10)==0: print('  '+str(t)+'   | '+str(mus[t+1])+'   | '+str(np.exp(lsigmas[t+1]))+'   | '+str(grad_mu)+'  | '+str(grad_lsigma))
    # end for

    print()
    print('Final μ: '+str(mus[-1]))
    print('Final σ: '+str(np.exp(lsigmas[-1])))
    return mus,lsigmas

"""
##########################
##########################
#       gradients        #
##########################
##########################
"""

def mu_gradient(B,mu,lsigma,lp,randp,divergence='rev-kl'):
    # calculate the gradient of the KL w.r.t. the mean mu
    if divergence=='rev-kl':
        sample=mu+np.exp(lsigma)*np.random.randn(B) #~N(mu,sigma^2)
        return np.mean((lq(sample,mu,lsigma)-lp(sample))*(sample-mu)*np.exp(-2*lsigma))
    if divergence=='fwd-kl':
        sample=randp(B) #~p
        return -np.mean(np.exp(-2*lsigma)*(sample-mu))

def ls_gradient(B,mu,lsigma,lp,randp,divergence='rev-kl'):
    # calculate the gradient of the KL w.r.t. the log sd lsigma
    if divergence=='rev-kl':
        sample=mu+np.exp(lsigma)*np.random.randn(B) #~N(mu,sigma^2)
        return np.mean((lq(sample,mu,lsigma)-lp(sample))*(np.exp(-2*lsigma)*(sample-mu)**2-1.))
    if divergence=='fwd-kl':
        sample=randp(B) #~p
        return -np.mean(np.exp(-2*lsigma)*(sample-mu)**2-1.)


"""
##########################
##########################
#     auxiliary fns      #
##########################
##########################
"""

# gaussian variational approximation
def lq(x,mu,logsigma):
    """
    Log density of a Gaussian

    Inputs:
        x        : (d,) array, points at which to evaluate the log density
        mu       : float, mean
        logsigma : float, log standard deviation

    Outputs:
        lq : (d,) array, log density evaluated at each point x_i
    """
    sigma=np.exp(logsigma)
    return -0.5*((x-mu)/sigma)**2-0.5*np.log(2*np.pi*sigma**2)

def kl(lq,lp,randq,B=1000):
    """
    Estimate KL(q||p)
    (for fwd KL simply input lp first and supply a randp sampler instead of randq)

    Inputs:
        lq    : function, log density of first argument in KL
        lp    : function, log density of second argument in KL
        randq : function, pseudo-random number generator of first argument in KL
        B     : int, Monte Carlo sample size for KL estimation

    Outputs:
        kl : float, estimate of KL(q||p)
    """
    sample=randq(B)
    return np.mean(lq(sample)-lp(sample))
