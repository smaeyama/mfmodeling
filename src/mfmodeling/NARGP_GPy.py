#!/usr/bin/env python
# coding: utf-8

import numpy as np
import GPy

class NARGP():
    '''
    Nonlinear autoregressive multi-fidelity Gaussian process regression (NARGP)
    P. Perdikaris, et al., "Nonlinear information fusion algorithms for data-efficient multi-fidelity modeling"
    Proc. R. Soc. A 473, 20160751 (2017). http://dx.doi.org/10.1098/rspa.2016.0751
    '''
    
    def __init__(self, *args, **kwargs):
        '''
        Parameters
        ----------
        data_list : list[nfidelity]
            List of multi-fidelity training data of y=f(x), where input x and output y are vectors.
            
            Structure of the list of data is as follow.
            data_list[ 0] = [data_input_lowest[nsample_lowest,ninput],   data_output_lowest[nsample_lowest,noutput_lowest]]
            data_list[ 1] = [data_input_1st[nsample_1st,ninput],         data_output_1st[nsample_1st,noutput_1st]]
            ...
            data_list[-1] = [data_input_highest[nsample_highest,ninput], data_output_highest[nsample_highest,noutput_highest]]
            
            "ninput" (dimension of the input vector x) is the same for all fidelities, while "noutput_***" (dimension of 
            the output vector y) and "nsample_***" (number of sampling of data set) can be different for each fidelity.
        '''
        self.__data_list = kwargs.get("data_list")   # (List) data_list[nfidelity]
        return

    def optimize(self, optimize_restarts=30, max_iters=400, verbose=True):
        '''
        Parameters
        ----------
        optimize_restarts : Int
        max_iters : Int
        verbose : Bool
            Parameters in GPy.models.GPRegression.optimize_restarts()
        '''
        nfidelity = len(self.__data_list)
        if verbose:
            print("nfidelity=",nfidelity,", optimize_restarts=",optimize_restarts,", max_iters=",max_iters)
        self.__model_list = []
        self.__kernel_list = []

        # Single-fidelity GP for the lowest fidelity data
        ifidelity = 0
        X, Y = self.__data_list[ifidelity]
        nsample, ninput = X.shape
        nsample, noutput = Y.shape
        # Design kernel function
        k = GPy.kern.RBF(ninput, ARD=True)
        m = GPy.models.GPRegression(X=X, Y=Y, kernel=k)
        # Initialization of hyper parameters
        m[".*Gaussian_noise"] = m.Y.var()*0.01
        m[".*Gaussian_noise"].fix()
        # Optimization of hyper parameters
        m.optimize(max_iters=max_iters)
        m[".*Gaussian_noise"].unfix()
        m[".*Gaussian_noise"].constrain_positive()
        m.optimize_restarts(optimize_restarts, optimizer="bfgs", max_iters=max_iters, verbose=verbose)
        self.__model_list.append(m.copy())
        self.__kernel_list.append(k.copy())

        # Multi-fidelity GP for higher fidelity data
        if nfidelity > 1:
            for ifidelity in range(1,nfidelity):
                X, Y = self.__data_list[ifidelity]
                nsample, ninput = X.shape
                nsample, noutput = Y.shape
                mu, v = self.predict(X,ifidelity=ifidelity-1)
                noutput1 = mu.shape[1]
                XX = np.hstack((X,mu))
                # Design kernel function
                k = GPy.kern.RBF(ninput, active_dims=np.arange(ninput), ARD=True) \
                  * GPy.kern.RBF(noutput1, active_dims=np.arange(ninput,ninput+noutput1), ARD=True) \
                  + GPy.kern.RBF(ninput, active_dims=np.arange(ninput), ARD=True)
                m = GPy.models.GPRegression(X=XX, Y=Y, kernel=k)
                m[".*Gaussian_noise"] = m.Y.var()*0.01
                m[".*Gaussian_noise"].fix()
                m.optimize(max_iters=max_iters)
                m[".*Gaussian_noise"].unfix()
                m[".*Gaussian_noise"].constrain_positive()
                m.optimize_restarts(optimize_restarts, optimizer="bfgs", max_iters=max_iters, verbose=verbose)
                self.__model_list.append(m.copy())
                self.__kernel_list.append(k.copy())
                
        return
    
    def predict(self, x, ifidelity=None, nMonteCarlo=1000):
        '''
        Parameters
        ----------
        x : Numpy.ndarray[..., ninput]
            The points at which to make a prediction
        ifidelity : Int
            0 < ifidelity < nfidelity-1
            Prediction for the specified fidelity model
        nMonteCarlo : Int
            Sampling number of Monte Carlo integration of Eq. (2.14)
        
        Returns
        -------
        mean : Numpy.ndarray[..., noutput]
            Mean for i-fidelity model
        variance : Numpy.ndarray[..., noutput]
            Variance for i-th fidelity model
        '''
        if ifidelity is None:
            ifidelity = len(self.__data_list)-1

        if ifidelity == 0: # Evaluate at fidelity level 0
            m = self.__model_list[0]
            mu0, var0 = m.predict(x)
            return mu0, var0
        
        if ifidelity == 1: # Evaluate at fidelity level 1
            m0, m1 = self.__model_list[0:2]
            # mu0, cov0 = m0.predict(x, full_cov=True)
            # z0 = np.random.multivariate_normal(mu0.ravel(),cov0,nMonteCarlo)
            z0 = m0.posterior_samples_f(x,nMonteCarlo)
            tmp_m = []
            tmp_v = []
            for i in range(nMonteCarlo):
                wmu1, wvar1 = m1.predict(np.hstack((x, z0[:,:,i])))
                tmp_m.append(wmu1)
                tmp_v.append(wvar1)
            # get posterior mean and variance
            mu1 = np.mean(tmp_m, axis=0)
            var1 = np.mean(tmp_v, axis=0) + np.var(tmp_m, axis=0)
            var1 = np.abs(var1)
            return mu1, var1

        if ifidelity == 2:
            m0, m1, m2 = self.__model_list[0:3]
            # mu0, cov0 = m0.predict(x, full_cov=True)
            # z0 = np.random.multivariate_normal(mu0.ravel(),cov0,nMonteCarlo)
            z0 = m0.posterior_samples_f(x,nMonteCarlo)
            tmp_m = []
            tmp_v = []
            for i in range(nMonteCarlo):
                # wmu1, wcov1 = m1.predict(np.hstack((x, z0[i,:][:,None])), full_cov=True)
                # z1 = np.random.multivariate_normal(wmu1.ravel(),wcov1,nMonteCarlo)
                z1 = m1.posterior_samples_f(np.hstack((x, z0[:,:,i])),nMonteCarlo)
                for j in range(nMonteCarlo):
                    wmu2, wvar2 = m2.predict(np.hstack((x, z1[:,:,j])))
                    tmp_m.append(wmu2)
                    tmp_v.append(wvar2)
            # get posterior mean and variance
            mu2 = np.mean(tmp_m, axis=0)
            var2 = np.mean(tmp_v, axis=0) + np.var(tmp_m, axis=0)
            var2 = np.abs(var2)
            return mu2, var2

        if ifidelity >= 3:
            print("This fidelity level is not supported:", ifidelity)
            return
            
    @property
    def data_list(self):
        return self.__data_list

    @property
    def model_list(self):
        return self.__model_list

    @property
    def kernel_list(self):
        return self.__kernel_list

