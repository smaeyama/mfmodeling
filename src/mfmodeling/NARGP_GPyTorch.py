import numpy as np
from .SingleGP_GPyTorch import SingleGP

class NARGP:
    """
    Nonlinear autoregressive multi-fidelity GP (Perdikaris 2017)
    Uses SingleGP at each fidelity level.
    """
    def __init__(self, data_list):
        self.__data_list = data_list
        self.__model_list = []

    def optimize(self, verbose=True):
        nfidelity = len(self.__data_list)
        if verbose:
            print(f"--- Training NARGP with {nfidelity} fidelities ---")
        self.__model_list = []

        # Fidelity 0 (Lowest)
        X, Y = self.__data_list[0]
        if verbose: print(f"\n[Fidelity 0] Training on {len(X)} points")
        model = SingleGP([X, Y])
        model.optimize(verbose=verbose)
        self.__model_list.append(model)

        # Higher fidelities
        for i in range(1, nfidelity):
            X, Y = self.__data_list[i]
            if verbose: print(f"\n[Fidelity {i}] Training on {len(X)} points")
            # Predict previous fidelity
            mu, _ = self.predict(X, ifidelity=i-1)
            # Augment X with the mean prediction from the lower fidelity
            XX = np.hstack((X, mu))
            model = SingleGP([XX, Y], low_fidelity_noutput=mu.shape[-1])
            model.optimize(verbose=verbose)
            self.__model_list.append(model)

    def predict(self, x, ifidelity=None, nMonteCarlo=200):
        if ifidelity is None:
            ifidelity = len(self.__model_list) - 1
        if ifidelity == 0:
            return self.__model_list[0].predict(x)

        # Recursive prediction for higher fidelity
        mu_prev, var_prev = self.predict(x, ifidelity-1, nMonteCarlo=nMonteCarlo)

        # Monte Carlo integration for the non-linear mapping
        # Ensure samples shape matches [nMonteCarlo, n_samples, n_outputs]
        std_prev = np.sqrt(np.maximum(var_prev, 1e-12))
        samples = mu_prev[None, :, :] + std_prev[None, :, :] * np.random.randn(nMonteCarlo, *mu_prev.shape)

        tmp_m = []
        tmp_v = []
        model = self.__model_list[ifidelity]

        for i in range(nMonteCarlo):
            z = samples[i]
            XX = np.hstack((x, z))
            m, v = model.predict(XX)
            tmp_m.append(m)
            tmp_v.append(v)

        tmp_m = np.array(tmp_m) # [nMC, N, nout]
        tmp_v = np.array(tmp_v) # [nMC, N, nout]

        # Law of Total Expectation and Variance
        mu = np.mean(tmp_m, axis=0)
        var = np.mean(tmp_v, axis=0) + np.var(tmp_m, axis=0)

        return mu, np.abs(var)

    @property
    def model_list(self):
        return self.__model_list

    @property
    def data_list(self):
        return self.__data_list
