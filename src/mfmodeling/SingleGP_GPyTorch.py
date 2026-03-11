import numpy as np
import torch
import gpytorch

class SingleGP(gpytorch.models.ExactGP):
    def __init__(self, data, dtype=torch.float64, device=None, low_fidelity_noutput=0):
        self.__data = data
        device = device if device else torch.device("cpu")
        train_x = torch.as_tensor(data[0], dtype=dtype, device=device)
        train_y = torch.as_tensor(data[1], dtype=dtype, device=device)
        if train_x.ndim == 1: train_x = train_x[:, None]
        if train_y.ndim == 1: train_y = train_y[:, None]

        self.train_x, self.train_y = train_x, train_y
        nsample, ninput = self.train_x.shape
        nsample, noutput = self.train_y.shape
        self._ninput, self._noutput, self._dtype, self._device = ninput, noutput, dtype, device
        self._low_fidelity_noutput = low_fidelity_noutput
        self._ninput_original = self._ninput - self._low_fidelity_noutput
        if low_fidelity_noutput > 0:
            assert self._ninput > low_fidelity_noutput

        batch_shape = torch.Size([noutput])
        likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=noutput, has_global_noise=False, has_task_noise=True,
            noise_constraint=gpytorch.constraints.GreaterThan(1e-6) # Lowering the noise constraint
        )
        super().__init__(self.train_x, self.train_y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=batch_shape)

        if self._low_fidelity_noutput > 0:
            # High-fidelity specific kernel (Product + Sum as per GPy example)
            # RBF(ninput_original) * RBF(low_fidelity_noutput) + RBF(ninput_original)

            # RBF for original input dimensions
            kernel_original_prod = gpytorch.kernels.RBFKernel(
                ard_num_dims=self._ninput_original,
                active_dims=list(range(self._ninput_original)),
                batch_shape=batch_shape
            )
            # RBF for low-fidelity output dimensions (mu)
            kernel_low_fidelity_prod = gpytorch.kernels.RBFKernel(
                ard_num_dims=self._low_fidelity_noutput,
                active_dims=list(range(self._ninput_original, self._ninput)),
                batch_shape=batch_shape
            )
            # Combine using operators + and *
            product_kernel = kernel_original_prod * kernel_low_fidelity_prod

            # RBF for original input dimensions (sum component)
            kernel_original_sum = gpytorch.kernels.RBFKernel(
                ard_num_dims=self._ninput_original,
                active_dims=list(range(self._ninput_original)),
                batch_shape=batch_shape
            )
            sum_kernel = product_kernel + kernel_original_sum
            self.covar_module = gpytorch.kernels.ScaleKernel(sum_kernel, batch_shape=batch_shape)
        else:
            # Default kernel for SingleGP
            self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=ninput, batch_shape=batch_shape),
                batch_shape=batch_shape
            )
        self._initialize_params()
        self.to(device=device, dtype=dtype)

    def _initialize_params(self):
        with torch.no_grad():
            yvar = torch.var(self.train_y, dim=0).clamp(min=1e-6)
            self.covar_module.outputscale = yvar

            # Initialize noise. Default constraint is GreaterThan(1e-6) for MultitaskGaussianLikelihood
            initial_noise_val = 0.001 * yvar
            self.likelihood.task_noises = initial_noise_val

            xrange = (self.train_x.max(0).values - self.train_x.min(0).values).clamp(min=1e-6)

            if self._low_fidelity_noutput > 0:
                # Initialize lengthscales for the complex kernel
                xrange_original = xrange[:self._ninput_original]
                xrange_low_fidelity = xrange[self._ninput_original:]

                # Product Kernel components - lengthscales are handled by accessing base_kernel.kernels[idx].lengthscale
                # The product kernel is now the first base_kernel component, and its sub-kernels are at [0] and [1]
                # For the RBF(ninput_original) part of the product
                self.covar_module.base_kernel.kernels[0].kernels[0].lengthscale = (
                    0.2 * xrange_original).view(1, 1, -1).expand(self._noutput, 1, self._ninput_original)
                # For the RBF(low_fidelity_noutput) part of the product
                self.covar_module.base_kernel.kernels[0].kernels[1].lengthscale = (
                    0.2 * xrange_low_fidelity).view(1, 1, -1).expand(self._noutput, 1, self._low_fidelity_noutput)
                # Sum Kernel component - this is the second base_kernel component
                self.covar_module.base_kernel.kernels[1].lengthscale = (
                    0.2 * xrange_original).view(1, 1, -1).expand(self._noutput, 1, self._ninput_original)
            else:
                # Default lengthscale initialization
                self.covar_module.base_kernel.lengthscale = (
                    0.2 * xrange).view(1, 1, -1).expand(self._noutput, 1, self._ninput)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(gpytorch.distributions.MultivariateNormal(mean_x, covar_x))

    def optimize(self, optimize_restarts=4, max_iters=400, verbose=True):
        best_mll = float('-inf')
        best_state = None

        for r in range(optimize_restarts):
            if r > 0: # Perturb parameters for restarts
                for param in self.parameters():
                    param.data.add_(torch.randn_like(param.data) * 0.1)

            self.train(); self.likelihood.train()
            mll_obj = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self)

            # Step 1: Fix noise
            self.likelihood.raw_task_noises.requires_grad_(False)
            self._run_lbfgs(mll_obj, max_iters, lr=0.1)

            # Step 2: Full optimization
            self.likelihood.raw_task_noises.requires_grad_(True)
            current_loss = self._run_lbfgs(mll_obj, max_iters, lr=0.1)

            if -current_loss > best_mll:
                best_mll = -current_loss
                best_state = {k: v.clone() for k, v in self.state_dict().items()}

            if verbose: print(f"Restart {r+1}/{optimize_restarts}, f = {current_loss:.4f}")

        self.load_state_dict(best_state)
        return self

    def _run_lbfgs(self, mll, max_iter, lr):
        opt = torch.optim.LBFGS(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe")
        def closure():
            opt.zero_grad()
            loss = -mll(self(self.train_x), self.train_y).sum()
            loss.backward(); return loss
        return opt.step(closure).item()

    def predict(self, x, return_numpy=True):
        self.eval(); self.likelihood.eval()
        x_t = torch.as_tensor(x, dtype=self._dtype, device=self._device)
        if x_t.ndim == 1: x_t = x_t[:, None]
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self(x_t))
        if return_numpy: return pred.mean.cpu().numpy(), pred.variance.cpu().numpy()
        return pred.mean, pred.variance
