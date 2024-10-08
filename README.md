# mfmodeling
Multi-fidelity modeling using Gaussian process regression

### Purpose of this project
This project is motivated to apply multi-fidelity data fusion algorithms to the regression problem in turbulent transport modeling in magnetic fusion plasma.
The developed module will be available as a general tool for multi-fidelity regression problems.

### Usage
**mfmodeling** module requires external packages: **numpy**, **GPy**.

The following is an explanation on the simple usage of NARGP (Nonlinear AutoRegressive Gaussian Process regression [P. Perdikaris (2017)]) as a multi-fidelity regression algorithm. See also ```tests/NARGP_example_2d/Demo_May2024_NARGP_example_2d.ipynb```.
1. Prepare multi-fidelity datasets as a list of each fidelity data.
```
    data_list = [[data_lowfid_x,  data_lowfid_y],
                 [data_highfid_x, data_highfid_y]]
```
2. Instantiate the NARGP object using the above dataset.
```
    from mfmodeling import NARGP
    model_nargp = NARGP(data_list = data_list)
```
3. Optimize hyperparameters of the kernel function in NARGP.
```
    model_nargp.optimize()
```
4. Make a prediction.
```
    mean, var = model_nargp.prediction(x_pred)
```
where ```mean``` and ```var``` are the prediction of posterior mean and variance at your evaluating position ```x_pred```.

## Reference
(Preprint) [S. Maeyama, M. Honda, E. Narita, S. Toda, "Multi-Fidelity Information Fusion for Turbulent Transport Modeling in Magnetic Fusion Plasma", under review.](https://doi.org/10.21203/rs.3.rs-4791765/v1)
[![doi](https://img.shields.io/badge/doi-10.21203/rs.3.rs--4791765/v1-5077AB.svg)](https://doi.org/10.21203/rs.3.rs-4791765/v1)
