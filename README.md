# mfmodeling
Multi-fidelity modeling using Gaussian processes regression.

---

This project is motivated to apply multi-fidelity data fusion algorithms to the regression problem in turbulent transport modeling in magnetic fusion plasma, where theoretical models, numerical simulations, and experimental data have different fidelity levels.

The module is designed as a general-purpose tool for multi-fidelity regression problems.

---

## Installation

Install from PyPI:

```bash
pip install mfmodeling
```

Or install the latest development version from GitHub:

```bash
pip install git+https://github.com/smaeyama/mfmodeling.git
```

---

## Algorithms

Currently implemented multi-fidelity regression methods:

- **NARGP** – Nonlinear AutoRegressive Gaussian Process (Perdikaris et al., 2017)

---

## Usage

The following is an explanation on the simple usage of NARGP as a multi-fidelity regression algorithm. See also a Demo notebook,  [`examples/NARGP_example_2d/Demo_May2024_NARGP_example_2d.ipynb`](examples/NARGP_example_2d/Demo_May2024_NARGP_example_2d.ipynb).
1. Prepare multi-fidelity datasets as a list of each fidelity data.
```python
data_list = [[data_lowfid_x,  data_lowfid_y],
             [data_highfid_x, data_highfid_y]]
```
2. Instantiate the NARGP model using the above dataset.
```python
from mfmodeling import NARGP
model_nargp = NARGP(data_list = data_list)
```
3. Optimize hyperparameters of the kernel function in NARGP.
```python
model_nargp.optimize()
```
4. Make predictions.
```python
mean, var = model_nargp.prediction(x_pred)
```
- `mean` : posterior mean  
- `var` : posterior variance  
- `x_pred` : evaluation points

---

## Dependencies

mfmodeling requires the following Python packages:
- `numpy`, `gpytorch`
- (optional) `matplotlib` for visualization in `examples/`
- (optional) `pandas`, `scikit-learn` for `examples/mauna_loa_data/`

---

# Citation

If you use **mfmodeling** in your research, please cite:

```bibtex
@article{maeyama2024multifidelity,
  author  = {Maeyama, Shinya and Honda, Mitsuru and Narita, Emi and Toda, Shinichiro},
  title   = {Multi-Fidelity Information Fusion for Turbulent Transport Modeling in Magnetic Fusion Plasma},
  journal = {Scientific Reports},
  volume  = {14},
  pages   = {28242},
  year    = {2024},
  doi     = {10.1038/s41598-024-78394-3}
}
```

[![doi](https://img.shields.io/badge/doi-10.1038/s41598--024--78394--3-5077AB.svg)](https://doi.org/10.1038/s41598-024-78394-3)

---

## License

This project is licensed under the MIT License. See the [`LICENSE`](LICENSE) file for details.

## Author

Developed by Shinya Maeyama (maeyama.shinya@nifs.ac.jp)
