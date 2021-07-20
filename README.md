This Python package contains the data and code necessary to reproduce results found in the Nature Communications paper 'Turn-Key Constrained Parameter Space Exploration for Particle Accelerators Using Bayesian Active Learning'.
It also contains a simple implementation of CPBE algorithm and a test problem to demonstrate the algorithm's effectiveness.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the development environment](#setting-up-the-development-environment)
- [License](#license)
- [Issues](https://github.com/neurodata/mgcpy/issues)

# Overview
The ``CPBE`` algorithm aims to be an algorithm that replaces common multi-parameter scans in order to characterize a target function. It adaptively samples input space to maximize information gain about the target function, respects unknown constraints in input space, and biases towards making small jumps in input space.


# System Requirements
## Hardware requirements
`CPBE` requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for any systems that can run python > 3.7. The package has been tested on the following systems:
+ Windows 10: Enterprise

### Python Dependencies
Code here mainly depends on the Python scientific stack.

```
numpy~=1.21.0
matplotlib~=3.4.2
pandas~=1.3.0
botorch~=0.4.0
gpytorch~=1.5.0
```

# Installation Guide:

### Install from Github
```
git clone https://github.com/roussel-ryan/turn_key_bayesian_exploration.git
cd turn_key_bayesian_exploration
python3 setup.py install
```

# Setting up the environment:
- Install Miniconda https://docs.conda.io/en/latest/miniconda.html
```
conda env create -f environment.yml
conda activate cpbe
```

# License

This project is covered under the **Apache 2.0 License**.