This Python package contains the data and code necessary to reproduce results found in the Nature Communications paper 'Turn-Key Constrained Parameter Space Exploration for Particle Accelerators Using Bayesian Active Learning'.
It also contains a simple implementation of CPBE algorithm and a test problem to demonstrate the algorithm's effectiveness.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Setting up the environment](#setting-up-the-environment)
- [License](#license)

# Overview
The ``CPBE`` algorithm aims to be an algorithm that replaces common multi-parameter scans in order to characterize a target function. It adaptively samples input space to maximize information gain about the target function, respects unknown constraints in input space, and biases towards making small jumps in input space.
## Repository contents
- data/ contains raw experimental measurements of the beam emittance as a function of input parameters for both the 2D scan case and the 4D CPBE case.
- demo/ contains an implementation of CPBE and a script to demonstrate its use in a simple exploration problem (see README in folder for details)
- plotting/ contains scripts used to generate plots in the paper

# System Requirements
## Hardware requirements
`CPBE` requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
This package is supported for any systems that can run python > 3.7. The package has been tested on the following systems:
+ Windows 10: Enterprise

# Installation Guide:

### Install from Github
Should take < 1 min
```
git clone https://github.com/roussel-ryan/turn_key_bayesian_exploration.git
cd turn_key_bayesian_exploration
python3 setup.py install
```

# Setting up the environment:
- Install Miniconda https://docs.conda.io/en/latest/miniconda.html
- Create and activate environment
- Should take < 10 min
```
conda env create -f environment.yml
conda activate cpbe
```

# License

This project is covered under the **MIT License**.
