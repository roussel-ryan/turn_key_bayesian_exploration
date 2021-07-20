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
numpy
scipy
Cython
scikit-learn
pandas
seaborn
```

# Installation Guide:

### Install from PyPi
```
pip3 install mgcpy
```

### Install from Github
```
git clone https://github.com/neurodata/mgcpy
cd mgcpy
python3 setup.py install
```
- `sudo`, if required
- `python3 setup.py build_ext --inplace  # for cython`, if you want to test in-place, first execute this

# Setting up the development environment:
- To build image and run from scratch:
  - Install [docker](https://docs.docker.com/install/)
  - Build the docker image, `docker build -t mgcpy:latest .`
    - This takes 10-15 mins to build
  - Launch the container to go into mgcpy's dev env, `docker run -it --rm --name mgcpy-env mgcpy:latest`
- Pull image from Dockerhub and run:
  - `docker pull tpsatish95/mgcpy:latest` or `docker pull tpsatish95/mgcpy:development`
  - `docker run -it --rm -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:latest` or `docker run -it --rm -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:development`


- To run demo notebooks (from within Docker):
  - `cd demos`
  - `jupyter notebook --ip 0.0.0.0 --no-browser --allow-root`
  - Then copy the url it generates, it looks something like this: `http://(0de284ecf0cd or 127.0.0.1):8888/?token=e5a2541812d85e20026b1d04983dc8380055f2d16c28a6ad`
  - Edit this: `(0de284ecf0cd or 127.0.0.1)` to: `127.0.0.1`, in the above link and open it in your browser
  - Then open `mgc.ipynb`

- To mount/load local files into docker container:
  - Do `docker run -it --rm -v <local_dir_path>:/root/workspace/ -p 8888:8888 --name mgcpy-env tpsatish95/mgcpy:latest`, replace `<local_dir_path>` with your local dir path.
  - Do `cd ../workspace` when you are inside the container to view the mounted files. The **mgcpy** package code will be in `/root/code` directory.


## MGC Algorithm's Flow
![MGCPY Flow](https://raw.githubusercontent.com/neurodata/mgcpy/master/MGCPY.png)

## Power Curves
- Recreated Figure 2 in https://arxiv.org/abs/1609.05148, with the addition of MDMR and Fast MGC
![Power Curves](https://raw.githubusercontent.com/neurodata/mgcpy/master/power_curves_dimensions.png)

# License

This project is covered under the **Apache 2.0 License**.
