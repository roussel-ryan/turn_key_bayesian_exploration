# Algorithm Demonstration Scrips

## Lengthscale Demonstration - lengthscale_demo_with_optimization.py
Reproduces Figure 1 from the paper, showing how a posterior acquisition function behaves when the GP length scales for each axis are isotropic or anisotropic.

## CPBE Demonstration - example_search.py
Demonstrates how CPBE behaves when exploring a simple test problem. It tries to explore the function f(x_1,x_2) = \sin(2\pi x_1) \sin(\pi x_2) while respecting the binary constraint defined by the region (x_1 - 0.5)^2 + (x_2 - 0.5)^2 < 0.35^2 U x_1 < x_2. The algorithm uses a proximal biasing term where \Sigma = 0.3^2 I and one initial valid sample.
