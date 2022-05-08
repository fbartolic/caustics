# caustics
`caustics` enables computation of microlensing light curves for binary and triple lens systems using the method of [contour integration](https://academic.oup.com/mnras/article-abstract/503/4/6143/6149166?redirectedFrom=fulltext&login=false). It is built on top of the [JAX](https://github.com/google/jax) library which enables efficient computation of *exact* gradients of the model with respect to all input parameters through the use of [automatic differentiation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html). 

## Features
- Accurate computation of binary and triple lens microlensing light curves for extended limb-darkend sources in a fraction of a second using either a CPU or a GPU backend.
- Evaluation of the gradients of all model outputs enables the use of gradient-based inference methods such as Hamiltonian Monte Carlo when fitting multiple lens microlensing light curves.
- A differentiable JAX/CUDA version of a complex polynomial root solver [CompEA](https://github.com/trcameron/CompEA) which uses the Ehrlich-Aberth method. Running the solver on an NVIDIA GPU enables evaluation of >100k binary or triple lens equation solutions in miliseconds. The gradient of the solutions with respect to the polynomial coefficients is obtained through [implicit differentiation](http://implicit-layers-tutorial.org/implicit_functions/).
- Hexadecapole approximation from [Cassan 2017](https://academic.oup.com/mnras/article/468/4/3993/3103057?login=true) is used to substantially speed up the computation of magnification outside of the regions close to the caustics.


## References
- `caustics` paper coming soon!
- [Light-curve calculations for triple microlensing systems](https://academic.oup.com/mnras/article-abstract/503/4/6143/6149166?redirectedFrom=fulltext&login=false)
- [On a compensated Ehrlich-Aberth method for the accurate computation of all polynomial roots](https://hal.archives-ouvertes.fr/hal-03335604)
- [A robust and efficient method for calculating the magnification of extended sources caused by gravitational lenses](https://ui.adsabs.harvard.edu/abs/1998A%26A...333L..79D/abstract)
- [VBBINARYLENSING: a public package for microlensing light-curve computation](https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.5157B/abstract)

