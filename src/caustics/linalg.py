from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import numpy as np
import jax.numpy as jnp
from jax import  jit 

def weighted_least_squares(y, y_err, M):
    C_inv = 1/y_err**2
    MTCinvM = M.T @ jnp.diag(C_inv) @ M
    beta = jnp.linalg.solve(MTCinvM, M.T @ jnp.diag(C_inv) @ y[:, None])
    return beta


@jit
def marginalized_log_likelihood(
    M_list, fobs_list, C_inv_list,  Lam_sd=1e04, 
):
    """
    Compute the (pointwise) log-likelihood for a microlensing light curve 
    marginalized over over the linear flux parameters 
    $\\boldsymbol\\beta\equiv(F_s, F_b)^\intercal$. The function takes a list 
    of magnification vectors, a list of observed fluxes and a list of inverse 
    data covariance matrices as an input, one element of the list represents 
    and independent set of observations, for instance light curves from different
    observatories. The total log-likelihood is then a sum of the log-likelihoods
    for each light curve.

    The inverse covariance matrices are assumed to be dense matrices. In that if
    you have a diagonal covariance matrix you should use the function 
    `weighted_least_squares` instead because it is much faster and equivalent to
    analytic marginalization. 

    Args:
        M_list (list[array_like]): List of matrices which map the linear flux
            parameters to the observed fluxes. Each matrix should have shape
            `(n_obs, 2)`.
        fobs_list (list[array_like]): List of 1D observed flux arrays. 
        C_inv_list (list[array_like]): List of nverse data covariance matrices. 
            If `dense_covariance`is False, this is a 1D vector containing the 
            diagonal elements of the
            covariance matrix. If `dense_covariance` is True, this is a dense
            matrix.
        Lam_sd (float, optional): Standard deviation of the Gaussian prior on 
            the linear flux parameters. The prior is assumed to be a zero mean
            independent Gaussian with standard deviation `Lam_sd` for both linear 
            parameters. Defaults to 1e04.

    Returns:
        tuple: If `dense_covariance` is False, returns a tuple 
        $(\\boldsymbol\\beta, \mathrm{LL}(\\boldsymbol\\theta))$ containing the 
        least-squares solution for the linear parameters and the log-likelihood.
        Otherwise, the first element of the tuple is 0. 
    """
    ll_list = []
    for M, fobs, C_inv in zip(M_list, fobs_list, C_inv_list):
        # Prior mean and (assumed to be diagonal) covariance
        Lam_diag = Lam_sd**2*jnp.ones(2)
        Lam_inv = jnp.diag(1./Lam_diag)
        mu = jnp.zeros(2)
        C = jnp.linals.solve(C_inv, jnp.eye(C_inv.shape[0]))
        C_det = jnp.linalg.det(C)

        # Inverse covariance matrix of the marginalized log-likelihood
        MTCinvM = M.T @ C_inv @ M
        cov_inverse = C_inv -\
            C_inv @ M @ jnp.linalg.solve(Lam_inv + MTCinvM, M.T @ C_inv)
        # Log determinant of the marginalized log-likelihood
        log_det_cov = jnp.log(C_det) + jnp.log(Lam_diag).sum() +\
                jnp.log(jnp.linalg.det(Lam_diag + MTCinvM))

        # Compute the likelihood 
        r = fobs - M @ mu
        ll = -0.5*r.T @ cov_inverse @ r - 0.5*log_det_cov
        ll_list.append(ll)

    return 0., ll_list
