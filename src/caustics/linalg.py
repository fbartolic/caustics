from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import numpy as np
import jax.numpy as jnp
from jax import  jit 


@partial(jit, static_argnames=("dense_covariance",))
def marginalized_log_likelihood(
    A_list, fobs_list, C_inv_list, dense_covariance=False, Lam_sd=1e04
):
    """
    Compute the log-likelihood for a microlensing light curve marginalized over
    over the linear flux parameters $\\boldsymbol\\beta\equiv(F_s, F_b)^\intercal$. 
    The function takes a list of magnification vectors, a list of observed 
    fluxes and a list of inverse data covariance matrices as an input, one element 
    of the list represents and independent set of observations, for instance light 
    curves from different observatories. The total log-likelihood is then a sum 
    of the log-likelihoods for each light curve.

    If `dense_covariance` is False, the inverse data covariance matrices 
    are assumed to be 1D vectors containing the elements on the diagonal. In 
    that case, the most efficient way to marginalize over the linear parameters 
    in the likelihood is to solve the linear least squares problem conditional 
    on fixed values of the nonlinear parameters (which determine the 
    magnification). If `dense_covariance` is True, the inverse covariance matrices 
    are assumed to be dense matrices. In that case we have to compute the full
    marginal likelihood. This increases the computational cost of the likelihood 
    evaluation.

    Args:
        A_list (list[array_like]): List of 1D magnification arrays, one per 
            independent observation.
        fobs_list (list[array_like]): List of 1D observed flux arrays. 
        C_inv_list (list[array_like]): List of nverse data covariance matrices. 
            If `dense_covariance`is False, this is a 1D vector containing the 
            diagonal elements of the
            covariance matrix. If `dense_covariance` is True, this is a dense
            matrix.
        dense_covariance (bool, optional): Flag indicating whether `C_inv` is 
            dense or diagonal. Defaults to False.
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
    # Solve for the linear parameters with linear least squares
    if dense_covariance is False:
        ll = 0.
        beta_list = []
        for A, fobs, C_inv in zip(A_list, fobs_list, C_inv_list):
            M = jnp.stack([A, jnp.ones_like(A)]).T
            MTCinvM = M.T @ jnp.diag(C_inv) @ M
            MTCinvM_inv = jnp.linalg.solve(MTCinvM, jnp.eye(2))
            beta = MTCinvM_inv @ M.T @ jnp.diag(C_inv) @ fobs[:, None]
            fpred = (M @ beta).reshape(-1)

            # Compute the likelihood
            Sigma = MTCinvM_inv
            ll += -0.5*jnp.sum((fobs - fpred)**2*C_inv) + 0.5*jnp.log(jnp.linalg.det(2*np.pi*Sigma))
            beta_list.append(beta.reshape(-1))
        return beta_list, ll

    # Do full analytic marginalization
    else:
        ll = 0.
        for A, fobs, C_inv in zip(A_list, fobs_list, C_inv_list):
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
            log_det_cov = jnp.log(C_det) + jnp.log(Lam_diag).sum() + jnp.log(jnp.linalg.det(Lam_diag + MTCinvM))

            # Compute the likelihood 
            r = fobs - M @ mu
            ll += -0.5*r.T @ cov_inverse @ r - 0.5*log_det_cov

        return 0., ll
 