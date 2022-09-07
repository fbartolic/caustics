from functools import partial
from jax.config import config
config.update("jax_enable_x64", True)
config.update('jax_platform_name', 'cpu')
import numpy as np
import jax.numpy as jnp
from jax import  jit 


@partial(jit, static_argnames=("dense_covariance",))
def marginalized_log_likelihood(A, fobs, C_inv, dense_covariance=False, Lam_sd=1e04):
    """
    Compute the log-likelihood for a microlensing light curve marginalized over
    over the linear flux parameters $\\boldsymbol\\beta\equiv(F_s, F_b)^\intercal$. 
    If `dense_covariance` is False, the inverse data covariance matrix `C_inv` 
    is assumed to be a 1D vector containing the elements on the diagonal. In 
    that case, the most efficient way to marginalize over the linear parameters 
    in the likelihood is to solve the linear least squares problem conditional 
    on fixed values of the nonlinear parameters (which determine the 
    magnification `A`). If `dense_covariance` is True, `C_inv` is assumed to be 
    a dense matrix and in that case we have to compute the full marginal 
    likelihood. This increases the computational cost of the likelihood 
    evaluation.

    Args:
        A (array_like): 1D magnification array.
        fobs (array_like): 1D observed flux array.
        C_inv (array_like): Inverse data covariance matrix. If `dense_covariance`
            is False, this is a 1D vector containing the diagonal elements of the
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
    # Compute matrices relevant matrices 
    M = jnp.stack([A, jnp.ones_like(A)]).T

    # Solve for the linear parameters with linear least squares
    if dense_covariance is False:
        MTCinvM = M.T @ jnp.diag(C_inv) @ M
        MTCinvM_inv = jnp.linalg.solve(MTCinvM, jnp.eye(2))
        beta = MTCinvM_inv @ M.T @ jnp.diag(C_inv) @ fobs[:, None]
        fpred = (M @ beta).reshape(-1)

        # Compute the likelihood
        Sigma = MTCinvM_inv
        ll = -0.5*jnp.sum((fobs - fpred)**2*C_inv) + 0.5*jnp.log(jnp.linalg.det(2*np.pi*Sigma))
        return beta, ll

    # Do full analytic marginalization
    else:
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
        ll = -0.5*r.T @ cov_inverse @ r - 0.5*log_det_cov

        return 0., ll
 