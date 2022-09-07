# Linear algebra 
This module contains a function `marginalized_log_likelihood` which computes
the log likelihood of a microlensing model marginalized over the linear flux
parameters $\beta\equiv(F_s,F_b)^\intercal$. This function is intended to be 
used a as replacement for the standard likelihood function when optimizing 
the likelihood or doing MCMC.

The marginalization is standard practice when fitting microlensing events 
because it reduces the number of 
parameters in the model by the number of linear parameters in the model. For 
light curves consisting of multiple independent observations (for example, those 
observed by different observatories) this is a necessary step because there 
could be dozens of linear parameters in the model. The way this is traditionally 
done in microlensing is by solving for the linear parameters conditional on 
fixed values of the nonlinear parameters using linear least squares. This 
procedure is justified in some circumstances but it is not valid if the 
data covariance matrix is dense (for example, if we are using a Gaussian Process
to model correlated noise in the light curves or stellar variability of the source 
star). Below, I show how to analytically marginalize over the linear parameters 
in the general case.


The observed flux $\mathbf f$ can be written as a linear model 
\begin{equation}
\mathbf f = \mathbf M\,\boldsymbol\beta
\end{equation}
where $\mathbf M$ is the design matrix which depends on some nonlinear parameters $\boldsymbol\theta$:

\begin{equation}
    \mathbf{M}\equiv \begin{pmatrix}
        \tilde{A}(t_1;\boldsymbol\theta)     & 1   \\
        \tilde{A}(t_2;\boldsymbol\theta)     & 1  \\
        \vdots                          & \vdots \\
        \tilde{A}(t_{N};\boldsymbol\theta) & 1
    \end{pmatrix}
\end{equation}

Assuming that we place a Gaussian prior on the linear parameters $\boldsymbol\beta$
with mean $\boldsymbol\mu$ and covariance $\boldsymbol\Lambda$, [one can show](https://arxiv.org/abs/2005.14199) 
that the marginal likelihood 
$\ln\int p(\mathbf f|\boldsymbol\theta)\,p(\boldsymbol\beta|\boldsymbol\theta)d\boldsymbol\beta$
is given by

\begin{equation} 
\mathrm{LL}(\boldsymbol\theta)=-\frac{1}{2}\left( \mathbf{f}-\mathbf{M}\boldsymbol{\mu}\right)^\intercal\left(\mathbf{C} + \mathbf{M}\boldsymbol{\Lambda}\mathbf{M}^\intercal\right)^{-1}
\left( \mathbf{f}-\mathbf{M}\boldsymbol{\mu}\right)     
- \frac{1}{2}\ln\left| \mathbf{C} + \mathbf{M}\boldsymbol{\Lambda}\mathbf{M}^\intercal\right|
\end{equation}

To compute the inverse and the determinant of the covariance matrix of marginalized likelihood we 
can use the matrix inversion lemma (see Appendix A3 of [R&W](https://gaussianprocess.org/gpml/)):

\begin{align}
     & \left(\mathbf{C}+\mathbf{M} \boldsymbol{\Lambda} \mathbf{M}^{\intercal}\right)^{-1}=\mathbf{C}^{-1}-\mathbf{C}^{-1} \mathbf{M}\left(\boldsymbol{\Lambda}^{-1}+\mathbf{M}^{\intercal} \mathbf{C}^{-1} \mathbf{M}\right)^{-1} \mathbf{M}^{\intercal} \mathbf{C}^{-1} \\
     & \ln\left|\mathbf{C}+\mathbf{M} \mathbf{\Lambda} \mathbf{M}^{\intercal}\right|=\ln|\mathbf{C}| +\ln|\boldsymbol{\Lambda}| + \ln\left|\boldsymbol{\Lambda}^{-1}+\mathbf{M}^{\intercal} \mathbf{C}^{-1} \mathbf{M}\right|
\end{align}

However, if we marginalize over the linear parameters, how can we obtain their values in order to, for example, produce 
a plot of the model fit? The answer, from the paper linked above, is that the distribution over $\boldsymbol\beta$
conditional on particular values of the nonlinear parameters $\boldsymbol\theta$ is a Gaussian 
$\mathcal{N}(\boldsymbol\beta;\mathbf a,\mathbf A)$ where

\begin{align}
&\mathbf{A}^{-1}=\boldsymbol\Lambda^{-1}+\mathbf{M}^{\intercal}\mathbf{C}^{-1}\mathbf{M} \\
&\mathbf a = \mathbf A\left(\boldsymbol\Lambda^{-1}\boldsymbol\mu+\mathbf{M}^{\intercal}\mathbf{C}^{-1}\mathbf{f}\right)
\end{align}

If we had posterior samples of the nonlinear parameters $\boldsymbol\theta$ we could 
could use this distribution to generate samples of the linear parameters.

In `marginalized_log_likelihood` I use the standard least squares solution for $\boldsymbol\beta$ 
if the data covariance matrix is diagonal because it's faster than the matrix 
operations in $\mathrm{LL}(boldsymbol\theta)$. Otherwise I use the full analytic marginalization.

::: caustics.linalg
    selection:
        members:
        - marginalized_log_likelihood 

