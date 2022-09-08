# Trajectory
This module implements functions for computing the trajectory of the source star 
in the source plane. For now, only the annual parallax trajectory is implemented 
(`caustics.trajectory.AnnualParallaxTrajectory`).

## Annual parallax
If one were to observe a source star from a barycentric frame of reference, in absence of 
acceleration of the source star (due to the source star being in a binary system), the 
apparent motion of the source star on the plane of the sky would be rectilinear. However,
we do not observe the stars from a barycentric frame of reference, we observe them from a 
non-intertial geocentric frame (Earth). \emph{Annual parallax} is the apparent motion of
star due to the Earth's motion around the Sun. It is important for longer timescale events 
when the event timescale is equal to some non-negligible fraction of the Earth's orbital
period.

In a barycentric frame of reference we can write down the position of the source star 
$\mathbf w_S(t)$ and the lens star $\mathbf w_L(t)$ as 

\begin{align}
&\mathbf{w}_S(t)=\mathbf{w}_{S, 0}+\left(t-t_0\right) \boldsymbol{\mu}_S \\
&\mathbf{w}_L(t)=\mathbf{w}_{L, 0}+\left(t-t_0\right) \boldsymbol{\mu}_L
\end{align}

where $\mu_S$ and $\mu_L$ are the proper motion vectors of the source and lens 
stars, respectively and $t_0$ is the time of closest approach of the source to the lens.
The relative position vector with respect to the source in units of Einstein radii is 

\begin{equation}
\boldsymbol{u}(t) \equiv \frac{\mathbf{w}_L(t)-\mathbf{w}_S(t)}{\theta_E}=\frac{\mathbf{w}_{L S, 0}}{\theta_E}+\frac{t-t_0}{\theta_E} \boldsymbol{\mu}_{L S}
\end{equation}

where $\mathbf{w}_{L S, 0} \equiv \mathbf{w}_{L, 0}-\mathbf{w}_{S, 0}$ and 
$\boldsymbol{\mu}_{L S} \equiv \boldsymbol{\mu}_L-\boldsymbol{\mu}_S$.
When observing the source star from Earth, the from Earth the apparent position of the star 
is shifted by the position vector of the Sun relative to the Earth $\mathbf s$ onto a 
Geocentric coordinate system on the plane of the sky which is defined at some reference 
time $t_0^\prime$ by the unit vector $\hat{\mathbf n}$ normal to the plane of the 
sky at the source star (a function of the celestial coordinates of the source star) 
and the unit vectors 
$\hat{\mathbf e}_n$ and $\hat{\mathbf e}_e$, pointing in the direction of the celestial
north and east, respectively. The coordinate system is defined to be right-handed. The 
unit vectors $\hat{\mathbf e}_n$ and $\hat{\mathbf e}_e$ are thus

\begin{align}
\hat{\mathbf{e}}_e &=\hat{\mathbf{z}} \times \hat{\mathbf{n}} \\
\hat{\mathbf{e}}_n &=\hat{\mathbf{n}} \times \hat{\mathbf{e}}_e
\end{align}

where $\hat{\mathbf n}$ is the 3D unit vector pointing the direction of the source star
and $\hat{\mathbf z}=(0,0,1)$. We can compute the 3D position vector of the Sun 
$\mathbf s(t)$ (and its time derivative) using the `astropy` function
[`astropy.coordinates.get_body_barycentric_posvel`](https://docs.astropy.org/en/stable/api/astropy.coordinates.get_body_barycentric_posvel.html)
at arbitrary times $t$ and obtain its projection on the plane of the sky by dotting it 
into the unit vectors: 

\begin{align}
\zeta_e(t ; \alpha, \delta) & \equiv \mathbf{s} \cdot \hat{\mathbf{e}}_e \\
\zeta_n(t ; \alpha, \delta) &\equiv \mathbf{s} \cdot \hat{\mathbf{e}}_n
\end{align}

The sky positions of the source and lens stars in the geocentric frame of reference defined at 
time $t_0^\prime$ are then given by

\begin{align}
&\mathbf{w}_S(t)=\mathbf{w}_{S, 0}+\left(t-t_0^{\prime}\right) \boldsymbol{\mu}_S+\pi_S \boldsymbol{\zeta}(t) \\
&\mathbf{w}_L(t)=\mathbf{w}_{L, 0}+\left(t-t_0^{\prime}\right) \boldsymbol{\mu}_L+\pi_L \boldsymbol{\zeta}(t)
\end{align}

where $\pi_S \equiv 1 \mathrm{au} / D_S$ is the source parallax and $\pi_L \equiv 1 \mathrm{au} / D_L$ is the lens parallax. The relative separation vector is 

\begin{equation}
\boldsymbol{u}(t)=\frac{\mathbf{w}_{L S, 0}}{\theta_E}+\frac{t-t_0^{\prime}}{\theta_E} \boldsymbol{\mu}_{L S}+\pi_E \boldsymbol{\zeta}(t)
\end{equation}

where $\pi_E\equiv \pi_{LS}/\theta_E$. Because parallax is usually a small effect, it makes 
sense to decompose the trajectory as a sum of rectilinear motion plus a deviation due to 
parallax
(see [An et al. 2002](https://ui.adsabs.harvard.edu/abs/2002ApJ...572..521A/abstract) for example). Mathematically, 

\begin{align}
    \mathbf{u}(t_0^\prime)&\equiv \mathbf{u}_0=\mathbf{w}_{LS}/\theta_E+\pi_E\mathbf{\zeta}(t_0^\prime)\\
    \dot{\mathbf u}(t_0^\prime)&\equiv \dot{\mathbf u}_0=\mathbf{\mu}_{LS}+\pi_E\dot{\mathbf{\zeta}}(t_0^\prime)
\end{align}

It then follows that  

\begin{equation}
    \boldsymbol{u}(t)=\mathbf{u}(t_0^\prime) + (t-t_0')\,\dot{\mathbf{u}}(t_0^\prime) +
    \pi_E\,\delta\mathbf\zeta(t)
\end{equation}

where 

\begin{equation}
    \delta\boldsymbol \zeta (t)=\boldsymbol \zeta (t)-\boldsymbol \zeta (t_0')-(t-t_0')
    \boldsymbol{\dot \zeta} (t_0')
    \label{eq:relative_separation_parallax_decomposed}
\end{equation}

is the position offset of the Sun on the plane of the sky relative to its position 
at the reference time $t_0^\prime$. By construction, we have 
$\delta\boldsymbol \zeta (t_0')=0$ and $\delta\dot{\boldsymbol \zeta} (t_0')=0$.
At the reference time $t_0^\prime$, the vectors $\mathbf{u}(t_0^\prime)$ and 
$\dot{\mathbf u}(t_0^\prime)$ are perpendicular to each other.

To evaluate this expression for the trajectory we need to choose a coordinate system.
A natural coordinate system for describing the trajectory of the source relative to the 
lens is one defined by unit vectors
$(\mathbf{\hat e}_\bot,\mathbf{\hat e}_\parallel)$ where $\mathbf{\hat e}_\parallel$ 
is parallel to the trajectory $\mathbf{u}(t)$ at time $t_0^\prime$.
We define the unit vectors as

\begin{equation}
    \mathbf{\hat e}_\bot\equiv \frac{\mathbf{u}_0}{|\mathbf{u}_0|},\quad
    \mathbf{\hat e}_\parallel\equiv \frac{\mathbf{\hat n}\times\mathbf{u}_0}{|\mathbf{u}_0|}\quad
\end{equation}

The coordinate system $(\mathbf{\hat e}_\bot,\mathbf{\hat e}_\parallel)$ is related to 
ecliptic coordinates by a simple rotation through an angle $\psi$.

By construction, at time $t_0^\prime$ we have $\mathbf{u}_0\,\bot\,\dot{\mathbf{u}}_0$ and 
the two components of $\mathbf{u}(t)$ are then

\begin{align}
    u_\bot(t)      & \equiv \mathbf{u}(t)\cdot \mathbf{\hat e}_\bot= u_0 +
    \pi_E\,\delta\boldsymbol \zeta(t)\cdot\mathbf{\hat e}_\bot                                                                                                                                       \\
    u_\parallel(t) & \equiv \mathbf{u}(t)\cdot \mathbf{\hat e}_\parallel= (t-t_0^\prime)\,\dot{\mathbf{u}}_0\cdot\mathbf{\hat e}_\parallel+ \pi_E\,\delta\boldsymbol \zeta(t)\cdot\mathbf{\hat e}_\parallel
\end{align}

using the definitions of the unit vectors and
$\delta\boldsymbol \zeta(t)=
    \delta \zeta_e(t)\,\mathbf{\hat e}_e+\delta \zeta_n(t)\,\mathbf{\hat e}_n$, we obtain

\begin{align}
    u_\bot(t)      & = u_0 + \pi_E\,\cos\psi\,\delta \zeta_e(t) - \pi_E\,\sin\psi\,\delta \zeta_n(t)
    \label{eq:u_t_parallel1}                                                                         \\
    u_\parallel(t) & =(t-t_0^\prime)/t_E^\prime + \pi_E\,\sin\psi\,\delta \zeta_e(t) +
    \pi_E\,\cos\psi\,\delta \zeta_n(t) \label{eq:u_t_parallel2}
\end{align}

where 

\begin{equation}
t_E^\prime =|\dot{\mathbf{u}}_0|=|\boldsymbol\mu_{LS}/\theta_E  + \pi_E\,\dot{\boldsymbol \zeta}(t_0^\prime)|
\end{equation}

The model parameters which determine the magnification $A(t)$ are then
$\left(u_0,t_0^\prime,t'_E,\pi_E,\psi\right)$. Notice that $u_0$ in this case can also be negative.

An alternative parametrization which is more commonly used in the literature is obtained by
defining components of a ``microlensing parallax vector'' as

\begin{align}
    \pi_{E,N} & \equiv \pi_E\cos\psi \\
    \pi_{E,E} & \equiv \pi_E\sin\psi
\end{align}

in which case we have

\begin{align}
    u_\bot(t)      & = u_0 + \pi_{E,N}\,\delta \zeta_e(t) - \pi_{E,E}\,\delta \zeta_n(t) \\
    u_\parallel(t) & =(t-t_0^\prime)/t_E^\prime + \pi_{E,E}\,\delta\zeta_e(t) +
    \pi_{E,N}\,\delta\zeta_n(t)
\end{align}

and the model parameters are $\left(u_0,t_0^\prime,t_E^\prime,\pi_{E,E},\pi_{E,N}\right)$.

Finally, we can also write down the expression for the trajectory vector in equatorial 
coordinates by applying a rotation matrix to the above vector, the result is 

\begin{align}
    u_e & =u_0\cos\psi + (t-t_0^\prime)/t_E^\prime\sin\psi + \pi_E\,\delta\zeta_e(t)  \\
    u_n & =-u_0\sin\psi + (t-t_0^\prime)/t_E^\prime\cos\psi + \pi_E\,\delta\zeta_n(t)
\end{align}

This last expression is what is computed in the `AnnualMicrolensingParallax` via the 
`compute()` method. One can either pass the parameters $\pi_{E,E}$ and $\pi_{E,N}$ or 
the angle $\psi$ and the magnitude $\pi_E$. The vector 
$(\delta\zeta_e(t), \delta\zeta_n(t))^\intercal$ is precomputed on a grid in time 
which spans the time interval of the light curve with a timestep of 1 day. It is then
interpolated at arbitrary times when the `compute()` method is called. This operation
is very cheap.

::: caustics.trajectory
    selection:
        members:
        - AnnualParallaxTrajectory 
