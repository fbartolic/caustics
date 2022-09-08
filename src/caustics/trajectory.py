# -*- coding: utf-8 -*-
__all__ = [
    "trajectory",
]

import numpy as np
import jax.numpy as jnp

from astropy.coordinates import get_body_barycentric_posvel
from astropy.time import Time

class AnnualParallaxTrajectory:
    """
    Compute the trajectory of the source star on the plane of the sky while 
    taking into account the annual parallax effect -- the fact that the apparent
    position of the source star on the plane of the sky is shifted by the 
    projected position vector of the Sun relative to the Earth.

    Example usage:
    ```Python
    trajectory = AnnualParallaxTrajectory(t, coords)
    w_points = trajectory.compute(t, t0=t0, tE=tE, u0=u0, piEE=piEE, piEN=piEN)
    ```
    """
    def __init__(self, t, coords):
        """
        Args:
            t (ndarray): Array containing the observation times in HJD format.
            coords (astropy.coordinates.SkyCoord): The coordinates of the source
                star.
        """
        self.t = t

        # Resample time uniformly between the beginning and end of in timestep 
        # of 1 day
        self.t_jpl = np.arange(t[0], t[-1] + 1, 1)

        self.coords = coords
        self.s_e = None  # Celestial East component of Sun position
        self.s_n = None  # Celestial North component of Sun position
        self.s_e_dot = None
        self.s_n_dot = None

        self._compute_sun_position_and_velocity()
        
    def _project_vector_onto_sky(self, vec):
        """
        This function takes a 3D cartesian vector specified in the ICRS coordinate 
        system and evaluated at times :math:`(t_i,\dots, t_N)` and projects it onto 
        a spherical coordinate system (the geocentric equatorial coordinate system)
        on the plane of the sky with the origin at the position defined by 
        `self.coords`.
        
        Args:
            vec (np.ndarray): An (n,3) 3D cartesian vector evaluated at `n` times.
        Returns: 
            tuple: A tuple (east_component, north_component). The projected 
            vector in geocentric equatorial coordinates.
        """
        # Unit vector normal to the plane of the sky in ICRS coordiantes
        direction = np.array(self.coords.cartesian.xyz.value)
        direction /= np.linalg.norm(direction)

        # Unit vector pointing north in ICRS coordinates
        e_north = np.array([0.0, 0.0, 1.0])

        # Spherical unit vectors of the coordinate system defined
        # on the plane of the sky which is perpendicular to the
        # source star direction
        e_east_sky = np.cross(e_north, direction)
        e_north_sky = np.cross(direction, e_east_sky)
        e_east_sky = e_east_sky / np.linalg.norm(e_east_sky)
        e_north_sky = e_north_sky / np.linalg.norm(e_north_sky)

        east_component = np.dot(vec, e_east_sky)
        north_component = np.dot(vec, e_north_sky)

        return east_component, north_component
    

    def _compute_sun_position_and_velocity(self):
        """
        Compute the position vector of the Sun relative to Earth projected 
        onto the plane of the sky (the geocentric equatorial coordinate system).
        """
        # Use NASA's JPL Horizons and compute orbital Earth's orbit at
        # observed times
        times = Time(self.t_jpl + 2450000, format="jd", scale="tdb")

        # Get Earth's position and velocity from JPL Horizons
        pos, vel = get_body_barycentric_posvel("earth", times)

        # Minus sign because `get_body_barycentric` returns Earth position
        # in heliocentric coordinates
        s_t = -pos.xyz.value.T
        v_t = -vel.xyz.value.T

        # Project position and velocity vector onto the plane of the sky
        self.s_e, self.s_n = self._project_vector_onto_sky(
            s_t, 
        )
        self.s_e_dot, self.s_n_dot = self._project_vector_onto_sky(
            v_t, 
        )
    
    def _compute_delta_sun_position_and_velocity(self, t, t0):
        # Interpolate Sun position at times t and t0
        s_e_t = jnp.interp(t, self.t_jpl, self.s_e)
        s_n_t = jnp.interp(t, self.t_jpl, self.s_n)

        s_e_t0 = jnp.interp(t0, self.t_jpl, self.s_e)
        s_n_t0 = jnp.interp(t0, self.t_jpl, self.s_n)
        s_e_dot_t0 = jnp.interp(t0, self.t_jpl, self.s_e_dot)
        s_n_dot_t0 = jnp.interp(t0, self.t_jpl, self.s_n_dot)

        # Compute deviation from rectilinear motion
        delta_s_e = s_e_t - s_e_t0 - (t - t0) * s_e_dot_t0
        delta_s_n = s_n_t - s_n_t0 - (t - t0) * s_n_dot_t0

        return delta_s_e, delta_s_n

    def compute(self, t, parametrization="cartesian", **params):
        """
        Compute the trajectory of the source star in a Geocentric Equatorial 
        coordinate system centered on the source star.

        Args:
            t (array_like): An array of times at which to evaluate the trajectory.
            parametrization (str, optional): If "cartesian" `**params` should 
            contain parameters `piEE` and `piEN`, otherwise `**params` should
            contain parameters `psi` and `piE` where $\pi_{EE}=\pi_E\sin\psi$ and
            $\pi_{EN}=\pi_E\cos\psi$. Defaults to "cartesian".

        Returns:
            array_like: A complex array of shape (len(t),) where the real part 
                is the East component and the imaginary part is the North
                component of the source trajectory in the source plane.
        """
        if parametrization == "polar":
            params['piEE']  = params['piE']*jnp.sin(params['psi'])
            params['piEN']  = params['piE']*jnp.cos(params['psi'])
        elif parametrization == "cartesian":
            params['psi'] = jnp.arctan2(params['piEE'], params['piEN'])
            params['piE'] = jnp.sqrt(params['piEN']**2 + params['piEE']**2)
        else:
            raise ValueError(
                "Invalid parametrization. Choose from 'polar' (piE, psi) or 'cartesian' (piEE, piEN)."
        )
        delta_s_e, delta_s_n = self._compute_delta_sun_position_and_velocity(t, params['t0'])

        # Compute components of trajectory in Geocentric Equatorial coordinates
        u_e = params['u0']*jnp.cos(params['psi']) + (t - params['t0'])/params['tE']*jnp.sin(params['psi']) +\
            params['piE']*delta_s_e
        u_n = -params['u0']*jnp.sin(params['psi']) + (t - params['t0'])/params['tE']*jnp.cos(params['psi']) +\
            params['piE']*delta_s_n

        return u_e + 1j*u_n

