import numpy as np
import os

from astropy.coordinates import SkyCoord
from astropy import units as u

from caustics.trajectory import AnnualParallaxTrajectory

import MulensModel as mm

def test_annual_parallax(t0=3619.):
    file_name = os.path.join(mm.DATA_PATH,
        "photometry_files", "OB05086", "starBLG234.6.I.218982.dat")
    dat = mm.MulensData(file_name=file_name, add_2450000=True)

    t = dat.time - 2450000.

    RA = "18:04:45.70"
    Dec = "-26:59:15.5"
    coords = SkyCoord(
        RA, Dec, unit=(u.hourangle, u.deg, u.arcminute)
    )
    trajectory = AnnualParallaxTrajectory(t, coords)
    delta_s_e, delta_s_n = trajectory._compute_delta_sun_position_and_velocity(t, t0)

    # Compare to MulensModel
    params = dict()
    params['t_0'] = t0 + 2450000.
    params['u_0'] = 0.37
    params['t_E'] = 100.
    params['pi_E_N'] = 0.
    params['pi_E_E'] = 0.
    my_model = mm.Model(params, coords=coords)
    trajectory_mm = mm.Trajectory(
            dat.time, parameters=my_model.parameters,
            parallax={'earth_orbital': True}, coords=coords)

    delta_s_e_mm = trajectory_mm.parallax_delta_N_E['E']
    delta_s_n_mm = trajectory_mm.parallax_delta_N_E['N']

    np.testing.assert_allclose(delta_s_e, delta_s_e_mm, atol=1e-3)
    np.testing.assert_allclose(delta_s_n, delta_s_n_mm, atol=1e-3)
