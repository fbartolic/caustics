# -*- coding: utf-8 -*-
__all__ = [
    "__version__",
    "lens_eq_binary",
    "lens_eq_triple",
    "images_point_source_binary",
    "images_point_source_triple",
    "mag_point_source_binary",
    "mag_point_source_triple",
    "critical_and_caustic_curves_binary",
    "critical_and_caustic_curves_triple",
    "mag_extended_source_binary",
    "mag_extended_source_triple",
    "mag_binary",
    "mag_triple",
]

from .point_source_magnification import (
    lens_eq_binary,
    lens_eq_triple,
    images_point_source_binary,
    images_point_source_triple,
    mag_point_source_binary,
    mag_point_source_triple,
    critical_and_caustic_curves_binary,
    critical_and_caustic_curves_triple,
)

from .extended_source_magnification import (
    mag_extended_source_binary,
    mag_extended_source_triple,
)

from .lightcurve import (
    mag_binary,
    mag_triple,
)
