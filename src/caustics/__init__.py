# -*- coding: utf-8 -*-
__all__ = [
    "__version__",
    "lens_eq",
    "mag_point_source",
    "critical_and_caustic_curves",
    "mag_extended_source",
    "mag",
]

from .point_source_magnification import (
    lens_eq,
    mag_point_source,
    critical_and_caustic_curves,
)

from .extended_source_magnification import (
    mag_extended_source,
)

from .lightcurve import (
    mag,
)
