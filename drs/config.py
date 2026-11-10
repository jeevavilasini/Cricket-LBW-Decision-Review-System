"""
Physical constants and the geometry of the pitch / wicket.

All quantities are SI (metres, seconds, kilograms) and follow the coordinate
convention documented in :mod:`drs`:

    x along the pitch (x = 0 at the striker's stumps, +x towards the bowler)
    y across the pitch (y = 0 on the middle-stump line, +y to the off side)
    z vertical (z = 0 on the pitch surface)

Where the Laws of Cricket specify a range, the mid-range value is used and the
source is noted.  These are the values that a mathematically honest model must
state: they are *assumptions*, and every derived number inherits their error.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Fundamental physics
# --------------------------------------------------------------------------

#: Gravitational acceleration (m/s^2).
G: float = 9.81

#: Gravity as a vector in the package coordinate frame.
G_VEC: np.ndarray = np.array([0.0, 0.0, -G])

#: Air density at ~20 C and sea level (kg/m^3).
RHO_AIR: float = 1.225

# --------------------------------------------------------------------------
# The ball (Laws of Cricket, Law 4 / Appendix; typical first-class ball)
# --------------------------------------------------------------------------

#: Mass of the ball (kg).  Regulation range is 156-163 g.
BALL_MASS: float = 0.160

#: Radius of the ball (m).  Regulation diameter is 2.80-2.86 in.
BALL_RADIUS: float = 0.0363

#: Cross-sectional area of the ball (m^2), used by the quadratic drag model.
BALL_AREA: float = float(np.pi * BALL_RADIUS**2)

#: Drag coefficient.  ~0.47 for a smooth sphere at sub-critical Reynolds
#: number; a cricket ball with a raised seam sits a little higher, and the
#: exact value depends on seam orientation and surface roughness.
DRAG_COEFFICIENT: float = 0.50

# --------------------------------------------------------------------------
# The pitch (Law 6 / Law 7)
# --------------------------------------------------------------------------

#: Distance between the two wickets (m).  Law 7: 22 yards = 20.12 m.
PITCH_LENGTH: float = 20.12

#: Popping crease: 4 feet in front of the stumps (m).
POPPING_CREASE: float = 1.22

#: A "good length" delivery bounces roughly this far in front of the stumps (m).
TYPICAL_BOUNCE_DISTANCE: float = 6.0

# --------------------------------------------------------------------------
# The wicket (Law 8 / Law 9)
# --------------------------------------------------------------------------

#: Height of a stump above the pitch (m).  Law 8: 28 in = 0.7112 m.
STUMP_HEIGHT: float = 0.7112

#: Radius of a stump (m).  Law 8: diameter 1.38-1.50 in; mid-range used.
STUMP_RADIUS: float = 0.0190

#: Overall width of the wicket, outer edge to outer edge (m).  9 in = 0.2286 m.
WICKET_WIDTH: float = 0.2286

#: Half of the overall wicket width (m), i.e. the outer edge of the outer stumps.
WICKET_HALF_WIDTH: float = WICKET_WIDTH / 2.0

#: Lateral separation of the three stump centres (m).
STUMP_SPACING: float = (WICKET_WIDTH - 2.0 * STUMP_RADIUS) / 2.0

#: Centre-line (y) of the three stumps: off stump, middle, leg stump.
STUMP_OFFSETS: tuple[float, ...] = (+STUMP_SPACING, 0.0, -STUMP_SPACING)
STUMP_NAMES: tuple[str, ...] = ("off", "middle", "leg")

#: Height of a bail above the top of the stumps (m) and its radius (m).
BAIL_HEIGHT: float = 0.0127
BAIL_RADIUS: float = 0.0125

#: Vertical centre of a bail when resting in its groove (m).
BAIL_CENTER_Z: float = STUMP_HEIGHT + BAIL_RADIUS

# --------------------------------------------------------------------------
# DRS-style decision margins
# --------------------------------------------------------------------------

#: The on-field decision stands ("umpire's call") when the relevant geometric
#: quantity is within this distance of a zone boundary, in either direction.
#: A tracking system quoting "a few millimetres to a centimetre" of accuracy
#: justifies a band of about that size; 15 mm is a deliberately conservative
#: reading of a 1 cm accuracy claim.  Commercial ball tracking is more accurate
#: than this, and the value is exposed so that its influence can be studied
#: rather than hidden -- see :func:`drs.decision.review_lbw`.
DEFAULT_UMPIRE_CALL_MARGIN: float = 0.015  # 15 mm


def stump_centers() -> np.ndarray:
    """Return the (x, y) centres of the three stumps at the striker's end.

    Returns
    -------
    ndarray, shape (3, 2)
        Rows ordered off stump, middle stump, leg stump.
    """
    return np.array([[0.0, y] for y in STUMP_OFFSETS], dtype=float)


def pitch_point(x: float, y: float, z: float = 0.0) -> np.ndarray:
    """Convenience constructor for a point in the pitch frame."""
    return np.array([x, y, z], dtype=float)
