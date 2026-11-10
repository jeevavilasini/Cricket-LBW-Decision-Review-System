"""
drs - Mathematical modelling and computational reconstruction of cricket's
Decision Review System (DRS) from ball-trajectory estimation.

The package is organised by the four mathematical problems of the project:

    detection   ->  where is the ball in each frame?
    tracking    ->  assemble reliable observations over time
    estimate    ->  recover the continuous trajectory (the inverse problem)
    geometry    ->  does the predicted path intersect the wicket?

plus the supporting physics, uncertainty-quantification and decision layers.

Coordinate convention (metres, right-handed), used everywhere in the package:

    x : along the pitch.  x = 0 at the plane of the striker's stumps,
        positive towards the bowler's end.  A delivery therefore starts at
        x ~ -18.5 and travels towards x = 0.
    y : across the pitch.  y = 0 on the middle-stump line, positive towards
        the off side of a right-handed batter, negative towards the leg side.
    z : vertical.  z = 0 on the pitch surface, positive upwards.
"""

from drs.config import (  # noqa: F401
    BALL_AREA,
    BALL_MASS,
    BALL_RADIUS,
    G,
    G_VEC,
    PITCH_LENGTH,
    STUMP_HEIGHT,
    STUMP_RADIUS,
    WICKET_HALF_WIDTH,
)

__version__ = "0.1.0"
