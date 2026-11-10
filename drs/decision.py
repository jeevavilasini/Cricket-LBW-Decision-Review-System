"""
The LBW decision layer.

Ball tracking alone does not decide an LBW appeal; it answers three geometric
questions that the Laws of Cricket then combine.  The ICC's DRS protocol names
them exactly:

``PITCHING``   where did the ball pitch, relative to the line of the stumps?
``IMPACT``     where did the ball strike the batter, relative to that line?
``WICKET``     would the ball have hit the stumps?

This module computes those three from a fitted trajectory and applies a
simplified reading of Law 36:

1.  If the ball pitched **outside the line of leg stump**, the batter cannot be
    out LBW -- no matter what happens afterwards.
2.  If the point of **impact** was outside the line of leg stump, not out.
3.  If the ball struck the batter **outside the line of off stump** and the
    batter was not offering a shot, not out.
4.  Otherwise, the batter is out if the ball **would have hit** the wicket.

On top of that sits **umpire's call**.  A prediction that lands within a
tolerance ``margin`` of a zone boundary is not treated as decisive: the
on-field decision stands.  That tolerance is the honest acknowledgement of
tracking error, and the project exposes it as a parameter precisely so its
influence can be measured.  Real DRS rules are more detailed than this model;
the zone predicates here are a documented approximation, not a replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from drs.config import BALL_RADIUS, DEFAULT_UMPIRE_CALL_MARGIN
from drs.estimate import FitResult, time_at_plane
from drs.geometry import Wicket

IN_LINE = "in_line"
OUTSIDE_OFF = "outside_off"
OUTSIDE_LEG = "outside_leg"

HITTING = "hitting"
MISSING = "missing"
UMPIRES_CALL = "umpire's_call"


@dataclass
class ZoneVerdict:
    """The outcome for one of pitching / impact / wicket."""

    name: str
    zone: str
    value: float
    marginal: bool = False
    point: np.ndarray | None = None

    @property
    def is_conclusive(self) -> bool:
        return not self.marginal

    def describe(self) -> str:
        label = {
            IN_LINE: "in line",
            OUTSIDE_OFF: "outside off",
            OUTSIDE_LEG: "outside leg",
            HITTING: "hitting",
            MISSING: "missing",
            UMPIRES_CALL: "umpire's call",
        }.get(self.zone, self.zone)
        tag = " (marginal)" if self.marginal else ""
        return f"{self.name}: {label}{tag}"


@dataclass
class LBWVerdict:
    """A complete (simplified) LBW review outcome."""

    pitching: ZoneVerdict
    impact: ZoneVerdict
    wicket: ZoneVerdict
    decision: str
    on_field: str
    umpires_call: bool
    reasons: list[str] = field(default_factory=list)
    wicket_point: np.ndarray | None = None
    clearance: float = float("nan")
    margin: float = DEFAULT_UMPIRE_CALL_MARGIN

    def describe(self) -> str:
        lines = [self.pitching.describe(), self.impact.describe(), self.wicket.describe()]
        lines.append(f"Decision: {self.decision.upper()}" + (" (umpire's call)" if self.umpires_call else ""))
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "pitching_zone": self.pitching.zone,
            "pitching_marginal": self.pitching.marginal,
            "impact_zone": self.impact.zone,
            "impact_marginal": self.impact.marginal,
            "wicket_zone": self.wicket.zone,
            "wicket_marginal": self.wicket.marginal,
            "clearance_m": self.clearance,
            "decision": self.decision,
            "on_field": self.on_field,
            "umpires_call": self.umpires_call,
        }


# --------------------------------------------------------------------------
# Derived geometric quantities
# --------------------------------------------------------------------------


def fitted_pitch_point(
    fit_result: FitResult,
    z_ground: float = BALL_RADIUS,
    t_max: float | None = None,
    x_max: float = 0.0,
) -> np.ndarray:
    """Where the fitted trajectory first meets the pitch.

    Inside the window that ends when the ball reaches the stumps, the fitted
    height is *unimodal*: it falls to the bounce and rises afterwards.  The
    bounce is therefore the **interior minimum** of ``z(t)``, and finding it that
    way is far more robust than scanning for a sign change -- a fixed sampling
    grid can easily straddle the bounce, where ``z`` touches the ground and
    turns, and report that the ball never pitched.

    A minimum sitting on the window boundary means the arc was still descending
    when it reached the stumps: the model is telling us it believes the ball
    never pitched, and this function returns ``nan`` rather than inventing a
    fictional pitch point.  That is the honest answer for a single parabola
    fitted across a bounce, and it is why the caller treats a missing pitch
    point as indeterminate.

    Accuracy note: the lateral coordinate is good to about a millimetre, which
    is what the pitching-*line* decision depends on.  The along-pitch coordinate
    inherits the quantisation of the break search -- the bounce is located to
    within one frame interval -- so it should not be read too precisely.  The
    fitted height at the reported point is within a few centimetres of the ball
    radius, because a piecewise fit is only piecewise continuous.
    """
    from scipy.optimize import brentq, minimize_scalar

    if t_max is None:
        try:
            t_max = time_at_plane(fit_result, x_plane=x_max)
        except ValueError:
            t_max = 1.0
    t_max = float(t_max)
    if t_max <= 0:
        return np.full(3, np.nan)

    def height(s: float) -> float:
        return float(fit_result.predict(np.array([s]))[0, 2])

    def point(s: float) -> np.ndarray:
        return fit_result.predict(np.array([s]))[0]

    # Stage 1: locate the turning point, if the fitted height has one. A
    # piecewise fit does (the bounce); a single ballistic arc does not.
    result = minimize_scalar(height, bounds=(0.0, t_max), method="bounded", options={"xatol": 1e-12})
    if not result.success:
        return np.full(3, np.nan)
    interior = float(result.x) < t_max - 1e-6

    # Stage 2: solve for ground contact on the *descending* branch. For a
    # bouncing trajectory that branch ends just before the turning point, where
    # the fitted model has a kink (and, for a piecewise fit, a small jump) -- so
    # the root is only well defined on this side. For a monotone arc the
    # descending branch is the whole window.
    if interior:
        hi = float(result.x) - 1e-7
        if hi <= 0.0:
            return np.full(3, np.nan)
    else:
        hi = t_max

    def g(s: float) -> float:
        return height(s) - z_ground

    if g(0.0) <= 0.0:
        candidate = point(0.0)  # already at ground level on the first sample
    elif g(hi) > 0.0:
        # The arc turns away before touching the ground. Accept it only if it
        # comes close enough to count as having pitched.
        tolerance = max(0.02, 3.0 * fit_result.residual_rmse)
        if not (interior and result.fun <= z_ground + tolerance):
            return np.full(3, np.nan)
        candidate = point(float(result.x))
    else:
        candidate = point(brentq(g, 0.0, hi, xtol=1e-12))

    # A contact that would happen level with or behind the striker is not a
    # pitching point: the ball reached the batter without pitching.
    return candidate if candidate[0] <= x_max else np.full(3, np.nan)


def classify_lateral(y: float, wicket: Wicket, margin: float) -> tuple[str, bool]:
    """Classify a lateral coordinate against the line of the stumps.

    Returns the zone and whether it is within ``margin`` of a boundary (i.e.
    an umpire's-call candidate).
    """
    d = abs(y) - wicket.half_width
    if y < 0:
        return (OUTSIDE_LEG if d > 0 else IN_LINE), abs(d) <= margin
    return (OUTSIDE_OFF if d > 0 else IN_LINE), abs(d) <= margin


# --------------------------------------------------------------------------
# The review
# --------------------------------------------------------------------------


def review_lbw(
    fit_result: FitResult,
    wicket: Wicket | None = None,
    on_field: str = "not_out",
    impact_x: float = -0.25,
    margin: float = DEFAULT_UMPIRE_CALL_MARGIN,
    offered_shot: bool = True,
) -> LBWVerdict:
    """Run the simplified LBW review on a fitted trajectory.

    Parameters
    ----------
    fit_result:
        Any estimator's output.  The trajectory must be valid in the plane of
        the stumps, which is why single-arc fits over a bounced delivery give
        such poor answers here.
    wicket:
        Wicket geometry; defaults to the standard one.
    on_field:
        The original umpire's decision, ``"out"`` or ``"not_out"``.  It is
        returned unchanged whenever the technology is not decisive.
    impact_x:
        The x plane at which the ball is taken to have struck the batter.  On a
        real appeal this comes from the pad position; ``-0.25 m`` (a little in
        front of the stumps) is a reasonable stand-in.
    margin:
        The umpire's-call tolerance in metres.
    offered_shot:
        Whether the batter was playing a shot.  Only relevant when the ball
        strikes outside the line of off stump.
    """
    wicket = wicket or Wicket()
    reasons: list[str] = []

    # -- WICKET: the predicted arrival in the plane of the stumps ----------
    t_w = time_at_plane(fit_result, x_plane=wicket.x)
    wicket_point = fit_result.predict(np.array([t_w]))[0]
    clearance = float(wicket.clearance(wicket_point)[0])
    # The umpire's-call band is symmetric about the contact threshold: a ball
    # predicted to clip the stumps and a ball predicted to miss them narrowly
    # are both inside the tracking error, so neither is decisive.
    if clearance > margin:
        wicket_zone, wicket_marginal = MISSING, False
    elif clearance < -margin:
        wicket_zone, wicket_marginal = HITTING, False
    else:
        wicket_zone, wicket_marginal = (HITTING if clearance <= 0.0 else MISSING), True
    wicket_verdict = ZoneVerdict(
        name="wicket", zone=wicket_zone, value=clearance, marginal=wicket_marginal, point=wicket_point
    )

    # -- PITCHING ----------------------------------------------------------
    pitch_point = fitted_pitch_point(fit_result)
    if np.any(np.isnan(pitch_point)):
        pitching_zone, pitching_marginal = IN_LINE, True
        reasons.append("the ball never pitched; the pitching zone is indeterminate")
    else:
        pitching_zone, pitching_marginal = classify_lateral(float(pitch_point[1]), wicket, margin)
    pitching_verdict = ZoneVerdict(
        name="pitching",
        zone=pitching_zone,
        value=float(pitch_point[1]) if not np.any(np.isnan(pitch_point)) else float("nan"),
        marginal=pitching_marginal,
        point=pitch_point,
    )

    # -- IMPACT ------------------------------------------------------------
    if impact_x >= wicket.x:
        raise ValueError("impact_x must be in front of the wicket plane (more negative than wicket.x)")
    t_i = time_at_plane(fit_result, x_plane=impact_x)
    impact_point = fit_result.predict(np.array([t_i]))[0]
    impact_zone, impact_marginal = classify_lateral(float(impact_point[1]), wicket, margin)
    impact_verdict = ZoneVerdict(
        name="impact", zone=impact_zone, value=float(impact_point[1]), marginal=impact_marginal, point=impact_point
    )

    # -- combine -----------------------------------------------------------
    decisive_not_out = False
    if pitching_zone == OUTSIDE_LEG:
        decisive_not_out = True
        reasons.append("the ball pitched outside the line of leg stump")
    if impact_zone == OUTSIDE_LEG:
        decisive_not_out = True
        reasons.append("the ball struck the batter outside the line of leg stump")
    if impact_zone == OUTSIDE_OFF and not offered_shot:
        decisive_not_out = True
        reasons.append("the ball struck outside off stump with no shot offered")

    any_marginal = pitching_marginal or impact_marginal or wicket_marginal

    if decisive_not_out:
        decision = "not_out"
        umpires_call = False
    elif wicket_zone == MISSING:
        decision = "not_out"
        umpires_call = False
        reasons.append("the ball would have missed the wicket")
    elif any_marginal:
        # The technology is not decisive: the on-field call stands.
        decision = "out" if on_field == "out" else "not_out"
        umpires_call = True
        reasons.append("at least one zone is marginal, so the on-field decision stands")
    else:
        decision = "out"
        umpires_call = False
        reasons.append("the ball pitched in line, struck in line and would have hit the wicket")

    return LBWVerdict(
        pitching=pitching_verdict,
        impact=impact_verdict,
        wicket=wicket_verdict,
        decision=decision,
        on_field=on_field,
        umpires_call=umpires_call,
        reasons=reasons,
        wicket_point=wicket_point,
        clearance=clearance,
        margin=margin,
    )
