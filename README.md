# Mathematical modelling of cricket's Decision Review System

A computational reconstruction of a cricket ball's trajectory from noisy
observations, and a geometric model of whether that trajectory would have hit
the wicket.

This is a **research prototype**, not an officiating system. It reproduces the
*mathematical structure* of a DRS ball-tracking review — pitching, impact,
wicket, and a margin for marginal calls — while making every assumption
explicit enough to be measured.

---

## The problem

For an LBW appeal the umpire needs to know whether the ball would have hit the
stumps, given where it pitched and where it struck the batter. Reconstructing
that from measurements is an **inverse problem**:

    p_i = r(t_i) + eps_i          observations are noisy, discrete and finite
                                       |
                                       v
    r(t) = ?                      the trajectory is continuous and unknown
                                       |
                                       v
    does r(t) meet the wicket?    the decision is geometric and 3-D

The video gives a handful of samples. The physics gives a continuous path. The
gap between them is where the mathematics lives.

## Quick start

```bash
python3 -m venv --system-site-packages drs-env
source drs-env/bin/activate
pip install -r requirements.txt

python scripts/run_demo.py                       # one delivery, end to end
python scripts/run_demo.py --video               # ... through rendered frames
python scripts/run_video.py --synthetic good_length_middle
python scripts/run_experiments.py                # the full study, ~2 minutes
python -m pytest tests/ -q                       # 138 tests
streamlit run app/streamlit_app.py               # interactive demo
```

Figures land in `outputs/figures/`, tables and CSV results in
`outputs/results/`.

## How it is put together

The package follows the four mathematical problems of the project, one module
each, with supporting layers around them.

| Module | Role |
|---|---|
| `drs.config` | Physical constants and wicket geometry, each with its source |
| `drs.physics` | Forward model: ballistic arcs, quadratic drag, the bounce map |
| `drs.simulate` | The observation model: ground truth, sampling, noise, dropouts |
| `drs.estimate` | **Inverse problem**: four estimators plus covariance |
| `drs.kalman` | Recursive ballistic Kalman filter and RTS smoother |
| `drs.geometry` | Pinhole camera, DLT triangulation, wicket solids, ICC zones |
| `drs.decision` | The simplified Law 36 review and the umpire's-call band |
| `drs.uncertainty` | Monte Carlo and bootstrap propagation of uncertainty |
| `drs.detection` | HSV colour thresholding and sub-pixel blob centroids |
| `drs.tracking` | Temporal association, physical gating, 3-D reconstruction |
| `drs.synthvideo` | Renders synthetic camera frames with motion blur |
| `drs.viz` | Figures that answer the questions the report asks |
| `drs.experiments` | The six studies, as reproducible CSV-producing functions |

### Coordinate system

Everything uses one right-handed frame in metres:

- **x** along the pitch, `x = 0` in the plane of the striker's stumps, positive
  towards the bowler. A delivery starts near `x = -18.5` and travels to `x = 0`.
- **y** across the pitch, `y = 0` on the middle-stump line, positive to the off
  side of a right-handed batter.
- **z** vertical, `z = 0` on the pitch surface.

## The mathematics

### 1. Ball motion

With gravity the only force, Newton's second law integrates in closed form:

    r(t) = r0 + v0 t + 1/2 g t^2 ,     g = (0, 0, -9.81)

Adding quadratic drag,

    m dv/dt = m g - 1/2 rho C_D A ||v|| v ,

destroys the closed form, so `drs.physics` integrates it with
`scipy.integrate.solve_ivp`. Both models are available so that the cost of the
drag-free assumption can be *measured* rather than assumed: on a 140 km/h
delivery, drag shortens the length by about **18 cm** and moves the arrival
height by about **5 mm**.

### 2. The bounce

The bounce is a genuine discontinuity, not a smooth feature. The velocity is
mapped through

    v_z^+ = -e v_z^- ,    v_x^+ = k_x v_x^- ,    v_y^+ = k_y v_y^- ,

so the trajectory is a **piecewise** dynamical system. A single parabola fitted
across the bounce is structurally wrong, and the experiments show the
consequence directly: on noiseless data a single arc has ~100 mm of residual
while the piecewise model has ~0 mm. **Model error dominates measurement
error.**

### 3. The inverse problem

Four estimators, deliberately comparable:

| Model | Parameters | Notes |
|---|---|---|
| `polynomial` | 9 | Free quadratic per axis; `g_hat = -2 c_z` is *estimated* |
| `ballistic` | 6 | Gravity fixed; linear least squares with exact covariance |
| `post_bounce` | 6 | One arc on the post-bounce frames only |
| `piecewise` | 12 + break | Two arcs joined at a **searched** bounce index |
| `ballistic_drag` | 7 | Nonlinear; drag coefficient as a third unknown |
| `kalman` | 6 | Recursive; handles missing frames and grows its own uncertainty |

The ballistic model is linear *because* gravity is imposed: subtracting the
known `1/2 g t^2` term leaves `theta = (r0, v0)` in a 6-parameter linear
least-squares problem, whose parameter covariance is available in closed form.
That covariance is what the uncertainty layer propagates.

The bounce index is recovered by a one-dimensional discrete search over every
admissible split, keeping the one with the smallest total residual.

### 4. From pixels to metres

A single image point defines a **ray**, not a point:

    lambda [u, v, 1]^T = K [R | t] [x, y, z, 1]^T

so a monocular system cannot recover depth without an extra assumption. With two
calibrated views, DLT triangulation closes the loop. The pipeline demonstrates
both, and the experiment shows the anisotropy that results: with a 4 m stereo
baseline at 25 m, one pixel of centroid error costs roughly **65 mm in depth**
but only **3 mm laterally** and **11 mm vertically**.

### 5. The decision

Three zone questions, exactly as the ICC protocol names them:

- **PITCHING** — outside the line of leg stump makes it not out, full stop.
- **IMPACT** — outside leg is not out; outside off with no shot offered is not out.
- **WICKET** — the ball's *sphere* is tested against the wicket solid by exact
  Euclidean distance, not by comparing coordinates.

On top of that sits **umpire's call**: a prediction within a tolerance `margin`
of a zone boundary, in either direction, is not decisive and the on-field
decision stands. The tolerance is a parameter (`DEFAULT_UMPIRE_CALL_MARGIN`,
default 15 mm) precisely so its influence can be studied.

### 6. Uncertainty

Two propagation strategies, and the difference between them is the point:

- **Parameter-covariance Monte Carlo**: draw `theta_j ~ N(theta_hat, Sigma_hat)`,
  simulate, and count the fraction that hit. Cheap, and exact for the linear
  estimators — but it only knows about measurement error.
- **Bootstrap**: resample the data and re-run the whole estimator. Expensive,
  but it also sees **model error**.

Neither number is the official DRS probability. Both are honest statements of
uncertainty in this model.

## Headline results

Measured on a Monte Carlo ensemble of realistic deliveries; full tables in
`outputs/results/SUMMARY.md`.

**Model error dominates measurement error.** On noiseless observations:

| Model | residual RMSE | error at the wicket plane |
|---|---|---|
| `piecewise` | 0.0 mm | 0.0 mm |
| `post_bounce` | 0.0 mm | 0.0 mm |
| `kalman` | 0.0 mm | 0.0 mm |
| `polynomial` | 96 mm | 202 mm |
| `ballistic` (one arc across the bounce) | 221 mm | 939 mm |

**Geometry beats curve fitting.** With 5 mm measurement noise the piecewise
model predicts the wicket-plane position to a median of ~3 mm, while the
single-arc model is wrong by ~0.9 m.

**Depth is the hard coordinate.** Stereo triangulation with 0.5 px centroid
noise gives 7.3 mm of depth error against 0.2 mm laterally.

**The vision pipeline works.** Rendered synthetic footage, colour detection,
two-view triangulation and a piecewise fit recover the wicket-plane position to
**0.4 mm** — and 80/80 frames were detected at 0.35 px centroid accuracy.

**A razor-edge delivery is genuinely uncertain.** For a ball whose true
clearance is exactly zero, the model's hit probability averages ~0.5 across
noise realisations and ranges from 0.02 to 0.99 — which is the honest answer,
and it is what justifies an umpire's-call rule existing at all.

**Ball tracking alone is not the decision.** `leg_break_into_stumps` would
demolish middle stump, and the model says so with 30 mm of clearance — but it
pitched outside leg stump, so the batter is not out.

## What this is not

- Not an official system, and not Hawk-Eye. No real broadcast footage was used;
  all results are on synthetic data with exact ground truth.
- The Laws of Cricket are implemented in a **simplified** form. The three zones
  and the umpire's-call band are modelled; the full Law 36 conditions are not.
- Shot-offered and impact-position handling use a fixed impact plane
  (`x = -0.25 m` by default) rather than tracking the batter's pads.
- The detector is a colour threshold. Broadcast footage — motion blur, camera
  movement, crowds, occlusion — would need a trained detector.

## Reproducing the results

Every experiment is seeded, so the tables are reproducible:

```bash
python scripts/run_experiments.py --quick     # fast smoke test
python scripts/run_experiments.py             # the full study
python scripts/run_experiments.py --only models camera
```

## Licence and attribution

Written as a self-contained mathematical-modelling project. The physical
constants follow the Laws of Cricket; the decision zones are a documented
approximation of the ICC DRS protocol, not a reproduction of it.
