# Mathematical modelling and computational reconstruction of cricket's Decision Review System

**A ball-trajectory estimation prototype, with measured error and quantified uncertainty**

---

## 1. The research question

An LBW (leg before wicket) appeal turns on a counterfactual: *where would the ball
have gone?* Nobody observes that. What exists is a finite set of noisy
measurements of where the ball *did* go, and a set of geometric rules about
where a ball must go for the batter to be out.

The project asks:

> How do physical constraints, trajectory-estimation methods, and measurement
> uncertainty influence the accuracy of a computational model for predicting
> whether a cricket ball intersects the wicket?

That question decomposes into four mathematical problems, which are also the
four modules of the software:

1. **Detection** — where is the ball in each frame?
2. **Trajectory reconstruction** — what continuous path does it follow between
   the observed positions?
3. **Physical modelling** — how do gravity and the bounce shape that path?
4. **Decision geometry** — does the predicted path intersect the wicket?

The fifth, which turns an exercise into a research project, is **uncertainty
quantification**: the answer is a distribution, not a yes.

Everything below is implemented in the accompanying package and reproducible
from `scripts/run_experiments.py`. All results are on synthetic data with exact
ground truth, so every error figure is *measured* rather than asserted.

---

## 2. Coordinate system and constants

One right-handed frame, in metres, is used everywhere:

| Axis | Meaning |
|---|---|
| `x` | along the pitch; `x = 0` in the plane of the striker's stumps, `+x` towards the bowler |
| `y` | across the pitch; `y = 0` on the middle-stump line, `+y` to the off side of a right-handed batter |
| `z` | vertical; `z = 0` on the pitch surface |

A delivery starts at roughly `x = -18.5` and travels towards `x = 0`.

Physical constants follow the Laws of Cricket where the Law specifies a range;
the mid-range value is used.

| Quantity | Value | Source |
|---|---|---|
| Gravitational acceleration `g` | 9.81 m/s² | — |
| Ball mass | 0.160 kg | Law 4 (156–163 g) |
| Ball radius | 36.3 mm | Law 4 (2.80–2.86 in diameter) |
| Ball cross-section `A` | 4.14 × 10⁻³ m² | derived |
| Air density `rho` | 1.225 kg/m³ | sea level, 20 °C |
| Drag coefficient `C_D` | 0.50 | smooth sphere ≈ 0.47; raised seam raises it |
| Pitch length | 20.12 m | Law 7 (22 yards) |
| Stump height | 0.7112 m | Law 8 (28 in) |
| Stump radius | 19.0 mm | Law 8 (1.38–1.50 in diameter) |
| Wicket width | 0.2286 m | Law 8 (9 in) |

Each is an **assumption**, and the error in every derived quantity inherits
from them.

---

## 3. Forward model: how the ball moves

### 3.1 Ballistic motion

Newton's second law with gravity as the only force,

```
m d²r/dt² = m g ,        g = (0, 0, -9.81)
```

integrates in closed form:

```
x(t) = x0 + v_x t
y(t) = y0 + v_y t
z(t) = z0 + v_z t - 1/2 g t²
```

The video supplies finitely many samples; these equations supply a continuum.
That is the entire reason the inverse problem is well posed at all.

### 3.2 Quadratic drag

Adding air resistance,

```
m dv/dt = m g - 1/2 rho C_D A ||v - w|| (v - w)
```

destroys the closed form, so the model is integrated numerically
(`scipy.integrate.solve_ivp`, RK45, `rtol = 1e-9`).

The drag-free model is retained as the default precisely so that its cost can
be measured: on the 140 km/h reference delivery, drag

- shortens the delivery by **18.1 cm** (the ball pitches at `x = -5.868 m`
  instead of `x = -5.687 m`), and
- changes the arrival height by **5 mm**.

Five millimetres is small compared with the ball's radius but is the same order
as the umpire's-call margin, which is exactly why it is worth quantifying rather
than dismissing.

### 3.3 The bounce

The bounce is **not** a smooth feature of the trajectory. It is an impulsive
event modelled by a velocity map:

```
v_z⁺ = -e v_z⁻ ,     v_x⁺ = k_x v_x⁻ ,     v_y⁺ = k_y v_y⁻ ,     (0 ≤ e ≤ 1)
```

so the trajectory is a **piecewise** dynamical system with an unknown switch
time `t_b`:

```
r(t) = r_pre(t)   for t < t_b
r(t) = r_post(t)  for t ≥ t_b
```

The parameters `e`, `k_x`, `k_y` are not universal constants; they depend on the
ball, the surface and the delivery. An optional `spin_kick` term adds a lateral
velocity impulse to represent turn off the pitch.

The bounce time is found analytically by solving `z(t) = r_ball` for the
smallest positive root, and the simulation loops through bounces until the ball
reaches the wicket plane.

---

## 4. Observation model

The estimators never see the true trajectory. They see

```
p_i = r(t_i) + eps_i ,      eps_i ~ N(0, sigma² I)
```

sampled at the camera frame rate, with a configurable fraction of frames
dropped — isolated (detector failure) or clustered (occlusion). Separating the
generative model from the inverse model is what makes the error analysis
meaningful: **ground truth is known exactly**.

Two observation paths are implemented:

- an **ideal** path that samples the true 3-D trajectory directly — the right
  way to study the estimators;
- a **vision** path that renders synthetic camera frames, detects the ball by
  colour, and triangulates two views — the right way to study the pipeline.

---

## 5. The inverse problem

Given `{ (t_i, p_i) }`, recover a continuous `r(t)` and, from it, the ball's
state in the plane of the stumps. Four estimators are implemented so they can be
*compared*.

### 5.1 Unconstrained polynomial

Per-axis least squares on `a + b t + c t²`. Nine free parameters, no physics.
The fitted `c_z` yields a free estimate of gravity, `g_hat = -2 c_z`, which
recovers 9.81 exactly on a noiseless single arc — and drifts badly once the
noise rises, which is the pedagogical point.

### 5.2 Gravity-constrained ballistic arc

With gravity *imposed*, write

```
r(t) = r0 + v0 (t - t_ref) + 1/2 g (t - t_ref)²
```

Subtracting the known `1/2 g (t - t_ref)²` term leaves a model that is **linear
in the six unknowns** `theta = (r0, v0)`. These are therefore a single
least-squares solve, and — because the model is linear — the parameter
covariance is available in closed form:

```
Sigma_theta = sigma² (Aᵀ A)⁻¹ ⊗ I₃ ,      sigma² = ||Y - A Theta||² / (3n - 6)
```

where the Kronecker structure follows from treating the three coordinates as
independent observations of the same design. This covariance is exactly what
the Monte-Carlo layer propagates, so the linearity of this model is not a
convenience — it is what makes the uncertainty analysis rigorous.

Valid on **one arc only**.

### 5.3 Piecewise ballistic: the bounce as an unknown

Since a real delivery bounces, the correct model is two arcs joined at an
unknown index:

```
r(t) = r_pre(t; theta_pre)     t < t_b
r(t) = r_post(t; theta_post)   t ≥ t_b
```

The switch index is recovered by a one-dimensional discrete search: fit both
arcs for *every* admissible split (each arc needs at least three points) and
keep the split minimising the total residual. The covariance is the
block-diagonal assembly of the two arcs' covariances.

This converts a smooth-fitting problem into a small **inverse problem in its own
right**, and it is the single most important estimator in the project.

### 5.4 Post-bounce single arc

Once the break index is known, a single ballistic arc on the post-bounce frames
alone is a fair, cheap and accurate alternative — because the arc that
determines the wicket-plane prediction is the post-bounce one. Separating this
from `piecewise` isolates how much the pre-bounce arc actually contributes.

### 5.5 Ballistic with drag

`(r0, v0, C_D)` fitted by nonlinear least squares with the ODE integrated
inside the residual. The Jacobian is numerical, so this is materially more
expensive, and it is included mainly to *quantify* the drag-free approximation
error rather than as a production model.

### 5.6 Ballistic Kalman filter

State `s = [p, v]` with the exact discrete transition for constant gravity:

```
F = [[I, dt I], [0, I]] ,    s_{k+1} = F s_k + [dt²/2 I ; dt I] g + w
```

Measurement `z = H s`, `H = [I, 0]`, `R = sigma_meas² I`. Everything gravity
does not explain — drag, swing, the bounce, spin — enters through the process
noise `w`, whose intensity `q` is exposed as a tuning parameter. Missing frames
propagate without an update, which is the quantitative statement of "a tracker
becomes less sure when it cannot see the ball".

An optional Rauch–Tung–Striebel smoother provides the offline refinement, and
is a useful upper bound on what a causal filter can achieve.

---

## 6. Geometry

### 6.1 The camera and the depth ambiguity

The pinhole projection

```
lambda [u, v, 1]ᵀ = K [R | t] [x, y, z, 1]ᵀ
```

is irreversible from a single pixel: it defines a **ray**. Points at 5 m and at
40 m along that ray project to the same pixel. This is confirmed by a unit test.

Depth therefore requires either a second view or a physical assumption.

### 6.2 Triangulation

With two calibrated views, DLT triangulation recovers the 3-D point exactly from
noiseless pixels (verified to 10⁻¹³ m). The error under noise is strongly
**anisotropic**: with a 4 m baseline at 25 m, one pixel of centroid error costs

| coordinate | error per pixel |
|---|---|
| depth (`x`) | ~43 mm |
| lateral (`y`) | ~2 mm |
| vertical (`z`) | ~7 mm |

This anisotropy is not an implementation detail; it is why the umpire's-call
band exists at all, and why it is the lateral zone questions (in line / outside
off / outside leg) that dominate LBW rather than the height.

A least-squares ray-intersection ("midpoint") triangulation is also provided for
comparison.

### 6.3 Wicket geometry

The wicket is modelled as three finite **cylinders** plus two **bail boxes**.
The hit test uses the exact Euclidean distance from the ball centre to the solid:

```
d_radial = hypot(x - c_x, y - c_y) - R_stump
d_axial  = max(0, -z, z - H_stump)
dist     = d_axial                 if d_radial ≤ 0
         = d_radial                if d_axial = 0
         = hypot(d_radial, d_axial) otherwise
```

and the ball (a sphere of radius `r_b`) touches when `dist ≤ r_b`. This is
strictly better than checking whether the centre lies between two coordinates:
a ball whose centre is *outside* the outer stump line can still be hitting, and
the model gets that right. The signed **clearance** `dist - r_b` is the natural
decision variable: negative means the ball hits.

### 6.4 The ICC zones

The protocol names three questions, and the model answers each explicitly:

- **Pitching** — is the bounce point inside the line of the stumps? Outside leg
  is decisive.
- **Impact** — is the contact point inside that line? Outside leg is decisive;
  outside off with no shot offered is also decisive.
- **Wicket** — does the predicted sphere intersect the wicket solid?

---

## 7. The decision

The review applies a simplified reading of Law 36:

1. If the ball pitched outside the line of leg stump → **not out**, full stop.
2. If the point of impact was outside the line of leg stump → **not out**.
3. If the ball struck outside off stump with no shot offered → **not out**.
4. Otherwise the batter is **out** if the ball would have hit the wicket.

On top of that sits **umpire's call**. A prediction within `margin` of a zone
boundary — *in either direction* — is not decisive, and the on-field decision
stands. Since the ball is a sphere, the wicket criterion is inherently fuzzy at
its edges, so the band is symmetric about the contact threshold: a prediction to
clip the stumps and a prediction to miss them narrowly are both inside the
tracking error.

The default margin is 15 mm (`DEFAULT_UMPIRE_CALL_MARGIN`), a deliberately
conservative reading of a "1 cm accuracy" claim. It is a **parameter**, not a
constant, precisely so its influence can be measured. The real ICC protocol is
more detailed than this model, and the zones here are a documented
approximation of it rather than a reproduction.

---

## 8. Uncertainty quantification

A single trajectory and a single yes/no answer is not a defensible output. The
prediction is (i) built from noisy measurements, (ii) based on an imperfect
functional form, and (iii) an **extrapolation** several metres beyond the last
observed frame.

Two propagation strategies are implemented, and their difference is the point:

**Parameter-covariance Monte Carlo.** Draw `theta_j ~ N(theta_hat, Sigma_hat)`
from the estimator's own covariance, simulate each draw, and estimate

```
P_hat(hit) = N_hits / N .
```

Cheap — and, for the *linear* estimators, the covariance is exact. But it only
knows about measurement error.

**Bootstrap Monte Carlo.** Resample the observations (fresh Gaussian noise or
resampled residuals) and re-run the whole estimator. Expensive, but it also sees
**model error**: if the functional form cannot represent the truth, the refits
disagree more than the covariance suggests.

Implementation note: because the geometric models are linear in their
parameters, the ensemble is propagated by a fully **batched bisection** — one
NumPy operation per iteration for all `N` trajectories — which makes 50 000
samples a fraction of a second.

---

## 9. Experiments and results

All experiments use a Monte Carlo ensemble of physically plausible deliveries,
drawn by sampling *cricketing* quantities (a release speed, a length, a line)
and solving for the release velocity that delivers them — so the ensemble looks
like an over, not a cloud of arbitrary parabolas.

### 9.1 Measurement noise

Median error at the wicket plane (mm), by per-coordinate measurement noise:

| noise (mm) | 0 | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| `piecewise` | **0.0** | 1.7 | 4.1 | 8.3 | 16.3 | 33.5 |
| `post_bounce` | **0.0** | 1.6 | 4.0 | 8.4 | 16.8 | 33.6 |
| `polynomial` | 133.8 | 133.6 | 133.2 | 132.8 | 135.3 | 139.3 |
| `ballistic` (one arc) | 901.8 | 901.5 | 901.0 | 900.1 | 898.4 | 896.9 |

Two things are visible at once.

For the **correctly specified** models the error rises perfectly linearly with
the noise: 5 mm of measurement noise produces about 4 mm of prediction error.
There is no amplification — the extrapolation is well conditioned.

For the **misspecified** models the error is almost independent of the noise.
This is the central result of the project: **model error dominates measurement
error.** Reducing the noise from 40 mm to zero would not help the single-arc
model at all; fixing its structure takes it from 900 mm to zero.

### 9.2 The frame budget — and why the count is the wrong question

Median error at the wicket plane (mm), `piecewise` model, 5 mm noise:

| frames retained | 6 | 10 | 15 | 20 | 25 | 30 |
|---|---|---|---|---|---|---|
| **first k** (lost the ball late) | 2013 | 1988 | 1904 | 705 | 6.1 | 4.0 |
| **last k** (acquired it late) | **7.2** | 5.1 | 4.1 | 4.0 | 4.0 | 4.0 |

The answer to "how many frames are needed?" is: *it is not about the number.*

Keeping the first 20 frames — 20 observations, plenty by any naive standard —
gives **0.7 metres** of error, and the first 15 give **1.9 metres**, because
those frames stop before the bounce. Keeping the last **6** frames gives
**7.2 mm**, because they are all post-bounce.

The threshold in the "first k" column is exactly where the retained frames start
to include the bounce (frame ≈ 17 at 50 Hz for a 0.34 s bounce). Before that the
estimator is being asked to extrapolate a pre-bounce parabola across a
discontinuity, and it fails by two metres.

A tracker that misses the bounce is helpless no matter how much data it has.
That is a design requirement, not a numerical accident.

### 9.3 Missing detections

Median error at the wicket plane (mm), `piecewise`, 5 mm noise:

| dropout | 0% | 10% | 25% | 40% |
|---|---|---|---|---|
| error | 4.0 | 4.0 | 5.0 | 5.5 |

Remarkably robust: losing 40% of frames costs 1.5 mm. Because the model is
structurally correct and the geometry is well conditioned, the least-squares fit
simply interpolates across the gaps. (Clustered gaps, which model an occlusion,
are the harder case and are also implemented.)

### 9.4 Estimator comparison

Median error at the wicket plane (mm):

| model | noise 0 mm | noise 5 mm | held-out RMSE | note |
|---|---|---|---|---|
| `piecewise` | **0.0** | **7.4** | **2.9 mm** | two arcs, searched break |
| `post_bounce` | **0.0** | **7.4** | 1256 mm | one arc, post-bounce only |
| `kalman` | **0.0** | 10.7 | 1254 mm | recursive, handles dropouts |
| `polynomial` | 133.8 | 129.9 | 99.9 mm | free gravity, one arc |
| `ballistic_drag` | 952.0 | 950.6 | 227.3 mm | drag, but pre-bounce only |
| `ballistic` | 901.8 | 900.7 | 256.0 mm | fixed gravity, one arc |

Three observations.

The three correctly-specified models recover the truth **exactly** from
noiseless data. That is the expected result and it validates both the forward
model and the estimators.

The held-out column is measured on *interleaved* frames — every other
observation withheld and predicted from its neighbours — so it is a genuine
generalisation test, and it separates the models far more sharply than the
in-sample residual does. `piecewise` reconstructs unseen observations to
**2.9 mm**, roughly two orders of magnitude better than the best single-arc
model (227 mm). It also exposes two failure modes that the in-sample numbers
hide completely: `post_bounce` and `kalman` are exact at the wicket plane yet
score above a metre on held-out frames, because a model fitted to the
post-bounce arc cannot predict the pre-bounce observations it is asked about.
A single number at the wicket plane would have called all three of them
"perfect".

`ballistic_drag` is the *worst* model at the wicket plane, which looks
paradoxical and is not: it estimates the drag coefficient correctly but models
only the pre-bounce arc, so it is extrapolated across the bounce like the other
single-arc models. Its poor showing is a second demonstration that structural
correctness beats physical sophistication.

### 9.5 Camera geometry and calibration

Median 3-D reconstruction error (mm) against stereo baseline (0.5 px centroid
noise):

| baseline (m) | 0.5 | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| error (mm) | 157.9 | 81.4 | 41.7 | 20.9 | 12.7 | 8.3 |

The error falls approximately as `1 / baseline`, as the stereo depth relation
`dZ ≈ Z² / (f B) · d(uv)` predicts, and flattens only where other error sources
take over.

The front-end pipeline, end to end on rendered synthetic footage at 150 fps:

| stage | result |
|---|---|
| frames detected | 80 / 80 in both cameras |
| centroid accuracy | 0.35 px |
| 3-D reconstruction RMS | 7.4 mm depth, 0.2 mm lateral, 1.0 mm vertical |
| **wicket-plane error** | **0.4 mm** |

Sub-millimetre reconstruction from half-pixel detections, once the geometry is
correct — which is a statement about the *geometry*, not about the detector.

### 9.6 Uncertainty

**Convergence.** On a delivery whose true clearance is *exactly zero*, the
estimated hit probability settles as

| N | 10 | 100 | 1 000 | 10 000 | 50 000 |
|---|---|---|---|---|---|
| `P(hit)` | 0.900 | 0.940 | 0.941 | 0.938 | 0.941 |

A few thousand trajectories are enough; beyond that the estimate moves by less
than a percentage point. Across independent noise realisations on the same
delivery, however, `P(hit)` ranges from 0.02 to 0.99 with a mean of 0.54.

That spread is the honest answer to "would it have hit?" for a ball grazing the
stump, and it is precisely the situation the umpire's-call rule exists to
handle.

**Coverage.** Across 40 independent noisy realisations, the nominal 95 %
interval of the predicted arrival point covered the truth

| coordinate | empirical coverage |
|---|---|
| lateral (`y`) | 95.0 % |
| vertical (`z`) | 92.5 % |
| both jointly | 87.5 % |

So the intervals are honest laterally and mildly **over-confident** vertically.
Reporting that, rather than a nominal 95 % and a confident claim, is the point:
the interval is derived from measurement error alone and takes no account of
model error.

---

## 10. Discussion and limitations

**What the results establish.**

1. *Structure beats precision.* A structurally correct model on 5 mm-noisy data
   (4 mm error) beats a structurally wrong one on perfect data (900 mm error) by
   more than two orders of magnitude. The corollary is a practical design rule:
   **spend effort on the model, not on the detector.** Improving the centroid
   from 1 px to 0.1 px helps the depth coordinate tenfold; it does nothing for a
   model that has the wrong number of arcs.

2. *The bounce is the whole problem.* Both the mechanistic analysis (§5.3) and
   the frame-budget experiment (§9.2) point the same way. A tracker must observe
   the post-bounce flight; six frames of it are worth more than twenty frames
   without it.

3. *The geometry is well conditioned where it matters.* The lateral and vertical
   coordinates — the ones the LBW zones are defined on — reconstruct to
   millimetres. The poorly conditioned depth coordinate barely matters for the
   decision, which is why a real DRS review can be reliable despite the depth
   ambiguity.

4. *Uncertainty is not decoration.* The estimator's covariance correctly tracks
   measurement noise, but it is mildly over-confident because it cannot see
   model error. The bootstrap comparison exists to make that gap visible.

**Limitations.**

- **No real footage.** Every result is synthetic, with exact ground truth. The
  vision path uses rendered frames. Broadcast footage — motion blur, camera
  movement, crowds, occlusion, a ball that is 8 px across — would need a trained
  detector, and the detector here is a colour threshold.
- **The Laws are simplified.** Three zones and a symmetric umpire's-call band
  are modelled; the full Law 36 conditions, and the real ICC margin values, are
  not.
- **A fixed impact plane.** Contact with the batter is taken at
  `x = -0.25 m` (parameterisable) rather than from the batter's actual pad
  position, which a real system would estimate.
- **Drag is modelled but not exploited.** The drag-aware estimator fits only the
  pre-bounce arc; a full drag model would need to handle the bounce as a
  non-smooth event inside the integration.
- **A single bounce model.** `e`, `k_x`, `k_y` are assumed constant per
  delivery. Real pitches vary along their length, and ball ageing changes both.
- **No spin dynamics.** Turn is represented by a velocity impulse at the bounce,
  not by integrating the Magnus force, which would also be needed for drift and
  dip in flight.

---

## 11. Future work

The natural extensions, in order of how much they would add:

1. **Estimate the bounce parameters, not just the bounce time.** Treat
   `(e, k_x, k_y)` as unknowns alongside `(r0, v0)`, and study the identifiability
   of restitution from a short post-bounce arc.
2. **Full drag with an event-handled bounce.** Replace the impulse model with a
   continuous impact model and re-measure the drag-free approximation error.
3. **A realistic image chain.** Train or adapt a detector on real footage, add
   camera-motion compensation, and compare against the synthetic baseline.
4. **A proper state-space model for spin.** Integrate the Magnus force so that
   drift and dip are predicted above the bounce rather than inferred from it.
5. **Bayesian model comparison.** Replace the RSS-based break search with a
   marginal-likelihood criterion (BIC/WAIC), and study whether the discrete
   search can be replaced by a continuous relaxation.
6. **Calibration uncertainty.** Propagate uncertainty in `K`, `R`, `t` — not
   just in the detections — into the wicket prediction. §9.5 shows the focal
   length matters; the project currently only perturbs it deterministically.

---

## 12. Reproducing everything

```bash
pip install -r requirements.txt

python -m pytest tests/ -q              # 138 tests
python scripts/run_demo.py              # one delivery, all figures
python scripts/run_video.py --synthetic good_length_middle
python scripts/run_experiments.py       # all six studies, ~3 minutes
streamlit run app/streamlit_app.py      # interactive
```

Every experiment is seeded, so the tables above are reproducible. Raw trial data
lands in `outputs/results/*.csv`, figures in `outputs/figures/`, and a generated
digest in `outputs/results/SUMMARY.md`.

### Reference

The starting point for the literature review is the paper *3D Ball Trajectory
from a Single Camera: A Cost-Effective Ball Estimation Technique for Cricket*,
which covers monocular camera geometry, a physics-based model, Kalman filtering,
bounce estimation and LBW analysis. Because that work is close to the present
proposal, the contribution here is deliberately placed in the **comparative and
uncertainty-aware** direction rather than in reproducing a single pipeline:
physics-constrained estimation versus polynomial fitting under controlled noise,
the effect of missing detections and calibration error, and an
uncertainty-aware wicket-intersection model. Publication venue, method details
and data availability should be verified before relying on it.
