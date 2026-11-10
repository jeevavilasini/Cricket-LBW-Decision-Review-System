"""
Interactive demonstration of the DRS model.

Run with:

    streamlit run app/streamlit_app.py

The interface follows the four modules of the project: choose a delivery, see
what the tracker observes, see what the estimators reconstruct, and see what
the decision layer concludes -- with the uncertainty made visible rather than
hidden behind a single number.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from drs import config, physics, viz  # noqa: E402
from drs.config import DEFAULT_UMPIRE_CALL_MARGIN  # noqa: E402
from drs.decision import review_lbw  # noqa: E402
from drs.estimate import fit, state_at_plane  # noqa: E402
from drs.geometry import Wicket  # noqa: E402
from drs.simulate import build_ground_truth, make_observations  # noqa: E402
from drs.uncertainty import bootstrap_wicket, monte_carlo_wicket  # noqa: E402

st.set_page_config(page_title="DRS ball tracking", layout="wide")

MODELS = ("polynomial", "ballistic", "post_bounce", "piecewise", "kalman")


# --------------------------------------------------------------------------
# Cached heavy lifting
# --------------------------------------------------------------------------
# ``st.cache_resource`` rather than ``st.cache_data``: the fitted trajectories
# capture their parameters in a closure, which is not picklable, and
# ``cache_data`` pickles its return value.  ``cache_resource`` stores objects by
# reference, which is the right primitive for them.  The scenario objects are
# treated as immutable here.


@st.cache_resource(show_spinner=False)
def build_scenario(preset_name: str, fps: float, noise_mm: float, dropout: float, seed: int):
    """Simulate a delivery, observe it, and fit every model."""
    truth = build_ground_truth(physics.preset(preset_name), label=preset_name)
    obs = make_observations(truth, fps=fps, noise_std=noise_mm / 1000.0, dropout=dropout, rng=seed)

    rows = []
    fits = {}
    for name in MODELS:
        try:
            f = fit(obs, name)
        except (ValueError, np.linalg.LinAlgError, RuntimeError) as exc:
            rows.append({"model": name, "error": str(exc)})
            continue
        fits[name] = f
        p = state_at_plane(f)
        t_true, r_true, _ = truth.state_at_wicket()
        rows.append(
            {
                "model": name,
                "fit RMSE (mm)": round(f.residual_rmse * 1000, 2),
                "arrival y (m)": round(float(p[1]), 4),
                "arrival z (m)": round(float(p[2]), 4),
                "3-D error (mm)": round(float(np.linalg.norm(p - r_true)) * 1000, 1),
            }
        )
    return truth, obs, fits, pd.DataFrame(rows)


@st.cache_resource(show_spinner=False)
def run_monte_carlo(preset_name: str, fps: float, noise_mm: float, dropout: float, seed: int,
                    model: str, n: int):
    """Monte-Carlo hit probability for one model, cached across widgets."""
    truth, obs, fits, _ = build_scenario(preset_name, fps, noise_mm, dropout, seed)
    f = fits.get(model)
    if f is None:
        return None, None
    unc = monte_carlo_wicket(f, n=n, rng=seed)
    return f, unc


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------

st.sidebar.title("Delivery")
preset_name = st.sidebar.selectbox("Preset delivery", sorted(physics.PRESETS), index=sorted(physics.PRESETS).index("leg_spin_marginal"))
fps = st.sidebar.slider("Camera frame rate (Hz)", 25.0, 300.0, 50.0, 25.0)
noise_mm = st.sidebar.slider("Measurement noise (mm)", 0.0, 40.0, 5.0, 0.5)
dropout = st.sidebar.slider("Frames not detected", 0.0, 0.6, 0.0, 0.05)
seed = st.sidebar.number_input("Random seed", 0, 10_000, 7)

st.sidebar.title("Decision")
on_field = st.sidebar.radio("On-field decision", ("not_out", "out"), horizontal=True)
margin_mm = st.sidebar.slider("Umpire's-call margin (mm)", 0.0, 60.0, DEFAULT_UMPIRE_CALL_MARGIN * 1000, 1.0)
offered_shot = st.sidebar.checkbox("Batter offered a shot", value=True)
impact_x = st.sidebar.slider("Impact plane x (m)", -1.5, -0.05, -0.25, 0.05)

st.sidebar.title("Uncertainty")
model = st.sidebar.selectbox("Model for the decision", MODELS, index=3)
n_mc = st.sidebar.select_slider("Monte-Carlo trajectories", (1000, 5000, 10000, 20000, 50000), value=10000)
n_boot = st.sidebar.select_slider("Bootstrap resamples", (0, 50, 100, 200), value=0)

truth, obs, fits, table = build_scenario(preset_name, fps, noise_mm, dropout, int(seed))
wicket = Wicket()
t_true, r_true, v_true = truth.state_at_wicket()
true_clearance = float(wicket.clearance(r_true.reshape(1, 3))[0]) * 1000

# --------------------------------------------------------------------------
# Headline
# --------------------------------------------------------------------------

st.title("Mathematical modelling of cricket's Decision Review System")
st.caption(
    "A trajectory-estimation prototype: simulate or observe a delivery, reconstruct its path, "
    "and decide whether it would have hit the wicket. This is a research model, not an "
    "officiating system."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Release speed", f"{np.linalg.norm(truth.release_velocity) * 3.6:.1f} km/h")
c2.metric("Bounces at", f"x = {truth.trajectory.pitch_point()[0]:.2f} m")
c3.metric("Arrival height", f"{r_true[2] * 1000:.0f} mm")
c4.metric("True clearance", f"{true_clearance:+.0f} mm",
          help="Distance from the ball surface to the nearest point of the wicket. "
               "Negative means the ball would hit.")

tab_pipeline, tab_decision, tab_uncertainty, tab_data = st.tabs(
    ["Pipeline", "Decision", "Uncertainty", "Data"]
)

# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

with tab_pipeline:
    col_a, col_b = st.columns([3, 2])
    with col_a:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))
        viz.plot_side_view(truth, fits=list(fits.values()), obs=obs, ax=axes[0])
        viz.plot_plan_view(truth, fits=list(fits.values()), obs=obs, wicket=wicket, ax=axes[1])
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    with col_b:
        st.subheader("Trajectory in three dimensions")
        fig = viz.plot_trajectory_3d(truth, fits.get("piecewise"), wicket)
        st.pyplot(fig, width="stretch")
        plt.close(fig)

    st.subheader("Estimator comparison")
    st.dataframe(table, width="stretch", hide_index=True)
    st.caption(
        "A single parabola fitted across the bounce is structurally wrong: the bounce is a "
        "discontinuity in velocity, so the model error dwarfs the measurement error."
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
    viz.plot_fit_comparison(obs, list(fits.values()), truth=truth, ax=axes[0])
    viz.plot_residuals(obs, list(fits.values()), ax=axes[1])
    st.pyplot(fig, width="stretch")
    plt.close(fig)

# --------------------------------------------------------------------------
# Decision
# --------------------------------------------------------------------------

with tab_decision:
    chosen = fits.get(model)
    if chosen is None:
        st.error(f"The '{model}' model could not be fitted to this scenario.")
    else:
        verdict = review_lbw(
            chosen,
            wicket=wicket,
            on_field=on_field,
            margin=margin_mm / 1000.0,
            impact_x=impact_x,
            offered_shot=offered_shot,
        )
        p = state_at_plane(chosen)

        left, right = st.columns([2, 3])
        with left:
            st.subheader("Verdict")
            banner = f"### {'OUT' if verdict.decision == 'out' else 'NOT OUT'}"
            if verdict.umpires_call:
                banner += "  \n_on-field decision stands (umpire's call)_"
            st.markdown(banner)

            st.markdown(
                f"- **Pitching** — {verdict.pitching.zone.replace('_', ' ')}"
                + (" (marginal)" if verdict.pitching.marginal else "")
            )
            st.markdown(
                f"- **Impact** — {verdict.impact.zone.replace('_', ' ')}"
                + (" (marginal)" if verdict.impact.marginal else "")
            )
            st.markdown(
                f"- **Wicket** — {verdict.wicket.zone.replace('_', ' ')}"
                + f" (clearance {verdict.clearance * 1000:+.1f} mm)"
            )
            st.markdown("**Reasons**")
            for reason in verdict.reasons:
                st.markdown(f"- {reason}")

        with right:
            st.subheader("Wicket plane")
            _, unc = run_monte_carlo(preset_name, fps, noise_mm, dropout, int(seed), model, int(n_mc))
            fig, ax = plt.subplots(figsize=(5.6, 5.2))
            viz.plot_wicket_plane(truth, fits=list(fits.values()), uncertainty=unc, wicket=wicket, ax=ax)
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        st.info(
            "The verdict follows a simplified reading of Law 36. Umpire's call means the "
            "prediction is within the tolerance of a zone boundary, so the on-field decision "
            "is left standing rather than overruled."
        )

# --------------------------------------------------------------------------
# Uncertainty
# --------------------------------------------------------------------------

with tab_uncertainty:
    chosen, unc = run_monte_carlo(preset_name, fps, noise_mm, dropout, int(seed), model, int(n_mc))
    if chosen is None or unc is None:
        st.error("No fit available for this configuration.")
    else:
        a, b, c = st.columns(3)
        a.metric("P(hit)", f"{unc.hit_probability:.3f}", help="Fraction of simulated trajectories that hit the wicket.")
        b.metric("Mean clearance", f"{unc.clearances.mean() * 1000:+.2f} mm")
        c.metric("Clearance spread", f"{unc.clearances.std() * 1000:.2f} mm")

        st.caption(
            "This probability describes uncertainty in *our* model — from noisy measurements and "
            "the fitted parameter covariance. It is not the official DRS probability, and it is not "
            "the umpire's-call rule."
        )

        left, right = st.columns(2)
        with left:
            fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
            axes[0].hist(unc.clearances * 1000, bins=50, color="#3d5a80", alpha=0.85)
            axes[0].axvline(0.0, color="#d1495b", linestyle="--", linewidth=1.5, label="contact threshold")
            axes[0].set_xlabel("clearance (mm)")
            axes[0].set_ylabel("count")
            axes[0].set_title("Distribution of clearance")
            axes[0].legend(fontsize=8)
            axes[0].grid(alpha=0.25)

            from drs.uncertainty import hit_probability_convergence

            sizes = (100, 500, 1000, 5000, 10000, 50000)
            conv = hit_probability_convergence(chosen, sizes=sizes, rng=0)
            viz.plot_uncertainty_convergence([s for s, _ in conv], [p for _, p in conv], ax=axes[1])
            st.pyplot(fig, width="stretch")
            plt.close(fig)

        with right:
            st.markdown(
                f"- 95% interval for arrival **y**: "
                f"[{unc.interval(1)[0]:+.4f}, {unc.interval(1)[1]:+.4f}] m\n"
                f"- 95% interval for arrival **z**: "
                f"[{unc.interval(2)[0]:.4f}, {unc.interval(2)[1]:.4f}] m\n"
                f"- standard deviation of the arrival point: "
                f"({unc.std_point[1] * 1000:.2f}, {unc.std_point[2] * 1000:.2f}) mm in (y, z)"
            )
            if n_boot and n_boot > 0:
                boot = bootstrap_wicket(obs, model=model, n=int(n_boot), rng=int(seed))
                st.markdown(
                    f"**Bootstrap check** (refitting {boot.n_samples} resampled datasets): "
                    f"P(hit) = {boot.hit_probability:.3f}. Where this differs from the "
                    "parameter-covariance value, the difference is model error rather than "
                    "measurement error."
                )
            else:
                st.caption("Set 'Bootstrap resamples' in the sidebar to compare against a bootstrap interval.")

# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

with tab_data:
    st.subheader("Observations")
    df = pd.DataFrame(
        {
            "t (s)": obs.t,
            "x (m)": obs.positions[:, 0],
            "y (m)": obs.positions[:, 1],
            "z (m)": obs.positions[:, 2],
            "detected": obs.visible,
        }
    )
    st.dataframe(df, width="stretch", hide_index=True)
    st.download_button("Download observations as CSV", df.to_csv(index=False), "observations.csv")

    with st.expander("Physical constants and geometry used"):
        st.json(
            {
                "gravity_m_s2": config.G,
                "ball_radius_mm": config.BALL_RADIUS * 1000,
                "ball_mass_kg": config.BALL_MASS,
                "stump_height_mm": config.STUMP_HEIGHT * 1000,
                "wicket_width_mm": config.WICKET_WIDTH * 1000,
                "stump_spacing_mm": config.STUMP_SPACING * 1000,
                "pitch_length_m": config.PITCH_LENGTH,
                "umpire_call_margin_mm": DEFAULT_UMPIRE_CALL_MARGIN * 1000,
            }
        )
