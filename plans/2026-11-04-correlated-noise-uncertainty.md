# Plan: correct NLLS uncertainties for correlated (band-limited) noise

**Status:** implemented 2026-11-04.
**Follows:** `plans/2026-11-03-mtDCDirectivity-nonlinear-inversion.md`, which found that the Gauss–Newton (GN) σ were too small (z-scores up to 2.8).

## Problem
- The Gaussian likelihood, $-\tfrac12\sum (d-u)^2/\sigma^2$, treats every time sample as independent.
- The data are band-limited (2–210 Hz at 1 kHz sampling), so the noise has fewer independent samples, $n_{\rm eff} < n_t$.
- The likelihood is therefore too narrow, and the GN covariance too small by $\sqrt{n_t/n_{\rm eff}}$.

## Approach (simplest that works)
Inflate σ by a scalar factor estimated from the noise spectrum:

$$\sigma_{\rm eff} = \sigma\sqrt{f}, \qquad f = \frac{n_t}{n_{\rm eff}} = \frac{n_f\sum_f P(f)^2}{\big(\sum_f P(f)\big)^2}$$

- $P(f)$ is the power spectrum averaged over traces.
- $f$ = 1 for white noise, and $f$ = 1/(fraction of the band occupied) for noise with a flat spectrum in a band. It is exact in that case and approximate otherwise.
- **The exact alternative**, not done, is to whiten the data and synthetics by $1/\sqrt{P(f)}$, i.e. a frequency-domain likelihood. It is more code and needs a stable noise-spectrum estimate.

## Code
- **`misfit.noise_correlation_factor(noise)`:** the factor above, from any (n_rec, n_t) noise traces: a residual, or a pre-event window.
- **`solvers.nonlinear_least_squares(..., correlated_noise=True)`:**
  - after the fit (and the χ² estimate of σ, if σ was not given), computes $f$ from the residual at the optimum;
  - inflates σ and refits once. The refit matters because σ sets the weight of the prior damping on **v**: the covariance of v_u, set by the prior alone, stays at σ_v.
  - `info['sigma']` is the effective σ, and `info['noise_factor']` is $f$.
  - Pass `info['sigma']` to `mcmc` / `make_loglike` so that they use the same likelihood.
- **`solvers.monte_carlo`:** the default σ from the model's starting fit is inflated in the same way.

## Verification (`examples/directional_inversion_demo.ipynb` §4)
- **White noise:** $f$ = 1.008. **Band-limited demo noise:** $f$ = 2.62 (2.60 ± 0.01 from residuals over 40 noise draws).
- **30 noise realisations, NLLS each.** The ratio of the scatter of the estimates to the mean GN σ:

  | | Mnn | Mee | Mdd | Mne | Mnd | Med | fc0 | v_e | v_n |
  |---|---|---|---|---|---|---|---|---|---|
  | scatter / σ (white) | 1.60 | 1.65 | 1.60 | 1.94 | 1.31 | 1.93 | 1.61 | 1.36 | 1.42 |
  | scatter / σ (corrected) | 0.99 | 1.02 | 0.99 | 1.20 | 0.81 | 1.19 | 1.00 | 0.84 | 0.88 |

  The corrected ratios are about 1, within the ±13 % sampling uncertainty of a 30-run scatter.
- **Demo data:** the largest |z| of NLLS against the truth falls from 2.8 to 1.7.

## Limitations
- One scalar factor for all traces and components. It assumes a stationary noise spectrum that is the same on every trace.
- Estimated from the residual, so model error is counted as noise. That is conservative.
