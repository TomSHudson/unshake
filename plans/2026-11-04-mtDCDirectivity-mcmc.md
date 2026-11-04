# Plan: MCMC (adaptive Metropolis) for the MT + DC directivity inversion

**Status:** implemented 2026-11-04.
**Follows:** Phase 2 of `plans/2026-11-03-mtDCDirectivity-nonlinear-inversion.md`. Uses the corrected likelihood from `plans/2026-11-04-correlated-noise-uncertainty.md`.

## Why
- Prior-sampling MC has an ESS of about 1 in 10-D.
- NLLS gives only a local Gaussian (GN) uncertainty, so it can't show non-Gaussian posteriors, e.g. for sources with little DC, where the data barely constrain **v**, or the ball prior |**v**| ≤ v_max.
- MCMC started from the NLLS optimum samples the full posterior efficiently.

## Method: adaptive Metropolis (Haario et al. 2001), pure numpy
- **Target:** log posterior = `make_loglike` (Gaussian, σ = `info['sigma']` from NLLS) + `model.log_prior`.
- **Chains:** `n_chains` (8) chains start at draws from N(p_NLLS, cov_GN). Draws outside the prior are reset to p_NLLS.
- **Proposals:**
  - each step proposes p' = p + N(0, 2.38²/d · C) for every chain, with Metropolis acceptance;
  - all chains go through one batched `forward` call;
  - C starts as cov_GN.
- **Adaptation:** during burn-in (default n_steps/2), C is re-estimated every 100 steps from the latest half of all chains. After burn-in it is fixed, so the kept samples are from a standard Metropolis sampler.
- **Diagnostics:** acceptance rate, Gelman–Rubin R-hat per parameter, and the integrated autocorrelation time τ. For τ, the chain-averaged autocorrelation is summed up to its first negative value; ESS = n_kept · n_chains / τ.
- **No gradients needed.** HMC/NUTS (NumPyro, PyMC) stays a possible improvement and would need a differentiable forward model.

## Code
- **`models.MTDCDirectivityModel.log_prior(params)`:**
  - flat in the MT within `bounds()`;
  - log-uniform fc0 over `fc0_range`;
  - a truncated isotropic Gaussian on **v** (σ_v, |**v**| ≤ v_max);
  - −inf outside. It matches `sample_prior` for fc0 and **v**.
  - Listed as an optional interface method in the `models.py` docstring.
- **`solvers.mcmc(model, dataset, p0, cov, n_steps=4000, n_chains=8, burn=None, sigma=None, adapt=True, rng=None, **loglike_kw)`:**
  - returns a `Result` of the kept samples;
  - `meta` holds acceptance, rhat, tau, ess and chain (all steps, for trace plots);
  - small helpers `_sqrtm`, `_rhat` and `_autocorr_time`.
- **`results.Result`:**
  - new field `equal_weights` (True for MCMC posterior draws), so `weights` are uniform;
  - `ess` uses `meta['ess']` when the sampler sets it.
- **Exports and docs:** `unshake.mcmc` is exported, and the README is updated.
- **Notebook:** `examples/directional_inversion_demo.ipynb` §5 (MCMC on the demo data) and §6 (MCMC on the pure crack).

## Results (demo data: 60 % DC, fc0 = 60 Hz, **v** = 0.6 vs towards 45°, SNR 50)

**Demo source:**
- 8 chains × 4000 steps (burn-in 2000) in 30 s;
- acceptance 0.25, max R-hat 1.011, τ ≈ 26–36 steps, ESS 441 (min over parameters). For comparison, MC gave ESS 1 from 50k samples.
- The MCMC means and σ match NLLS ± corrected GN σ, because the posterior is close to Gaussian:
  - fc0 59.97 ± 0.24 Hz;
  - v_e 745 ± 15 m/s, v_n 763 ± 16 m/s (true 772, 772);
  - horizontal |**v**| 0.587 vs, azimuth 44.3°.
- v_u: 61 ± 597 m/s. That is narrower than the prior σ_v = 900 because of the ball |**v**| ≤ 1600 m/s, which MCMC respects and NLLS does not.

**Pure crack (C_DC ≈ 0):**
- acceptance 0.25, max R-hat 1.046;
- the posterior of |**v**| is close to the prior: the data carry no information on **v**, as they shouldn't. The NLLS point estimate alone could not show this.

## Possible improvements (not done)
- **HMC/NUTS** (NumPyro/PyMC) with a JAX port of the forward model. It needs a smoothed |·| in the DC/CLVD split.
- **Speed:**
  - cache `rfft(gf)` for faster forwards;
  - profile the MT amplitude analytically.
- **Multimodal posteriors:** start chains from several NLLS optima, e.g. ±**v**.
- **Correlated noise:** whiten the data by the noise spectrum, in place of the scalar factor.
