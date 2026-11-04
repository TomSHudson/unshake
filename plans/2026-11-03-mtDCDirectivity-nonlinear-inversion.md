# Plan: non-linear inversion for MT + fc0 + rupture velocity vector (MTDCDirectivityModel)

## Context
- The forward model `models.MTDCDirectivityModel` was committed in b6ef12a. It splits the MT into a DC part, convolved with a Doppler-shifted Brune, and a non-DC part, convolved with the symmetric Brune at fc0.
- Its parameters are `p = [Mnn, Mee, Mdd, Mne, Mnd, Med, fc0, v_e, v_n, v_u]`, with **v** = vr · d̂.
- The inversion is now non-linear. unshake has only the linear `least_squares` and prior-sampling `monte_carlo`.
- **Goal:** invert for m6, fc0 and **v** with simple, logical code.

**User decisions:**
- Target MTDCDirectivityModel only, with fc0 free (10 parameters).
- Every parameter gets **finite, symmetric bounds**. The signs of the MT and **v** components stay free: they are signed by definition, and only fc0, |**v**| and M0 are positive.
- **Starting fc0:** a variable-projection scan plus Brent refinement, with an `fc0=` override so that an fc0 measured from the data (the user's method, to come later) can be plugged in.
- **Phasing:** implement **MC + deterministic NLLS now**. The MCMC (adaptive Metropolis) comes later, as a separate step. HMC/NUTS is flagged as a possible improvement.
- **Plan note filename:** `plans/2026-11-03-mtDCDirectivity-nonlinear-inversion.md`.

**Measured cost:** one `forward` takes about 1.1 ms on the demo Dataset (150 traces, n_t = 100) and scales linearly with batch size. The NLLS should take about 1 s; 100k MC samples take about 2 min.

## Phase 1 (implement now)

### MC: existing `monte_carlo`, one fix
- `monte_carlo(MTDCDirectivityModel(...), ds, n)` already runs, since the model has `sample_prior` and `forward`.
- **Fix needed: the prior amplitude.**
  - `_Scaled.prepare` sets `scale` from LSQ on the *formal* GFs (no STF).
  - The data are low-passed by the Brune STF, so that norm is biased.
  - Override `prepare` in `MTDCDirectivityModel` to set `scale = mt.norm(m0)` from `find_non_linear_starting_model_params` (single-Brune variable projection, below), and cache `self.p0`.
- Expect a tiny ESS in 10-D. The demo reports it as the baseline.

### Deterministic NLLS: `scipy.optimize.least_squares`
- **Unknowns:** p in natural units (no log transforms).
  - Conditioning comes from explicit `x_scale = [scale]*6 + [fc0_start] + [σ_v]*3`, so every dimension is O(1) for the optimiser.
  - **Answer to "log of the MT components?":** no point. They are signed, so a log can't be taken. The problem is scale, which a linear scale fixes.
  - The alternative, log M0 plus angles, brings back the singularities and wrap-around that the m6 and **v** parameterisation avoids.
- **Residuals:**
  - data: `sqrt(w_i)(d − u(p))/σ_i`, flattened;
  - prior damping: **v**/σ_v, which matches the model's Gaussian prior on **v** and keeps low-DC / tensile sources well-posed.
- **Bounds** (`model.bounds()`):
  - |m_k| ≤ `a_max · scale`, with `a_max = scale_range[1]` if set, else 3 (|M_ij| ≤ ‖M‖_F);
  - fc0 within `fc0_range`;
  - |v_j| ≤ v_max. The ball |**v**| ≤ v_max is checked after the fit, with a warning if it is violated.
- **Method:** TRF with a 2-point numerical Jacobian (about 11 forwards per iteration).
- **Start** (`model.find_non_linear_starting_model_params(dataset, fc0=None, fc0_grid=None)`), variable projection:
  1. For a given fc0, convolve the GFs with a single Brune (`forward.convolve_stf` + `stfs.brune_stf_spectrum`, on `dataclasses.replace(dataset, gf=...)`).
  2. Solve the MT with the existing `solvers.least_squares(..., check_band=False)`, and record the weighted RSS.
  3. Scan a coarse log grid of about 15 points over `fc0_range`.
  4. Refine with `scipy.optimize.minimize_scalar(method='bounded')` in log fc0 between the neighbours of the grid minimum.
  - This takes about 25 cheap LSQ solves and gives an accurate fc0, not a wide spread.
  - `fc0=` skips the scan.
  - Returns `p0 = [m0, fc0, 0, 0, 0]`. At **v** = 0, ∂u/∂**v** ≠ 0, so the solver can move away from it.
- **Multi-start:**
  - start from p0;
  - add `n_starts` (default 4) starts that perturb **only v**, in random directions with |**v**| = σ_v/2;
  - keep fc0 and m0 fixed;
  - keep the lowest cost.
  - This covers the weak sign ambiguity of **v** and the kink at γ = 0.
- **Noise σ:**
  - Use `sigma=` or `dataset.noise_std` if either is set.
  - Otherwise, run pass 1 with σ from the residual std of the variable-projection start, then re-estimate σ² = χ²/(N − p) at the optimum and refit once (warm start). This keeps the prior damping weighted correctly.
- **Uncertainty:** Gauss–Newton `cov = (JᵀJ)⁻¹` from the scipy result, in natural units, because the residuals are already σ-scaled.
- **Returns:** `(p, cov, info)`, with `info = {sigma, cost, vr, nfev, start_costs}`, and a printed one-line summary.

### Code changes

#### `src/unshake/models.py` (`MTDCDirectivityModel`)
- `find_non_linear_starting_model_params(dataset, fc0=None, fc0_grid=None)`, plus a small module-level helper `_single_brune_lsq(dataset, fc0) -> (m, rss)`.
- `prepare(dataset)`: the override described above (it sets `scale` and caches `p0`).
- `bounds() -> (lo, hi)` and `param_scales(p0) -> (n_params,)` for the optimiser.
- `prior_residuals(params) -> (3,)`: **v**/σ_v.
- Extend the module docstring: these are *optional* interface methods used by `nonlinear_least_squares`.

#### `src/unshake/solvers.py`
- `nonlinear_least_squares(model, dataset, p0=None, sigma=None, n_starts=4, rng=None, **lsq_kw)`.
  - It is generic: it uses only `forward`, `bounds`, `param_scales`, the optional `prior_residuals`, and `find_non_linear_starting_model_params` when p0 is None.
  - It applies the dataset weights in the same way as `least_squares`.
- Update the module docstring.

#### `src/unshake/__init__.py`, `README.md`
- Export `nonlinear_least_squares`.
- Update the solvers line in the README.

#### `examples/directional_inversion_demo.ipynb` (new)
- **Load:** the cached `stf_demo_out/stf_demo_dataset.pkl`. Rebuild the true source and data exactly as in `directional_stf_demo.ipynb` §3/§5: 60 % DC, fc0 = 60 Hz, **v** = 0.6 vs towards 45°, SNR 50.
- **Sections:**
  1. The variable-projection fc0 start: the RSS-vs-fc0 curve, with the refined fc0 marked.
  2. MC prior sampling (about 100k samples): ESS, best VR, and the marginals of the top samples.
  3. NLLS:
     - recovered parameters against the true ones, with GN 1σ;
     - the rupture speed, azimuth and plunge with uncertainties, from draws of N(p, cov) through `model.rupture` (no forwards needed);
     - waveform fit (`plot.wavefield`), lune and beachball.
  4. A noise-free check and a pure-crack check (**v** → about 0).
  5. A short comparison table of MC and NLLS (cost, best VR, uncertainty).

#### `plans/2026-11-03-mtDCDirectivity-nonlinear-inversion.md` (new)
This plan, the method assessment below, and an Outcome section with the numbers from the demo.

## Phase 2 (later, separate approval): an efficient sampler
- **Adaptive Metropolis (Haario 2001), pure numpy, about 50 lines.**
  - The chains move in lockstep (one batched `forward` per step), so they give a split R-hat.
  - Initialise at the NLLS optimum, with the proposal covariance 2.38²/d · cov_GN.
  - Adapt only during burn-in.
- **What it needs:**
  - `model.log_prior` (flat m6 within the bounds, log-uniform fc0, truncated Gaussian **v**);
  - `Result.log_weights`, so that MCMC samples are equally weighted.
- **Improvements to flag (not now):**
  - **NumPyro / PyMC NUTS.**
    - Needs a differentiable forward model: a JAX or PyTensor port of `forward`, `convolve_stf` and `mt.decompose`.
    - The gradient through `eigh` is singular at degenerate eigenvalues (ISO, CLVD, crack) and kinked at γ = 0 (|·| in M_DC), so it would need a smoothed |·|.
    - Worth it if the dimension grows (finite or multiple sources).
  - **numba:** little gain. The cost is already in compiled BLAS/FFT, and numba doesn't support `np.fft`.
  - **Other options:**
    - emcee as an optional extra;
    - importance sampling with a Laplace (GN) proposal;
    - caching `rfft(gf)`;
    - analytic profiling of the MT amplitude.

## Outcome (Phase 1 implemented 2026-11-03)

### Code
- **`models.MTDCDirectivityModel`:**
  - `prepare` sets `p0` (variable projection) and `scale = ||m0||`.
  - `find_non_linear_starting_model_params(dataset, fc0=None, fc0_grid=15)` gives a 15-point log scan, then `minimize_scalar` (bounded) in log fc0. `fc0=` skips the scan.
  - New methods: `bounds()`, `param_scales(p)`, `prior_residuals(p)` (**v**/σ_v), `starts(p0, n, rng)` (perturbs **v** only) and `check_params(p)` (warns if |**v**| > v_max).
  - New helper: `models._single_brune_lsq(dataset, fc0)`.
  - The module docstring lists the optional NLLS interface.
- **`solvers.nonlinear_least_squares(model, dataset, p0=None, sigma=None, n_starts=4, rng=None, **lsq_kw)`:**
  - generic, using TRF with bounds, `x_scale` from the model, and the prior residuals;
  - multi-start;
  - σ is re-estimated from χ²/(N − p) with one refit when no noise level is given;
  - the GN covariance is computed in scaled coordinates.
  - Returns `(p, cov, info)`.
- **`solvers.monte_carlo`:** if the model has a starting fit (`model.p0`) and no noise level is given, the Gaussian σ comes from that fit's residual. The old default used LSQ on the formal GFs and ignored the STF.
- **Exports and docs:** `unshake.nonlinear_least_squares` is exported, and the README is updated.
- **New demo:** `examples/directional_inversion_demo.ipynb`, with the same source and noisy data as `directional_stf_demo.ipynb` §5.

### Results (demo: 60 % DC, fc0 = 60 Hz, **v** = 0.6 vs towards 45°, SNR 50; true-model VR 0.9271)

**Variable projection:**
- single-Brune data at 60 Hz → 60.13 Hz;
- DC-directivity data → 63.13 Hz (0.09 s).

**MT amplitude for the MC prior:**
- old `prepare` (LSQ on formal GFs): scale 0.164;
- new: 1.07;
- true ||m||: 1.12.

**MC (prior sampling):** 50k samples in 42 s, ESS 1.0. The best sample has VR 0.822, fc0 70.7 Hz, |**v**| 0.43 vs at 235°.

**NLLS (noisy):**
- 5 starts, all reaching the same minimum; 32 forward evaluations in 0.3–0.4 s; VR 0.9272.
- fc0 = 59.97 ± 0.16 Hz.
- (v_E, v_N) = (745 ± 10, 763 ± 10) m/s, against a true (772, 772). Horizontal |**v**| = 0.586 vs (95 %: 0.574–0.597), azimuth 44.3° (43.4–45.2).
- Lune (γ, δ) = (−6.0°, 17.1°), against a true (−6.6°, 18.0°). C_DC = 0.62 (true 0.60).
- **Some z-scores reach 2–2.8.** The GN covariance assumes white noise, but the noise is band-limited (2–210 Hz at 1 kHz), so σ is too small by about √(n_t/n_eff) ≈ 1.5. This is documented in the docstring and the notebook.
- v_u = 0 ± 900 m/s is prior-only, because all receivers lie in the horizontal plane of the source.

**NLLS (noise-free):** relative errors about 1e-13 (MT, fc0, **v**).

**Pure crack (3:1:1):**
- converges; fc0 = 60.2 Hz; estimated C_DC = 0.006.
- **v** = (−677, 834) ± (433, 463) m/s, not significant (1.5–2σ from 0).
- The spurious sub-1 % DC part, fitted to noise, gives **v** weak sensitivity. So for low-C_DC sources, **v** must be reported together with C_DC and not over-interpreted. This is weaker than the "**v** → ~0" expected in the plan.

**Regression:** `examples/synthetic_mt_recovery.py` runs unchanged.

### Next
Phase 2: the adaptive Metropolis MCMC, seeded from the NLLS optimum and `cov`. See above.

---

## Changes commited in commit: Initial NLLS (non-linear least squares) MT + DC directivity inversion

### New files
- **`examples/directional_inversion_demo.ipynb`:** the non-linear inversion demo for `MTDCDirectivityModel`, on the same GFs, source and noisy data as `directional_stf_demo.ipynb` §5. It needs the cached `examples/stf_demo_out/stf_demo_dataset.pkl`. Sections:
  1. the variable-projection starting model (RSS against fc0);
  2. Monte Carlo prior sampling (50k samples, ESS about 1);
  3. NLLS: parameters against the truth with GN 1σ, rupture speed, azimuth and lune from GN draws, (v_E, v_N) ellipses, waveform fit and beachballs;
  4. checks on noise-free data and a pure tensile crack;
  5. a comparison table of the methods.
- **`plans/2026-11-03-mtDCDirectivity-nonlinear-inversion.md`:** this plan. It covers the method assessment (MC, NLLS, MCMC, HMC), Phase 1 (implemented), Phase 2 (MCMC, planned) and the Outcome.

### Modified files
- **`src/unshake/models.py`:**
  - The module docstring lists the optional interface for the non-linear solvers: `bounds`, `param_scales`, `prior_residuals`, `find_non_linear_starting_model_params`, `starts` and `check_params`.
  - New module-level helper `_single_brune_lsq(dataset, fc0)`: linear MT least squares with one causal Brune at fc0 for the whole MT. Returns `(m, rss)`.
  - **`MTDCDirectivityModel`:**
    - **`prepare(dataset)`** (overrides `_Scaled.prepare`): sets `self.p0` from `find_non_linear_starting_model_params` and, if it is not given, `scale = ||m0||`. The base class took the scale from LSQ on the formal GFs, which ignores the STF: 0.164 against a true 1.12 in the demo.
    - **`find_non_linear_starting_model_params(dataset, fc0=None, fc0_grid=15)`:** variable projection.
      - A log-spaced fc0 scan, then `scipy.optimize.minimize_scalar` (bounded) in log fc0 between the neighbours of the grid minimum.
      - `fc0=` uses a given value, e.g. one measured from the data.
      - Returns `[m0, fc0, 0, 0, 0]`.
    - **`bounds()`:** finite, symmetric bounds: |m_k| ≤ a_max · scale (a_max = `scale_range[1]` or 3), fc0 within `fc0_range`, |v_j| ≤ v_max.
    - **`param_scales(p)`:** `[scale]*6, fc0, [sigma_v]*3`, for conditioning the optimiser.
    - **`prior_residuals(params)`:** **v**/σ_v, the Gaussian prior on **v** as damping.
    - **`starts(p0, n, rng)`:** p0 with |**v**| = σ_v/2 in random directions, for the multi-start.
    - **`check_params(params)`:** warns if |**v**| > v_max.
- **`src/unshake/solvers.py`:**
  - **New `nonlinear_least_squares(model, dataset, p0=None, sigma=None, n_starts=4, rng=None, **lsq_kw)`:**
    - `scipy.optimize.least_squares` (TRF, bounds, finite-difference Jacobian, `x_scale` from the model);
    - weighted, σ-scaled data residuals plus the model's prior residuals;
    - multi-start, keeping the lowest cost;
    - if no noise level is given, σ is re-estimated as χ²/(N − p) at the optimum and the fit is repeated once;
    - the Gauss–Newton covariance is computed in scaled coordinates;
    - prints a summary and returns `(p, cov, info)`.
  - **`monte_carlo`:** if the model has a starting fit (`model.p0`) and no noise level is given, the Gaussian σ comes from that fit's residual, not from LSQ on the GFs alone.
  - The module docstring lists the new solver.
- **`src/unshake/__init__.py`:** exports `nonlinear_least_squares`.
- **`README.md`:** the solvers line, and a pointer to the new demo notebook.

### Not changed
- The forward model, the other models, `least_squares`, `make_loglike`, `Result`, the plotting and the Salvus driver.
- `examples/synthetic_mt_recovery.py` still runs, with unchanged results.
- `examples/directional_stf_demo*.ipynb`.

### Notes
- **The GN covariance assumes white noise.** With band-limited noise it is too small by about √(n_t/n_eff), about 1.5 in the demo. This is stated in the docstring and the notebook.
- **For sources with little DC (C_DC ≈ 0), v is weakly and noisily constrained.** Report it together with C_DC.
- **The notebook is committed with outputs** (about 0.4 MB). Executing it takes about 1 min, most of it in the 50k-sample MC.

### Suggested commit message
```
Initial NLLS (non-linear least squares) MT + DC directivity inversion

Adds a non-linear least-squares (NLLS) inversion for MTDCDirectivityModel
(MT, fc0 and rupture velocity vector v). It recovers the MT, fc0 and the
horizontal v in under half a second; Monte Carlo runs as a baseline.

What was added
- Starting model: MTDCDirectivityModel.find_non_linear_starting_model_params(ds, fc0=None)
  finds a starting fc0 and MT. It scans 15 fc0 values on a log grid, solving
  the MT linearly at each with one Brune STF for the whole MT, then refines
  with a 1-D search. About 0.1 s; gives 60.1 Hz on single-Brune data with
  true fc0 = 60 Hz. Pass fc0= to plug in a value measured from the data.
- Model helpers: finite symmetric bounds on every parameter (the MT and v
  components keep their signs), per-parameter scales, the prior on v as a
  damping term, multi-start perturbations of v only, and a warning when
  |v| > v_max.
- unshake.nonlinear_least_squares(model, ds): a bounded trust-region solve
  with several starts. With no noise level given, it estimates sigma from the
  misfit and refits once. Returns (p, cov, info), with a Gauss-Newton
  covariance.
- The MT components are not log-transformed: they are signed, so there is no
  log to take; scaling each parameter to O(1) solves the conditioning problem.
- Monte Carlo fixes. Both came from fitting the MT on the GFs alone, which
  ignores the source time function:
  - the prior amplitude was 0.164 against a true MT norm of 1.12 (now 1.07);
  - the default noise level was about 3.5x too large.
  Both now come from the starting model.

Results on the demo data (examples/directional_inversion_demo.ipynb)

| | fc0 (Hz) | |v| / vs | azimuth (deg) | VR | cost |
|---|---|---|---|---|---|
| true | 60.0 | 0.600 | 45.0 | 0.927 | - |
| Starting model (v = 0) | 63.1 | 0 | - | 0.876 | 0.1 s |
| MC, best of 50k samples | 70.7 | 0.43 | 235 | 0.822 | 42 s, effective sample size ~1 |
| NLLS | 59.97 +/- 0.16 | 0.586 | 44.3 | 0.927 | 32 forward evaluations, 0.4 s |

Plain prior-sampling MC is no use in 10 dimensions. On noise-free data,
NLLS recovers every parameter to about 1e-13.

Things to know
- The uncertainties are too small. The covariance assumes white noise, but
  the noise is band-limited, so the error bars are about 1.5x too tight; some
  parameters sit 2-2.8 sigma from the truth. Documented, not corrected.
- v_u is not constrained in the demo: all receivers lie in the horizontal
  plane of the source, so v_u = 0 +/- 900 m/s, the prior.
- Pure-crack check: the fit puts a spurious DC part of about 0.6 % into the
  MT, which makes the data slightly sensitive to v. Result
  v = (-677, 834) +/- (433, 463) m/s: not significant, but not zero. For
  sources with little DC, report v alongside the DC fraction.

Next step (not built): an adaptive Metropolis MCMC (pure numpy) started from
the NLLS result. HMC/NUTS (NumPyro or PyMC) is flagged for later: it needs a
differentiable forward model, and the gradient through the MT decomposition
is badly behaved. numba would not help, as the time is already spent in
compiled matrix and FFT routines.

Files: src/unshake/{models,solvers,__init__}.py, README.md,
examples/directional_inversion_demo.ipynb (new),
plans/2026-11-03-mtDCDirectivity-nonlinear-inversion.md (new)
```
