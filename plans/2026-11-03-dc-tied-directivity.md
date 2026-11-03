# Plan: DC-tied directivity (option B), and notes on the non-linear inversion

**Status:** forward model implemented 2026-11-03 (see Outcome). The non-linear inversion is not implemented yet.
**Follows:** `plans/2026-11-03-directional-stf.md`, which implemented the shared-fc0 two-Brune model with a free weight w2.

## Summary: the forward problem as built (2026-11-03)

**1. Green's functions (path only).**
- Salvus applies its wavelet $s(t)$ as the moment $M(t)$. `Dataset.remove_stf(wavelet.get_stf_rate(), band)` divides out the wavelet's moment rate and applies the band filter $B$ = `spectra.flat_band(f, *band)`.
- $G_{ik}(t)$ is then the response of trace $i$ to a unit-impulse moment rate of MT component $k$, times $B$.
- The data must be band-limited by the same $B$ (`forward.bandlimit`; `Dataset.check_band` reminds and warns).
- Conventions:
  - MT components are NED, $[nn, ee, dd, ne, nd, ed]$;
  - coordinates, $\hat r_i$ (`Dataset.ray_dirs`) and $\mathbf{v}$ are ENU;
  - spectra use the np.fft sign, $X(f) = \int x\,e^{-2\pi i f t}\,dt$.

**2. Split of the moment tensor** (`mt.decompose`, Vavryčuk 2015).
- Eigenvalues are $\lambda_1 \ge \lambda_2 \ge \lambda_3$, with T and P axes $\mathbf{e}_1$, $\mathbf{e}_3$:

$$M_{DC} = \tfrac12\big(\lambda_1-\lambda_3-|\lambda_1+\lambda_3-2\lambda_2|\big), \qquad
\mathbf{m}^{DC} = M_{DC}(\mathbf{e}_1\mathbf{e}_1^\mathsf{T}-\mathbf{e}_3\mathbf{e}_3^\mathsf{T}), \qquad
\mathbf{m}^{nDC} = \mathbf{m}-\mathbf{m}^{DC}\ (\text{ISO + CLVD})$$

- The fractions are $C_X = M_X/(|M_{ISO}|+|M_{CLVD}|+M_{DC})$, with $M_{ISO} = \tfrac13\sum\lambda$ and $M_{CLVD} = \tfrac23(\lambda_1+\lambda_3-2\lambda_2)$.

**3. Source time functions** (`stfs`). The causal Brune moment rate with unit moment:

$$\dot M(t) = \omega_c^2\, t\, e^{-\omega_c t} H(t), \qquad S(f; f_c) = \frac{1}{(1+i f/f_c)^2}, \qquad |S| = \texttt{spectra.brune}$$

**4. Directivity** (`spectra.apparent_fc_vector`). Unilateral Doppler, with rupture velocity vector $\mathbf{v} = (v_E, v_N, v_U)$:

$$f_{c2,i} = \frac{f_{c0}}{1 - \mathbf{v}\cdot\hat r_i / c_i}$$

- $c_i$ = `dataset.receivers.velocity`.
- $v_r = |\mathbf{v}|$, and azimuth and plunge come from `spectra.rupture_from_vector`.

**5. Forward model** (`models.MTDCDirectivityModel.forward`, `forward.convolve_stf`):

$$u_i(t) = \mathcal{F}^{-1}\Big[S(f; f_{c2,i})\,\mathcal{F}\big[\textstyle\sum_k m^{DC}_k G_{ik}\big] + S(f; f_{c0})\,\mathcal{F}\big[\textstyle\sum_k m^{nDC}_k G_{ik}\big]\Big]$$

- The FFTs are zero-padded to at least $2n_t$ (linear, not circular, convolution) and cut back to $n_t$.
- Parameters (10): $[M_{nn}, M_{ee}, M_{dd}, M_{ne}, M_{nd}, M_{ed}, f_{c0}, v_E, v_N, v_U]$.
- Prior (`sample_prior`):
  - MT from `MTModel(kind)`;
  - $\log f_{c0}$ uniform over `fc0_range`;
  - $\mathbf{v}$ isotropic zero-mean Gaussian ($\sigma_v$ = 900 m/s), truncated at $|\mathbf{v}| \le v_{max}$ = 1600 m/s.

**6. Limits and properties (all verified).**
- $\mathbf{v} = 0$ gives a single Brune at $f_{c0}$ for the whole MT.
- A pure DC is fully directional, identical to `MTTwoBruneModel` with $w_2 = 1$.
- ISO, CLVD and tensile sources have no sensitivity to $\mathbf{v}$.
- $u$ is linear in the MT amplitude ($\mathbf{m}^{DC}$ is homogeneous of degree 1). It is non-linear in the MT shape and orientation, $f_{c0}$ and $\mathbf{v}$.

**7. Assumptions and limitations.**
- The STFs are point-source Brune ($\omega^{-2}$) STFs. Directivity shifts only the corner frequency; there is no Haskell sinc term and no change in amplitude with direction.
- The DC part and the non-DC part share $f_{c0}$, have the same origin time and differ only through directivity.
- One phase velocity $c_i$ per trace ($v_p$ in the demo), although the window holds P and S.
- Straight rays from a point source; homogeneous medium in the demo.
- The DC/CLVD split is the Vavryčuk convention. $\mathbf{m}^{DC}$ has a kink at $\gamma = 0$ (the DC meridian).

**Older models kept for comparison:**
- `MTTwoBruneModel`: shared fc0, free w2, rupture as (vr, azimuth, plunge);
- `MTSpectrumModel`: zero-phase stub.

**Demo:** `examples/directional_stf_demo.ipynb` (this model, 60 % DC source). `examples/directional_stf_demo_v1.ipynb` is the free-w2 version.

## Context
- **The problem.** In the shared-fc0 two-Brune model, the directional moment fraction w2 and the rupture velocity vr trade off. To first order the spectra constrain only w2·vr/c. Notebook section 5.6 shows a flat VR ridge along w2·vr = const: the true model (w2 = 0.6, vr = 0.6 vs) has VR 0.9405, against 0.9398 for w2 = 1, vr = 0.36 vs.
- **The fix.** Remove w2 and tie directivity to the moment tensor itself. The double-couple (shear) part ruptures with a Doppler-shifted STF. The non-DC part (ISO + CLVD) radiates with the symmetric STF.

**Decisions (user, 2026-11-03):**
1. **Option B:** split the MT into DC and non-DC parts, each convolved with its own STF. Option A, a scalar weight w2 = C_DC applied to the whole wavefield, is not used.
2. **Tensile sources can have no directivity.** A pure tensile crack, CLVD or ISO source has C_DC = 0, so vr is unconstrained and in effect zero.
3. **No fault-plane constraint** on the rupture direction for now. The rupture direction is free in 3-D.
4. **The rupture is parameterised as a velocity vector** $\mathbf{v} = v_r\hat d = (v_E, v_N, v_U)$ (ENU, m/s), replacing (vr, rupt_az, rupt_plunge). See "Rupture as a velocity vector" below.

## Equations

### DC / non-DC split of the moment tensor
Uses Vavryčuk (2015), *Moment tensor decompositions revisited*, J. Seismol., doi:10.1007/s10950-014-9463-y.

Eigen-decomposition $\mathbf{M} = \sum_k \lambda_k \mathbf{e}_k\mathbf{e}_k^\mathsf{T}$, with $\lambda_1 \ge \lambda_2 \ge \lambda_3$. Here $\mathbf{e}_1$ = T axis, $\mathbf{e}_2$ = B (null) axis and $\mathbf{e}_3$ = P axis, in the order `mt.eig` returns them.

$$M_{ISO} = \tfrac13(\lambda_1+\lambda_2+\lambda_3), \qquad
M_{CLVD} = \tfrac23(\lambda_1+\lambda_3-2\lambda_2), \qquad
M_{DC} = \tfrac12\big(\lambda_1-\lambda_3-|\lambda_1+\lambda_3-2\lambda_2|\big)$$

$$M = |M_{ISO}| + |M_{CLVD}| + M_{DC}, \qquad C_X = M_X / M, \qquad C_{DC} \in [0, 1]$$

$$\mathbf{M}^{DC} = M_{DC}\,(\mathbf{e}_1\mathbf{e}_1^\mathsf{T} - \mathbf{e}_3\mathbf{e}_3^\mathsf{T}), \qquad
\mathbf{M}^{nDC} = \mathbf{M} - \mathbf{M}^{DC} \quad (\text{pure ISO + CLVD})$$

**Checked numerically:**
- C_DC is 1 for a DC, and 0 for CLVD (2, −1, −1), ISO (1, 1, 1) and a tensile crack (3, 1, 1).
- For (1, 0.2, −1.2), C_DC = 0.667, which agrees with $1 - 2|\varepsilon|$ for deviatoric sources.
- $\mathbf{M}-\mathbf{M}^{DC}$ is always pure ISO + CLVD.
- **Homogeneity:** $\mathbf{m}^{DC}(a\mathbf{m}) = a\,\mathbf{m}^{DC}(\mathbf{m})$ for every real $a$, including negative $a$.

**Properties:**
- C_DC depends only on eigenvalue ratios, so it is a function of lune position $(\gamma, \delta)$ alone.
- $\mathbf{M}^{DC}$ is continuous everywhere. It is 0 wherever the eigenvectors are degenerate (ISO, CLVD, crack), so arbitrary eigenvectors there don't matter.
- It is **not differentiable** at $\lambda_1+\lambda_3 = 2\lambda_2$, i.e. $\gamma = 0$, the DC meridian, because of the $|\cdot|$.

### Forward model (option B)

$$u_i = \mathcal{F}^{-1}\Big[\,S\big(f; f_{c2}(\hat r_i)\big)\,\hat U^{DC}_i(f) \;+\; S(f; f_{c0})\,\hat U^{nDC}_i(f)\Big]$$

$$\hat U^{DC}_i = \mathcal{F}\big[\mathbf{G}_i\cdot\mathbf{m}^{DC}\big], \quad \hat U^{nDC}_i = \mathcal{F}\big[\mathbf{G}_i\cdot\mathbf{m}^{nDC}\big], \quad
S(f; f_c) = \frac{1}{(1+i f/f_c)^2}, \quad f_{c2}(\hat r_i) = \frac{f_{c0}}{1 - \mathbf{v}\cdot\hat r_i / c_i}$$

- $\mathbf{v} = (v_E, v_N, v_U)$ is the rupture velocity vector in ENU (m/s). $c_i$ is the phase velocity for trace $i$ (`dataset.receivers.velocity`), and $\hat r_i$ the straight-ray take-off vector (`Dataset.ray_dirs`).
- This is the same unilateral Doppler form as `spectra.apparent_fc`, with $v_r\cos\theta_i = \mathbf{v}\cdot\hat r_i$, so $v_r = |\mathbf{v}|$ and $\hat d = \mathbf{v}/|\mathbf{v}|$. Written with $\mathbf{v}\cdot\hat r_i$, it needs no direction when $\mathbf{v} = 0$.
- For reporting: $v_r = |\mathbf{v}|$, rupture azimuth $\phi_r = \operatorname{atan2}(v_E, v_N)$ (clockwise from north), plunge $p = \arcsin(-v_U/|\mathbf{v}|)$ (positive down), matching `spectra.rupture_direction`.
- **Validity:** physical only for $|\mathbf{v}| < c_i$. Bound the parameter by $|\mathbf{v}| \le v_{max}$ (e.g. about $0.9\,v_s$ for sub-shear ruptures).
- Equivalent form: $u_i = S(f_{c0})\otimes \mathbf{G}_i\cdot\mathbf{m} + \big[S(f_{c2,i}) - S(f_{c0})\big]\otimes \mathbf{G}_i\cdot\mathbf{m}^{DC}$. The directivity is therefore a perturbation that acts only on the DC radiation.
- **Limits:** pure DC → whole wavefield directional; C_DC = 0 → single symmetric Brune at fc0, with no sensitivity to vr or $\hat d$.
- **Parameters:** 6 MT + fc0 + $(v_E, v_N, v_U)$ = **10** (w2 removed).

### First-order (small vr/c) sensitivity
$f_{c2,i} \approx f_{c0}\,(1 + \mathbf{v}\cdot\hat r_i/c)$, with rupture velocity vector $\mathbf{v} = v_r\hat d$. Then:

$$u_i \approx \mathcal{F}^{-1}\big[S(f; f_{c0})\,\hat U_i\big] + \sum_{j=1}^{3} v_j\,\frac{\hat r_{ij}}{c}\,\mathcal{F}^{-1}\big[K(f)\,\hat U^{DC}_i\big], \qquad
K(f) = f_{c0}\frac{\partial S}{\partial f_{c0}} = \frac{2 i x}{(1 + i x)^3},\ x = f/f_{c0}$$

The formula for K(f) was checked against finite differences.

- The spectra constrain $\mathbf{v}$ scaled by the amount of DC radiation at each receiver. The DC amount is fixed by the radiation pattern of the whole waveform, so vr is resolvable.
- **Errors in the non-DC part propagate into vr.** In the scalar analogue, $\delta v_r/v_r \approx -\delta C_{DC}/C_{DC}$. Spurious CLVD from GF errors or poor coverage biases vr.
- For low-C_DC events, vr and $\hat d$ are poorly constrained. That is physically expected (decision 2).

## Notes on solving the non-linear inversion

### What is non-linear
- **The MT amplitude stays linear.** $\mathbf{m}^{DC}$ is homogeneous of degree 1, so $u(a\mathbf{m}) = a\,u(\mathbf{m})$. The amplitude can be solved or profiled analytically, as `solvers.lsq_loglike` and `MTModel` already do.
- **Non-linear parameters (9):**
  - MT orientation (3);
  - lune position (γ, δ), which sets C_DC;
  - fc0;
  - the rupture velocity vector $\mathbf{v}$ (3).
- **The STF parameters are cheap to evaluate.** For a fixed MT, precompute $\hat U^{DC}_i$ and $\hat U^{nDC}_i$ once. Each (fc0, $\mathbf{v}$) evaluation is then two spectrum multiplies and one irfft.
- **The MT is expensive and kinked.** It enters through a 3×3 eigen-decomposition per sample (batched `np.linalg.eigh` is cheap), with the kink at γ = 0.

### Rupture as a velocity vector (decision 4)
- Use $\mathbf{v} = v_r\hat d$ (3 components, |v| ≤ v_max) instead of (vr, φr, p).
- This removes the coordinate singularity at vr = 0, where the direction is undefined. The posterior is then smooth and near-Gaussian around $\mathbf{v} = 0$. Tensile events just collapse to $\mathbf{v} \approx 0$.
- To first order the forward model is linear in $\mathbf{v}$ (see above), so Gauss–Newton is well behaved.
- **The prior must be chosen deliberately** (`sample_prior` and the solvers must use the same one).
  - A uniform prior on the ball $|\mathbf{v}| \le v_{max}$ is **not** uniform in vr. It implies $p(v_r) \propto v_r^2$, which favours fast ruptures where the data are weak.
  - Uniform vr with an isotropic direction is $p(\mathbf{v}) \propto 1/|\mathbf{v}|^2$. This is integrable but singular at $\mathbf{v} = 0$, which is awkward for MCMC and Laplace proposals.
  - **Recommended default (to confirm when implementing):** an isotropic zero-mean Gaussian on $\mathbf{v}$, $\sigma_v \approx 0.5\,v_s$, truncated at $|\mathbf{v}| \le v_{max}$. It is smooth, it matches the damping in the linearised step, and its mean is $\mathbf{v} = 0$. That is the "no directivity" answer when the data have no sensitivity, e.g. tensile sources (decision 2).
  - **Reporting.** Any smooth prior on $\mathbf{v}$ gives $p(v_r) \to 0$ as $v_r \to 0$ (the 3-D shell volume), so report the vector $\mathbf{v}$ and its covariance. Interpret marginals of $|\mathbf{v}|$ with that in mind.
- **Implementation.** Compute $f_{c2,i} = f_{c0}/(1 - \mathbf{v}\cdot\hat r_i/c_i)$ directly, e.g. a small `spectra.apparent_fc_vector(fc0, v, c, ray_dirs)` helper. This avoids normalising $\mathbf{v}$ (no guard at $|\mathbf{v}| = 0$). `apparent_fc` is kept for the existing models.

### Recommended pipeline
1. **Starting MT and fc0.** Run the existing linear `least_squares` for the MT with a single Brune at trial fc0 values. This is variable projection: a 1-D scan or `scipy.optimize.minimize_scalar` over log fc0, with the MT solved linearly at each fc0. The result is $\mathbf{m}_0$, $f_{c0}$.
2. **Linearised directivity.** From $\mathbf{m}_0$, form $\mathbf{m}^{DC}_0$ and solve the 3-column linear least squares $r_i \approx \sum_j v_j\,a_{ij}(t)$ for $\mathbf{v}$, where $a_{ij} = (\hat r_{ij}/c)\,\mathcal{F}^{-1}[K\,\hat U^{DC}_i]$. Add a little damping, which acts as a zero-mean Gaussian prior on $\mathbf{v}$. This gives a starting $\mathbf{v}_0$.
3. **Full non-linear refinement.**
   - Use `scipy.optimize.least_squares` (trust region / Levenberg–Marquardt) over the 10 parameters (m6, log fc0, $\mathbf{v}$), with weighted residuals and bounds $|v_j| \le v_{max}$. The ball constraint $|\mathbf{v}| \le v_{max}$ can be enforced by a penalty or checked afterwards.
   - Use a numerical Jacobian (about 11 forward evaluations per iteration, each ≲ 1 ms for 150 traces).
   - Multi-start: from $\mathbf{m}_0$, from small perturbations on the lune either side of γ = 0 (the kink), and from $\pm\mathbf{v}_0$.
   - Optionally replace $|x|$ in $M_{DC}$ with $\sqrt{x^2+\epsilon^2}$ for the optimiser only.
4. **Uncertainty.**
   - **Fast, local:** Gauss–Newton covariance $(J^\mathsf{T}WJ)^{-1}$ at the optimum, as for `least_squares` now.
   - **Full posterior, either of:**
     - **(a) Importance sampling.** Use a Gaussian (Laplace) proposal centred on the optimum, widened by about 1.5–2×. Weights are likelihood × prior / proposal, and they drop straight into the existing `Result` (samples + weights). Check the ESS.
     - **(b) MCMC** (Metropolis in (m6, log fc0, $\mathbf{v}$), or an affine-invariant ensemble sampler), initialised at the optimum. This needs an optional dependency, e.g. `emcee`; core unshake stays numpy/scipy/matplotlib.
   - **Plain prior-sampling `monte_carlo` won't work.** It is too inefficient in 10-D, with a very low ESS. Keep it only for low-dimensional slices, e.g. STF parameters with the MT fixed.
5. **Optional: block-coordinate alternation.** Alternate between the MT (non-linear least squares in m6, STF fixed) and the STF parameters (4-D, cheap with precomputed $\hat U$). This is useful for diagnosing the trade-offs, but slower to converge than the joint solve where the MT and STF parameters are correlated.

### A relaxed linear diagnostic (cheap, no eigen-decomposition)
- For fixed STF parameters, the model $u_i = \mathcal{F}^{-1}[S(f_{c2,i})\,\mathcal{F}[\mathbf{G}_i\cdot\mathbf{m}^A] + S(f_{c0})\,\mathcal{F}[\mathbf{G}_i\cdot\mathbf{m}^B]]$ is **linear** in two unconstrained 6-vectors $\mathbf{m}^A$ and $\mathbf{m}^B$ (a 12-column design matrix).
- Fitting it and then checking whether $\mathbf{m}^A$ is close to the DC part of $\mathbf{m}^A + \mathbf{m}^B$ tests the option-B assumption on real data.
- It is ill-conditioned as $\mathbf{v} \to 0$, because the two STFs coincide, so it needs damping.

### Solver details to watch
- **Kink at γ = 0.** It sits exactly on the DC meridian, where many icequakes will be. Gradient methods may stall there, so use the multi-starts or the smoothed |·|. Sampling methods are unaffected.
- **Degenerate eigenvalues** (ISO, CLVD, crack): $\mathbf{m}^{DC} = 0$, so the forward model is continuous. But $\mathbf{v}$ has zero sensitivity there, so it needs a prior or damping.
- **Multimodality:**
  - the sign of $\mathbf{v}$ is only weakly ambiguous, because the unilateral 1/(1 − x) is asymmetric;
  - local minima can come from cycle-skipping between the P and S windows.
  - Multi-start covers both.
- **Cost:** cache `rfft(gf)` (n_rec, 6, n_f) once per Dataset. Then $\hat U^{DC}$ and $\hat U^{nDC}$ are matrix products in the frequency domain, and each model needs one irfft. For batches, chunk to bound memory: 10k samples × 150 traces × 129 frequencies × 16 B ≈ 3 GB.
- **Single phase velocity.** c = vp per trace is still used, although the window holds P and S. Separate P and S windows, each with its own c, would make $\mathbf{v}$ more meaningful.

## Proposed implementation (for approval; nothing done yet)
- **`spectra.py`:** `apparent_fc_vector(fc0, v, c, ray_dirs)` gives $f_{c0}/(1 - \mathbf{v}\cdot\hat r_i/c_i)$, vectorised to (n, n_rec).
- **`mt.py`:** `decompose(m6)` returns `(c_iso, c_clvd, c_dc, m_dc, m_ndc)`, vectorised over (n, 6), following Vavryčuk (2015).
- **`models.py`:** a new `MTDCDirectivityModel(MTModel)`.
  - Parameters: 6 MT, fc0, `v_e`, `v_n`, `v_u` (rupture velocity vector, ENU, m/s).
  - `sample_prior`: the truncated isotropic Gaussian on $\mathbf{v}$ above (σ_v and v_max are constructor arguments), log-uniform fc0, and the MT from `MTModel`.
  - A helper converts $\mathbf{v}$ to (vr, azimuth, plunge) for reporting and plots.
  - `forward`: two `convolve_stf` calls, or one combined frequency-domain pass, as in the forward-model equation above.
  - `corner_frequencies`: unchanged in meaning.
  - Keep `MTTwoBruneModel` (free w2) for comparison.
- **`solvers.py`, later:** a `nonlinear_least_squares(model, dataset, p0, ...)` wrapper around `scipy.optimize.least_squares`, returning the optimum and the Gauss–Newton covariance. Add an importance sampler with a Laplace proposal that returns a `Result`.
- **Notebook (new section 6):**
  - a true MT with C_DC ≈ 0.6 (a lune point off the DC);
  - the DC and non-DC partial synthetics, and per-receiver effective directivity;
  - a repeat of the 5.6 trade-off scan, now over horizontal $(v_E, v_N)$. It should show a closed minimum around the true vector, instead of a ridge;
  - an interactive cell with sliders for fc0, $v_E$, $v_N$ and $v_U$ (or vr and azimuth, converted to $\mathbf{v}$), printing true and model parameters.

## Summary of the notes (2026-11-03)
1. **Only 9 parameters are non-linear.** $\mathbf{m}^{DC}$ is homogeneous of degree 1, so the MT amplitude is solved analytically. The 9 are MT orientation (3), lune (2), fc0 and $\mathbf{v}$ (3). For a fixed MT, an STF evaluation is cheap: two spectrum multiplies and one irfft, using the precomputed $\hat U^{DC}$ and $\hat U^{nDC}$.
2. **Velocity vector $\mathbf{v}$ (decision 4).** It removes the singularity at vr = 0, tensile events collapse smoothly to $\mathbf{v} \approx 0$, and the model is nearly linear in $\mathbf{v}$, which suits Gauss–Newton. The prior on $\mathbf{v}$ must be chosen deliberately: uniform on the ball implies $p(v_r) \propto v_r^2$. The recommended default is a truncated zero-mean Gaussian.
3. **Pipeline:**
   - linear MT least squares with an fc0 scan (variable projection) for the starting MT and fc0;
   - a 3-parameter linearised least squares for $\mathbf{v}_0$;
   - multi-start `scipy.optimize.least_squares` over all 10 parameters, including starts either side of γ = 0;
   - Gauss–Newton covariance;
   - importance sampling with a Laplace proposal (fits `Result`) or MCMC for the full posterior.
   Plain prior-sampling Monte Carlo is too inefficient in 10-D.
4. **Testing option B on data:** fit two unconstrained MTs, one per STF. This is linear with 12 columns, needs damping as $\mathbf{v} \to 0$, and checks whether the directional MT is close to the DC part of the total.
5. **Pitfalls:**
   - the kink in $\mathbf{m}^{DC}$ at γ = 0 (the DC meridian), so use multi-start or a smoothed |·| in the optimiser;
   - zero sensitivity to $\mathbf{v}$ for ISO, CLVD and tensile sources, so it needs a prior or damping;
   - multimodality (a weak sign ambiguity in $\mathbf{v}$, P/S cycle-skipping);
   - batch memory, so cache `rfft(gf)` and chunk;
   - the single phase velocity c per trace.


## Outcome (implemented 2026-11-03)

### Code
- **`mt.decompose(m6)`:** returns `(c_iso, c_clvd, c_dc, m_dc, m_ndc)`, vectorised (Vavryčuk 2015).
- **`spectra.apparent_fc_vector(fc0, v, c, ray_dirs)`** and **`spectra.rupture_from_vector(v)`**, the inverse of `rupture_direction`.
- **`models.MTDCDirectivityModel`:**
  - parameters `[Mnn, Mee, Mdd, Mne, Mnd, Med, fc0, v_e, v_n, v_u]`;
  - the prior on **v** is a truncated isotropic Gaussian (defaults `sigma_v=900`, `v_max=1600` m/s);
  - `rupture(params)` returns (vr, azimuth, plunge) in degrees.
  - It is exported from `unshake`. `MTTwoBruneModel` (free w2) is kept.

### Checks
- `decompose` gives DC/CLVD/ISO/crack = 1/0/0/0 DC.
- (1, 0.1, −0.5) → ISO 0.2, CLVD 0.2, DC 0.6.
- `decompose` is homogeneous of degree 1 and safe for a zero MT.
- In the model:
  - **v** = 0 equals a single Brune at fc0 (difference 4e-16);
  - a pure DC equals `MTTwoBruneModel` with w2 = 1 (difference 4e-16);
  - a tensile crack has zero sensitivity to **v**;
  - `apparent_fc_vector` equals `apparent_fc`.

### Demo notebook
- `examples/directional_stf_demo_v1.ipynb` is a copy of the free-w2 version.
- `examples/directional_stf_demo.ipynb` now uses `MTDCDirectivityModel` throughout section 5.
- **Source:** eigenvalues (1, 0.1, −0.5) on the principal axes of the DC strike 30, dip 60, rake −60. That gives ISO 20 %, CLVD 20 % (opening-crack-like) and DC 60 %, and the DC part's nodal planes are 30/60/−60 and 160.9/41.4/−130.9.
- **True values:** fc0 = 60 Hz, **v** = 0.6 vs towards azimuth 45°, horizontal.
- **Results** (with the notebook band as edited by the user, spectra to about 200 Hz):
  - Single Brune: the VR scan peaks at 60.1 Hz.
  - Best single Brune on the DC-directivity data: 62.6 Hz, VR 0.875, against VR 0.927 for the true model.
  - The DC share of the 3-component RMS amplitude is about 0.65–0.9 across azimuths, more than the 60 % moment fraction, because ISO + CLVD radiate weaker S waves.
  - **Section 5.6 scan over (v_E, v_N)** (MT, fc0 and v_U fixed): a **closed maximum** at the true vector. Noise-free: best |v|/vs = 0.60 at 48° (VR 0.9997). Noisy: 0.60 at 42° (VR 0.9270, true 0.9271). Against VR 0.8733 at **v** = 0. There is no ridge, unlike the free-w2 model.

---

## Summary of changes for commit (2026-11-03)
This commit covers both `plans/2026-11-03-directional-stf.md` and this plan.

### New files
- **`src/unshake/stfs.py`:** causal source time functions with unit moment:
  - `brune_stf(t, fc)`: Brune moment rate in time;
  - `brune_stf_spectrum(f, fc)` = 1/(1 + i f/fc)²;
  - `two_brune_spectrum(f, fc1, fc2, w2)`.
- **`examples/directional_stf_demo.ipynb`:** forward demo on small full-space Salvus GFs (50 three-component receivers on a 35 m ring, flat-spectrum wavelet to 150 Hz, formal GFs). It covers:
  - single-Brune fc sweeps (spectra, waveforms, VR scan, slider);
  - the DC-tied directivity model with a 60 % DC / 20 % ISO / 20 % CLVD source;
  - a VR scan over (v_E, v_N).
- **`examples/directional_stf_demo_v1.ipynb`:** the previous version with a free directional weight w2, including its w2–vr trade-off scan.
- **`plans/2026-11-03-directional-stf.md`:** Brune / two-Brune forward demo (equations, checks, shared-fc0 revision).
- **`plans/2026-11-03-dc-tied-directivity.md`:** DC-tied directivity, the summary of the forward problem, and notes on the non-linear inversion (not implemented).

### Modified files
- **`forward.py`:** `convolve_stf(synth, dt, spec)` convolves with a complex STF spectrum in the frequency domain, zero-padded so the convolution is linear.
- **`mt.py`:**
  - `from_sdr(strike, dip, rake, m0)`: DC from fault angles (Aki & Richards);
  - `decompose(m6)`: ISO/CLVD/DC fractions and the DC / non-DC tensors (Vavryčuk 2015).
- **`spectra.py`:**
  - `apparent_fc_vector(fc0, v, c, ray_dirs)`: Doppler corner frequency from a rupture velocity vector;
  - `rupture_from_vector(v)`: speed, azimuth and plunge from **v**.
- **`models.py`:**
  - `MTTwoBruneModel`: MT + (1 − w2)·S(fc0) + w2·S(Doppler fc0), with a free w2. Kept for comparison.
  - `MTDCDirectivityModel`: MT + fc0 + **v**. The DC part gets the Doppler-shifted Brune, the non-DC part the symmetric Brune. It has a truncated-Gaussian prior on **v**.
- **`__init__.py`:** exports `stfs`, `MTTwoBruneModel` and `MTDCDirectivityModel`.
- **`README.md`:** module list (stfs, models, the mt decomposition) and the demo notebook.
- **`.gitignore`:** `examples/stf_demo_out/` (cached Salvus GFs and Dataset pickle).

### Not changed
- `apply_source_spectrum`, `MTSpectrumModel`, the solvers, the Salvus driver and the existing examples.
- `examples/synthetic_mt_recovery.py` still runs, and `import unshake` works without Salvus.

### Notes
- **Rebuilding the demo:** the notebooks rely on `examples/stf_demo_out/`, which is git-ignored. On a fresh clone, re-run the Salvus cell (about 3.5 min on 4 ranks). It writes to the `PROJECT_PATH` set at the top of the notebook; the driver deletes and recreates that directory.
- **Notebook sizes:** the notebooks are committed with outputs, about 2 MB each.

### Suggested commit message
```
Directional source time functions: causal Brune STFs and DC-tied directivity

- stfs.py: causal Brune moment rate and spectrum (1/(1 + i f/fc)^2), two-Brune sum
- forward.convolve_stf: zero-padded frequency-domain STF convolution
- mt.from_sdr; mt.decompose (Vavrycuk 2015 ISO/CLVD/DC fractions and DC tensor)
- spectra.apparent_fc_vector / rupture_from_vector (Doppler fc from a rupture
  velocity vector)
- MTTwoBruneModel (shared fc0, free directional weight w2) and
  MTDCDirectivityModel (DC part Doppler-shifted, non-DC symmetric; params
  m6, fc0, v_e, v_n, v_u)
- examples/directional_stf_demo.ipynb (+ _v1): forward demo on small
  full-space Salvus GFs, 60 % DC source
- plans: directional STFs, DC-tied directivity, non-linear inversion notes
```
