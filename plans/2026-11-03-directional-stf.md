# Plan: directional source time functions (Brune) — forward demo

## Context
The next step for unshake is inverting for source-time-function (STF) properties that can vary with direction from the source. The formal-GF work (`plans/2026-10-23-formal-gfs.md`) set up the prerequisite. `Dataset.remove_stf` divides the simulation wavelet's moment rate out of Salvus GFs, so the GFs are path-only (band-limited by B = `spectra.flat_band`).

This step adds the forward side only, with no inversion yet:
1. Convolve formal GFs with a **causal Brune** moment-rate STF, and show how the corner frequency fc changes the spectra and waveforms compared with synthetic "data" from a known MT.
2. Convolve with a **sum of two Brune STFs**:
   - STF 1 (symmetric) has an azimuth-independent corner frequency fc0.
   - STF 2 (directional) has a corner frequency fc2 perturbed by direction through the existing Doppler `spectra.apparent_fc`. **Constraint (revision of 2026-11-03):** its corner frequency perpendicular to the rupture equals fc0, so both STFs share one corner-frequency parameter. Independent corners can be added back later.
   - STF 2 is added with a weight w2 (its moment fraction) that is the same at every receiver. Only fc2 varies with direction.
3. Demonstrate both in a notebook, on GFs from a very small full-space Salvus run (pure DC source, 50 receivers in a ring at evenly spaced azimuths).

**Decisions (from the user):**
- Doppler form for the fc2 perturbation.
- Causal Brune STF.
- Full-space geometry.
- One shared corner frequency fc0 for both STFs (replaces separate fc1 and fc2_0).

## Equations implemented

> **Next revision (forward model implemented 2026-11-03):** tie directivity to the double-couple part of the MT and drop w2. See `plans/2026-11-03-dc-tied-directivity.md`. It also contains notes on solving the non-linear inversion.

### Brune source time function (`stfs.brune_stf`, `stfs.brune_stf_spectrum`)
Moment rate with unit moment, causal, where ω_c = 2π f_c:

$$\dot M(t) = \omega_c^2\, t\, e^{-\omega_c t}\, H(t), \qquad \int \dot M\, dt = 1$$

Its spectrum, in the np.fft sign convention $X(f) = \int x(t)\, e^{-2\pi i f t}\, dt$:

$$S(f; f_c) = \frac{1}{(1 + i f / f_c)^2}, \qquad |S(f; f_c)| = \frac{1}{1 + (f/f_c)^2} = \texttt{spectra.brune}(f, f_c)$$

### Sum of two Brunes (`stfs.two_brune_spectrum`, `models.MTTwoBruneModel`)
For receiver $i$:

$$S_i(f) = (1 - w_2)\, S(f; f_{c0}) + w_2\, S\big(f; f_{c2}(\hat r_i)\big), \qquad f_{c2}(\hat r_i)\big|_{\theta_i = 90°} = f_{c0}$$

- $f_{c0}$ is the corner frequency of the symmetric STF, the same at every receiver. It is **constrained** to equal the directional STF's corner frequency perpendicular to the rupture.
- $w_2 \in [0, 1]$ is the moment fraction of STF 2, also the same at every receiver. Both terms have unit moment, so the total moment is unchanged.
- Only $f_{c2}$ depends on direction.
- Model parameters: 6 MT components plus $(f_{c0}, w_2, v_r, \phi_r, p)$, 11 in total, one fewer than with separate $f_{c1}$ and $f_{c2,0}$.
- `stfs.two_brune_spectrum(f, fc1, fc2, w2)` stays general; `MTTwoBruneModel` passes $f_{c0}$ as `fc1`.

### Directional (Doppler) corner frequency (`spectra.apparent_fc`)
Unilateral, Haskell-type: `mode='unilateral'`, the default and the one used in the demo.

$$f_{c2}(\hat r_i) = \frac{f_{c0}}{1 - \dfrac{v_r}{c}\cos\theta_i}, \qquad \cos\theta_i = \hat r_i \cdot \hat d$$

Bilateral, symmetric about the rupture direction: `mode='bilateral'`, not used in the demo.

$$f_{c2}(\hat r_i) = \frac{f_{c0}}{1 + \dfrac{v_r}{c}\,|\cos\theta_i|}$$

(In bilateral mode $f_{c0}$ is still the corner frequency at $\theta = 90°$.)

Symbols:
- $f_{c0}$: corner frequency of the symmetric STF, and of the directional STF seen perpendicular to the rupture ($\theta = 90°$).
- $v_r$: rupture velocity.
- $c$: phase velocity at the source, from `dataset.receivers.velocity`.
- $\hat r_i$: unit source-to-receiver vector along a straight ray, in ENU (`Dataset.ray_dirs`).
- $\hat d$: unit rupture direction, from `spectra.rupture_direction`. Rupture azimuth $\phi_r$ is clockwise from north; plunge $p$ is positive downwards.

$$\hat d = (\cos p \sin\phi_r,\; \cos p \cos\phi_r,\; -\sin p) \quad \text{(ENU)}$$

### Forward model (`MTTwoBruneModel.forward`, `forward.convolve_stf`)

$$u_i(t) = \mathcal{F}^{-1}\Big[\, S_i(f)\; \mathcal{F}\big[\textstyle\sum_k m_k\, G_{ik}(t)\big] \Big]$$

- $G_{ik}$ are the formal band-limited GFs: the response to a unit-impulse moment rate, multiplied by B.
- $m_k$ are the NED MT components.
- The FFTs are zero-padded to at least $2 n_t$, so the convolution is linear, and the result is cut back to $n_t$ samples.

### Trade-off between $w_2$ and $v_r$
With a shared $f_{c0}$, the two STFs differ only through the Doppler shift. For small $v_r/c$, $f_{c2} \approx f_{c0}\,(1 + (v_r/c)\cos\theta_i)$, so

$$S_i(f) \approx S(f; f_{c0}) + w_2\,\frac{v_r}{c}\,\cos\theta_i\; f_{c0}\,\frac{\partial S}{\partial f_{c0}}$$

- To first order only the product $w_2\, v_r / c$ is constrained. Higher-order terms, from the forward/backward asymmetry of $1/(1 - x)$, separate $w_2$ and $v_r$ only weakly.
- $w_2 = 0$ or $v_r = 0$ both reduce to a single Brune with corner $f_{c0}$. At those values the rupture direction (and $v_r$ or $w_2$) is undetermined.

### Demo values and limitations
**Demo values:**
- $f_{c0} = 60$ Hz, $w_2 = 0.6$, $v_r = 0.6\, v_s = 1092$ m/s and $c = v_p = 3540$ m/s, so $v_r / c = 0.308$.
- The rupture is horizontal towards 45°, and the receiver ring lies in the source plane, so $\cos\theta_i = \cos(\phi_i - 45°)$.
- $f_{c2}$ therefore ranges from about 46 Hz (225°, away from the rupture) to about 87 Hz (45°, towards it). The symmetric STF has 60 Hz everywhere.

**Limitations:**
- Only the corner frequency is shifted. The full Haskell finite-fault spectrum (sinc-type directivity term, amplitude changes) is not used, and $w_2$ does not vary with direction.
- One $c$ is used per trace, $v_p$ in the demo, although the window contains both P and S. With $c = v_s$ the Doppler factor would be larger ($v_r/c = 0.6$).

## Code changes in unshake (plain functions plus one model class, matching the existing style)

### New `src/unshake/stfs.py` (all new source time functions live here)
- The module docstring says:
  - these are moment-rate STFs with unit moment, causal, in the time and frequency domains;
  - they are convolved with formal GFs through `forward.convolve_stf`;
  - `spectra.py` keeps the existing zero-phase amplitude spectra and the Doppler `apparent_fc`.
- Import it in `__init__.py` (`from . import stfs`).
- `brune_stf(t, fc)`: Brune moment rate Ṁ(t) = (2π fc)² t e^(−2π fc t) H(t), with unit area (unit moment). Used for plots.
- `brune_stf_spectrum(f, fc)`: its complex spectrum, 1 / (1 + i f/fc)², in the np.fft sign convention (e^(−iωt)). Notes:
  - Its modulus equals the existing `brune(f, fc)`. The docstring says so and points out that `source_spectrum` / `brune` are the zero-phase amplitude versions.
  - `fc` broadcasts, so `fc[..., None]` against `f` gives (..., n_rec, n_f).
- `two_brune_spectrum(f, fc1, fc2, w2)`: (1 − w2)·S(f; fc1) + w2·S(f; fc2).
  - Both terms are unit moment, so w2 is STF 2's moment fraction and the total moment is unchanged.
  - fc1 and w2 are per sample and the same at every receiver. Only fc2 broadcasts per receiver.

### `src/unshake/forward.py`
- `convolve_stf(synth, dt, spec)`: multiply by the complex `spec(f)` using the existing zero-padded `_filter`, which gives linear rather than circular convolution, truncated to n_t.
  - `spec` is a callable f → array broadcastable to (..., n_rec, n_f).
  - Needs formal GFs, as the docstring says (same note as `apply_source_spectrum`).
- Leave `apply_source_spectrum` (zero-phase, used by the `MTSpectrumModel` stub) unchanged.
- **Time domain or frequency domain.**
  - GFs, data and synthetics stay in the time domain. The time domain is where the Dataset windowing, shifts, plotting and misfits all happen.
  - `convolve_stf` does the convolution in the frequency domain internally: rfft, multiply by S(f), irfft. This is mathematically identical to time-domain convolution but costs O(n log n), and the STF is defined analytically in frequency anyway.
  - Zero-padding to at least 2·n_t makes it linear, not circular, so causal STF energy can't wrap around to early times.
  - The output is truncated back to n_t. Any tail past the window end is lost, as for real data.
  - Linearity also lets the STF be applied after summing the MT components: (Σ mᵢ Gᵢ) ∗ S, one FFT per trace.
  - Future inversion speed-up, not done now: cache `rfft(gf)` once per Dataset. Each sample then needs only a multiply and one irfft.

### `src/unshake/mt.py`
- `from_sdr(strike, dip, rake, m0=1)`: DC moment tensor in NED (Aki & Richards box 4.4), returned as a 6-vector `[nn, ee, dd, ne, nd, ed]`. The notebook uses it to set the known DC. Check it by round-tripping through the existing `mt.sdr`.

### `src/unshake/models.py`
Add `MTTwoBruneModel(MTModel)`, so a future inversion can plug straight into `monte_carlo`. Its interface matches the others:
- `param_names = MT_NAMES + ['fc0', 'w2', 'vr', 'rupt_az', 'rupt_plunge']` (revised from separate `fc1`, `fc2_0`).
- `sample_prior`:
  - fc ranges are log-uniform;
  - w2 is uniform on [0, 1];
  - vr, rupture azimuth and plunge are drawn as in `MTSpectrumModel.sample_prior`.
- `corner_frequencies(params, ds)`: returns fc2 per receiver, `(n, n_rec)`, from `spectra.apparent_fc(fc0, vr, ds.receivers.velocity, ds.ray_dirs, rupture_direction(az, plunge), mode)`. The same method name means `plot.fc_azimuth` works unchanged.
- `forward`: `convolve_stf(synthetics(gf, m), dt, lambda f: stfs.two_brune_spectrum(f, fc0[:, None, None], fc2[..., None], w2[:, None, None]))`.
- Special cases:
  - w2 = 0 reduces to a single azimuth-independent Brune;
  - vr = 0 gives two non-directional Brunes.
- Add it to the `__init__.py` exports.

### Docs
- README module list: add one line about directional / two-Brune STFs.
- Save this plan as `plans/2026-11-03-directional-stf.md`.

## Notebook: `examples/directional_stf_demo.ipynb`
Kernel: `/Users/tomhudson/miniforge3/envs/salvus_src_inv` (Salvus 2024.1.1, ipywidgets 8.1.5).

### 1. Small Salvus run, cached
Uses `unshake.salvus_fwd_driver.gen_synths_salvus` with:
- **Full space:** `gen_body_waves_only=True`.
- **Medium:** homogeneous ice (vp 3540, vs 1820, ρ 917), Q high (e.g. Qμ = 1000, Qκ = 2Qμ) so attenuation doesn't hide the STF effect. Q is a notebook parameter.
- **Box:** about ±50 m in x, y and z. Medium grid ±200 m.
- **Source:** at (0, 0, −10) m.
- **Receivers:** 50 three-component velocity receivers at 35 m radius, at azimuths 0, 7.2, …, 352.8°, named `R00`–`R49`. The name must not start with "D", or the driver applies the DAS projection.
- **Driver quirks to respect:**
  - coordinates are passed in metres / 1e6, because the driver multiplies by 1e6;
  - in full space it moves receivers down by 10 m, so the ring sits in the source's horizontal plane at z = −10 m.
- **Wavelet:** `FlatSpectrumWavelet(5, 10, 140, 150, time_shift_in_seconds=0.04)`, with `max_frequency_in_hertz=150`, `elements_per_wavelength=2`, `end_time_in_seconds=0.12`, and `centre_freq_Hz=100` (now only the absorbing-boundary reference frequency).
- **Output:** fs = 1000 Hz.
- **Paths:**
  - dedicated project path, a variable such as `/Volumes/4tb_extssd/salvus_tmp_workspace/unshake_stf_demo` (the driver deletes it if it exists);
  - outputs to `examples/stf_demo_out/` (add to `.gitignore`).
- **Cache guard:** `run_fwd = False` reuses the files on disk.
- **Expected cost:** roughly 5–15 min on 4 ranks. The earlier ±220 m / 40 Hz full-space test took 2.5 min.

### 2. Build the formal GF Dataset
Same recipe as notebook section 3.y.2, without the data-specific shifts:
- **Load the GFs:** read `*_{HHE,HHN,HHZ}_{Mxx..Myz}.m` into a (150, 6, n_t) array.
- **Remove the STF:** `forward.remove_stf(gf, dt, (t_rate − wavelet.get_auto_start_time(), rate), band=(10, 20, 120, 140))`, where `(t_rate, rate) = wavelet.get_stf_rate()`. Sample 0 is then the origin time.
- **Reorder to NED:** `ned_idx = [1, 0, 2, 3, 5, 4]`, `ned_sign = [1, 1, 1, 1, −1, −1]`.
- **Rotate to Z/R/T per receiver,** so that P falls mostly on R and S on T/Z, which makes the plots easier to read.
- **Build the `Dataset`:**
  - `Receivers(labels=R00Z…, coords, channel_dirs, phase='P', velocity=vp)`;
  - `Source(coords=…)` and `t0=0`;
  - `meta` holds `gf_band`, `data_band` and the provenance.
- Save it as a pickle.

**Note:** `apparent_fc` uses a single phase velocity per trace, vp here, while the window contains both P and S. This is fine for the demo. Separate P and S windows, each with its own Doppler c, is a later refinement.

### 3. Known source and synthetic "data"
- **Source:** an oblique DC via `mt.from_sdr`, e.g. strike 30, dip 60, rake −60, chosen so the horizontal ring has no complete P nodes.
- **Single-Brune data:** fc_true = 60 Hz.
- **Two-Brune data:** fc0 = 60, w2 = 0.6, vr = 0.6·vs, rupture azimuth 45°, plunge 0 (revised from fc1 = 40, fc2_0 = 90, w2 = 0.4).
- **Noise:** Gaussian, band-limited by B, SNR about 20. Also keep the noise-free version.

### 4. Single Brune: varying fc
- **STF plots:** Ṁ(t) and |S(f)| for fc ∈ {20, 40, 60, 90, 130} Hz, with the band B shaded.
- **Spectra:** data and model amplitude spectra (`spectra.amplitude_spectrum`) for three or four azimuths and for the median over receivers, one curve per fc.
- **Waveforms:**
  - record sections sorted by azimuth (`plot.wavefield(ds, synth, x=ds.azimuth)`), for under-, true- and over-estimated fc;
  - trace overlays (`plot.traces`).
- **Misfit scan:** VR (`misfit.vr`) against fc as a 1-D scan. This is a visualisation of misfit, not an inversion.
- **Interactive cell:** an `ipywidgets.interact` slider over fc (log scale) that redraws the spectra and the trace panel.

### 5. Two Brunes with directional fc2
- **fc against azimuth:** `plot.fc_azimuth(model, params, ds)` for fc2, with a horizontal line for fc0.
- **Effective STF:** the summed STF in the time and frequency domains at forward, perpendicular and backward azimuths relative to the rupture.
- **Spectral ratio by azimuth:** a heatmap of data spectrum / spectrum of the single-Brune best-fit fc, sorted by azimuth, so the directional signature stands out.
- **Fit comparison:** the true two-Brune model against the best single Brune (from the VR scan), compared on waveforms (`plot.wavefield` and `plot.traces`) and per-receiver VR against azimuth.
- **Interactive cell:** sliders for fc0, w2, vr/vs and rupture azimuth, with a printed table of true and model parameters.
- **Trade-off scan (5.6):** VR over (w2, vr) with fc0 and rupture direction at their true values.

## Verification
1. **Unit checks** (in a scratch script; also include brief asserts in the notebook):
   - `|stfs.brune_stf_spectrum| == spectra.brune` to machine precision.
   - The FFT of the sampled `brune_stf` matches `brune_stf_spectrum`.
   - `two_brune_spectrum(w2=0)` equals the single Brune.
   - `MTTwoBruneModel.forward` with w2 = 0 equals `convolve_stf` with a single Brune.
   - With vr = 0, fc2 is the same at every receiver.
   - `mt.sdr(mt.from_sdr(s, d, r))` recovers (s, d, r).
2. **Linear convolution:**
   - `convolve_stf` matches `np.convolve(gf, brune_stf(t, fc) * dt)[:n_t]` (direct time-domain convolution) to about 1e-6, with the STF band-limited by B for a fair comparison;
   - the causal pulse onset appears after the GF arrival, with no wrap-around energy at early times.
3. **Salvus run:** the GFs show P and S at about 9.9 ms and 19.2 ms; the radiation pattern varies across azimuth as expected for the DC; the band note from `ds.check_band()` has no warning.
4. **Notebook:** runs top to bottom in the salvus env, with `run_fwd=True` once and `False` afterwards. The VR scan peaks at fc_true = 60 Hz for noise-free single-Brune data.
5. **Regression:** `python examples/synthetic_mt_recovery.py` still runs, and `import unshake` works in base Anaconda without Salvus.

---

## Outcome (implemented 2026-11-03)

### Code
- **New `stfs.py`:** `brune_stf`, `brune_stf_spectrum` (1/(1 + i f/fc)², causal, unit moment) and `two_brune_spectrum`.
- **`forward.convolve_stf`:** frequency-domain product, zero-padded so the convolution is linear.
- **`mt.from_sdr`:** Aki & Richards box 4.4, NED.
- **`models.MTTwoBruneModel`:** parameters are 6 MT, fc0, w2, vr, rupt_az and rupt_plunge. fc0 and w2 are the same at every receiver; fc2 is fc0 Doppler-shifted via `apparent_fc`.
- Also updated: README and `.gitignore` (`examples/stf_demo_out/`).

### Checks
- `|brune_stf_spectrum| = spectra.brune` to machine precision.
- The FFT of the sampled `brune_stf` matches the analytic spectrum to 1e-6.
- `convolve_stf` matches a long-window FFT reference to 4e-8, so there is no wrap-around. Against `np.convolve` with the sampled STF the difference is 2 % at dt = 1 ms and 2e-4 at dt = 0.1 ms. That residual is the sampling error of the reference STF (its kink at t = 0), not an error in `convolve_stf`.
- `MTTwoBruneModel` with w2 = 0 equals the single Brune exactly, and with vr = 0 fc2 is constant.
- `from_sdr` round-trips through `sdr`.
- `examples/synthetic_mt_recovery.py` runs, and `import unshake` works without Salvus.

### Demo notebook (`examples/directional_stf_demo.ipynb`)
- **Salvus run:** full space, ±50 m box, 150 Hz mesh with 2 elements per wavelength. It took 213 s on 4 ranks, with outputs cached in `examples/stf_demo_out/`. P arrives at about 10 ms and S at about 19 ms, and the DC radiation pattern looks right on Z, R and T.
- **Single Brune** (fc_true = 60 Hz, peak SNR 50): the VR scan peaks at 60.1 Hz, with VR 0.979 noisy and 1.000 noise-free.
- **Two Brunes, first version** (separate fc1 = 40, fc2_0 = 90; w2 = 0.6, vr = 0.6 vs towards azimuth 45°, c = vp; superseded by the shared-fc0 revision below):
  - the best single Brune is 70.7 Hz with VR 0.899, against VR 0.970 for the true model;
  - the 3-component data / single-Brune spectral ratio rises at high frequency towards 45° and falls towards 225°, and the pattern is still visible with noise.
- **Not handled yet:** Doppler with a single c = vp, although the window contains both P and S. For an inversion, cache `rfft(gf)` to save time.

### Revision: shared corner frequency (2026-11-03)
- `MTTwoBruneModel` now uses one corner frequency, fc0, for the symmetric STF and for the directional STF perpendicular to the rupture. That leaves 11 parameters.
- Notebook truth: fc0 = 60 Hz, w2 = 0.6, vr = 0.6 vs towards 45°. Checks: w2 = 0 and vr = 0 each reduce exactly to a single Brune at fc0.
- Results, with the notebook band as later edited by the user (spectra to about 200 Hz):
  - the best single Brune is 62.6 Hz with VR 0.906, against VR 0.940 for the true two-Brune model;
  - the directional signature in the spectral ratio is still clear.
- **Trade-off scan (notebook 5.6).** VR is flat along w2·vr = const:
  - noisy data: the true model has VR 0.9405, the best grid point (w2 0.64, vr/vs 0.57) also 0.9405, and w2 = 1 with vr/vs = 0.36 (same product) 0.9398;
  - noise-free data: 1.0000 against 0.9991.
- **Conclusion.** The data constrain fc0, the rupture direction and the product w2·vr/c, but not w2 and vr separately. Before inverting, either fix one of w2 or vr (e.g. w2 = 1, a fully directional source), or sample in (w2·vr/c, w2) and expect a broad marginal in w2.
