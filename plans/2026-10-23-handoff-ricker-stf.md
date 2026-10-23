# Handoff: the Ricker STF in the Salvus Green's functions

This note starts a new working session. Read it together with `plans/2026-10-22-milestone1-architecture.md`: that file covers the architecture, what was verified, and next steps 1–7. This note is about next step 2.

## Where things stand (2026-10-23)

- **Package and install.** The `unshake` milestone 1 package lives in `git/unshake/src/unshake`, with modules `data`, `mt`, `spectra`, `forward`, `misfit`, `models`, `solvers`, `results` and `plot`. It is installed editable in:
  - base Anaconda Python 3.11;
  - the miniforge env `/Users/tomhudson/miniforge3/envs/salvus_src_inv`, which is the notebook kernel.
- **Nothing is committed in git yet.**
- **Added since the milestone 1 plan:**
  - `solvers.lsq_loglike(m_dirs, m_lsq, cov)`: the LSQ Gaussian log-density, profiled over amplitude.
  - `solvers.lsq_lune_profile(m_lsq, cov)`: optimises orientation per lune bin; about 25 s.
  - `plot.lune(..., values=, stat='max', vmin=, colorbar=)`.
- **Notebook.** `example_crevasse_iq_joint_inv_testing.ipynb`, section 3, has the unshake cross-check cells:
  - build and save a `Dataset`;
  - least squares compared with SeisSrcInv;
  - Monte Carlo with 10,000 samples;
  - lune and beachball plots.

  The LSQ lune defaults to covariance samples; set `plot_full_lune_lsq = True` for the full-lune profile.
- **Saved Dataset.** The notebook saves the `Dataset` to `data_out/<uid>/<uid>_unshake_dataset.pkl`. `ds.meta` records the GF provenance, including `centre_freq_Hz`.

## The problem

The GFs in `run_salvus_sm_synthetics.py` (`gen_synths_salvus`) are made as follows:
- **Source wavelet:** `sn.simple_config.stf.Ricker(center_frequency=centre_freq_Hz)`. `centre_freq_Hz` is the mean spectral peak of the data, from `get_stf_and_attn_starting_model_params` in 25–150 Hz.
- **Mesh:** `max_frequency_in_hertz = 2 * centre_freq_Hz` and `elements_per_wavelength = 2.0`. The GFs are therefore only accurate up to about 2 × f0, and coarsely so.
- **Attenuation:** on, with Q from the spectral-ratio t* estimate. Medium: vp 3540, vs 1820, ρ 917, homogeneous, with a free surface.
- **Output:** DAS strain projected onto the fibre axis, resampled to the data rate. The notebook then rolls the synthetics by the padding offset, an anisotropy travel-time correction, and cross-correlation shifts picked against the `Mxx` GF.

So every GF already contains a Ricker source spectrum, and its band is limited by the mesh. In `MTSpectrumModel` the source spectrum S(f; fc) is applied on top, so the Ricker has to be removed or accounted for first. Otherwise the fitted corner frequencies are biased.

## Options to evaluate

1. **Divide out the Ricker inside its band.** This hook already exists: `forward.apply_source_spectrum(..., reference=lambda f: spectra.ricker_spectrum(f, f0))`, with a crude water level of 1e-3 × max.
   - Restrict the misfit to the band where the Ricker is above a threshold and the mesh is accurate (below about 2 f0).
   - Band-pass the data to the same band.
   - Cheap, but the usable band may be too narrow around fc.
2. **Re-run Salvus with a broadband STF.** For example a Gaussian or smoothed Heaviside with a higher `max_frequency_in_hertz` and more elements per wavelength.
   - Cleanest result, but 3D cost scales roughly with f^4.
3. **Analytic or semi-analytic broadband GFs for the homogeneous medium.** For example far-field plus near-field full-space terms (the free surface matters for surface-receiver DAS), or fk / reflectivity for a half-space.
   - Fast and gives an arbitrary band, but loses Salvus fidelity.
4. **Keep the Ricker and make it relative.** The model spectrum becomes S(f; fc) / R(f; f0) only within the passband; equivalently, fit spectral ratios between receivers so the common Ricker cancels.
   - This links naturally to the Doppler idea: directivity shows up as azimuthal variation of fc, and a common STF cancels in ratios.

## Things to check first

- **Salvus Ricker timing.** Check its time origin and shift (it may be centred at t > 0) and its normalisation. These affect phase alignment and the deconvolution, not just amplitude. Compare `spectra.ricker_spectrum` against the Salvus Ricker definition.
- **Usable band.** Compare the data spectra with the GF spectra per channel; the data are band-passed at 10–500 Hz.
- **Typical f0 and fc range** for the icequake events.

## Suggested first step

Plot, for the saved `Dataset`:
- data spectra versus GF spectra;
- the Ricker spectrum and the 2 f0 mesh limit;
- a Brune spectrum for plausible fc.

Use this to decide between options 1/4 (no re-simulation) and 2/3 (new GFs).
