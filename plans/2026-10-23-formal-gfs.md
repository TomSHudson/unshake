# Plan: formal (STF-free) Green's functions in unshake

## Context
The Salvus GFs in unshake are not formal Green's functions. Each MT column is the medium response convolved with a Ricker STF, g = G · R(f; f0), where f0 is about 98 Hz for the saved event `20231022080438720752`. The mesh is accurate up to about 2·f0 with only 2 elements per wavelength.

The target is a full-waveform joint MT + fc inversion (`MTSpectrumModel`). For that, the source spectrum S(f; fc) must be the only source-time term. Otherwise:
- fc is biased, because the fit has to make up for R;
- the azimuthal (Doppler) fc pattern is distorted.

The goal is to compute GFs once and keep the STF separate from the path effects.

A GF from a numerical simulation is always band-limited. The practical "formal" GF is therefore G_B = G · B, where B is a known, zero-phase, almost-flat band window. **The data must be band-limited with the same B.** The band is capped at **≤150 Hz for now**.

One definition of B is used everywhere: `spectra.flat_band(f, f0, f1, f2, f3)`.
- It is a cosine ramp up over f0–f1, flat over f1–f2, and a cosine ramp down over f2–f3.
- It is the same shape as Salvus's `FlatSpectrumWavelet` / `Ormsby`.
- It covers both routes to formal GFs:
  - existing Ricker GFs: deconvolve R, then apply B;
  - new runs: simulate directly with a `FlatSpectrumWavelet`, so the GFs already equal G·B.

Keep unshake simple: a few plain functions, with no new classes or framework.

## Recommended wavelet for real GFs (new Salvus runs)
Use `sn.simple_config.stf.FlatSpectrumWavelet(f0=2, f1=8, f2=130, f3=150, time_shift_in_seconds=t_s)`.

Why:
- **Known band filter.** Its spectrum is exactly B: flat over 8–130 Hz and zero above 150 Hz. The simulated GF is therefore G·B with nothing to deconvolve, and the STF is "removed" by construction.
- **Later source spectra.** A proper Brune/Boatwright S(f; fc), or a finite-source STF, can be convolved on later. Below 150 Hz the result is exact.
- **Clean high-frequency band.** Nothing is above f3, so no energy reaches frequencies the mesh cannot resolve. Use `max_frequency_in_hertz=150` and `elements_per_wavelength` of about 3 (tensor order 2).
- **Low ramp below the data band.** The 2–8 Hz ramp sits below the data's 10 Hz high-pass and avoids the long DC tail that a flat spectrum to 0 Hz would need. Use a `time_shift_in_seconds` large enough for the wavelet to be causal from the simulation start; the GF shift picking absorbs it.

Alternatives considered:
- **Ricker:** narrow band, and it vanishes at DC.
- **`GaussianRate`:** flat to DC, but it is a Gaussian, not flat in band, and puts energy above the mesh limit unless it is made wide.
- **`Delta` / `Heaviside`:** not band-limited, so they give numerical dispersion.

To check at execution:
- Whether Salvus injects the wavelet as moment or as moment-rate for MT sources, by comparing `get_stf` with `get_stf_rate` and reading the docs. unshake treats S(f) as a moment-rate spectrum, so if Salvus uses moment there is a factor of iω to correct.
- That `flat_band` matches the spectrum of the sampled `FlatSpectrumWavelet`.

## Steps

### 0. Move `run_salvus_sm_synthetics.py` into unshake
- Copy it to `git/unshake/src/unshake/salvus_fwd_driver.py`.
- **Salvus is optional.**
  - At the top of the module, `try: import salvus.namespace as sn ...` (plus `xarray`, `pandas`, `pyproj`, `obspy`). On `ImportError`, set `HAS_SALVUS = False`.
  - `gen_synths_salvus` raises a clear `ImportError` only when it is called without Salvus.
  - `__init__.py` does not import the driver. `import unshake` works without Salvus.
- Add optional arguments to `gen_synths_salvus`. Their defaults reproduce current behaviour:
  - `wavelet=None`, which means `Ricker(center_frequency=centre_freq_Hz)`;
  - `max_frequency_in_hertz=None`, which means `2*centre_freq_Hz`;
  - `elements_per_wavelength=2.0`.
- **Docstring.** Update the docstring to cover:
  - the defaults;
  - that the GFs are convolved with `wavelet`, so the default Ricker GFs are *not* formal GFs;
  - the recommended `FlatSpectrumWavelet` call above, for formal band-limited GFs;
  - that the data must later be band-limited with the same B (`forward.bandlimit`).
- Do not otherwise refactor the code.
- Replace the old `python/initial_joint_inv_development/run_salvus_sm_synthetics.py` with a one-line shim, `from unshake.salvus_fwd_driver import *`. This keeps the gornergletscher notebooks working.
- In the notebook, change the import to `from unshake.salvus_fwd_driver import gen_synths_salvus`.
- **pyproject:**
  - The core `dependencies` stay `numpy`, `scipy` and `matplotlib`.
  - Add an optional extra, `salvus = ["xarray", "pandas", "pyproj", "obspy"]`.
  - README note: install Salvus via Mondaic, then `pip install -e ".[salvus]"`.

### 1. Diagnostic spectra plot (saved Dataset)
- In `spectra.py`:
  - `amplitude_spectrum(x, dt, nfft=None)` applies a Hann taper, zero-pads and takes the rfft. It returns `(f, |X|)` and is vectorised over leading axes.
  - `flat_band(f, f0, f1, f2, f3)` gives B, as above.
- In `plot.py`, add `spectra_check(ds, m, f0, fmax=150, fc=(50, 100, 200))`. It draws three panels:
  1. Data, LSQ-synthetic and pre-pick noise spectra. Each shows the median and the 10–90 % range over channels. The noise comes from the 50 ms before the pick (`ds.t0 = -0.05`).
  2. Reference curves:
     - R(f; f0);
     - the 2·f0 mesh limit and the 150 Hz cap;
     - the 10–500 Hz filter band.
  3. The data/synthetic ratio, which estimates S_true/R. It has Brune(fc)/R overlays, plus a version coloured by azimuth.
- Add a notebook cell (section 3) that:
  - loads the pickle and runs `least_squares`;
  - calls `plot.spectra_check`;
  - samples the Salvus `Ricker(center_frequency=f0)` with `get_stf`, compares it with `spectra.ricker_spectrum`, and prints the time shift;
  - checks moment versus moment-rate.

### 2. Formal band-limited GFs from the existing Ricker GFs
- In `forward.py`:
  - `bandlimit(x, dt, band)` applies B = `flat_band(f, *band)` by FFT multiplication, with zero-padding. Here `band = (f0, f1, f2, f3)`.
  - `remove_stf(x, dt, stf, band)` returns x · B / STF, set to zero wherever B = 0, which needs no water level. `stf` is a callable f → complex spectrum, or a sampled STF trace, which keeps the Salvus time shift exact.
- In `data.py`, `Dataset.remove_stf(stf, band=(2, 8, 130, 150))` returns a new Dataset in which:
  - `gf` is set to `remove_stf(gf)` and `data` to `bandlimit(data)`;
  - `meta['gf_band']` and `meta['data_band']` are both set to `band`, and `meta['gf']` notes the STF that was removed.
- For future `FlatSpectrumWavelet` runs, the notebook sets `meta['gf_band']` when it builds the Dataset, and calls `bandlimit` on the data, which sets `meta['data_band']`.
- **Printed band note.** A small `data.check_band(ds)` prints a one-line note: "GFs and data must be band-limited by the same filter B: gf_band=…, data_band=…".
  - It calls `warnings.warn` if either band is missing or if the two differ.
  - It is called at the start of `least_squares`, `monte_carlo` and `make_loglike`.
- Drop the `reference=` argument from `apply_source_spectrum` and `MTSpectrumModel`, since it is redundant. Update the `spectra.py` TODO.

### 3. Decision gate (no implementation now)
After the diagnostic and the formal-GF ratio plot:
- **If 8–130 Hz brackets plausible fc:** continue with the deconvolved Ricker GFs.
- **If it does not, or for production:** re-run Salvus with the recommended `FlatSpectrumWavelet`. No unshake code changes are needed.

## Files
- New: `git/unshake/src/unshake/salvus_fwd_driver.py`. The old script becomes a shim.
- Edited:
  - `spectra.py`: `amplitude_spectrum`, `flat_band` and the TODO;
  - `forward.py`: `bandlimit`, `remove_stf`, and dropping `reference`;
  - `data.py`: `Dataset.remove_stf` and `check_band`;
  - `solvers.py`: the `check_band` calls;
  - `models.py`: dropping `reference`;
  - `plot.py`: `spectra_check`;
  - `pyproject.toml`, `README.md` and the notebook.
- Save this plan as `git/unshake/plans/2026-10-23-formal-gfs.md`.

## Verification
Run in `/Users/tomhudson/miniforge3/envs/salvus_src_inv` on `data_out/20231022080438720752/20231022080438720752_unshake_dataset.pkl`.
1. **Imports:**
   - `import unshake` works in base Anaconda, which has no Salvus, and `gen_synths_salvus` raises the clear `ImportError` there.
   - The Salvus env imports the driver, and the old shim works.
2. **Round trip.** Re-applying the Ricker to `remove_stf(gf)` gives `bandlimit(gf)` to numerical precision.
3. **Spectrum checks.**
   - The Salvus Ricker spectrum matches `ricker_spectrum`, and the shift is reported.
   - The sampled `FlatSpectrumWavelet(2, 8, 130, 150)` spectrum matches `flat_band`.
4. **Band note.** `least_squares` on the formal-GF Dataset prints the band note without a warning. On the original Dataset it warns.
5. **LSQ.** The MT orientation is essentially unchanged, and the VR before and after is reported.
6. **Ratio panel.** With formal GFs it is roughly flat or Brune-like within 8–130 Hz, not 1/R-shaped.
7. **Synthetic smoke test.** `examples/synthetic_mt_recovery.py` still runs.

---

## Outcome (implemented 2026-11-02)

### Deviations from the plan above
- **Band.** The default band is **B = (10, 20, 120, 140) Hz**, not (2, 8, 130, 150).
  - The data are already high-passed at 10 Hz.
  - Below about 10 Hz, 1/R amplifies by 50–100×.
  - B has to sit inside the wavelet's own band.
- **Recommended wavelet:** `FlatSpectrumWavelet(f0=5, f1=10, f2=140, f3=150, time_shift_in_seconds=0.04)` with `max_frequency_in_hertz=150`.
  - Salvus truncates this wavelet to ±0.2/f0 s, so its spectrum has about 20 % ripple.
  - Its GFs therefore also go through `Dataset.remove_stf(wavelet.get_stf(), band)`. A single code path covers any wavelet.
- **No separate `stf` argument type.** `remove_stf` accepts either a callable or a sampled `(t, s)`, as planned.
- **`least_squares(check_band=True)`.** Internal calls pass `False`, so the band note prints once per inversion.

### Verification results (event 20231022080438720752, f0 = 98.2 Hz)
- **Imports.** unshake imports in base Anaconda without Salvus, and `gen_synths_salvus` raises a clear `ImportError` there. The old script path works through the shim. `examples/synthetic_mt_recovery.py` still runs.
- **Salvus Ricker.** It is zero-phase and centred on t = 0, with auto start at −0.0159 s, and matches `ricker_spectrum` to 1e-5. `get_stf_rate` = d/dt `get_stf`. **Still open:** whether Salvus injects the wavelet as moment or as moment-rate for MT sources (check the Salvus docs).
- **Round trip.** Re-applying the Ricker to `remove_stf(gf)` gives `bandlimit(gf)` to a relative error of 3.5e-3. The residual comes from cropping to the 0.25 s window, which cuts filter tails.
- **Band note.** It warns on the original Dataset and is silent on the formal one.
- **LSQ.** The MT is essentially unchanged: cos ∠ = 0.996, lune (−12.9, 53.9) becomes (−15.2, 58.1). VR goes from 0.029 to 0.031. The fit is still poor, as in milestone step 5.
- **Diagnostic.**
  - With Ricker GFs, the data/synthetic ratio follows 1/R × Brune, with fc roughly 50–100 Hz.
  - With formal GFs the ratio is close to flat over 20–120 Hz, falling slightly (between the Brune 50 and 100 Hz curves).
  - The data are only about 3–10× above the pre-pick noise, from about 20 Hz to more than 200 Hz.
  - No clear azimuthal fc pattern is visible yet.
- **Decision.** fc appears to sit near the top of the usable band (about 120 Hz). This favours re-running Salvus with the flat-spectrum wavelet up to 150 Hz, and probably higher later, so that the high-frequency side of the corner is constrained.

### Found while moving the driver (fixed 2026-11-02)
- **Mxz / Myz sources.** `gen_synths_salvus` built `source_Mxz` and `source_Myz` with `mxx=1` as well as `mxz=1` / `myz=1`. Fixed, so each source is now a single component.
  - GFs made before the fix have "Mxz" = Gxx + Gxz and "Myz" = Gxx + Gyz.
  - Correct them as G_Mxz − G_Mxx and G_Myz − G_Mxx, or re-run Salvus.
- **`domain_dict`.** It is now used. `None` gives `DEFAULT_DOMAIN_DICT`, which is the previously hard-coded box.

### Moment versus moment-rate (resolved 2026-11-02)
**Test setup.** A small full-space Salvus run, kept in a scratchpad project so nothing existing was touched:
- homogeneous ice (vp 3540, vs 1820, ρ 917), no attenuation;
- a ±220 m box with a 40 Hz mesh and a 20 Hz Ricker;
- Mzz and Mxz sources;
- receivers at 100 and 140 m in five directions;
- run time about 2.5 min on 4 ranks.

The output was compared with the full Aki & Richards eq. 4.29 solution (near, intermediate and far field).

**Results:**
- **The wavelet is the moment M(t).** With M(t) = M·s(t), the displacement misfit is 0.5 % and the amplitude ratio is 0.998. With s(t) as the moment rate, the misfit is 100 %. Salvus `mxz=1` means Mxz = Mzx = 1.
- **Strain channels.**
  - Channels 0–2 are εxx, εyy, εzz.
  - Channels 3–5 are the *engineering* shears 2εyz, 2εxz, 2εxy.
  - x = east, y = north, z = up.
- **DAS projection fixed in the driver.** The old projection, ch0·sinθ + ch1·cosθ with an unsigned θ, was wrong. It is now ε_nn = nᵢnⱼεᵢⱼ, using all six channels and a signed 3D fibre direction from neighbouring channels. Checked against the analytic strain for random fibre directions: misfit 1–2 %.

**Fix applied.** The GF STF to divide out is the wavelet's **moment rate**: `Dataset.remove_stf(wavelet.get_stf_rate(), band)`.
- The formal GFs are then the strain response to a unit-impulse moment rate. That matches the moment-rate Brune/Boatwright spectra.
- The DAS data record strain, as confirmed by the user, so no further factor of iω is needed.
- Docstrings and the notebook cell are updated.

**Re-run diagnostic** (saved Dataset, moment-rate deconvolution):
- The data/synthetic ratio, which now estimates Ṁ_true(f), is flat over 20–120 Hz. This suggests fc is at or above the top of the band.
- The MT changes more than before: cos ∠ = 0.88 to the original. VR is 0.004.
- These GFs predate the Mxz/Myz and DAS-projection fixes, so this interpretation is provisional until Salvus is re-run.

---

## Summary of changes for commit (2026-11-02)

### Files in `git/unshake`
**New:**
- **`src/unshake/salvus_fwd_driver.py`:** `run_salvus_sm_synthetics.py` moved into the package.
  - Salvus and its helpers are an optional import. `HAS_SALVUS` records whether they loaded, and calling `gen_synths_salvus` without them raises a clear `ImportError`.
  - New optional arguments, whose defaults keep the old behaviour: `wavelet` (default Ricker), `max_frequency_in_hertz` (default 2·f0) and `elements_per_wavelength` (default 2.0).
  - Fixed: the `Mxz` and `Myz` sources also set `mxx=1`.
  - Fixed: `domain_dict` was ignored. `None` now gives `DEFAULT_DOMAIN_DICT`, which is the old hard-coded box.
  - Fixed: the DAS strain projection is now ε_nn = nᵢnⱼεᵢⱼ, using all six Salvus strain channels (shears are engineering strains) and a signed 3D fibre direction.
  - The docstring documents the Salvus moment convention and the recommended `FlatSpectrumWavelet`.
- **`plans/2026-10-23-formal-gfs.md`:** this file.

**Modified:**
- **`spectra.py`:** new `flat_band` (the band filter B) and `amplitude_spectrum`. The module docstring now explains formal band-limited GFs, replacing the old TODO.
- **`forward.py`:** new `bandlimit`, `stf_spectrum` and `remove_stf`. Removed the `reference=` argument from `apply_source_spectrum`.
- **`data.py`:** new `Dataset.remove_stf`, `Dataset.bandlimit` and `Dataset.check_band`.
- **`solvers.py`:** `least_squares` (with a new `check_band=True` argument) and `make_loglike` print the band note and warn when the bands are missing or differ.
- **`models.py`:** removed `reference=` from `MTSpectrumModel`. Internal least-squares calls skip the band note.
- **`plot.py`:** new `spectra_check` diagnostic.
- **`pyproject.toml`:** new optional extra, `salvus = ["xarray", "pandas", "pyproj", "obspy"]`.
- **`README.md`:** install note for the Salvus extra, and an updated module list.

### Suggested commit message
```
Formal band-limited GFs, Salvus driver moved into unshake

- Move run_salvus_sm_synthetics.py to unshake.salvus_fwd_driver (optional
  Salvus import; new wavelet / max_frequency_in_hertz /
  elements_per_wavelength args; [salvus] extra)
- Fix driver bugs: Mxz/Myz sources also set mxx=1; domain_dict ignored;
  DAS strain projection (now n_i n_j eps_ij with engineering shears)
- Remove the simulation STF from GFs: forward.remove_stf / bandlimit,
  Dataset.remove_stf / bandlimit / check_band, spectra.flat_band.
  Salvus applies the wavelet as moment M(t) (verified against Aki &
  Richards), so the moment rate (get_stf_rate) is divided out
- Solvers print a GF/data band-filter note; drop reference= hook
- plot.spectra_check diagnostic; spectra.amplitude_spectrum
```

### Outside the repo (`python/initial_joint_inv_development`, not under git)
- **`run_salvus_sm_synthetics.py`** is now a shim that re-exports from `unshake.salvus_fwd_driver`, so the Gornergletscher notebooks keep working.
- **`example_crevasse_iq_joint_inv_testing.ipynb`:**
  - the commented import in cell 1 now points to the new module;
  - two new cells at the end: a heading, and the spectra diagnostic with formal GFs using `get_stf_rate()`.

### Before relying on results
- **Existing GFs are stale.** The GFs on disk and in the saved Dataset predate the Mxz/Myz and projection fixes, so re-run Salvus before relying on them.
- **Changed default:** `gen_synths_salvus` now writes correctly projected DAS strain, so new runs won't reproduce old outputs.
- **Removed argument:** any code still passing `reference=` to `apply_source_spectrum` or `MTSpectrumModel` will now raise an error.

## Notebook: new Salvus run for the example event (added 2026-11-02, not yet run)
Section 3.y of `example_crevasse_iq_joint_inv_testing.ipynb` has three new cells.

**3.y.1: Salvus run.**
- Uses the fixed driver with `FlatSpectrumWavelet(5, 10, 140, 150, shift 0.04 s)`, `max_frequency_in_hertz=150`, `elements_per_wavelength=3`, and an end time of 0.25 s, which covers the whole data window.
- Project: `/Volumes/4tb_extssd/salvus_tmp_workspace/src_inv`. The driver deletes and recreates this project if it exists.
- Output goes to a new directory, `data_out/<event_uid>_salvus_v2_flat150`. The cell refuses to run if that directory already holds synthetics.

**3.y.2: formal band-limited GFs.**
- Divides out the moment rate with `remove_stf((t_rate − t_sim_start, rate), band)`, where t_sim_start = `wavelet.get_auto_start_time()`. Output sample k is then at origin + k·dt.
- Timing referencing checked on the full-space test: offset 0–2 samples, against 16 if the reference were wrong.
- The rest of the prep matches section 2.2: pre-pad shift (zero-filled), anisotropy correction, and cross-correlation shifts against Mxx. The shift window is now symmetric, [−20, 20], because the old one-sided [−25, 0] window was absorbing the Ricker start-time lag.
- Band-limits the data by the same B, uses NED ordering, and saves `ds_v2` to the new directory.

**3.y.3: inversion check.**
- Least squares (VR, lune, strike/dip/rake, and the angle to the old MT) and a 10k-sample Monte Carlo.
- Lune, beachball and wavefield plots, plus `spectra_check`.
