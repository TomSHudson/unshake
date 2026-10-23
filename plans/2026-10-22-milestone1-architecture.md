# unshake: initial package architecture (milestone 1)

## Context
SeisSrcInv (`/Users/tomhudson/Documents/python/git_repositories/SeisSrcInv`, **not to be modified**) does full-waveform MT inversion. It uses least squares plus Monte Carlo random sampling. Its weaknesses:
- Logic is buried in 1500-line, file-IO-coupled functions (`inversion.run`), with the `inversion_type` and misfit branches copy-pasted many times.
- It parses NonLinLoc through `os.system`.
- Its likelihood is weak: `exp(-(1-sim)/2)`.
- It has several bugs.

The research goal is a joint MT + asymmetric corner-frequency (Doppler / rupture-velocity) inversion, starting with point sources and later extending to finite and moving sources. The new package `unshake` already exists as a placeholder at `/Users/tomhudson/Documents/projects/joint_mt_fc_inversion/git/unshake`, with a hatchling `pyproject.toml`, a `src/` layout and an empty `__init__.py`.

Milestone 1 delivers three things:
1. A small, array-based, modular core with the SeisSrcInv essentials.
2. A source-spectrum / Doppler slot ready for the joint inversion.
3. A pluggable sampler, starting with vectorised Monte Carlo.

Data prep and Green's function generation (Salvus, DAS projection, shift picking) stay in the notebooks. Formal tests come later.

## Design principles
- **Arrays in, arrays out.** The core never reads files. One small `io` helper handles saving and loading results.
- **Two extension seams:**
  - a `Model`: it maps a parameter vector to synthetics and owns its prior;
  - a `Sampler`: it explores a log-likelihood.

  New physics means a new `Model`. A new inference method means a new sampler function.
- **Vectorised over samples.** Use `einsum` and batches, not Python loops. Multiprocessing is optional.
- **Flat module layout.** Use few files and no deep class hierarchies.

## Package layout (`src/unshake/`)
| Module | Contents |
|---|---|
| `__init__.py` | Expose `Dataset`, the models, `least_squares`, `monte_carlo` and `Result` |
| `data.py` | `Dataset` class, described below |
| `mt.py` | MT conventions and samplers, described below |
| `spectra.py` | Source spectra and the Doppler apparent-fc stub, described below |
| `forward.py` | `synthetics(gf, m)`, `apply_spectrum(...)`, described below |
| `misfit.py` | `vr`, `pcc`, `ncc` and `gaussian_loglike`, described below |
| `models.py` | `Model` protocol plus `MTModel`, `ForceModel` and `MTSpectrumModel` (stub) |
| `solvers.py` | `least_squares` (linear, with posterior covariance) and `monte_carlo(model, dataset, n, misfit=..., batch=..., n_jobs=...)` |
| `results.py` | `@dataclass Result` (samples, log-likelihood, posterior weights, best fit, metadata) with `.save()`/`.load()` via pickle or `np.savez` |
| `plot.py` | Lune histogram, beachball (obspy), waveform-fit panels, and (later) fc versus azimuth |

**`data.py`: the `Dataset` class**

`Dataset` is a full class: the one place that holds everything about an event that an inversion needs. Models, solvers and plots receive only the `Dataset`; they are never passed loose arrays.

*What it holds (attributes):*
- **Waveforms:** `data`, shape (n_rec, n_t), plus `dt` and `t0`.
- **Green's functions:** `gf`, a dict keyed by source type, so MT and single-force GFs can be held together:
  - `gf['mt']` has shape (n_rec, 6, n_t);
  - `gf['force']` has shape (n_rec, 3, n_t).
- **Receivers:** `receivers` holds per-receiver `labels`, `coords` (n_rec, 3) in the local ENU frame, `component` / channel-axis vectors (for DAS), `phase` (P or S), and the phase velocity.
- **Source:** `source` holds the event uid, the hypocentre coordinates and the origin time.
- **Weights and noise:** `weights`, plus `noise_std` (given, or estimated from a pre-event window).
- **Metadata:** a `meta` dict for free-form provenance, such as the GF generator, filter band and STF used for the GFs.

*Derived properties (computed from coordinates, not stored):*
- `ray_dirs`: unit take-off vectors along straight rays, used by the Doppler term.
- `azimuth`, `takeoff`, `distance`.

*Methods:*
- `window(t1, t2)`, `subset(idx_or_labels)`, `shift(per_rec_samples)` (alignment as preprocessing) and `estimate_noise()`.
- `save(path)` / `Dataset.load(path)` via pickle, so a prepared event can be reloaded without redoing the notebook prep.
- `plot()`: a quick look at the traces next to the GF columns.

**`mt.py`: MT conventions and samplers**
- A single documented convention: NED 6-vector `[nn, ee, dd, ne, nd, ed]`, the same order as `green_func_array` in the notebooks.
- Conversions: 6 ↔ 3×3, to Lune γ/δ, to strike/dip/rake.
- Vectorised samplers: `sample_full_mt`, `sample_lune`, `sample_dc`, `sample_force`. They take `(n, rng)` and return shape (n, 6) or (n, 3).
- A single random-rotation helper fixes the inconsistent `arccos`/`arctan2` rotations in SeisSrcInv.

**`spectra.py`**
- Source spectra: `brune(f, fc, n=2, gamma=1)` and `boatwright(...)`.
- `apparent_fc(fc0, vr, c, ray_dirs, rupture_dir)`, the Doppler term: `fc0 / (1 - (vr/c)·cosθ)` for a unilateral line source, with a bilateral variant.
- These are written as plain functions so that the theory can be swapped easily.

**`forward.py`**
- `synthetics(gf, m)`: `einsum('rct,sc->srt')`, batched over samples.
- `apply_spectrum(synth, dt, spec_per_rec)`: multiplies by the source spectrum in the frequency domain (rfft, multiply, irfft), per receiver.

**`misfit.py`**
- `vr`, `pcc`, `ncc` (with an optional max-lag shift) and `gaussian_loglike`.
- All are vectorised over samples.
- One `mode='per_trace'|'stacked'` argument replaces the four copy-pasted branches in SeisSrcInv.

**`models.py`**
- `Model` protocol:
  - `param_names`
  - `sample_prior(n, rng)`, which returns shape (n, p)
  - `forward(params, dataset)`, which returns shape (n, n_rec, n_t)
- `MTModel(kind='full'|'lune'|'dc')` handles MT magnitude either way: free scalar `M0` sampled log-uniformly around the least-squares norm, or fixed to it (the SeisSrcInv behaviour).
- `ForceModel`.
- `MTSpectrumModel` (stub): its parameters are MT, `fc0`, `vr` and the rupture azimuth/plunge, and its forward is `apply_spectrum(synthetics(...), apparent_fc(...))`. Its prior and forward are wired up but the physics is left minimal. It is the first thing to flesh out in milestone 2.

**`solvers.py`**
- `least_squares` is a linear solve with posterior covariance.
- `monte_carlo(...)` builds the log-likelihood from the chosen misfit as a proper Gaussian likelihood. It also reports `VR` / `PCC` for interpretation.
- A future `mcmc(model, dataset, ...)` (emcee) uses the same `Model` and log-likelihood. Nothing else changes.

### Reused from SeisSrcInv (copied and rewritten vectorised, not imported)
| Source in SeisSrcInv | Destination in unshake |
|---|---|
| `inversion.generate_random_MT` (424), `generate_random_MT_Lune_samp` (438), `generate_random_DC_MT` (474), `generate_random_single_force_vector` (499) | `mt.py` samplers |
| `get_full_MT_array` (227), `get_six_MT_from_full_MT_array` (234) | `mt.py` conversions. Drop the √2 off-diagonal scaling to match the Salvus GF columns, and document this. |
| `perform_inversion` (270), `perform_inversion_lsq_with_posterior` (313) | `solvers.least_squares` |
| `variance_reduction` (691), `pearson_correlation_comparison` (735) | `misfit.py` |
| `plot.find_delta_gamm_values_from_sixMT` (1111), `MT33_SDR` (190), `plot_Lune` (1250) | `mt.py` / `plot.py` |

### Deliberately left out of milestone 1
- The multi-medium mode.
- Relative-magnitude switches. These are replaced by an M0 parameter.
- Auto time-shift inside the sampling loop. Shifts become preprocessing, or an optional `ncc` max-lag.
- NonLinLoc parsing, MTFIT `.mat` output, phase cutting, the fk unit factors and `prep_inv_input_data`.
- Mixed DC + crack + force coupling types. These can return later as a `Model`.

## Extension path (not built now, but the structure allows it)
- **Joint MT+fc (milestone 2).** Flesh out `MTSpectrumModel` and switch to the MCMC sampler for the roughly 9-parameter space.
- **Finite and moving sources.** `Dataset.gf` gains a sub-source axis, shape (n_rec, n_sub, n_comp, n_t). A `FiniteSourceModel` sums sub-sources with rupture-time delays. `forward` and the samplers stay unchanged.

### Open issue to note in the code
The Salvus GFs are generated with a Ricker STF. For the spectral model, either regenerate the GFs with a broadband or impulsive STF, or deconvolve the Ricker in `apply_spectrum`. This plan does not resolve it. It is left as a `TODO` in `spectra.py`.

## First implementation step
Create `unshake/plans/` and save this plan there as `unshake/plans/2026-10-22-milestone1-architecture.md`. Later plans go in the same folder.

## Other file changes
- `pyproject.toml`: add `dependencies = ["numpy", "scipy", "matplotlib"]` and an optional `[project.optional-dependencies] plot = ["obspy"]`.
- `README.md`: add a short "dev install: `pip install -e .`" note and a sketch of the architecture.
- No tests directory yet.

## Verification
1. Install with `pip install -e /Users/tomhudson/Documents/projects/joint_mt_fc_inversion/git/unshake`.
2. **Synthetic recovery script** (`examples/synthetic_mt_recovery.py`):
   - Use random GFs or a simple analytic GF, a known MT and added noise.
   - Check that `least_squares` and `monte_carlo` recover the MT.
   - Check that the Lune and beachball plots render.
3. **Real data cross-check.** In new cells at the end of `example_crevasse_iq_joint_inv_testing.ipynb`:
   - Build a `Dataset` from the notebook's existing `green_func_array`, data, receiver coordinates and event.
   - Save it with `Dataset.save`.
   - Run `unshake.least_squares`.
   - Confirm the MT matches the notebook's `SeisSrcInv.inversion.perform_inversion` output to numerical precision.
   - Run `monte_carlo` with the Lune model and a small sample size (about 1e4) so it is fast, then plot the Lune and beachball.
4. **Doppler stub smoke test.** Call `MTSpectrumModel.forward` with `vr=0`. It should reproduce the MT synthetics filtered by a symmetric Brune spectrum. Plot `apparent_fc` against azimuth to check the asymmetry.

---

## Status after implementation (2026-10-22)

All modules in the layout above are implemented. The package is installed as an editable install in the base Anaconda Python 3.11, and nothing is committed yet.

### Verified
- **Synthetic recovery** (`examples/synthetic_mt_recovery.py`, analytic P + S Green's functions): least squares recovers the true moment tensor to within about 0.01 per normalised component.
- **MT maths:** `mt.sdr` returns the input strike/dip/rake and agrees with obspy's `mt2plane` applied to `mt.to_use`. The lune conversions and the DC sampler give exact round-trips.
- **Doppler stub:** `MTSpectrumModel.forward` with `vr = 0` equals a symmetric Brune filter. With rupture to the north, the highest apparent corner frequencies are at northward receivers.
- **Real data:** on cached event `data_out/20231022013049927909/*_MT.pkl` (398 channels), `unshake.least_squares` matches `SeisSrcInv.inversion.perform_inversion` exactly, covariance included.
- **Speed:** `monte_carlo` with 10,000 samples takes about 5.5 s on that event, using matmul-based synthetics and batches of 1000.

### Not yet run
The cross-check cells (section 3) appended to `example_crevasse_iq_joint_inv_testing.ipynb` need the notebook's in-memory state from sections 1–2.3, and have not been run.

### Deviations from the plan
- **No `n_jobs` option in `monte_carlo`.** numpy/BLAS threading does the parallel work instead.
- **Uniform random orientations.** They are now drawn uniformly over SO(3) (`scipy` `Rotation.random`). SeisSrcInv used two angles only, which does not cover all orientations uniformly.
- **`sample_lune` defaults to SeisSrcInv behaviour.** That is uniform in γ and δ; `equal_area=True` samples uniformly by area instead.

## Findings and suggested next steps

1. **Add an MCMC sampler: highest priority for milestone 2.**
   - With a proper Gaussian likelihood, the effective sample size from independent Monte Carlo is tiny: about 1–4 out of 10,000 on real data, and about 1 out of 50,000 on synthetics. The posterior is far too peaked for prior sampling.
   - SeisSrcInv's likelihood (`misfit='vr'`, `-(1 - sim)/2`) has the opposite problem: it is almost flat, with ESS ≈ n.
   - Add `solvers.mcmc(model, dataset, ...)` using `make_loglike`, for example with emcee. Options for the starting point:
     - start at the least-squares solution;
     - parametrise the MT with unconstrained Gaussian components for the full MT;
     - use lune + orientation angles.
   - This is needed before the joint MT + fc space of about 10 parameters is tractable.
2. **Resolve the Ricker source-time function in the Salvus GFs before fitting corner frequencies on real data.** Either:
   - regenerate the GFs with a broadband or impulsive STF; or
   - divide out the Ricker using `forward.apply_source_spectrum(..., reference=lambda f: spectra.ricker_spectrum(f, f0))`. This hook is implemented, and the water level is crude.
3. **Flesh out `MTSpectrumModel`.**
   - Decide on a phase-dependent `c`, using vp or vs per receiver/window.
   - Define the bilateral and asymmetric rupture forms, which are currently only simple placeholders.
   - Decide on the priors for `fc0`, `vr` and rupture direction.
   - Consider fitting spectra directly (an amplitude-spectrum misfit) as well as waveforms.
4. **Check `mt_ned2use` in the notebooks.**
   - The notebook's version gives `[dd, nn, ee, -ed, -ne, nd]`.
   - The standard NED → USE conversion (`unshake.mt.to_use`, confirmed against obspy) is `[dd, nn, ee, nd, -ed, -ne]`.
   - Earlier notebook beachballs may therefore have been plotted with the wrong orientation.
5. **Investigate the poor fit of the cached event.** Event `20231022013049927909` has a least-squares variance reduction of only 0.001. Check alignment, time shifts and the windowing before using it as a test case.
6. **Smaller items:**
   - Run the notebook cross-check cells, and use the saved `Dataset` pickle for further development without redoing the prep.
   - Consider moving time-shift picking into `Dataset`, or a small `prep` helper, once it stabilises.
   - Add parallel batches only if BLAS threading proves insufficient.
   - Add formal tests once the joint inversion works. The synthetic example and the sanity checks above are the natural starting point.
7. **Later milestones:** finite and moving sources (a sub-source axis in `gf` plus `FiniteSourceModel`), as described under "Extension path".

### Update 2026-10-23
- **Added LSQ uncertainty over the whole lune:**
  - `solvers.lsq_loglike` and `solvers.lsq_lune_profile`, exported at package level;
  - `plot.lune(values=..., stat=..., vmin=..., colorbar=...)`.
- **Notebook default.** The notebook's LSQ lune uses fast covariance sampling by default; set `plot_full_lune_lsq = True` for the slower full-lune profile.
- **Extra install.** `unshake` is also installed editable in `/Users/tomhudson/miniforge3/envs/salvus_src_inv`, the notebook kernel.
- **Next step 2 (Ricker STF)** continues in `plans/2026-10-23-handoff-ricker-stf.md`.
