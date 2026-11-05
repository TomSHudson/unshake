# unshake

**Full waveform seismic source inversion in Python.**

`unshake` works backwards from recorded ground shaking to the source that caused it.

> ⚠️ This package is in early planning. Functionality is under active development.

## Planned scope

- Moment tensor inversion from full waveforms
- Directivity, Doppler-shifted and asymmetric sources
- Spatially distributed (finite) sources
- Multiple-source / multi-event inversion

## Installation

```bash
pip install unshake
```

## Development install

```bash
pip install -e .            # editable install
pip install -e ".[plot]"    # + obspy for beachballs
pip install -e ".[salvus]"  # + helpers for the Salvus GF driver
```

Generating Green's functions (`unshake.salvus_fwd_driver.gen_synths_salvus`) needs
Salvus, which is not on PyPI: install it via Mondaic first. The rest of unshake
imports and runs without it.

## Architecture (milestone 1)

```
src/unshake/
  data.py     Dataset (+ Receivers, Source): waveforms, GFs, geometry, metadata for one event
  mt.py       MT convention (NED [nn, ee, dd, ne, nd, ed]), conversions, lune, strike/dip/rake,
              ISO/CLVD/DC decomposition, samplers
  spectra.py  source spectra (Brune/Boatwright) and Doppler apparent corner frequency
  stfs.py     causal source time functions (Brune, sum of two Brunes) for convolution with formal GFs
  forward.py  synthetics = GF x source; source-spectrum filter; band-limit / remove GF STF
  misfit.py   VR, PCC, NCC, Gaussian log-likelihood (vectorised over samples)
  models.py   source models (MTModel, ForceModel, MTTwoBruneModel, MTDCDirectivityModel, MTSpectrumModel stub): params -> synthetics + prior
  solvers.py  least_squares, nonlinear_least_squares, monte_carlo, mcmc (pluggable: any model, any sampler)
  results.py  Result: samples, log-likelihoods, posterior weights, save/load
  plot.py     lune, beachball, wavefield / trace fits, fc vs azimuth, spectra check
  qc.py       post-inversion QC: data-synthetic cross-correlation lags vs take-off direction
  salvus_fwd_driver.py  optional: run Salvus to generate GFs (needs Salvus)
```

New physics = a new model class in `models.py`; new inference = a new function in `solvers.py`.
Plans live in `plans/`; a synthetic example is in `examples/synthetic_mt_recovery.py`, and a
forward demo of (directional) Brune STFs on small Salvus GFs is in `examples/directional_stf_demo.ipynb`,
and the non-linear inversion of the DC-directivity model (MC, NLLS and MCMC) is in
`examples/directional_inversion_demo.ipynb`.

```python
import unshake
ds = unshake.Dataset(data=data, gf={'mt': gf}, dt=dt, receivers=labels)
m, cov = unshake.least_squares(ds)
res = unshake.monte_carlo(unshake.MTModel(kind='lune'), ds, n=10_000)
```

## License

MIT

## Developer notes

For the build:
```bash
pip install build twine
python -m build
twine upload dist/*
```