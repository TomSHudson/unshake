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
```

## Architecture (milestone 1)

```
src/unshake/
  data.py     Dataset (+ Receivers, Source): waveforms, GFs, geometry, metadata for one event
  mt.py       MT convention (NED [nn, ee, dd, ne, nd, ed]), conversions, lune, strike/dip/rake, samplers
  spectra.py  source spectra (Brune/Boatwright) and Doppler apparent corner frequency
  forward.py  synthetics = GF x source; frequency-domain source-spectrum filter
  misfit.py   VR, PCC, NCC, Gaussian log-likelihood (vectorised over samples)
  models.py   source models (MTModel, ForceModel, MTSpectrumModel stub): params -> synthetics + prior
  solvers.py  least_squares, monte_carlo (pluggable: any model, any sampler)
  results.py  Result: samples, log-likelihoods, posterior weights, save/load
  plot.py     lune, beachball, wavefield / trace fits, fc vs azimuth
```

New physics = a new model class in `models.py`; new inference = a new function in `solvers.py`.
Plans live in `plans/`; a synthetic example is in `examples/synthetic_mt_recovery.py`.

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