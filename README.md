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

## License

MIT

## Developer notes

For the build:
```bash
pip install build twine
python -m build
twine upload dist/*
```