# Changelog

## Initial development

First version: low-field MRI corruption for training segmentation models on good scans that must
work on low-field open-magnet ones. Only intensities change, so labels stay exact.

### Forward model
- Stages in acquisition order (`corrupt`): normalisation to the foreground median, monotonic
  contrast mapping blended by a tissue weight, canvas framing, thick-slab slice selection with
  per-acquisition slab offsets, smooth receive shading, k-space crop with taper and FOV-aligned
  phase (`kspace`), line-by-line phase errors (ghosting), optional EMI tones, complex noise on the
  acquired grid less the noise the source already carries, magnitude (Rician floor), linear
  resampling, averaging over acquisitions, and a warped FOV with exact zeros outside.
- Slice-to-slice contrast from crosstalk in 2D multi-slice spoiled gradient echo (`crosstalk`).
- Raw k-space route with field-strength scaling that keeps coil and sample noise apart
  (`physics.corrupt_kspace`, `physics.scale_covariance`).

### Profiles and calibration
- Strict YAML profiles: unknown keys raise; `[low, high]` ranges are drawn per sample.
- `low_field_breast` bundled, fitted to 0.5 T open-magnet breast scans.
- Image statistics (`metrics`, `masks`) and calibration (`calibrate`): acquisition geometry comes
  from the spectra and the protocol; noise and shading are fitted by corrupting good scans until
  their statistics match the target's.

### Integrations
- `LowFieldCorrupt` (dict transform; labels follow the image onto the output grid), MONAI
  (`LowFieldCorruptd`), TorchIO (`RandomLowField`).
- CLI: `lofi profiles | corrupt | measure | calibrate`.

## v0.1.0 (2026-11-23)

### Feat

- initial lofi-aug package [skip ci]
