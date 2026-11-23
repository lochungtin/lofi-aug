# LOFI

**Lo**w-**Fi**eld MRI corruption. LOFI makes good scans look like low-field open-magnet scans, for
training models that must work on the bad ones. It only changes intensities, so anatomy stays where
it is and labels stay exact.

[![Branch CI](https://github.com/lochungtin/lofi-aug/actions/workflows/branch-ci.yml/badge.svg)](https://github.com/lochungtin/lofi-aug/actions/workflows/branch-ci.yml)

> Status: pre-release.

## Install

```bash
pip install lofi-aug            # core
pip install "lofi-aug[io]"      # NIfTI support (needed by the command line)
```

Other extras: `viz`, `monai`, `torchio`.

## How it works

A forward model runs in acquisition order, so the artefacts emerge rather than being painted on:

1. **Normalise and map contrast.** The foreground median becomes 1. An optional monotonic map moves
   the tissue histogram to the target's.
2. **Frame.** Pad to the scanner's canvas.
3. **Select slices.** Box-average thick slabs. Each acquisition's slabs can be displaced.
4. **Shade.** Multiply by a smooth receive sensitivity.
5. **Sample k-space.** Add a smooth phase, then crop k-space to the acquired matrix with a taper.
6. **Corrupt k-space.** Add line-by-line phase errors (ghosting), optional EMI tones, then complex
   noise, less what the source already carries at that voxel size.
7. **Magnitude.** This is where the Rician floor comes from.
8. **Resample and combine.** Linear onto the output grid, then average the acquisitions.
9. **FOV.** Warped edge, exact zeros outside it, and a fade towards it.

Scanner-specific choices live in profiles. `low_field_breast` is bundled and fitted to real scans.

## Command line

```bash
lofi profiles
lofi corrupt good.nii.gz --label good_label.nii.gz --profile low_field_breast --n 4 --out out/
lofi measure scan_*.nii.gz
lofi calibrate --target bad_*.nii.gz --sources good_*.nii.gz --out my_scanner.yaml
```

## Python

```python
from lofi.augment import LowFieldCorrupt

corrupt = LowFieldCorrupt("low_field_breast", seed=0)
sample = corrupt({"image": image, "label": label, "spacing": (0.8, 0.8, 1.1)})  # (C, *spatial)
```

The result also holds `"fov"`, and `"spacing"` becomes the output grid's. Labels are moved onto that
grid with the image and set to background outside the FOV.

Framework adapters keep the input grid, since those pipelines fix it themselves. Apply them before
random cropping:

- MONAI: `lofi.integrations.monai.LowFieldCorruptd(["image", "label"], "low_field_breast")`.
- TorchIO: `lofi.integrations.torchio.RandomLowField("low_field_breast")`.

With raw k-space and a noise-only scan, `lofi.physics.corrupt_kspace` applies the field-strength
model directly, including the separate scaling of coil and sample noise.

## Development

```bash
uv sync --extra dev --extra io
uv run pytest
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

`CLAUDE.md` records the conventions and the design as built. The research behind the package
(calibration study, research notes, downstream segmentation studies) is kept outside it, in the
`_lofi` folder of `lochungtin/PhD_Codebase`.

## Licence

MIT.
