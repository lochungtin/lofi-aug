"""Command line: ``lofi profiles | corrupt | measure | calibrate``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from lofi import calibrate, io
from lofi import profile as profiles
from lofi.augment import resample_label
from lofi.corrupt import corrupt
from lofi.metrics import measure


def _device(name: str | None) -> torch.device:
    return torch.device(name or ("cuda" if torch.cuda.is_available() else "cpu"))


def _corrupt(args: argparse.Namespace) -> None:
    profile = profiles.load(args.profile)
    device = _device(args.device)
    generator = torch.Generator(device).manual_seed(args.seed)
    image, spacing = io.read(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).name.removesuffix(".gz").removesuffix(".nii")
    for i in range(args.n):
        result = corrupt(image.to(device), spacing, profile, generator)
        io.write(out / f"{stem}_lofi{i:02d}.nii.gz", result.image, result.spacing)
        if args.label:
            label = io.read(args.label)[0].to(device)
            kept = resample_label(label, result.fov.shape, result.padding) * result.fov
            io.write(out / f"{stem}_lofi{i:02d}_label.nii.gz", kept.short(), result.spacing)
    print(f"wrote {args.n} scan(s) to {out}")


def _measure(args: argparse.Namespace) -> None:
    device = _device(args.device)
    stats = []
    for path in args.images:
        image, spacing = io.read(path)
        stats.append(measure(image.to(device), spacing))
    print(json.dumps(calibrate.summarise(stats), indent=2))


def _calibrate(args: argparse.Namespace) -> None:
    device = _device(args.device)
    stats = []
    for path in args.target:
        image, spacing = io.read(path)
        stats.append(measure(image.to(device), spacing))
    target = calibrate.summarise(stats)
    profile = calibrate.initial(profiles.load(args.start), target)
    if args.sources:
        scans = [(s[0].to(device), s[1]) for s in map(io.read, args.sources)]
        profile, history = calibrate.fit(profile, target, scans, args.iterations)
        for step, record in enumerate(history):
            print(step, {k: round(v, 3) for k, v in record.items()})
    profiles.save(profile, args.out)
    print(f"wrote {args.out}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lofi", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("profiles", help="list bundled profiles")

    run = commands.add_parser("corrupt", help="corrupt one NIfTI scan")
    run.add_argument("image")
    run.add_argument("--label")
    run.add_argument("--profile", default="low_field_breast")
    run.add_argument("--n", type=int, default=1)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--out", default=".")
    run.add_argument("--device")
    run.set_defaults(run=_corrupt)

    stats = commands.add_parser("measure", help="print scan statistics as JSON")
    stats.add_argument("images", nargs="+")
    stats.add_argument("--device")
    stats.set_defaults(run=_measure)

    fit = commands.add_parser("calibrate", help="fit a profile to real target scans")
    fit.add_argument("--target", nargs="+", required=True)
    fit.add_argument("--sources", nargs="*", default=[])
    fit.add_argument("--start", default="low_field_breast")
    fit.add_argument("--iterations", type=int, default=4)
    fit.add_argument("--out", required=True)
    fit.add_argument("--device")
    fit.set_defaults(run=_calibrate)

    args = parser.parse_args(argv)
    if args.command == "profiles":
        print("\n".join(profiles.bundled()))
    else:
        args.run(args)
