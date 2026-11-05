# brkraw-sordino

SORDINO reconstruction hook for BrkRaw. SORDINO is a
zero-echo-time (ZTE) radial 3D acquisition from Bruker ParaVision; its raw data are k-space
samples along radial spokes, not a Cartesian image. This hook reconstructs them when BrkRaw
reads or converts a matching scan: it computes the spoke trajectory from the method
parameters (default radial, golden sampling and golden grid lists, with the gradient-ramp
timing of the sequence), corrects ramp-time phase, fills the k-space centre that the dead time
leaves unmeasured, and reconstructs with an adjoint NUFFT. It returns magnitude images by
default (complex on request), merges receive channels, and writes NIfTI files with the affine
BrkRaw gives the scan.

## Install

```bash
pip install -e .
```

## Hook install

```bash
brkraw hook install brkraw-sordino
```

This installs the hook rule from the package manifest (`brkraw_hook.yaml`).

## Usage

Once installed, BrkRaw applies the hook automatically when a dataset matches the rule. The
hook behaves the same through the CLI and the Python API (same entrypoint, same arguments).

```bash
brkraw convert /path/to/study --scan-id 3 --reco-id 1
```

To pass options, for example a larger field of view, use `--hook-arg` or `--hook-args-yaml`:

```bash
brkraw convert /path/to/study -s 3 -r 1 --hook-arg sordino:ext_factors=1.2
```

## More

- Options and defaults, one line each: `brkraw hook docs sordino`
  (the file `src/brkraw_sordino/docs.md`).
- Algorithms and equations (trajectories, density weights, k-space centre estimate, frames,
  affine, caches) and, in its Appendix A, every option in detail with the memory and disk
  cautions and YAML examples: [`docs/technical-note.md`](docs/technical-note.md).
- BIDS extension proposal draft for ZTE-based fMRI: [`docs/bep/bep-zte-fmri.md`](docs/bep/bep-zte-fmri.md).
- Release history: [`RELEASE_NOTES.md`](RELEASE_NOTES.md).
- Axis orientation of the output: `notebooks/orientation.ipynb`.
