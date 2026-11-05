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

## Options

User options added since 0.2.4. Pass them as `--hook-arg sordino:KEY=VALUE` or in a hook-args YAML
file. The last column is the section of [`docs/technical-note.md`](docs/technical-note.md) that
describes the option; section 10 is the frame overview. Every option is listed in
`brkraw hook docs sordino`.

| Option | Values | Default | Effect | Technical note |
|---|---|---|---|---|
| `frame_spokes` | `subset`, `repetition`, `all` or a positive spoke count | `null` | Spokes per frame; `all` gives one image from every spoke. | 10 |
| `frame_step` | positive spoke count | `null` (= `frame_spokes`) | Spokes between frame starts. | 10 |
| `frame_accumulate` | `true`, `false` | `false` | Every frame starts at the first spoke. | 10 |
| `peri_event` | `period=<s>,bin=<s>` plus optional `onset`, `start`, `bins`, `events` | `null` | Average the frames into bins of time around a periodic event. | 10.1 |
| `peri_censor` | path to a 0/1 file | `null` | Leave out event ranges whose spokes are not all usable; needs `peri_event`. | 10.1 |
| `subtract_rep_mean` | `true`, `false` | `false` | Subtract the repetition mean from every frame (needs frames and 2 or more repetitions). | 10.2 |
| `rep_pattern` | `none`, `model`, `measured` | `none` | Remove the pattern repeated in every repetition. | 10.2 |
| `anat_solver` | `adjoint`, `cg` | `adjoint` | Solver of the `frame_spokes=all` anatomical image. | 10.3 |
| `anat_cg_iter` | positive integer | `10` | CG iterations of `anat_solver=cg`. | 10.3 |
| `anat_lambda` | number >= 0 | `0.0` | Absolute Tikhonov weight of `anat_solver=cg`. | 10.3 |
| `frame_static` | `adjoint`, `cg` | `adjoint` | Solver of the static part of `rep_pattern=model` frames. | 10.4 |
| `frame_static_iter` | positive integer | `10` | CG iterations of `frame_static=cg`. | 10.4 |
| `frame_static_lambda` | number >= 0 | `0.0` | Absolute Tikhonov weight of `frame_static=cg`. | 10.4 |
| `physio` | `off`, `on` | `off` | Write respiration and heart-rate traces from the innermost spoke samples. | 10.5 |
| `physio_n_inner` | positive integer | `6` | Innermost samples per spoke used for the traces. | 10.5 |
| `physio_resp_band` | `low,high` in Hz | `0.5,5` | Respiratory band. | 10.5 |
| `physio_cardiac_band` | `low,high` in Hz | `5,15` | Cardiac band. | 10.5 |
| `physio_window_s` | `respiration,cardiac` in s (one number serves both) | `3,1` | Window of the rate estimate. | 10.5 |
| `physio_pca` | `true`, `false` | `true` | PCA trace per band (`false`: mean). | 10.5 |
| `physio_prefix` | text | `null` | Output prefix of `<prefix>_physio.tsv.gz` and `<prefix>_physio.json`. | 10.5 |
| `physio_traj_period` | `auto`, `off` or spokes (integer >= 2) | `auto` | Trajectory period removed from the traces. | 10.5 |
| `physio_traj_window` | `off` or seconds (> 0) | `1` | Window of the local cycle mean of that removal (`off`: whole repetition). | 10.5 |
| `frame_export` | `off`, `complex` | `off` | Also write complex frames, pure-noise frames and the k-space noise level. | 10.6 |
| `frame_export_prefix` | text | `null` | Output prefix of the exported files; needs `frame_export=complex`. | 10.6 |

## More

- Options and defaults, one line each: `brkraw hook docs sordino`
  (the file `src/brkraw_sordino/docs.md`).
- Algorithms and equations (trajectories, density weights, k-space centre estimate, frames,
  affine, caches) and, in its Appendix A, every option in detail with the memory and disk
  cautions and YAML examples: [`docs/technical-note.md`](docs/technical-note.md).
- BIDS extension proposal draft for ZTE-based fMRI: [`docs/bep/bep-zte-fmri.md`](docs/bep/bep-zte-fmri.md).
- Release history: [`RELEASE_NOTES.md`](RELEASE_NOTES.md).
- Axis orientation of the output: `notebooks/orientation.ipynb`.
