# brkraw-sordino

SORDINO-ZTE reconstruction hook for BrkRaw.

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

Once installed, `brkraw` applies the hook automatically when a dataset matches the rule.

Basic conversion:

```bash
brkraw convert /path/to/study --scan-id 3 --reco-id 1
```

The hook behaves the same whether invoked via the CLI or via the Python API (the same hook entrypoint and arguments are used).

To explicitly pass hook options (or override defaults), use `--hook-arg` / `--hook-args-yaml` below.

## Hook options

Hook arguments can be passed via `brkraw convert` using `--hook-arg` with the
entrypoint name (`sordino`):

```bash
brkraw convert /path/to/study -s 3 -r 1 \
  --hook-arg sordino:ext_factors=1.2 \
  --hook-arg sordino:offset=2 \
  --hook-arg sordino:split_ch=false
```

### Pass hook options via YAML (`--hook-args-yaml`)

BrkRaw can also load hook arguments from YAML. Generate a template like this:

```bash
brkraw hook preset sordino -o hook_args.yaml
```

Edit the generated YAML, then pass it to `brkraw convert` (repeatable):

```bash
brkraw convert /path/to/study -s 3 -r 1 --hook-args-yaml hook_args.yaml
```

Example:

```yaml
hooks:
  sordino:
    ext_factors: 1.2
    offset: 2
    split_ch: false
    # as_complex: true  # optional, return (real, imag)
    # cache_dir: ~/.brkraw/cache/sordino  # optional (add manually if needed)
```

Notes:

- CLI `--hook-arg` values override YAML.
- YAML supports both `{hooks: {sordino: {...}}}` and `{sordino: {...}}` shapes.
- You can also set `BRKRAW_CONVERT_HOOK_ARGS_YAML` (comma-separated paths).

Supported keys:

- `ext_factors`: scalar or 3-item sequence (default: 1.0)
- `ignore_samples`: int (default: 1)
- `offset`: int (default: 0)
- `num_frames`: int or null (default: None)
- `correct_spoketiming`: bool (default: false)
- `correct_ramptime`: bool (default: true). One switch for the ramp-time
  correction. On: each sample is placed at the time integral of the ramping
  gradient from the RF centre (with the ramp window of the sequence version:
  v1 `mjm_zte`, v2 `sordino_260122_trig`, v3 `sordino`, or a general ZTE with
  a constant gradient) and the per-projection phase that the FOV-offset
  frequency (`ACQ_O1_list`) leaves while the gradient ramps is removed. Off:
  every spoke keeps one fixed gradient vector and no phase correction.
  Timing values and adjustments are declared in `timing.py`.
- `estimate_k0`: bool (default: false). Estimates the k-space centre samples
  that the dead time leaves unmeasured, by a least-squares image (one FOV
  grid, 10 conjugate-gradient iterations), and adds them to the adjoint
  reconstruction. SORDINO v1-v3 only; on a general ZTE it is ignored with an
  info message. It stops with an error when `correct_ramptime` is false. It
  changes the image noticeably, no ground truth exists yet, and it makes the
  reconstruction slower, so it is off by default.
- `offreso_freqs`: float or list of floats in Hz, one per receive channel
  (default: none).
- `mem_limit`: float (default: 0.5)
- `clear_cache`: bool (default: true)
- `split_ch`: bool (default: false, merge channels)
- `as_complex`: bool (default: false, return complex as (real, imag))
- `cache_dir`: string path (default: ~/.brkraw/cache/sordino)

Boolean values are read case-insensitively: `true`, `True`, `TRUE`, `false`,
`False` (also `1`, `0`, `yes`, `no`, `on`, `off`), from YAML or from
`--hook-arg`; any other value is an error. A key the hook does not know
(including the removed `ramp_model` and `correct_phase`, which have no alias)
is ignored with a one-line warning.

## Notes

- The hook reconstructs data using an adjoint NUFFT and returns magnitude images by default.
- The k-space centre inside the dead time is not sampled and, unless `estimate_k0` is on, not filled. Its radius
  (k-grid units, first acquired and first kept sample) is logged (info for a general
  ZTE gap over one unit, debug otherwise) and kept as `scan._sordino_recon_meta["kspace_gap"]`
  and in the recon cache `.json`. With `estimate_k0`, `scan._sordino_recon_meta["k0"]` and the
  cache `.json` also hold the estimated K0 (`[real, imag]` per channel, for every frame).
- Converted NIfTI outputs apply slope/intercept scaling for uint16 storage.
- `ext_factors` scales the affine around the FOV center during conversion.
- Multi-channel data defaults to merged channels; set `split_ch=true` to keep channels split.
- When `split_ch=false`, magnitude uses RSS while complex uses coherent sum.
- Orientation is normalized when the first 3D axes are spatial; see `notebooks/orientation.ipynb`.
- Cache files live under `~/.brkraw/cache/sordino` (or `BRKRAW_CONFIG_HOME`) and are cleared when `clear_cache=true`.
