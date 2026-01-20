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
- `correct_ramptime`: bool (default: true)
- `offreso_ch`: int or null (default: None)
- `offreso_freq`: float (default: 0.0)
- `mem_limit`: float (default: 0.5)
- `clear_cache`: bool (default: true)
- `split_ch`: bool (default: false, merge channels)
- `as_complex`: bool (default: false, return complex as (real, imag))
- `cache_dir`: string path (default: ~/.brkraw/cache/sordino)

## Notes

- The hook reconstructs data using an adjoint NUFFT and returns magnitude images by default.
- Converted NIfTI outputs apply slope/intercept scaling for uint16 storage.
- `ext_factors` scales the affine around the FOV center during conversion.
- Multi-channel data defaults to merged channels; set `split_ch=true` to keep channels split.
- When `split_ch=false`, magnitude uses RSS while complex uses coherent sum.
- Orientation is normalized when the first 3D axes are spatial; see `notebooks/orientation.ipynb`.
- Cache files live under `~/.brkraw/cache/sordino` (or `BRKRAW_CONFIG_HOME`) and are cleared when `clear_cache=true`.
