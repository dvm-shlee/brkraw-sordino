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
  --hook-arg sordino:rss=false
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
    rss: false
    # cache_dir: ~/.brkraw/cache/sordino  # optional (add manually if needed)
```

Notes:

- CLI `--hook-arg` values override YAML.
- YAML supports both `{hooks: {sordino: {...}}}` and `{sordino: {...}}` shapes.
- You can also set `BRKRAW_CONVERT_HOOK_ARGS_YAML` (comma-separated paths).

Supported keys:

- `ext_factors`: scalar or 3-item sequence (default: 1.0)
- `pass_samples`: int (default: 1)
- `offset`: int (default: 0)
- `num_frames`: int or null (default: None)
- `traj_offset`: float or null (default: None)
- `spoketiming`: bool (default: false)
- `ramp_time`: bool (default: false)
- `offreso_ch`: int or null (default: None)
- `offreso_freq`: float (default: 0.0)
- `mem_limit`: float (default: 0.5)
- `traj_denom`: float or null (default: None)
- `clear_cache`: bool (default: true)
- `operator`: string (default: "finufft")
- `rss`: bool (default: true)
- `cache_dir`: string path (default: ~/.brkraw/cache/sordino)

## Notes

- The hook reconstructs data using an adjoint NUFFT and returns magnitude images.
- Multi-channel data defaults to RSS combination.
- Cache files live under `~/.brkraw/cache/sordino` (or `BRKRAW_CONFIG_HOME`).
