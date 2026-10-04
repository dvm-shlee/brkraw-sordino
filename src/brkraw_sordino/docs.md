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

- `ext_factors`: scalar or 3-item sequence (default: 1.0). Enlarges the
  reconstruction grid, and so the field of view, at the same voxel size: axis `i`
  gets `int(Matrix[i] * ext_factors[i])` voxels (16 x 1.5 -> 24, 16 x 1.1 -> 17).
  The three values follow the **read/phase/slice acquisition order** of
  `PVM_Matrix` (the reconstruction axes): index 0 = read, 1 = phase, 2 = slice.
  They are not anatomical axes and not the axes of the output array: the hook
  reorders the axes after reconstruction, so the anatomical direction each index
  widens depends on the scan (slice orientation, read direction, subject type and
  position). Measured example, two approved coronal scans (readout `H_F`,
  Quadruped, `Head_Supine`, written in brkraw's default `subject_ras`): index 0
  (read, along the bore, which is the animal's posterior-anterior axis for a
  quadruped lying head first) widens posterior-anterior, index 1 (phase)
  left-right, index 2 (slice, vertical in the magnet) inferior-superior; the output array holds index 1 on its axis 0
  and index 0 on its axis 1. Check the mapping of your own protocol once (for
  example with `[1.5, 1, 1]` and a look at which side grew) before relying on it.
  A scalar or three equal values widen every axis, so the order does not matter
  for them.
  Naming caution: "RPS" is sometimes used for read/phase/slice, but in
  neuroimaging RPS is also an orientation code (Right-Posterior-Superior). The
  order here is the acquisition order, never an anatomical code.
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

Read-time options (they choose what is returned from the reconstruction
cache and are not part of the cache key):

- `frames`: which reconstructed frames to return, with the brkraw rules: an int
  picks one frame and removes the frame axis, a list keeps the axis in that
  order, `"start:stop[:step]"` is a Python slice. Frames count the
  reconstructed frames (0 is frame `offset`). Only the selected frames are
  read from the cache. The whole scan (or `num_frames` from `offset`) is still
  reconstructed when no cache exists.
- `axis`: optional; the frame axis is data axis 3 (`3`, `-1`, `"cycle"` or
  `"repetition"`). `axis` without `frames` is an error.
- `max_memory_gb`: float (default: half of this computer's physical memory,
  4 GB when it cannot be read). Before it reconstructs or reads, the hook
  estimates the memory of the returned data and the disk space of a new
  cache, and stops with `SordinoResourceError` (a `MemoryError`) when the
  memory estimate is above this limit or the cache does not fit on the disk.
  Nothing is reconstructed or read in that case; ask for fewer frames or raise
  the limit. The error's `retry_kwargs` holds the smallest limit (0.1 GB steps)
  that would pass, for example `{"max_memory_gb": 1.8}`; the brkraw CLI uses it
  to ask "Proceed anyway?" in a terminal. The estimate covers the returned
  arrays and the read buffers, and, when no cache exists, the reconstruction
  step that runs first in the same process (`recon_nbytes`). The
  reconstruction goes through each frame in contiguous chunks of spokes (at
  most about 13 M samples each), so its memory is a fixed part (the image of
  every receiver and one NUFFT grid; with `estimate_k0` also its
  least-squares solve) plus one chunk, and does not grow with the number of
  spokes or frames. The `estimate_k0` solve runs either on grids (about 1 KiB
  per output voxel, whatever the sample count) or at the samples (about 128 B
  per voxel plus 80 B per sample of a frame). The grid solve is faster (1.2 to
  8 x on the measured shapes), so the sample solve is used only when it at
  most halves the whole estimate, that is when a frame has few samples for its
  grid, or when only the sample solve fits the memory limit (for example a
  160^3 scan with few spokes on an 8 GB computer); an info log line names the
  choice. Both give the same image and K0
  within the NUFFT tolerance. Example: a 128^3 grid with 12,800 x 64 samples
  per frame measured 0.57 GiB with the sample solve (2.08 GiB on grids); a
  64^3 grid with the same samples keeps the grid solve (0.42 GiB, 0.2 s per
  frame, against 0.25 GiB and 0.8 s). What the limit leaves after
  the read buffers sets the chunk size; the check stops only when even a
  256-spoke chunk does not fit. With spoke-timing correction the estimate is
  the larger of that and 5 x one FID segment, whose size follows `mem_limit`.
  Example: a 160^3 scan with NPro 80876, NPoints 640 (OverSampling 8) and 2
  receivers (synthetic FID, macOS, default limit) measured 3.0 GiB (estimate
  4.1 GiB; 18.9 GiB before the chunked reconstruction), and 4.9 GiB with
  `estimate_k0` (estimate 8.1 GiB; 27.3 GiB before). On an 8 GB computer
  (4 GiB limit) the plain reconstruction fits with smaller chunks; with
  `estimate_k0` it stops and offers a limit of 5.3 GB. The estimate is at or
  above every measurement, by up to 2 x.

`brkraw_sordino.get_dataobj_info(scan, reco_id, **options)` returns the same
estimate without reading data (shape, dtype and count of the returned arrays,
bytes, whether a valid cache exists, cache size, memory estimate and limit),
for callers that decide before loading.

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
- `ext_factors` keeps every object where it was: the affine keeps the voxel size and
  direction and moves the origin by `-(N // 2 - N0 // 2)` voxels on each output axis
  (`N` the extended size, `N0` the size at 1.0; the adjoint NUFFT puts the grid centre at
  index `N // 2` for odd and even sizes). When `N` and `N0` differ in parity the extra voxel
  sits on one side, so the FOV centre moves by half a voxel while the objects do not.
  Before this fix, a factor other than 1 on index 0 or 1 moved the image when the output
  reorders those axes (measured: 1.6 mm on a 16^3 scan, 10 mm on a 60^3 scan), and a size
  that is not a whole multiple (or an odd matrix) gave a sub-voxel shift (16 x 1.1: 0.3 mm).
  `ext_factors=1` is unchanged, and the reconstruction and its cache key are unchanged
  for every value.
- Multi-channel data defaults to merged channels; set `split_ch=true` to keep channels split.
- When `split_ch=false`, magnitude uses RSS while complex uses coherent sum.
- Orientation is normalized when the first 3D axes are spatial; see `notebooks/orientation.ipynb`.
- Cache files live under `~/.brkraw/cache/sordino` (or `BRKRAW_CONFIG_HOME`): `recon_<hash>.bin`
  (complex128 reconstruction, all reconstructed frames) with its `.json`, and `traj_<hash>.npy`
  (trajectory). `clear_cache` does not remove these files: they stay for reuse until
  `brkraw cache clear`, and a temporary file left by an interrupted run is replaced by the next run.
- The recon cache is keyed by the scan, reco, FID and every option that changes the reconstructed
  values. `as_complex`, `split_ch`, `clear_cache` and `cache_dir` are not in the key (they choose
  what is returned, what is cleaned up, or where the cache lives), so changing them reads the same
  cache instead of reconstructing again. Recon caches written before this change are not read
  (reconstructed once); remove them with `brkraw cache clear`.
- The trajectory file is keyed only by the values that generate the trajectory (gradient
  scheme, samples per spoke, and the sample offset or the ramp-model sample times), so options
  such as `ext_factors` or frames reuse it. Trajectory files written before 0.6.0 (`<md5>.npy`)
  are no longer read; they can be removed with `brkraw cache clear`.
