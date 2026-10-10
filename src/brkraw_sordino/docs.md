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
  that the dead time leaves unmeasured, by a least-squares image (one channel
  at a time, 10 conjugate-gradient iterations), and adds them to the adjoint
  reconstruction. SORDINO v1-v3 only; on a general ZTE it is ignored with an
  info message. It stops with an error when `correct_ramptime` is false. It
  changes the image noticeably, no ground truth exists yet, and it makes the
  reconstruction slower, so it is off by default. The least-squares solve has
  two forms that give the same image and K0 within the NUFFT tolerance. The
  Toeplitz solve works on grids only (memory about 1 KiB per output voxel plus
  64 MiB, whatever the sample count) and is the faster one. The sample solve
  works at the samples (about 128 B per voxel plus 80 B per sample of a frame
  plus 16 MiB). There is no option to choose: the hook picks the sample solve
  only when its whole reconstruction estimate is at most half of the Toeplitz
  one (a frame with few samples for its grid), or when the Toeplitz solve does
  not fit the memory limit even with the smallest chunk and the sample solve
  does; otherwise it uses the Toeplitz solve. An info log line names the solve
  and the estimates. `get_dataobj_info` reports the choice (below).
- `offreso_freqs`: float, list of floats or a string such as `"120,-80"` in Hz,
  one per receive channel; every form reads as the same values and a value that
  is not a finite number is refused (default: none).
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
- `allow_short_fid`: bool (default: true). A scan stopped before its last
  repetition leaves a FID shorter than `PVM_NRepetitions` frames need. The
  hook measures the FID first; when it is short, it logs one warning (bytes
  found, bytes the parameters need, bytes missing, complete frames out of the
  planned ones) and reconstructs the complete frames only, so the output has
  fewer frames than the parameters say (the NIfTI frame count is the number
  of frames returned; voxel size, affine and TR are unchanged). The bytes of
  the incomplete last frame are not used. `offset` and `num_frames` count
  within the complete frames. It stops with a `ValueError` when no frame is
  complete or `offset` is at or after the last complete frame. With `false`
  a short FID stops with a `ValueError` before anything is reconstructed, as
  earlier versions stopped (they stopped during the reconstruction). A FID of
  the planned size or longer is read as before.

`brkraw_sordino.get_dataobj_info(scan, reco_id, **options)` returns the same
estimate without reading data, for callers that decide before loading. It
takes the same options as `get_dataobj` (`frames`, `axis`, `max_memory_gb`, the
reconstruction options) and returns a dict with these keys:

- `shape`, `dtype`, `count`, `nbytes`: shape and real dtype string of each
  returned array, how many arrays are returned (channels when `split_ch`, times
  2 with `as_complex`), and the bytes of all of them.
- `frames`, `frames_reconstructed`: frames returned, and frames in the
  reconstruction (and its cache).
- `cached`: a valid recon cache exists. `cache_path`, `cache_dir`,
  `cache_dtype` (complex128 until a cache exists) and `cache_nbytes` describe
  the cache file.
- `peak_nbytes`: the memory estimate that is compared with the limit (the
  returned arrays, three cache frames and, without a cache, `recon_nbytes`).
- `recon_nbytes`, `recon_chunk_spokes`, `recon_chunks`: the reconstruction
  step's estimate, spokes per chunk and number of chunks. Without a cache
  these are the planned values; with a cache `recon_nbytes` is 0 and the other
  two are null.
- `recon_k0_method`: `"toeplitz"` or `"samples"` with `estimate_k0` and no
  cache, otherwise null. `recon_k0_reason`: why it was chosen, `"rule"` (the
  half rule above) or `"limit"` (only the sample solve fits the memory limit),
  null when `recon_k0_method` is null.
- `disk_nbytes`, `disk_free_nbytes`: disk a new cache needs (0 when cached)
  and the free space in `cache_dir` (null when cached or unreadable).
- `limit_nbytes`, `limit_source`: the memory limit that `get_dataobj`
  applies, and where it comes from (`max_memory_gb option`,
  `half of physical memory` or `fallback 4 GB`).
- `frames_planned`, `fid_short_nbytes`: frames in the parameters
  (`PVM_NRepetitions`), and the bytes the FID is short of them (0 when the FID
  is complete, null when its size or the frame count cannot be read). With a short FID,
  `frames_reconstructed` is the number of complete frames.

Boolean values are read case-insensitively: `true`, `True`, `TRUE`, `false`,
`False` (also `1`, `0`, `yes`, `no`, `on`, `off`), from YAML or from
`--hook-arg`; any other value is an error. A key the hook does not know
(including the removed `ramp_model` and `correct_phase`, which have no alias)
is ignored with a one-line warning.

## Golden trajectories and frames

The `sordino` sequence (from `sordino_260801`) chooses the spoke directions with
the method parameter `TrajectoryMode`, and the hook reads it; there is no option:
`Default` (the radial list), `GoldenSampling` (`goldensamp.c`: golden means cut
into subsets of `NGoldenSpokesPerSubset` spokes, reordered in z-stacks) and
`GoldenGridSampling` (`goldengrid.c`: one spoke per equal-area cell per grid
frame, plus the opposite spoke with `GridMirror`). Older data without the key are
`Default`. The spoke order is checked against `ACQ_O1_list` (a warning when it
does not match).

The golden lists do not use the Default-trajectory parameters ProUnderSampling,
`Reorder`, `HalfAcquisition` or `DirectMode`. A golden protocol may still carry an
old value (the phantom scans 11 and 17 of WI-0112 keep ProUnderSampling 3.6147
from an earlier Default setting); it changes nothing here. ParaVision's online
reconstruction (pdata/1) does use it for Golden Grid: the first 12,732 lines of
scan 17's `traj` file are the Default list for that value, so pdata/1 of a golden
scan is not a reference image. In the sequence version of those scans
(`sordino_260801`) the data show that the same list also reached the
acquisition: the first 12,732 spokes of the Golden Grid scans (17, and 21 of the
same session) fit the Default directions with the golden `ACQ_O1_list`
frequencies, not the golden list; from spoke 12,732 on they fit the golden list
(inferred from the data; ParaVision's internal order is not documented). The
hook reconstructs every spoke along the golden list, so these first spokes are
misplaced (4.4 % of such a scan; grid frames 0-49 of a frame series). They look
like a movement of a few mm in the first 8 s; it is not motion.

Frames (golden scans; a Default scan with a spoke count):

- A Default scan keeps one repetition per frame unless `frame_spokes` is a spoke
  count (`subset` is refused: the Default list has no method subset). Its list goes
  through the sphere in order, so a frame shorter than a repetition covers only part
  of the sphere (a warning); accumulation then shows the coverage growing.

- `frame_spokes`: spokes per frame. Default: the method subset, the smallest
  count that covers the sphere (`NGoldenSpokesPerSubset`; one grid frame, cells
  x 2 with `GridMirror`). `subset` says the same, `repetition` gives one
  repetition per frame (as a Default scan), an integer is a spoke count.
- `frame_step`: spokes between frame starts (default: `frame_spokes`). Smaller
  gives a sliding window; larger leaves spokes out.
- `frame_accumulate`: every frame starts at the first spoke; frame k holds
  `frame_spokes` + k x `frame_step` spokes (default step `frame_spokes`: 1, 2, 3
  ... x `frame_spokes`). The start does not move. The last accumulated frame always
  holds every spoke read, also when the count is not a multiple of the step.
- The repetitions read (`offset`, `num_frames`) form one stream of spokes, so
  windows, sliding windows and accumulation run across repetition boundaries
  (an info line counts such frames).
- Any spoke range can be reconstructed. A frame that starts or ends inside a
  method subset gets a warning (its directions cover the sphere less evenly:
  3,200 spokes of scan 11 had a largest gap of 4.50 degrees from inside a
  subset, 3.58 degrees from a subset start).
- Each frame has the brightness of a one-repetition image (scaled by NPro over
  its spoke count); the frames are the 4th axis of the result and of the NIfTI.
- NIfTI time step: pixdim[4] is `frame_step` x the spoke TR (`PVM_RepetitionTime`),
  in the time unit asked for (with accumulation: the growth between frames).
  The frame list (spoke ranges, frame centre times, interval) is in
  `scan._sordino_recon_meta["frames"]`.
- `estimate_k0` with frames: the k-space centre is estimated once from all
  spokes read and used by every frame. A window estimate is far too low (scan
  11: 3-4 % of the all-spoke value from 160 spokes, 26-29 % from 3,200). With
  several repetitions this also removes repetition-to-repetition changes of the
  filled centre. If the object moves during the scan, the one value is the
  average over its positions.
- `correct_spoketiming` cannot be combined with frames (it moves every spoke of
  a repetition to one time point); use `frame_spokes: repetition` with it.
- Size: the recon cache holds one complex128 volume per frame. The default for
  scan 11 (288,000 spokes, 160-spoke subsets, 120^3) is 1,800 frames: 49.8 GB of
  cache (46.4 GiB), 24.9 GB as float64 magnitude in memory, 6.2 GB as a uint16
  NIfTI. The size check runs before anything is reconstructed and stops with
  the sizes; ask for larger frames, fewer repetitions (`num_frames`) or
  `frame_spokes: repetition`. A sliding window also holds one copy of the
  running sum per open frame start (`frame_spokes` / `frame_step` copies).

```bash
brkraw convert /path/to/study -s 11 -r 1 \
  --hook-arg sordino:frame_spokes=3200 \
  --hook-arg sordino:frame_step=1600
```

```yaml
hooks:
  sordino:
    frame_spokes: 3200      # 20 subsets of 160 spokes, 2 s at a TR of 0.625 ms
    frame_accumulate: true  # frames of 3,200, 6,400, 9,600 ... spokes
```

## Notes

- The hook reconstructs data using an adjoint NUFFT and returns magnitude images by default.
- The k-space centre inside the dead time is not sampled and, unless `estimate_k0` is on, not filled. Its radius
  (k-grid units, first acquired and first kept sample) is logged (info for a general
  ZTE gap over one unit, debug otherwise) and kept as `scan._sordino_recon_meta["kspace_gap"]`
  and in the recon cache `.json`. With `estimate_k0`, `scan._sordino_recon_meta["k0"]` and the
  cache `.json` also hold the estimated K0 (`[real, imag]` per channel, for every frame;
  with golden frames one estimate for all frames).
- A short FID (see `allow_short_fid`) is kept with the result as
  `scan._sordino_recon_meta["short_fid"]` and in the recon cache `.json`:
  `fid_nbytes`, `expected_nbytes`, `frame_nbytes`, `frames_planned`,
  `frames_complete` (null for a complete FID). Its recon cache is separate
  from the cache of the complete scan.
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
