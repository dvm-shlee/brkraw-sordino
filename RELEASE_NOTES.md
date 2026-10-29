# Unreleased (golden-sordino)

- fix: NIfTI time step for repetition frames. `pixdim[4]` is now `NPro` x TR (for example 180 s for 288,000
  spokes at 0.625 ms) when each frame is one repetition; it stayed at 1.0 before. Frame modes are unchanged.

New options for event-related golden scans (the peri-event work is by Tzu-Hao Chao; the review fixes below are
on top of it). Details and equations: `docs/technical-note.md`, section 10.1.

- `peri_event` (`perievent.py`): one image per time bin relative to an event that repeats every `period` seconds
  (`period=`, `bin=`, optional `onset=`, `start=`, `bins=`, `events=`). The frame axis is bins; `pixdim[4]` is the
  bin width.
- `peri_censor` (`perievent.py:126-151`): a 0/1 file, one row per equal block of the stream's spokes. A peri-event
  range is used only when all its spokes are usable.
- `subtract_rep_mean` (`frames.py:462-483`): every spoke is reconstructed as its deviation from the mean of its
  spoke position over the repetitions read.
- `rep_pattern` (`none`, `model`, `measured`; default `none`; `frames.py:261-271`, `frames.py:584-668`): removes the
  repetition-locked undersampling pattern from every complex frame before the magnitude. `model` subtracts the
  pattern of a forward model of the frame's own spokes, `measured` writes the `subtract_rep_mean` residual plus the
  all-spoke image. Needs a frame plan; refused with `subtract_rep_mean`; `measured` needs 2 repetitions and is
  refused with `peri_censor`. Keyed only when set. Equations: `docs/technical-note.md`, section 10.2.
- `anat_solver` (`adjoint`, `cg`; default `adjoint`), `anat_cg_iter` (default 10), `anat_lambda` (default 0) and
  the new `frame_spokes: all` (every spoke of every repetition in one image): the anatomical reference of the
  scan itself, by conjugate gradients on the normal equations of all spokes (same weights, kept samples and
  predicted centre samples as the frames) instead of the adjoint. One 3D image per channel, same affine; the
  log and the cache `.json` record the solver, iterations, lambda and the final relative residual. Keyed only
  when not `adjoint`. Equations: `docs/technical-note.md`, section 10.3.
- `frame_static` (`adjoint`, `cg`; default `adjoint`), `frame_static_iter` (default 10) and `frame_static_lambda`
  (default 0): with `rep_pattern: model` and `cg` every frame is `x_hat + g_k A_k^H W_k (d_k - A_k x_hat)`, the
  CG estimate of all spokes plus the frame's own residual, instead of the blurred adjoint image of `x_hat`.
  A static object gives every frame `x_hat`; the dynamic resolution is still the frame's. Needs
  `rep_pattern: model`. The cache `.json` records the iterations, lambda and the final residual; keyed only
  when not `adjoint`. Default frames are unchanged. Equations: `docs/technical-note.md`, section 10.4.
- Dependency pin `finufft<2.5` (`pyproject.toml:19`); the suite is run with finufft 2.4.1.

Behaviour changes in the peri-event stage:

- With `subtract_rep_mean` and `estimate_k0`, the all-spoke pass of the centre solve now uses the raw data (the
  residuals summed to zero there), and residual frames no longer get predicted centre samples added
  (`frames.py:549-555`, `frames.py:592-595`, `frames.py:633-634`).

- A peri bin that gets no spokes no longer raises: a warning names the empty bins and they are all-zero frames
  (`perievent.py:220-225`). It raises only when no spokes are selected at all (`perievent.py:218-219`).
- The default bin count is `ceil(period / bin)` instead of the floor, so the whole period is binned; a warning is
  logged when the bins run past the period and spokes are gridded in two bins (`perievent.py:179-189`).
- `subtract_rep_mean` leaves censored spokes (`peri_censor`) out of the repetition mean, so they do not leak into
  the residual (`frames.py:466-476`). A position censored in every repetition has mean 0.
- `subtract_rep_mean` or `peri_censor` without a frame plan raises a `ValueError` naming the option and what it
  needs, instead of being ignored (`frames.py:229-235`).
- The memory estimate counts every bin of a peri plan as open (`FramePlan.max_open` is `n_frames`,
  `frames.py:170-177`, used at `hook.py:715`), so a plan with many bins is refused under a small `max_memory_gb`.

Tests: two tolerances changed to match the NUFFT tolerance (`serial.NUFFT_EPS`), not a change of results: the
golden density comparison of the weights path and the product path is 1e-10 (was 1e-12;
`tests/test_golden_density.py:182-184`), and the chunked-sum comparison of the serial reconstruction is 1e-6 times
the maximum (was 1e-10; `tests/test_serial_recon.py:423-425`); both failed under finufft 2.4.1.

Documentation: measured project numbers were removed from the package docs.

# Unreleased (branch golden-weights-k0, on brkraw-060)

Documentation: `brkraw hook docs sordino` (`src/brkraw_sordino/docs.md`) is now a CLI usage page of
37 lines, 80 columns: three usage lines and one line per option with its default. The option tables, cautions,
developer checks and the size-estimate table it held moved unchanged to Appendix A of `docs/technical-note.md`.

Behaviour change, `estimate_k0` on every trajectory: `estimate_k0` and `correct_ramptime` are now
independent. With `correct_ramptime: false` the k-space centre is estimated on the straight trajectory (the virtual
samples continue each spoke's line back to the RF centre); a general ZTE (no `RampTime`) is no longer reconstructed
without it. The default stays `estimate_k0: true`, so **results with `correct_ramptime: false` and general ZTE results
change** (the centre is filled); their recon caches get a new key, and `estimate_k0: false` gives the previous result
and finds the old caches. An explicit `estimate_k0: true` with `correct_ramptime: false` is no longer an error. Every
other combination keeps its results and cache keys. The least-squares solve of the centre uses 10 iterations up to an
unsampled centre of 1.3 k-grid units (every SORDINO scan: unchanged), 40 up to 1.8 and 160 above (synthetic sweep:
with 10 the fill degrades above 1.3); the count enters the cache key only when it is not 10, and a centre wider than
2.0 k-grid units gives a warning (the reconstruction still runs). ZTE-mode scans of `sordino_260801` (`SequenceMode = ZTE`) are not detected from the header: set
`correct_ramptime: false` for them.

New option `recon_matrix`: reconstruct on a smaller matrix at the same FOV (`-H sordino:recon_matrix=60`
or `[60, 60, 60]`; one ratio on every axis, not above `PVM_Matrix`). Each spoke is cut at the new Nyquist radius (a
sphere, no taper), the density weights and `estimate_k0` work on the kept samples, and the values follow the full image's
low-pass. The affine is the one ParaVision writes for that matrix (voxels `Matrix / recon_matrix` larger, the FOV centre
kept). Leaving it out changes nothing (same results and cache keys).

Documentation: `README.md` is now the short entry (description, install, hook install,
minimal use). `brkraw hook docs sordino` (`src/brkraw_sordino/docs.md`) shows the option tables, defaults and cautions.
The algorithms, with their equations, are in `docs/technical-note.md`. `tests/test_docs.py` keeps the option tables in
step with the code.

Sequence generations: the sequences are now named 0.1 (`mjm_zte_231005`),
0.2 (`sordino_260122_trig`), 0.3 (`sordino_260801`, golden sample extension) and 1.x (`sordino`, the stable release);
zte stays. The earlier "v1/v2/v3" wording is gone from the docs and log messages. 0.3 and 1.x share one ramp model and
are told apart by the method name only (`"0.3 or 1.x"` when it says neither). `get_dataobj_info` has two new items,
`sequence_generation` and `sequence_method`, and the recon cache `.json` keeps `sequence_generation`; no existing item
changed. Internally the model values `v1`/`v2`/`v3`/`zte` stay, so trajectory caches are not rebuilt.

Behaviour changes for every scan:

- Image grid: images are now placed where the scan's affine says
  (grid shift = affine index of the FOV centre minus the NUFFT centre `N // 2`).
  For axial scans this moves them half a voxel down the slice axis. Default and golden trajectories,
  frames and `estimate_k0` all follow; the affine does not change. Recon caches
  written before this change are not read (reconstructed once).

- `estimate_k0` is now on by default. The dead-time centre of k-space is
  estimated and filled for SORDINO 0.1 to 1.x data; general ZTE data and scans
  without the sequence timing values are reconstructed without it (info line).
  With `correct_ramptime: false` and no explicit `estimate_k0` it is off (info
  line); `estimate_k0: true` with `correct_ramptime: false` is still an error.
  Images differ from earlier versions, reconstruction takes longer and needs
  more memory (on small computers the memory check may now stop and offer a
  higher limit). Set `estimate_k0: false` for the previous result. Recon caches
  made with the old default are not reused under the new default (they are
  found again with `estimate_k0: false`).

Golden trajectories (`TrajectoryMode` GoldenSampling / GoldenGridSampling of
`sordino_260801`; merged into brkraw-060 at da72d83):

- New option `golden_weights`: `capped` (default; Pipe-Menon weights capped at
  the |k|^2 weight, then one common factor), `pipe` (uncapped Pipe-Menon) or
  `k2` (|k|^2). Default scans are unchanged. The default keeps the recon caches
  made at da72d83.
- Already on brkraw-060 (da72d83): golden trajectory lists, frames (subset,
  sliding window, accumulation; `frame_spokes`, `frame_step`,
  `frame_accumulate`), the Golden Grid head rule for `sordino_260801`, and the
  capped sample-based density weights. See "Golden trajectories and frames" in
  the README.

# Release v0.2.4

Date: 2026-01-31
Changes since 0.2.2

- chore: prepare release v0.2.4 (edca175)
- Refactor: standardize logging and enhance off-resonance options (b4f7aa1)
- fix: update release_pr ensure_pr call (222d7a8)
- docs: release notes for v0.2.3 (6c842b5)
- chore: prepare release v0.2.3 (9a6b87d)
- fix: apply off-resonance correction during recon (f473ca6)
- ci: patch method to retreive PR (9b4d011)
