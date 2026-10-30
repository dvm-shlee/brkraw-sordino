brkraw-sordino: SORDINO-ZTE reconstruction hook for BrkRaw

usage: brkraw convert STUDY -s SCAN [-r RECO] -H sordino:KEY=VALUE [-H ...]
       brkraw convert STUDY -s SCAN --hook-args-yaml FILE
       brkraw hook preset sordino -o FILE    (YAML template, every default)

reconstruction options (KEY=default):
  ext_factors=1.0            grid (FOV) scale; one value or read,phase,slice
  recon_matrix=null          smaller matrix, same FOV (60 or 60,60,60)
  ignore_samples=1           samples dropped at spoke start (0 is read as 1)
  offset=0                   first repetition to reconstruct
  num_frames=null            repetitions from offset (null: all)
  correct_ramptime=true      ramp-time correction (false: straight spokes)
  estimate_k0=true           fill the unmeasured k-space centre (any spokes)
  correct_spoketiming=false  align spokes to repetition time (no frames)
  offreso_freqs=null         off-resonance Hz per channel (120 or "120,-80")
  golden_weights=capped      golden density weights: capped, pipe or k2
  split_ch=false             true: one image per channel (false: merged)
  as_complex=false           return (real, imag) instead of magnitude
  mem_limit=0.5              GB for correct_spoketiming FID segments
  clear_cache=true           remove temporary files after conversion
  cache_dir=~/.brkraw/cache/sordino   cache folder (or BRKRAW_CONFIG_HOME)

frame options (golden scans; a Default scan only with a spoke count):
  frame_spokes=null          spokes per frame: subset, repetition, all or count
  frame_step=null            spokes between frame starts (null: frame_spokes)
  frame_accumulate=false     every frame starts at the first spoke
  peri_event=null            event bins: period=,bin= (bins=ceil(period/bin))
  peri_censor=null           0/1 file, one row per spoke block; needs peri_event
  subtract_rep_mean=false    subtract repetition mean (needs frames, 2+ reps)
  rep_pattern=none           remove the repeated pattern: none, model, measured
  anat_solver=adjoint        frame_spokes=all image: adjoint or cg (all spokes)
  anat_cg_iter=10            CG iterations of anat_solver=cg
  anat_lambda=0.0            Tikhonov weight of anat_solver=cg (absolute)
  frame_static=adjoint       rep_pattern=model frames: static part adjoint or cg
  frame_static_iter=10       CG iterations of frame_static=cg
  frame_static_lambda=0.0    Tikhonov weight of frame_static=cg (absolute)

physio options (side file; not in the cache key):
  physio=off                 on: respiration/heart rate from the spokes
  physio_n_inner=6           innermost samples per spoke (after ignore_samples)
  physio_resp_band=0.5,5     respiratory band, low,high in Hz
  physio_cardiac_band=5,15   cardiac band, low,high in Hz
  physio_window_s=3,1        rate window in s: respiration,cardiac
  physio_pca=true            PCA trace per band (false: mean)
  physio_prefix=null         output prefix (null: <cache_dir>/scan<id>)
  physio_traj_period=auto    trajectory periodicity removed from the traces: auto (2 x method subset
                             of a golden scan), off, or a number of spokes
  writes <prefix>_physio.tsv.gz (+ .json): time_s,respiratory,cardiac,quality

read-time options (not in the cache key):
  frames=null                frames to return: int, list or "start:stop:step"
  axis=null                  frame axis: 3, -1, cycle or repetition
  max_memory_gb=auto         size-check limit (auto: half of physical memory)
  allow_short_fid=true       short FID: reconstruct the complete frames only

ZTE-mode scans of sordino_260801: set correct_ramptime=false yourself.
Details, caches, sizes: docs/technical-note.md, Appendix A (repository).
