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
  frame_spokes=null          spokes per frame: subset, repetition or count
  frame_step=null            spokes between frame starts (null: frame_spokes)
  frame_accumulate=false     every frame starts at the first spoke
  peri_event=null            event bins: period=,bin= (bins=ceil(period/bin))
  peri_censor=null           0/1 file, one row per spoke block; needs peri_event
  subtract_rep_mean=false    subtract repetition mean (needs frames, 2+ reps)

read-time options (not in the cache key):
  frames=null                frames to return: int, list or "start:stop:step"
  axis=null                  frame axis: 3, -1, cycle or repetition
  max_memory_gb=auto         size-check limit (auto: half of physical memory)
  allow_short_fid=true       short FID: reconstruct the complete frames only

ZTE-mode scans of sordino_260801: set correct_ramptime=false yourself.
Details, cautions, caches and sizes: docs/technical-note.md, Appendix A,
in the brkraw-sordino repository.
