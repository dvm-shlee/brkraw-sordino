# brkraw-sordino: technical note

How the hook turns SORDINO raw data into images. Every equation here is the one the code
evaluates; the module that holds it is named at the start of each section. `brkraw hook docs
sordino` (`src/brkraw_sordino/docs.md`) lists the options one per line; their full description and the
practical advice are in Appendix A.
Measured results, figures and the reasoning behind design choices are kept in the project
research notes, not in this package. Statements marked **inference** are readings of the data
that the vendor documentation does not confirm.

## 1. Notation

| Symbol | Meaning (method parameter or option) |
| --- | --- |
| $i$, $j$, $c$, $r$ | spoke, sample, receive channel, repetition (all counted from 0) |
| $N_{\mathrm{pro}}$ | spokes per repetition (`NPro`) |
| $\mathbf N=(N_1,N_2,N_3)$ | acquisition matrix in read, phase, slice order (`PVM_Matrix`) |
| $\mathbf f$, $\mathbf M$ | `ext_factors` and the reconstruction grid, $M_a=\lfloor N_a f_a\rfloor$ |
| $\sigma$, $B$ | oversampling (`OverSampling`) and effective bandwidth in Hz (`PVM_EffSWh`) |
| $N_s$ | samples per spoke, $N_s=\lfloor N_1\sigma/2\rfloor$ |
| $\Delta t$ | dwell time, $\Delta t=1/(B\sigma)$ |
| $T_0$ | time from the RF centre to the first sample (`AcqDelayTotal`) |
| $\mathbf g_i$ | direction of spoke $i$ (unit vector, or $\mathbf 0$ for the origin spoke) |
| $\mathbf k_{ij}$ | sample position in cycles per sample ($\lvert k\rvert=\tfrac12$ is Nyquist) |
| $\boldsymbol\omega_{ij}$ | sample position in radians, $\boldsymbol\omega=2\pi\mathbf k$ |
| $y_{ij}$, $y'_{ij}$ | measured sample and the sample after the corrections of section 5 |
| $O_i$ | receiver frequency of spoke $i$ in Hz (`ACQ_O1_list`) |
| $n_{\mathrm{ign}}$ | samples dropped at the start of each spoke (`ignore_samples`) |

## 2. Samples and units

Sample $j$ of every spoke is taken at the time (from the RF centre)

$$t_j = T_0 + j\,\Delta t .$$

The first $n_{\mathrm{ign}}$ samples of each spoke are left out of the reconstruction. Positions
are kept in cycles per sample and converted with $\boldsymbol\omega=2\pi\mathbf k$ (the code
writes $\mathbf k/0.5\cdot\pi$) before they enter the NUFFT.

## 3. Spoke directions (`traj.py`, `golden.py`)

The method parameter `TrajectoryMode` selects the list of directions $\mathbf g_i$; a method
without it is `Default`. Only the direction list depends on the mode; the ramp model (section
4), the corrections (5) and the reconstruction (7) are the same for all.

### 3.1 Default

With $U$ the undersampling (`ProUnderSampling`; found by root finding so that the list has
`NPro` spokes) and $N=N_1$:

$$n_\theta=\Big\lceil\frac{\pi N}{2\sqrt U}\Big\rceil,\qquad
\theta_a=\frac{\pi\,(a+\tfrac12)}{n_\theta},\qquad
n_\phi(a)=\Big\lceil\frac{\pi N\sin\theta_a}{2\sqrt U}\Big\rceil,\qquad
\phi_{a,b}=\frac{\pi\,(b+\tfrac12)}{n_\phi(a)},$$

$$\mathbf g_{a,b}=(\sin\theta_a\cos\phi_{a,b},\ \sin\theta_a\sin\phi_{a,b},\ \cos\theta_a),$$

for $a=0,\dots,n_\theta-1$ and $b=0,\dots,n_\phi(a)-1$ (one hemisphere). The list is reordered
(`Reorder`: the scheme of the Bruker ZTE sequence, bouncing between the table edges in both
indices; otherwise every second $\theta$ ring is reversed), then, unless `HalfAcquisition`, followed by
the same vectors negated, $\mathbf g_{N_{\mathrm{hemi}}+i}=-\mathbf g_i$ ($N_{\mathrm{hemi}}$ the length of the first
part). With `UseOrigin` a zero vector is
put in front (spoke 0).

### 3.2 Golden Sampling

Two-dimensional golden means $\varphi_1=0.465571231876768$ and $\varphi_2=0.6823278038280193$
(multidimensional golden means, Chan et al. [5]; the golden-ratio profile order goes back to
Winkelmann et al. [6]). The sequence's list has $N_{\mathrm{pro}}=n_{\mathrm{subsets}}N_{\mathrm{sub}}$ spokes
(with `GoldenReorder=No`, `NGoldenSteps`). For $n=0,1,2,\dots$

$$z_n = 1-2\,\{n\varphi_2\},\qquad \vartheta_n=2\pi\{n\varphi_1\},\qquad
\mathbf g_n=\big(\sqrt{1-z_n^2}\cos\vartheta_n,\ \sqrt{1-z_n^2}\sin\vartheta_n,\ z_n\big),$$

where $\{x\}$ is the fractional part. The list is cut into subsets of $N_{\mathrm{sub}}$
consecutive spokes (`NGoldenSpokesPerSubset`, `NGoldenSubsets` subsets). With
`GoldenReorder=Yes` each subset is reordered into z-stacks of width `ZStackAngleDeg` (north to
south in even subsets, south to north in odd ones, by azimuth inside a stack, the stack edges
twisted with the azimuth) so that the step between consecutive spokes stays short; with `No`
the plain golden order is kept. The code is a vectorised copy of the sequence's C functions
(`goldensamp.c`) and is tested against a statement-by-statement copy.

### 3.3 Golden Grid

`NGridRing` $=R$ latitude rings of equal-area cells (`goldengrid.c`). With $\Delta\beta=180^\circ/R$ the
centre latitude of ring $l=0,\dots,R-1$ is $\beta_l=90^\circ-\Delta\beta/2-l\,\Delta\beta$ and it holds
$m_l=\mathrm{round}(360^\circ\cos\beta_l/\Delta\beta)$ cells; $n_{\mathrm{cell}}=\sum_l m_l$. The cell solid angle is
$\Omega=4\pi/n_{\mathrm{cell}}$, and the ring edges follow from $\sin b_{l+1}=\sin b_l-\Omega\,m_l/(2\pi)$,
$b_0=\pi/2$. Cell $q$ (counted from 1 over all rings, upper edge $z_u=\sin b_l$, height
$\Delta z=\lvert\sin b_l-\sin b_{l+1}\rvert$, longitude start $\alpha_0$, width $\Delta\alpha=360^\circ/m_l$) gets
in grid frame $m=1,2,\dots$ the spoke

$$\alpha=\alpha_0+\Delta\alpha\,\big\{m\varphi_2+\{q\,0.61803398875\}\big\},\qquad
z=z_u-\Delta z\,\big\{m\varphi_1+\{q\,0.41421356237\}\big\},$$

$$\mathbf g=\big(\sqrt{1-z^2}\cos\alpha,\ \sqrt{1-z^2}\sin\alpha,\ z\big).$$

One spoke per cell per frame; with `GridMirror` the opposite spokes $-\mathbf g$ of the frame
follow its cells. Hence $N_{\mathrm{pro}}=n_{\mathrm{frames}}\,n_{\mathrm{cell}}$ (times 2 with
`GridMirror`), and the hook stops when the count differs from `NPro`.

The golden lists do not use ProUnderSampling, `Reorder`, `HalfAcquisition` or `DirectMode`. A golden
protocol may still carry an old value of them (the development phantom scans keep ProUnderSampling
3.6147 from an earlier Default setting); it changes nothing here. ParaVision's online reconstruction
(pdata/1) does use it for Golden Grid (the first lines of such a scan's `traj` file are the Default list
for that value), so the pdata/1 image of a golden scan is not a reference image.

### 3.4 Golden Grid head (automatic, a warning, no option)

For sequence `sordino_260801` (whose reconstruction relation fills the gradient lists with `radialGrad3D`
whenever `GoldenSampTraj = No`, the case for Golden Grid) the data show that the first $N_h$ spokes of every
repetition of a Golden Grid scan ran along the Default list (section 3.1 with the scan's ProUnderSampling), with
the golden `ACQ_O1_list` frequencies, and from spoke $N_h$ on along the golden list (**inference**
from the data; ParaVision's internal order is not documented; it looks like a movement of a few mm
in the first seconds, but it is not motion). $N_h$ is the count that the Default loop gives for
ProUnderSampling (12,732 spokes in the two Golden Grid scans examined), at most $N_{\mathrm{pro}}$. The hook decides from the scan's `traj` file: its first lines are the
Default list (rule on) or the golden list (rule off, as a fixed sequence would write it); without
a `traj` file the rule applies only to `sordino_260801`; for `GoldenSampTraj = Yes` it is off.
When on, those spokes are placed along the Default list and the receiver-frequency difference is
applied over the sample time (section 5.1).

### 3.5 Order check

The sequence records $O_i=o_0+\mathbf o\cdot\mathbf g_i$ (offsets times direction). The hook fits
$o_0,\mathbf o$ by least squares and compares the relative RMS residual with $10^{-6}$; a golden
mode above it gets one warning (the computed order does not match the scanner's list).

## 4. Sample positions: the gradient ramp (`timing.py`, `ramp.py`, `traj.py`)

During spoke $i$ the gradient moves linearly from $\mathbf g_{i-1}$ to $\mathbf g_i$,

$$\mathbf G(t)=\mathbf g_{i-1}+(\mathbf g_i-\mathbf g_{i-1})\,f(t),\qquad
f(t)=\min\Big\{1,\max\Big\{0,\frac{t-t_r}{L}\Big\}\Big\},$$

and the sample position is the integral of $\mathbf G$ from the RF centre,

$$\mathbf k_{ij}=u\,\big(\tilde t_j\,\mathbf g_{i-1}+\tilde F_j\,(\mathbf g_i-\mathbf g_{i-1})\big),\qquad
u=\frac{1}{2(N_s-1)},\quad \tilde t_j=\frac{t_j}{\Delta t},\quad
\tilde F_j=\frac{1}{\Delta t}\int_0^{t_j}\! f(s)\,\mathrm ds .$$

$\mathbf g_{-1}$ is the last vector of the list (the dummy spokes and the previous repetition end
on it); every spoke, the last one included, has the ramp. The integral has the closed form
$h(t)-h(0)$ with $h(x)=0$ for $x\le t_r$, $(x-t_r)^2/(2L)$ for $t_r<x<t_r+L$ and
$L/2+x-t_r-L$ after. The ramp window $(t_r,L)$ (from the RF centre) depends on the sequence generation,
detected from method keys (0.x are development generations, 1.x is the stable `sordino` release):

| Generation | Sequence | Detected by | Ramp start $t_r$ | Length $L$ |
| --- | --- | --- | --- | --- |
| 0.1 | `mjm_zte_231005` | `RampDelay` and `RampTime` | $-(10\,\mu\mathrm s+\text{RampDelay}+T_{\mathrm{RF}}/2)$ | `RampTime` |
| 0.2 | `sordino_260122_trig` | `TrigSegmentMode` | $T_{\mathrm{RF}}/2$ | `RampTime` |
| 0.3 | `sordino_260801` (golden sample extension) | `MaximizeRampTime` or `RFWait`, method name | $T_{\mathrm{RF}}/2$ | `RampTime` |
| 1.x | `sordino` (stable release) | `MaximizeRampTime` or `RFWait`, method name | $T_{\mathrm{RF}}/2$ | `RampTime` |
| zte | general ZTE | no `RampTime` | none: constant gradient | none |

0.3 and 1.x share one timing model; they differ only in the name shown, taken from the method name (a
scan whose method name is neither is shown as "0.3 or 1.x"). Internally the models are `v1`, `v2`, `v3`
and `zte`; these values are part of the trajectory cache key and do not change.

$T_{\mathrm{RF}}$ is the excitation pulse length (`ExcPul`). For the general ZTE the gradient is
constant during RF and sampling, $\tilde F_j=\tilde t_j$ and $\mathbf k_{ij}=u\,\tilde t_j\,\mathbf g_i$. Timing values
and the per-version adjustments (all zero) are declared once in `timing.py`.

With `correct_ramptime=false` every spoke keeps its fixed vector,

$$\mathbf k_{ij}=u\,(j+\tau_0)\,\mathbf g_i,\qquad \tau_0=T_0/\Delta t ,$$

the dead time in samples. The hook computes trajectory rows on demand (spoke range by spoke range)
and keeps no whole trajectory.

## 5. Corrections of the samples (`recon.py`, `spoketiming.py`)

### 5.1 Ramp phase (`correct_ramptime`)

The receiver frequency $O_i$ is set for $\mathbf g_i$ while the gradient still moves from
$\mathbf g_{i-1}$, so spins away from the isocentre collect a phase during the ramp,

$$y'_{ij}=y_{ij}\,\exp\!\big[-2\pi\mathrm i\,(O_{i-1}-O_i)\,\tau_j\big],\qquad
\tau_j=\int_0^{t_j}\!\big(1-f(s)\big)\,\mathrm ds=t_j-\int_0^{t_j}\! f(s)\,\mathrm ds,$$

with $\tau_j$ in seconds and the phase reference at the RF centre. It applies when `ACQ_O1_list`
has one value per spoke and the gradient ramps (not for a single value, nor for the general
ZTE). The sign follows the phase observed on 0.2 data.

For the Golden Grid head (section 3.4) the true frequency is $O^{\mathrm{true}}_i=o_0+\mathbf o\cdot\mathbf g^{\mathrm{Default}}_i$
on the head ($o_0,\mathbf o$ fitted to the golden list and the recorded values) and the recorded value
elsewhere. With $e_i=O^{\mathrm{true}}_i-O_i$ the correction becomes

$$y'_{ij}=y_{ij}\exp\!\big[-2\pi\mathrm i\,\big((O^{\mathrm{true}}_{i-1}-O^{\mathrm{true}}_i)\tau_j+e_i\,t_j\big)\big],$$

(the $\tau_j$ term only with `correct_ramptime`, the $e_i t_j$ term always).

### 5.2 Off-resonance (`offreso_freqs`)

For channel $c$ with frequency $f_c$ (Hz), counting the kept samples as $j'=0,1,\dots$:

$$y'_{j'}=y_{j'}\exp\!\big[-2\pi\mathrm i\,f_c\,(j'+1)\,\Delta t\big].$$

### 5.3 Spoke timing (`correct_spoketiming`)

Spoke $i$ of repetition $r$ is measured at $t_{ir}=(rN_{\mathrm{pro}}+i)\,\mathrm{TR}$, with TR the
spoke repetition time (`PVM_RepetitionTime`). For each spoke, sample and channel, the series over $r$ of the
magnitude and of the unwrapped phase is interpolated linearly (extrapolated at the ends) to the centre of
the repetition's scan time, $(r+\tfrac12)\,N_{\mathrm{pro}}\,\mathrm{TR}$. It needs more than one repetition and
cannot be combined with frames. The corrected FID is a temporary file in the segments that `mem_limit` sets.

## 6. Density weights (`serial.py`, `dcf.py`)

The adjoint image of non-uniformly sampled data needs a weight $w_m$ per sample $m=(i,j)$.

**Default trajectory** (uniform radial sampling): the weight of 3D radial sampling at radius
$\lvert\mathbf k\rvert$, normalised by the largest over all kept samples of the repetition (and over the
virtual samples of section 9 when they are used):

$$w_m=\frac{\lvert\mathbf k_m\rvert^2}{\max_{m'}\lvert\mathbf k_{m'}\rvert^2}.$$

**Golden trajectories, Pipe-Menon** [1]. Weights are estimated from the actual sample positions of
one repetition, because the ramp bends each golden spoke from the previous direction (unevenly in
the Golden Grid order) and $\lvert k\rvert^2$ then leaves direction-dependent streaks. With the
spreading operator $\mathcal S$ (samples onto a fine grid with the NUFFT kernel) and the
interpolation $\mathcal I$ (grid back to the samples with the same kernel), starting from
$w^{(0)}=1$,

$$w^{(n+1)}_m=\frac{w^{(n)}_m}{\big\lvert(\mathcal I\mathcal S\,w^{(n)})_m\big\rvert},\qquad n=0,\dots,9 .$$

The kernel is finufft's [2], used for spreading and interpolation only (upsampling 2, the same
tolerance $10^{-6}$ as the reconstruction); the spread of all samples is a sum and is accumulated
chunk by chunk on one grid. After ten iterations the weights are scaled to the brightness of the
$\lvert k\rvert^2$ image, $\hat w_m=w_m\,\dfrac{\sum_{m'}d_{m'}}{\sum_{m'}w_{m'}}$ with
$d_m=\lvert\mathbf k_m\rvert^2/\max\lvert\mathbf k\rvert^2$ (sums over the measured samples).

**Cap** (`golden_weights: capped`, the default). Pipe-Menon also raises the weight of thinly covered
k-space and with it its noise, so no weight may exceed that of uniform radial sampling at its radius,
and the sum is restored with one common factor $\lambda\ge 1$:

$$w^\star_m=\lambda\,\min\!\big(\hat w_m,\ d_m\big),\qquad
\lambda=\frac{\sum_{m'}d_{m'}}{\sum_{m'}\min(\hat w_{m'},d_{m'})}.$$

Relative to $\lvert k\rvert^2$ the weights are thus clipped at one level and keep Pipe-Menon's
relative profile below it. `pipe` uses $\hat w$ without the cap; `k2` uses $d$ (no estimate, no extra
memory or time). The weights of one repetition serve every repetition and frame, and the K0 solve
(section 9); a golden scan holds one float64 per sample of a repetition for them (the kept samples, plus the
virtual ones with `estimate_k0`) and one complex128 spreading grid,
$8\,N_{\mathrm{pro}}\,n_{\mathrm{samples}}+16\,M_1M_2M_3$ bytes (`recon_dcf_nbytes`, counted in the size check;
0.58 GB for a 288,000-spoke scan at matrix 120), and needs time of the order of tens of seconds for a large scan. The choice is recorded in `scan._sordino_recon_meta["density_weights"]`
(`k2` for Default scans).

## 7. Adjoint reconstruction (`recon.py`, `serial.py`)

Per repetition (or frame) and channel the image on the grid $\mathbf M$ is the type-1 NUFFT (sign $+$,
tolerance $10^{-6}$, finufft [2]) of the weighted samples. With the centred grid index
$\boldsymbol\nu=\mathbf n-\lfloor\mathbf M/2\rfloor$, $n_a=0,\dots,M_a-1$, and the grid shift $\mathbf s$ of
section 8,

$$X_c[\mathbf n]=\frac1\eta\sum_{i,j} w_{ij}\; y'_{ijc}\;e^{-\mathrm i\,\boldsymbol\omega_{ij}\cdot\mathbf s}\;
e^{+\mathrm i\,\boldsymbol\omega_{ij}\cdot\boldsymbol\nu},\qquad
\eta=\sqrt{2^3\,M_1M_2M_3}.$$

The sum runs over the kept samples. It is linear in the spokes, so the reconstruction goes through
contiguous chunks of spokes (at most about 13 million samples each) and sums their images; the
weight maximum $d_{\max}$ is found over all spokes before the first chunk, so the chunked result
equals the whole-scan adjoint up to the summation order.

Channels: with `split_ch=false` the magnitude output is the root sum of squares
$\sqrt{\sum_c\lvert X_c\rvert^2}$ and the complex output the coherent sum $\sum_c X_c$; with `split_ch=true` the
channels are returned separately. `as_complex` returns real and imaginary parts as two arrays.

## 8. Image grid, affine and orientation (`recon.py`, `hook.py`, `orientation.py`)

### 8.1 Grid shift (WI-0119)

The image is placed where the BrkRaw affine says. The k-space phase origin of the scan is the FOV
centre $\mathbf c$ (the sequence moves it to the slice-pack position with the receiver frequencies
`ACQ_O1_list`). The NUFFT of section 7 puts it at index $\lfloor M_a/2\rfloor$ of axis $a$ ($\lfloor N_a/2\rfloor$ with
`ext_factors` 1; 8.2 treats the extended grid); the affine
$A$ (output index to world; the voxel-axis flips that brkraw applies for `ACQ_scaling_phase` < 0 and for coronal slices are in it) puts it at $A^{-1}\mathbf c$. The grid is moved by

$$\mathbf s=A^{-1}\mathbf c-\lfloor\mathbf N/2\rfloor ,$$

per output axis, moved to the reconstruction axis it holds (`orientation.axis_order`), rounded to
$10^{-6}$ voxel, with $\mathbf N$ the acquisition matrix (`ext_factors` 1). In the NUFFT the shift is the
phase $e^{-\mathrm i\boldsymbol\omega\cdot\mathbf s}$ on the samples: an object at displacement $\mathbf d$ (voxels)
from the phase origin lands on index $\lfloor\mathbf M/2\rfloor+\mathbf s+\mathbf d$. Being a phase only, it leaves the density weights and
the K0 normal operator unchanged. The affine itself is not changed by it.

The FOV centre is the one ParaVision records,

$$\mathbf c=\mathbf p_0+R^{\mathsf T}\big(\mathbf v\odot\boldsymbol\rho\odot\mathbf n\big),$$

with $\mathbf p_0$ `VisuCorePosition` (first voxel centre), $R$ `VisuCoreOrientation` (rows: image axis
directions), $\mathbf v$ the voxel size (`VisuCoreExtent` / `VisuCoreSize`), $\boldsymbol\rho$ `RECO_rotate` and
$\mathbf n$ the reconstruction size (`RECO_transposition` 1 swaps the first two image axes; other
transpositions give no centre; 3D data only). On the phantom it equals the slice-pack position (`PVM_SliceGeo`) to
$10^{-6}$ mm. Without reco parameters (no affine), with several slice packs, or when the centre cannot
be read, the grid stays at $\lfloor\mathbf M/2\rfloor$ (an info line says so).

For the phantom scans (axial, $120^3$ and $60^3$) $\mathbf s=(0,0,-0.5)$ voxel: before this change every
image sat half a voxel too high on the slice axis (0.1 mm at 0.2 mm voxels, 0.2 mm at 0.4 mm). Odd sizes,
other planes and flipped axes follow from the affine and are tested on synthetic affines only.

**Why the slice axis differs (inference).** ParaVision writes `RECO_rotate` $=(0.5,\ 0.5,\ 0.5-0.5/N)$,
that is $(N-1)/(2N)$ on the third axis, for this method and for Bruker's own 3D methods seen
(FieldMap $64^3$ sagittal, UTE3D $128^3$ radial in the PV 360.3.5 and 3.6 example protocols), with every
offset and plane seen (axial, sagittal). Then $\rho_3N=(N-1)/2$ against $\lfloor N/2\rfloor=N/2$ for an even
$N$: the FOV centre sits at the FFT centre $N/2$ on read and phase but at the geometric centre
$(N-1)/2$ of the partitions on the third axis, as the slices of a slice pack sit symmetrically about its
position. The Bruker manuals and method sources available do not give the formula
(`ATB_SetRecoRotate` is in a closed library), so this reading is not confirmed.

ParaVision's own reconstruction of the same scans (pdata/1) sits about half a voxel too low on the slice
axis; its geometry parameters do not explain it, so it is presumably in ParaVision's regridding
(**inference**, not tested), and it is not corrected here.

### 8.2 `ext_factors`

The grid becomes $M_a=\lfloor N_af_a\rfloor$ at the same voxel size, so the field of view grows by
$M_a/N_a$. The adjoint NUFFT puts the phase origin at index $\lfloor M_a/2\rfloor$ for odd and even sizes,
and the grid shift of 8.1 (derived for $\mathbf N$) is applied on the extended grid as well. The affine keeps the voxel size and
direction and moves its origin by whole voxels on each output axis $b$ (holding reconstruction axis $a$):

$$v_b=-\big(\lfloor M_a/2\rfloor-\lfloor N_a/2\rfloor\big),\qquad A'=A\,\begin{pmatrix}I&\mathbf v\\ 0&1\end{pmatrix},$$

so that every object stays where it was and the phase origin still maps to $\mathbf c$. When $M_a$ and $N_a$
differ in parity the extra voxel sits on one side and the FOV centre moves by half a voxel while
the objects do not. Before this fix a factor other than 1 on index 0 or 1 moved the image when the output
reorders those axes (measured: 1.6 mm on a $16^3$ scan, 10 mm on a $60^3$ scan), and a size that is not a
whole multiple (or an odd matrix) gave a sub-voxel shift ($16\times1.1$: 0.3 mm). `ext_factors=1` is
unchanged; the reconstruction and its cache key were not changed for any value by that fix.

### 8.3 Orientation

The reconstruction axes (read, phase, slice) are permuted to the data axes of the scan: by the
largest entry of each row of the gradient orientation matrix, then by the slice plane (axial
$(0,1,2)$, coronal $(0,2,1)$, sagittal $(1,2,0)$). Orientation is normalised when the first 3D axes are
spatial; `notebooks/orientation.ipynb` shows it on a test scan.

## 9. K-space centre (`estimate_k0`; `kcentre.py`, `serial.py`)

The dead time leaves the centre unmeasured. The first acquired sample lies at the radius
$\kappa_0=T_0/(\Delta t\,\sigma)$ k-grid units ($1/\mathrm{FOV}$; one unit is $\sigma$ samples along a spoke), the
first kept sample at $(T_0+n_{\mathrm{ign}}\Delta t)/(\Delta t\,\sigma)$. Without `estimate_k0` this centre is left empty. Both radii are logged (info for a general ZTE gap over
one unit, debug otherwise) and kept as `scan._sordino_recon_meta["kspace_gap"]` and in the recon cache
`.json`. The estimate runs on every trajectory (WI-0137): the ramp-corrected SORDINO one, the straight one of
`correct_ramptime=false` and a general ZTE; it is switched off with an info message only when the values the
virtual samples need are missing (and stops with an error when `estimate_k0` was given).

**Gap size.** About $\tfrac43\pi\kappa^3$ grid points lie inside the gap of radius $\kappa$ (4 at $\kappa=1$, 14 at 1.5),
and only the FOV support determines them. Synthetic sweep (WI-0137: straight spokes, two objects filling about 60 % and
90 % of the FOV, matrix 24 and 48, $\sigma=4$, noise free and with 2 % and 5 % noise): with 10 iterations the
relative $K_0$ error (noise free) is at most 0.4 % up to $\kappa=1.3$, 0.2–8 % at 1.4 and 4–73 % at 1.5 (with 5 % noise:
at most 2 % up to 1.3, matrix 24); with 40 iterations it
stays below 0.1 % up to 1.8 (noise free) and with 160 below 0.5 % at 2.0 (with noise), so above 1.3 the iteration
count, not the data, limits the fill. The iteration count therefore follows the gap of the scan
(`kcentre.iterations_for_gap`): 10 up to $\kappa=1.3$, 40 up to 1.8, 160 above. SORDINO data have
$\kappa\approx0.5$–$1.2$ and keep 10 (same result and cache key as before); a wider gap stores the count in the recon
cache key (`K0Iterations`) and logs it. Above $\kappa=2.0$, outside the measured range, the hook warns
(`kcentre.GAP_WARN_KGRID`) and still reconstructs.

**Estimate.** With $A$ the type-2 NUFFT onto the measured sample positions,
$(Ax)_m=\sum_{\mathbf n}x[\mathbf n]\,e^{-\mathrm i\boldsymbol\omega_m\cdot\boldsymbol\nu}$, $W=\mathrm{diag}(w)$ and the phased data
$\tilde y_m=y'_me^{-\mathrm i\boldsymbol\omega_m\cdot\mathbf s}$, one channel at a time:

$$\hat x=\arg\min_x\ \big\lVert W^{1/2}(Ax-\tilde y)\big\rVert^2
\iff A^{\mathsf H}WA\,\hat x=A^{\mathsf H}W\tilde y .$$

$\hat x$ lives on the reconstruction grid $\mathbf M$ itself (the FOV, or the extended FOV); the finite grid is the support constraint
that determines the centre. It is solved by conjugate gradients [4] from $x_0=0$ in the iteration count of the
gap rule above (10 up to $\kappa=1.3$, 40 up to 1.8, 160 above; set by the hook, no option: `estimate_k0` is an
on/off switch). The estimated signal at $k=0$ is

$$K_0=\sum_{\mathbf n}\hat x[\mathbf n]$$

(one value per channel; with frames, one per channel for all frames). The image is then the adjoint
(section 7) over the measured samples and over **virtual leading samples**: for each spoke the times
$t_m=t_{\mathrm{first}}-m\,\Delta t\ge 0$ ($m=1,2,\dots$, $t_{\mathrm{first}}$ the time of the first kept sample, down to the RF
centre) on the spoke's own ramp trajectory (section 4; with `correct_ramptime=false` on the spoke's straight
line, sample index $j+T_0/\Delta t$, $j=n_{\mathrm{ign}}-m$, the fixed-vector form of section 4), with predicted values $\hat y^{\mathrm v}=A_{\mathrm v}\hat x$ and weights
$w^{\mathrm v}$ of the same rule as section 6 (the maximum $d_{\max}$ also covers them). They carry no shift factor: $\hat x$ is
already on the shifted grid. Only the centre comes from the iterative solution; every measured sample
enters as in the plain adjoint.

**Normal operator and the choice of solve.** Two forms of $A^{\mathsf H}WA$ give the same image and $K_0$
within the NUFFT tolerance, and the hook picks one (no option):

- *Toeplitz* [3]: $(A^{\mathsf H}WAx)[\mathbf n]=\sum_{\mathbf m}T[\mathbf n-\mathbf m]\,x[\mathbf m]$ with
  $T[\mathbf d]=\sum_m w_m\,e^{\mathrm i\boldsymbol\omega_m\cdot\mathbf d}$, $\mathbf d\in[-(M-1),M-1]^3$. $T$ is one type-1 NUFFT
  onto a $2\mathbf M$ grid, accumulated chunk by chunk, and each CG step is two FFTs of size $2\mathbf M$ per axis
  (circulant embedding, Wajer and Pruessmann [7]). Memory about 1 KiB per output voxel plus 64 MiB, whatever the sample count;
  the faster form (1.2 to 8 times on the measured shapes).
- *Samples*: a type-2 NUFFT to the samples, the weights, a type-1 NUFFT back. Memory about 128 B per voxel plus
  80 B per sample of a frame plus 16 MiB.

The sample solve is used only when its whole reconstruction estimate is at most half of the Toeplitz
one (a frame with few samples for its grid), or when the Toeplitz solve does not fit the memory limit even
with the smallest chunk and the sample solve does. An info line names the solve and the estimates;
`get_dataobj_info` reports it (`recon_k0_method`, `recon_k0_reason`). Example: a $128^3$ grid with
$12{,}800\times64$ samples per frame measured 0.57 GiB with the sample solve (2.08 GiB on grids); a $64^3$ grid
with the same samples keeps the grid solve (0.42 GiB, 0.2 s per frame, against 0.25 GiB and 0.8 s).

**With frames** the estimate is made once from all spokes read, with the kernel of the repetitions
added ($n_{\mathrm{rep}}$ times one repetition's), and its predicted leading samples are added spoke by spoke; every frame
uses that one $\hat x$, so frames stay linear in their spokes. A window estimate alone is far too low (on a
development phantom scan: 3-4 % of the all-spoke value from 160 spokes, 26-29 % from 3,200), and with several repetitions the
common estimate also removes repetition-to-repetition changes of the filled centre. If the object moves
during the scan, the one value is the average over its positions. The estimated $K_0$ is also kept in
`scan._sordino_recon_meta["k0"]` and the cache `.json`.

## 10. Frames (`frames.py`, `frameplan.py`)

The repetitions read (`offset`, `num_frames`) form one stream of $N_{\mathrm{tot}}=n_{\mathrm{rep}}N_{\mathrm{pro}}$
spokes; global spoke $g$ is spoke $g \bmod N_{\mathrm{pro}}$ of repetition $\lfloor g/N_{\mathrm{pro}}\rfloor$, so windows run across
repetition boundaries (an info line counts them). With window $W$ (`frame_spokes`) and step $S$
(`frame_step`, default $W$) the frame $k=0,1,\dots$ holds the spokes

$$[\,l_k,h_k)=[\,kS,\ kS+W)\quad\text{or, with }\texttt{frame\_accumulate},\quad[\,0,\ W+kS),$$

as long as $h_k\le N_{\mathrm{tot}}$; with accumulation a last frame $[0,N_{\mathrm{tot}})$ is added when the
final step did not end there. Defaults: $W=N_{\mathrm{pro}}$ for a Default scan (one repetition per frame), and the method subset
for a golden scan: $N_{\mathrm{sub}}$ (`NGoldenSpokesPerSubset`; 1 with `GoldenReorder=No`) for Golden
Sampling, one grid frame $n_{\mathrm{cell}}$ (times 2 with `GridMirror`) for Golden Grid. `subset` is refused for a Default scan (no method subset).

A frame image is the adjoint of its spokes with the weights of one repetition (section 6),
scaled to the brightness of a one-repetition image:

$$X_k=\frac{N_{\mathrm{pro}}}{h_k-l_k}\cdot\frac1\eta\sum_{g=l_k}^{h_k-1}\sum_j w_{g \bmod N_{\mathrm{pro}},\,j}\;y'_{gj}\,e^{-\mathrm i\boldsymbol\omega_{gj}\cdot\mathbf s}\,e^{\mathrm i\boldsymbol\omega_{gj}\cdot\boldsymbol\nu}.$$

The engine keeps one running sum over the stream and a copy of it at each open frame start (so a sliding
window holds $W/S$ copies); a frame is the difference of the sum at its end and the copy at its start, so
every spoke is reconstructed once however much the frames overlap. Any spoke range can be
reconstructed; a frame that starts or ends inside a method subset gets a warning (its directions cover
the sphere less evenly), as does a Default scan with a frame shorter than a repetition (its list goes
through the sphere in order, so such a frame covers only part of it and accumulation shows the coverage
growing).

The frames are the fourth axis of the result and of the NIfTI. Time: the NIfTI time step is $\mathrm{pixdim}[4]=S\times\mathrm{TR}_{\mathrm{spoke}}$ (`PVM_RepetitionTime`, in the unit asked for;
with accumulation, the growth between frames). The frame list (spoke ranges, centre times
$(l_k+h_k)\,\mathrm{TR}/2$, interval) is in `scan._sordino_recon_meta["frames"]`.

**Size.** The recon cache holds one complex128 volume per frame (and channel):
$16\,n_{\mathrm{ch}}M_1M_2M_3$ bytes per frame. A sliding window also holds one copy of the running sum per open frame
start. The size check runs before anything is reconstructed and stops with the sizes.

## 11. Output scaling (`hook.convert`)

Converted NIfTI files are stored as unsigned 16-bit integers with a linear scaling in the header. For each
returned array $X$ (all frames; one channel, the merged result, or the real or imaginary part)

$$q=\mathrm{round}\!\Big(\frac{X-\mathrm{inter}}{\mathrm{slope}}\Big)\in\{0,\dots,65535\},\qquad
\mathrm{inter}=\min X,\quad \mathrm{slope}=\frac{\max X-\min X}{65535},$$

with slope 1 when $\max X=\min X$, and a reader recovers $x=\mathrm{inter}+\mathrm{slope}\cdot q$. The maximum maps to 65535
($2^{16}$ would wrap to 0 in uint16).

## 12. Caches and memory (`hook.py`, `memguard.py`, `cacheio.py`)

**Recon cache.** `recon_<hash>.bin` (complex128, all reconstructed frames, one headerless block per frame, Fortran order
inside it) with `recon_<hash>.json` (dtype, shape, `kspace_gap`, `k0`, `short_fid`), in `cache_dir`
(default `~/.brkraw/cache/sordino`, or under `BRKRAW_CONFIG_HOME`). `clear_cache` does not remove these files;
they stay for reuse until `brkraw cache clear`, and a temporary file left by an interrupted run is replaced
by the next run. The hash is the SHA-1 of the JSON of: scan, reco, FID identity, every option that changes
the reconstructed values, the decoded method parameters, the timing adjustments, the grid rule and shift
(section 8.1), the frame plan and, for golden scans, the density-weight rule. `as_complex`, `split_ch`,
`clear_cache` and `cache_dir` (they choose what is returned, what is cleaned up, or where the cache lives)
and the read-time options (`frames`, `axis`, `max_memory_gb`, `allow_short_fid`) are not in it, so changing them reads the same cache.
`golden_weights: capped` keeps the key made before the option existed, and Default scans have no density
entry in the key (their key is unchanged). A cache written under an older key
(for example before the grid change of section 8.1) is not read and is reconstructed once; caches made before
`estimate_k0` became the default are found again only with `estimate_k0: false`. Remove old ones with
`brkraw cache clear`. A short FID (`allow_short_fid`) has its own cache, separate from that of the complete scan.

**Trajectory files.** The hook does not write a trajectory file (rows are computed on demand, section 4).
`traj_<hash>.npy` files are written only by `traj.get_trajectory` (tools and tests). Their key
covers only the values that generate the trajectory (the gradient scheme, samples per spoke, and the sample
offset or the ramp-model sample times, plus a formula version), so options such as `ext_factors` or frames reuse
it. Files from before 0.6.0 (`<md5>.npy`) are no longer read.

**Size check.** Before it reconstructs or reads, the hook estimates the memory of what it returns, the
read buffers and, without a cache, the reconstruction step, and the disk space of a new cache; it stops with
`SordinoResourceError` over `max_memory_gb` (default half of physical memory, 4 GB when unreadable). The
reconstruction memory is a fixed part (the image of every receiver and one NUFFT grid, with `estimate_k0`
also its solve) plus one chunk of spokes; what the limit leaves after the read buffers sets the chunk size,
and the check stops only when even a 256-spoke chunk does not fit. With `correct_spoketiming` the estimate
is the larger of that and 5 times one FID segment, whose size follows `mem_limit`. The estimate is at or above every
measurement, by up to 2 times.

## 13. References

1. Pipe JG, Menon P. Sampling density compensation in MRI: rationale and an iterative numerical solution.
   *Magn Reson Med* 41:179-186, 1999. doi: `10.1002/(SICI)1522-2594(199901)41:1<179::AID-MRM25>3.0.CO;2-V`
2. Barnett AH, Magland J, af Klinteberg L. A parallel nonuniform fast Fourier transform library based on an
   "exponential of semicircle" kernel. *SIAM J Sci Comput* 41(5):C479-C504, 2019. doi:10.1137/18M120885X
3. Fessler JA, Lee S, Olafsson VT, Shi HR, Noll DC. Toeplitz-based iterative image reconstruction for MRI with
   correction for magnetic field inhomogeneity. *IEEE Trans Signal Process* 53(9):2866-2878, 2005.
   doi:10.1109/TSP.2005.853152
4. Hestenes MR, Stiefel E. Methods of conjugate gradients for solving linear systems. *J Res Natl Bur Stand*
   49(6):409-436, 1952. doi:10.6028/jres.049.044
5. Chan RW, Ramsay EA, Cunningham CH, Plewes DB. Temporal stability of adaptive 3D radial MRI using
   multidimensional golden means. *Magn Reson Med* 61:354-363, 2009. doi:10.1002/mrm.21837
6. Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An optimal radial profile order based on the
   golden ratio for time-resolved MRI. *IEEE Trans Med Imaging* 26(1):68-76, 2007. doi:10.1109/TMI.2006.885337
7. Wajer F, Pruessmann KP. Major speedup of reconstruction for sensitivity encoding with arbitrary
   trajectories. *Proc ISMRM* 9:767, 2001 (conference abstract, no DOI: not verified).

Sources 1-6 were checked against Crossref and the other public registries with the company tool
`scripts/litcheck.py` (identifier exists, first author, year, title and journal agree).

## Appendix A. Options and practical notes

`brkraw hook docs sordino` shows one line per option (`src/brkraw_sordino/docs.md`). This appendix is the full
description that page used to hold (moved unchanged, WI-0138): how to pass options, what each option does, and what
to watch for.

### A.1 Passing options

```bash
brkraw convert /path/to/study -s 3 -r 1 \
  --hook-arg sordino:ext_factors=1.2 --hook-arg sordino:offset=2 --hook-arg sordino:split_ch=false
```

Or from YAML. `brkraw hook preset sordino -o hook_args.yaml` writes a template with every
reconstruction option at its default; edit it and pass it (repeatable):

```bash
brkraw convert /path/to/study -s 3 -r 1 --hook-args-yaml hook_args.yaml
```

```yaml
hooks:
  sordino:
    ext_factors: 1.2
    offset: 2
    split_ch: false
    # as_complex: true  # optional, return (real, imag)
    # cache_dir: ~/.brkraw/cache/sordino  # optional (add manually if needed)
    # frame_spokes: 3200      # golden scan: 20 subsets of 160 spokes, 2 s at a TR of 0.625 ms
    # frame_accumulate: true  # frames of 3,200, 6,400, 9,600 ... spokes
```

- `--hook-arg` (short `-H`) values override YAML. YAML takes `{hooks: {sordino: {...}}}` or `{sordino: {...}}`.
- `BRKRAW_CONVERT_HOOK_ARGS_YAML` holds comma-separated YAML paths.
- Booleans are case-insensitive: `true`, `True`, `TRUE`, `false`, `False` (also `1`, `0`,
  `yes`, `no`, `on`, `off`); anything else is an error.
- A key the hook does not know is ignored with a one-line warning. `ramp_model` and
  `correct_phase` were removed and have no alias.

### A.2 Reconstruction options

| Option | Default | Meaning |
| --- | --- | --- |
| `ext_factors` | `1.0` | Scalar or 3 values. Enlarges the grid (so the FOV) at the same voxel size: axis `i` gets `int(Matrix[i] * ext_factors[i])` voxels (16 x 1.5 = 24, 16 x 1.1 = 17). Order: read, phase, slice of `PVM_Matrix`. |
| `recon_matrix` | `null` | Lower resolution, same FOV: one value or 3 (read, phase, slice of `PVM_Matrix`; `60`, `[60, 60, 60]`, `60,60,60`). Voxels get `Matrix / recon_matrix` times larger; k-space is cut at the new Nyquist radius (samples beyond it are not used, no taper); weights and `estimate_k0` work on the kept samples. Not above `Matrix`, one ratio on every axis; `null` or `Matrix` is the full grid. Works with `ext_factors` (on the new grid), frames and `split_ch`. See Cautions. |
| `ignore_samples` | `1` | Samples dropped at the start of every spoke (0 is read as 1). |
| `offset` | `0` | First repetition to reconstruct. |
| `num_frames` | `null` | Repetitions to reconstruct from `offset` (`null`: all). |
| `correct_ramptime` | `true` | Ramp-time correction: places each sample on the ramping gradient and removes the phase the FOV-offset frequency (`ACQ_O1_list`) leaves while it ramps. Off: every spoke keeps one fixed gradient vector (a straight line from the RF centre), no phase correction. Independent of `estimate_k0`. Timing values live in `timing.py`. |
| `estimate_k0` | `true` | Fills the k-space centre that the dead time leaves unmeasured: the dark centre of the adjoint image is filled. Works on every trajectory: the ramp-corrected one, the straight one of `correct_ramptime: false` and a general ZTE (independent of `correct_ramptime`). Changes the image and costs time and memory; a wide centre gap gives a warning (see Cautions). `false` gives the plain adjoint. |
| `correct_spoketiming` | `false` | Aligns every spoke to the centre of its repetition's scan time (needs several repetitions; not with frames). |
| `offreso_freqs` | `null` | Off-resonance in Hz, one per receive channel: a number, a list or `"120,-80"`; all forms read the same, a non-finite value is refused. |
| `golden_weights` | `capped` | Density weights of golden scans (case-insensitive): `capped` Pipe-Menon weights capped at the \|k\|^2 weight; `pipe` the same without the cap (less streaking, more noise in thinly covered k-space); `k2` the \|k\|^2 weights of uniform straight spokes (no estimate, no extra memory or time; streaks). Each choice has its own cache; `capped` keeps the caches made before the option existed. A Default scan always uses `k2` (a warning when the option is not `capped`). |
| `split_ch` | `false` | `false` merges channels (magnitude: root sum of squares; complex: coherent sum). |
| `as_complex` | `false` | Return `(real, imag)` instead of magnitude. |
| `mem_limit` | `0.5` | GB for the FID segments of `correct_spoketiming` (divided by a safety factor 3.4). Not the reconstruction memory limit. |
| `clear_cache` | `true` | Removes leftover temporary files after conversion; finished caches stay. |
| `cache_dir` | `~/.brkraw/cache/sordino` | Cache folder (or `BRKRAW_CONFIG_HOME`). |

Frames (golden scans; a Default scan only when `frame_spokes` is a spoke count). The spoke
directions come from the method parameter `TrajectoryMode` (`Default`, `GoldenSampling`,
`GoldenGridSampling`; older data without it are `Default`); the hook reads it, there is no option.

| Option | Default | Meaning |
| --- | --- | --- |
| `frame_spokes` | `null` | Spokes per frame. `null` or `subset`: the method subset, the smallest count that covers the sphere (`NGoldenSpokesPerSubset`; one grid frame). `repetition`: one repetition per frame. Integer: a spoke count. Default scans keep one repetition per frame (`subset` is refused). |
| `frame_step` | `null` | Spokes between frame starts (`null`: `frame_spokes`). Smaller is a sliding window, larger leaves spokes out. |
| `frame_accumulate` | `false` | Every frame starts at the first spoke and holds `frame_spokes` + k x `frame_step` spokes; the last one holds every spoke read. |
| `peri_event` | `null` | Event-related frames: one image per time bin relative to an event that repeats every `period` seconds, each bin holding that bin's spokes from every event. Comma-separated fields `period=` and `bin=` (seconds, both required), with optional `onset=` (seconds to the first event, default 0), `start=` (seconds from onset to the first bin, default 0, may be negative), `bins=` (default: enough to tile one period) and `events=` (default: every event that fits). Example: `peri_event=period=20,bin=1,onset=1`. A bin is a union of one spoke range per event, so the engine sums per bin in one forward pass instead of differencing prefix sums; each spoke is still gridded once. The golden trajectory repeats every `NPro` spokes, so events land on `NPro / gcd(period_spokes, NPro)` distinct phases: when the period divides the repetition evenly the bins re-sample a few directions many times, and the log reports the unique spoke count to judge against the Nyquist count of the reconstruction matrix. |
| `peri_censor` | `null` | Path to a 1D file of 0/1 over the spokes of the stream, 1 = usable. A peri-event range is used only when all of its spokes are usable; the per-bin scale divides by the events that reached each bin, so censoring changes the averaging weight and nothing else. The file's granularity must match the bin width, or bins end up with unequal event counts. |
| `subtract_rep_mean` | `false` | Average each spoke position over the repetitions read and subtract that average from every occurrence before gridding. Whatever is fixed for a spoke position - its own undersampling pattern, and the time-averaged anatomy - cancels, leaving only what varies between repetitions of that position. Task-neutral only when the repetition length is not a multiple of the event period: at 1810 subsets a repetition is 181 s against a 20 s cycle, so the 20 repetitions of a spoke cover all 20 task phases once each; at 1800 subsets they would all land on the same phase and the response would be subtracted away with the background. |

Read-time options (they choose what is returned from the cache and are not part of its key):

| Option | Default | Meaning |
| --- | --- | --- |
| `frames` | `null` | Frames to return, brkraw rules: an int picks one and removes the frame axis, a list keeps the axis in that order, `"start:stop[:step]"` is a slice. Frame 0 is frame `offset`. Only these frames are read from a cache; with no cache the whole scan (or `num_frames`) is reconstructed. |
| `axis` | `null` | The frame axis is data axis 3 (`3`, `-1`, `"cycle"` or `"repetition"`). `axis` without `frames` is an error. |
| `max_memory_gb` | half of physical memory | Limit of the size check (4 GB when memory cannot be read). Over it, the hook stops before reconstructing or reading with `SordinoResourceError` (a `MemoryError`) whose `retry_kwargs` holds the smallest passing limit (0.1 GB steps), for example `{"max_memory_gb": 1.8}`; the brkraw CLI uses it to ask "Proceed anyway?". The cache must also fit on the disk. Ask for fewer frames or raise the limit. |
| `allow_short_fid` | `true` | A scan stopped before its last repetition leaves a short FID. `true`: the hook measures the FID first and, when short, logs one warning (bytes found, needed, missing; complete frames of planned) and reconstructs only the complete frames; the bytes of the incomplete last frame are not used, the NIfTI frame count is the number of frames returned (voxel size, affine and TR unchanged), `offset` and `num_frames` count complete frames; a `ValueError` if none is complete or `offset` is at or after the last. `false`: a short FID stops with a `ValueError` before anything is reconstructed. A FID of the planned size or longer is read as usual. |

Frames from the CLI (sliding window of 3,200 spokes every 1,600):

```bash
brkraw convert /path/to/study -s 11 -r 1 \
  --hook-arg sordino:frame_spokes=3200 --hook-arg sordino:frame_step=1600
```

### A.3 Cautions

- **`ext_factors` axes.** The three values follow the acquisition order (read, phase,
  slice), not anatomical axes and not the output array axes: the hook reorders axes after
  reconstruction, so which anatomical direction an index widens depends on the scan. Check
  your protocol once (for example `[1.5, 1, 1]` and see which side grew). A scalar or three
  equal values widen every axis, so the order does not matter for them. "RPS" is also an orientation code (Right-Posterior-Superior); here it is never
  one. Example (two approved coronal scans, readout `H_F`, quadruped, `Head_Supine`,
  `subject_ras`): index 0 widens posterior-anterior, 1 left-right, 2 inferior-superior; the
  output array holds index 1 on axis 0 and index 0 on axis 1.
- **`estimate_k0` is on by default** and costs time and memory (a 160^3 scan with NPro 80,876, NPoints 640, OverSampling 8 and 2 receivers, synthetic FID, macOS, default limit: 3.0 GiB measured without it, 4.9 GiB with it). Memory is a fixed part plus one chunk of spokes and does not grow with spokes or frames. On an 8 GB computer (limit 4 GiB) such a scan stops with an offer of a limit of 5.3 GB; `estimate_k0: false` fits. Set to `true` explicitly it stops with an error when the values it needs are missing (the sequence timing, or with `correct_ramptime: false` only `AcqDelayTotal`, `PVM_EffSWh`, `OverSampling`); left at the default it is switched off with an info line then. Caches made before it became the default are found again only with `estimate_k0: false`.
- **`estimate_k0` with `correct_ramptime: false` and on a general ZTE (WI-0137).** Before, `correct_ramptime: false` switched `estimate_k0` off (an explicit `true` was an error) and a general ZTE was always reconstructed without it. Now both follow `estimate_k0` (default `true`): those results change and their caches get a new key (`estimate_k0: false` gives the old result and finds the old caches). The virtual centre samples continue each spoke's straight line back to the RF centre. On SORDINO data with `correct_ramptime: false` the straight line only approximates the ramped spokes: treat that combination as an experiment.
- **Wide centre gap.** The fill is a least-squares solve with the FOV as the only constraint. Its iterations follow the gap (the first kept sample's distance from k = 0 in k-grid units; SORDINO data: 0.5-1.2): 10 up to 1.3, 40 up to 1.8, 160 above (synthetic data: K0 within 0.1 % up to 1.8 and 0.5 % at 2.0; 10 iterations alone degrade above 1.3). Above 2.0 a warning gives the gap; the reconstruction still runs. `ignore_samples: 0` narrows the gap by 1 / OverSampling when the first sample is usable. The gap is in `kspace_gap` of the cache `.json`.
- **ZTE-mode scans of the SORDINO sequence.** `sordino_260801` can run in ZTE mode (`SequenceMode = ZTE`: per the sequence source the gradient reaches each spoke's vector before the RF pulse and stays constant while sampling). The method file still holds `RampTime` and `MaximizeRampTime`, and the hook does not tell such scans apart from the header: it applies the SORDINO ramp model. Set `correct_ramptime: false` (and keep `estimate_k0: true`) for them yourself.
- **Golden frames need disk.** The cache holds one complex128 volume per frame and channel.
  The default frames of a 288,000-spoke, 160-spoke-subset scan at 120^3 are 1,800 frames:
  49.8 GB of cache (46.4 GiB), 24.9 GB as float64 magnitude in memory, 6.2 GB as uint16 NIfTI. The size
  check runs first and stops with the sizes; use larger `frame_spokes`, fewer repetitions
  (`num_frames`) or `frame_spokes: repetition`. A sliding window also keeps one running-sum
  copy per open frame start (`frame_spokes` / `frame_step` copies).
- **`recon_matrix` grid and values.** The grid is the one ParaVision writes for a scan of that matrix with the same FOV and centre (phantom: scan 13 at 120^3 reduced to 60 has the affine of the native 60^3 scan 18), so the FOV edges can move by half a full-resolution voxel. Values follow the full image's low-pass (phantom, Default: scale within 0.2 %, NRMSE 0.03); golden scans with `capped` or `pipe` weights re-estimate the weights on the kept samples, which put the global scale at 0.74 of the full image (shape kept; `k2`: 1.04).
- **NIfTI is uint16 with slope and intercept.** Converted files are stored as 16-bit integers; the true value is `inter + slope * stored`.
- **Frames:** `correct_spoketiming` cannot be combined with frames (use
  `frame_spokes: repetition`). `estimate_k0` is estimated once from all spokes read. NIfTI
  `pixdim[4]` is `frame_step` x the spoke TR in the unit asked for. Frame spokes, centre times
  and interval are in `scan._sordino_recon_meta["frames"]`.
- **Grid shift and cache regeneration.** The image is placed where the brkraw affine says; the
  reconstruction grid is moved by a derived shift (section 8). For the phantom
  scans (axial) it is (0, 0, -0.5) voxel: earlier images sat half a voxel too high on the slice
  axis. Recon caches written before this change are not read; they are reconstructed once.
  ParaVision's own reconstruction (pdata/1) sits about half a voxel low on the slice axis; this
  is not corrected here.
- **Cache files** (`cache_dir`): `recon_<hash>.bin` (complex128, all frames) with its `.json`
  (also `kspace_gap`, `k0`, `short_fid`). They stay for reuse until `brkraw cache clear`;
  `clear_cache` removes only temporary files. The key covers scan, reco, FID and every option that
  changes the values; `as_complex`, `split_ch`, `clear_cache`, `cache_dir` and the read-time
  options are not in it, so changing them reads the same cache.
- **Result metadata** (`scan._sordino_recon_meta`, also in the cache `.json`): `kspace_gap` (radius of
  the unmeasured centre, first acquired and first kept sample), `k0` (`[real, imag]` per channel and
  frame; with frames one estimate for all), `short_fid` (`fid_nbytes`, `expected_nbytes`,
  `frame_nbytes`, `frames_planned`, `frames_complete`; null for a complete FID; a short FID has its
  own cache), `frames`, `density_weights`, `sequence_generation`, `recon_matrix` (null, or the matrix, ratio, samples kept per spoke, k-space radius kept and output scale). A Golden Grid head, if applied, logs a warning; no option.
- **Sequence generations.** 0.1 (`mjm_zte_231005`), 0.2 (`sordino_260122_trig`), 0.3 (`sordino_260801`, golden sample extension) and 1.x (`sordino`, the stable release) are development generations and the release; zte is a general ZTE. 0.3 and 1.x share one ramp model, so the result does not depend on which a scan is: they are told apart by the method name only, and a scan whose method name says neither is shown as `0.3 or 1.x`. The x of 1.x is not read (the method header has no version string). Details: section 4.

### A.4 For developers: tests and manual checks

`tests/` holds only tests on synthetic data; `pytest tests` needs no dataset and the CI never reads one.
Checks that read a real dataset are scripts in `scripts/validate/` (named `check_*.py`, so a plain `pytest` does not
collect them) and are run by hand, when wanted: `BRKRAW_SORDINO_GOLDEN_ZIP=/path/to/phantom.zip pytest -v
scripts/validate/check_golden_phantom.py scripts/validate/check_grid_centre_phantom.py`. Without the variable they stop
with a message. They compare the golden trajectories with the scanner's `traj` file and the derived grid shift with
the phantom's recorded geometry.

### A.5 Size estimate without reading data

`brkraw_sordino.get_dataobj_info(scan, reco_id, **options)` takes the options of `get_dataobj`
and returns a dict:

| Key | Content |
| --- | --- |
| `shape`, `dtype`, `count`, `nbytes` | each returned array, how many (channels with `split_ch`, x 2 with `as_complex`), total bytes |
| `frames`, `frames_reconstructed` | frames returned; frames in the reconstruction and its cache |
| `cached`, `cache_path`, `cache_dir`, `cache_dtype`, `cache_nbytes` | a valid cache exists and its file (`cache_dtype` is complex128 until one exists) |
| `peak_nbytes` | the estimate compared with the limit: returned arrays, three cache frames, and without a cache `recon_nbytes` |
| `recon_nbytes`, `recon_chunk_spokes`, `recon_chunks` | reconstruction step estimate, spokes per chunk, chunks (0 and null when cached) |
| `recon_frames_open`, `recon_dcf_nbytes` | running-sum copies held at once with frames (null without); bytes of the golden density weights and their grid (0 otherwise) |
| `recon_k0_method`, `recon_k0_reason` | `"toeplitz"` or `"samples"` (with `estimate_k0`, no cache); `"rule"` or `"limit"` |
| `disk_nbytes`, `disk_free_nbytes` | disk a new cache needs (0 when cached), free space (null when cached or unreadable) |
| `limit_nbytes`, `limit_source` | the limit `get_dataobj` applies; `max_memory_gb option`, `half of physical memory` or `fallback 4 GB` |
| `frames_planned`, `fid_short_nbytes` | frames in `PVM_NRepetitions`; bytes the FID is short (0 complete, null unknown); with a short FID `frames_reconstructed` is the complete frames |
| `sequence_generation`, `sequence_method` | the sequence generation: `"0.1"`, `"0.2"`, `"0.3"`, `"1.x"`, `"0.3 or 1.x"` (method name does not tell), `"zte"` (null when not recognised); the scan's method name (null when missing). The same label is kept as `sequence_generation` in the recon cache `.json` |
