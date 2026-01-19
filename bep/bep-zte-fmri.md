---
Title: "ZTE-based fMRI extension"
Status: Draft
Authors: ["Your Name / BrkRaw team"]
Created: "2025-01-18"
License: "CC BY 4.0"
---

# ZTE-based fMRI BIDS Extension Proposal (Draft)

This document follows the BIDS Extension Proposal (BEP) guidance and process:

- BEP process: https://bids.neuroimaging.io/extensions/process.html
- BEP guidelines: https://bids.neuroimaging.io/extensions/guidelines.html

## Proposal summary

We propose a new BIDS Extension Proposal (BEP) to support Zero Echo Time (ZTE)-based
functional MRI (fMRI) acquisitions.

This extension focuses on explicitly recording how ZTE-based fMRI data were acquired
and how they were reconstructed, without prescribing reconstruction algorithms or
requiring access to raw k-space data. The goal is to improve data provenance,
interpretability, and cross-study comparability, while remaining fully backwards
compatible with existing BIDS datasets and tools.

## Defined scope

### In scope

- Functional (time-series) ZTE-based MRI acquisitions
- ZTE-derived fMRI sequence families (e.g., ZTE, SORDINO, MB-SWIFT)
- Metadata describing:
  - How the data were acquired (acquisition strategy, trajectory, key parameters)
  - Which reconstruction approach and software were used
  - Minimal, human- and machine-readable reconstruction descriptors
- An extensible framework that can accommodate additional ZTE-based fMRI variants
  not yet explicitly listed

### Out of scope

- Definition or standardization of reconstruction algorithms
- Raw k-space reconstruction requirements
- Vendor- or site-specific reconstruction pipelines
- Derivatives specification

## Planned deliverables

A BIDS Extension Proposal that:

- Introduces a dedicated suffix for ZTE-based fMRI data (tentative: `zte`)
- Defines acquisition-related metadata fields specific to ZTE-based fMRI
- Defines reconstruction-related metadata fields to document how data were
  reconstructed
- Example BIDS-compliant filenames and JSON sidecars
- Clear guidance on backwards compatibility and tool behavior

## Use cases

- Sharing and archiving ZTE-based fMRI datasets in a BIDS-compliant manner
- Transparently documenting acquisition and reconstruction provenance for
  ZTE-based fMRI studies
- Enabling correct interpretation and comparison of ZTE-based fMRI data across
  sites, vendors, and studies
- Supporting tool developers and data users who need to understand how ZTE-based
  fMRI data were generated, without requiring reconstruction reproducibility

## Proposed data organization

- Data type: `func`
- Proposed suffix: `zte` (tentative)
- Example filename:
  - `sub-01_task-rest_run-01_zte.nii.gz`
  - `sub-01_task-rest_run-01_zte.json`

The suffix is intentionally distinct from `bold` to avoid implying T2*-weighted
contrast. The final suffix selection should be aligned with community feedback.

## Metadata fields

### Standard BIDS fields (reused)

- `RepetitionTime` (s)
- `EchoTime` (s)
- `FlipAngle` (deg)
- `MRAcquisitionType` ("3D")
- `Manufacturer`
- `SequenceName`

### Proposed ZTE extension fields

The following fields are proposed as ZTE-specific metadata to support radial
and non-radial sampling and reconstruction reproducibility:

- `ZTETechnique`: sequence family (e.g., `ZTE`, `SORDINO`, `MB-SWIFT`)
- `ZTETrajectoryType`: trajectory type (e.g., `radial`, `spiral`, `cones`, `cartesian`)
- `ZTEReadoutPoints`: number of readout samples per spoke (from `NPoints`)
- `ZTEProjections`: number of projections/spokes per frame (from `NPro`)
- `ZTEOverSampling`: oversampling factor (from `OverSampling`)
- `ZTEEffectiveBandwidth`: effective bandwidth (Hz) (from `PVM_EffSWh`)
- `ZTEAcqDelayTotal`: acquisition delay (us) (from `AcqDelayTotal`)
- `ZTEUnderSampling`: under-sampling factor (from `ProUnderSampling`)
- `ZTEReceiverChannels`: number of receivers (from `PVM_EncNReceivers`)
- `ReconMatrix`: reconstruction matrix (from `PVM_Matrix`)
- `FieldOfView`: FOV in mm (from `PVM_Fov`)
- `SpatialResolution`: nominal resolution in mm (from `PVM_SpatResol`)
- `SliceOrientation`: slice orientation labels (from `PVM_SPackArrSliceOrient`)
- `ReadoutOrientation`: readout orientation labels (from `PVM_SPackArrReadOrient`)
- `ReconstructionMethod`: method or filter used for reconstruction
- `ReconstructionType`: "vendor", "custom", or "unknown"
- `ReconstructionSoftware`: software name
- `ReconstructionSoftwareVersion`: software version
- `ReconstructionDescription`: brief free-text description of the approach

These fields are intended to be machine-readable and aligned with existing
parameter names to minimize ambiguity.

## Example

The example below illustrates how a ZTE-based fMRI dataset can document both
how the data were acquired and how they were reconstructed, without
prescribing or standardizing reconstruction algorithms.

This example is provided for illustration only. The listed fields are not
intended to be exhaustive or mandatory.

```json
{
  "RepetitionTime": 0.05,
  "EchoTime": 0.0,
  "FlipAngle": 2.0,
  "MRAcquisitionType": "3D",
  "Manufacturer": "Bruker",
  "SequenceName": "sordino",

  "ZTETechnique": "SORDINO",
  "ZTETrajectoryType": "radial",
  "ZTEReadoutPoints": 256,
  "ZTEProjections": 12000,
  "ZTEOverSampling": 2.0,
  "ZTEEffectiveBandwidth": 100000.0,

  "ReconstructionMethod": "RecoRegridNFilter",
  "ReconstructionType": "vendor",
  "ReconstructionSoftware": "ParaVision",
  "ReconstructionSoftwareVersion": "360.3.5",
  "ReconstructionDescription": "Vendor-provided filters for non-Cartesian reconstruction"
}
```

In this example:

- Acquisition-related fields describe how k-space data were sampled and how the
  ZTE-based fMRI time-series was acquired.
- Reconstruction-related fields record which reconstruction approach and
  software were used to generate the final images.
- These fields record what was done, but do not define or prescribe how
  reconstruction should be performed.

## Community input and open participation

While this proposal includes example acquisition and reconstruction metadata for
selected ZTE-based fMRI sequences (e.g., ZTE and SORDINO), a complete and
community-agreed parameter set is not yet defined for all ZTE-based fMRI
implementations.

In particular, sequence families such as MB-SWIFT, as well as other ZTE-derived
or related fMRI implementations not explicitly listed here, may require additional
or sequence-specific metadata.

Input and contributions from the broader ZTE-based fMRI community are explicitly
welcomed to help identify missing parameters, refine existing fields, and ensure
that the extension remains inclusive and extensible.

## Current status

A draft specification document is under active development. This issue is intended
to gauge community interest, collect early feedback, and identify potential
contributors before proceeding to a formal BEP submission.
