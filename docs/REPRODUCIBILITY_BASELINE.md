# FELT v2 reproducibility baseline

Baseline recorded on 2026-09-06 before repository cleanup.

## Repository state

- Working branch: `pyfeat-2.0.3-migration`
- Baseline revision: `e6edbd3ff76b933a9f546b1446f245e6c592d50c`
- The worktree contained tracked and untracked development changes.
- `01_data/` is excluded from Git and is not included in the source snapshot.
- `01_data/02_output/062/` contains historical Py-Feat 0.6.2 material and is
  excluded from the FELT v2 production contract.

## Restored RAVDESS input

| Measure | Count |
| --- | ---: |
| All MP4 files | 4,904 |
| Selected full audiovisual files (`01-*`) | 2,452 |
| Full audiovisual speech files | 1,440 |
| Full audiovisual song files | 1,012 |

The canonical input root at baseline was `01_data/01_input/`.

## Existing v2 output inventory

| Directory | Files | Approximate size |
| --- | ---: | ---: |
| `01_raw_motion/` | 2,452 | 6.47 GiB |
| `02_smoothed_motion/` | 2,452 | 10.47 GiB |
| `03_smoothed_video/` | 17,165 | 11.17 GiB |
| `logs/` | 26 | 0.07 GiB |
| `migration_smoke/` | 14 | 0.01 GiB |
| `plots/` | 38 | 0.08 GiB |
| `qc/` | 5,016 | 0.32 GiB |

The `03_smoothed_video/` count consists of 17,164 canonical MP4 files plus the
batch manifest.

## Canonical visualization inventory

| View | MP4 files |
| --- | ---: |
| AU region heatmap | 2,452 |
| AU-to-mesh animation | 2,452 |
| Blendshape region heatmap | 2,452 |
| Landmark-only contour mesh | 2,452 |
| Landmark-only tessellation mesh | 2,452 |
| Source-overlay contour mesh | 2,452 |
| Source-overlay tessellation mesh | 2,452 |
| Total | 17,164 |

## Historical Py-Feat 0.6.2 inventory

`01_data/02_output/062/` contained 5,547 files totalling approximately
14.61 GiB. This material is retained as history and is not an input to the
canonical Py-Feat 2.0.3 workflow.

## Protected archive backup

The archives under `01_data/02_output/` were compared with copies in a
separately stored local backup. File sizes and SHA-256 hashes matched for every
archive.

| Archive | Bytes | SHA-256 |
| --- | ---: | --- |
| `01_raw_motion.zip` | 2,134,548,348 | `AF07E815BD06DFF807AFB4A7139F7EF3A057DA0E33D29DCEC42D1A7F03470826` |
| `02_smoothed_motion.zip` | 4,654,410,454 | `59A8D681ED36EA057E713163F06B9CE3AED03C8C7F8D5EA1789A82849F13E182` |
| `03_smoothed_video.zip` | 11,997,243,642 | `85393B596FEB37102475D82C9ADDE05199C66FAA6B111814FEA5677E4FC982DE` |
| `logs_plots_qc_smoke.7z` | 169,771,243 | `E26E6E5F8675A7F4DB50A84F6F0AF9FBD25AB2235262111CAC5C46E0E4690981` |

These hashes identify the protected pre-cleanup backups. The raw and smoothed
CSV working trees were subsequently migrated from extraction-machine absolute
`input` values to portable `Actor_XX/filename.mp4` references. That intentional
metadata-only change means newly built motion archives will not match the two
backup ZIP hashes above; the numerical tracking and smoothing values were not
reserialized.

This establishes a recoverable copy of the completed v2 artifacts before code
and documentation cleanup begins.
