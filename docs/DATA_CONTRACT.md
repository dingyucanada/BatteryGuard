# Data Contract and Leakage Rules

All units are explicit and all identifiers are stable strings. CSV and Parquet are supported. A source directory should contain `cells`, `cycles`, optional `timeseries`, optional `protocols`, and optionally `splits` with the corresponding extension.

The HTTP ingest surface accepts a directory path relative to the configured `data/raw` root only. It resolves symlinks before enforcing containment and rejects absolute paths, traversal, symlink escape, blank paths, and missing directories. The trusted local CLI may use an explicitly configured path outside that root.

## `cells`

Required fields: `cell_id`, `chemistry`, `nominal_capacity_ah`, `batch_id` (nullable), `protocol_id`, `cycle_life` (nullable for censored/hidden inputs), `eol_threshold`, and `censored`.

`cycle_life` is a target, never a feature. Censored cells keep `cycle_life=null`; no lifetime is fabricated.

## `cycles`

Primary key: (`cell_id`, `cycle_index`). Required: `charge_capacity_ah`, `discharge_capacity_ah`, `coulombic_efficiency`, `charge_time_s`, `discharge_time_s`. Optional: `dcir_ohm`, `avg_temp_c`, `max_temp_c`, charge/discharge energy.

## `timeseries`

Primary key: (`cell_id`, `cycle_index`, `sample_index`). Required: time, current, voltage, charge/discharge capacity. Temperature and `step_type` are optional. Time must be monotonic inside a cycle.

## `splits`

Exactly one row per `cell_id`: `train`, `calibration`, `test`, `demo_hidden`, or `external_ood`. Calibration cells cannot train or tune the point model. External OOD cells cannot select models or thresholds.

## Hard leakage invariants

The pipeline raises rather than warns when:

1. one cell appears in multiple splits;
2. protected/full-life target fields enter feature input or prediction input;
3. a feature references a cycle after `early_cycles`;
4. a filename/channel/protocol identifier directly encodes lifetime;
5. calibration cells were used to fit the point model;
6. external OOD data changed model selection.

Protocol is represented through interpretable stage variables. Raw protocol IDs are not one-hot encoded in a protocol-holdout experiment.

## MATR use

The adapter targets the structure associated with Severson et al., [“Data-driven prediction of battery cycle life before capacity degradation”](https://doi.org/10.1038/s41560-019-0356-8), *Nature Energy* 4, 383–391 (2019). Acquire data from the original research distribution and verify its current license and citation requirements before use. Put local files in `data/raw/matr/`; the project does not download or redistribute them. Run ingestion, inspect the Dataset Card and quality report, freeze the split manifest, and only then train. This repository contains no MATR result.
