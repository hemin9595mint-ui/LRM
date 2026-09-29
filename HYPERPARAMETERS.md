# Hyperparameter Selection Record

This file documents the hyperparameter tuning procedure used for the experiments reported in:

**A Physics-Guided Framework for Underwater Video Enhancement in Aquaculture Environments**

The hyperparameters were selected using the **validation set only**. The test sets were not used for hyperparameter tuning, checkpoint selection, or final-setting selection.

## Final Configuration

| Hyperparameter | Final value |
|---|---:|
| `lambda_deg` | 1.0 |
| `lambda_cdg` | 1.0 |
| `lambda_A` | 0.01 |
| `lambda_N` | 0.01 |
| `lambda_temp` | 0.01 |
| `lambda_adv` | 0.01 |
| `lambda_col` | 1.0 |
| `lambda_sty` | 0.01 |
| `lambda_str` | 0.1 |

## Candidate Values Considered

The loss weights were tuned manually around the final configuration.  
`lambda_deg` and `lambda_col` were fixed at 1.0 as reference weights, while the remaining loss weights were explored over the following candidate values:

| Hyperparameter | Candidate values |
|---|---|
| `lambda_cdg` | {0.5, 1.0, 2.0} |
| `lambda_A` | {0.001, 0.01, 0.05} |
| `lambda_N` | {0.001, 0.01, 0.05} |
| `lambda_temp` | {0.001, 0.01, 0.05} |
| `lambda_adv` | {0.001, 0.01, 0.05} |
| `lambda_sty` | {0.001, 0.01, 0.1} |
| `lambda_str` | {0.05, 0.1, 0.2} |

The above values define the candidate values considered during manual tuning. We did **not** perform an exhaustive Cartesian grid search over all possible combinations.

## Validation Trials

A total of **eight candidate configurations** were evaluated on the validation set.

| Trial | `lambda_cdg` | `lambda_A` | `lambda_N` | `lambda_temp` | `lambda_adv` | `lambda_sty` | `lambda_str` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.1 |
| 2 | 2.0 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.1 |
| 3 | 1.0 | 0.001 | 0.001 | 0.01 | 0.01 | 0.01 | 0.1 |
| 4 | 1.0 | 0.05 | 0.05 | 0.01 | 0.01 | 0.01 | 0.1 |
| 5 | 1.0 | 0.01 | 0.01 | 0.001 | 0.01 | 0.01 | 0.1 |
| 6 | 1.0 | 0.01 | 0.01 | 0.05 | 0.01 | 0.01 | 0.1 |
| 7 | 1.0 | 0.01 | 0.01 | 0.01 | 0.05 | 0.1 | 0.2 |
| 8 (final) | 1.0 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.1 |

For all eight trials:

- `lambda_deg = 1.0`
- `lambda_col = 1.0`

## Selection Criterion

The final configuration was selected according to the overall validation performance, jointly considering:

- **frame-level enhancement quality**, including visual quality and restoration fidelity; and
- **temporal consistency**, with particular attention to temporal stability and flicker reduction across consecutive frames.

The final setting was chosen as the configuration providing the best overall balance between enhancement quality and temporal consistency on the validation set.

## Reproducibility Note

The selected hyperparameters were fixed after validation-based tuning and were subsequently used for all reported test-set evaluations. No test-set results were used to adjust the hyperparameters.
