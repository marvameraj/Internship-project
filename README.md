# Internship-project
# Project Panopticon: Intelligent Exam Proctoring

Internship project for EduGuard AI (ML division). A time-series classification pipeline that flags suspicious exam behaviour while keeping false accusations of innocent students low.

**Author:** Marva Meraj

## The problem
The existing proctoring system flags students for sneezing, reading aloud or glancing at the keyboard. This project builds a model that flags only sustained, suspicious behaviour and only when it is at least 90% certain.

## What the notebook does
| Step | Technique |
|---|---|
| Align asynchronous logs | `pd.merge_asof` (1 second tolerance) joins tab-switch events to the 1-second video telemetry |
| Handle dropped video frames | forward-fill for gaze, interpolation for audio (732 missing values, no rows dropped) |
| Reduce noise | 10-second rolling mean of gaze and rolling max of audio |
| Handle class imbalance | `RandomForestClassifier(class_weight='balanced')` (only 0.77% of seconds are cheating) |
| Protect innocent students | `predict_proba` with a 0.90 decision threshold instead of `.predict()` |

## Results (held-out test set, 1,999 seconds, 20 real cheating seconds)
| | Default (0.50) | Strict (0.90) |
|---|---|---|
| False accusations | 1 | 1 |
| Cheating seconds caught | 16 | 10 |
| Precision | 94% | 91% |
| Recall | 80% | 50% |

The strict threshold meets the 90% precision requirement at the cost of recall.

## Limitations
- The test set has only 20 cheating seconds, so the numbers can shift easily.
- The split is random by second, so neighbouring seconds can appear in both train and test.
- Cheating labels exist only at logged tab-switch moments, so the model relies heavily on tab switches.

## Files
- `Project_Panopticon.ipynb` – the full notebook with outputs
- `video_telemetry.csv` – eye gaze angle and audio level, every second
- `system_events.csv` – tab-switch events with cheating labels

## How to run
Open the notebook in Google Colab or Jupyter, keep the two CSV files in the same folder, and run all cells.
Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn.
