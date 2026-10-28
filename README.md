# Satellite Telemetry Anomaly Detection

A Python-based project exploring anomaly detection in synthetic satellite telemetry. The project combines temporal feature engineering, an Isolation Forest baseline, and rule-based detectors for drift, frozen readings, and spikes.

> **Status:** Exploratory portfolio project. Reported metrics are preliminary diagnostics, not an unbiased estimate of performance on unseen data.

## Project objective

Satellite telemetry can contain unusual patterns that may indicate sensor or system anomalies. This project explores how telemetry time series can be analysed to identify:

- **Drift:** gradual or sustained changes in a telemetry signal.
- **Freeze:** readings that remain unchanged for an extended period.
- **Spike:** sudden changes that reverse in the following sample.

The dataset is synthetic; the results should not be interpreted as validation on real spacecraft telemetry.

## Dataset

The dataset contains 20,000 timesteps, seven telemetry channels, and anomaly labels.

Telemetry channels:

- `temperature`
- `voltage`
- `current`
- `battery_level`
- `orientation_x`
- `orientation_y`
- `orientation_z`

Additional columns include `timestep`, `anomaly`, and `anomaly_type`. The dataset contains 582 labelled anomalies (2.91%).

## Methodology

### 1. Temporal feature engineering

The project constructs 42 features across the seven telemetry channels, including raw values, differences, rolling variability, deviation from a trailing mean, and unchanged-reading counts.

### 2. Baseline model

An Isolation Forest is trained using normal observations. A normal-only calibration subset is used to establish an anomaly-score threshold.

### 3. Hybrid detection

The notebook explores a combination of rule-based and model-based approaches, including drift, freeze, spike, and short-freeze detection. Exploratory variants and threshold comparisons are retained to show the development process.

Some rules use future samples or full-run information, so the current implementation is retrospective rather than strictly real-time.

## Validation diagnostics

On the inspected validation segment (timesteps 12,000–15,999):

| Metric | Result |
|---|---:|
| Precision | 0.600 |
| Recall | 0.980 |
| F1 score | 0.744 |
| True positives | 99 |
| False positives | 66 |
| False negatives | 2 |
| True negatives | 3,833 |

The two missed anomalies were labelled as spikes. False alarms were associated with the drift and freeze rules.

**Important:** this segment was inspected during development and contains no labelled drift anomalies. Detector thresholds were also explored using the test set. These results must not be presented as an unbiased final estimate of generalization.

## Limitations and future work

- Evaluate on data that has not been used for threshold or detector selection.
- Assess drift, freeze, and spike performance separately on representative examples.
- Investigate false alarms and missed anomalies without changing the frozen detector retrospectively.
- Develop causal alternatives for rules that currently use future observations.
- Verify that the notebook runs from top to bottom in a fresh environment.
- Consolidate plots and document the synthetic data-generation process.

## Reproducibility

1. Clone or download the repository.
2. Install the required Python packages.
3. Place the dataset at the path expected by the notebook.
4. Open the notebook in Jupyter.
5. Restart the kernel and run all cells from top to bottom.

Dependencies used in the notebook include Python, NumPy, pandas, Matplotlib, and scikit-learn. Confirm the exact package versions and dataset path before publishing.

## Project structure

A suggested repository layout:

```text
satellite-telemetry-anomaly-detection/
├── data/
│   └── telemetry.csv
├── notebooks/
│   └── satellite_telemetry_anomaly_detection_labeled.ipynb
├── README.md
└── requirements.txt
```

Adjust filenames and paths to match the files actually committed.

## Disclaimer

This is an educational project using synthetic telemetry. It is not a flight-qualified anomaly-detection system and has not been validated for operational spacecraft use.