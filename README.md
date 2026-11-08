# Satellite Telemetry Anomaly Detection

A Python-based machine learning project exploring anomaly detection in synthetic satellite telemetry. The project combines temporal feature engineering, an Isolation Forest baseline, and rule-based detectors to identify unusual patterns such as gradual drift, frozen sensor readings, and sudden spikes.

> **Project status:** Exploratory portfolio project. Reported metrics are development diagnostics, not an unbiased estimate of performance on unseen data.

## 1. Project Objective

Satellite telemetry consists of measurements collected from different onboard sensors and subsystems. Unusual patterns in these measurements may indicate sensor faults or abnormal system behaviour.

This project explores how multivariate time-series data can be analysed to identify three types of anomalies:

- **Drift:** Gradual or sustained changes in a telemetry signal.
- **Freeze:** Sensor readings that remain unchanged for an extended period.
- **Spike:** Sudden changes that reverse in the following sample.

The project uses synthetic data for experimentation. Its results should not be interpreted as validation on real spacecraft telemetry.

## 2. Dataset

The synthetic dataset contains **20,000 timesteps**, seven telemetry channels, and anomaly labels.

### Telemetry features

| Feature | Description |
|---|---|
| `temperature` | Temperature measurement |
| `voltage` | Voltage measurement |
| `current` | Current measurement |
| `battery_level` | Battery-level measurement |
| `orientation_x` | X-axis orientation |
| `orientation_y` | Y-axis orientation |
| `orientation_z` | Z-axis orientation |

Additional columns:

- `timestep`: Time-step index.
- `anomaly`: Binary anomaly label.
- `anomaly_type`: Label describing the anomaly category.

The dataset contains **582 labelled anomalies (2.91%)**.

## 3. Methodology

### 3.1 Temporal Feature Engineering

The project constructs 42 features across the seven telemetry channels.

These include:

- Raw sensor values.
- Differences between consecutive readings.
- Rolling variability.
- Deviation from a trailing mean.
- Counts of unchanged readings.

The purpose is to capture both the current sensor state and changes in its recent behaviour.

### 3.2 Isolation Forest Baseline

An Isolation Forest is used as an unsupervised anomaly-detection baseline.

The model is trained using normal observations. A normal-only calibration subset is used to establish an anomaly-score threshold.

### 3.3 Hybrid Anomaly Detection

The notebook explores a combination of model-based and rule-based detection approaches.

The hybrid approach includes:

- Isolation Forest anomaly scores.
- Drift detection rules.
- Frozen-reading detection.
- Spike detection.
- Short-freeze detection.

Exploratory model variants and threshold comparisons are retained in the notebook to document the development process.

**Important implementation limitation:** Some rules use future samples or full-run information. These components are retrospective and should not be interpreted as strictly real-time detectors.

## 4. Exploratory Performance Diagnostics

The following results were obtained on the inspected segment covering timesteps **12,000–15,999**.

| Metric | Result |
|---|---:|
| Precision | 0.600 |
| Recall | 0.980 |
| F1 score | 0.744 |
| True positives | 99 |
| False positives | 66 |
| False negatives | 2 |
| True negatives | 3,833 |

### Understanding the results

- **Precision (0.600):** Approximately 60% of the detected anomalies in this segment were labelled anomalies.
- **Recall (0.980):** Approximately 98% of the labelled anomalies in this segment were detected.
- **F1 score (0.744):** A combined measure of precision and recall.

The two missed anomalies were labelled as spikes. False alarms were associated with the drift and freeze rules.

> **Evaluation caution:** This segment was inspected during development and contains no labelled drift anomalies. Detector thresholds were also explored using the test set. Consequently, these metrics are exploratory diagnostics and must not be presented as an unbiased estimate of generalisation performance.

## 5. Limitations and Future Work

Potential improvements include:

- Evaluate the detector on data that has not been used for threshold or detector selection.
- Assess drift, freeze, and spike detection separately using representative examples.
- Investigate false alarms and missed anomalies using a separate evaluation procedure.
- Develop causal alternatives for rules that currently use future observations.
- Verify that the notebook runs from beginning to end in a fresh environment.
- Further document the synthetic data-generation process.
- Explore additional anomaly-detection methods and compare their performance fairly.

## 6. Installation and Reproducibility

### Prerequisites

- Python 3.10 or compatible version.
- Git.
- Jupyter Notebook or JupyterLab.

### Clone the repository

```bash
git clone https://github.com/A2ZRahul/satellite-telemetry-anomaly-detection.git
cd satellite-telemetry-anomaly-detection
```

### Create and activate a virtual environment

Windows (Git Bash):

```bash
python -m venv venv
source venv/Scripts/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Launch Jupyter Notebook

```bash
jupyter notebook
```

Open the main notebook:

`notebooks/satellite_telemetry_anomaly_detection.ipynb`

Restart the kernel and run all cells from top to bottom.

The notebook expects the dataset at:

`data/telemetry.csv`

## 7. Repository Structure

```text
satellite-telemetry-anomaly-detection/
│
├── data/
│   └── telemetry.csv
│
├── notebooks/
│   ├── 01_eda..ipynb
│   └── satellite_telemetry_anomaly_detection.ipynb
│
├── src/
│   └── generate_data.py
│
├── .gitignore
├── README.md
└── requirements.txt
```

## 8. Disclaimer

This is an educational machine learning project using synthetic satellite telemetry. It is not a flight-qualified anomaly-detection system and has not been validated for operational spacecraft use.

**Author:** Rahul Panda  
**GitHub:** [A2ZRahul](https://github.com/A2ZRahul)