"""
Generates synthetic satellite telemetry data for V1 of the anomaly detection project.

Channels simulated (loosely inspired by real spacecraft telemetry):
- temperature   (deg C)   - daily orbital thermal cycle
- voltage       (V)       - power bus voltage, tied to solar panel illumination
- current       (A)       - current draw, inversely related to voltage under load
- battery_level (%)       - slow charge/discharge cycle
- orientation_x/y/z (deg) - spacecraft attitude, slow drift + small oscillation

Anomalies injected (labelled in `anomaly` column, but treat this as EVALUATION-ONLY —
a real anomaly detector doesn't get to see these labels during training):
- spikes    : sudden single-timestep jumps (point anomalies)
- drifts    : sustained deviation over a window (collective/contextual anomalies)
- freezes   : sensor "stuck" at a constant value for a window (a real failure mode)

Run: python src/generate_data.py
Output: data/telemetry.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N_TIMESTEPS = 20_000  # ~14 days at 1-minute resolution


def generate_normal_telemetry(n: int) -> pd.DataFrame:
    t = np.arange(n)
    orbital_period = 1500  # timesteps per orbit (~sinusoidal thermal/power cycle)

    temperature = (
        15 * np.sin(2 * np.pi * t / orbital_period)
        + RNG.normal(0, 0.8, n)
        + 5
    )
    illumination = np.clip(np.sin(2 * np.pi * t / orbital_period), 0, None)
    voltage = 28 + 3 * illumination + RNG.normal(0, 0.3, n)
    current = 2.5 - 0.8 * illumination + RNG.normal(0, 0.15, n)
    battery_level = 60 + 25 * np.sin(2 * np.pi * t / orbital_period - 0.5) + RNG.normal(0, 1.5, n)
    battery_level = np.clip(battery_level, 0, 100)

    orientation_x = 2 * np.sin(2 * np.pi * t / (orbital_period * 3)) + RNG.normal(0, 0.2, n)
    orientation_y = 1.5 * np.cos(2 * np.pi * t / (orbital_period * 2)) + RNG.normal(0, 0.2, n)
    orientation_z = RNG.normal(0, 0.3, n).cumsum() * 0.001  # slow random-walk drift

    return pd.DataFrame({
        "temperature": temperature,
        "voltage": voltage,
        "current": current,
        "battery_level": battery_level,
        "orientation_x": orientation_x,
        "orientation_y": orientation_y,
        "orientation_z": orientation_z,
    })


def inject_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)
    df["anomaly"] = 0
    df["anomaly_type"] = "none"

    n_spikes = 15
    n_drifts = 6
    n_freezes = 6

    used = np.zeros(n, dtype=bool)

    def free_start(min_gap=200, window=1):
        for _ in range(200):
            s = RNG.integers(500, n - 500)
            if not used[max(0, s - min_gap): s + window + min_gap].any():
                return s
        return None

    channels = df.columns[:-2].tolist()

    # Point anomalies: single-timestep spikes
    for _ in range(n_spikes):
        s = free_start(window=1)
        if s is None:
            continue
        ch = RNG.choice(channels)
        direction = RNG.choice([-1, 1])
        magnitude = RNG.uniform(4, 8) * df[ch].std()
        df.loc[s, ch] += direction * magnitude
        df.loc[s, "anomaly"] = 1
        df.loc[s, "anomaly_type"] = "spike"
        used[s] = True

    # Contextual/collective anomalies: sustained drift over a window
    for _ in range(n_drifts):
        window = RNG.integers(30, 80)
        s = free_start(window=window)
        if s is None:
            continue
        ch = RNG.choice(channels)
        magnitude = RNG.uniform(2, 4) * df[ch].std()
        ramp = np.linspace(0, magnitude, window)
        df.loc[s: s + window - 1, ch] += ramp
        df.loc[s: s + window - 1, "anomaly"] = 1
        df.loc[s: s + window - 1, "anomaly_type"] = "drift"
        used[s: s + window] = True

    # Sensor freezes: value gets "stuck" for a window (common real failure mode)
    for _ in range(n_freezes):
        window = RNG.integers(20, 60)
        s = free_start(window=window)
        if s is None:
            continue
        ch = RNG.choice(channels)
        stuck_value = df.loc[s, ch]
        df.loc[s: s + window - 1, ch] = stuck_value
        df.loc[s: s + window - 1, "anomaly"] = 1
        df.loc[s: s + window - 1, "anomaly_type"] = "freeze"
        used[s: s + window] = True

    return df


def main():
    df = generate_normal_telemetry(N_TIMESTEPS)
    df = inject_anomalies(df)
    df.insert(0, "timestep", np.arange(len(df)))

    out_path = "data/telemetry.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    print(f"Anomalous timesteps: {df['anomaly'].sum()} ({df['anomaly'].mean()*100:.2f}%)")
    print(df["anomaly_type"].value_counts())


if __name__ == "__main__":
    main()
