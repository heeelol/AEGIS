
### Table 3.4.3.1: Pipeline decoupling validation

| Metric | CV Thread Alone | CV Thread + HMI + Load Cell Polling (full load) | Δ | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **CV thread FPS** | 29.7 FPS | 29.6 FPS | -0.1 FPS (-0.4%) | CV processing rate stays identical under full load. |
| **End-to-end alert latency (mean, ms)** | 13.87 ms | 12.75 ms | -1.12 ms | Includes shared memory sync + HTTP serialization. |
| **End-to-end alert latency (max, ms)** | 29.18 ms | 14.82 ms | -14.36 ms | Peak HTTP network query overhead. |
| **HMI refresh responsiveness** | N/A (HMI disabled) | 2.11 ms mean (3.4 ms max) | N/A | Measures FastAPI responsiveness under full concurrent request load. |
| **CPU utilization (% / cores used)** | 0.9% / 0.01 cores | 6.2% / 0.06 cores | N/A | Demonstrates multi-threaded core distribution. |
| **Load-cell polling rate (Hz)** | N/A (Disabled) | 11.3 Hz | N/A | Confirms load cell background serial ingestion rate. |
