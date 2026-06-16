# Scenario 4.2.4: Latency Evaluation #4A

## Overview

This scenario evaluates the **impact of different traffic loads on latency** in a **contention scenario involving four overlapping BSSs**. It investigates the **latency anomaly** reported in [6], where Single-Link Operation (SLO) exhibits **lower latency than Multi-Link Operation (MLO)**.

**Based on:** 
- Paper [6] and Section 3.2.9 of the thesis
- "Performance Analysis of MLO in IEEE 802.11 Networks"

## Scenario Description

### Network Topology

Four independent BSSs (A, B, C, D) operating in overlapping channels:
- Each BSS: 1 AP + 1 STA (downlink traffic only)
- All APs and STAs are within mutual communication range (~10 meters apart)
- 4 orthogonal 80 MHz channels available:
  - 5 GHz: Channel 1 (5005 MHz), Channel 100 (5500 MHz)
  - 6 GHz: Channel 1 (5955 MHz), Channel 100 (6425 MHz)

### Modes of Operation

Three modes with different channel allocation schemes:

#### 1. **SLO (Single-Link Operation)** – No Contention
```
BSS A: Channel 1 (5 GHz)
BSS B: Channel 100 (5 GHz)
BSS C: Channel 1 (6 GHz)
BSS D: Channel 100 (6 GHz)
```
- Each BSS has **1 exclusive channel** (80 MHz)
- No contention between BSSs
- Baseline for comparison

#### 2. **MLO-STR:2 (Spatial Reuse with 2 Links)** – Light Contention
```
BSS A: Channels 1, 100 (5 GHz) – shared with BSS B
BSS B: Channels 1, 100 (5 GHz) – shared with BSS A
BSS C: Channels 1, 100 (6 GHz) – shared with BSS D
BSS D: Channels 1, 100 (6 GHz) – shared with BSS C
```
- Each BSS has **2 channels**, shared with one other BSS
- Moderate multi-link contention
- Device contention pairs: (A↔B) and (C↔D)

#### 3. **MLO-STR:4 (Spatial Reuse with 4 Links)** – Heavy Contention
```
BSS A: Channels 1, 100 (5 GHz), 1, 100 (6 GHz)
BSS B: Channels 1, 100 (5 GHz), 1, 100 (6 GHz)
BSS C: Channels 1, 100 (5 GHz), 1, 100 (6 GHz)
BSS D: Channels 1, 100 (5 GHz), 1, 100 (6 GHz)
```
- All 4 BSSs share **all 4 channels**
- High multi-link contention
- Each BSS can potentially block the others

### Physical Parameters

| Parameter | Value |
|-----------|-------|
| Band | 5 GHz & 6 GHz |
| Channels | 1, 100 (in each band) |
| Channel Width | 80 MHz |
| AP-STA Distance | 5 m |
| Propagation Model | Friis (line-of-sight) |
| PHY Standard | 802.11be (EHT) |
| MCS | 8 |
| MPDU Aggregation | 1024 |
| Spatial Streams | 2 |
| Guard Interval | 800 ns |

### Traffic Model

- **Direction:** Downlink only (AP → STA)
- **Type:** Constant-rate OnOff traffic
- **Distribution:** Symmetric across all BSSs (total load / 4 per BSS)
- **Packet Size:** 1472 bytes
- **Traffic Loads:** 100, 1000, 2500 Mbps (total)

### Key Expected Behaviors

#### Low Load (100 Mbps)
- **Minimal contention:** STAs have ample opportunities to transmit
- **Positive MLO impact:** Multiple links reduce latency (STR exploits link diversity)
- **Result:** MLO latency < SLO latency (as in scenario 4.2.3)

#### High Load (1000, 2500 Mbps)
- **Severe contention:** Multiple BSSs frequently compete for channels
- **Anomaly occurs:** Multi-link devices consume many channels, blocking neighbors
- **Starvation effect:** STAs waiting for all links to be free
- **Result:** MLO latency > SLO latency (the anomaly)

## Files

### 1. `wifi-mlo-latency-scenario4a.cc`
**C++ NS-3 simulation executable**

**Key Features:**
- 4 BSSs (A, B, C, D)
- Dynamic channel assignment based on `--strMode`
- Flow monitor for latency collection
- Percentile latency calculation (50th, 95th, 99th)

**Command-line Parameters:**
```bash
--offeredLoad=VALUE    Total traffic load (100–2500 Mbps)
--strMode=VALUE        Channel allocation: SLO, STR2, or STR4
--simTime=VALUE        Simulation duration (default 2.0 seconds)
--RngRun=VALUE         Random seed index (default 1)
--verbose              Enable detailed logging
```

**Output Format:**
```
mean_latency,p50_latency,p95_latency,p99_latency,throughput
```

**Example:**
```bash
./ns3 run "scratch/4.2.4/wifi-mlo-latency-scenario4a --offeredLoad=1000 --strMode=STR2 --RngRun=1"
```

### 2. `run_all_rerun.py`
**Python runner script for batch execution**

**Sweep Parameters:**
- Traffic loads: **100, 1000, 2500 Mbps**
- STR modes: **SLO, STR2, STR4**
- RNG runs: **1–10** (per configuration)
- **Total: 3 × 3 × 10 = 90 simulations**

**Features:**
- Parallel execution (8 workers by default)
- Progress persistence (resume on crash)
- Automatic result aggregation
- CSV summary output

**Output Files:**
- `results/L{load}_M{mode}_R{rng}_result.csv` – Individual run results
- `results/summary.csv` – Aggregated statistics per (load, mode)
- `results/progress.json` – Execution progress tracking
- `plots/4.2.4_latency_vs_load.svg` – Delay plot with p50/p95/p99 stacked bars
- `plots/4.2.4_throughput_vs_load.svg` – Aggregate throughput vs. load

### 3. `plot_results.py`
**Python plotting utility**

- Reads `results/summary.csv`
- Produces publication-ready SVG figures in `plots/`
- Supports legacy summary files without `P95 (ms)` column

**Usage:**
```bash
python3 scratch/4.2.4/plot_results.py
```

**Usage:**
```bash
# Run all 90 simulations
python3 scratch/4.2.4/run_all_rerun.py

# Resume interrupted run (automatic)
python3 scratch/4.2.4/run_all_rerun.py
```

## Execution Instructions

### Single Run
```bash
cd /Users/pwityk/ns-3.45

# Test with low load, short duration
./ns3 run "scratch/4.2.4/wifi-mlo-latency-scenario4a --offeredLoad=100 --strMode=SLO --simTime=2 --RngRun=1"

# Production run: high load, STR:4 mode
./ns3 run "scratch/4.2.4/wifi-mlo-latency-scenario4a --offeredLoad=2500 --strMode=STR4 --simTime=2 --RngRun=1"
```

### Batch Execution
```bash
cd /Users/pwityk/ns-3.45/scratch/4.2.4

# Run all 90 simulations in parallel
python3 run_all_rerun.py

# Re-generate plots only
python3 plot_results.py

# Monitor progress in another terminal
watch "tail -20 results/progress.json"
```

## Expected Results

Based on [6] and section 3.2.9, the expected latency trends are:

### 50th Percentile Latency
```
Load: 100 Mbps
  SLO:     ~0.5 ms
  STR:2:   ~0.4 ms (↓ improvement)
  STR:4:   ~0.3 ms (↓ improvement)

Load: 2500 Mbps
  SLO:     ~8 ms
  STR:2:   ~12 ms (↑ degradation - anomaly)
  STR:4:   ~14 ms (↑ degradation - anomaly)
```

### 99th Percentile Latency
```
Load: 100 Mbps
  SLO:     ~2 ms
  STR:2:   ~1.5 ms
  STR:4:   ~1 ms

Load: 2500 Mbps
  SLO:     ~10 ms
  STR:2:   ~14 ms (anomaly emerges strongly)
  STR:4:   ~16 ms (anomaly most pronounced)
```

## Key Findings (from [6])

### The Latency Anomaly Explanation

1. **Multi-link contention:** When STR devices access multiple links frequently (24–34% at high load), they occupy channels that contending neighbors need
   
2. **Channel starvation:** A BSS with STR occasionally finds all 4 links occupied simultaneously, causing complete deferral

3. **Packet aggregation side-effect:** Due to longer backoff times, STR devices aggregate more packets when they finally transmit, creating long channel occupancy

4. **Result:** The 99th percentile latency increases (anomaly) despite the median being relatively stable

### Takeaway
*In the presence of high load and contention, STR EMLMR devices frequently access multiple links, thereby occasionally blocking contending neighbors for long periods of time and causing larger delays than those experienced by legacy SLO under a static orthogonal channel allocation.*

## Troubleshooting

### Build Issues
```bash
# Clean and rebuild
cd /Users/pwityk/ns-3.45
./ns3 clean
./ns3 build scratch/4.2.4/wifi-mlo-latency-scenario4a

# Verify compilation
./ns3 run scratch/4.2.4/wifi-mlo-latency-scenario4a --help
```

### Runtime Errors

**Error: "Unknown STR mode"**
- Use only: `SLO`, `STR2`, or `STR4` (case-sensitive)

**Error: "No packets received"**
- Increase `simTime` (default 2s should be sufficient)
- Check traffic generation logic

**Python Script Hangs**
- Check `results/progress.json` for stuck jobs
- Kill and restart (automatic resume)
- Increase timeout in runner script if needed

## Comparison with Other Scenarios

| Aspect | 4.2.1 | 4.2.2 | 4.2.3 | 4.2.4 |
|--------|-------|-------|-------|-------|
| **Scenario Name** | Throughput #1 | Throughput #2 | Latency #3 | **Latency #4A** |
| **Focus** | Multi-link benefit | Load scaling | Latency modes | **Contention anomaly** |
| **BSSs** | 1 | 1 | 1 | **4 (contending)** |
| **Links** | 1, 2 | 1, 2, 4 | 1, 2, 3 | **Fixed per mode** |
| **Bands** | 80/160 MHz | 5/6 GHz | 5 GHz | **5/6 GHz** |
| **MLO Modes** | STR | STR | STR, EMLSR | **STR only** |
| **Metric** | Throughput | Throughput | Latency | **Latency + Anomaly** |

## References

- [6] Paper title (from thesis bibliography)
- Section 2.3: Latency Anomaly Overview
- Section 3.2.9: Detailed Scenario Setup
- Section 4.2.4: Results and Analysis
