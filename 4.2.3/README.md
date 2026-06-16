# Latency Scenario #3 (4.2.3)

**Based on:** [15] Kozioł et al., "Performance Analysis of MLO in IEEE 802.11 Networks" (accepted work)

## Overview

Scenario 4.2.3 evaluates how **link configurations** and **MLO modes (STR/EMLSR)** affect downlink latency in multi-link 802.11be networks.

### Key Parameters

| Parameter | Value |
|-----------|-------|
| **Band** | 5 GHz only |
| **Channels** | 1, 100, 136 (80 MHz each) |
| **Number of Links** | 1, 2, or 3 |
| **Number of Stations** | 1 |
| **MLO Modes** | STR (default) or EMLSR |
| **PHY Standard** | 802.11be EHT |
| **MCS** | 8 |
| **Spatial Streams** | 2 |
| **Guard Interval** | 800 ns |
| **A-MPDU Limit** | 1024 MPDUs |
| **AP–STA Distance** | 5 m (line of sight) |
| **Traffic Model** | Constant-rate On-Off (downlink) |

## Topology

```
5 GHz Channel 1      5 GHz Channel 100    5 GHz Channel 136
     ↓                    ↓                     ↓
  [AP]  ←────────────────→  [STA]  ←─────────────→
   |                                             |
   └─────────────────────────────────────────────┘
           5m line-of-sight
```

### Link Configuration Modes

- **L=1 (SLO):** Single Link Operation
  - 5 GHz Channel 1, 80 MHz

- **L=2 (STR/EMLSR):** Dual Link Operation
  - Link 0: 5 GHz Channel 1, 80 MHz
  - Link 1: 5 GHz Channel 100, 80 MHz

- **L=3 (STR/EMLSR):** Triple Link Operation
  - Link 0: 5 GHz Channel 1, 80 MHz
  - Link 1: 5 GHz Channel 100, 80 MHz
  - Link 2: 5 GHz Channel 136, 80 MHz

### MLO Modes

- **STR (Simultaneous Transmit/Receive):** Default mode
  - AP and STA can transmit/receive on multiple links simultaneously
  - Parallel transmission increases throughput and reduces latency

- **EMLSR (Enhanced Multi-Link Single-Radio):** When enabled
  - STA reduces power consumption by operating on select links
  - Provides latency/power-efficiency trade-off

## Building the Scenario

```bash
cd /Users/pwityk/ns-3.45

# Build the scenario executable
./ns3 build scratch/4.2.3/wifi-mlo-latency-scenario3

# Verify build
./ns3 run "scratch/4.2.3/wifi-mlo-latency-scenario3 --helpFull | grep -E 'numLinks|emlsr|offeredLoad'"
```

## Running Single Simulations

```bash
./ns3 run "scratch/4.2.3/wifi-mlo-latency-scenario3 \
  --simTime=10 \
  --payloadSize=1500 \
  --nMpdus=1024 \
  --numLinks=2 \
  --emlsr=false \
  --offeredLoad=500 \
  --RngRun=1"
```

### Command-Line Arguments

| Argument | Default | Range | Unit |
|----------|---------|-------|------|
| `simTime` | 10.0 | > 0 | seconds |
| `payloadSize` | 1500 | > 0 | bytes |
| `nMpdus` | 1024 | > 0 | MPDUs |
| `numLinks` | 2 | {1, 2, 3} | links |
| `emlsr` | false | {true, false} | boolean |
| `offeredLoad` | 500.0 | > 0 | Mb/s |
| `RngRun` | 1 | ≥ 1 | run index |

### Expected Output

```
=== Latency Scenario #3 ===
numLinks: 2  MLO mode: STR  offeredLoad: 500 Mb/s

Mean DL Latency: X.XXX ms
Mean DL Throughput: X.XX Mb/s
```

## Running Full Sweep (All Combinations)

```bash
cd /Users/pwityk/ns-3.45
chmod +x scratch/4.2.3/run_all_rerun.py
python3 scratch/4.2.3/run_all_rerun.py
```

### Sweep Parameters

- **numLinks:** 1, 2, 3
- **MLO Mode:**
  - STR (emlsr=false) for numLinks = 1, 2, 3
  - EMLSR (emlsr=true) for numLinks = 2, 3
- **Offered Load:** 100 – 2500 Mb/s (25 points, 100 Mb/s steps)
- **Runs:** 10 per configuration

### Total Jobs

```
[(1 link × 1 mode) + (2 links × 2 modes)] × 25 loads × 10 runs = 1,250 simulations
```

### Output Storage

Results are stored in `thr_runs_4_2_3/` directory with filenames:

```
L{numLinks}_M{mode}_Load{load}_rng{run}_result.csv
```

Example:
```
L2_MSTR_Load500_rng1_result.csv    # Link-2, STR mode, 500 Mb/s, run 1
L3_MEMLSR_Load1000_rng5_result.csv # Link-3, EMLSR mode, 1000 Mb/s, run 5
```

Each file contains a single line with comma-separated values:

```
<latency_ms>,<throughput_mbps>
```

After the sweep finishes, the runner also writes a summary figure to [plots/4_2_3.svg](plots/4_2_3.svg).

## Performance Analysis Tools

The batch runner generates a two-panel SVG showing mean latency and mean throughput versus offered load, with separate series for each valid link-count and MLO mode combination.

### 1. Aggregation Script (Future)

After collecting results, aggregate by configuration:

```python
import pandas as pd
import glob

# Load all results
dfs = []
for fname in glob.glob("thr_runs_4_2_3/L*_result.csv"):
    # Parse filename: L{nl}_M{mode}_Load{load}_rng{rng}_result.csv
    parts = fname.split('_')
    nl = int(parts[0][1:])
    mode = parts[1][1:]
    load = int(parts[2][4:])
    rng = int(parts[3][3:])
    
    # Read result
    lat, thr = map(float, open(fname).read().strip().split(','))
    
    dfs.append({'numLinks': nl, 'mode': mode, 'load': load, 'rng': rng, 'latency': lat, 'thr': thr})

df = pd.DataFrame(dfs)
```

### 2. Comparative Analysis

```python
# Group by configuration (mode, numLinks, load) and compute statistics
grouped = df.groupby(['numLinks', 'mode', 'load'])['latency'].agg(['mean', 'std', 'min', 'max'])

# Compare STR vs EMLSR
str_data = df[df['mode'] == 'STR']
emlsr_data = df[df['mode'] == 'EMLSR']

# Improvement metric
improvement = ((str_data['latency'].mean() - emlsr_data['latency'].mean()) / 
               str_data['latency'].mean() * 100)
print(f"EMLSR latency improvement: {improvement:.2f}%")
```

## Troubleshooting

### Configuration Errors

If the scenario doesn't compile:

```bash
# EMLSR is configured through EhtConfiguration (WifiHelper::ConfigEhtOptions).
# Do not set EmlsrActivated on ns3::StaWifiMac attributes.
```

### Runtime Crashes

**Multi-link route population error:**

```
FATAL: Do not call WifiNetDevice::GetChannel() when using multiple channels
```

**Solution:** Already fixed in this scenario — global routing population removed (not needed for this simple AP↔STA topology).

**Invalid EMLSR + single-link combination:**

```
NS_ASSERT failed, cond="mle", msg="AssocReq should contain a Multi-Link Element"
```

**Solution:** EMLSR is valid only for multi-link operation (`numLinks >= 2`). The runner skips invalid `(numLinks=1, mode=EMLSR)` jobs.

### Slow Execution

- Reduce `simTime` for quick tests: `--simTime=2`
- Increase `MAX_WORKERS` in runner if available CPU cores > 8
- Check system load: `top` or `htop`

## Scenario Evolution

**From Paper [15]:**
- Table 3.9 specifies channel plan, modes, and distance
- Latency metric: Mean downlink packet delay [ms]
- Throughput metric: Mean downlink throughput [Mb/s]

**Implementation Notes:**
- Uses Friis propagation loss model (free-space path loss)
- FlowMonitor for latency aggregation across packets
- OnOff traffic captures steady-state offered load behavior
- 1 second warm-up period (tStart=1.0) to stabilize channel

## References

- [15] Kozioł, J., "Performance Analysis of MLO in IEEE 802.11 Networks"
  - Available: https://ieeexplore.ieee.org/document/10118829

## Related Scenarios

- **Scenario 4.2.1:** Normalized-load sweep with 802.11be STR
- **Scenario 4.2.2:** Offered-load sweep across 1/2/4 links (5/6 GHz)
- **Scenario 4.2.3:** This scenario — MLO mode comparison on 5 GHz 1/2/3 links
