#!/usr/bin/env python3
"""
4.2.8 — EMLSR parameter summary table + consolidated comparison plot.

Produces:
  - thesis_tables/4_2_8_emlsr_summary.md   (markdown table)
  - thesis_tables/4_2_8_emlsr_summary.tex  (LaTeX table)
  - plots/4_2_8_summary.svg, .pdf          (consolidated comparison figure)

The detailed per-parameter throughput / latency plots remain produced by
plot_emlsr_params.py — this script is the high-level overview.
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                                 _os.pardir, "_common"))
import thesis_style as _thesis_style  # noqa: F401

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

# ===== CONFIG =====
OUTPUT_DIR   = "thr_runs_4_2_8"
PLOT_DIR     = "plots"
TABLE_DIR    = "thesis_tables"
NSTA_VALUES  = [1, 4]
CONFIDENCE   = 0.95

# Parameter metadata: values, default, human description, preferred direction,
# why it matters. `prefer` is "less" / "more" / "on" / "off" / "—" (no impact).
PARAMS = {
    "transDelay": dict(
        values=[0, 16, 32, 64, 128, 256], default=0, unit="µs", prefer="less",
        description=(
            "Microseconds the STA needs after receiving an EMLSR Action frame "
            "before its main PHY can actually start receiving on the other link. "
            "Pure dead time during which neither link is usable."
        ),
        impact=(
            "Less is better. Every µs is air-time lost on both links. Real Wi-Fi 7 "
            "chips report 16–32 µs; values above ~64 µs are abnormal and exist "
            "only for stress-testing."
        ),
    ),
    "padDelay": dict(
        values=[0, 32, 64, 128, 256], default=0, unit="µs", prefer="less",
        description=(
            "Padding bytes the AP prepends to every downlink frame to give an "
            "EMLSR client time to wake / refocus its main PHY for the new link "
            "before the data starts arriving."
        ),
        impact=(
            "Less is better. Padding inflates every DL transmission, eating "
            "throughput and adding head-of-line latency. Set to 0 unless the "
            "STA reports a wake-up budget."
        ),
    ),
    "timeout": dict(
        values=[128, 256, 512, 1024, 2048, 4096, 8192, 16384], default=128, unit="µs", prefer="—",
        description=(
            "Transition Timeout: how long the AP waits for an EMLSR Action-frame "
            "round-trip to complete before declaring the EMLSR setup failed. "
            "Used only during EMLSR mode entry / exit, not in steady state."
        ),
        impact=(
            "Doesn't matter for steady-state throughput or latency — both stay "
            "flat across the whole 128 µs … 16 ms range. Pick any value that "
            "fits link-budget assumptions for mode negotiation."
        ),
    ),
    "auxWidth": dict(
        values=[20, 40, 80], default=80, unit="MHz", prefer="more",
        description=(
            "Channel bandwidth the auxiliary PHY (listening-only radio) is "
            "capable of. Determines whether the aux can decode/demodulate a "
            "full 80 MHz frame on the secondary link or only a narrow slice."
        ),
        impact=(
            "More is better. With aux width below the operating bandwidth, the "
            "main PHY has to be switched for every wider TX, raising latency. "
            "Match the operating channel width (80 MHz here)."
        ),
    ),
    "sleep": dict(
        values=[0, 1], default=0, unit="bool", prefer="off",
        description=(
            "Aux PHY enters a low-power sleep state when idle. Saves battery on "
            "the STA but loses signal-detection capability until it wakes."
        ),
        impact=(
            "Off is better for throughput / latency (aux stays alert and reacts "
            "instantly). On is the right choice when the priority is battery "
            "life — but the sweep shows no measurable steady-state impact in "
            "this saturated single-BSS test."
        ),
    ),
    "txCap": dict(
        values=[0, 1], default=1, unit="bool", prefer="on",
        description=(
            "Whether the auxiliary PHY may itself transmit. If off, uplink "
            "frames destined for the aux's link force a main-PHY switch first."
        ),
        impact=(
            "On is better. With txCap = on, EMLSR can opportunistically pick "
            "either PHY for UL → lower channel-access latency (≈ 16 % drop in "
            "this sweep). Throughput differences are within noise."
        ),
    ),
    "switchAux": dict(
        values=[0, 1], default=1, unit="bool", prefer="on",
        description=(
            "Whether the auxiliary PHY follows the main PHY when it switches "
            "active links — so the previously-active link still has a radio "
            "monitoring it and EMLSR remains symmetric."
        ),
        impact=(
            "On is dramatically better — by far the most impactful setting in "
            "the whole sweep (≈ 14 % more throughput, ≈ 81 % lower latency). "
            "Off leaves one link unmonitored after every switch, breaking "
            "EMLSR's whole symmetry guarantee."
        ),
    ),
}


def ci_mean(values, confidence=CONFIDENCE):
    n = len(values)
    if n == 0:
        return float("nan"), 0.0
    if n == 1:
        return float(values[0]), 0.0
    arr = np.asarray(values, dtype=float)
    m = float(arr.mean())
    se = sp_stats.sem(arr)
    h = sp_stats.t.ppf((1 + confidence) / 2, n - 1) * se
    return m, float(h)


def load_results():
    """data[param][value][nStas] = list of (latency_ms, throughput_Mbps)."""
    pattern = re.compile(r"^P(\w+)_V([^_]+)_S(\d+)_rng\d+_result\.csv$")
    data = {p: {v: {ns: [] for ns in NSTA_VALUES}
                for v in meta["values"]}
            for p, meta in PARAMS.items()}
    if not os.path.isdir(OUTPUT_DIR):
        return data
    for fname in os.listdir(OUTPUT_DIR):
        m = pattern.match(fname)
        if not m:
            continue
        param, val_s, ns = m.group(1), m.group(2), int(m.group(3))
        if param not in data or ns not in NSTA_VALUES:
            continue
        try:
            val = int(val_s)
        except ValueError:
            val = float(val_s)
        if val not in data[param]:
            continue
        try:
            with open(os.path.join(OUTPUT_DIR, fname)) as f:
                lat, thr = (float(x) for x in f.read().strip().split(","))
        except (ValueError, IndexError):
            continue
        data[param][val][ns].append((lat, thr))
    return data


def aggregate(data):
    """Return summary[param][nStas] = list of per-value dicts {val, thr_mean, lat_mean, ...}."""
    summary = {}
    for param, meta in PARAMS.items():
        summary[param] = {}
        for ns in NSTA_VALUES:
            rows = []
            for v in meta["values"]:
                pairs = data[param][v][ns]
                if not pairs:
                    rows.append(dict(val=v, n=0, thr_mean=float("nan"),
                                     thr_ci=0.0, lat_mean=float("nan"), lat_ci=0.0))
                    continue
                thrs = [t for _, t in pairs]
                lats = [la for la, _ in pairs]
                tm, tc = ci_mean(thrs)
                lm, lc = ci_mean(lats)
                rows.append(dict(val=v, n=len(pairs),
                                 thr_mean=tm, thr_ci=tc,
                                 lat_mean=lm, lat_ci=lc))
            summary[param][ns] = rows
    return summary


def fmt(v, prec=1):
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return "—"
    return f"{v:.{prec}f}"


_PREFER_ICON = {"less": "↓ less", "more": "↑ more",
                 "on": "● on", "off": "○ off", "—": "—"}


def write_markdown_table(summary):
    """Write a concise EMLSR parameter summary as markdown."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    out = os.path.join(TABLE_DIR, "4_2_8_emlsr_summary.md")
    lines = []
    lines.append("# 4.2.8 — EMLSR parameter summary\n")
    lines.append(
        "Per-parameter sweep at fixed load (ρ=0.7), 2 EMLSR links (5 GHz ch42 + 6 GHz ch7), "
        "EHT MCS 11, A-MPDU 1024, 5 RNG seeds per data point, simTime 3 s. "
        "Channel access delay is the MAC-to-PHY interval (Mac TX → Phy TX begin).\n"
    )
    lines.append("**Prefer column legend:** ↓ less = smaller value is better, "
                 "↑ more = larger value is better, ● on / ○ off = boolean preference, "
                 "— = no measurable steady-state impact (set to any value).\n")
    lines.append("| Parameter | Description | Tested | Prefer | Δ thr | Δ lat | Why |")
    lines.append("|---|---|---|---|---|---|---|")
    for param, meta in PARAMS.items():
        rows = summary[param][4]  # n_STA = 4 is the more sensitive case
        thrs = [r["thr_mean"] for r in rows if not np.isnan(r["thr_mean"])]
        lats = [r["lat_mean"] for r in rows if not np.isnan(r["lat_mean"])]
        if not thrs or not lats:
            continue
        thr_range = max(thrs) - min(thrs)
        lat_range = max(lats) - min(lats)
        thr_pct = 100.0 * thr_range / max(thrs) if max(thrs) > 0 else 0.0
        lat_pct = 100.0 * lat_range / max(lats) if max(lats) > 0 else 0.0
        values_s = ", ".join(str(v) for v in meta["values"])
        unit_suffix = f" {meta['unit']}" if meta["unit"] != "bool" else ""
        prefer_s = _PREFER_ICON.get(meta["prefer"], meta["prefer"])
        line = (
            f"| `{param}` | {meta['description']} | "
            f"{values_s}{unit_suffix} | **{prefer_s}** | "
            f"{thr_pct:.0f} % ({thr_range:.0f} Mb/s) | "
            f"{lat_pct:.0f} % ({lat_range:.2f} ms) | "
            f"{meta['impact']} |"
        )
        lines.append(line)
    lines.append("")
    lines.append("**Δ thr / Δ lat** are the relative (%) and absolute span across "
                 "the tested values at n_STA = 4 (the more sensitive case). "
                 "Small Δ → parameter has little steady-state effect.\n")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")
    return out


_PREFER_TEX = {"less": r"$\downarrow$ less", "more": r"$\uparrow$ more",
                "on": "on", "off": "off", "—": "---"}


def write_latex_table(summary):
    """LaTeX version for thesis."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    out = os.path.join(TABLE_DIR, "4_2_8_emlsr_summary.tex")
    lines = []
    lines.append(r"% 4.2.8 EMLSR parameter summary table — generated from summary_table.py")
    lines.append(r"\begin{table}[H]")
    lines.append(r"  \centering")
    lines.append(r"  \footnotesize")
    lines.append(
        r"  \caption{EMLSR parameter sensitivity. The \emph{Prefer} column "
        r"indicates the direction (\textit{less}, \textit{more}, \textit{on}, "
        r"\textit{off}, or no preference) that maximises throughput / minimises "
        r"channel-access delay. $\Delta$ values are the relative spread across "
        r"the tested range at $n_{STA}=4$, $\rho=0.7$. A near-zero $\Delta$ "
        r"means the parameter has no measurable steady-state effect.}"
    )
    lines.append(r"  \label{tab:emlsr_params}")
    lines.append(r"  \begin{tabular}{@{}lp{4.0cm}lrr@{}}")
    lines.append(r"    \toprule")
    lines.append(r"    Parameter & Description & Prefer & $\Delta$ thr & $\Delta$ lat \\")
    lines.append(r"    \midrule")
    for param, meta in PARAMS.items():
        rows = summary[param][4]
        thrs = [r["thr_mean"] for r in rows if not np.isnan(r["thr_mean"])]
        lats = [r["lat_mean"] for r in rows if not np.isnan(r["lat_mean"])]
        if not thrs or not lats:
            continue
        thr_pct = 100.0 * (max(thrs) - min(thrs)) / max(thrs) if max(thrs) > 0 else 0.0
        lat_pct = 100.0 * (max(lats) - min(lats)) / max(lats) if max(lats) > 0 else 0.0
        prefer_s = _PREFER_TEX.get(meta["prefer"], meta["prefer"])
        lines.append(
            r"    \texttt{" + param.replace("_", r"\_") + r"} & "
            + meta["description"].replace("→", r"$\rightarrow$") + r" & "
            + f"{prefer_s} & "
            + f"{thr_pct:.0f}\\% & {lat_pct:.0f}\\% \\\\"
        )
    lines.append(r"    \bottomrule")
    lines.append(r"  \end{tabular}")
    lines.append(r"\end{table}")
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"Wrote {out}")
    return out


def plot_summary(summary):
    """One 2-row × 7-column comparison figure: throughput and latency normalised
    to each parameter's own baseline value."""
    fig, axes = plt.subplots(2, len(PARAMS), figsize=(2.6 * len(PARAMS), 5.4),
                              sharey="row")
    if len(PARAMS) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    for col, (param, meta) in enumerate(PARAMS.items()):
        # Top: throughput sensitivity
        ax_t = axes[0, col]
        ax_l = axes[1, col]
        for ns, marker, color in [(1, "o", "#2c5aa0"), (4, "s", "#c0392b")]:
            rows = summary[param][ns]
            xs = [r["val"] for r in rows if not np.isnan(r["thr_mean"])]
            ts = [r["thr_mean"] for r in rows if not np.isnan(r["thr_mean"])]
            ls = [r["lat_mean"] for r in rows if not np.isnan(r["lat_mean"])]
            if xs:
                ax_t.plot(xs, ts, marker=marker, color=color,
                          linewidth=1.4, markersize=5,
                          label=f"{ns} STA{'s' if ns > 1 else ''}")
                ax_l.plot(xs, ls, marker=marker, color=color,
                          linewidth=1.4, markersize=5)
        ax_t.set_title(f"{param}", fontsize=10)
        unit_s = "" if meta["unit"] == "bool" else f" [{meta['unit']}]"
        ax_l.set_xlabel(f"{param}{unit_s}", fontsize=9)
        if param == "timeout":
            ax_t.set_xscale("log", base=2)
            ax_l.set_xscale("log", base=2)
        if meta["unit"] == "bool":
            ax_t.set_xticks([0, 1])
            ax_l.set_xticks([0, 1])
            ax_t.set_xticklabels(["off", "on"])
            ax_l.set_xticklabels(["off", "on"])
        for ax in (ax_t, ax_l):
            ax.grid(True, linestyle="--", alpha=0.35)
        ax_t.tick_params(axis='x', labelbottom=False)

    axes[0, 0].set_ylabel("Throughput [Mbit/s]")
    axes[1, 0].set_ylabel("Channel access delay [ms]")
    axes[0, -1].legend(loc="lower right", fontsize=8, framealpha=0.9)
    fig.suptitle("4.2.8 — EMLSR parameter sensitivity (ρ=0.7, MCS 11, A-MPDU 1024)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    base = os.path.join(PLOT_DIR, "4_2_8_summary")
    fig.savefig(base + ".svg", format="svg", bbox_inches="tight")
    fig.savefig(base + ".pdf", format="pdf", bbox_inches="tight")
    print(f"Wrote {base}.{{svg,pdf}}")
    plt.close(fig)


def main():
    data = load_results()
    summary = aggregate(data)
    write_markdown_table(summary)
    write_latex_table(summary)
    plot_summary(summary)


if __name__ == "__main__":
    main()
