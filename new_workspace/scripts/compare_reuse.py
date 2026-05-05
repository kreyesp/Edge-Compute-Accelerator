import accelforge as af
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent

FANOUT_X = 8
FANOUT_Y = 8

LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]

CONFIGS = [
    {
        "label"   : "No Reuse\n(Arch + Workload)",
        "arch"    : str(WORKSPACE / "arches"    / "accelerator_sweep_no_reuse.yaml"),
        "workload": str(WORKSPACE / "workloads" / "tiny_yolo_test_no_reuse.yaml"),
        "jinja"   : {"GLB_KB": 1024, "FREQ_GHZ": 1.0, "FANOUT_X": FANOUT_X, "FANOUT_Y": FANOUT_Y},
    },
    {
        "label"   : "Workload Reuse Only\n(Layer Fusion)",
        "arch"    : str(WORKSPACE / "arches"    / "custom_accelerator_sweep_NON_IDEAL.yaml"),
        "workload": str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml"),
        "jinja"   : {"GLB_KB": 1024, "FREQ_GHZ": 1.0, "FANOUT_X": FANOUT_X, "FANOUT_Y": FANOUT_Y},
    },
    {
        "label"   : "Arch Reuse Only\n(No Workload Fusion)",
        "arch"    : str(WORKSPACE / "arches"    / "custom_accelerator_sweep_NON_IDEAL.yaml"),
        "workload": str(WORKSPACE / "workloads" / "tiny_yolo_test_no_reuse.yaml"),
        "jinja"   : {"GLB_KB": 1024, "FREQ_GHZ": 1.0, "FANOUT_X": FANOUT_X, "FANOUT_Y": FANOUT_Y},
    },
    {
        "label"   : "Local Buffer\n(768KB GLB + 4KB/PE)",
        "arch"    : str(WORKSPACE / "arches"    / "custom_accelerator_sweep_local_buffer.yaml"),
        "workload": str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml"),
        "jinja"   : {"GLB_KB": 768, "LOCAL_KB": 4, "FREQ_GHZ": 1.0, "FANOUT_X": FANOUT_X, "FANOUT_Y": FANOUT_Y},
    },
]

def min_edp_filter(df):
    return (df["Total<SEP>energy"] * df["Total<SEP>latency"]).argmin()

# ---------------------------------------------------------------------------
# Run all configs
# ---------------------------------------------------------------------------
all_results = []

for cfg in CONFIGS:
    print(f"Running: {cfg['label'].replace(chr(10), ' ')}...", end=" ", flush=True)
    spec     = af.Spec.from_yaml(cfg["arch"], cfg["workload"], jinja_parse_data=cfg["jinja"])
    mappings = spec.map_workload_to_arch()
    data     = mappings.data.iloc[[min_edp_filter(mappings.data)]]
    cols     = data.columns.tolist()

    layer_energy  = {}
    layer_latency = {}

    for layer in LAYERS:
        e_cols = [c for c in cols if c.startswith(f"{layer}<SEP>energy<SEP>")]
        l_cols = [c for c in cols if c.startswith(f"{layer}<SEP>latency<SEP>")]
        layer_energy[layer]  = data[e_cols].sum(axis=1).values[0]
        layer_latency[layer] = data[l_cols].max(axis=1).values[0]

    layer_edp = {l: layer_energy[l] * layer_latency[l] for l in LAYERS}

    total_energy  = data["Total<SEP>energy"].values[0]
    total_latency = data["Total<SEP>latency"].values[0]
    total_edp     = total_energy * total_latency

    print(f"EDP={total_edp:.3e} pJ·s")

    all_results.append({
        "label"        : cfg["label"],
        "layer_energy" : layer_energy,
        "layer_latency": layer_latency,
        "layer_edp"    : layer_edp,
        "total_energy" : total_energy,
        "total_latency": total_latency,
        "total_edp"    : total_edp,
        "data"         : data,
    })

# ---------------------------------------------------------------------------
# Print table
# ---------------------------------------------------------------------------
for r in all_results:
    print(f"\n=== {r['label'].replace(chr(10), ' ')} ===")
    print(f"{'Layer':<6}  {'Energy (pJ)':>14}  {'Latency (s)':>14}  {'EDP (pJ·s)':>14}")
    print("-" * 56)
    for l in LAYERS:
        print(f"{l:<6}  {r['layer_energy'][l]:>14.4e}  {r['layer_latency'][l]:>14.4e}  {r['layer_edp'][l]:>14.4e}")
    print("-" * 56)
    print(f"{'Total':<6}  {r['total_energy']:>14.4e}  {r['total_latency']:>14.4e}  {r['total_edp']:>14.4e}")

# ---------------------------------------------------------------------------
# MAC utilization for each config
# ---------------------------------------------------------------------------
NUM_PES    = FANOUT_X * FANOUT_Y
FREQ_GHZ   = 1.0
CYCLE_TIME = 1 / (FREQ_GHZ * 1e9)

for r in all_results:
    print(f"\n=== MAC Utilization: {r['label'].replace(chr(10), ' ')} ===")
    print(f"{'Layer':<6}  {'MAC ops':>12}  {'Total cycles':>14}  {'Min cycles':>12}  {'Utilization':>12}")
    print("-" * 64)
    data = r["data"]
    for layer in LAYERS:
        try:
            mac_ops      = data[f"{layer}<SEP>action<SEP>MAC<SEP>None<SEP>compute"].values[0]
            mac_latency  = data[f"{layer}<SEP>latency<SEP>MAC"].values[0]
            total_cycles = mac_latency / CYCLE_TIME
            min_cycles   = mac_ops / NUM_PES
            utilization  = min_cycles / total_cycles
            print(f"{layer:<6}  {mac_ops:>12.0f}  {total_cycles:>14.0f}  {min_cycles:>12.0f}  {utilization:>11.1%}")
        except KeyError:
            print(f"{layer:<6}  (MAC columns not found)")

# ---------------------------------------------------------------------------
# Plot — side by side grouped bars per layer
# ---------------------------------------------------------------------------
colors    = ["steelblue", "darkorange", "seagreen", "mediumpurple"]
n_configs = len(all_results)
x         = np.arange(len(LAYERS))
width     = 0.18
offsets   = np.linspace(-(n_configs - 1) / 2, (n_configs - 1) / 2, n_configs) * width

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle(f"TinyYOLOv2 — Reuse Comparison ({FANOUT_X}×{FANOUT_Y} PEs, 1GHz)", fontsize=13)

def grouped_bar_plot(ax, metric_key, title, ylabel):
    for i, r in enumerate(all_results):
        values = [r[metric_key][l] for l in LAYERS]
        bars   = ax.bar(x + offsets[i], values, width,
                        label=r["label"].replace("\n", " "),
                        color=colors[i], edgecolor="white", alpha=0.9)
        ax.bar_label(bars, fmt="%.1e", fontsize=5, padding=2, rotation=90)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(LAYERS, rotation=45)
    ax.legend(fontsize=7)

grouped_bar_plot(axes[0], "layer_energy",  "Energy per Layer",  "Energy (pJ)")
grouped_bar_plot(axes[1], "layer_latency", "Latency per Layer", "Latency (s)")
grouped_bar_plot(axes[2], "layer_edp",     "EDP per Layer",     "EDP (pJ·s)")

plt.tight_layout()

out = WORKSPACE / "outputs"
out.mkdir(exist_ok=True)
plt.savefig(out / "reuse_comparison.png", dpi=150)
print(f"\nPlot saved to {out / 'reuse_comparison.png'}")
plt.show()
