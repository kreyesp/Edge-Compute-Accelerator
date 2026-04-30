import accelforge as af
from pathlib import Path
import matplotlib.pyplot as plt

WORKSPACE    = Path(__file__).resolve().parent.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")

# Define the three architectures to compare
ARCHITECTURES = {
    "Custom": "custom_accelerator_sweep_NON_IDEAL.yaml",
    "NVDLA": "nvdla_accelerator.yaml",
    "TPUv4": "tpuv4_accelerator.yaml"
}

FANOUT_X = 8
FANOUT_Y = 8
LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]

def min_edp_filter(df):
    return (df["Total<SEP>energy"] * df["Total<SEP>latency"]).argmin()

# Dictionaries to store the layer-wise results for all architectures
results_energy  = {arch: [] for arch in ARCHITECTURES}
results_latency = {arch: [] for arch in ARCHITECTURES}
results_edp     = {arch: [] for arch in ARCHITECTURES}

print("Running workload mapping for all accelerators. This may take a moment...\n")

# Iterate through each architecture, run the mapping, and extract data
for arch_name, arch_filename in ARCHITECTURES.items():
    arch_src = str(WORKSPACE / "arches" / arch_filename)
    
    # Jinja params are safely passed to all; unused ones will be ignored by NVDLA/TPUv4
    spec = af.Spec.from_yaml(
        arch_src, 
        WORKLOAD_SRC,
        jinja_parse_data={
            "GLB_KB"  : 1024,
            "FREQ_GHZ": 1.0,
            "FANOUT_X": FANOUT_X,
            "FANOUT_Y": FANOUT_Y,
        }
    )

    mappings = spec.map_workload_to_arch()
    data     = mappings.data.iloc[[min_edp_filter(mappings.data)]]
    cols     = data.columns.tolist()

    # Extract per-layer energy and latency
    for layer in LAYERS:
        e_cols = [c for c in cols if c.startswith(f"{layer}<SEP>energy<SEP>")]
        l_cols = [c for c in cols if c.startswith(f"{layer}<SEP>latency<SEP>")]
        
        e = data[e_cols].sum(axis=1).values[0]
        l = data[l_cols].max(axis=1).values[0]
        
        results_energy[arch_name].append(e)
        results_latency[arch_name].append(l)
        results_edp[arch_name].append(e * l)
        
    print(f"[{arch_name}] processing complete.")

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("TinyYOLOv2 — Multi-Accelerator Architecture Comparison", fontsize=14)

markers = ['o', 's', '^']
colors  = ['steelblue', 'darkorange', 'seagreen']

def line_plot(ax, data_dict, title, ylabel):
    for i, (arch_name, values) in enumerate(data_dict.items()):
        ax.plot(
            LAYERS, values, 
            marker=markers[i], 
            color=colors[i], 
            label=arch_name, 
            linewidth=2, 
            markersize=6
        )
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_yscale("log")  # Enforce log scale due to magnitude disparities
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()
    ax.tick_params(axis="x", rotation=45)

line_plot(axes[0], results_energy,  "Energy per Layer",  "Energy (pJ) [Log]")
line_plot(axes[1], results_latency, "Latency per Layer", "Latency (s) [Log]")
line_plot(axes[2], results_edp,     "EDP per Layer",     "EDP (pJ·s) [Log]")

plt.tight_layout()

out = WORKSPACE / "outputs"
out.mkdir(exist_ok=True)
out_file = out / "mapping_results_comparison.png"
plt.savefig(out_file, dpi=150)
print(f"\nComparison plot saved to {out_file}")
plt.show()