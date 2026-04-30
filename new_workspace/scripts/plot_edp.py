import accelforge as af
from pathlib import Path
import matplotlib.pyplot as plt

WORKSPACE    = Path(__file__).resolve().parent.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")
# ARCH_SRC     = str(WORKSPACE / "arches"    / "custom_accelerator_sweep.yaml")
ARCH_SRC     = str(WORKSPACE / "arches"    / "nvdla_accelerator.yaml")
FANOUT_X = 8
FANOUT_Y = 8

LAYERS = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]

def min_edp_filter(df):
    return (df["Total<SEP>energy"] * df["Total<SEP>latency"]).argmin()

spec = af.Spec.from_yaml(ARCH_SRC, WORKLOAD_SRC,
                         jinja_parse_data={
                             "GLB_KB"  : 1024,
                             "FREQ_GHZ": 1.0,
                             "FANOUT_X": FANOUT_X,
                             "FANOUT_Y": FANOUT_Y,
                         })

mappings = spec.map_workload_to_arch()
data     = mappings.data.iloc[[min_edp_filter(mappings.data)]]
cols     = data.columns.tolist()

# --- per-layer energy: sum all LN<SEP>energy<SEP>* columns ---
# --- per-layer latency: max of LN<SEP>latency<SEP>* columns (critical path) ---
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

print(f"{'Layer':<6}  {'Energy (pJ)':>14}  {'Latency (s)':>14}  {'EDP (pJ·s)':>14}")
print("-" * 56)
for l in LAYERS:
    print(f"{l:<6}  {layer_energy[l]:>14.4e}  {layer_latency[l]:>14.4e}  {layer_edp[l]:>14.4e}")
print("-" * 56)
print(f"{'Total':<6}  {total_energy:>14.4e}  {total_latency:>14.4e}  {total_edp:>14.4e}")


#checking MAC utilization
NUM_PES    = FANOUT_X * FANOUT_Y   # FANOUT_X * FANOUT_Y
FREQ_GHZ   = 1.0
CYCLE_TIME = 1 / (FREQ_GHZ * 1e9)

print(f"\n{'Layer':<6}  {'MAC ops':>12}  {'Total cycles':>14}  {'Min cycles':>12}  {'Utilization':>12}")
print("-" * 64)
for layer in LAYERS:
    mac_ops      = data[f"{layer}<SEP>action<SEP>MAC<SEP>None<SEP>compute"].values[0]
    mac_latency  = data[f"{layer}<SEP>latency<SEP>MAC"].values[0]
    total_cycles = mac_latency / CYCLE_TIME          # actual cycles taken
    min_cycles   = mac_ops / NUM_PES                  # minimum possible if PEs never stall
    utilization  = min_cycles / total_cycles
    print(f"{layer:<6}  {mac_ops:>12.0f}  {total_cycles:>14.0f}  {min_cycles:>12.0f}  {utilization:>11.1%}")

# --- Plot ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("TinyYOLOv2 — Mapping Results", fontsize=13)

def bar_plot(ax, values, title, ylabel, color):
    bars = ax.bar(LAYERS, [values[l] for l in LAYERS], color=color, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.bar_label(bars, fmt="%.1e", fontsize=7, padding=3)
    ax.tick_params(axis="x", rotation=45)

bar_plot(axes[0], layer_energy,  "Energy per Layer",  "Energy (pJ)",  "steelblue")
bar_plot(axes[1], layer_latency, "Latency per Layer", "Latency (s)",  "darkorange")
bar_plot(axes[2], layer_edp,     "EDP per Layer",     "EDP (pJ·s)",   "seagreen")

plt.tight_layout()

out = WORKSPACE / "outputs"
out.mkdir(exist_ok=True)
plt.savefig(out / "mapping_results.png", dpi=150)
print(f"\nPlot saved to {out / 'mapping_results.png'}")
plt.show()
