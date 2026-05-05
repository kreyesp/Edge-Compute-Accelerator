import accelforge as af
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from accelforge.frontend.mapper.metrics import Metrics

WORKSPACE    = Path(__file__).resolve().parent.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")
ARCH_SRC     = str(WORKSPACE / "arches"    / "custom_accelerator_sweep_local_buffer.yaml")

LAYERS   = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
FANOUT   = 8
FREQ_GHZ = 1.0
N_PES    = FANOUT * FANOUT   # 64

# 3 options: total memory = GLB + 64 * LOCAL = 1MB
# Option A: GLB dominant  — 768KB GLB + 4KB per PE
# Option B: equal split   — 512KB GLB + 8KB per PE
# Option C: local dominant— 256KB GLB + 12KB per PE
CONFIGS = [
    {"name": "A: 768KB GLB + 4KB Local", "GLB_KB": 768,  "LOCAL_KB": 4},
    {"name": "B: 512KB GLB + 8KB Local", "GLB_KB": 512,  "LOCAL_KB": 8},
    {"name": "C: 256KB GLB + 12KB Local","GLB_KB": 256,  "LOCAL_KB": 12},
]

def min_edp_filter(df):
    return (df["Total<SEP>energy"] * df["Total<SEP>latency"]).argmin()

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = {}

for cfg in CONFIGS:
    name = cfg["name"]
    print(f"Running {name}...", end=" ", flush=True)
    try:
        spec = af.Spec.from_yaml(ARCH_SRC, WORKLOAD_SRC,
                                 jinja_parse_data={
                                     "GLB_KB"  : cfg["GLB_KB"],
                                     "LOCAL_KB": cfg["LOCAL_KB"],
                                     "FREQ_GHZ": FREQ_GHZ,
                                     "FANOUT_X": FANOUT,
                                     "FANOUT_Y": FANOUT,
                                 })
        spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY  # ← add this
        mappings = spec.map_workload_to_arch()
        data     = mappings.data.iloc[[min_edp_filter(mappings.data)]]

        total_energy  = float(data["Total<SEP>energy"].values[0])
        total_latency = float(data["Total<SEP>latency"].values[0])
        total_edp     = total_energy * total_latency

        per_layer = {}
        for layer in LAYERS:
            e_cols = [c for c in data.columns if c.startswith(f"{layer}<SEP>energy<SEP>")]
            l_cols = [c for c in data.columns if c.startswith(f"{layer}<SEP>latency<SEP>")]
            e = data[e_cols].sum(axis=1).values[0] if e_cols else 0.0
            l = data[l_cols].max(axis=1).values[0] if l_cols else 0.0
            per_layer[layer] = {"energy": e, "latency": l, "edp": e * l}

        results[name] = {
            "total_energy" : total_energy,
            "total_latency": total_latency,
            "total_edp"    : total_edp,
            "per_layer"    : per_layer,
            "glb_kb"       : cfg["GLB_KB"],
            "local_kb"     : cfg["LOCAL_KB"],
        }
        print(f"EDP={total_edp:.3e} pJ·s")
    except Exception as ex:
        print(f"FAILED: {ex}")

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Config':<30}  {'Energy (pJ)':>14}  {'Latency (s)':>14}  {'EDP (pJ·s)':>14}")
print("-" * 78)
for name, r in results.items():
    print(f"{name:<30}  {r['total_energy']:>14.4e}"
          f"  {r['total_latency']:>14.4e}  {r['total_edp']:>14.4e}")

# ── Plot 1: Total metrics comparison ─────────────────────────────────────────
names     = list(results.keys())
# short_names = ["A: 768KB+4KB", "B: 512KB+8KB", "C: 256KB+12KB"]
short_names = ["A", "B", "C"]
energies  = [r["total_energy"]  for r in results.values()]
latencies = [r["total_latency"] for r in results.values()]
edps      = [r["total_edp"]     for r in results.values()]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"TinyYOLOv2 — GLB vs LocalBuffer Split (Total=1MB, {FANOUT}×{FANOUT}={N_PES} PEs, {FREQ_GHZ}GHz)",
             fontsize=11)

colors = ["steelblue", "darkorange", "seagreen"]

def bar_plot(ax, values, title, ylabel):
    bars = ax.bar(short_names, values, color=colors, edgecolor="white")
    ax.set_title(title)
    ax.set_xlabel("Memory Split Config")
    ax.set_ylabel(ylabel)
    ax.bar_label(bars, fmt="%.2e", fontsize=8, padding=3)
    ax.tick_params(axis="x", rotation=15)

bar_plot(axes[0], energies,  "Total Energy",  "Energy (pJ)")
bar_plot(axes[1], latencies, "Total Latency", "Latency (s)")
bar_plot(axes[2], edps,      "Total EDP",     "EDP (pJ·s)")

plt.tight_layout()
out = WORKSPACE / "outputs"
out.mkdir(exist_ok=True)
plt.savefig(out / "sweep_local_buffer_totals.png", dpi=150)
print(f"\nSaved: {out / 'sweep_local_buffer_totals.png'}")

# ── Plot 2: Per-layer EDP breakdown ──────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("Per-Layer Breakdown by Memory Split Config", fontsize=12)

layer_colors = plt.cm.tab10.colors

for ax, metric, ylabel, title in [
    (axes[0], "energy",  "Energy (pJ)",  "Energy per Layer"),
    (axes[1], "latency", "Latency (s)",  "Latency per Layer"),
    (axes[2], "edp",     "EDP (pJ·s)",  "EDP per Layer"),
]:
    bottoms = [0.0] * len(results)
    for j, layer in enumerate(LAYERS):
        values = [r["per_layer"][layer][metric] for r in results.values()]
        ax.bar(short_names, values, bottom=bottoms, label=layer,
               color=layer_colors[j], edgecolor="white")
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_title(title)
    ax.set_xlabel("Memory Split Config")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(out / "sweep_local_buffer_per_layer.png", dpi=150)
print(f"Saved: {out / 'sweep_local_buffer_per_layer.png'}")

plt.show()
