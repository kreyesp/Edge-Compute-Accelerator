import accelforge as af
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

WORKSPACE    = Path(__file__).resolve().parent.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")
ARCH_SRC     = str(WORKSPACE / "arches"    / "custom_accelerator_sweep_NON_IDEAL.yaml")
# WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test_no_reuse.yaml")
# ARCH_SRC     = str(WORKSPACE / "arches"    / "accelerator_sweep_no_reuse.yaml")

LAYERS   = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
# FANOUTS  = [1, 4, 16, 64]
FANOUT = 8
FREQ_GHZ = 1.0
# The benchmark in the YAML is based on 64KB having an area of 2.4e-7 m² (0.24 mm²).
# Realistic GlobalBuffer capacities for edge ML range from ~128KB to 8MB.
# Therefore, realistic area sizes in mm² are:
# 128KB = 0.48 mm²
# 256KB = 0.96 mm²
# 512KB = 1.92 mm²
# 1MB   = 3.84 mm²
# 2MB   = 7.68 mm²
# 4MB   = 15.36 mm²
# 8MB   = 30.72 mm²
AREA_SIZES = [0.48, 0.96, 1.92, 3.84, 7.68, 15.36, 30.72]



def min_edp_filter(df):
    return (df["Total<SEP>energy"] * df["Total<SEP>latency"]).argmin()

def extract_metrics(data, layers):
    cols = data.columns.tolist()
    energy  = sum(data[[c for c in cols if c.startswith(f"{l}<SEP>energy<SEP>")]].sum(axis=1).values[0] for l in layers)
    latency = sum(data[[c for c in cols if c.startswith(f"{l}<SEP>latency<SEP>")]].max(axis=1).values[0] for l in layers)
    return energy, latency, energy * latency

# ── Sweep ────────────────────────────────────────────────────────────────────
results = {}   # fanout -> {n_pes, total_energy, total_latency, total_edp, per_layer}

for area in AREA_SIZES:
    label = f"{area:.2f} mm²"
    # The YAML expects AREA_SIZE as a scaler based on the 64KB = 2.4e-7 m² rule.
    # area_m² = 2.4e-7 * (yaml_scaler / 64)
    # area_mm² * 1e-6 = 2.4e-7 * (yaml_scaler / 64)
    # yaml_scaler = (area_mm² / 0.24) * 64
    yaml_area_scaler = (area / 0.24) * 64
    print(f"Running Area={label} (fanout={FANOUT}, {FANOUT*FANOUT} PEs)...",
          end=" ", flush=True)
    try:
        spec = af.Spec.from_yaml(ARCH_SRC, WORKLOAD_SRC,
                                 jinja_parse_data={
                                     "AREA_SIZE": yaml_area_scaler,
                                     "FREQ_GHZ": FREQ_GHZ,
                                     "FANOUT_X": FANOUT,
                                     "FANOUT_Y": FANOUT,
                                 })
        mappings = spec.map_workload_to_arch()
        data     = mappings.data.iloc[[min_edp_filter(mappings.data)]]

        # Leak energy debug printing
        for layer in LAYERS:
            leak_cols = [c for c in data.columns if f"{layer}<SEP>energy" in c and "leak" in c]
            if leak_cols:
                total_leak = data[leak_cols].sum(axis=1).values[0]
                print(f"{layer} leak energy: {total_leak:.4e}")
            else:
                print(f"{layer}: NO leak columns found")

        total_energy  = data["Total<SEP>energy"].values[0]
        total_latency = data["Total<SEP>latency"].values[0]
        total_edp     = total_energy * total_latency

        per_layer = {}
        for layer in LAYERS:
            e_cols = [c for c in data.columns if c.startswith(f"{layer}<SEP>energy<SEP>")]
            l_cols = [c for c in data.columns if c.startswith(f"{layer}<SEP>latency<SEP>")]
            e = data[e_cols].sum(axis=1).values[0]
            l = data[l_cols].max(axis=1).values[0]
            per_layer[layer] = {"energy": e, "latency": l, "edp": e * l}

        results[area] = {
            "area"         : area,
            "label"        : label,
            "total_energy" : total_energy,
            "total_latency": total_latency,
            "total_edp"    : total_edp,
            "per_layer"    : per_layer,
        }
        print(f"EDP={total_edp:.3e} pJ·s")
    except Exception as ex:
        print(f"FAILED: {ex}")

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Area':>10}  {'Energy (pJ)':>14}  {'Latency (s)':>14}  {'EDP (pJ·s)':>14}")
print("-" * 56)
for fanout, r in results.items():
    print(f"{r['label']:>10}  {r['total_energy']:>14.4e}"
          f"  {r['total_latency']:>14.4e}  {r['total_edp']:>14.4e}")

# ── Plot 1: Total metrics vs PE count ────────────────────────────────────────
area_labels = [r["label"]         for r in results.values()]
areas       = [r["area"]        for r in results.values()]
energies  = [r["total_energy"]  for r in results.values()]
latencies = [r["total_latency"] for r in results.values()]
edps      = [r["total_edp"]     for r in results.values()]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"TinyYOLOv2 — Sweep over Area Size ((Fanout={FANOUT}×{FANOUT}={FANOUT*FANOUT} PEs, {FREQ_GHZ}GHz)", fontsize=12)

def total_plot(ax, y, title, ylabel, color):
    ax.plot(areas, y, marker="o", color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Area Size (mm²)")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(areas)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.2f}"))
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    for x, y_val in zip(areas, y):
        ax.annotate(f"{y_val:.2e}", (x, y_val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)

total_plot(axes[0], energies,  "Total Energy vs Area",  "Energy (pJ)",  "steelblue")
total_plot(axes[1], latencies, "Total Latency vs Area", "Latency (s)",  "darkorange")
total_plot(axes[2], edps,      "Total EDP vs Area",     "EDP (pJ·s)",  "seagreen")

plt.tight_layout()
out = WORKSPACE / "outputs"
out.mkdir(exist_ok=True)
plt.savefig(out / "sweep_area_totals.png", dpi=150)
print(f"\nSaved: {out / 'sweep_area_totals.png'}")

# ── Plot 2: Per-layer EDP breakdown as stacked bar per fanout ────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(f"Per-Layer Breakdown by Area Size (Fanout={FANOUT}×{FANOUT})", fontsize=12)

colors       = plt.cm.tab10.colors
bar_labels   = [r["label"] for r in results.values()]

for ax, metric, ylabel, title in [
    (axes[0], "energy",  "Energy (pJ)",  "Energy per Layer"),
    (axes[1], "latency", "Latency (s)",  "Latency per Layer"),
    (axes[2], "edp",     "EDP (pJ·s)",  "EDP per Layer"),
]:
    bottoms = [0.0] * len(results)
    for j, layer in enumerate(LAYERS):
        values = [r["per_layer"][layer][metric] for r in results.values()]
        ax.bar(bar_labels, values, bottom=bottoms, label=layer,
               color=colors[j], edgecolor="white")
        bottoms = [b + v for b, v in zip(bottoms, values)]
    ax.set_title(title)
    ax.set_xlabel("Area Size")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(out / "sweep_area_per_layer.png", dpi=150)
print(f"Saved: {out / 'sweep_area_per_layer.png'}")

plt.show()
