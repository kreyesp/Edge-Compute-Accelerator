import accelforge as af
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

WORKSPACE    = Path(__file__).resolve().parent.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")
ARCH_SRC     = str(WORKSPACE / "arches"    / "custom_accelerator_sweep_NON_IDEAL.yaml")

LAYERS   = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
FANOUT   = 8                                          # fixed
FREQ_GHZ = 1.0
# 1MB center, 3 smaller, 3 larger (all powers of 2)
GLB_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]  # KB

def min_edp_filter(df):
    return (df["Total<SEP>energy"] * df["Total<SEP>latency"]).argmin()

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = {}   # glb_kb -> {glb_kb, total_energy, total_latency, total_edp, per_layer}

for glb_kb in GLB_SIZES:
    label = f"{glb_kb}KB" if glb_kb < 1024 else f"{glb_kb//1024}MB"
    print(f"Running GLB={label} (fanout={FANOUT}, {FANOUT*FANOUT} PEs)...",
          end=" ", flush=True)
    try:
        spec = af.Spec.from_yaml(ARCH_SRC, WORKLOAD_SRC,
                                 jinja_parse_data={
                                     "GLB_KB"  : glb_kb,
                                     "FREQ_GHZ": FREQ_GHZ,
                                     "FANOUT_X": FANOUT,
                                     "FANOUT_Y": FANOUT,
                                 })
        mappings = spec.map_workload_to_arch()
        data     = mappings.data.iloc[[min_edp_filter(mappings.data)]]

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

        results[glb_kb] = {
            "glb_kb"       : glb_kb,
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
print(f"\n{'GLB':>8}  {'Energy (pJ)':>14}  {'Latency (s)':>14}  {'EDP (pJ·s)':>14}")
print("-" * 56)
for glb_kb, r in results.items():
    print(f"{r['label']:>8}  {r['total_energy']:>14.4e}"
          f"  {r['total_latency']:>14.4e}  {r['total_edp']:>14.4e}")

# ── Plot 1: Total metrics vs GLB size ─────────────────────────────────────────
glb_labels = [r["label"]         for r in results.values()]
glb_kbs    = [r["glb_kb"]        for r in results.values()]
energies   = [r["total_energy"]  for r in results.values()]
latencies  = [r["total_latency"] for r in results.values()]
edps       = [r["total_edp"]     for r in results.values()]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"TinyYOLOv2 — Sweep over GLB Size (Fanout={FANOUT}×{FANOUT}={FANOUT*FANOUT} PEs, {FREQ_GHZ}GHz)",
             fontsize=12)

def total_plot(ax, y, title, ylabel, color):
    ax.plot(glb_kbs, y, marker="o", color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("GLB Size")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    ax.set_xticks(glb_kbs)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x)}KB" if x < 1024 else f"{int(x)//1024}MB"
    ))
    ax.tick_params(axis="x", rotation=45)
    ax.axvline(x=1024, color="gray", linestyle="--", alpha=0.5, label="1MB center")
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    for x, y_val in zip(glb_kbs, y):
        ax.annotate(f"{y_val:.2e}", (x, y_val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)

total_plot(axes[0], energies,  "Total Energy vs GLB",  "Energy (pJ)",  "steelblue")
total_plot(axes[1], latencies, "Total Latency vs GLB", "Latency (s)",  "darkorange")
total_plot(axes[2], edps,      "Total EDP vs GLB",     "EDP (pJ·s)",  "seagreen")

plt.tight_layout()
out = WORKSPACE / "outputs"
out.mkdir(exist_ok=True)
plt.savefig(out / "sweep_glb_totals.png", dpi=150)
print(f"\nSaved: {out / 'sweep_glb_totals.png'}")

# ── Plot 2: Per-layer breakdown as stacked bars ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle(f"Per-Layer Breakdown by GLB Size (Fanout={FANOUT}×{FANOUT})", fontsize=12)

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
    ax.set_xlabel("GLB Size")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=7, loc="upper right")

plt.tight_layout()
plt.savefig(out / "sweep_glb_per_layer.png", dpi=150)
print(f"Saved: {out / 'sweep_glb_per_layer.png'}")

plt.show()
