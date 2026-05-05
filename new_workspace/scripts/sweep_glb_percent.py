import accelforge as af
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

WORKSPACE    = Path(__file__).resolve().parent.parent
WORKLOAD_SRC = str(WORKSPACE / "workloads" / "tiny_yolo_test.yaml")
ARCH_SRC     = str(WORKSPACE / "arches"    / "custom_accelerator_sweep_NON_IDEAL.yaml")

LAYERS   = ["L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9"]
FANOUT   = 8
FREQ_GHZ = 1.0
GLB_SIZES = [128, 256, 512, 1024, 2048, 4096, 8192]  # KB

def min_edp_filter(df):
    return (df["Total<SEP>energy"] * df["Total<SEP>latency"]).argmin()

# ── Sweep ─────────────────────────────────────────────────────────────────────
results = {}

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

        total_energy  = float(data["Total<SEP>energy"].values[0])
        total_latency = float(data["Total<SEP>latency"].values[0])
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

# ── Compute percentage difference from best (minimum) value ──────────────────
best_energy  = min(r["total_energy"]  for r in results.values())
best_latency = min(r["total_latency"] for r in results.values())
best_edp     = min(r["total_edp"]     for r in results.values())

glb_kbs    = [r["glb_kb"]        for r in results.values()]
glb_labels = [r["label"]         for r in results.values()]

pct_energy  = [(r["total_energy"]  - best_energy)  / best_energy  * 100 for r in results.values()]
pct_latency = [(r["total_latency"] - best_latency) / best_latency * 100 for r in results.values()]
pct_edp     = [(r["total_edp"]     - best_edp)     / best_edp     * 100 for r in results.values()]

print(f"\n{'GLB':>8}  {'Energy % vs best':>18}  {'Latency % vs best':>18}  {'EDP % vs best':>14}")
print("-" * 68)
for i, (glb_kb, r) in enumerate(results.items()):
    print(f"{r['label']:>8}  {pct_energy[i]:>17.2f}%  {pct_latency[i]:>17.2f}%  {pct_edp[i]:>13.2f}%")

# ── Plot 1: Absolute values ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(f"TinyYOLOv2 — Sweep over GLB Size (Fanout={FANOUT}×{FANOUT}={FANOUT*FANOUT} PEs, {FREQ_GHZ}GHz)",
             fontsize=12)

energies  = [r["total_energy"]  for r in results.values()]
latencies = [r["total_latency"] for r in results.values()]
edps      = [r["total_edp"]     for r in results.values()]

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
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    for x, y_val in zip(glb_kbs, y):
        ax.annotate(f"{y_val:.2e}", (x, y_val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)

total_plot(axes[0], energies,  "Total Energy vs GLB",  "Energy (pJ)",  "steelblue")
total_plot(axes[1], latencies, "Total Latency vs GLB", "Latency (s)",  "darkorange")
total_plot(axes[2], edps,      "Total EDP vs GLB",     "EDP (pJ·s)",   "seagreen")

plt.tight_layout()
out = WORKSPACE / "outputs"
out.mkdir(exist_ok=True)
plt.savefig(out / "sweep_glb_totals.png", dpi=150)
print(f"\nSaved: {out / 'sweep_glb_totals.png'}")

# ── Plot 2: Percentage difference from best ───────────────────────────────────
fig2, axes2 = plt.subplots(1, 3, figsize=(15, 5))
fig2.suptitle(
    f"TinyYOLOv2 — GLB Sweep: % Overhead vs Best Config\n"
    f"(Fanout={FANOUT}×{FANOUT}={FANOUT*FANOUT} PEs, {FREQ_GHZ}GHz, 0% = best)",
    fontsize=12
)

def pct_plot(ax, pct_values, title, color):
    ax.plot(glb_kbs, pct_values, marker="o", color=color, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("GLB Size")
    ax.set_ylabel("% above best")
    ax.set_xscale("log", base=2)
    ax.set_xticks(glb_kbs)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x)}KB" if x < 1024 else f"{int(x)//1024}MB"
    ))
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(y=0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    for x, y_val in zip(glb_kbs, pct_values):
        ax.annotate(f"{y_val:.2f}%", (x, y_val), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=7)
    # highlight the best point
    best_idx = pct_values.index(min(pct_values))
    ax.scatter([glb_kbs[best_idx]], [pct_values[best_idx]],
               color="red", zorder=5, s=80, label=f"Best: {glb_labels[best_idx]}")
    ax.legend(fontsize=8)

pct_plot(axes2[0], pct_energy,  "Energy % vs Best",  "steelblue")
pct_plot(axes2[1], pct_latency, "Latency % vs Best", "darkorange")
pct_plot(axes2[2], pct_edp,     "EDP % vs Best",     "seagreen")

plt.tight_layout()
plt.savefig(out / "sweep_glb_pct.png", dpi=150)
print(f"Saved: {out / 'sweep_glb_pct.png'}")

# ── Plot 3: Per-layer stacked bars ────────────────────────────────────────────
fig3, axes3 = plt.subplots(1, 3, figsize=(16, 6))
fig3.suptitle(f"Per-Layer Breakdown by GLB Size (Fanout={FANOUT}×{FANOUT})", fontsize=12)

layer_colors = plt.cm.tab10.colors
bar_labels   = [r["label"] for r in results.values()]

for ax, metric, ylabel, title in [
    (axes3[0], "energy",  "Energy (pJ)",  "Energy per Layer"),
    (axes3[1], "latency", "Latency (s)",  "Latency per Layer"),
    (axes3[2], "edp",     "EDP (pJ·s)",   "EDP per Layer"),
]:
    bottoms = [0.0] * len(results)
    for j, layer in enumerate(LAYERS):
        values = [r["per_layer"][layer][metric] for r in results.values()]
        ax.bar(bar_labels, values, bottom=bottoms, label=layer,
               color=layer_colors[j], edgecolor="white")
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
