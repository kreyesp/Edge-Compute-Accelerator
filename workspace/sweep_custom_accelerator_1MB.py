"""
Memory Allocation Sweep: Fixed 1MB SRAM Budget

Explores different ways to split 1MB between shared GLB and
per-PE LocalBuffers. Tests the tradeoff between having a large
shared buffer vs many small private buffers.

Photos saved to: photos/memory_allocation_<workload_name>/
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import accelforge as af
from accelforge.frontend.mapper.metrics import Metrics

# ---------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------
ARCH_FILE = 'workspace/arches/custom_accelerator_sweep_v3.yaml'

WORKLOADS = {
    '1': ('workspace/workloads/tinyyolo.yaml',      'TinyYOLOv2_200x200'),
    '2': ('workspace/workloads/tinyyolo_400.yaml',   'TinyYOLOv2_400x400'),
    '3': ('workspace/workloads/tinyyolo_600.yaml',   'TinyYOLOv2_600x600'),
}

TOTAL_SRAM_KB = 1024  # 1MB budget
FIXED_MAC_TPT = 2048

# ---------------------------------------------------------------
# MEMORY ALLOCATION CONFIGS
# ---------------------------------------------------------------
# Each tuple: (fanout_x, fanout_y, glb_kb, lb_kb_per_pe, description)
# Constraint: glb_kb + (fanout_x * fanout_y * lb_kb) <= TOTAL_SRAM_KB

CONFIGS = [
    # Baseline: no LocalBuffer (v1 equivalent)
    (8, 8, 1024, 1,   '1024KB GLB, 1KB LB, 64 PEs'),
    (4, 4, 1024, 1,   '1024KB GLB, 1KB LB, 16 PEs'),

    # Heavy GLB, light LocalBuffer
    (8, 8, 896,  2,   '896KB GLB, 2KB LB, 64 PEs'),
    (4, 4, 960,  4,   '960KB GLB, 4KB LB, 16 PEs'),

    # Balanced split
    (8, 8, 512,  8,   '512KB GLB, 8KB LB, 64 PEs'),
    (4, 4, 768,  16,  '768KB GLB, 16KB LB, 16 PEs'),
    (4, 8, 512,  16,  '512KB GLB, 16KB LB, 32 PEs'),

    # Heavy LocalBuffer, light GLB
    (4, 4, 512,  32,  '512KB GLB, 32KB LB, 16 PEs'),
    (2, 4, 512,  64,  '512KB GLB, 64KB LB, 8 PEs'),

    # Fewer PEs, bigger everything
    (2, 2, 768,  64,  '768KB GLB, 64KB LB, 4 PEs'),
    (2, 4, 768,  32,  '768KB GLB, 32KB LB, 8 PEs'),
]


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

def select_workload():
    print('\nAvailable workloads:')
    for key, (path, desc) in WORKLOADS.items():
        print(f'  [{key}] {desc} ({path})')
    choice = input('\nSelect workload: ').strip()
    if choice in WORKLOADS:
        path, desc = WORKLOADS[choice]
        print(f'Selected: {desc}')
        return path, desc
    else:
        path, desc = WORKLOADS['1']
        return path, desc


def min_edp_filter(data):
    best_idx = 0
    best_edp = float('inf')
    for i, result in enumerate(data):
        try:
            edp = float(result.energy()) * float(result.latency())
            if edp < best_edp:
                best_edp = edp
                best_idx = i
        except Exception:
            continue
    return best_idx


def get_per_layer_energy(mappings):
    best = mappings[min_edp_filter(mappings.data)]
    try:
        per_einsum = best.energy(per_einsum=True)
        return {k: float(v) for k, v in per_einsum.items()}
    except Exception:
        return {}


def run_config(workload_file, fx, fy, glb_kb, lb_kb, desc):
    total_pes = fx * fy
    total_sram = glb_kb + (total_pes * lb_kb)
    config_name = f'FX{fx}_FY{fy}_GLB{glb_kb}_LB{lb_kb}'

    print(f'\n{"="*60}')
    print(f'{desc}')
    print(f'  Config: {config_name}')
    print(f'  Total SRAM: {total_sram}KB ({total_sram/1024:.2f}MB)')
    print(f'  GLB: {glb_kb}KB | LB: {lb_kb}KB x {total_pes} = {total_pes*lb_kb}KB')
    print(f'{"="*60}')

    if total_sram > TOTAL_SRAM_KB + 64:
        print(f'  -> SKIPPED: exceeds {TOTAL_SRAM_KB}KB budget ({total_sram}KB)')
        return None

    try:
        spec = af.Spec.from_yaml(
            ARCH_FILE,
            workload_file,
            jinja_parse_data={
                'BATCH_SIZE': 1,
                'FANOUT_X': fx,
                'FANOUT_Y': fy,
                'GLB_KB': glb_kb,
                'LB_KB': lb_kb,
                'MAC_TPT': FIXED_MAC_TPT,
            },
        )

        spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY
        mappings = spec.map_workload_to_arch()

        mapping_data = mappings.data
        n_mappings = len(mapping_data) if hasattr(mapping_data, '__len__') else 0
        per_layer = get_per_layer_energy(mappings)

        try:
            edp_series = mapping_data["Total<SEP>energy"] * mapping_data["Total<SEP>latency"]
            best_edp = float(edp_series.min())
            best_idx = edp_series.idxmin()
            best_energy = float(mapping_data["Total<SEP>energy"].iloc[best_idx])
            best_latency = float(mapping_data["Total<SEP>latency"].iloc[best_idx])
        except (KeyError, TypeError):
            best_mapping = mappings[min_edp_filter(mappings.data)]
            best_energy  = float(best_mapping.energy())
            best_latency = float(best_mapping.latency())
            best_edp     = best_energy * best_latency

        print(f'  -> EDP={best_edp:.4e}, E={best_energy:.4e}, L={best_latency:.4e}')

        return {
            'config': config_name,
            'desc': desc,
            'fanout_x': fx,
            'fanout_y': fy,
            'total_pes': total_pes,
            'glb_kb': glb_kb,
            'lb_kb': lb_kb,
            'total_lb_kb': total_pes * lb_kb,
            'total_sram_kb': total_sram,
            'n_mappings': n_mappings,
            'energy': best_energy,
            'latency': best_latency,
            'edp': best_edp,
            'per_layer': per_layer,
        }

    except Exception as e:
        print(f'  -> FAILED: {e}')
        return None


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    workload_file, workload_name = select_workload()
    out_dir = os.path.join('photos', f'memory_allocation_{workload_name}')
    os.makedirs(out_dir, exist_ok=True)

    print(f'\nWorkload:     {workload_name}')
    print(f'SRAM Budget:  {TOTAL_SRAM_KB}KB ({TOTAL_SRAM_KB/1024:.1f}MB)')
    print(f'Photos:       {out_dir}/')
    print(f'Configs:      {len(CONFIGS)}')

    results = []
    for fx, fy, glb, lb, desc in CONFIGS:
        result = run_config(workload_file, fx, fy, glb, lb, desc)
        if result:
            results.append(result)

    if not results:
        print('\nNo valid results!')
        return

    results.sort(key=lambda r: r['edp'])
    best = results[0]

    # --- Print results table ----------------------------------
    print(f'\n{"="*100}')
    print(f'RESULTS — Memory Allocation Sweep — {workload_name} (sorted by EDP)')
    print(f'{"="*100}')
    print(f'{"Description":<40} {"PEs":>4} {"GLB":>6} {"LB/PE":>6} {"TotLB":>6} '
          f'{"Energy":>12} {"Latency":>12} {"EDP":>12}')
    print('-' * 105)
    for r in results:
        print(f'{r["desc"]:<40} {r["total_pes"]:>4} {r["glb_kb"]:>5}K '
              f'{r["lb_kb"]:>5}K {r["total_lb_kb"]:>5}K '
              f'{r["energy"]:>12.4e} {r["latency"]:>12.4e} {r["edp"]:>12.4e}')

    print(f'\nBest: {best["desc"]} (EDP={best["edp"]:.4e})')

    # --- Plot 1: 4-panel sweep summary (same style as fanout sweep) ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    pes = [r['total_pes'] for r in results]
    edps = [r['edp'] for r in results]
    energies = [r['energy'] for r in results]
    latencies = [r['latency'] for r in results]
    glb_sizes = [r['glb_kb'] for r in results]

    # Color by GLB size to show allocation effect
    glb_colors = [r['glb_kb'] for r in results]

    # EDP vs PEs
    ax = axes[0, 0]
    sc = ax.scatter(pes, edps, c=glb_colors, cmap='coolwarm', s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total PEs (Fanout_X × Fanout_Y)')
    ax.set_ylabel('EDP')
    ax.set_title('EDP vs PE Count')
    ax.set_yscale('log')
    plt.colorbar(sc, ax=ax, label='GLB Size (KB)')

    # Energy vs PEs
    ax = axes[0, 1]
    sc = ax.scatter(pes, energies, c=glb_colors, cmap='coolwarm', s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total PEs')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs PE Count')
    ax.set_yscale('log')
    plt.colorbar(sc, ax=ax, label='GLB Size (KB)')

    # Latency vs PEs
    ax = axes[1, 0]
    sc = ax.scatter(pes, latencies, c=glb_colors, cmap='coolwarm', s=80, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Total PEs')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Latency vs PE Count')
    ax.set_yscale('log')
    plt.colorbar(sc, ax=ax, label='GLB Size (KB)')

    # GLB vs LB allocation with EDP as color
    ax = axes[1, 1]
    sc = ax.scatter(glb_sizes, [r['total_lb_kb'] for r in results],
                    c=edps, cmap='viridis_r', s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('GLB Size (KB)')
    ax.set_ylabel('Total LocalBuffer (KB)')
    ax.set_title('Memory Split (color = EDP)')
    # Draw 1MB budget line
    budget_glb = np.linspace(0, TOTAL_SRAM_KB, 100)
    budget_lb = TOTAL_SRAM_KB - budget_glb
    ax.plot(budget_glb, budget_lb, 'r--', alpha=0.5, label=f'{TOTAL_SRAM_KB}KB budget')
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='EDP')

    fig.suptitle(f'Memory Allocation Sweep — {workload_name} (1MB Budget)', fontsize=14)
    fig.tight_layout()
    path1 = os.path.join(out_dir, 'sweep_results.png')
    fig.savefig(path1, bbox_inches='tight', dpi=150)
    print(f"\nSaved '{path1}'")

    # --- Plot 2: EDP/Energy/Latency bar charts by config ------
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))

    labels = [r['desc'] for r in results]

    # EDP comparison
    ax = axes2[0, 0]
    colors = ['green' if r == best else 'steelblue' for r in results]
    ax.barh(range(len(results)), edps, color=colors)
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('EDP')
    ax.set_title('EDP by Memory Allocation')
    ax.invert_yaxis()

    # Energy comparison
    ax = axes2[0, 1]
    ax.barh(range(len(results)), energies, color='orange')
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Energy')
    ax.set_title('Energy by Memory Allocation')
    ax.invert_yaxis()

    # Latency comparison
    ax = axes2[1, 0]
    ax.barh(range(len(results)), latencies, color='green')
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('Latency (s)')
    ax.set_title('Latency by Memory Allocation')
    ax.invert_yaxis()

    # Memory split visualization
    ax = axes2[1, 1]
    glb_vals = [r['glb_kb'] for r in results]
    lb_vals = [r['total_lb_kb'] for r in results]
    ax.barh(range(len(results)), glb_vals, color='royalblue', label='GLB')
    ax.barh(range(len(results)), lb_vals, left=glb_vals, color='coral', label='LocalBuffers')
    ax.axvline(x=TOTAL_SRAM_KB, color='red', linestyle='--', alpha=0.5, label=f'{TOTAL_SRAM_KB}KB budget')
    ax.set_yticks(range(len(results)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel('SRAM (KB)')
    ax.set_title('Memory Split: GLB vs LocalBuffers')
    ax.legend(fontsize=8)
    ax.invert_yaxis()

    fig2.suptitle(f'Memory Allocation Details — {workload_name} (1MB Budget)', fontsize=14)
    fig2.tight_layout()
    path2 = os.path.join(out_dir, 'allocation_comparison.png')
    fig2.savefig(path2, bbox_inches='tight', dpi=150)
    print(f"Saved '{path2}'")

    # --- Plot 3: Per-layer breakdown for top 3 configs --------
    top3 = [r for r in results[:3] if r['per_layer']]
    if top3:
        layer_names = list(top3[0]['per_layer'].keys())

        fig3, ax3 = plt.subplots(figsize=(14, 7))
        x = np.arange(len(layer_names))
        width = 0.8 / len(top3)

        for i, r in enumerate(top3):
            layer_energies = [r['per_layer'].get(ln, 0) for ln in layer_names]
            offset = (i - len(top3)/2 + 0.5) * width
            ax3.bar(x + offset, layer_energies, width, label=r['desc'])

        ax3.set_xlabel('Layer')
        ax3.set_ylabel('Energy')
        ax3.set_title(f'Per-Layer Energy — Top 3 Allocations — {workload_name}')
        ax3.set_xticks(x)
        ax3.set_xticklabels(layer_names, rotation=45, ha='right')
        ax3.legend(fontsize=8)
        ax3.set_yscale('log')
        fig3.tight_layout()
        path3 = os.path.join(out_dir, 'per_layer_comparison.png')
        fig3.savefig(path3, bbox_inches='tight', dpi=150)
        print(f"Saved '{path3}'")

    # --- Plot 4: Best config per-layer with percentages -------
    best_layers = best['per_layer']
    if best_layers:
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        names = list(best_layers.keys())
        values = list(best_layers.values())
        colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

        bars = ax4.bar(names, values, color=colors)
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Energy')
        ax4.set_title(f'Per-Layer Energy — {workload_name}\nBest: {best["desc"]}')
        ax4.tick_params(axis='x', rotation=45)

        total = sum(values)
        for bar, val in zip(bars, values):
            pct = 100 * val / total if total > 0 else 0
            if pct > 2:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

        fig4.tight_layout()
        path4 = os.path.join(out_dir, 'best_per_layer.png')
        fig4.savefig(path4, bbox_inches='tight', dpi=150)
        print(f"Saved '{path4}'")

    # --- Summary ----------------------------------------------
    print(f'\n--- Summary ---')
    print(f'Workload          : {workload_name}')
    print(f'SRAM Budget       : {TOTAL_SRAM_KB}KB')
    print(f'Configs tested    : {len(results)}')
    print(f'Best allocation   : {best["desc"]}')
    print(f'  GLB: {best["glb_kb"]}KB, LB: {best["lb_kb"]}KB/PE x {best["total_pes"]} PEs = {best["total_lb_kb"]}KB')
    print(f'  EDP: {best["edp"]:.4e}')
    print(f'  Energy: {best["energy"]:.4e}')
    print(f'  Latency: {best["latency"]:.4e}')
    if best['per_layer']:
        total_e = sum(best['per_layer'].values())
        print(f'\nPer-layer energy (best config):')
        for layer, e in sorted(best['per_layer'].items(), key=lambda x: -x[1]):
            pct = 100 * e / total_e if total_e > 0 else 0
            print(f'  {layer:<20} {e:.4e}  ({pct:.1f}%)')


if __name__ == '__main__':
    main()
