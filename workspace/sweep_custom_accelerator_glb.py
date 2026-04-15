"""
GLB Size Sweep: Tensix NEO Edge Accelerator + TinyYOLOv2

Sweeps GLB size with fixed fanout (8x8) and MAC throughput (2048)
to test whether increasing on-chip buffer reduces L8 energy by
keeping more weights in GLB instead of refetching from DRAM.

Photos saved to: photos/GLB_sweep_<workload_name>/
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
ARCH_FILE = 'workspace/arches/custom_accelerator_sweep.yaml'

WORKLOADS = {
    '1': ('workspace/workloads/tinyyolo.yaml',      'TinyYOLOv2_200x200'),
    '2': ('workspace/workloads/tinyyolo_400.yaml',   'TinyYOLOv2_400x400'),
    '3': ('workspace/workloads/tinyyolo_600.yaml',   'TinyYOLOv2_600x600'),
}

# ---------------------------------------------------------------
# SWEEP CONFIGURATIONS
# ---------------------------------------------------------------
# Fixed spatial and MAC config (best from previous sweep)
FIXED_FANOUT_X = 8
FIXED_FANOUT_Y = 8
FIXED_MAC_TPT  = 2048

# GLB sizes to sweep (in KB)
# 512-1024 = within edge budget
# 2048-8192 = over budget but shows theoretical benefit
GLB_KB_VALUES = [512, 1024, 2048, 4096, 6144, 8192, 10240]

# For reference: L8 weight size = 1024*1024*3*3 = 9,437,184 bytes = 9216 KB


# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

def select_workload():
    print('\nAvailable workloads:')
    for key, (path, desc) in WORKLOADS.items():
        print(f'  [{key}] {desc} ({path})')
    print(f'  [c] Custom path')

    choice = input('\nSelect workload: ').strip()

    if choice in WORKLOADS:
        path, desc = WORKLOADS[choice]
        print(f'Selected: {desc}')
        return path, desc
    elif choice.lower() == 'c':
        path = input('Enter workload YAML path: ').strip()
        desc = input('Enter folder name for photos: ').strip() or 'custom'
        return path, desc
    else:
        print(f'Invalid choice "{choice}", defaulting to option 1.')
        path, desc = WORKLOADS['1']
        return path, desc


def make_output_dir(workload_name):
    out_dir = os.path.join('photos', f'GLB_sweep_{workload_name}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


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


def run_config(arch_file, workload_file, glb_kb):
    config_name = f'GLB{glb_kb}KB'
    print(f'\n{"="*60}')
    print(f'Config: {config_name}')
    print(f'  GLB: {glb_kb}KB ({glb_kb/1024:.1f}MB)')
    print(f'  Fanout: {FIXED_FANOUT_X}x{FIXED_FANOUT_Y} = {FIXED_FANOUT_X*FIXED_FANOUT_Y} PEs')
    print(f'{"="*60}')

    try:
        spec = af.Spec.from_yaml(
            arch_file,
            workload_file,
            jinja_parse_data={
                'BATCH_SIZE': 1,
                'FANOUT_X': FIXED_FANOUT_X,
                'FANOUT_Y': FIXED_FANOUT_Y,
                'GLB_KB': glb_kb,
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
        if per_layer:
            for layer, e in per_layer.items():
                print(f'     {layer}: {e:.4e}')

        return {
            'config': config_name,
            'glb_kb': glb_kb,
            'n_mappings': n_mappings,
            'energy': best_energy,
            'latency': best_latency,
            'edp': best_edp,
            'per_layer': per_layer,
        }

    except Exception as e:
        print(f'  -> FAILED: {e}')
        return {
            'config': config_name,
            'glb_kb': glb_kb,
            'n_mappings': 0,
            'energy': float('inf'),
            'latency': float('inf'),
            'edp': float('inf'),
            'per_layer': {},
        }


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    workload_file, workload_name = select_workload()
    out_dir = make_output_dir(workload_name)

    print(f'\nWorkload:  {workload_name}')
    print(f'Arch:      {ARCH_FILE}')
    print(f'Photos:    {out_dir}/')
    print(f'Fixed:     FX={FIXED_FANOUT_X}, FY={FIXED_FANOUT_Y}, MAC={FIXED_MAC_TPT}')
    print(f'Sweeping:  GLB = {GLB_KB_VALUES} KB')
    print(f'Note:      L8 weights = 9216KB. GLB > 9216KB should fit all weights.\n')

    results = []
    for glb in GLB_KB_VALUES:
        result = run_config(ARCH_FILE, workload_file, glb)
        results.append(result)

    valid = [r for r in results if r['edp'] < float('inf')]
    if not valid:
        print('\nNo valid results!')
        return

    valid.sort(key=lambda r: r['glb_kb'])

    # --- Print results table ----------------------------------
    print(f'\n{"="*80}')
    print(f'RESULTS — GLB Sweep — {workload_name}')
    print(f'{"="*80}')
    print(f'{"GLB (KB)":<12} {"GLB (MB)":<10} {"Energy":>12} {"Latency":>12} {"EDP":>12} {"L8 Energy":>12} {"L8 %":>8}')
    print('-' * 80)
    for r in valid:
        l8_energy = r['per_layer'].get('L8_out', 0)
        total_e = sum(r['per_layer'].values()) if r['per_layer'] else 1
        l8_pct = 100 * l8_energy / total_e if total_e > 0 else 0
        print(f'{r["glb_kb"]:<12} {r["glb_kb"]/1024:<10.1f} {r["energy"]:>12.4e} '
              f'{r["latency"]:>12.4e} {r["edp"]:>12.4e} {l8_energy:>12.4e} {l8_pct:>7.1f}%')

    # --- Plot 1: Total energy + EDP vs GLB size ---------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    glb_sizes = [r['glb_kb'] for r in valid]
    glb_mb = [g / 1024 for g in glb_sizes]

    # Total energy vs GLB
    ax = axes[0, 0]
    ax.plot(glb_mb, [r['energy'] for r in valid], 'o-', color='orange', linewidth=2)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1MB edge budget')
    ax.axvline(x=9.0, color='blue', linestyle='--', alpha=0.5, label='L8 weight size (9MB)')
    ax.set_xlabel('GLB Size (MB)')
    ax.set_ylabel('Total Energy')
    ax.set_title('Total Energy vs GLB Size')
    ax.legend(fontsize=8)

    # EDP vs GLB
    ax = axes[0, 1]
    ax.plot(glb_mb, [r['edp'] for r in valid], 's-', color='steelblue', linewidth=2)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1MB edge budget')
    ax.axvline(x=9.0, color='blue', linestyle='--', alpha=0.5, label='L8 weight size (9MB)')
    ax.set_xlabel('GLB Size (MB)')
    ax.set_ylabel('EDP')
    ax.set_title('EDP vs GLB Size')
    ax.legend(fontsize=8)

    # Latency vs GLB
    ax = axes[1, 0]
    ax.plot(glb_mb, [r['latency'] for r in valid], '^-', color='green', linewidth=2)
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='1MB edge budget')
    ax.axvline(x=9.0, color='blue', linestyle='--', alpha=0.5, label='L8 weight size (9MB)')
    ax.set_xlabel('GLB Size (MB)')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Latency vs GLB Size')
    ax.legend(fontsize=8)

    # L8 energy vs GLB
    ax = axes[1, 1]
    l8_energies = [r['per_layer'].get('L8_out', 0) for r in valid]
    l7_energies = [r['per_layer'].get('L7_out', 0) for r in valid]
    ax.plot(glb_mb, l8_energies, 'o-', color='red', linewidth=2, label='L8 (1024→1024)')
    ax.plot(glb_mb, l7_energies, 's-', color='purple', linewidth=2, label='L7 (512→1024)')
    ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.3, label='1MB edge budget')
    ax.axvline(x=9.0, color='blue', linestyle='--', alpha=0.3, label='L8 weight size')
    ax.set_xlabel('GLB Size (MB)')
    ax.set_ylabel('Energy')
    ax.set_title('L7 & L8 Energy vs GLB Size')
    ax.legend(fontsize=8)

    fig.suptitle(f'GLB Size Sweep — {workload_name} (FX{FIXED_FANOUT_X}_FY{FIXED_FANOUT_Y})', fontsize=14)
    fig.tight_layout()
    path1 = os.path.join(out_dir, 'glb_sweep_summary.png')
    fig.savefig(path1, bbox_inches='tight', dpi=150)
    print(f"\nSaved '{path1}'")

    # --- Plot 2: Per-layer energy at each GLB size (stacked) --
    configs_with_layers = [r for r in valid if r['per_layer']]
    if configs_with_layers:
        layer_names = list(configs_with_layers[0]['per_layer'].keys())

        fig2, ax2 = plt.subplots(figsize=(14, 7))
        x = np.arange(len(configs_with_layers))
        width = 0.6
        bottom = np.zeros(len(configs_with_layers))
        colors = plt.cm.tab10(np.linspace(0, 1, len(layer_names)))

        for i, layer in enumerate(layer_names):
            values = [c['per_layer'].get(layer, 0) for c in configs_with_layers]
            ax2.bar(x, values, width, bottom=bottom, label=layer, color=colors[i])
            bottom += np.array(values)

        ax2.set_xlabel('GLB Size')
        ax2.set_ylabel('Total Energy (stacked)')
        ax2.set_title(f'Per-Layer Energy Stacked by GLB Size — {workload_name}')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{c["glb_kb"]}KB\n({c["glb_kb"]/1024:.1f}MB)' for c in configs_with_layers],
                           fontsize=8)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        fig2.tight_layout()
        path2 = os.path.join(out_dir, 'glb_sweep_stacked_energy.png')
        fig2.savefig(path2, bbox_inches='tight', dpi=150)
        print(f"Saved '{path2}'")

        # --- Plot 3: Per-layer % breakdown at each GLB size ---
        fig3, ax3 = plt.subplots(figsize=(14, 7))
        bottom_pct = np.zeros(len(configs_with_layers))

        for i, layer in enumerate(layer_names):
            values = [c['per_layer'].get(layer, 0) for c in configs_with_layers]
            totals = [sum(c['per_layer'].values()) for c in configs_with_layers]
            pcts = [100 * v / t if t > 0 else 0 for v, t in zip(values, totals)]
            ax3.bar(x, pcts, width, bottom=bottom_pct, label=layer, color=colors[i])
            bottom_pct += np.array(pcts)

        ax3.set_xlabel('GLB Size')
        ax3.set_ylabel('Energy Share (%)')
        ax3.set_title(f'Per-Layer Energy % by GLB Size — {workload_name}')
        ax3.set_xticks(x)
        ax3.set_xticklabels([f'{c["glb_kb"]}KB\n({c["glb_kb"]/1024:.1f}MB)' for c in configs_with_layers],
                           fontsize=8)
        ax3.set_ylim(0, 105)
        ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        fig3.tight_layout()
        path3 = os.path.join(out_dir, 'glb_sweep_pct_breakdown.png')
        fig3.savefig(path3, bbox_inches='tight', dpi=150)
        print(f"Saved '{path3}'")

    # --- Summary ----------------------------------------------
    print(f'\n--- Summary ({workload_name}) ---')
    print(f'Photos saved to   : {out_dir}/')
    print(f'GLB sizes tested  : {[r["glb_kb"] for r in valid]} KB')

    best_by_edp = min(valid, key=lambda r: r['edp'])
    print(f'Best EDP config   : {best_by_edp["config"]} (EDP={best_by_edp["edp"]:.4e})')

    # Show L8 energy reduction
    if len(valid) >= 2:
        smallest = valid[0]
        largest = valid[-1]
        l8_small = smallest['per_layer'].get('L8_out', 0)
        l8_large = largest['per_layer'].get('L8_out', 0)
        if l8_small > 0:
            reduction = (1 - l8_large / l8_small) * 100
            print(f'\nL8 energy at {smallest["glb_kb"]}KB GLB: {l8_small:.4e}')
            print(f'L8 energy at {largest["glb_kb"]}KB GLB:  {l8_large:.4e}')
            print(f'L8 energy reduction: {reduction:.1f}%')


if __name__ == '__main__':
    main()
