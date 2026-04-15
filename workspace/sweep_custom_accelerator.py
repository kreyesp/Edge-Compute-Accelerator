"""
Parameter Sweep: Tensix NEO Edge Accelerator + TinyYOLOv2

Sweeps spatial fanout, GLB size, and MAC throughput to find
optimal configurations under edge constraints.
Includes per-layer energy breakdown for the best configuration.

Prompts for both architecture and workload selection.
Photos saved to: photos/<arch_name>_<workload_name>/
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import accelforge as af
from accelforge.frontend.mapper.metrics import Metrics

# ---------------------------------------------------------------
# AVAILABLE ARCHITECTURES AND WORKLOADS
# ---------------------------------------------------------------
ARCHITECTURES = {
    '1': ('workspace/arches/custom_accelerator_sweep.yaml',    'v1_single_GLB'),
    '2': ('workspace/arches/custom_accelerator_sweep_v2.yaml', 'v2_with_LocalBuffer'),
}

WORKLOADS = {
    '1': ('workspace/workloads/tinyyolo.yaml',      'TinyYOLOv2_200x200'),
    '2': ('workspace/workloads/tinyyolo_400.yaml',   'TinyYOLOv2_400x400'),
    '3': ('workspace/workloads/tinyyolo_600.yaml',   'TinyYOLOv2_600x600'),
}

# ---------------------------------------------------------------
# SWEEP CONFIGURATIONS
# ---------------------------------------------------------------
FANOUT_X_VALUES = [1, 2, 4, 8]
FANOUT_Y_VALUES = [1, 2, 4, 8]
GLB_KB_VALUES   = [256, 512, 768, 1024]
MAC_TPT_VALUES  = [512, 1024, 2048]

FANOUT_ONLY = True

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

def select_architecture():
    """Prompt user to select an architecture."""
    print('\nAvailable architectures:')
    for key, (path, desc) in ARCHITECTURES.items():
        print(f'  [{key}] {desc} ({path})')
    print(f'  [c] Custom path')

    choice = input('\nSelect architecture: ').strip()

    if choice in ARCHITECTURES:
        path, desc = ARCHITECTURES[choice]
        print(f'Selected: {desc}')
        return path, desc
    elif choice.lower() == 'c':
        path = input('Enter architecture YAML path: ').strip()
        desc = input('Enter short name for folder: ').strip() or 'custom_arch'
        return path, desc
    else:
        print(f'Invalid choice "{choice}", defaulting to option 1.')
        path, desc = ARCHITECTURES['1']
        return path, desc


def select_workload():
    """Prompt user to select a workload."""
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
        desc = input('Enter short name for folder: ').strip() or 'custom_workload'
        return path, desc
    else:
        print(f'Invalid choice "{choice}", defaulting to option 1.')
        path, desc = WORKLOADS['1']
        return path, desc


def make_output_dir(arch_name, workload_name):
    """Create photos/<arch_name>_<workload_name>/ directory."""
    out_dir = os.path.join('photos', f'{arch_name}_{workload_name}')
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


def run_config(arch_file, workload_file, fanout_x, fanout_y, glb_kb, mac_tpt):
    config_name = f'FX{fanout_x}_FY{fanout_y}_GLB{glb_kb}_MAC{mac_tpt}'
    print(f'\n{"="*60}')
    print(f'Config: {config_name}')
    print(f'  Fanout: {fanout_x}x{fanout_y} = {fanout_x*fanout_y} PEs')
    print(f'  GLB: {glb_kb}KB, MAC throughput: {mac_tpt} ops/clk')
    print(f'{"="*60}')

    try:
        spec = af.Spec.from_yaml(
            arch_file,
            workload_file,
            jinja_parse_data={
                'BATCH_SIZE': 1,
                'FANOUT_X': fanout_x,
                'FANOUT_Y': fanout_y,
                'GLB_KB': glb_kb,
                'MAC_TPT': mac_tpt,
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

        print(f'  -> {n_mappings} mappings, EDP={best_edp:.4e}, '
              f'E={best_energy:.4e}, L={best_latency:.4e}')
        if per_layer:
            for layer, e in per_layer.items():
                print(f'     {layer}: {e:.4e}')

        return {
            'config': config_name,
            'fanout_x': fanout_x,
            'fanout_y': fanout_y,
            'total_pes': fanout_x * fanout_y,
            'glb_kb': glb_kb,
            'mac_tpt': mac_tpt,
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
            'fanout_x': fanout_x,
            'fanout_y': fanout_y,
            'total_pes': fanout_x * fanout_y,
            'glb_kb': glb_kb,
            'mac_tpt': mac_tpt,
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
    arch_file, arch_name = select_architecture()
    workload_file, workload_name = select_workload()
    run_label = f'{arch_name} + {workload_name}'
    out_dir = make_output_dir(arch_name, workload_name)

    print(f'\n{"="*60}')
    print(f'Architecture: {arch_name} ({arch_file})')
    print(f'Workload:     {workload_name} ({workload_file})')
    print(f'Photos:       {out_dir}/')
    print(f'{"="*60}')

    if FANOUT_ONLY:
        configs = [
            (fx, fy, 1024, 2048)
            for fx, fy in product(FANOUT_X_VALUES, FANOUT_Y_VALUES)
        ]
        print(f'FANOUT-ONLY sweep: {len(configs)} configurations')
    else:
        configs = list(product(
            FANOUT_X_VALUES, FANOUT_Y_VALUES,
            GLB_KB_VALUES, MAC_TPT_VALUES
        ))
        print(f'Full sweep: {len(configs)} configurations')

    results = []
    for fx, fy, glb, mac in configs:
        result = run_config(arch_file, workload_file, fx, fy, glb, mac)
        results.append(result)

    valid = [r for r in results if r['edp'] < float('inf')]
    if not valid:
        print('\nNo valid results!')
        return

    valid.sort(key=lambda r: r['edp'])

    # --- Print results table ----------------------------------
    print(f'\n{"="*80}')
    print(f'RESULTS — {run_label} (sorted by EDP)')
    print(f'{"="*80}')
    print(f'{"Config":<30} {"PEs":>5} {"GLB":>6} {"MAC":>6} '
          f'{"Energy":>12} {"Latency":>12} {"EDP":>12} {"Maps":>5}')
    print('-' * 95)
    for r in valid:
        print(f'{r["config"]:<30} {r["total_pes"]:>5} {r["glb_kb"]:>5}K '
              f'{r["mac_tpt"]:>5} {r["energy"]:>12.4e} {r["latency"]:>12.4e} '
              f'{r["edp"]:>12.4e} {r["n_mappings"]:>5}')

    best = valid[0]
    print(f'\nBest config: {best["config"]} with EDP={best["edp"]:.4e}')

    # --- Plot 1: 4-panel sweep summary ------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    pes = [r['total_pes'] for r in valid]
    edps = [r['edp'] for r in valid]
    ax.scatter(pes, edps, c='steelblue', s=60, alpha=0.7)
    ax.set_xlabel('Total PEs (Fanout_X × Fanout_Y)')
    ax.set_ylabel('EDP')
    ax.set_title('EDP vs PE Count')
    ax.set_yscale('log')

    ax = axes[0, 1]
    energies = [r['energy'] for r in valid]
    ax.scatter(pes, energies, c='orange', s=60, alpha=0.7)
    ax.set_xlabel('Total PEs')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs PE Count')
    ax.set_yscale('log')

    ax = axes[1, 0]
    latencies = [r['latency'] for r in valid]
    ax.scatter(pes, latencies, c='green', s=60, alpha=0.7)
    ax.set_xlabel('Total PEs')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Latency vs PE Count')
    ax.set_yscale('log')

    ax = axes[1, 1]
    if FANOUT_ONLY:
        fx_vals = sorted(set(r['fanout_x'] for r in valid))
        fy_vals = sorted(set(r['fanout_y'] for r in valid))
        edp_grid = np.full((len(fy_vals), len(fx_vals)), np.nan)
        for r in valid:
            xi = fx_vals.index(r['fanout_x'])
            yi = fy_vals.index(r['fanout_y'])
            edp_grid[yi, xi] = r['edp']
        im = ax.imshow(edp_grid, aspect='auto', origin='lower', cmap='viridis_r')
        ax.set_xticks(range(len(fx_vals)))
        ax.set_xticklabels(fx_vals)
        ax.set_yticks(range(len(fy_vals)))
        ax.set_yticklabels(fy_vals)
        ax.set_xlabel('Fanout X (input reuse)')
        ax.set_ylabel('Fanout Y (output reuse)')
        ax.set_title('EDP Heatmap (Fanout X vs Y)')
        plt.colorbar(im, ax=ax, label='EDP')
    else:
        ax.text(0.5, 0.5, 'Heatmap only in\nFANOUT_ONLY mode',
                ha='center', va='center', transform=ax.transAxes)

    fig.suptitle(f'{run_label}', fontsize=14)
    fig.tight_layout()
    path1 = os.path.join(out_dir, 'sweep_results.png')
    fig.savefig(path1, bbox_inches='tight', dpi=150)
    print(f"\nSaved '{path1}'")

    # --- Plot 2: Per-layer energy comparison ------------------
    configs_to_plot = []
    if len(valid) >= 3:
        configs_to_plot = [valid[0], valid[len(valid)//2], valid[-1]]
        labels = ['Best EDP', 'Mid EDP', 'Worst EDP']
    elif len(valid) >= 1:
        configs_to_plot = [valid[0]]
        labels = ['Best EDP']

    configs_with_layers = [(c, l) for c, l in zip(configs_to_plot, labels) if c['per_layer']]

    if configs_with_layers:
        layer_names = list(configs_with_layers[0][0]['per_layer'].keys())

        fig2, ax2 = plt.subplots(figsize=(12, 6))
        x = np.arange(len(layer_names))
        width = 0.8 / len(configs_with_layers)

        for i, (conf, label) in enumerate(configs_with_layers):
            energies = [conf['per_layer'].get(ln, 0) for ln in layer_names]
            offset = (i - len(configs_with_layers)/2 + 0.5) * width
            ax2.bar(x + offset, energies, width, label=f'{label} ({conf["config"]})')

        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Energy')
        ax2.set_title(f'Per-Layer Energy Comparison — {run_label}')
        ax2.set_xticks(x)
        ax2.set_xticklabels(layer_names, rotation=45, ha='right')
        ax2.legend()
        ax2.set_yscale('log')
        fig2.tight_layout()
        path2 = os.path.join(out_dir, 'per_layer_comparison.png')
        fig2.savefig(path2, bbox_inches='tight', dpi=150)
        print(f"Saved '{path2}'")

        # --- Plot 3: Best config per-layer breakdown ----------
        best_layers = best['per_layer']
        if best_layers:
            fig3, ax3 = plt.subplots(figsize=(10, 6))
            names = list(best_layers.keys())
            values = list(best_layers.values())
            colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

            bars = ax3.bar(names, values, color=colors)
            ax3.set_xlabel('Layer')
            ax3.set_ylabel('Energy')
            ax3.set_title(f'Per-Layer Energy — {run_label}\nBest ({best["config"]}, {best["total_pes"]} PEs)')
            ax3.tick_params(axis='x', rotation=45)

            total = sum(values)
            for bar, val in zip(bars, values):
                pct = 100 * val / total if total > 0 else 0
                if pct > 2:
                    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                             f'{pct:.1f}%', ha='center', va='bottom', fontsize=9)

            fig3.tight_layout()
            path3 = os.path.join(out_dir, 'best_per_layer.png')
            fig3.savefig(path3, bbox_inches='tight', dpi=150)
            print(f"Saved '{path3}'")
    else:
        print('No per-layer energy data available for plotting.')

    # --- Summary ----------------------------------------------
    print(f'\n--- Summary ---')
    print(f'Architecture      : {arch_name}')
    print(f'Workload          : {workload_name}')
    print(f'Photos saved to   : {out_dir}/')
    print(f'Configs tested    : {len(valid)}')
    print(f'Best config       : {best["config"]}')
    print(f'Best EDP          : {best["edp"]:.4e}')
    print(f'Best Energy       : {best["energy"]:.4e}')
    print(f'Best Latency      : {best["latency"]:.4e}')
    if best['per_layer']:
        total_e = sum(best['per_layer'].values())
        print(f'\nPer-layer energy (best config):')
        for layer, e in sorted(best['per_layer'].items(), key=lambda x: -x[1]):
            pct = 100 * e / total_e if total_e > 0 else 0
            print(f'  {layer:<20} {e:.4e}  ({pct:.1f}%)')


if __name__ == '__main__':
    main()
