"""
Parameter Sweep: Tensix NEO Edge Accelerator + TinyYOLOv2

Sweeps spatial fanout, GLB size, and MAC throughput to find
optimal configurations under edge constraints.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import accelforge as af
from accelforge.frontend.mapper.metrics import Metrics

# ---------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------
ARCH_FILE     = 'workspace/arches/custom_accelerator_sweep.yaml'
WORKLOAD_FILE = 'workspace/workloads/tinyyolo.yaml'

# ---------------------------------------------------------------
# SWEEP CONFIGURATIONS
# ---------------------------------------------------------------
# Each list defines values to try for that parameter.
# Total configs = product of all list lengths.

FANOUT_X_VALUES = [1, 2, 4, 8]       # Input reuse dimension
FANOUT_Y_VALUES = [1, 2, 4, 8]       # Output reuse dimension
GLB_KB_VALUES   = [256, 512, 768, 1024]     # GlobalBuffer size in KB
MAC_TPT_VALUES  = [512, 1024, 2048]   # MAC ops/clk throughput

# Set to True to only sweep fanout (faster, good for initial exploration)
FANOUT_ONLY = True

# ---------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------

def min_edp_filter(data):
    """Return the index of the mapping with minimum EDP."""
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


def run_config(fanout_x, fanout_y, glb_kb, mac_tpt):
    """Run mapper for a single configuration. Returns dict of results."""
    config_name = f'FX{fanout_x}_FY{fanout_y}_GLB{glb_kb}_MAC{mac_tpt}'
    print(f'\n{"="*60}')
    print(f'Config: {config_name}')
    print(f'  Fanout: {fanout_x}x{fanout_y} = {fanout_x*fanout_y} PEs')
    print(f'  GLB: {glb_kb}KB, MAC throughput: {mac_tpt} ops/clk')
    print(f'{"="*60}')

    try:
        spec = af.Spec.from_yaml(
            ARCH_FILE,
            WORKLOAD_FILE,
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

        # Get best EDP
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
        }


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    # Build sweep configurations
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

    # Run all configurations
    results = []
    for fx, fy, glb, mac in configs:
        result = run_config(fx, fy, glb, mac)
        results.append(result)

    # Filter out failed runs
    valid = [r for r in results if r['edp'] < float('inf')]

    if not valid:
        print('\nNo valid results!')
        return

    # Sort by EDP
    valid.sort(key=lambda r: r['edp'])

    # --- Print results table ----------------------------------
    print(f'\n{"="*80}')
    print('RESULTS (sorted by EDP)')
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

    # --- Plot 1: EDP vs Total PEs ----------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # EDP vs PEs
    ax = axes[0, 0]
    pes = [r['total_pes'] for r in valid]
    edps = [r['edp'] for r in valid]
    ax.scatter(pes, edps, c='steelblue', s=60, alpha=0.7)
    ax.set_xlabel('Total PEs (Fanout_X × Fanout_Y)')
    ax.set_ylabel('EDP')
    ax.set_title('EDP vs PE Count')
    ax.set_yscale('log')

    # Energy vs PEs
    ax = axes[0, 1]
    energies = [r['energy'] for r in valid]
    ax.scatter(pes, energies, c='orange', s=60, alpha=0.7)
    ax.set_xlabel('Total PEs')
    ax.set_ylabel('Energy')
    ax.set_title('Energy vs PE Count')
    ax.set_yscale('log')

    # Latency vs PEs
    ax = axes[1, 0]
    latencies = [r['latency'] for r in valid]
    ax.scatter(pes, latencies, c='green', s=60, alpha=0.7)
    ax.set_xlabel('Total PEs')
    ax.set_ylabel('Latency (s)')
    ax.set_title('Latency vs PE Count')
    ax.set_yscale('log')

    # EDP heatmap: Fanout_X vs Fanout_Y (FANOUT_ONLY mode)
    ax = axes[1, 1]
    if FANOUT_ONLY:
        fx_vals = sorted(set(r['fanout_x'] for r in valid))
        fy_vals = sorted(set(r['fanout_y'] for r in valid))
        edp_grid = np.full((len(fy_vals), len(fx_vals)), np.nan)
        for r in valid:
            xi = fx_vals.index(r['fanout_x'])
            yi = fy_vals.index(r['fanout_y'])
            edp_grid[yi, xi] = r['edp']

        im = ax.imshow(edp_grid, aspect='auto', origin='lower',
                        cmap='viridis_r')
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

    fig.suptitle('Tensix NEO Edge — Parameter Sweep Results', fontsize=14)
    fig.tight_layout()
    fig.savefig('sweep_results.png', bbox_inches='tight', dpi=150)
    print("\nSaved 'sweep_results.png'")


if __name__ == '__main__':
    main()
