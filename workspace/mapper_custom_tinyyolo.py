"""
Mapper: Custom Tensix NEO Edge Accelerator + TinyYOLOv2
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import copy

import accelforge as af
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.plotting.mappings import plot_energy_breakdown, plot_latency_comparison

# ---------------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------------
ARCH_FILE     = 'workspace/arches/custom_accelerator_sweep_v2.yaml'
WORKLOAD_FILE = 'workspace/workloads/tinyyolo.yaml'

FIXED_MAC_TPT = 2048

JINJA_DATA = {
    'BATCH_SIZE': 1,
    'FANOUT_X': 2,
    'FANOUT_Y': 4,
    'GLB_KB': 512,
    'LB_KB': 8,
    'MAC_TPT': FIXED_MAC_TPT,
}


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


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    print('Building spec: Custom Tensix NEO Edge + TinyYOLOv2...')
    spec = af.Spec.from_yaml(
        ARCH_FILE,
        WORKLOAD_FILE,
        jinja_parse_data=JINJA_DATA,
    )

    # --- Calculate area ---------------------------------------
    print('Calculating component area, energy, latency, leak...')
    evaluated_spec = spec.calculate_component_area_energy_latency_leak()
    area = evaluated_spec.arch.total_area
    print(f'Total architecture area: {area}')

    # --- Configure mapper -------------------------------------
    spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY

    # Print available mapper fields for future tuning
    print(f'\nAvailable FFM mapper fields: {list(spec.mapper.model_fields.keys())}')

    # --- Run mapper -------------------------------------------
    print('\nRunning mapper (this may take a while)...')
    mappings = spec.map_workload_to_arch()

    mapping_data = mappings.data
    n_total = len(mapping_data) if hasattr(mapping_data, '__len__') else 'unknown'
    print(f'Mapper produced {n_total} mappings.')

    # --- Extract best EDP -------------------------------------
    try:
        edp_series = mapping_data["Total<SEP>energy"] * mapping_data["Total<SEP>latency"]
        best_edp = edp_series.min()
        best_idx = edp_series.idxmin()
        best_energy = float(mapping_data["Total<SEP>energy"].iloc[best_idx])
        best_latency = float(mapping_data["Total<SEP>latency"].iloc[best_idx])

        print(f'\n{"Metric":<25} {"Value":>20}')
        print('-' * 47)
        print(f'{"Total Area":<25} {area!s:>20}')
        print(f'{"Total Energy":<25} {best_energy:>20.4e}')
        print(f'{"Total Latency (s)":<25} {best_latency:>20.4e}')
        print(f'{"Best EDP":<25} {float(best_edp):>20.4e}')

    except (KeyError, TypeError) as e:
        print(f'DataFrame-style access failed ({e}), trying object-style...')
        best_mapping = mappings[min_edp_filter(mappings.data)]
        best_energy  = float(best_mapping.energy())
        best_latency = float(best_mapping.latency())
        best_edp     = best_energy * best_latency

        print(f'\n{"Metric":<25} {"Value":>20}')
        print('-' * 47)
        print(f'{"Total Area":<25} {area!s:>20}')
        print(f'{"Total Energy":<25} {best_energy:>20.4e}')
        print(f'{"Total Latency (s)":<25} {best_latency:>20.4e}')
        print(f'{"Best EDP":<25} {best_edp:>20.4e}')

    # --- Best mapping details ---------------------------------
    best_mapping = mappings[min_edp_filter(mappings.data)]

    print('\nPer-einsum energy breakdown:')
    print(f'  {"Einsum":<30} {"Energy":>15}')
    print('  ' + '-' * 47)
    try:
        per_einsum = best_mapping.energy(per_einsum=True)
        for einsum_name, energy_val in per_einsum.items():
            print(f'  {einsum_name:<30} {float(energy_val):>15.4e}')
    except Exception as e:
        print(f'  (per-einsum breakdown unavailable: {e})')

    print('\n=== Best Mapping (YAML) ===')
    try:
        print(best_mapping.mapping().to_yaml())
    except Exception as e:
        print(f'  (YAML output unavailable: {e})')

    # --- Plotting ---------------------------------------------
    print('\nGenerating plots...')
    n_to_show = 1

    # Relevant column names from accelforge
    energy_col = 'Total<SEP>energy'
    latency_col = 'Total<SEP>latency'

    df = mapping_data.copy()

    df['sort_metric'] = df[energy_col].astype(float) * df[latency_col].astype(float)

    # Sorting
    sorted_df = df.sort_values(by='sort_metric')

    top_n_df = sorted_df.head(n_to_show).drop(columns=['sort_metric'])

    print(f'Total mappings explored: {len(mapping_data)}')

    top_n = copy.copy(mappings)
    top_n.data = top_n_df

    # if we want to show all top n
    best_n = mappings[min_edp_filter(top_n.data)]

    print(f'Total mappings explored: {len(mapping_data)}')

    try:
        fig_energy, axes_energy = plot_energy_breakdown(
            [top_n], ['einsum', 'component'], ['action'], ['Best EDP Mapping']
        )
        
        fig_energy.set_size_inches(12, 8) 
        
        # Altering axis label text size:
        for ax in axes_energy:
            ax.tick_params(axis='x', labelsize=8)

        fig_energy.suptitle('Energy Breakdown (Custom Tensix NEO Edge)')
        
        fig_energy.tight_layout() 
        fig_energy.savefig('custom_accel_energy_breakdown.png', bbox_inches='tight')
        print("Saved 'custom_accel_energy_breakdown.png'.")

        fig_lat, ax_lat = plot_latency_comparison(
            [top_n], ['All Mappings']
        )
        
        fig_lat.set_size_inches(2, 5)
        
        fig_lat.suptitle('Latency Comparison (Custom Tensix NEO Edge)')
        fig_lat.tight_layout()
        fig_lat.savefig('custom_accel_latency_comparison.png', bbox_inches='tight')
        print("Saved 'custom_accel_latency_comparison.png'.")

    except Exception as e:
        print(f'Failed to generate plots: {e}')

    print(f'\n--- Summary ---')
    print(f'Architecture area : {area}')
    print(f'Mappings explored : {n_total}')
    print(f'Best EDP          : {float(best_edp):.4e}')


if __name__ == '__main__':
    main()
