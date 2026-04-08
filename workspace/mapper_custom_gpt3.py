"""
Mapper: Custom Tensix NEO Edge Accelerator

Runs the accelforge FFM mapper and prints per-einsum EDP results.
Adjust WORKLOAD_FILE and JINJA_DATA for your target workload.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import accelforge as af
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.plotting.mappings import plot_energy_breakdown, plot_latency_comparison

ARCH_FILE     = 'workspace/arches/custom_accelerator_v0.yaml'

WORKLOAD_FILE = 'workspace/workloads/gpt3_175B_kv_cache.yaml'
# WORKLOAD_FILE = 'workspace/workloads/tinyyolo.yaml'

JINJA_DATA = {
    'BATCH_SIZE': 1,
}


def min_edp_filter(data):
    """Return the index of the mapping with minimum energy-delay product."""
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


def main():
    print('Building spec: Custom Tensix NEO Edge + workload...')
    spec = af.Spec.from_yaml(
        ARCH_FILE,
        WORKLOAD_FILE,
        jinja_parse_data=JINJA_DATA,
    )

    spec.mapper.metrics = Metrics.LATENCY | Metrics.ENERGY

    print('Running mapper (this may take a while)...')
    all_mappings = spec.map_workload_to_arch()

    best = all_mappings[min_edp_filter(all_mappings.data)]
    total_energy  = float(best.energy())
    total_latency = float(best.latency())
    total_edp     = total_energy * total_latency

    print(f'\n{"Metric":<20} {"Value":>20}')
    print('-' * 42)
    print(f'{"Total Energy (J)":<20} {total_energy:>20.4e}')
    print(f'{"Total Latency (s)":<20} {total_latency:>20.4e}')
    print(f'{"EDP (J·s)":<20} {total_edp:>20.4e}')

    # Per-einsum breakdown
    print('\nPer-einsum energy breakdown:')
    print(f'  {"Einsum":<30} {"Energy (J)":>15}')
    print('  ' + '-' * 47)
    try:
        per_einsum = best.energy(per_einsum=True)
        for einsum_name, energy_val in per_einsum.items():
            print(f'  {einsum_name:<30} {float(energy_val):>15.4e}')
    except Exception as e:
        print(f'  (per-einsum breakdown unavailable: {e})')

    print('\n=== Best Mapping (YAML) ===')
    print(best.mapping().to_yaml())

    # Plotting
    print('\nGenerating plots...')
    mapping_list = all_mappings.data
    print(f'Total mappings explored: {len(mapping_list)}')

    try:
        fig_energy, axes_energy = plot_energy_breakdown(
            [best], ['einsum', 'component'], ['action'], ['Best EDP Mapping']
        )
        fig_energy.suptitle('Energy Breakdown (Custom Tensix NEO Edge)')
        fig_energy.tight_layout()
        fig_energy.savefig('custom_accel_energy_breakdown.png', bbox_inches='tight')
        print("Saved 'custom_accel_energy_breakdown.png'.")

        fig_lat, ax_lat = plot_latency_comparison(
            [all_mappings], ['All Mappings']
        )
        fig_lat.suptitle('Latency Comparison (Custom Tensix NEO Edge)')
        fig_lat.tight_layout()
        fig_lat.savefig('custom_accel_latency_comparison.png', bbox_inches='tight')
        print("Saved 'custom_accel_latency_comparison.png'.")

    except Exception as e:
        print(f'Failed to generate plots: {e}')


if __name__ == '__main__':
    main()
