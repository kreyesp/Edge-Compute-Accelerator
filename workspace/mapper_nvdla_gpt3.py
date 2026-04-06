"""
Mapper: NVDLA architecture + GPT-3 175B KV-Cache workload.

Runs the accelforge FFM mapper over the dense workload and prints
per-einsum EDP results.
"""

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt

import accelforge as af
from accelforge.frontend.mapper.metrics import Metrics
from accelforge.plotting.mappings import plot_energy_breakdown, plot_latency_comparison

ARCH_FILE     = 'workspace/arches/nvdla.yaml'
# UPDATED: Pointing to the new 175B workload
WORKLOAD_FILE = 'workspace/workloads/gpt3_175B_kv_cache.yaml' 

# Jinja2 template variables for the workload.
# UPDATED: N_TOKENS increased to 1024 to guarantee a large search space 
# and populate the latency plot, while remaining fast enough to test.
JINJA_DATA = {
    'BATCH_SIZE':    1,
    'N_TOKENS':      1024, 
    'N_NEW_TOKENS':  1,
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
    print('Building spec: NVDLA arch + GPT-3 175B KV-Cache workload...')
    spec = af.Spec.from_yaml(
        ARCH_FILE,
        WORKLOAD_FILE,
        jinja_parse_data=JINJA_DATA,
    )

    # Optimise for both energy and latency (EDP).
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

    # ---------------------------------------------------------
    # NEW CODE: Plotting the relevant mapping data
    # ---------------------------------------------------------
    print('\n Generating plots...')
    
    # Use .data to get the true number of generated mappings
    mapping_list = all_mappings.data
    print("THIS IS THE TRUE MAPPINGS LENGTH: " + str(len(mapping_list)))

    try:
        # 1. Plot energy breakdown for the single best EDP mapping
        fig_energy, axes_energy = plot_energy_breakdown(
            [best], ['einsum', 'component'], ['action'], ['Best EDP Mapping']
        )
        fig_energy.suptitle('Energy Breakdown per Einsum (Best Mapping)')
        fig_energy.tight_layout()
        fig_energy.savefig('best_mapping_energy_breakdown.png', bbox_inches='tight')
        print("Saved 'best_mapping_energy_breakdown.png'.")

        # 2. Plot latency comparison across ALL generated mappings
        # Pass the wrapper object, just as accelforge expects
        fig_lat_comp, ax_lat_comp = plot_latency_comparison(
            [all_mappings], ['All Mappings']
        )
        fig_lat_comp.suptitle('Total Latency Comparison Across All Mappings')
        fig_lat_comp.tight_layout()
        fig_lat_comp.savefig('all_mappings_latency_comparison.png', bbox_inches='tight')
        print("Saved 'all_mappings_latency_comparison.png'.")

    except Exception as e:
        print(f"Failed to generate plots: {e}")


if __name__ == '__main__':
    main()